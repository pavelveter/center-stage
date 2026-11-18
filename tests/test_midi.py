"""Hardware-free tests for the MIDI mapping engine, manager and TUI bridge."""

import os
import tempfile
import threading
import time
import types
from unittest import mock

import pytest

from hyper_stage import tui
from hyper_stage.config import DEFAULTS, load_config, save_config
from hyper_stage.midi import MidiManager
from hyper_stage.midi_mapping import (
    ControlEvent,
    build_mappings,
    Feedback,
    Incoming,
    MidiRouter,
    normalize_value,
    parse_mapping,
    target_for_control,
)


def make_cfg(**overrides):
    result = dict(DEFAULTS)
    result.update(overrides)
    return result


def cc(channel, number, value):
    return Incoming("cc", channel - 1, number, value)


def note(channel, number, velocity=127):
    return Incoming("note_on", channel - 1, number, velocity)


class TestConfiguration:
    def test_midi_section_round_trips_without_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "hyper-stage.ini")
            original = load_config(path)
            original["midi_enabled"] = True
            original["midi_input_device"] = " nanoKONTROL2 "
            original["midi_channel"] = 0
            original["midi_feedback_interval"] = 35
            original["midi_button_debounce"] = 120
            original["midi_mappings"] = dict(original["midi_mappings"])
            original["midi_mappings"]["zoom"] = "cc:2:7:relative_0_127:invert"
            save_config(path, original)

            with open(path) as handle:
                text = handle.read()
            assert "[midi]" in text
            assert "zoom = cc:2:7:relative_0_127:invert" in text
            assert "{" not in text and "json" not in text.lower()
            assert "; Optional MIDI controller" in text

            loaded = load_config(path)
            assert loaded["midi_enabled"] is True
            assert loaded["midi_input_device"] == "nanoKONTROL2"
            assert loaded["midi_channel"] == 0
            assert loaded["midi_feedback_interval"] == 35
            assert loaded["midi_button_debounce"] == 120
            assert loaded["midi_mappings"]["zoom"] == "cc:2:7:relative_0_127:invert"
            assert loaded["midi_mappings"]["tracking_enabled"] == "cc:1:20:toggle"

    def test_one_target_can_hold_several_pipe_separated_mappings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "multi.ini")
            with open(path, "w") as handle:
                handle.write(
                    "[midi]\nzoom = cc:1:10:absolute|cc:1:14:relative_0_127\n"
                )
            loaded = load_config(path)
            mappings, warnings = build_mappings(loaded)
            assert not warnings
            assert [
                (mapping.number, mapping.mode)
                for mapping in mappings
                if mapping.target == "zoom"
            ] == [(10, "absolute"), (14, "relative_0_127")]

    def test_old_ini_without_midi_section_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "old.ini")
            with open(path, "w") as handle:
                handle.write("[camera]\nindex = 2\n[tracking]\nzoom = 2.5\n")
            loaded = load_config(path)
            assert loaded["camera_index"] == 2
            assert loaded["zoom"] == 2.5
            assert loaded["midi_enabled"] is False
            assert loaded["midi_mappings"]["zoom"] == "cc:1:10:absolute"

    def test_invalid_mappings_warn_and_are_skipped(self, capsys):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.ini")
            with open(path, "w") as handle:
                handle.write(
                    "[midi]\nenabled = true\n"
                    "zoom = cc:1:bad:absolute\n"
                    "lut_scale = note:1:200:absolute\n"
                    "bogus_target = cc:1:1:absolute\n"
                    "tracking_enabled = cc:1:20:toggle\n"
                )
            loaded = load_config(path)
            warnings = capsys.readouterr().out
            assert "invalid MIDI mapping" in warnings
            assert "bogus_target" in warnings
            assert loaded["midi_mappings"]["zoom"] == "cc:1:10:absolute"
            assert loaded["midi_mappings"]["lut_scale"] == "cc:1:13:absolute"
            assert "tracking_enabled" in loaded["midi_mappings"]
            assert "bogus_target" not in loaded["midi_mappings"]

    def test_mapping_grammar_validation(self):
        with pytest.raises(ValueError):
            parse_mapping("zoom", "cc:1:10")
        with pytest.raises(ValueError):
            parse_mapping("zoom", "cc:1:10:toggle")  # wrong mode for tuning
        with pytest.raises(ValueError):
            parse_mapping("zoom", "cc:17:10:absolute")
        with pytest.raises(ValueError):
            parse_mapping("zoom", "cc:1:128:absolute")
        with pytest.raises(ValueError):
            parse_mapping("zoom", "cc:1:10:absolute:typo")
        assert parse_mapping("zoom", "cc:0:10:relative_0_127:invert:accel").invert


    def test_invalid_debounce_falls_back_to_default(self, capsys):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.ini")
            with open(path, "w") as handle:
                handle.write("[midi]\nbutton_debounce = -5\n")
            loaded = load_config(path)
            assert loaded["midi_button_debounce"] == DEFAULTS["midi_button_debounce"]
            assert "button_debounce" in capsys.readouterr().out


class TestControlRegistry:
    def test_pip_steppers_resolve_to_their_midi_targets(self):
        assert target_for_control("btn-pip-prev") == "pip_prev"
        assert target_for_control("btn-pip-next") == "pip_next"
        assert target_for_control("pip-select") == "pip_next"
        assert target_for_control("btn-lut-prev") == "lut_prev"
        assert target_for_control("btn-lut-next") == "lut_next"
        assert target_for_control("btn-shader-prev") == "shader_prev"
        assert target_for_control("btn-shader-next") == "shader_next"
        assert target_for_control("text-input") is None


class TestMappingEngine:
    @staticmethod
    def router(*specs, channel=1):
        return MidiRouter([parse_mapping(*spec) for spec in specs], channel)

    def test_cc_absolute_scales_and_clamps_to_target_range(self):
        router = self.router(("zoom", "cc:1:10:absolute"))
        assert router.route(cc(1, 10, 0)) == [
            ControlEvent("zoom", "set_value", 0.0)
        ]
        assert router.route(cc(1, 10, 127))[0].value == 1.0
        assert router.route(cc(1, 10, 64))[0].value == pytest.approx(64 / 127)

    def test_relative_twos_complement(self):
        router = self.router(("zoom", "cc:1:10:relative_twos"))
        assert router.route(cc(1, 10, 3))[0].value == pytest.approx(0.15)
        assert router.route(cc(1, 10, 126))[0].value == pytest.approx(-0.1)
        assert router.route(cc(1, 10, 64)) == []
        assert router.route(cc(1, 10, 0)) == []

    def test_relative_binary_offset(self):
        router = self.router(("zoom", "cc:1:10:relative_binary"))
        assert router.route(cc(1, 10, 2))[0].value == pytest.approx(0.1)
        assert router.route(cc(1, 10, 66))[0].value == pytest.approx(-0.1)
        assert router.route(cc(1, 10, 64)) == []

    def test_relative_0_127_native_direction_and_ignore(self):
        router = self.router(("zoom", "cc:1:10:relative_0_127"))
        assert router.route(cc(1, 10, 0)) == [ControlEvent("zoom", "step", 0.05)]
        assert router.route(cc(1, 10, 127)) == [ControlEvent("zoom", "step", -0.05)]
        # Values that are not movement must not be treated as absolute.
        for value in (1, 63, 64, 65, 126):
            assert router.route(cc(1, 10, value)) == []

    def test_relative_0_127_invert_and_accel(self):
        plain = self.router(("zoom", "cc:1:10:relative_0_127"))
        inverted = self.router(("zoom", "cc:1:10:relative_0_127:invert"))
        accel = self.router(("zoom", "cc:1:10:relative_0_127:accel"))
        assert inverted.route(cc(1, 10, 0))[0].value == -0.05
        assert inverted.route(cc(1, 10, 127))[0].value == 0.05
        assert accel.route(cc(1, 10, 0))[0].value == 0.1
        assert accel.route(cc(1, 10, 127))[0].value == -0.1
        assert plain.route(cc(1, 10, 64)) == []

    def test_note_on_velocity_zero_is_note_off(self):
        router = self.router(("pip_enabled", "note:1:60:momentary"))
        assert router.route(note(1, 60)) == [
            ControlEvent("pip_enabled", "set_bool", True)
        ]
        assert router.route(note(1, 60, 0)) == [
            ControlEvent("pip_enabled", "set_bool", False)
        ]
        assert router.route(Incoming("note_off", 0, 60, 0)) == [
            ControlEvent("pip_enabled", "set_bool", False)
        ]

    def test_toggle_is_edge_triggered(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        router = self.router(("tracking_enabled", "cc:1:20:toggle"))
        toggle = ControlEvent("tracking_enabled", "toggle")
        assert router.route(cc(1, 20, 127)) == [toggle]
        assert router.route(cc(1, 20, 127)) == []  # still held
        assert router.route(cc(1, 20, 100)) == []
        assert router.route(cc(1, 20, 0)) == []
        clock["t"] += 0.2  # a real second press, past the calm window
        assert router.route(cc(1, 20, 127)) == [toggle]

    def test_rapid_button_chatter_produces_one_toggle(self, monkeypatch):
        """One physical press that bounces must not toggle on/off/on/off."""
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        router = MidiRouter(
            [parse_mapping("tracking_enabled", "cc:1:20:toggle")], 1,
            debounce_ms=80,
        )
        toggle = ControlEvent("tracking_enabled", "toggle")
        events = []
        for index, value in enumerate([127, 0, 127, 0, 127, 0, 127, 0]):
            clock["t"] = 1000.0 + index * 0.01
            events.extend(router.route(cc(1, 20, value)))
        assert events == [toggle]

        # A genuine later press still works.
        clock["t"] += 0.2
        events.extend(router.route(cc(1, 20, 127)))
        assert events == [toggle, toggle]

    def test_note_toggle_fires_once_per_press(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        router = self.router(("lut_enabled", "note:1:61:toggle"))
        toggle = ControlEvent("lut_enabled", "toggle")
        assert router.route(note(1, 61)) == [toggle]
        assert router.route(note(1, 61)) == []
        assert router.route(note(1, 61, 0)) == []
        clock["t"] += 0.2
        assert router.route(note(1, 61)) == [toggle]

    def test_momentary_hold_and_release(self):
        router = self.router(("sharpen_enabled", "note:1:62:momentary"))
        assert router.route(note(1, 62)) == [
            ControlEvent("sharpen_enabled", "set_bool", True)
        ]
        assert router.route(note(1, 62, 0)) == [
            ControlEvent("sharpen_enabled", "set_bool", False)
        ]
        inverted = self.router(("sharpen_enabled", "note:1:62:momentary:invert"))
        assert inverted.route(note(1, 62))[0].value is False

    def test_channel_filtering_global_and_per_mapping(self):
        global_one = self.router(("zoom", "cc:1:10:absolute"), channel=1)
        assert global_one.route(cc(2, 10, 127)) == []
        assert global_one.route(cc(1, 10, 127))[0].value == 1.0
        any_channel = self.router(("zoom", "cc:0:10:absolute"), channel=0)
        assert any_channel.route(cc(16, 10, 127))[0].value == 1.0
        explicit = self.router(("zoom", "cc:2:10:absolute"), channel=0)
        assert explicit.route(cc(1, 10, 127)) == []
        assert explicit.route(cc(2, 10, 127))[0].value == 1.0

    def test_one_input_can_drive_multiple_targets(self):
        router = self.router(
            ("zoom", "cc:1:10:absolute"),
            ("lut_scale", "cc:1:10:absolute"),
        )
        events = router.route(cc(1, 10, 127))
        assert {event.target for event in events} == {"zoom", "lut_scale"}

    def test_one_target_can_have_multiple_inputs(self):
        router = self.router(
            ("zoom", "cc:1:10:absolute"),
            ("zoom", "cc:1:11:relative_twos"),
        )
        assert router.route(cc(1, 10, 127))[0].op == "set_value"
        assert router.route(cc(1, 11, 2))[0].op == "step"

    def test_next_prev_and_capture(self):
        router = self.router(("lut_next", "note:1:60:trigger"))
        assert router.route(note(1, 60)) == [ControlEvent("lut_next", "trigger", None)]
        assert router.route(note(1, 60)) == []
        captured = router.capture(note(1, 60))
        assert (captured.type, captured.channel, captured.number) == ("note", 1, 60)
        assert str(captured) == "note:1:60"
        assert router.capture(types.SimpleNamespace(type="clock", channel=0)) is None
        assert router.learn_mapping(captured, "lut_next") == "note:1:60:trigger"
        assert router.learn_mapping(captured, "zoom") == "note:1:60:absolute"

    def test_set_mappings_replaces_routing_at_runtime(self):
        router = self.router(("zoom", "cc:1:10:absolute"))
        router.set_mappings([parse_mapping("lut_scale", "cc:1:11:absolute")])
        assert router.route(cc(1, 10, 127)) == []
        assert router.route(cc(1, 11, 64))[0].target == "lut_scale"


class TestFeedback:
    @staticmethod
    def router(*specs):
        return MidiRouter([parse_mapping(*spec) for spec in specs], 1)

    def test_boolean_and_continuous_feedback_messages(self):
        router = self.router(
            ("tracking_enabled", "cc:1:20:toggle"),
            ("sharpen_enabled", "note:1:62:momentary"),
            ("zoom", "cc:1:10:absolute"),
        )
        assert router.feedback(router.mappings[0], True) == Feedback("cc", 1, 20, 127)
        assert router.feedback(router.mappings[0], False) == Feedback("cc", 1, 20, 0)
        assert router.feedback(router.mappings[1], True).type == "note_on"
        assert router.feedback(router.mappings[1], False).type == "note_off"
        assert router.feedback(router.mappings[2], 4.0) == Feedback("cc", 1, 10, 127)
        assert router.feedback(router.mappings[2], 1.0) == Feedback("cc", 1, 10, 0)

    def test_dedupe_throttle_and_loopback_guard(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        router = self.router(("tracking_enabled", "cc:1:20:toggle"))
        feedback = router.feedback(router.mappings[0], True)
        assert router.should_send(feedback, 20)
        assert not router.should_send(feedback, 20)  # same value
        other = router.feedback(router.mappings[0], False)
        assert not router.should_send(other, 20)  # throttled
        clock["t"] += 0.05
        assert router.should_send(other, 20)
        assert router.is_loopback(other)
        clock["t"] += 0.1
        assert not router.is_loopback(other)


class FakePort:
    def __init__(self, name, callback=None):
        self.name = name
        self.callback = callback
        self.closed = False
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


class FakeMido:
    def __init__(self):
        self.inputs = ["In A"]
        self.outputs = ["Out A"]
        self.in_ports = []
        self.out_ports = []

    def get_input_names(self):
        return list(self.inputs)

    def get_output_names(self):
        return list(self.outputs)

    def open_input(self, name, callback=None):
        port = FakePort(name, callback)
        self.in_ports.append(port)
        return port

    def open_output(self, name):
        port = FakePort(name)
        self.out_ports.append(port)
        return port

    @staticmethod
    def Message(msg_type, **kwargs):
        assert msg_type in ("control_change", "note_on", "note_off")
        return types.SimpleNamespace(type=msg_type, **kwargs)


class TestManager:
    @staticmethod
    def make_manager(**overrides):
        settings = make_cfg(midi_enabled=True, **overrides)
        manager = MidiManager(
            settings,
            [
                parse_mapping("zoom", "cc:1:10:absolute"),
                parse_mapping("lut_enabled", "note:1:60:toggle"),
            ],
        )
        fake = FakeMido()
        manager._mido = fake
        return manager, fake

    def test_manager_uses_configured_debounce(self):
        manager, _ = self.make_manager(midi_button_debounce=250)
        assert manager.router.debounce_seconds == 0.25
        manager.stop()

    def test_no_backend_is_graceful_and_app_starts(self):
        manager = MidiManager(make_cfg(midi_enabled=True), [])
        manager._mido = None
        manager._backend_error = "no backend"
        assert manager.list_devices()["error"] == "no backend"
        manager.start()
        assert manager.status == "no backend"
        assert not manager.connected
        manager.stop()

    def test_note_messages_use_velocity_not_value(self):
        """mido note messages have .velocity; they must never be dropped."""
        msg = types.SimpleNamespace(
            type="note_on", channel=0, note=60, velocity=127
        )
        router = MidiRouter([parse_mapping("lut_enabled", "note:1:60:toggle")], 1)
        assert router.route(
            Incoming("note_on", 0, msg.note, normalize_value(msg))
        ) == [ControlEvent("lut_enabled", "toggle")]

    def test_missing_configured_device_is_not_replaced(self):
        manager, fake = self.make_manager(midi_input_device="Missing")
        manager.start()
        assert "not found" in manager.status
        assert manager._in_port is None  # no silent fallback to another input
        manager.stop()

    def test_input_dispatch_and_feedback_output(self):
        received = []
        arrived = threading.Event()
        manager, fake = self.make_manager()

        def on_event(event):
            received.append(event)
            arrived.set()

        manager.on_event = on_event
        manager.start()
        assert any(port.name == "In A" for port in fake.in_ports)
        fake.in_ports[-1].callback(
            types.SimpleNamespace(
                type="control_change", channel=0, control=10, value=127
            )
        )
        assert arrived.wait(2.0)
        assert received[0] == ControlEvent("zoom", "set_value", 1.0)

        arrived.clear()
        fake.in_ports[-1].callback(
            types.SimpleNamespace(
                type="note_on", channel=0, note=60, velocity=127
            )
        )
        assert arrived.wait(2.0)
        assert received[1] == ControlEvent("lut_enabled", "toggle")

        manager.send_state("zoom", 4.0)
        for _ in range(50):
            if fake.out_ports and fake.out_ports[-1].sent:
                break
            time.sleep(0.02)
        assert fake.out_ports[-1].sent[0].type == "control_change"
        assert fake.out_ports[-1].sent[0].value == 127
        manager.stop()

    def test_feedback_does_not_swallow_button_release(self, monkeypatch):
        """press -> feedback -> release -> press must toggle twice."""
        clock = {"t": 1000.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        received, arrived = [], threading.Event()
        settings = make_cfg(midi_enabled=True, midi_feedback=True)
        manager = MidiManager(
            settings, [parse_mapping("tracking_enabled", "cc:1:20:toggle")]
        )
        fake = FakeMido()
        manager._mido = fake

        def on_event(event):
            received.append(event)
            arrived.set()

        manager.on_event = on_event
        manager.start()
        port = fake.in_ports[-1]

        def send(value):
            port.callback(
                types.SimpleNamespace(
                    type="control_change", channel=0, control=20, value=value
                )
            )

        toggle = ControlEvent("tracking_enabled", "toggle")
        send(127)  # press
        assert arrived.wait(2.0)
        assert received[-1] == toggle

        manager.send_state("tracking_enabled", True)  # LED feedback for the press
        arrived.clear()
        clock["t"] += 0.02  # quick tap: release only 20 ms after the feedback
        send(0)  # release -- must not be swallowed by the loop guard
        time.sleep(0.2)
        assert len(received) == 1  # releases never produce a toggle

        arrived.clear()
        clock["t"] += 0.08
        send(127)  # next press must be a fresh edge
        assert arrived.wait(2.0)
        assert len(received) == 2
        assert received[-1] == toggle
        manager.stop()

    def test_note_velocity_zero_release_is_not_swallowed(self, monkeypatch):
        """Note On velocity 0 is a release even right after note feedback."""
        clock = {"t": 500.0}
        monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
        received, arrived = [], threading.Event()
        settings = make_cfg(midi_enabled=True, midi_feedback=True)
        manager = MidiManager(
            settings, [parse_mapping("lut_next", "note:1:60:trigger")]
        )
        fake = FakeMido()
        manager._mido = fake
        manager.on_event = lambda event: (received.append(event), arrived.set())
        manager.start()
        port = fake.in_ports[-1]

        def send(velocity):
            port.callback(
                types.SimpleNamespace(
                    type="note_on", channel=0, note=60, velocity=velocity
                )
            )

        trigger = ControlEvent("lut_next", "trigger", None)
        send(127)
        assert arrived.wait(2.0)
        manager.send_pulse("lut_next")
        arrived.clear()
        clock["t"] += 0.01
        send(0)  # release sent as Note On velocity 0
        time.sleep(0.2)
        clock["t"] += 0.1
        send(127)
        assert arrived.wait(2.0)
        assert received == [trigger, trigger]
        manager.stop()

    def test_disconnect_and_reconnect(self):
        manager, fake = self.make_manager(midi_auto_reconnect=True)
        manager.start()
        original = fake.in_ports[-1]
        original.closed = True  # simulate OS-level disconnect
        deadline = time.monotonic() + 3.0
        while len(fake.in_ports) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(fake.in_ports) >= 2
        manager.stop()
        assert all(port.closed for port in fake.in_ports)

    def test_learn_consumes_message_without_routing_it(self):
        events, learned = [], []
        manager, fake = self.make_manager()
        manager.on_event = events.append
        manager.on_learn = learned.append
        manager.start()
        manager.start_learn()
        fake.in_ports[-1].callback(
            types.SimpleNamespace(
                type="control_change", channel=0, control=10, value=64
            )
        )
        deadline = time.monotonic() + 2.0
        while not learned and time.monotonic() < deadline:
            time.sleep(0.02)
        assert str(learned[0]) == "cc:1:10"
        assert events == []
        assert not manager.learning
        manager.cancel_learn()
        manager.stop()

    def test_clean_shutdown_leaves_no_live_workers(self):
        manager, _ = self.make_manager()
        manager.start()
        threads = [manager._dispatcher, manager._output, manager._monitor]
        assert all(thread.is_alive() for thread in threads)
        manager.stop()
        assert not any(thread.is_alive() for thread in threads)
        manager.stop()  # idempotent


class FakePipeline:
    ident = None

    def __init__(self, cfg, state, preview=False):
        self.state = state

    def start(self):
        pass

    def stop(self):
        pass

    def join(self, timeout=None):
        pass

    def reload_shader(self):
        pass


def run_async(func):
    import asyncio
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class TestTuiBridge:
    @run_async
    async def test_midi_controls_reuse_existing_actions(self):
        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            with tempfile.TemporaryDirectory() as folder:
                cfg = make_cfg(lut_file="", shader_file="")
                app = tui.HyperStageApp(cfg, os.path.join(folder, "t.ini"))
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    # Switch toggle goes through flip_switch / Changed handler.
                    app._on_midi_event(ControlEvent("cam_flip", "toggle"))
                    await pilot.pause()
                    assert app.state.snapshot()["cam_flip"]

                    # Continuous absolute goes through apply_tuning (clamped).
                    app._on_midi_event(ControlEvent("zoom", "set_value", 0.5))
                    await pilot.pause()
                    assert app.state.snapshot()["zoom"] == pytest.approx(2.5)
                    assert app.query_one("#in-zoom", tui.Input).value == "2.50"
                    app._on_midi_event(ControlEvent("zoom", "set_value", 9.0))
                    await pilot.pause()
                    assert app.state.snapshot()["zoom"] == 4.0

                    # Relative steps use the registry step size.
                    app._on_midi_event(ControlEvent("lut_scale", "step", 1))
                    await pilot.pause()
                    assert app.state.snapshot()["lut_scale"] == 9

                    if len(app.lut_files) >= 2:
                        first = app.state.snapshot()["lut_file"]
                        app._on_midi_event(ControlEvent("lut_next", "trigger"))
                        await pilot.pause()
                        assert app.state.snapshot()["lut_file"] != first
                        assert (
                            app.query_one("#lut-select", tui.Select).value
                            == app.state.snapshot()["lut_file"]
                        )

                    if len(app.shader_files) >= 2:
                        first = app.state.snapshot()["shader_file"]
                        app._on_midi_event(ControlEvent("shader_next", "trigger"))
                        await pilot.pause()
                        assert app.state.snapshot()["shader_file"] != first
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_magic_loop_dependency_from_midi(self):
        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            cfg = make_cfg(lut_file="")
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                tracking = app.query_one("#sw-tracking", tui.Switch)
                assert tracking.value
                app._on_midi_event(ControlEvent("magic_loop_enabled", "toggle"))
                await pilot.pause()
                assert app.state.snapshot()["magic_loop_enabled"]
                assert not app.state.snapshot()["tracking_enabled"]
                assert tracking.disabled
                app._on_midi_event(ControlEvent("tracking_enabled", "toggle"))
                await pilot.pause()
                assert not app.state.snapshot()["tracking_enabled"]  # still frozen
                app._on_midi_event(ControlEvent("magic_loop_enabled", "toggle"))
                await pilot.pause()
                assert app.state.snapshot()["tracking_enabled"]
                assert not tracking.disabled
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_pip_selection_and_corner_from_midi(self):
        import cv2
        import numpy as np

        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            with tempfile.TemporaryDirectory() as folder:
                images = []
                for name in ("a.png", "b.png"):
                    path = os.path.join(folder, name)
                    cv2.imwrite(path, np.zeros((20, 30, 3), dtype=np.uint8))
                    images.append(path)
                cfg = make_cfg(lut_file="", pip_enabled=True)
                app = tui.HyperStageApp(cfg, os.path.join(folder, "t.ini"))
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    with mock.patch("os.getcwd", return_value=folder):
                        app.refresh_pip_files()
                    app.apply_pip_image(images[0])
                    await pilot.pause(0.2)
                    app._on_midi_event(ControlEvent("pip_next", "trigger"))
                    await pilot.pause(0.2)
                    assert app.state.snapshot()["pip_image"] == images[1]
                    app._on_midi_event(
                        ControlEvent("pip_corner_br", "trigger")
                    )
                    await pilot.pause()
                    assert app.state.snapshot()["pip_corner"] == 3
                    app._on_midi_event(ControlEvent("pip_index", "select_index", 0.0))
                    await pilot.pause(0.2)
                    assert app.state.snapshot()["pip_image"] == images[0]
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_app_starts_without_midi_hardware(self):
        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            cfg = make_cfg(lut_file="", midi_enabled=True)
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            with mock.patch.object(app.midi, "start"):
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    assert app.midi.router.mappings
                    assert app.state.snapshot()["tracking_enabled"]
        finally:
            tui.VideoPipeline = original

class TestContextAwareLearn:
    """Learn binds the last control the user actually interacted with."""

    @pytest.fixture(autouse=True)
    def fake_pipeline(self, monkeypatch):
        monkeypatch.setattr(tui, "VideoPipeline", FakePipeline)

    @staticmethod
    def make_app(folder, **overrides):
        return tui.HyperStageApp(
            make_cfg(lut_file="", **overrides), os.path.join(folder, "t.ini")
        )

    @staticmethod
    def notice(app):
        return str(app.query_one("#notice", tui.Static).render())

    @staticmethod
    async def touch_zoom(pilot, app, value="2.5"):
        field = app.query_one("#in-zoom", tui.Input)
        field.focus()
        field.value = value
        await pilot.press("enter")
        await pilot.pause()

    @run_async
    async def test_zoom_interaction_targets_zoom(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await self.touch_zoom(pilot, app)
                assert app._active_midi_target == "zoom"
                app.action_midi_learn()
                await pilot.pause()
                assert app.midi.learning
                assert "MIDI LEARN → Zoom" in self.notice(app)

    @run_async
    async def test_lut_scale_button_targets_lut_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                app.query_one("#inc-lutscale", tui.Button).press()
                await pilot.pause()
                assert app._active_midi_target == "lut_scale"
                app.action_midi_learn()
                await pilot.pause()
                assert "MIDI LEARN → Scale" in self.notice(app)

    @run_async
    async def test_switch_interaction_targets_tracking(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await pilot.press("t")
                await pilot.pause()
                assert app._active_midi_target == "tracking_enabled"
                app.action_midi_learn()
                await pilot.pause()
                assert "MIDI LEARN → Face Tracking" in self.notice(app)

    @run_async
    async def test_dropdown_interaction_targets_its_selection_action(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                assert len(app.lut_files) >= 2
                app.query_one("#lut-select", tui.Select).value = app.lut_files[1]
                await pilot.pause()
                assert app._active_midi_target == "lut_next"
                app.action_midi_learn()
                await pilot.pause()
                assert "MIDI LEARN → Next LUT" in self.notice(app)

    @run_async
    async def test_programmatic_update_does_not_change_active_target(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await pilot.press("t")
                await pilot.pause()
                app.apply_tuning("zoom", 3.0)
                app.state.update(zoom=2.0)
                await pilot.pause()
                assert app._active_midi_target == "tracking_enabled"

    @run_async
    async def test_widget_refresh_does_not_change_active_target(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                app.query_one("#lut-select", tui.Select).value = app.lut_files[1]
                await pilot.pause()
                assert app._active_midi_target == "lut_next"
                app.refresh_pip_files()
                await pilot.pause()
                assert app._active_midi_target == "lut_next"

    @run_async
    async def test_non_mappable_interaction_keeps_previous_target(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await self.touch_zoom(pilot, app)
                text = app.query_one("#text-input", tui.Input)
                text.focus()
                await pilot.press("h", "i", "enter")
                await pilot.pause()
                assert app._active_midi_target == "zoom"

    @run_async
    async def test_no_interaction_requests_target_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                app.action_midi_learn()
                await pilot.pause()
                assert not app.midi.learning
                assert "select a TUI control" in self.notice(app)

    @run_async
    async def test_most_recent_valid_interaction_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await pilot.press("t")
                await pilot.pause()
                app.query_one("#inc-lutscale", tui.Button).press()
                await pilot.pause()
                await self.touch_zoom(pilot, app)
                assert app._active_midi_target == "zoom"

    @run_async
    async def test_midi_traffic_does_not_change_active_target(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await self.touch_zoom(pilot, app)
                app._on_midi_event(ControlEvent("lut_enabled", "toggle"))
                app.midi.send_state("zoom", 2.0)
                await pilot.pause()
                assert app._active_midi_target == "zoom"

    @run_async
    async def test_capture_is_shown_in_learn_notice(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                await self.touch_zoom(pilot, app)
                app.action_midi_learn()
                await pilot.pause()
                message = types.SimpleNamespace(
                    type="control_change", channel=0, control=10, value=127
                )
                app._on_midi_learn(app.midi.router.capture(message))
                await pilot.pause()
                assert "MIDI LEARN → Zoom → CC 10 CH 1" in self.notice(app)

    @run_async
    async def test_lut_and_shader_stepper_buttons_cycle_and_bind(self):
        with tempfile.TemporaryDirectory() as folder:
            app = self.make_app(folder)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                assert len(app.lut_files) >= 2
                assert len(app.shader_files) >= 2

                app.query_one("#btn-lut-next", tui.Button).press()
                await pilot.pause(0.2)
                assert app.state.snapshot()["lut_file"] == app.lut_files[0]
                assert app._active_midi_target == "lut_next"
                app.action_midi_learn()
                assert tui.LEARN_TARGETS[app._learn_index] == "lut_next"
                app.midi.cancel_learn()

                app.query_one("#btn-lut-prev", tui.Button).press()
                await pilot.pause(0.2)
                assert app._active_midi_target == "lut_prev"
                app.action_midi_learn()
                assert tui.LEARN_TARGETS[app._learn_index] == "lut_prev"
                app.midi.cancel_learn()

                app.query_one("#btn-shader-next", tui.Button).press()
                await pilot.pause()
                assert app.state.snapshot()["shader_file"] == app.shader_files[0]
                assert app._active_midi_target == "shader_next"
                app.action_midi_learn()
                assert tui.LEARN_TARGETS[app._learn_index] == "shader_next"
                app.midi.cancel_learn()

                app.query_one("#btn-shader-prev", tui.Button).press()
                await pilot.pause()
                assert app._active_midi_target == "shader_prev"
                app.action_midi_learn()
                assert tui.LEARN_TARGETS[app._learn_index] == "shader_prev"

    @run_async
    async def test_pip_stepper_buttons_cycle_images_and_bind(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            images = []
            for name in ("a.png", "b.png"):
                path = os.path.join(folder, name)
                cv2.imwrite(path, np.zeros((20, 30, 3), dtype=np.uint8))
                images.append(path)
            app = self.make_app(folder, pip_enabled=True)
            async with app.run_test(size=(100, 45)) as pilot:
                await pilot.pause()
                with mock.patch("os.getcwd", return_value=folder):
                    app.refresh_pip_files()
                app.apply_pip_image(images[0])
                await pilot.pause(0.2)

                app.query_one("#btn-pip-next", tui.Button).press()
                await pilot.pause(0.2)
                assert app.state.snapshot()["pip_image"] == images[1]
                assert app._active_midi_target == "pip_next"
                app.action_midi_learn()
                await pilot.pause()
                assert "Next PiP image" in self.notice(app)
                app.midi.cancel_learn()

                app.query_one("#btn-pip-prev", tui.Button).press()
                await pilot.pause(0.2)
                assert app.state.snapshot()["pip_image"] == images[0]
                assert app._active_midi_target == "pip_prev"
                app.action_midi_learn()
                await pilot.pause()
                assert "Previous PiP image" in self.notice(app)

    @run_async
    async def test_learn_keys_start_browse_and_cancel(self):
        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            cfg = make_cfg(lut_file="", midi_enabled=True)
            app = tui.HyperStageApp(cfg, os.path.join(tempfile.mkdtemp(), "t.ini"))
            with mock.patch.object(app.midi, "start"):
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    await pilot.press("t")  # last touched: Face Tracking
                    await pilot.pause()
                    await pilot.press("ctrl+m")
                    await pilot.pause()
                    assert app.midi.learning
                    await pilot.press("ctrl+n")
                    await pilot.pause()
                    assert app._learn_index == 1
                    await pilot.press("ctrl+b")
                    await pilot.pause()
                    assert app._learn_index == 0
                    await pilot.press("ctrl+m")
                    await pilot.pause()
                    assert not app.midi.learning
        finally:
            tui.VideoPipeline = original

    @run_async
    async def test_learn_persists_and_requires_explicit_replacement(self):
        original = tui.VideoPipeline
        tui.VideoPipeline = FakePipeline
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "t.ini")
                cfg = make_cfg(lut_file="", midi_enabled=True)
                app = tui.HyperStageApp(cfg, path)
                async with app.run_test(size=(100, 45)) as pilot:
                    await pilot.pause()
                    await pilot.press("t")  # last touched: Face Tracking
                    await pilot.pause()
                    app.action_midi_learn()
                    assert app.midi.learning
                    # First target is tracking_enabled and it is already mapped.
                    captured = types.SimpleNamespace(
                        type="control_change", channel=1, control=7, value=127
                    )
                    app._on_midi_learn(app.midi.router.capture(captured))
                    await pilot.pause()
                    assert "capture again to replace" in str(
                        app.query_one("#notice", tui.Static).render()
                    )
                    assert app.cfg["midi_mappings"]["tracking_enabled"] == "cc:1:20:toggle"
                    # Second capture is the explicit replacement.
                    app._on_midi_learn(app.midi.router.capture(captured))
                    await pilot.pause()
                    assert app.cfg["midi_mappings"]["tracking_enabled"] == "cc:2:7:toggle"
                    assert not app.midi.learning
                    assert load_config(path)["midi_mappings"]["tracking_enabled"] == (
                        "cc:2:7:toggle"
                    )
        finally:
            tui.VideoPipeline = original
