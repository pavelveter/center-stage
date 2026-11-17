import pytest

from hyper_stage.config import DEFAULTS
from hyper_stage.state import SharedState


class TestSharedState:
    def test_update_marks_dirty_and_merges_into_config(self):
        cfg = dict(DEFAULTS)
        state = SharedState(cfg)
        assert state.pending_revision() is None
        state.update(zoom=2.0)
        assert state.pending_revision() is not None
        assert state.snapshot()['zoom'] == 2.0
        assert state.config(cfg)['zoom'] == 2.0
        assert state.config(cfg)['fps'] == cfg['fps']

    def test_telemetry_does_not_dirty_the_config(self):
        state = SharedState(dict(DEFAULTS))
        state.set_telemetry(fps=29.9, tracking=True)
        assert state.pending_revision() is None
        assert state.telemetry()['fps'] == 29.9
        assert state.telemetry()['tracking']

    def test_edit_during_save_stays_pending(self):
        state = SharedState(dict(DEFAULTS))
        state.update(zoom=2.0)
        revision = state.pending_revision()
        state.update(zoom=3.0)
        state.mark_saved(revision)
        assert state.pending_revision() is not None
        state.mark_saved(state.pending_revision())
        assert state.pending_revision() is None

    def test_unknown_setting_is_rejected(self):
        state = SharedState(dict(DEFAULTS))
        with pytest.raises(KeyError):
            state.update(no_such_setting=1)

