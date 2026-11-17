# src-layout and pytest migration

## Intent

Move all application modules into an installable `src/hyper_stage/` package,
convert the suite from `unittest` to `pytest`, and split the two monolithic
test files into focused modules. Keep runtime behaviour and the user's
uncommitted `hyper-stage.ini` PiP edits unchanged.

## Decisions

- **Absolute imports** (`from hyper_stage.framebus import ...`) throughout;
  `src/` added to pytest's `pythonpath`, and the project installed editable by
  `uv sync` via a new setuptools build-system section.
- **Console script** `hyper-stage = hyper_stage.cli:main`; `main.py` at the repo
  root keeps working with an updated import, as does `start.sh`.
- **`app.tcss` moves into the package**: Textual resolves a relative `CSS_PATH`
  against the App subclass's module, so `CSS_PATH = "app.tcss"` stays valid.
- **pytest only**: plain `assert`, `pytest.raises`, `pytest.skip`, and a small
  `run_async` decorator in `conftest.py` for the Textual async tests (avoids a
  new `pytest-asyncio` dependency). `unittest.mock` remains as a stdlib utility.
- Test modules: `test_framebus.py`, `test_workers.py`, `test_controllers.py`,
  `test_effects_and_lut.py`, `test_shaders.py`, `test_tui.py`, plus
  `test_config.py` and `test_state.py` so every existing test has a logical
  home. Slow/integration tests keep their timing-based assertions.

## Verification

- `.venv/bin/pytest` → all tests pass, exit code 0.
- `git diff --check` clean; `hyper-stage.ini` diff identical to the pre-existing
  user edit.
