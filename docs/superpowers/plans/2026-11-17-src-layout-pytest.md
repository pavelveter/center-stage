# src-layout and pytest Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the app into `src/hyper_stage/`, convert and split the test suite to pytest, and make all entry points work with the new layout.

**Architecture:** `src/hyper_stage/` holds the installable package; top-level `tests/` holds focused pytest modules with `pythonpath = ["src"]`.

**Tech Stack:** Python 3.13, setuptools editable install, pytest, uv.

**Spec:** `docs/specs/src-layout-pytest-migration.md`

## Global Constraints

- Preserve runtime behaviour, CLI flags, INI keys and the uncommitted `hyper-stage.ini` PiP edit.
- No new test dependencies beyond pytest; async tests use a local decorator.
- All test commands run with `.venv/bin/pytest` and must exit 0.

## Review Focus

- Hidden patch targets (`mock.patch("detector.ensure_model")`) must be rewritten to package paths.
- Textual `CSS_PATH` resolution must still find `app.tcss` after the move.
- `benchmark.py` and `start.sh` must import the installed package.
- Async Textual tests must run without `pytest-asyncio`.

### Task 1: Package move and import rewrite

- [ ] `mkdir -p src/hyper_stage`, `git mv` the 16 modules and `app.tcss`, add `__init__.py`.
- [ ] Rewrite every local import to `from hyper_stage.<module> import ...`.
- [ ] Update `main.py`, `benchmark.py` and `start.sh`.
- [ ] Verify: `.venv/bin/python -c "import hyper_stage.pipeline, hyper_stage.tui"`.

### Task 2: pyproject and tooling

- [ ] Add `[build-system]`, `[project.scripts]`, `[tool.setuptools]` package config and `[tool.pytest.ini_options]`.
- [ ] `uv add --dev pytest`; sync the editable install.
- [ ] Verify: `.venv/bin/pytest --version` and `pythonpath` resolution.

### Task 3: Test conversion and split

- [ ] Convert unittest assertions to plain `assert` / `pytest.raises` / `pytest.skip`.
- [ ] Convert async tests with `run_async` from `tests/conftest.py`.
- [ ] Split into the eight focused modules; delete `test_tracker.py` and `test_pipeline.py`.
- [ ] Verify: `.venv/bin/pytest` → 100% pass, exit 0.

### Task 4: Final verification

- [ ] Run `.venv/bin/pytest`, `git diff --check`, inspect `git status`.
- [ ] Confirm `hyper-stage.ini` untouched and no stray artifacts.
