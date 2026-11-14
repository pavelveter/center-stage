#!/bin/sh
# One-shot bootstrap and launcher for hyper-stage.
#
# Installs uv if missing, creates the virtual environment, installs the
# pinned dependencies from pyproject.toml/uv.lock, pre-downloads the
# MediaPipe face-detector model, and execs the TUI app.

set -e
cd "$(dirname "$0")"

# uv manages Python and the dependencies; installed to ~/.local/bin via the
# official installer when absent.
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create/sync the environment from the lockfile (idempotent, near no-op when
# everything already matches).
uv sync

# Pre-download the face-detector model so the first launch starts offline.
python_dir="$(uv run python -c 'import sys; print(sys.executable)')"
"$python_dir" - <<'PY'
import tracker
path = tracker.ensure_model()
print(f"model ready: {path}")
PY

# Hand over to the app; the virtual camera needs a real terminal session.
exec uv run tracker.py "$@"
