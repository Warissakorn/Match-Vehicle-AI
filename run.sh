#!/usr/bin/env bash
#
# Launcher for the Match-Vehicle-AI vehicle Re-ID GUI (Linux / macOS).
#
# On first run it creates a local virtual environment, installs the
# dependencies, then opens the Tkinter desktop app. On later runs it reuses the
# environment and only reinstalls when requirements.txt has changed, so startup
# is fast. Just run:  ./run.sh   (or double-click on some desktops).
#
set -euo pipefail

# Always operate from the project root (the directory this script lives in).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
STAMP="$VENV_DIR/.requirements.sha256"

# Pick an available Python 3 interpreter.
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: Python 3 not found. Please install Python 3.10+ and retry." >&2
    exit 1
fi

# Create the virtual environment if it does not exist yet.
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install / update dependencies only when requirements.txt changed.
REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$REQ_HASH" ]; then
    echo "Installing dependencies (first run or requirements changed) ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo "$REQ_HASH" > "$STAMP"
fi

echo "Launching Match-Vehicle-AI GUI ..."
exec python app/gui.py "$@"
