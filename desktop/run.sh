#!/usr/bin/env bash
#
# Bootstrap and launch the raw_sr Electron desktop app.
#
# Usage:
#   desktop/run.sh                  # dev mode: Vite + Electron with hot reload
#   RAW_SR_PROD=1 desktop/run.sh    # production: build, then launch the bundle
#   RAW_SR_DEV=1  desktop/run.sh    # also install requirements-dev.txt
#   RAW_SR_PYTHON=/path/to/python   # interpreter used by the Electron backend
#
# Any extra arguments are forwarded to `npm run dev` / `npm run preview`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$REPO_ROOT/venv"
PY="$VENV/bin/python"

step() { printf '\n==> %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# 1. Python 3.11+ is required (README).
# --------------------------------------------------------------------------- #
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found. Install Python 3.11+ or set PYTHON=/path/to/python3." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "error: Python 3.11+ required, found $("$PYTHON" --version 2>&1)." >&2
  exit 1
fi

# --------------------------------------------------------------------------- #
# 2. Ensure a virtualenv at the repository root.
# --------------------------------------------------------------------------- #
if [[ ! -x "$PY" ]]; then
  step "Creating Python virtualenv at $VENV"
  "$PYTHON" -m venv "$VENV"
else
  step "Using existing Python virtualenv at $VENV"
fi

# --------------------------------------------------------------------------- #
# 3. Install Python dependencies (root runtime + backend) when they change.
# --------------------------------------------------------------------------- #
ROOT_REQ="$REPO_ROOT/requirements.txt"
BACKEND_REQ="$REPO_ROOT/desktop/backend/requirements.txt"
REQ_FILES=("$ROOT_REQ" "$BACKEND_REQ")
if [[ "${RAW_SR_DEV:-0}" == "1" ]]; then
  REQ_FILES+=("$REPO_ROOT/requirements-dev.txt")
fi
STAMP="$VENV/.desktop-requirements.stamp"

want="$("$PY" - "${REQ_FILES[@]}" <<'PYEOF'
import hashlib, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    with open(path, 'rb') as fh:
        h.update(fh.read())
print(h.hexdigest())
PYEOF
)"
have="$(cat "$STAMP" 2>/dev/null || true)"

if [[ "$want" != "$have" ]]; then
  step "Installing Python dependencies"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT_REQ"
  "$PY" -m pip install -r "$BACKEND_REQ"
  if [[ "${RAW_SR_DEV:-0}" == "1" ]]; then
    "$PY" -m pip install -r "$REPO_ROOT/requirements-dev.txt"
  fi
  printf '%s\n' "$want" > "$STAMP"
else
  step "Python dependencies up to date"
fi

# --------------------------------------------------------------------------- #
# 4. Ensure Node dependencies.
# --------------------------------------------------------------------------- #
cd "$SCRIPT_DIR"
if [[ ! -d node_modules ]]; then
  step "Installing Node dependencies (npm install)"
  npm install
else
  step "Node dependencies present"
fi

# --------------------------------------------------------------------------- #
# 5. Launch.
# --------------------------------------------------------------------------- #
if [[ "${RAW_SR_PROD:-0}" == "1" ]]; then
  step "Building production bundle (npm run build)"
  npm run build
  step "Launching raw_sr desktop (production preview)"
  exec npm run preview -- "$@"
else
  step "Launching raw_sr desktop (development)"
  exec npm run dev -- "$@"
fi
