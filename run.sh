#!/usr/bin/env bash
#
# Launch the raw_sr Electron desktop app (self-contained bootstrap).
#
# Usage:
#   ./run.sh                  # dev mode: Vite + Electron with hot reload
#   RAW_SR_PROD=1 ./run.sh    # production: build, then launch the bundle
#   RAW_SR_DEV=1  ./run.sh    # also install requirements-dev.txt
#   RAW_SR_PYTHON=/path/to/python ./run.sh
#       Interpreter the Electron app uses for its backend (it takes priority
#       over the repo venv). It must ALREADY have the runtime + backend
#       requirements installed; run.sh checks this before launching.
#
# Any extra arguments are forwarded to `npm run dev` / `npm run preview`.
# Set RAW_SR_SKIP_PREFLIGHT=1 to launch even if another instance is detected.
# Set RAW_SR_PREFLIGHT_ONLY=1 to install Python deps, print the launch summary
# (engine diagnostics) and exit without launching — useful for CI and for
# "is my GPU seen?" checks.
#
# Everything is created inside this repository (venv/, desktop/node_modules/),
# so a fresh `git clone` needs no other setup. Nothing is installed globally.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$HERE"
VENV="$REPO_ROOT/venv"
PY="$VENV/bin/python"
DESKTOP_DIR="$REPO_ROOT/desktop"

step() { printf '\n==> %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# 1. Python 3.11+ is required.
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
#
# The stamp is a hash of the requirement files, so editing one re-installs and
# re-running unchanged is a no-op. Installation is idempotent by design.
# --------------------------------------------------------------------------- #
ROOT_REQ="$REPO_ROOT/requirements.txt"
BACKEND_REQ="$DESKTOP_DIR/backend/requirements.txt"
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
# 4. Launch summary — which interpreter, and is GPU acceleration available?
#
# Electron resolves its backend interpreter as RAW_SR_PYTHON when set, else the
# repo venv, so report the same choice here. This is diagnostic only: a missing
# pyopencl is a warning (the app still runs on the CPU), never a hard error.
# Runs after the dependency install so a fresh clone reports real capabilities.
# --------------------------------------------------------------------------- #
LAUNCH_PY="${RAW_SR_PYTHON:-$PY}"
step "Launch summary"
echo "    python:   $LAUNCH_PY"
if ! "$LAUNCH_PY" - <<'PYEOF'
import sys
try:
    import pyopencl as cl
except Exception as exc:  # noqa: BLE001 - absence is a normal CPU state
    print(f"    pyopencl: MISSING ({type(exc).__name__}: {exc})")
    print(f"              fix: {sys.executable} -m pip install -r "
          "desktop/backend/requirements.txt")
    print("    opencl:   none (CPU fallback — GPU acceleration stays off)")
    raise SystemExit(0)
try:
    devices = []
    for plat in cl.get_platforms():
        try:
            devices.extend(plat.get_devices(device_type=cl.device_type.GPU))
        except Exception:  # noqa: BLE001 - platform without a GPU
            continue
    if devices:
        dev = max(devices, key=lambda d: d.global_mem_size)
        print("    pyopencl: OK")
        print(f"    opencl:   {str(dev.name).strip()}")
    else:
        print("    pyopencl: OK")
        print("    opencl:   none (no GPU device; CPU fallback)")
except Exception as exc:  # noqa: BLE001 - probe must never abort the launch
    print("    pyopencl: OK")
    print(f"    opencl:   probe failed ({type(exc).__name__}: {exc})")
raise SystemExit(0)
PYEOF
then
  echo "    (engine probe failed — launching anyway)"
fi

if [[ "${RAW_SR_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  step "Preflight-only requested (RAW_SR_PREFLIGHT_ONLY=1); not launching"
  exit 0
fi

# --------------------------------------------------------------------------- #
# 5. Ensure Node dependencies.
#
# `npm ci` installs exactly package-lock.json into desktop/node_modules (the
# reproducible clean-clone path); a missing lock falls back to `npm install`.
# A stamp of package.json + package-lock.json makes re-runs a no-op and only
# reinstalls when the manifests change.
# --------------------------------------------------------------------------- #
if ! command -v npm >/dev/null 2>&1; then
  echo "error: 'npm' not found. Install Node.js 20+ (https://nodejs.org) and re-run." >&2
  exit 1
fi
node_major="$(node -v 2>/dev/null | sed -e 's/^v//' -e 's/[^0-9].*//' || true)"
if [[ "$node_major" =~ ^[0-9]+$ ]] && (( node_major < 20 )); then
  echo "error: Node.js 20+ required to build the desktop app, found $(node -v)." >&2
  echo "       Install a newer Node (https://nodejs.org) and re-run." >&2
  exit 1
fi
if [[ ! -d "$DESKTOP_DIR" ]]; then
  echo "error: desktop app directory not found at $DESKTOP_DIR" >&2
  exit 1
fi
cd "$DESKTOP_DIR"
NODE_STAMP="$DESKTOP_DIR/node_modules/.raw-sr-node.stamp"
node_want="$("$PY" - "$DESKTOP_DIR/package.json" "$DESKTOP_DIR/package-lock.json" <<'PYEOF'
import hashlib, os, sys
h = hashlib.sha256()
for path in sys.argv[1:]:
    try:
        with open(path, 'rb') as fh:
            h.update(fh.read())
    except FileNotFoundError:
        h.update(os.path.basename(path).encode() + b':missing\0')
print(h.hexdigest())
PYEOF
)"
node_have="$(cat "$NODE_STAMP" 2>/dev/null || true)"

if [[ -d node_modules && -z "$node_have" ]]; then
  # Existing tree from before stamps existed: trust it, then track it.
  printf '%s\n' "$node_want" > "$NODE_STAMP"
  node_have="$node_want"
fi

if [[ "$node_want" != "$node_have" ]]; then
  if [[ -f "$DESKTOP_DIR/package-lock.json" ]]; then
    step "Installing Node dependencies (npm ci)"
    if ! npm ci; then
      echo "error: 'npm ci' failed. Check network access to registry.npmjs.org" >&2
      echo "       (and delete desktop/node_modules if it is in a bad state)." >&2
      exit 1
    fi
  else
    step "Installing Node dependencies (npm install; no package-lock.json)"
    if ! npm install; then
      echo "error: 'npm install' failed. Check network access to registry.npmjs.org." >&2
      exit 1
    fi
  fi
  printf '%s\n' "$node_want" > "$NODE_STAMP"
else
  step "Node dependencies present"
fi

# --------------------------------------------------------------------------- #
# 6. Preflight checks for the launch itself.
# --------------------------------------------------------------------------- #
# Electron uses RAW_SR_PYTHON (if set) for the backend, in preference to the
# repo venv — so it must already have the dependencies installed there.
if [[ -n "${RAW_SR_PYTHON:-}" ]]; then
  if [[ -x "$RAW_SR_PYTHON" ]] || command -v "$RAW_SR_PYTHON" >/dev/null 2>&1; then
    if ! "$RAW_SR_PYTHON" -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
      echo "error: RAW_SR_PYTHON=$RAW_SR_PYTHON cannot import fastapi/uvicorn." >&2
      echo "       The Electron app would start its backend with that interpreter" >&2
      echo "       and fail. Fix it with:" >&2
      echo "         $RAW_SR_PYTHON -m pip install -r '$BACKEND_REQ'" >&2
      echo "       or unset RAW_SR_PYTHON to use the repo venv." >&2
      exit 1
    fi
  fi
fi

# Electron holds a single-instance lock: a second launch exits silently while
# its Vite server keeps running. Detect that up front instead of confusing
# the user (RAW_SR_SKIP_PREFLIGHT=1 downgrades this to a warning).
if command -v pgrep >/dev/null 2>&1; then
  running="$(pgrep -f 'node_modules/electron/dist/electron' 2>/dev/null | tr '\n' ' ' || true)"
  if [[ -n "$running" ]]; then
    if [[ "${RAW_SR_SKIP_PREFLIGHT:-0}" == "1" ]]; then
      step "Warning: another desktop instance seems to be running (pids: $running) - Electron may refuse this launch"
    else
      echo "error: a raw_sr desktop instance is already running (pids: $running)." >&2
      echo "       Close that app window (or kill those pids) and re-run," >&2
      echo "       or set RAW_SR_SKIP_PREFLIGHT=1 to launch anyway." >&2
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# 7. Launch.
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
