#!/usr/bin/env bash
#
# Build the self-contained Linux AppImage of the raw-sr desktop app.
#
#   scripts/build_appimage.sh            # -> desktop/release/raw-sr-<version>-x86_64.AppImage
#
# The AppImage bundles a relocatable CPython (python-build-standalone) with the
# runtime dependencies from desktop/requirements-appimage.txt, so users need
# nothing installed. CUDA/CuPy is NOT bundled (about 2 GB); NVIDIA users add it
# once with `raw-sr-<version>-x86_64.AppImage --install-gpu`.
#
# Needs: bash, curl, tar, sha256sum, Node.js 20+ and npm, network access.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$HERE/desktop"
BUNDLE="$DESKTOP/bundle"
CACHE="$DESKTOP/bundle-cache"

# python-build-standalone release (https://github.com/astral-sh/python-build-standalone)
PBS_TAG="${PBS_TAG:-20261124}"
PBS_PY="${PBS_PY:-3.12.16}"
PBS_FILE="cpython-${PBS_PY}+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}"

mkdir -p "$CACHE"
if [ ! -f "$CACHE/$PBS_FILE" ]; then
  echo "==> Downloading $PBS_FILE"
  curl -fL --retry 3 -o "$CACHE/$PBS_FILE.part" "$PBS_BASE/${PBS_FILE//+/%2B}"
  mv "$CACHE/$PBS_FILE.part" "$CACHE/$PBS_FILE"
fi
echo "==> Verifying checksum"
curl -fsSL --retry 3 "$PBS_BASE/SHA256SUMS" -o "$CACHE/SHA256SUMS-$PBS_TAG"
expected="$(grep " $PBS_FILE\$" "$CACHE/SHA256SUMS-$PBS_TAG" | awk '{print $1}')"
actual="$(sha256sum "$CACHE/$PBS_FILE" | awk '{print $1}')"
if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
  echo "Checksum mismatch for $PBS_FILE (expected '$expected', got '$actual')" >&2
  exit 1
fi

echo "==> Unpacking Python into $BUNDLE/python"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"
tar -xzf "$CACHE/$PBS_FILE" -C "$BUNDLE"
PY="$BUNDLE/python/bin/python3"

echo "==> Installing runtime dependencies"
PYTHONNOUSERSITE=1 "$PY" -m pip install --no-cache-dir --upgrade pip
PYTHONNOUSERSITE=1 "$PY" -m pip install --no-cache-dir -r "$DESKTOP/requirements-appimage.txt"

echo "==> Trimming the bundle"
# Console-script wrappers carry the absolute build path in their shebang and
# are not used by the app (it runs `python3 -m ...`); keep only the
# interpreter itself.
find "$BUNDLE/python/bin" -maxdepth 1 -type f ! -name 'python3*' ! -name 'python' -delete
find "$BUNDLE/python/bin" -maxdepth 1 -type l ! -name 'python3*' ! -name 'python' -delete
rm -rf "$BUNDLE/python/lib/python3."*/test "$BUNDLE/python/lib/python3."*/idlelib \
       "$BUNDLE/python/lib/python3."*/tkinter "$BUNDLE/python/lib/python3."*/turtledemo
find "$BUNDLE/python/lib" -path '*/site-packages/*/tests' -type d -prune -exec rm -rf {} +

echo "==> Smoke test of the bundled interpreter"
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$HERE" "$PY" -c \
  "import numpy, cv2, rawpy, scipy, tifffile, fastapi, uvicorn; import raw_sr.core.superres, desktop.backend.server; print('bundle imports OK')"

# Bytecode caches embed the build path; drop them all (Python recompiles).
find "$BUNDLE/python" -name '__pycache__' -type d -prune -exec rm -rf {} +
if grep -rlaF "$HERE" "$BUNDLE/python" | head -5 | grep .; then
  echo "The bundle still contains the build path $HERE (see above)." >&2
  exit 1
fi

echo "==> Building the Electron app and the AppImage"
cd "$DESKTOP"
[ -d node_modules ] || npm ci --no-audit --no-fund
npm run build
npx electron-builder --linux AppImage --publish never

ls -lh "$DESKTOP"/release/*.AppImage
