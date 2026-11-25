#!/usr/bin/env bash
# Reproduce the 12-run HDR+ public dataset evaluation (docs/HDRPLUS_EVAL.md).
#
# Usage:
#   scripts/eval_hdrplus.sh [DATA_DIR] [OUT_DIR]
#
# Environment variables:
#   RAW_SR_PYTHON  Python interpreter to use (default: venv/bin/python)
#   RAW_SR_WRAP    Optional command prefix for memory/timeout guarding, e.g.:
#                  RAW_SR_WRAP="timeout 3000 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${1:-$REPO_ROOT/data/hdrplus}"
OUT_DIR="${2:-$REPO_ROOT/output/hdrplus}"
PY="${RAW_SR_PYTHON:-$REPO_ROOT/venv/bin/python}"

mkdir -p "$OUT_DIR"

# burst_id:subdir:suffix
RUNS=(
  "0919_20150910_131720_137:0919_20150910_131720_137:_0919_20150910_131720_137"
  "0382_20150912_130914_161:0382_20150912_130914_161:_0382_20150912_130914_161"
  "33TJ_20150810_140637_256:short/33TJ_20150810_140637_256:_33TJ_20150810_140637_256_short"
  "6G7M_20150404_132758_328:short/6G7M_20150404_132758_328:_6G7M_20150404_132758_328_short"
)

cd "$REPO_ROOT"

for entry in "${RUNS[@]}"; do
  IFS=':' read -r burst subdir suffix <<< "$entry"
  bdir="$DATA_DIR/$subdir"
  if [[ ! -f "$bdir/payload_N000.dng" ]]; then
    echo "ERROR: missing $bdir/payload_N000.dng (run scripts/fetch_hdrplus.sh first)" >&2
    exit 1
  fi

  read -r roi_x roi_y < <("$PY" -I -c \
    "import rawpy, sys; r=rawpy.imread(sys.argv[1]); h,w=r.raw_image_visible.shape; print((w//2-512)//2*2, (h//2-512)//2*2)" \
    "$bdir/payload_N000.dng")

  for ratio in 2 3 4; do
    tag="${suffix#_}"
    log_file="$OUT_DIR/log_${tag}_${ratio}x.txt"
    echo "== $tag ${ratio}x (roi=${roi_x},${roi_y},1024x1024)"

    cmd=("$PY" "$REPO_ROOT/scripts/sr_acceptance.py"
         --ratio "$ratio"
         --roi-x "$roi_x" --roi-y "$roi_y"
         --heldout-sweep
         --out-dir "$OUT_DIR"
         --out-suffix "$suffix")

    if [[ -n "${RAW_SR_WRAP:-}" ]]; then
      # shellcheck disable=SC2206
      wrap_arr=( $RAW_SR_WRAP )
      cmd=("${wrap_arr[@]}" "${cmd[@]}")
    fi

    RAW_SR_TEST_DATA="$bdir" "${cmd[@]}" > "$log_file" 2>&1 || true
    grep -E "^(S2|S3|P1)|verdict:" "$log_file" || true
  done
done

echo "ALLDONE"
