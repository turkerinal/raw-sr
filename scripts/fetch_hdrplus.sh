#!/usr/bin/env bash
# Fetch the 4 people-free HDR+ Burst Photography Dataset bursts used in docs/HDRPLUS_EVAL.md.
# Requires only bash, curl, and python3.
#
# Usage:
#   scripts/fetch_hdrplus.sh [--list] [TARGET_DIR]
#
# Default TARGET_DIR: ./data/hdrplus

set -euo pipefail

BASE_URL="https://storage.googleapis.com/hdrplusdata/20171106_subset/bursts"

# burst_id:frame_count (payload_N000.dng .. payload_N<count-1>.dng)
BURSTS=(
  "0919_20150910_131720_137:8"
  "0382_20150912_130914_161:8"
  "33TJ_20150810_140637_256:10"
  "6G7M_20150404_132758_328:10"
)

# Bursts whose final frame (N009) is a ~3.8x longer exposure; create N000-N008 symlink dirs.
SHORT_BURSTS=(
  "33TJ_20150810_140637_256"
  "6G7M_20150404_132758_328"
)

LIST_ONLY=0
TARGET_DIR="./data/hdrplus"

for arg in "$@"; do
  case "$arg" in
    --list)
      LIST_ONLY=1
      ;;
    -h|--help)
      echo "Usage: $0 [--list] [TARGET_DIR]"
      echo "  --list      Print download URLs only without downloading."
      echo "  TARGET_DIR  Destination directory (default: ./data/hdrplus)."
      exit 0
      ;;
    *)
      TARGET_DIR="$arg"
      ;;
  esac
done

emit_urls() {
  python3 - "$BASE_URL" "${BURSTS[@]}" <<'PY'
import sys
base = sys.argv[1]
for item in sys.argv[2:]:
    burst, count_s = item.split(":")
    for i in range(int(count_s)):
        print(f"{burst} payload_N{i:03d}.dng {base}/{burst}/payload_N{i:03d}.dng")
PY
}

if [[ "$LIST_ONLY" -eq 1 ]]; then
  emit_urls | while read -r _burst _fname url; do
    echo "$url"
  done
  exit 0
fi

mkdir -p "$TARGET_DIR"

emit_urls | while read -r burst fname url; do
  bdir="$TARGET_DIR/$burst"
  mkdir -p "$bdir"
  dest="$bdir/$fname"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "skip (exists): $dest"
    continue
  fi
  echo "fetch: $url -> $dest"
  curl -fsSL --retry 3 -o "${dest}.tmp" "$url"
  mv "${dest}.tmp" "$dest"
done

for burst in "${SHORT_BURSTS[@]}"; do
  sdir="$TARGET_DIR/short/$burst"
  mkdir -p "$sdir"
  for i in $(seq -w 0 8); do
    fname="payload_N0${i}.dng"
    ln -sfn "../../$burst/$fname" "$sdir/$fname"
  done
done

cat > "$TARGET_DIR/ATTRIBUTION.txt" <<'EOF'
Dataset: HDR+ Burst Photography Dataset (20171106_subset)
Authors: Samuel W. Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams,
         Jonathan T. Barron, Florian Kainz, Jiawen Chen, and Marc Levoy
Paper:   "Burst photography for high dynamic range and low-light imaging on mobile cameras",
         ACM Transactions on Graphics (SIGGRAPH Asia 2016), 35(6).
URL:     https://hdrplusdata.org/dataset.html
License: Creative Commons license (CC-BY-SA)

The authors request scientific use and citation. Only bursts without people
were selected for raw_sr evaluation:
  - 0919_20150910_131720_137 (building facade, 8 frames)
  - 0382_20150912_130914_161 (glass objects indoors, 8 frames)
  - 33TJ_20150810_140637_256 (brick hotel facade, 10 frames; short/ uses N000-N008)
  - 6G7M_20150404_132758_328 (victorian house, 10 frames; short/ uses N000-N008)
EOF

echo "Done. Attribution written to $TARGET_DIR/ATTRIBUTION.txt"
