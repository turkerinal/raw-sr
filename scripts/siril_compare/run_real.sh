#!/bin/bash
cd "$(dirname "$0")"
D=data
r(){ timeout 5000 flock /tmp/rawsr/heavy.lock systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 venv/bin/python real_cmp.py "$@" 2>&1 | grep -E "^SUMMARY|Error|Traceback" ; }
r --burst $D/hdrplus/0919_20150910_131720_137 --name hdrplus0919 --roi 1504 1000
r --burst $D/own9 --name own9 --roi 1650 1240
r --burst $D/samsung14 --name samsung14 --roi 2400 520 --n-heldout 5
r --burst $D/inria_friant20 --name inria20 --roi 2080 800 --n-heldout 5
echo ALLDONE
