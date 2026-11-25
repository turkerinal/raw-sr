#!/usr/bin/env python3
"""nvidia-smi proof for the GPU scatter kernel at product scale.

Loads the real ROI-2048 burst, then runs ONLY the scatter deposits (9 frames
into the 4096^2 tables, exactly drizzle's GPU work) while sampling
nvidia-smi utilization inside that window. Reports per-sample utilization
during the deposit window plus the CPU-engine wall for the same work.
"""

import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_sr.core.superres import (  # noqa: E402
    _gpu_scatter_begin,
    _scatter_fast,
    load_raw_linear,
    to_drizzle_matrix,
)
from scripts.bench_gpu_scatter import GpuSampler, find_burst  # noqa: E402

roi = (1650, 1240, 2048, 2048)
data_dir, files = find_burst()
frames = [load_raw_linear(f, roi=roi) for f in files]
print(f"{len(frames)} frames loaded, ROI 2048 -> HR 4096^2")

# Same transforms the CPU arm used (translation-only here is fine: the kernel
# work is identical per frame regardless of M content).
aligns = [to_drizzle_matrix(np.eye(3), 2.0) for _ in frames]
M = np.asarray(aligns[0], np.float64) / aligns[0][2, 2]

eng, why = _gpu_scatter_begin(4096, 4096, "bilinear")
assert eng is not None, why
cl = eng._cl
q = eng.queue

# Warm: upload+kernel timing is dominated by atomics, but do one frame first
# so buffer allocs/first-touch are excluded from the sampled window.
eng.deposit(frames[0].rgb, M, 0.5, 3, "bilinear", 1.0)

s = GpuSampler()
s.start()
t0 = time.perf_counter()
for fr in frames:
    eng.deposit(fr.rgb, M, 0.5, 3, "bilinear", 1.0)
acc, cov = eng.fetch()
t_gpu = time.perf_counter() - t0
s.stop()
u = np.array(s.util)
m = np.array(s.mem)
print(f"GPU 9-frame scatter window: {t_gpu:.2f}s "
      f"({t_gpu * 1000 / len(frames):.0f} ms/frame)")
print(f"  nvidia-smi samples in window: n={len(u)} "
      f"mean={u.mean():.0f}% median={np.median(u):.0f}% "
      f"p10={np.percentile(u, 10):.0f}% max={u.max():.0f}%")
print(f"  VRAM during window: mean={m.mean():.0f} max={m.max():.0f} MiB")
print("  per-sample util:", " ".join(f"{x:.0f}" for x in u))
eng.close()

t0 = time.perf_counter()
for fr in frames:
    a, c = _scatter_fast(fr.rgb, M, 4096, 4096, 0.5, 3)
t_cpu = time.perf_counter() - t0
print(f"CPU banded _scatter_fast, same 9 frames: {t_cpu:.1f}s "
      f"= {t_cpu / t_gpu:.1f}x GPU")
