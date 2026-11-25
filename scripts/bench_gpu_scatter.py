#!/usr/bin/env python3
"""A/B benchmark: banded CPU scatter vs GPU (pyopencl) scatter deposition.

Runs the real DNG burst through SuperResolution with EVERYTHING identical
between arms (warp routing OFF in both — ``use_opencl=False``) except the
scatter engine (``use_gpu_scatter`` False vs True), and reports:

  wall | scatter-phase wall (drizzle profiler) | CPU-s (user+sys incl.
  children) | nvidia-smi GPU-util & VRAM samples (GPU arm) | HR max-abs-diff.

Usage:
  venv/bin/python scripts/bench_gpu_scatter.py --roi 1024
  venv/bin/python scripts/bench_gpu_scatter.py --roi 2048
  venv/bin/python scripts/bench_gpu_scatter.py --roi 512 --arms auto,gpu
"""

import argparse
import glob
import os
import resource
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_sr.core.superres import load_raw_linear, SuperResolution  # noqa: E402


def find_burst():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.environ.get("RAW_SR_TEST_DATA"),
        os.path.join(here, "test_data"),
    ]
    for c in cands:
        if not c:
            continue
        files = sorted(glob.glob(os.path.join(c, "*.dng")))
        if len(files) >= 2:
            return c, files
    raise SystemExit("no DNG burst found")


class GpuSampler(threading.Thread):
    """Samples nvidia-smi utilization/VRAM every ~150 ms in the background."""

    def __init__(self):
        super().__init__(daemon=True)
        self.util = []
        self.mem = []
        self._stop = threading.Event()

    def run(self):
        try:
            proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "-lms", "150"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            return
        while not self._stop.is_set():
            line = proc.stdout.readline()
            if not line:
                break
            try:
                u, m = (float(x) for x in line.split(","))
            except ValueError:
                continue
            self.util.append(u)
            self.mem.append(m)
        proc.terminate()

    def stop(self):
        self._stop.set()
        self.join(timeout=3)

    def summary(self):
        if not self.util:
            return "no samples"
        u = np.array(self.util)
        m = np.array(self.mem)
        return (f"util mean={u.mean():.0f}% p95={np.percentile(u, 95):.0f}% "
                f"max={u.max():.0f}% n={len(u)} | "
                f"VRAM max={m.max():.0f} MiB")


def rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def run_arm(frames, arm, roi):
    use_gpu = arm == "gpu"
    sr = SuperResolution(ratio=2.0, use_opencl=False,
                         use_gpu_scatter=use_gpu)
    sampler = GpuSampler() if use_gpu else None
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    c0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    res = sr.run(frames)
    wall = time.perf_counter() - t0
    if sampler:
        sampler.stop()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    c1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
             + (c1.ru_utime - c0.ru_utime) + (c1.ru_stime - c0.ru_stime))
    out = {
        "engine": res.metrics.get("opencl_scatter"),
        "scatter_s": res.metrics.get("scatter_seconds"),
        "wall": wall,
        "cpu_s": cpu_s,
        "rss_mb": rss_mb(),
        "gpu": sampler.summary() if sampler else "",
        "res": res,
    }
    print(f"  [{arm:>3}] engine={out['engine']:<11} wall={wall:7.2f}s  "
          f"scatter={out['scatter_s']:7.2f}s  cpu-s={cpu_s:7.1f}  "
          f"rss={out['rss_mb']:6.0f} MB")
    if sampler:
        print(f"        GPU: {out['gpu']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi", type=int, default=1024)
    ap.add_argument("--roi-x", type=int, default=1650)
    ap.add_argument("--roi-y", type=int, default=1240)
    ap.add_argument("--arms", default="cpu,gpu",
                    help="comma list subset/order of cpu,gpu,auto")
    args = ap.parse_args()

    data_dir, files = find_burst()
    roi = (args.roi_x, args.roi_y, args.roi, args.roi)
    print(f"burst: {len(files)} frames ({data_dir})  ROI={roi}")
    frames = [load_raw_linear(f, roi=roi) for f in files]
    print(f"loaded, rss={rss_mb():.0f} MB")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = {}
    for arm in arms:
        if arm == "auto":
            t0 = time.perf_counter()
            res = SuperResolution(ratio=2.0).run(frames)
            wall = time.perf_counter() - t0
            print(f"  [auto] engine={res.metrics['opencl_scatter']} "
                  f"opencl_active={res.metrics['opencl_active']} "
                  f"wall={wall:.2f}s scatter={res.metrics['scatter_seconds']:.2f}s")
            results[arm] = {"res": res, "wall": wall}
        else:
            results[arm] = run_arm(frames, arm, args.roi)

    if "cpu" in results and "gpu" in results:
        d = float(np.abs(results["gpu"]["res"].hr.astype(np.float64)
                         - results["cpu"]["res"].hr.astype(np.float64)).max())
        dc = float(np.abs(results["gpu"]["res"].coverage.astype(np.float64)
                          - results["cpu"]["res"].coverage.astype(np.float64)).max())
        print(f"  HR max-abs-diff GPU-vs-CPU: {d:.3e}   coverage: {dc:.3e}")
    if "cpu" in results and "auto" in results:
        d = float(np.abs(results["auto"]["res"].hr.astype(np.float64)
                         - results["cpu"]["res"].hr.astype(np.float64)).max())
        print(f"  HR max-abs-diff AUTO-vs-CPU: {d:.3e}")
    hf = {a: r["res"].metrics.get("hf_gain_db") for a, r in results.items()}
    print(f"  hf_gain_db: " + "  ".join(f"{a}={v:.5f}" for a, v in hf.items()))


if __name__ == "__main__":
    main()
