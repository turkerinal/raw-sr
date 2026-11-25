#!/usr/bin/env python3
"""Benchmark the SR core on a real RAW burst: CPU vs OpenCL (brand-agnostic GPU).

Runs `raw_sr.core.superres.SuperResolution` with `use_opencl=False` vs
`use_opencl=True` on the SAME decoded frames and prints a compact table
(wall seconds, speedup, opencl_active). The OpenCL arm is skipped cleanly
when unavailable (no device, or the installed core predates the flag).

To keep runs cheap and safe against parallel profiling, the pipeline is
ALWAYS run on a centered ROI (default 512 px, hard cap 1024 px) — never
full-frame by default.

Usage (from the repository root, using the venv):
    venv/bin/python desktop/scripts/bench_sr.py [burst.dng ...] \
        [--roi 512] [--ratio 2.0] [--data-dir DIR]

With no paths, RAW files are looked up in --data-dir, then <repo>/test_data,
then $RAW_SR_TEST_DATA.
Exits 0 when at least the CPU arm ran, 1 on failure (or <2 RAW frames).
"""
from __future__ import annotations

import argparse
import glob
import inspect
import os
import sys
import time

import cv2  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MAX_ROI_SIDE = 1024  # hard cap: this bench must never run full-frame
EXTENSIONS = ('dng', 'cr2', 'cr3', 'nef', 'arw', 'raf', 'rw2', 'orf', 'pef')


def find_default_paths() -> list[str]:
    candidates = [
        os.environ.get('RAW_SR_TEST_DATA'),
        os.path.join(REPO_ROOT, 'test_data'),
    ]
    for d in (c for c in candidates if c):
        files: list[str] = []
        for ext in EXTENSIONS:
            files += glob.glob(os.path.join(d, f'*.{ext}'))
            files += glob.glob(os.path.join(d, f'*.{ext.upper()}'))
        if len(files) >= 2:
            return sorted(set(files))
    return []


def opencl_available() -> bool:
    try:
        return bool(cv2.ocl.haveOpenCL())
    except Exception:  # noqa: BLE001
        return False


def opencl_active_after(sr: object, res: object, supports_flag: bool):
    """Best effort: did the run actually use an OpenCL path?

    Prefers an explicit flag from the core (attribute on the engine or the
    result). OpenCV's global runtime setting is consulted only when the core
    understands `use_opencl` (a core without the flag never toggles it, so
    its value would just echo OpenCV's default and mislead). Returns None
    when nothing authoritative is known.
    """
    for source in (sr, res):
        for attr in ('opencl_active', 'used_opencl', 'use_opencl'):
            value = getattr(source, attr, None)
            if isinstance(value, bool):
                return value
    if supports_flag:
        try:
            if callable(getattr(cv2.ocl, 'useOpenCL', None)):
                return bool(cv2.ocl.useOpenCL())
        except Exception:  # noqa: BLE001
            pass
    return None


def build_sr(core_cls, ratio: float, use_opencl: bool):
    """Instantiate the core forcing the OpenCL flag, or None if unsupported."""
    params = inspect.signature(core_cls.__init__).parameters
    kwargs = {'ratio': ratio}
    if 'use_opencl' not in params:
        return None
    kwargs['use_opencl'] = use_opencl
    return core_cls(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('paths', nargs='*', help='RAW burst files (default: auto-discover)')
    parser.add_argument('--data-dir', default=None, help='directory to auto-discover RAW files in')
    parser.add_argument('--roi', type=int, default=512,
                        help=f'centered ROI side in px (default 512, max {MAX_ROI_SIDE})')
    parser.add_argument('--ratio', type=float, default=2.0, help='SR ratio (default 2.0)')
    args = parser.parse_args()

    if args.roi < 8 or args.roi > MAX_ROI_SIDE:
        parser.error(f'--roi must be between 8 and {MAX_ROI_SIDE} '
                     '(this bench never runs full-frame by default)')

    paths = list(args.paths)
    if not paths and args.data_dir:
        paths = sorted(p for e in EXTENSIONS
                       for p in glob.glob(os.path.join(args.data_dir, f'*.{e}')))
    if not paths:
        paths = find_default_paths()
    if len(paths) < 2:
        print(f'bench: fewer than 2 RAW frames found ({len(paths)}). Skipping.')
        return 1

    from raw_sr.core.superres import SuperResolution, load_raw_linear

    ref_shape = load_raw_linear(paths[0]).rgb.shape
    h, w = ref_shape[0], ref_shape[1]
    side = min(args.roi, h, w)
    roi = ((w - side) // 2, (h - side) // 2, side, side)

    print('bench_sr: CPU vs OpenCL (brand-agnostic GPU)')
    print(f'  frames: {len(paths)}   roi: {roi}   ratio: {args.ratio}')
    frames = []
    for p in paths:
        t0 = time.perf_counter()
        frames.append(load_raw_linear(p, roi=roi))
        print(f'  loaded {os.path.basename(p)} ({time.perf_counter() - t0:.2f}s)')

    supports_flag = 'use_opencl' in inspect.signature(SuperResolution.__init__).parameters
    has_device = opencl_available()
    results: list[tuple[str, float | None, bool | None, str]] = []  # (arm, secs, active, note)

    # --- CPU arm ----------------------------------------------------------
    try:
        sr = build_sr(SuperResolution, args.ratio, False)
        if sr is None:  # core without the flag: defaults are the CPU path
            sr = SuperResolution(ratio=args.ratio)
        t0 = time.perf_counter()
        res_cpu = sr.run(frames)
        cpu_s = time.perf_counter() - t0
        cpu_active = opencl_active_after(sr, res_cpu, supports_flag)
        results.append(('cpu', cpu_s, cpu_active, ''))
    except Exception as exc:  # noqa: BLE001
        print(f'bench_sr: CPU arm FAILED: {exc}')
        return 1

    # --- OpenCL arm (skipped cleanly when unavailable) ---------------------
    if not supports_flag:
        results.append(('opencl', None, False,
                        'skipped: installed core SuperResolution has no use_opencl flag'))
    elif not has_device:
        results.append(('opencl', None, False, 'skipped: no OpenCL device (cv2.ocl.haveOpenCL()=False)'))
    else:
        try:
            sr = build_sr(SuperResolution, args.ratio, True)
            t0 = time.perf_counter()
            res_gpu = sr.run(frames)
            gpu_s = time.perf_counter() - t0
            results.append(('opencl', gpu_s, opencl_active_after(sr, res_gpu, supports_flag), ''))
            hr = getattr(res_gpu, 'hr', None)
            if hr is not None:
                print(f'  hr_shape: {tuple(hr.shape)} used: {getattr(res_gpu, "used_frames", "?")}'
                      f'/{getattr(res_gpu, "total_frames", "?")}')
        except Exception as exc:  # noqa: BLE001
            results.append(('opencl', None, False, f'skipped: OpenCL arm failed ({exc})'))

    # --- report -------------------------------------------------------------
    print()
    print(f'{"arm":8s} {"wall_s":>8s} {"speedup":>8s} {"opencl_active":>14s}  note')
    print('-' * 72)
    for arm, secs, active, note in results:
        if secs is None:
            print(f'{arm:8s} {"—":>8s} {"—":>8s} {"—":>14s}  {note}')
        else:
            speedup = cpu_s / secs if arm != 'cpu' and cpu_s > 0 else 1.0
            active_s = 'unknown' if active is None else str(active)
            print(f'{arm:8s} {secs:8.2f} {speedup:7.2f}x {active_s:>14s}  {note}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
