"""Tests for the parallel registration pool (superres.py v1.4.0).

The pool is only allowed to change *when* frames are registered, never *what*
is registered: per-frame results are collected BY INDEX and the ECC fit is
bit-identical across cv2 thread counts (measured on real ROI-2048 frames), so

* pooled transforms == serial transforms EXACTLY (same bytes), and
* the final HR image is bit-identical pooled vs serial (max-abs-diff 0).

Also pins the auto-gate (small frames must stay serial; explicit N must pool)
and the serial fallback when the environment cannot host child processes.
All bursts are tiny synthetic frames; the pool tests spawn real processes, so
the file is kept to a few small runs and still finishes in seconds.
"""

import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import raw_sr.core.superres as core  # noqa: E402
from raw_sr.core.superres import (  # noqa: E402
    RawFrame,
    SuperResolution,
    _map_in_processes,
    _REG_POOL_MIN_REF_PIXELS,
)

_HAS_CPUS = (os.cpu_count() or 1) >= 4


# --------------------------------------------------------------------------- #
# Synthetic burst with known per-frame homographies (texture-rich so the ECC
# path does real work; same generator style as test_superres_ghosting).
# --------------------------------------------------------------------------- #

def _rigid(dx, dy, ang_deg, persp, cx, cy):
    th = np.deg2rad(ang_deg)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    T2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    P = np.eye(3)
    P[2, 0], P[2, 1] = persp
    Tr = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
    return Tr @ T2 @ P @ R @ T1


def _synth_burst(n=5, size=128, seed=7):
    rng = np.random.default_rng(seed)
    truth = cv2.GaussianBlur(rng.random((2 * size, 2 * size, 3),
                                        dtype=np.float32), (0, 0), 1.0)
    shifts = [(0.0, 0.0), (0.5, 0.1), (-0.3, 0.6), (0.2, -0.4), (-0.6, -0.2)]
    frames = []
    for i in range(n):
        dx, dy = shifts[i % len(shifts)]
        A = _rigid(dx, dy, 0.3 * (i - n // 2) / n, (2e-5, -1e-5),
                   size / 2.0, size / 2.0)
        xs, ys = np.meshgrid(np.arange(size, dtype=np.float32),
                             np.arange(size, dtype=np.float32))
        pts = np.stack([xs, ys, np.ones_like(xs)]).reshape(3, -1)
        pr = A @ pts
        pr = pr[:2] / pr[2]
        mx = (2.0 * pr[0]).reshape(size, size).astype(np.float32)
        my = (2.0 * pr[1]).reshape(size, size).astype(np.float32)
        lr = cv2.remap(truth, mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        lr = np.clip(lr + rng.normal(0, 0.003, lr.shape).astype(np.float32),
                     0, 1)
        frames.append(RawFrame(name=f"p{i}", rgb=lr,
                               green=lr[:, :, 1].copy()))
    return frames


@pytest.fixture(scope="module")
def serial_vs_pooled():
    """Run the same burst serially and with a forced pool, once, for reuse.

    ``use_opencl=False`` pins the deposit to the deterministic CPU engine:
    the bit-identity this fixture feeds is about the registration POOL, and
    the optional GPU scatter is deliberately not bit-reproducible run to run
    (float32 atomics, unordered).
    """
    frames = _synth_burst()
    common = dict(ratio=2.0, blur_threshold_ratio=0.0, use_opencl=False)
    res_serial = SuperResolution(**common, registration_workers=1).run(frames)
    res_pool = SuperResolution(**common, registration_workers=2).run(frames)
    return res_serial, res_pool


class TestPooledEquivalence:
    def test_reference_selection_matches(self, serial_vs_pooled):
        a, b = serial_vs_pooled
        assert b.metrics["reference_index"] == a.metrics["reference_index"]
        assert b.kept_indices == a.kept_indices

    @pytest.mark.skipif(not _HAS_CPUS, reason="pool needs >= 4 CPUs")
    def test_transforms_bit_identical(self, serial_vs_pooled):
        a, b = serial_vs_pooled
        assert len(b.transforms) == len(a.transforms)
        assert len(a.transforms) >= 3  # guard: the burst really registered
        for ta, tb in zip(a.transforms, b.transforms):
            assert tb.tobytes() == ta.tobytes()

    @pytest.mark.skipif(not _HAS_CPUS, reason="pool needs >= 4 CPUs")
    def test_final_hr_bit_identical(self, serial_vs_pooled):
        a, b = serial_vs_pooled
        assert b.hr.shape == a.hr.shape and b.hr.dtype == a.hr.dtype
        d = float(np.abs(b.hr.astype(np.float64)
                         - a.hr.astype(np.float64)).max())
        assert d == 0.0
        assert np.array_equal(b.coverage, a.coverage)
        assert b.shifts == a.shifts
        assert b.align_scores == a.align_scores
        assert b.align_residuals == a.align_residuals
        assert b.weights.tolist() == a.weights.tolist()

    def test_pool_repeated_runs_bit_identical(self):
        frames = _synth_burst(n=4, size=96)
        r1 = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                             registration_workers=2).run(frames)
        r2 = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                             registration_workers=3).run(frames)
        for ta, tb in zip(r1.transforms, r2.transforms):
            assert tb.tobytes() == ta.tobytes()


# --------------------------------------------------------------------------- #
# Auto gate: pool only pays off above the size threshold
# --------------------------------------------------------------------------- #

class TestAutoGate:
    @pytest.mark.skipif(not _HAS_CPUS, reason="auto pools need >= 4 CPUs")
    def test_auto_serial_below_threshold(self):
        sr = SuperResolution()
        side = int(np.sqrt(_REG_POOL_MIN_REF_PIXELS)) - 100
        assert sr._registration_workers(9, (side, side)) == 1
        assert sr._registration_workers(1, (4096, 4096)) == 1  # one task

    @pytest.mark.skipif(not _HAS_CPUS, reason="auto pools need >= 4 CPUs")
    def test_auto_pools_above_threshold(self):
        sr = SuperResolution()
        n = sr._registration_workers(9, (2048, 2048))
        ncpu = os.cpu_count()
        assert n >= 2 and n <= min(core._REG_POOL_MAX_WORKERS, 9, ncpu - 1)

    def test_forced_one_is_serial(self):
        sr = SuperResolution(registration_workers=1)
        assert sr._registration_workers(9, (4096, 4096)) == 1

    def test_forced_n_capped_by_tasks(self):
        sr = SuperResolution(registration_workers=64)
        assert sr._registration_workers(5, (2048, 2048)) == 5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SuperResolution(registration_workers=0)


# --------------------------------------------------------------------------- #
# Fallback + order guarantee of the pool primitive
# --------------------------------------------------------------------------- #

def test_pool_failure_falls_back_to_serial(monkeypatch):
    frames = _synth_burst(n=4, size=96)
    common = dict(ratio=2.0, blur_threshold_ratio=0.0, use_opencl=False)
    ref = SuperResolution(**common, registration_workers=1).run(frames)

    def boom(*a, **k):
        raise RuntimeError("simulated pool failure")

    monkeypatch.setattr(core, "_map_in_processes", boom)
    res = SuperResolution(**common, registration_workers=3).run(frames)
    assert np.abs(res.hr.astype(np.float64)
                  - ref.hr.astype(np.float64)).max() == 0.0
    for ta, tb in zip(ref.transforms, res.transforms):
        assert tb.tobytes() == ta.tobytes()


def _square_after(payload):
    """Module-level picklable worker: sleeps out of order, returns by index."""
    i, delay = payload
    if delay:
        import time
        time.sleep(delay)
    return i, i * i


def test_map_in_processes_orders_by_index():
    # Payload 0 is slow: completion order != submission order unless the map
    # really assembles results BY INDEX.
    payloads = [(0, 0.6), (1, 0.0), (2, 0.0), (3, 0.0)]
    if not _HAS_CPUS:
        pytest.skip("pool needs >= 4 CPUs")
    out = _map_in_processes(_square_after, payloads, 3)
    # The map returns each task's result (its first tuple element is the index
    # key), assembled in submission order even though task 0 finishes last.
    assert out == [0, 1, 4, 9]

