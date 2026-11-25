"""Deterministic tests for the GPU-scatter calibration and its cache policy.

``gpu_scatter_beneficial()`` decides whether AUTO routes the scatter deposit to
the GPU. It times the shipped kernel with :func:`_gpu_scatter_bench` (median of
:data:`_GPU_SCATTER_BENCH_REPS` runs per arm) and engages the GPU only when that
median is at least 5% faster. The cache policy is the fix for the "GPU works for
me, not for you" report:

* a **True** verdict is cached for the process lifetime (a measured win is
  stable);
* a **False** verdict is deliberately NOT cached, so the next call re-measures
  and one noisy sample under transient load cannot pin a whole session to the
  CPU.

Everything here is monkeypatched, so it is fast and needs no GPU.
"""

import time

import numpy as np
import pytest

import raw_sr.core.superres as core


def _fresh_cache(monkeypatch):
    monkeypatch.setattr(core, "_gpu_scatter_beneficial_cache", None)


def _have_device(monkeypatch):
    # A non-None env is all gpu_scatter_beneficial() checks before benching.
    monkeypatch.setattr(core, "_gpu_scatter_env", lambda: {"platform": "fake"})


def _bench(monkeypatch, fn):
    monkeypatch.setattr(core, "_gpu_scatter_bench", fn)


# --------------------------------------------------------------------------- #
# 5% margin
# --------------------------------------------------------------------------- #

def test_true_verdict_requires_five_percent_win(monkeypatch):
    _have_device(monkeypatch)
    _fresh_cache(monkeypatch)
    _bench(monkeypatch, lambda: (0.94, 1.00))
    assert core.gpu_scatter_beneficial() is True


def test_within_five_percent_declines(monkeypatch):
    _have_device(monkeypatch)
    _fresh_cache(monkeypatch)
    # 0.95 exactly is NOT faster ("at least 5% faster" is strict).
    _bench(monkeypatch, lambda: (0.95, 1.00))
    assert core.gpu_scatter_beneficial() is False

    _fresh_cache(monkeypatch)
    _bench(monkeypatch, lambda: (0.96, 1.00))
    assert core.gpu_scatter_beneficial() is False


# --------------------------------------------------------------------------- #
# Cache policy
# --------------------------------------------------------------------------- #

def test_true_verdict_is_cached(monkeypatch):
    _have_device(monkeypatch)
    _fresh_cache(monkeypatch)
    calls = {"n": 0}

    def bench():
        calls["n"] += 1
        return (0.5, 1.0)

    _bench(monkeypatch, bench)
    assert core.gpu_scatter_beneficial() is True
    assert core.gpu_scatter_beneficial() is True
    # A measured win is stable: benched once, then served from the cache.
    assert calls["n"] == 1


def test_false_verdict_is_remeasured_and_can_recover(monkeypatch):
    _have_device(monkeypatch)
    _fresh_cache(monkeypatch)
    results = iter([(1.0, 1.0), (0.5, 1.0)])
    calls = {"n": 0}

    def bench():
        calls["n"] += 1
        return next(results)

    _bench(monkeypatch, bench)
    # A transient (noisy) False must not stick: the next call re-measures.
    assert core.gpu_scatter_beneficial() is False
    assert core.gpu_scatter_beneficial() is True
    assert calls["n"] == 2
    # ...and the recovered True is now cached.
    assert core.gpu_scatter_beneficial() is True
    assert calls["n"] == 2


# --------------------------------------------------------------------------- #
# Graceful decline
# --------------------------------------------------------------------------- #

def test_no_device_is_false_without_benching(monkeypatch):
    monkeypatch.setattr(core, "_gpu_scatter_env", lambda: None)
    _fresh_cache(monkeypatch)
    calls = {"n": 0}

    def bench():
        calls["n"] += 1
        return (0.1, 1.0)

    _bench(monkeypatch, bench)
    assert core.gpu_scatter_beneficial() is False
    assert calls["n"] == 0        # never benched without a device


def test_bench_exception_declines(monkeypatch):
    _have_device(monkeypatch)
    _fresh_cache(monkeypatch)

    def boom():
        raise RuntimeError("simulated kernel build failure")

    _bench(monkeypatch, boom)
    assert core.gpu_scatter_beneficial() is False


# --------------------------------------------------------------------------- #
# _gpu_scatter_bench: median of N runs per arm (no GPU needed)
# --------------------------------------------------------------------------- #

class _FakeEngine:
    def __init__(self):
        self.deposits = 0

    def deposit(self, *a, **k):
        self.deposits += 1

    def fetch(self):
        return (np.zeros((4, 4, 3), np.float64), np.zeros((4, 4), np.float64))

    def close(self):
        pass


def test_gpu_scatter_bench_reports_median_not_mean(monkeypatch):
    # Median gpu reps (3) + median cpu reps (3) each call perf_counter twice.
    # The GPU warm-up run is NOT timed, so it consumes no timestamps. Feed
    # controlled deltas so median != mean and check the function returns medians.
    assert core._GPU_SCATTER_BENCH_REPS >= 3
    deltas = [2.0, 9.0, 4.0, 10.0, 12.0, 20.0]  # gpu x3, cpu x3
    values = []
    cursor = 0.0
    for d in deltas:
        values.append(cursor)
        cursor += d
        values.append(cursor)
    seq = iter(values)

    monkeypatch.setattr(time, "perf_counter", lambda: next(seq))
    monkeypatch.setattr(core, "_scatter_fast",
                        lambda *a, **k: (np.zeros((512, 512, 3)),
                                         np.zeros((512, 512))))
    monkeypatch.setattr(core, "_gpu_scatter_begin",
                        lambda *a, **k: (_FakeEngine(), "gpu"))

    t_gpu, t_cpu = core._gpu_scatter_bench()
    # gpu deltas [2,9,4] -> median 4 (mean 5); cpu [10,12,20] -> median 12
    # (mean 14). The median is the load-robust statistic the docstring promises.
    assert t_gpu == 4.0, f"gpu bench {t_gpu} is not the median of [2,9,4]"
    assert t_cpu == 12.0, f"cpu bench {t_cpu} is not the median of [10,12,20]"
