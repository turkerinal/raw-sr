"""Tests for the optional GPU scatter backend (superres.py, pyopencl path).

The contract this file pins:

1. The GPU engine deposits the SAME tables as :func:`_scatter_accumulate`
   within the documented float32-atomic tolerance (max rel error < 1e-5 on the
   supported set, coverage equal). Atomically unordered deposits are NOT
   bit-identical run to run — the tolerance, not equality, is the guarantee.
2. A full :meth:`SuperResolution.run` forced onto the GPU matches the forced
   CPU run within 1e-4 max-abs on the HR image.
3. Without pyopencl / without a device the whole module degrades to the
   identical CPU path (metrics report 'unavailable'), and the module import
   itself never depends on pyopencl being present.

Skips cleanly when no OpenCL GPU device exists (CPU-only machines).
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
    _gpu_scatter_begin,
    _scatter_accumulate,
    drizzle,
    translation_matrix,
)


HAS_GPU = core._gpu_scatter_env() is not None
needs_gpu = pytest.mark.skipif(
    not HAS_GPU, reason="no OpenCL GPU device / pyopencl unavailable")


def _matrices():
    """Translation (negative), rotation+perspective, fractional ratio,
    fractional ratio with a non-square HR target — the same spread used by the
    _scatter_fast-vs-generic-loop validation in test_superres_perf."""
    ratio = 2.0
    rot = cv2.getRotationMatrix2D((32.0, 24.0), 4.0, 1.0)
    Wr = np.vstack([rot, [0, 0, 1]])
    Wp = Wr @ np.array([[1, 0, 0], [0, 1, 0], [1e-5, -8e-6, 1.0]])
    return {
        "identity": core.to_drizzle_matrix(np.eye(3), ratio),
        "translation": core.to_drizzle_matrix(
            translation_matrix(-3.25, 2.75), ratio),           # negative dx
        "rotation": core.to_drizzle_matrix(rot, ratio),
        "rotation_perspective": core.to_drizzle_matrix(Wp, ratio),
        "fractional_ratio": core.to_drizzle_matrix(
            translation_matrix(0.5, -0.5), 2.3),
    }


def _burst(n=4, h=64, w=48, seed=11):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2.0)
    frames = []
    for i in range(n):
        rgb = cv2.GaussianBlur(
            np.dstack([base] * 3) + 0.02 * rng.random((h, w, 3), np.float32),
            (3, 3), 0.6)
        frames.append(RawFrame(name=str(i), rgb=rgb, green=rgb[:, :, 1]))
    return frames


def _deposit_gpu_one(src, M, oh, ow, pixfrac, subsamples, splat, wt,
                     rweight=None):
    eng, why = _gpu_scatter_begin(oh, ow, splat)
    assert eng is not None, f"GPU engine unavailable: {why}"
    try:
        eng.deposit(src, M, pixfrac, subsamples, splat, wt, rweight=rweight)
        acc, cov = eng.fetch()
    finally:
        eng.close()
    return acc, cov


# --------------------------------------------------------------------------- #
# (a) scatter tables: GPU vs CPU
# --------------------------------------------------------------------------- #

@needs_gpu
@pytest.mark.parametrize("name", list(_matrices().keys()))
@pytest.mark.parametrize("splat", ["bilinear", "area"])
def test_gpu_scatter_matches_cpu_tables(name, splat):
    M = _matrices()[name]
    src = _burst(1)[0].rgb
    oh, ow = 110, 128
    a_cpu, c_cpu = _scatter_accumulate(src, M, oh, ow, 0.5, 3, splat=splat)
    a_gpu, c_gpu = _deposit_gpu_one(src, M, oh, ow, 0.5, 3, splat, 1.0)

    sup_a = np.abs(a_cpu) > 1e-6
    sup_c = c_cpu > 1e-6
    assert sup_a.sum() > 100 and sup_c.sum() > 100  # guard: real support
    rel_a = (np.abs(a_gpu - a_cpu)[sup_a]
             / np.abs(a_cpu[sup_a])).max()
    rel_c = (np.abs(c_gpu - c_cpu)[sup_c] / c_cpu[sup_c]).max()
    assert rel_a < 1e-5, f"{name}/{splat} acc rel err {rel_a:.2e}"
    assert rel_c < 1e-5, f"{name}/{splat} cov rel err {rel_c:.2e}"
    # coverage support identical (no new/missing cells at 1e-6 resolution)
    assert np.array_equal(c_gpu > 1e-6, sup_c)


@needs_gpu
def test_gpu_scatter_weighted_frames_and_uneven_sizes():
    """drizzle() with frame weights + a non-square HR grid (fractional ratio).
    Mirrors the multi-frame fold the CPU path does in float64."""
    frames = _burst(3)
    aligns = [np.eye(3),
              translation_matrix(-2.3, 1.7),
              cv2.getRotationMatrix2D((24.0, 16.0), 3.0, 1.0)]
    aligns = [np.asarray(a, dtype=np.float64) for a in aligns]
    ws = [0.5, 0.3, 0.2]
    common = dict(pixfrac=0.7, subsamples=2, splat="bilinear")

    eng1 = {}
    hr_cpu, cov_cpu = drizzle(frames, aligns, 2.3, ws,
                              out_shape=(110, 148), use_gpu_scatter=False,
                              engine_out=eng1, **common)
    hr_gpu, cov_gpu = drizzle(frames, aligns, 2.3, ws,
                              out_shape=(110, 148), use_gpu_scatter=True,
                              engine_out=eng1, **common)
    assert eng1["scatter"] == "gpu"
    d_hr = np.abs(hr_gpu - hr_cpu).max()
    assert d_hr < 1e-5, f"hr max-abs {d_hr:.2e}"
    d_cov = np.abs(cov_gpu - cov_cpu).max()
    assert d_cov / max(cov_cpu.max(), 1e-9) < 1e-5


@needs_gpu
@pytest.mark.parametrize("splat", ["bilinear", "area"])
def test_gpu_scatter_matches_cpu_with_robust_weights(splat):
    """The robust-fusion per-LR-pixel ``rweight`` map must fold into BOTH the
    accumulated value and the coverage weight identically on the GPU kernel
    (max rel err < 1e-5), for both shipped splats and every matrix."""
    rng = np.random.default_rng(17)
    src = _burst(1)[0].rgb
    h, w = src.shape[:2]
    # smooth-ish weights in [0, 1] with some exact zeros (rejected pixels)
    rw = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    rw[::7, ::5] = 0.0
    for name, M in _matrices().items():
        a_cpu, c_cpu = _scatter_accumulate(src, M, 110, 128, 0.5, 3,
                                           splat=splat, rw=rw)
        a_gpu, c_gpu = _deposit_gpu_one(src, M, 110, 128, 0.5, 3, splat, 1.0,
                                        rweight=rw)
        sup_a = np.abs(a_cpu) > 1e-6
        sup_c = c_cpu > 1e-6
        assert sup_a.sum() > 100 and sup_c.sum() > 100
        rel_a = (np.abs(a_gpu - a_cpu)[sup_a]
                 / np.abs(a_cpu[sup_a])).max()
        rel_c = (np.abs(c_gpu - c_cpu)[sup_c] / c_cpu[sup_c]).max()
        assert rel_a < 1e-5, f"{name}/{splat} rw acc rel err {rel_a:.2e}"
        assert rel_c < 1e-5, f"{name}/{splat} rw cov rel err {rel_c:.2e}"


@needs_gpu
def test_full_run_robust_fusion_gpu_matches_cpu():
    """A full robust-fusion run forced onto the GPU matches the forced CPU run:
    the weights are computed on the CPU either way, so only the scatter deposit
    differs and must agree within the documented tolerance."""
    frames = _burst(4, h=96, w=96, seed=5)
    kw = dict(ratio=2.0, ibp_iterations=2, use_opencl=False,
              robust_fusion=True, robust_k=3.0, robust_iters=1)
    res_cpu = SuperResolution(**kw).run(frames)
    res_gpu = SuperResolution(**kw, use_gpu_scatter=True).run(frames)
    assert res_cpu.metrics["opencl_scatter"] == "cpu"
    assert res_gpu.metrics["opencl_scatter"] == "gpu"
    d = np.abs(res_gpu.hr - res_cpu.hr).max()
    assert d < 1e-4, f"robust run HR max-abs diff {d:.2e}"


# --------------------------------------------------------------------------- #
# (b) full run()
# --------------------------------------------------------------------------- #

@needs_gpu
def test_full_run_gpu_matches_cpu():
    frames = _burst(4, h=96, w=96, seed=5)
    kw = dict(ratio=2.0, ibp_iterations=2, use_opencl=False)
    res_cpu = SuperResolution(**kw).run(frames)
    res_gpu = SuperResolution(**kw, use_gpu_scatter=True).run(frames)
    assert res_cpu.metrics["opencl_scatter"] == "cpu"
    assert res_gpu.metrics["opencl_scatter"] == "gpu"
    d = np.abs(res_gpu.hr - res_cpu.hr).max()
    assert d < 1e-4, f"run HR max-abs diff {d:.2e}"
    assert np.abs(res_gpu.coverage - res_cpu.coverage).max() < 1e-4
    # The device name follows EITHER GPU engine: pure CPU reports None, a
    # run whose scatter deposit ran on the GPU names that device.
    assert res_cpu.metrics["opencl_device"] is None
    assert res_gpu.metrics["opencl_device"] == core.opencl_device_name()
    if core.opencl_device_name() is not None:
        assert res_gpu.metrics["opencl_device"]  # names the GPU device


@needs_gpu
def test_gpu_device_reported_when_only_scatter_uses_gpu():
    """metrics['opencl_device'] reports the GPU device when EITHER engine used
    it: warp routing OFF (CPU) + scatter deposit ON (GPU) must still name the
    device. D3: opencl_active now means "a GPU engine ran", so it is True here,
    while the separate opencl_warp stays False (it tracks the warps only)."""
    frames = _burst(3, h=64, w=64, seed=7)
    res = SuperResolution(ratio=2.0, ibp_iterations=1,
                          use_opencl=False, use_gpu_scatter=True).run(frames)
    assert res.metrics["opencl_scatter"] == "gpu"
    assert res.metrics["opencl_warp"] is False    # warps stayed on the CPU
    assert res.metrics["opencl_active"] is True   # scatter used the GPU
    assert res.metrics["opencl_device"] == core.opencl_device_name()
    if core.opencl_device_name() is not None:
        assert res.metrics["opencl_device"]  # non-None: scatter engaged the GPU


# --------------------------------------------------------------------------- #
# (c) graceful degradation
# --------------------------------------------------------------------------- #

def test_no_pyopencl_module_imports_and_runs_cpu(monkeypatch):
    """pyopencl absent (import raises ImportError) -> module still imports,
    the gate reports unavailable, and a FORCED GPU run produces the exact
    CPU result."""
    frames = _burst(3, h=64, w=64, seed=3)
    res_cpu = SuperResolution(ratio=2.0, ibp_iterations=1,
                              use_opencl=False).run(frames)

    import builtins
    real_import = builtins.__import__

    def _no_pyopencl(name, *a, **k):
        if name == "pyopencl" or name.startswith("pyopencl."):
            raise ImportError("simulated: pyopencl not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_pyopencl)
    monkeypatch.setattr(core, "_gpu_scatter_env_cache", None)
    monkeypatch.setattr(core, "_gpu_scatter_beneficial_cache", None)
    assert core._gpu_scatter_env() is None          # probed: unavailable
    assert core.gpu_scatter_beneficial() is False   # auto declines cleanly
    res = SuperResolution(ratio=2.0, ibp_iterations=1,
                          use_opencl=False, use_gpu_scatter=True).run(frames)
    assert res.metrics["opencl_scatter"] == "unavailable"
    assert res.metrics["opencl_device"] is None  # neither engine used a GPU
    assert np.array_equal(res.hr, res_cpu.hr)  # byte-identical CPU fallback


def test_auto_without_gpu_never_touches_device(monkeypatch):
    """Auto with no device behaves exactly as the pure-CPU setting."""
    frames = _burst(2, h=48, w=48, seed=9)
    monkeypatch.setattr(core, "_gpu_scatter_env_cache", False)
    monkeypatch.setattr(core, "_gpu_scatter_beneficial_cache", None)
    # Isolate the SCATTER gate from the separate warp-routing gate:
    monkeypatch.setattr(core, "opencl_beneficial", lambda: False)
    ref = SuperResolution(ratio=2.0, ibp_iterations=0,
                          use_opencl=False).run(frames)
    auto = SuperResolution(ratio=2.0, ibp_iterations=0).run(frames)
    assert auto.metrics["opencl_scatter"] == "cpu"
    assert np.array_equal(auto.hr, ref.hr)


def test_vram_guard_declines_oversized_tables():
    """Runs whose HR tables would exceed the VRAM budget stay on the CPU even
    when the GPU engine is requested (device untouched)."""
    if not HAS_GPU:
        pytest.skip("no GPU: guard logic identical, skip allocation check")
    giant = int(np.sqrt(core._OCL_SCATTER_MAX_TABLE_BYTES / 20)) + 8
    eng, why = _gpu_scatter_begin(giant, giant, "bilinear")
    assert eng is None and why == "cpu"


@needs_gpu
def test_unsupported_splat_falls_back_to_cpu():
    frames = _burst(2, h=48, w=48, seed=4)
    eng = {}
    hr, cov = drizzle(frames, [np.eye(3), translation_matrix(0.5, 0.5)],
                      2.0, [0.5, 0.5], splat="cubic", use_gpu_scatter=True,
                      engine_out=eng)
    assert eng["scatter"] == "cpu"
    assert np.isfinite(hr).all() and np.isfinite(cov).all()
