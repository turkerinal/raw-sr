"""Performance-engine consistency tests for the SR core (v1.3.0).

Two guarantees the speed work must never break:

1. The banded scatter engine (``_scatter_fast``) must be numerically equal to
   the generic per-tap deposition loop it replaced (tolerance 1e-6 — the
   measured difference is float64 round-off at ~1e-16).
2. The optional OpenCL (OpenCV T-API) routing must agree with the CPU path
   within a documented tolerance, and the CPU path must always be available
   regardless of device health. Skips cleanly when no OpenCL device exists.

All cases are small and deterministic so the whole file runs in seconds.
"""

import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raw_sr.core.superres import (  # noqa: E402
    RawFrame,
    SuperResolution,
    _scatter_accumulate,
    _scatter_fast,
    drizzle,
    iterative_back_projection,
    to_drizzle_matrix,
    translation_matrix,
)


# --------------------------------------------------------------------------- #
# Reference: the pre-v1.3.0 generic deposition loop, embedded verbatim so the
# fast engine is checked against the exact code whose output is contracted.
# --------------------------------------------------------------------------- #

def _scatter_reference(src, M, oh, ow, pixfrac, subsamples, splat="bilinear",
                       splat_sigma=0.5):
    h, w = src.shape[:2]
    M = np.asarray(M, dtype=np.float64)
    if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
        M = np.eye(3, dtype=np.float64)
    else:
        M = M / M[2, 2]

    ys, xs = np.mgrid[0:h, 0:w]
    X = xs.ravel().astype(np.float64)
    Y = ys.ravel().astype(np.float64)
    vals = src.reshape(-1, 3).astype(np.float64)

    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21 = M[2, 0], M[2, 1]

    subsamples = max(int(subsamples), 1)
    p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
    offs = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * p
    subw = 1.0 / float(subsamples * subsamples)

    acc = np.zeros(oh * ow * 3, dtype=np.float64)
    wt = np.zeros(oh * ow, dtype=np.float64)
    ar = np.arange(3)

    for oy in offs:
        Yp = Y + oy
        for ox in offs:
            Xp = X + ox
            den = m20 * Xp + m21 * Yp + 1.0
            U = (m00 * Xp + m01 * Yp + m02) / den
            V = (m10 * Xp + m11 * Yp + m12) / den
            if splat == "area":
                ui = np.floor(U + 0.5).astype(np.int64)
                vi = np.floor(V + 0.5).astype(np.int64)
                m = (ui >= 0) & (ui < ow) & (vi >= 0) & (vi < oh)
                if not m.any():
                    continue
                idx = vi[m] * ow + ui[m]
                wt += np.bincount(idx, weights=np.full(int(m.sum()), subw),
                                  minlength=oh * ow)
                idx3 = (idx[:, None] * 3 + ar[None, :]).ravel()
                acc += np.bincount(idx3, weights=(vals[m] * subw).ravel(),
                                   minlength=oh * ow * 3)
            else:
                base_u = np.floor(U).astype(np.int64)
                base_v = np.floor(V).astype(np.int64)
                tu = U - np.floor(U)
                tv = V - np.floor(V)
                wu_taps = [(0, 1.0 - tu), (1, tu)]
                wv_taps = [(0, 1.0 - tv), (1, tv)]
                for di, wu in wu_taps:
                    uu = base_u + di
                    ok_u = (uu >= 0) & (uu < ow)
                    if not ok_u.any():
                        continue
                    for dj, wv in wv_taps:
                        vv = base_v + dj
                        m = ok_u & (vv >= 0) & (vv < oh)
                        if not m.any():
                            continue
                        wgt = (wu[m] * wv[m]) * subw
                        idx = vv[m] * ow + uu[m]
                        wt += np.bincount(idx, weights=wgt, minlength=oh * ow)
                        idx3 = (idx[:, None] * 3 + ar[None, :]).ravel()
                        acc += np.bincount(
                            idx3, weights=(vals[m] * wgt[:, None]).ravel(),
                            minlength=oh * ow * 3)
    return acc.reshape(oh, ow, 3), wt.reshape(oh, ow)


def _transform_matrices():
    ang = 2e-3
    return {
        "identity": to_drizzle_matrix(np.eye(3), 2.0),
        "trans": to_drizzle_matrix((0.53, -0.29), 2.0),
        "rotpersp": to_drizzle_matrix(
            np.array([[np.cos(ang), -np.sin(ang), 0.37],
                      [np.sin(ang), np.cos(ang), -0.21],
                      [4e-6, -3e-6, 1.0]]), 2.0),
        "negoff": to_drizzle_matrix((-3.2, -2.1), 2.0),
    }


def _noisy_src(rng, shape=(24, 26, 3)):
    return rng.random(shape, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1. Fast scatter engine == generic loop
# --------------------------------------------------------------------------- #

class TestScatterEngineEquivalence:
    def test_matches_reference_across_transforms(self):
        rng = np.random.default_rng(7)
        src = _noisy_src(rng)
        oh, ow = 48, 52  # deliberately not exactly 2x: odd sizes
        for name, M in _transform_matrices().items():
            for splat in ("bilinear", "area"):
                for pf, ss in ((0.5, 3), (1.0, 2)):
                    ref = _scatter_reference(src, M, oh, ow, pf, ss, splat=splat)
                    fast = _scatter_fast(src, M, oh, ow, pf, ss, splat=splat)
                    d_acc = np.abs(fast[0] - ref[0]).max()
                    d_wt = np.abs(fast[1] - ref[1]).max()
                    assert d_acc <= 1e-6, f"acc {name}/{splat}/{pf}/{ss}: {d_acc}"
                    assert d_wt <= 1e-6, f"wt {name}/{splat}/{pf}/{ss}: {d_wt}"

    def test_matches_reference_impulses(self):
        lr = np.zeros((16, 16, 3), dtype=np.float32)
        lr[8, 3] = 1.0
        lr[2, 15] = 0.5
        for dx, dy in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.75)):
            M = to_drizzle_matrix(translation_matrix(dx, dy), 2.0)
            ref = _scatter_reference(lr, M, 32, 32, 0.5, 3)
            fast = _scatter_fast(lr, M, 32, 32, 0.5, 3)
            assert np.abs(fast[0] - ref[0]).max() <= 1e-6
            assert np.abs(fast[1] - ref[1]).max() <= 1e-6

    def test_chunk_size_does_not_matter(self):
        rng = np.random.default_rng(11)
        src = _noisy_src(rng, (20, 20, 3))
        M = to_drizzle_matrix((0.3, -0.7), 2.0)
        base = _scatter_fast(src, M, 40, 40, 0.5, 3, chunk=4)
        for chunk in (1, 7, 32, 1000):
            out = _scatter_fast(src, M, 40, 40, 0.5, 3, chunk=chunk)
            assert np.abs(out[0] - base[0]).max() <= 1e-6
            assert np.abs(out[1] - base[1]).max() <= 1e-6

    def test_dispatcher_routes_splats(self):
        # cubic/gaussian must still be served by the generic loop (the fast
        # engine only implements bilinear/area): check the generic path is
        # finite, has non-negative weights and never deposits more value than
        # the weight allows.
        rng = np.random.default_rng(3)
        src = _noisy_src(rng, (12, 12, 3))
        M = to_drizzle_matrix((0.4, 0.4), 2.0)
        for splat in ("cubic", "gaussian"):
            got = _scatter_accumulate(src, M, 24, 24, 0.5, 2, splat=splat)
            assert np.all(np.isfinite(got[0])) and np.all(np.isfinite(got[1]))
            assert np.all(got[1] >= 0)
            total_w = float(got[1].sum())
            # interior deposits carry (nearly) every sample's weight
            assert total_w > 0.5 * 12 * 12
            # every deposited value is a measured sample times its weight:
            # per channel, the accumulated value can never exceed wt * max.
            for c in range(3):
                assert np.all(got[0][..., c] <= got[1] * float(src.max())
                              + 1e-9)

    def test_drizzle_output_uses_engine(self):
        # drizzle() with the shipped (ratio-aware) defaults must reproduce a
        # manual weighted average of the reference-engine deposits.
        from raw_sr.core.superres import default_deposit_params
        pf, sub = default_deposit_params(2.0)
        rng = np.random.default_rng(5)
        frames, aligns = [], []
        for i in range(2):
            rgb = rng.random((10, 10, 3), dtype=np.float32)
            frames.append(RawFrame(name=f"m{i}", rgb=rgb,
                                   green=rgb[:, :, 1].copy()))
            aligns.append(translation_matrix(0.25 * i, -0.5 * i))
        hr, cov = drizzle(frames, aligns, 2.0, [0.6, 0.4])
        acc = np.zeros((20, 20, 3))
        wt = np.zeros((20, 20))
        for fr, al, w in zip(frames, aligns, [0.6, 0.4]):
            a, c = _scatter_reference(fr.rgb, to_drizzle_matrix(al, 2.0),
                                      20, 20, pf, sub)
            acc += a * w
            wt += c * w
        expected = np.zeros_like(acc)
        sel = wt >= 1e-6
        expected[sel] = acc[sel] / wt[sel][:, None]
        expected[~sel] = 0.0
        assert np.abs(hr.astype(np.float64) - expected).max() <= 2e-5
        assert np.abs(cov.astype(np.float64) - wt).max() <= 1e-5


# --------------------------------------------------------------------------- #
# 2. OpenCL (T-API) routing: consistency + always-available CPU fallback
# --------------------------------------------------------------------------- #

_HAS_OCL = cv2.ocl.haveOpenCL()

# Documented CPU/OpenCL agreement bound for the routed upsample warps.
# Measured deviations on real UMat execution are ~1e-3 (interpolation on the
# device rounds differently); with silent CPU fallback the difference is 0.
OCL_TOLERANCE = 2e-3


def _synthetic_burst(n=3, size=48, ratio=2.0, seed=13):
    rng = np.random.default_rng(seed)
    base = rng.random((size, size), dtype=np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.0)
    frames, aligns = [], []
    for k in range(n):
        dx, dy = (0.3 * k, -0.2 * k)
        mov = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]),
                             (size, size))
        rgb = np.stack([mov, np.roll(mov, 1, 1), np.roll(mov, 2, 1)], axis=2)
        frames.append(RawFrame(name=f"s{k}", rgb=rgb, green=mov.copy()))
        aligns.append(translation_matrix(dx, dy))
    hr = np.zeros((int(size * ratio), int(size * ratio), 3), dtype=np.float32)
    for fr in frames:
        hr += cv2.resize(fr.rgb, (hr.shape[1], hr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    hr /= len(frames)
    return hr, frames, aligns


@pytest.mark.skipif(not _HAS_OCL, reason="no OpenCL device available")
class TestOpenCLConsistency:
    def test_ibp_cpu_vs_opencl_within_tolerance(self):
        hr, frames, aligns = _synthetic_burst()
        out_cpu = iterative_back_projection(hr, frames, aligns, 2.0,
                                            iterations=2, use_opencl=False)
        out_ocl = iterative_back_projection(hr, frames, aligns, 2.0,
                                            iterations=2, use_opencl=True)
        assert out_cpu.shape == out_ocl.shape
        d = np.abs(out_cpu.astype(np.float64)
                   - out_ocl.astype(np.float64)).max()
        assert d <= OCL_TOLERANCE

    def test_drizzle_gather_cpu_vs_opencl(self):
        hr, frames, aligns = _synthetic_burst(n=2)
        a_cpu, c_cpu = drizzle(frames, aligns, 2.0, deposit="gather",
                               use_opencl=False)
        a_ocl, c_ocl = drizzle(frames, aligns, 2.0, deposit="gather",
                               use_opencl=True)
        assert np.abs(a_cpu.astype(np.float64)
                      - a_ocl.astype(np.float64)).max() <= OCL_TOLERANCE
        assert np.abs(c_cpu.astype(np.float64)
                      - c_ocl.astype(np.float64)).max() <= OCL_TOLERANCE

    def test_orchestrator_flags_and_state_restore(self):
        hr, frames, aligns = _synthetic_burst(n=2, size=32)
        prev = cv2.ocl.useOpenCL()
        sr_off = SuperResolution(ratio=2.0, ibp_iterations=1, use_opencl=False)
        res_off = sr_off.run(frames)
        assert res_off.metrics["opencl_active"] is False
        assert sr_off.opencl_active is False
        sr_on = SuperResolution(ratio=2.0, ibp_iterations=1, use_opencl=True)
        res_on = sr_on.run(frames)
        assert isinstance(res_on.metrics["opencl_active"], bool)
        assert sr_on.opencl_active == res_on.metrics["opencl_active"]
        # CPU and OpenCL orchestrations agree within the documented tolerance.
        assert np.abs(res_off.hr.astype(np.float64)
                      - res_on.hr.astype(np.float64)).max() <= OCL_TOLERANCE
        # The process-wide T-API switch must be restored by run().
        assert cv2.ocl.useOpenCL() == prev

    def test_auto_reports_bool_and_runs(self):
        _, frames, _ = _synthetic_burst(n=2, size=32)
        sr = SuperResolution(ratio=2.0, ibp_iterations=1)  # use_opencl=None
        res = sr.run(frames)
        assert isinstance(res.metrics["opencl_active"], bool)

    def test_failure_falls_back_to_cpu(self, monkeypatch):
        # Simulate a device/kernel failure inside the T-API call: the routed
        # warp must transparently return the CPU result.
        hr, frames, aligns = _synthetic_burst(n=2, size=32)
        out_cpu = iterative_back_projection(hr, frames, aligns, 2.0,
                                            iterations=1, use_opencl=False)

        real_warp = cv2.warpPerspective

        def broken_warp(src, *a, **kw):
            if isinstance(src, cv2.UMat):
                raise cv2.error("simulated device failure")
            return real_warp(src, *a, **kw)

        monkeypatch.setattr(cv2, "warpPerspective", broken_warp)
        out = iterative_back_projection(hr, frames, aligns, 2.0,
                                        iterations=1, use_opencl=True)
        assert np.abs(out.astype(np.float64)
                      - out_cpu.astype(np.float64)).max() <= 1e-6


class TestOrchestratorWithoutOpenCL:
    def test_forced_true_without_tapi_device_keeps_warps_on_cpu(self, monkeypatch):
        # No T-API OpenCL device, warp routing forced True -> warp engine runs
        # on the CPU. The pyopencl scatter engine is independent and may still
        # run; D3 makes opencl_active the OR of the two, so assert the warp
        # flag and the derived union, not a blanket False.
        monkeypatch.setattr(cv2.ocl, "haveOpenCL", lambda: False)
        _, frames, _ = _synthetic_burst(n=2, size=32)
        sr = SuperResolution(ratio=2.0, ibp_iterations=1, use_opencl=True)
        res = sr.run(frames)
        assert res.metrics["opencl_warp"] is False
        assert res.metrics["opencl_active"] == (
            res.metrics["opencl_scatter"] == "gpu")

    def test_auto_calibration_failure_keeps_warps_on_cpu(self, monkeypatch):
        # Warp auto-calibration says CPU; the scatter gate is a separate engine
        # and is unaffected (D3 opencl_warp reports the warp decision alone).
        _, frames, _ = _synthetic_burst(n=2, size=32)
        import raw_sr.core.superres as core
        monkeypatch.setattr(core, "opencl_beneficial", lambda: False)
        sr = SuperResolution(ratio=2.0, ibp_iterations=1)
        res = sr.run(frames)
        assert res.metrics["opencl_warp"] is False
        assert res.metrics["opencl_active"] == (
            res.metrics["opencl_scatter"] == "gpu")
