"""Registration work-removal tests (A3 #5): results must not move at all.

The registration phase used to do four pieces of provably redundant work per
frame. Each is removed here (``raw_sr/core/superres.py``) and this file pins the
guarantee that made the removal safe — **the numbers do not change**:

1. ``estimate_homography`` measured the ref->mov phase correlation itself while
   ``_register_frame`` measured the same pair one line earlier; the seed is now
   passed in (``seed=``).
2. The alignment score and the residual are two statistics of the *same*
   rendered pair (moved image + coverage mask), so ``_render_in_ref_frame`` is
   computed once instead of twice.
3. The reference frame's gradient mask was rebuilt per frame
   (``ref_mask=`` is now supplied once per run).
4. Pool tasks pickled 3 full float32 frames each; they now carry slot indices
   into one shared-memory block (``_SharedRegFrames``).

Together they must be **bit-identical**, not merely close: the transforms are
compared byte for byte, and on the real burst the recorded shifts/homographies
are compared against the pre-change reference (``_B2_BASELINE``).
"""

import glob
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
    _SharedRegFrames,
    _alignment_residual_px,
    _alignment_score,
    _register_frame,
    _render_in_ref_frame,
    estimate_homography,
    estimate_translation,
)

_HAS_CPUS = (os.cpu_count() or 1) >= 4

# Real-burst reference measured on the code BEFORE the work removal
# (/tmp/rawsr/b2_reg_probe.py, ROI (1650,1240,512), 9 frames, default
# ecc_scale 0.5, auto reference index 2). The fingerprint is a decimal hash of every transform
# entry; the shifts are the reported anchor-relative translations. If a future
# refactor changes the fit by even one float, this fails.
_B2_ROI512_REF_INDEX = 2
# B8 (2026-11-20): re-pinned after correcting the ECC seed sign in
# estimate_homography (it was seeded 2*shift away from the truth; the solve
# then started from the mirror/no-lock side). The phase-correlation shifts
# above are unchanged; the warps moved by <= 0.264 px per matrix (frame 6),
# 0.002-0.06 px on most frames, and the real ROI-512 mean registration
# residual improved 0.5983 -> 0.5929 px. At ROI 1024 the warps move by ~1e-6
# (see _B4_ROI1024_* below). Anything beyond those bounds is a new regression.
_B2_ROI512_TRANSFORM_HASH = 1375213667826279932
_B2_ROI512_SHIFTS = {
    0: (0.0, 0.0),
    1: (-1.861676, -1.767301),
    2: (2.492999, 8.826927),
    3: (6.599006, 9.153162),
    4: (-1.548165, 12.264489),
    5: (5.014712, 16.606665),
    6: (0.659043, 14.816667),
    7: (-3.351208, 8.029174),
    8: (-1.812108, 15.636671),
}

# Real-burst reference measured on the code BEFORE the ROI>=1024 homography
# shortcut (2026-11-20, /tmp/rawsr/b4_dump2.py, ROI (1650,1240,1024), default
# pipeline path = registration_robust False, auto reference index 2). Per
# moving frame: the four projected corners of the ref->mov transform in LR px,
# then its edge-free alignment score and residual in px. The shortcut must not
# move the fitted homography by more than 0.01 px at the corners, nor change
# the score or residual by more than 1e-3 (the task's acceptance budget), at
# both the shipped ecc_scale and the acceptance script's 0.25.
_B4_ROI1024_REF_INDEX = 2
_B4_ROI1024_ECC050 = {
    0: ((1.631679, 10.241097), (1023.835130, 6.178034), (4.146280, 1032.640119), (1027.663300, 1029.358361), 0.997735721, 0.064644406),
    1: ((4.171455, 11.707054), (1026.235377, 9.442556), (5.348191, 1034.022813), (1028.831293, 1032.584568), 0.997863442, 0.032049698),
    3: ((-2.992894, -1.428497), (1020.221407, 3.483857), (-6.877816, 1021.335068), (1016.368771, 1025.468913), 0.998492350, 0.058125756),
    4: ((5.698585, -5.070417), (1028.597488, 1.591441), (-0.182424, 1017.582754), (1022.716479, 1024.244612), 0.997626082, 0.020443822),
    5: ((-1.079213, -9.433326), (1022.223068, -2.920244), (-6.074562, 1013.803454), (1016.515869, 1019.620751), 0.997461293, 0.019911086),
    6: ((3.500423, -7.766369), (1026.816733, -1.065901), (-1.704954, 1015.452126), (1020.999749, 1021.701193), 0.997922039, 0.022005742),
    7: ((7.936882, -1.007708), (1030.694870, 6.592097), (0.494929, 1021.984304), (1023.252917, 1029.584109), 0.997886615, 0.020416086),
    8: ((7.707510, -9.274375), (1031.045319, 2.442653), (-2.633999, 1013.171583), (1019.811482, 1024.784150), 0.997813582, 0.032013880),
}
_B4_ROI1024_ECC025 = {
    0: ((1.631679, 10.241097), (1023.835130, 6.178034), (4.146280, 1032.640119), (1027.663300, 1029.358361), 0.997735721, 0.064644406),
    1: ((4.172295, 11.707847), (1026.234933, 9.443028), (5.348000, 1034.022863), (1028.831500, 1032.584780), 0.997863442, 0.032127610),
    3: ((-2.992983, -1.428616), (1020.221652, 3.483719), (-6.877781, 1021.334947), (1016.368771, 1025.468862), 0.998492352, 0.058109821),
    4: ((5.698585, -5.070417), (1028.597488, 1.591441), (-0.182424, 1017.582754), (1022.716479, 1024.244612), 0.997626082, 0.020443822),
    5: ((-1.079143, -9.433273), (1022.223019, -2.920186), (-6.074581, 1013.803536), (1016.515884, 1019.620842), 0.997461293, 0.019911187),
    6: ((3.500423, -7.766369), (1026.816733, -1.065901), (-1.704954, 1015.452126), (1020.999749, 1021.701193), 0.997922039, 0.022005742),
    7: ((7.936882, -1.007708), (1030.694870, 6.592097), (0.494929, 1021.984304), (1023.252917, 1029.584109), 0.997886615, 0.020416086),
    8: ((7.707510, -9.274375), (1031.045319, 2.442653), (-2.633999, 1013.171583), (1019.811482, 1024.784150), 0.997813582, 0.032013880),
}


def _corners(W, roi):
    """Project the four ROI corners through a ref->mov homography."""
    W = np.asarray(W, dtype=np.float64)
    W = W / W[2, 2]
    pts = np.array([[0.0, 0.0, 1.0], [roi - 1.0, 0.0, 1.0],
                    [0.0, roi - 1.0, 1.0], [roi - 1.0, roi - 1.0, 1.0]]).T
    p = W @ pts
    return (p[:2] / p[2]).T


def _real_burst_dir():
    """Repo-relative test_data first, then ``RAW_SR_TEST_DATA`` (A3 §2.3)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.environ.get("RAW_SR_TEST_DATA"),
             os.path.join(here, "test_data")]
    for c in cands:
        if not c:
            continue
        files = sorted(glob.glob(os.path.join(c, "*.dng")))
        if len(files) >= 9:
            return c, files
    return None, []


_REAL_DIR, _REAL_FILES = _real_burst_dir()
needs_real = pytest.mark.skipif(
    not _REAL_FILES,
    reason="no 9-frame DNG burst (set RAW_SR_TEST_DATA) for the real-burst "
           "registration regression check")


def _synth_burst(n=5, size=144, seed=13):
    """Deterministic burst with known sub-pixel/rotation/perspective motion."""
    rng = np.random.default_rng(seed)
    truth = cv2.GaussianBlur(rng.random((2 * size, 2 * size, 3),
                                        dtype=np.float32), (0, 0), 1.1)
    plan = [(0.0, 0.0, 0.0, 0.0), (0.62, -0.31, 0.4, 0.0),
            (-0.44, 0.53, -0.3, 3e-6), (0.21, 0.87, 0.15, -2e-6),
            (-0.73, -0.44, 0.5, 1e-6)]
    frames = []
    for i in range(n):
        dx, dy, ang, persp = plan[i % len(plan)]
        th = np.deg2rad(ang)
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        T1 = np.array([[1, 0, -size], [0, 1, -size], [0, 0, 1.0]])
        T2 = np.array([[1, 0, size], [0, 1, size], [0, 0, 1.0]])
        P = np.eye(3)
        P[2, 0], P[2, 1] = persp, -persp
        Tr = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])
        M = Tr @ T2 @ P @ R @ T1
        H = cv2.warpPerspective(truth, np.linalg.inv(M).astype(np.float32),
                                (2 * size, 2 * size))
        lr = cv2.resize(H, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(RawFrame(name=f"f{i}", rgb=lr.copy(),
                               green=lr[:, :, 1].copy()))
    return frames


def _hash_transforms(transforms):
    h = 0
    for t in transforms:
        for v in np.asarray(t, dtype=np.float64).ravel():
            h = (h * 1000003 + int(round(float(v) * 1e9))) % (2 ** 61 - 1)
    return int(h)


class TestSeededHomography:
    def test_seed_reproduces_internal_phase_correlation(self):
        frames = _synth_burst(n=3)
        ref, mov = frames[0].green, frames[2].green
        W_default = estimate_homography(ref, mov, ecc_scale=0.5)
        dx, dy = estimate_translation(ref, mov)
        W_seeded = estimate_homography(ref, mov, ecc_scale=0.5, seed=(dx, dy))
        assert W_default.tobytes() == W_seeded.tobytes()

    def test_ref_mask_is_reusable_across_frames(self):
        frames = _synth_burst(n=4)
        ref = frames[0].green
        mask = core.gradient_mask(ref)
        for mov in (frames[1].green, frames[3].green):
            W_a = estimate_homography(ref, mov, ecc_scale=0.5)
            W_b = estimate_homography(ref, mov, ecc_scale=0.5, ref_mask=mask)
            assert W_a.tobytes() == W_b.tobytes()

    def test_edge_weighted_off_ignores_mask(self):
        frames = _synth_burst(n=3)
        ref, mov = frames[0].green, frames[1].green
        dx, dy = estimate_translation(ref, mov)
        W_a = estimate_homography(ref, mov, ecc_scale=0.5, edge_weighted=False)
        W_b = estimate_homography(ref, mov, ecc_scale=0.5, edge_weighted=False,
                                  seed=(dx, dy))
        assert W_a.tobytes() == W_b.tobytes()


class TestSingleRender:
    def test_score_and_residual_match_standalone_helpers(self):
        frames = _synth_burst(n=3)
        ref, mov = frames[0].green, frames[2].green
        W = estimate_homography(ref, mov, ecc_scale=0.5)
        rendered = _render_in_ref_frame(ref, mov, W)
        assert _alignment_score(ref, mov, W) == \
            _alignment_score(ref, mov, W, rendered=rendered)
        r_plain = _alignment_residual_px(ref, mov, W)
        r_shared = _alignment_residual_px(ref, mov, W, rendered=rendered)
        assert r_plain == r_shared

    def test_singular_transform_degrades_the_same_way(self):
        frames = _synth_burst(n=2)
        ref, mov = frames[0].green, frames[1].green
        # Singular but normalised (W[2,2] == 1): rank-1 linear part, so the
        # inverse does not exist. (All-zero would be renormalised to identity.)
        bad = np.array([[1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]])
        assert _render_in_ref_frame(ref, mov, bad) == (None, None)
        assert _alignment_score(ref, mov, bad) == \
            _alignment_score(ref, mov, bad, rendered=(None, None))
        r1 = _alignment_residual_px(ref, mov, bad)
        r2 = _alignment_residual_px(ref, mov, bad, rendered=(None, None))
        assert np.isnan(r1) and np.isnan(r2)

    def test_register_frame_matches_manual_sequence(self):
        frames = _synth_burst(n=3)
        anchor, ref, mov = (frames[0].green, frames[1].green, frames[2].green)
        dx, dy, W, score, residual = _register_frame(
            anchor, ref, mov, False, "homography", 0.5, True)
        # The same measurements taken the long way round.
        adx, ady = estimate_translation(anchor, mov)
        rdx, rdy = estimate_translation(ref, mov)
        W_ref = estimate_homography(ref, mov, ecc_scale=0.5,
                                    seed=(rdx, rdy),
                                    ref_mask=core.gradient_mask(ref))
        assert (dx, dy) == (adx, ady)
        assert W.tobytes() == W_ref.tobytes()
        assert score == _alignment_score(ref, mov, W_ref)
        assert residual == _alignment_residual_px(ref, mov, W_ref)


class TestSharedFrameBlock:
    def test_dedupes_and_round_trips(self):
        rng = np.random.default_rng(3)
        a = rng.random((6, 7), dtype=np.float32)
        b = rng.random((6, 7), dtype=np.float32)
        store = _SharedRegFrames([a, b, a])
        try:
            assert len(store.specs) == 2          # `a` stored once
            assert store.slot(a) == 0
            assert store.slot(b) == 1
            view = np.ndarray((6, 7), dtype=np.float32,
                              buffer=store._shm.buf, offset=0)
            assert np.array_equal(view, a)
            off, h, w = store.specs[1]
            view_b = np.ndarray((h, w), dtype=np.float32,
                                buffer=store._shm.buf, offset=off)
            assert np.array_equal(view_b, b)
        finally:
            store.close()

    @pytest.mark.skipif(not _HAS_CPUS, reason="pool needs >= 4 CPUs")
    def test_pool_payload_carries_no_frames(self):
        frames = _synth_burst(n=4, size=96)
        res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                              registration_workers=2,
                              use_opencl=False).run(frames)
        assert res.used_frames >= 2
        # Payload size is what the change is about: 5 short ints instead of
        # 3 full frames. Check the contract directly.
        store = _SharedRegFrames([frames[0].green, frames[1].green])
        try:
            import pickle
            payload = (0, store.slot(frames[0].green),
                       store.slot(frames[1].green), store.slot(frames[1].green),
                       False, "homography", 0.5, True, False)
            assert len(pickle.dumps(payload)) < 1024
        finally:
            store.close()


class TestPoolEqualsSerial:
    @pytest.mark.skipif(not _HAS_CPUS, reason="pool needs >= 4 CPUs")
    def test_transforms_bit_identical(self):
        frames = _synth_burst()
        kw = dict(ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.5,
                  use_opencl=False)
        ser = SuperResolution(**kw, registration_workers=1).run(frames)
        par = SuperResolution(**kw, registration_workers=3).run(frames)
        assert [t.tobytes() for t in ser.transforms] == \
               [t.tobytes() for t in par.transforms]
        assert list(ser.shifts) == list(par.shifts)
        assert np.array_equal(np.asarray(ser.align_scores),
                              np.asarray(par.align_scores))
        assert np.array_equal(np.asarray(ser.align_residuals),
                              np.asarray(par.align_residuals))
        assert np.abs(ser.hr.astype(np.float64)
                      - par.hr.astype(np.float64)).max() == 0.0


@needs_real
class TestRealBurstRegistrationUnchanged:
    """The load-bearing check: real-burst shifts/homographies do not move."""

    @pytest.fixture(scope="class")
    @classmethod
    def frames(cls):
        from raw_sr.core.superres import load_raw_linear
        roi = (1650, 1240, 512, 512)
        return [load_raw_linear(f, roi=roi) for f in _REAL_FILES]

    def test_shifts_and_homographies_match_baseline(self, frames):
        sr = SuperResolution(ratio=2.0, registration_workers=1,
                             use_opencl=False)
        res = sr.run(frames)
        assert res.reference_index == _B2_ROI512_REF_INDEX
        got = {int(i): (round(float(a), 6), round(float(b), 6))
               for i, (a, b) in zip(range(len(frames)), res.shifts)}
        assert got == _B2_ROI512_SHIFTS
        assert _hash_transforms(res.transforms) == _B2_ROI512_TRANSFORM_HASH

    def test_register_frame_is_self_consistent_on_real_data(self, frames):
        # Direct check with the real reference frame and real masks.
        ref_idx = _B2_ROI512_REF_INDEX
        anchor, ref = frames[0].green, frames[ref_idx].green
        mov = frames[4].green
        mask = core.gradient_mask(ref)
        for reg in ("homography", "translation"):
            a = _register_frame(anchor, ref, mov, False, reg, 0.25, True,
                                ref_mask=mask)
            b = _register_frame(anchor, ref, mov, False, reg, 0.25, True,
                                ref_mask=mask)
            assert a[2].tobytes() == b[2].tobytes()
            assert a[3] == b[3] and a[4] == b[4]


@needs_real
class TestRoi1024HomographyShortcut:
    """The ROI>=1024 shortcut must stay inside the 0.01 px / 1e-3 budget.

    The branch warms the homography up from the refined affine fit instead of
    running a half-scale homography pass and an independent seed-based solve.
    These tests compare against the pre-change transforms recorded in
    ``_B4_ROI1024_*``; they are the guard that makes the speedup safe.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def frames(cls):
        from raw_sr.core.superres import load_raw_linear
        roi = (1650, 1240, 1024, 1024)
        return [load_raw_linear(f, roi=roi) for f in _REAL_FILES]

    def _check(self, frames, ecc_scale, reference):
        ref_idx = _B4_ROI1024_REF_INDEX
        ref = frames[ref_idx].green
        worst_corner = worst_score = worst_resid = 0.0
        for i, expected in sorted(reference.items()):
            _, _, W, score, resid = _register_frame(
                ref, ref, frames[i].green, False, "homography", ecc_scale,
                False)
            corners = _corners(W, 1024)
            exp = np.asarray(expected[:4], dtype=np.float64)
            worst_corner = max(worst_corner, float(np.max(np.linalg.norm(
                corners - exp, axis=1))))
            worst_score = max(worst_score, abs(score - expected[4]))
            worst_resid = max(worst_resid, abs(resid - expected[5]))
        assert worst_corner <= 0.01, f"corner moved {worst_corner:.6f} px"
        assert worst_score <= 1e-3, f"score moved {worst_score:.2e}"
        assert worst_resid <= 1e-3, f"residual moved {worst_resid:.2e}"

    def test_ecc_scale_shipped(self, frames):
        self._check(frames, 0.5, _B4_ROI1024_ECC050)

    def test_ecc_scale_acceptance(self, frames):
        self._check(frames, 0.25, _B4_ROI1024_ECC025)
