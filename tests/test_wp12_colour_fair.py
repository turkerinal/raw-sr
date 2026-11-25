"""Unit tests for WP12: colour-correct CFA fusions and colour-fair CFA held-out metric."""

from __future__ import annotations

import numpy as np
import pytest

from raw_sr.core.cfa_drizzle import channel_map, pattern_array
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.map_sr import estimate_colour_matrix
from raw_sr.core.superres import RawFrame, SuperResolution

RGGB = np.array([[0, 1], [3, 2]], dtype=np.int8)

# Realistic camera-to-sRGB inverse (output sRGB -> WB-scaled camera RGB)
C_REALISTIC = np.array(
    [
        [0.72, 0.22, 0.06],
        [0.08, 0.86, 0.06],
        [0.04, 0.16, 0.80],
    ],
    dtype=np.float64,
)


def _make_colour_burst(
    h: int = 64,
    w: int = 64,
    n: int = 4,
    C: np.ndarray = C_REALISTIC,
    gr_gb_imbalance: float = 1.0,
):
    """Create a smooth multi-colour synthetic burst where rgb is sRGB and mosaic is C @ rgb."""
    yy, xx = np.meshgrid(
        np.linspace(0.15, 0.80, h, dtype=np.float64),
        np.linspace(0.15, 0.80, w, dtype=np.float64),
        indexing="ij",
    )
    rgb = np.stack(
        [
            0.20 + 0.55 * xx,
            0.25 + 0.45 * yy,
            0.30 + 0.35 * (1.0 - xx) * yy,
        ],
        axis=-1,
    ).astype(np.float32)

    cam = np.matmul(rgb.astype(np.float64), C.T)
    ch = channel_map(RGGB, (h, w))
    mos = np.empty((h, w), dtype=np.float32)
    for c in range(3):
        mos[ch == c] = cam[:, :, c][ch == c].astype(np.float32)

    if abs(gr_gb_imbalance - 1.0) > 1e-6:
        mos[0::2, 1::2] *= np.float32(np.sqrt(gr_gb_imbalance))
        mos[1::2, 0::2] /= np.float32(np.sqrt(gr_gb_imbalance))

    frames = [
        RawFrame(
            name=f"f{k}",
            rgb=rgb.copy(),
            green=rgb[:, :, 1].copy(),
            mosaic=mos.copy(),
            pattern=RGGB,
        )
        for k in range(n)
    ]
    return frames, rgb, mos


@pytest.mark.parametrize(
    "fusion,extra",
    [
        ("demosaic_drizzle", {}),
        ("cfa_drizzle", {}),
        ("cfa_kr", {"cfa_adaptive": False}),
        ("cfa_kr", {"cfa_adaptive": True}),
        ("map", {"use_gpu": False}),
    ],
)
def test_constant_colour_roundtrip_all_fusions(fusion, extra):
    """Every fusion must return output in the same sRGB colour space as default."""
    frames, rgb, _ = _make_colour_burst(h=64, w=64, n=4, C=C_REALISTIC)
    sr = SuperResolution(
        ratio=2.0,
        blur_threshold_ratio=0.0,
        registration="translation",
        fusion=fusion,
        **extra,
    )
    res = sr.run(frames, reference_index=0)
    # Interior crop avoids border replication differences
    hr_mid = res.hr[24:-24, 24:-24]
    ref_mid = rgb[12:-12, 12:-12]
    # Compare channel means: must match sRGB within < 0.015 (not camera space which differs by ~0.05-0.10)
    mean_hr = hr_mid.mean(axis=(0, 1))
    mean_ref = ref_mid.mean(axis=(0, 1))
    max_err = float(np.max(np.abs(mean_hr - mean_ref)))
    assert max_err < 0.015, (
        f"fusion={fusion} {extra} colour mismatch: hr={mean_hr}, ref={mean_ref}, max_err={max_err:.4f}"
    )


@pytest.mark.parametrize("fusion", ["cfa_drizzle", "cfa_kr"])
def test_cfa_fusions_equalize_gr_gb_imbalance(fusion):
    """cfa_drizzle and cfa_kr must equalize Gr/Gb imbalance on input mosaics."""
    h, w = 64, 64
    rgb = np.full((h, w, 3), [0.40, 0.50, 0.35], dtype=np.float32)
    mos = np.full((h, w), 0.50, dtype=np.float32)
    mos[0::2, 0::2] = 0.40
    mos[1::2, 1::2] = 0.35
    mos[0::2, 1::2] = 0.50 * np.float32(np.sqrt(1.02))
    mos[1::2, 0::2] = 0.50 / np.float32(np.sqrt(1.02))
    frames = [
        RawFrame(name=f"f{k}", rgb=rgb.copy(), green=rgb[:, :, 1].copy(), mosaic=mos.copy(), pattern=RGGB)
        for k in range(4)
    ]
    sr = SuperResolution(
        ratio=2.0,
        blur_threshold_ratio=0.0,
        registration="translation",
        fusion=fusion,
        cfa_adaptive=True,
    )
    res = sr.run(frames, reference_index=0)
    ratio = res.metrics.get("cfa_kr_gr_gb_ratio")
    assert ratio is not None and abs(ratio - 1.02) < 0.002, f"Expected gr_gb_ratio ~1.02, got {ratio}"
    # Interior green must be flat (no period-2 checkerboard)
    g_std = float(np.std(res.hr[24:-24, 24:-24, 1]))
    assert g_std < 0.002, f"Expected flat green after Gr/Gb equalization, got std={g_std:.5f}"


def test_cfa_heldout_colour_fair_vs_raw():
    """Colour-fair CFA held-out must remove the camera-vs-sRGB penalty for sRGB HR outputs."""
    frames, rgb, mos = _make_colour_burst(h=64, w=64, n=5, C=C_REALISTIC)
    C_est, info = estimate_colour_matrix(rgb, mos, RGGB)
    assert not info["fallback"]
    assert np.allclose(C_est, C_REALISTIC, atol=1e-3)

    # Perfect sRGB HR image (2x upsampled smooth ramp)
    import cv2

    hr_srgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_CUBIC)
    mosaics_fused = [f.mosaic for f in frames[:4]]
    transforms = [np.eye(3, dtype=np.float64) for _ in range(4)]
    weights = [1.0] * 4
    ref_to_held = np.eye(3, dtype=np.float64)

    res_raw = evaluate_cfa_heldout(
        {"srgb": hr_srgb},
        frames[4].mosaic,
        mosaics_fused,
        transforms,
        weights,
        ref_to_held,
        ratio=2.0,
        pattern=RGGB,
        margin_lr=8,
        colour_matrix=None,
    )
    res_fair = evaluate_cfa_heldout(
        {"srgb": hr_srgb},
        frames[4].mosaic,
        mosaics_fused,
        transforms,
        weights,
        ref_to_held,
        ratio=2.0,
        pattern=RGGB,
        margin_lr=8,
        colour_matrix=C_est,
    )

    psnr_raw = res_raw["per_method"]["srgb"]["heldout_cfa_psnr_db"]
    psnr_fair = res_fair["per_method"]["srgb"]["heldout_cfa_psnr_db"]
    assert res_raw["config"]["colour_fair"] is False
    assert res_fair["config"]["colour_fair"] is True
    # Fair metric must be dramatically higher than raw (which suffers ~25-35 dB colour error)
    assert psnr_fair > psnr_raw + 15.0, f"Expected fair ({psnr_fair:.2f}) >> raw ({psnr_raw:.2f})"
    assert psnr_fair > 50.0, f"Expected near-exact reconstruction in fair metric, got {psnr_fair:.2f} dB"
