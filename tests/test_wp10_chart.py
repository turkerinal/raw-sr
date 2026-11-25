"""WP10 tests: deterministic chart generation, fiducial detection, and simulated accuracy."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from scripts.testchart.make_chart import generate_charts, render_chart_raster
from scripts.testchart.measure_chart import detect_fiducials_and_homography
from scripts.testchart.validate_sim import run_simulated_validation


def test_chart_generation_deterministic(tmp_path):
    """Rendering the chart twice produces bit-identical PNG and PDF outputs."""
    img1 = render_chart_raster(paper="A4", dpi=100)
    img2 = render_chart_raster(paper="A4", dpi=100)
    assert img1.shape == img2.shape
    assert np.array_equal(img1, img2)

    out1 = generate_charts(str(tmp_path / "run1"), sizes=("A4",), dpi=100)
    out2 = generate_charts(str(tmp_path / "run2"), sizes=("A4",), dpi=100)

    for key in ("A4_png", "A4_pdf"):
        assert os.path.isfile(out1[key])
        h1 = hashlib.sha256(open(out1[key], "rb").read()).hexdigest()
        h2 = hashlib.sha256(open(out2[key], "rb").read()).hexdigest()
        assert h1 == h2, f"{key} is not deterministic"

    # Verify all 4 ArUco fiducials are detected on the rendered chart
    H, px_h, detected = detect_fiducials_and_homography(img1)
    assert H.shape == (3, 3)
    assert set(detected.keys()) == {0, 1, 2, 3}
    assert abs(px_h - img1.shape[0]) / img1.shape[0] < 0.02


def test_simulated_camera_validation_within_10_pct(tmp_path):
    """End-to-end simulated camera test finds fiducials and measures resolution within 10%."""
    rep = run_simulated_validation(
        ratio=2,
        lr_h=540,
        n_frames=12,
        sigma_lens_lr=0.45,
        seed=42,
        out_json=str(tmp_path / "sim.json"),
        out_panel=str(tmp_path / "sim.png"),
    )
    assert os.path.isfile(tmp_path / "sim.json")
    assert os.path.isfile(tmp_path / "sim.png")

    sim = rep["simulation"]
    assert sim["sr_star_f50_rel_err_pct"] < 10.0, f"Star f50 error too high: {sim}"
    assert sim["sr_edge_f50_rel_err_pct"] < 10.0, f"Edge MTF50 error too high: {sim}"

    # Zoom GT: SR must beat single-frame bicubic on overall PSNR and SSIM
    z_single = rep["zoom_gt"]["single_vs_gt"]["overall"]
    z_sr = rep["zoom_gt"]["sr_vs_gt"]["overall"]
    assert z_sr["psnr_db"] > z_single["psnr_db"], f"SR did not beat single on PSNR: {z_single} vs {z_sr}"
    assert z_sr["ssim"] > z_single["ssim"], f"SR did not beat single on SSIM: {z_single} vs {z_sr}"
