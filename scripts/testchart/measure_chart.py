#!/usr/bin/env python3
"""Measure resolution and ground-truth accuracy on the WP10 test chart.

Supports two protocols in one script:
  A) "Zoom GT" (--burst DIR --gt DIR):
     Registers SR output, single-frame bicubic, and d/2 (or d/3) GT frames to
     canonical chart coordinates via the 4 corner ArUco fiducials (homography +
     sub-pixel refinement), photometrically matches each estimate to GT via
     per-channel affine fit on the 8-step grey wedge, and reports PSNR/SSIM per
     chart region.
  B) "Chart-only" (--burst DIR, no --gt):
     Measures directly from known chart geometry:
       - 144-spoke Siemens stars (sinusoidal & binary): resolved radius ->
         limiting resolution in cycles/LR px (50% and 10% contrast thresholds,
         global and per-angle), plus phase-reversal aliasing detection.
       - ISO 12233-style 5-deg slanted edges (high & low contrast): sfrmat-style
         ESF -> LSF -> MTF50 / MTF10, flagging above-Nyquist values as aliasing.
       - Hyperbolic & linear line-pair wedges: limiting frequency (cycles/LR px).
       - 8-step grey wedge: flat-patch noise sigma and mid-grey SNR.

Outputs a JSON report and a PNG summary panel (single | SR [| GT] crops).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.testchart import chart_spec as spec
else:
    from . import chart_spec as spec


# --------------------------------------------------------------------------- #
# 1. Fiducial detection, homography, and canonical chart warping
# --------------------------------------------------------------------------- #
def _to_u8_gray(img: np.ndarray) -> np.ndarray:
    """Convert float [0, 1] or uint8/uint16 RGB/gray image to uint8 grayscale."""
    if img.ndim == 3:
        # Green channel or luminance
        g = img[..., 1] if img.shape[2] >= 3 else img[..., 0]
    else:
        g = img
    if g.dtype == np.uint8:
        return g
    if g.dtype == np.uint16:
        return (g >> 8).astype(np.uint8)
    g_f = g.astype(np.float32)
    lo, hi = float(np.percentile(g_f, 1.0)), float(np.percentile(g_f, 99.5))
    if hi - lo < 1e-6:
        return np.clip(g_f * 255.0, 0, 255).astype(np.uint8)
    norm = np.clip((g_f - lo) / (hi - lo), 0.0, 1.0)
    # Mild gamma for linear RAW images so dark ArUco borders are crisp
    return np.clip(np.round(np.power(norm, 1.0 / 2.2) * 255.0), 0, 255).astype(np.uint8)


def _canonical_marker_corners(canon_w: int = spec.CANONICAL_W, canon_h: int = spec.CANONICAL_H) -> dict[int, np.ndarray]:
    """Return expected (4, 2) corner coordinates (TL, TR, BR, BL) per marker ID."""
    aspect = canon_w / float(canon_h)
    rects = spec.fiducial_rects_norm(aspect)
    out: dict[int, np.ndarray] = {}
    for fid, r in rects.items():
        x0, y0, x1, y1 = r.to_px(canon_w, canon_h)
        side = min(x1 - x0, y1 - y0)
        out[fid] = np.array(
            [
                [x0, y0],
                [x0 + side - 1, y0],
                [x0 + side - 1, y0 + side - 1],
                [x0, y0 + side - 1],
            ],
            dtype=np.float64,
        )
    return out


def detect_fiducials_and_homography(
    img: np.ndarray,
    canon_w: int = spec.CANONICAL_W,
    canon_h: int = spec.CANONICAL_H,
) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """Detect the 4 corner ArUco markers and compute img -> canonical homography.

    Returns:
        H: 3x3 float64 homography mapping input image pixels -> canonical chart pixels.
        img_px_per_chart_h: effective height of the chart in input image pixels.
        detected_corners: dict of marker_id -> (4, 2) refined corners in input image.
    """
    gray = _to_u8_gray(img)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.ARUCO_DICT_NAME))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners_list, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        found = [] if ids is None else [int(i) for i in ids.ravel()]
        raise RuntimeError(f"Expected 4 ArUco fiducials (IDs 0..3), found: {found}")

    canon_corners = _canonical_marker_corners(canon_w, canon_h)
    src_pts, dst_pts = [], []
    detected: dict[int, np.ndarray] = {}
    for c, mid_val in zip(corners_list, ids.ravel()):
        mid = int(mid_val)
        if mid in canon_corners:
            pts = c.reshape(4, 2).astype(np.float64)
            detected[mid] = pts
            src_pts.append(pts)
            dst_pts.append(canon_corners[mid])

    if len(detected) < 4:
        raise RuntimeError(f"Missing required fiducial IDs 0..3; detected: {sorted(detected.keys())}")

    src = np.vstack(src_pts)
    dst = np.vstack(dst_pts)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed")

    # Estimate effective chart height in input image pixels via inverse homography
    H_inv = np.linalg.inv(H)
    p_top = H_inv @ np.array([canon_w * 0.5, 0.0, 1.0])
    p_bot = H_inv @ np.array([canon_w * 0.5, float(canon_h), 1.0])
    p_top /= p_top[2]
    p_bot /= p_bot[2]
    img_px_per_chart_h = float(np.linalg.norm(p_bot[:2] - p_top[:2]))
    return H, img_px_per_chart_h, detected


def warp_to_canonical(
    img: np.ndarray,
    H: np.ndarray,
    canon_w: int = spec.CANONICAL_W,
    canon_h: int = spec.CANONICAL_H,
) -> np.ndarray:
    """Warp an RGB or greyscale float32 image into canonical chart coordinates."""
    f32 = img.astype(np.float32)
    return cv2.warpPerspective(
        f32, H, (canon_w, canon_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def refine_canonical_ecc(mov: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Sub-pixel homography refinement of already-warped ``mov`` to ``ref``."""
    g_mov = mov[..., 1].astype(np.float32) if mov.ndim == 3 else mov.astype(np.float32)
    g_ref = ref[..., 1].astype(np.float32) if ref.ndim == 3 else ref.astype(np.float32)
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5)
    try:
        cv2.findTransformECC(g_ref, g_mov, warp, cv2.MOTION_HOMOGRAPHY, crit)
        return cv2.warpPerspective(
            mov.astype(np.float32),
            warp,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except cv2.error:
        return mov.astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. Photometric matching on the 8-step grey wedge
# --------------------------------------------------------------------------- #
def _grey_wedge_patch_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (means, stds) of shape (8, C) across the 8 grey wedge steps."""
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    arr = img[..., None] if img.ndim == 2 else img
    gx0, gy0, gx1, gy1 = spec.ROIS_NORM["grey_wedge"].to_px(w, h)
    n_steps = len(spec.GREY_STEPS)
    step_w = (gx1 - gx0) / n_steps
    pad_y = max(2, int(round(0.22 * (gy1 - gy0))))
    pad_x = max(2, int(round(0.22 * step_w)))

    means = np.zeros((n_steps, c), dtype=np.float64)
    stds = np.zeros((n_steps, c), dtype=np.float64)
    for i in range(n_steps):
        x0 = int(round(gx0 + i * step_w)) + pad_x
        x1 = int(round(gx0 + (i + 1) * step_w)) - pad_x
        patch = arr[gy0 + pad_y : gy1 - pad_y, x0:x1, :].astype(np.float64)
        means[i] = patch.mean(axis=(0, 1))
        stds[i] = patch.std(axis=(0, 1))
    return means, stds


def photometric_match_grey_wedge(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-channel affine match of ``est`` to ``gt`` using the 8-step grey wedge."""
    m_est, _ = _grey_wedge_patch_stats(est)
    m_gt, _ = _grey_wedge_patch_stats(gt)
    out = est.astype(np.float32).copy()
    if out.ndim == 2:
        A = np.column_stack([m_est[:, 0], np.ones(len(m_est))])
        a, b = np.linalg.lstsq(A, m_gt[:, 0], rcond=None)[0]
        return (a * out + b).astype(np.float32)
    for ch in range(out.shape[2]):
        A = np.column_stack([m_est[:, ch], np.ones(len(m_est))])
        a, b = np.linalg.lstsq(A, m_gt[:, ch], rcond=None)[0]
        out[..., ch] = (a * out[..., ch] + b).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# 3. Protocol A: Zoom GT metrics (PSNR and SSIM per region)
# --------------------------------------------------------------------------- #
def compute_psnr(est: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((est.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range * data_range) / mse))


def compute_ssim(est: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Gaussian-windowed SSIM (sigma=1.5, 11x11 kernel) averaged over channels."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    x = est.astype(np.float64)
    y = gt.astype(np.float64)
    if x.ndim == 2:
        x, y = x[..., None], y[..., None]
    vals = []
    for ch in range(x.shape[2]):
        xc, yc = x[..., ch], y[..., ch]
        mu_x = cv2.GaussianBlur(xc, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
        mu_y = cv2.GaussianBlur(yc, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
        sig_x2 = cv2.GaussianBlur(xc * xc, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_x * mu_x
        sig_y2 = cv2.GaussianBlur(yc * yc, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_y * mu_y
        sig_xy = cv2.GaussianBlur(xc * yc, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_x * mu_y
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * sig_xy + c2)
        den = (mu_x * mu_x + mu_y * mu_y + c1) * (sig_x2 + sig_y2 + c2)
        vals.append(float(np.mean(num / np.maximum(den, 1e-12))))
    return float(np.mean(vals))


def evaluate_zoom_gt(est_canon: np.ndarray, gt_canon: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute per-region PSNR and SSIM of photometrically matched ``est`` vs ``gt``."""
    matched = photometric_match_grey_wedge(est_canon, gt_canon)
    h, w = gt_canon.shape[:2]
    data_range = max(0.1, float(np.percentile(gt_canon, 99.5) - np.percentile(gt_canon, 0.5)))
    regions: dict[str, dict[str, float]] = {}
    for name in ("star_sin", "star_bin", "edge_high", "edge_low", "wedge_hyp", "wedge_lin", "noise_patch", "grey_wedge"):
        x0, y0, x1, y1 = spec.ROIS_NORM[name].to_px(w, h)
        # Shrink slightly to exclude ROI border lines
        m = 4
        p_est = matched[y0 + m : y1 - m, x0 + m : x1 - m]
        p_gt = gt_canon[y0 + m : y1 - m, x0 + m : x1 - m]
        regions[name] = {
            "psnr_db": round(compute_psnr(p_est, p_gt, data_range), 3),
            "ssim": round(compute_ssim(p_est, p_gt, data_range), 4),
        }
    # Overall interior (excluding outer fiducial margin)
    mx, my = int(0.08 * w), int(0.14 * h)
    regions["overall"] = {
        "psnr_db": round(compute_psnr(matched[my:-my, mx:-mx], gt_canon[my:-my, mx:-mx], data_range), 3),
        "ssim": round(compute_ssim(matched[my:-my, mx:-mx], gt_canon[my:-my, mx:-mx], data_range), 4),
    }
    return regions


# --------------------------------------------------------------------------- #
# 4. Protocol B: Chart-only metrics (Siemens star, slanted edge, wedges, noise)
# --------------------------------------------------------------------------- #
def measure_siemens_star(
    canon_img: np.ndarray,
    center_norm: tuple[float, float],
    lr_px_per_chart_h: float,
    n_spokes: int = spec.STAR_N_SPOKES,
    n_sectors: int = 8,
) -> dict[str, Any]:
    """Measure Siemens star limiting resolution (50% & 10% contrast) and aliasing reversal.

    Uses exact harmonic-N Fourier projection along concentric rings so both
    modulation amplitude and phase (for aliasing reversal) are measured cleanly.
    """
    gray = canon_img[..., 1].astype(np.float64) if canon_img.ndim == 3 else canon_img.astype(np.float64)
    h, w = gray.shape
    cx = center_norm[0] * w
    cy = center_norm[1] * h
    r_outer = spec.STAR_RADIUS_H * h * 0.96
    r_inner = spec.STAR_INNER_RADIUS_H * h * 1.25

    n_radii = 90
    radii = np.linspace(r_outer, r_inner, n_radii)  # outer (low freq) -> inner (high freq)
    n_theta = max(2048, n_spokes * 16)
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_n = np.cos(n_spokes * thetas)
    sin_n = np.sin(n_spokes * thetas)

    # Sample rings via bilinear remap
    mx = (cx + radii[:, None] * np.cos(thetas[None, :])).astype(np.float32)
    my = (cy + radii[:, None] * np.sin(thetas[None, :])).astype(np.float32)
    rings = cv2.remap(gray.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    rings = rings - rings.mean(axis=1, keepdims=True)

    # Global harmonic projection per radius: complex amplitude c(r) = a + j*b
    a_r = (2.0 / n_theta) * (rings @ cos_n)
    b_r = (2.0 / n_theta) * (rings @ sin_n)
    amp_r = np.hypot(a_r, b_r)
    phase_r = np.arctan2(b_r, a_r)

    # Reference low-frequency amplitude & phase from outermost 5 rings
    ref_amp = float(np.median(amp_r[:5]))
    ref_a = float(np.median(a_r[:5]))
    ref_b = float(np.median(b_r[:5]))
    ref_phase = math.atan2(ref_b, ref_a)

    # Frequencies in cycles/LR px: f = n_spokes / (2 * pi * r_norm_h * lr_px_per_chart_h)
    r_norm_h = radii / float(h)
    freqs_lr = n_spokes / (2.0 * np.pi * r_norm_h * lr_px_per_chart_h)

    norm_amp = amp_r / max(ref_amp, 1e-8)
    # Signed contrast projected onto reference phase (negative = phase reversal / aliasing!)
    signed_contrast = norm_amp * np.cos(phase_r - ref_phase)

    def _find_threshold(curve: np.ndarray, thresh: float) -> float:
        below = np.where(curve < thresh)[0]
        if len(below) == 0:
            return float(freqs_lr[-1])
        idx = int(below[0])
        if idx == 0:
            return float(freqs_lr[0])
        c0, c1 = curve[idx - 1], curve[idx]
        f0, f1 = freqs_lr[idx - 1], freqs_lr[idx]
        t = (thresh - c0) / (c1 - c0) if abs(c1 - c0) > 1e-9 else 0.0
        return float(f0 + np.clip(t, 0.0, 1.0) * (f1 - f0))

    f50 = _find_threshold(signed_contrast, 0.50)
    f10 = _find_threshold(signed_contrast, 0.10)
    # Detect aliasing reversal: signed contrast goes significantly negative after dropping below 0.1
    aliasing_reversal = bool(np.any(signed_contrast < -0.08))

    # Per-sector measurements
    sec_step = n_theta // n_sectors
    sector_f50: list[float] = []
    sector_f10: list[float] = []
    for s in range(n_sectors):
        i0, i1 = s * sec_step, (s + 1) * sec_step
        sa = (2.0 / (i1 - i0)) * (rings[:, i0:i1] @ cos_n[i0:i1])
        sb = (2.0 / (i1 - i0)) * (rings[:, i0:i1] @ sin_n[i0:i1])
        s_ref = max(float(np.median(np.hypot(sa[:5], sb[:5]))), 1e-8)
        s_ref_ph = math.atan2(float(np.median(sb[:5])), float(np.median(sa[:5])))
        s_signed = (np.hypot(sa, sb) / s_ref) * np.cos(np.arctan2(sb, sa) - s_ref_ph)
        sector_f50.append(round(_find_threshold(s_signed, 0.50), 4))
        sector_f10.append(round(_find_threshold(s_signed, 0.10), 4))

    return {
        "f50_cyc_lr_px": round(f50, 4),
        "f10_cyc_lr_px": round(f10, 4),
        "aliasing_reversal": aliasing_reversal,
        "sector_f50": sector_f50,
        "sector_f10": sector_f10,
    }


def measure_slanted_edge_mtf(
    canon_img: np.ndarray,
    roi_key: str,
    lr_px_per_chart_h: float,
    is_single_frame: bool = False,
) -> dict[str, Any]:
    """sfrmat-style slanted-edge ESF -> LSF -> MTF50 / MTF10 in cycles/LR px."""
    gray = canon_img[..., 1].astype(np.float64) if canon_img.ndim == 3 else canon_img.astype(np.float64)
    h, w = gray.shape
    x0, y0, x1, y1 = spec.ROIS_NORM[roi_key].to_px(w, h)
    # Crop interior away from black frame
    pad_x = max(4, int(0.10 * (x1 - x0)))
    pad_y = max(4, int(0.08 * (y1 - y0)))
    patch = gray[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)

    gx = np.diff(patch, axis=1)
    w_g = np.abs(gx) ** 2
    row_sum = w_g.sum(axis=1)
    valid = row_sum > np.percentile(row_sum, 20)
    if valid.sum() < 10:
        return {"mtf50_cyc_lr_px": 0.0, "mtf10_cyc_lr_px": 0.0, "aliasing_flagged": False}

    xs_g = np.arange(pw - 1, dtype=np.float64) + 0.5
    centers = (w_g[valid] @ xs_g) / row_sum[valid]
    ys_v = np.arange(ph, dtype=np.float64)[valid]
    a, b = np.polyfit(ys_v, centers, 1)
    norm = math.sqrt(1.0 + a * a)
    dist = ((xx - (a * yy + b)) / norm).ravel()
    vals = patch.ravel()

    oversamp = 4
    bin_w = 1.0 / oversamp
    d_max = min(24.0, pw * 0.4)
    bins = np.arange(-d_max, d_max + bin_w, bin_w)
    idx = np.digitize(dist, bins) - 1
    n_bins = len(bins) - 1
    esf = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.float64)
    ok = (idx >= 0) & (idx < n_bins)
    np.add.at(esf, idx[ok], vals[ok])
    np.add.at(cnt, idx[ok], 1.0)
    good = cnt > 0
    if good.sum() < 16:
        return {"mtf50_cyc_lr_px": 0.0, "mtf10_cyc_lr_px": 0.0, "aliasing_flagged": False}
    esf[good] /= cnt[good]
    esf = np.interp(np.arange(n_bins), np.where(good)[0], esf[good])

    lsf = np.diff(esf)
    pk = int(np.argmax(np.abs(lsf)))
    win_half = min(pk, len(lsf) - 1 - pk, int(12 * oversamp))
    if win_half < 8:
        return {"mtf50_cyc_lr_px": 0.0, "mtf10_cyc_lr_px": 0.0, "aliasing_flagged": False}
    lsf_w = lsf[pk - win_half : pk + win_half + 1] * np.hamming(2 * win_half + 1)

    n_fft = max(1024, 1 << int(math.ceil(math.log2(len(lsf_w) * 4))))
    mtf = np.abs(np.fft.rfft(lsf_w, n=n_fft))
    if mtf[0] <= 1e-12:
        return {"mtf50_cyc_lr_px": 0.0, "mtf10_cyc_lr_px": 0.0, "aliasing_flagged": False}
    mtf /= mtf[0]

    # Frequency in cycles per canonical pixel -> cycles per LR pixel
    # 1 canonical pixel = (lr_px_per_chart_h / h) LR pixels
    freqs_canon = np.fft.rfftfreq(n_fft, d=bin_w)
    freqs_lr = freqs_canon * (h / lr_px_per_chart_h)

    def _crossing(thresh: float) -> float:
        below = np.where(mtf < thresh)[0]
        if len(below) == 0:
            return float(freqs_lr[-1])
        i = int(below[0])
        if i == 0:
            return float(freqs_lr[0])
        t = (thresh - mtf[i - 1]) / (mtf[i] - mtf[i - 1]) if abs(mtf[i] - mtf[i - 1]) > 1e-9 else 0.0
        return float(freqs_lr[i - 1] + np.clip(t, 0.0, 1.0) * (freqs_lr[i] - freqs_lr[i - 1]))

    mtf50 = _crossing(0.50)
    mtf10 = _crossing(0.10)
    # Single-frame MTF50 > 0.5 cyc/LR px is physically impossible without aliasing
    aliasing_flagged = bool(is_single_frame and mtf50 > 0.50)
    return {
        "mtf50_cyc_lr_px": round(mtf50, 4),
        "mtf10_cyc_lr_px": round(mtf10, 4),
        "aliasing_flagged": aliasing_flagged,
    }


def measure_wedge(
    canon_img: np.ndarray,
    roi_key: str,
    lr_px_per_chart_h: float,
) -> dict[str, float]:
    """Measure limiting frequency along the hyperbolic (horizontal) or linear (vertical) wedge."""
    gray = canon_img[..., 1].astype(np.float64) if canon_img.ndim == 3 else canon_img.astype(np.float64)
    h, w = gray.shape
    x0, y0, x1, y1 = spec.ROIS_NORM[roi_key].to_px(w, h)
    m = 6
    patch = gray[y0 + m : y1 - m, x0 + m : x1 - m]

    if roi_key == "wedge_hyp":
        # Vertical bars -> average vertically to suppress noise
        profile = patch.mean(axis=0)
        n = len(profile)
        u = np.linspace(0.0, 1.0, n)
        p0 = 1.0 / spec.WEDGE_FREQ_MIN_CYC_H
        p1 = 1.0 / spec.WEDGE_FREQ_MAX_CYC_H
        freqs_h = 1.0 / (p0 + (p1 - p0) * u)
    else:
        # Horizontal bars -> average horizontally
        profile = patch.mean(axis=1)
        n = len(profile)
        v = np.linspace(0.0, 1.0, n)
        freqs_h = spec.WEDGE_FREQ_MIN_CYC_H + (spec.WEDGE_FREQ_MAX_CYC_H - spec.WEDGE_FREQ_MIN_CYC_H) * v

    freqs_lr = freqs_h / lr_px_per_chart_h

    # Local RMS modulation in a sliding window (half-width ~12 px)
    win = 25
    kernel = np.ones(win, dtype=np.float64) / win
    local_mean = np.convolve(profile, kernel, mode="same")
    local_sq = np.convolve((profile - local_mean) ** 2, kernel, mode="same")
    mod = np.sqrt(np.maximum(local_sq, 0.0))
    ref_mod = max(float(np.percentile(mod[: max(15, n // 8)], 90)), 1e-8)
    norm_mod = mod / ref_mod

    def _lim(thresh: float) -> float:
        below = np.where(norm_mod < thresh)[0]
        if len(below) == 0:
            return float(freqs_lr[-1])
        return float(freqs_lr[int(below[0])])

    return {
        "f50_cyc_lr_px": round(_lim(0.50), 4),
        "f10_cyc_lr_px": round(_lim(0.10), 4),
    }


def measure_flat_noise(canon_img: np.ndarray) -> dict[str, float]:
    """Measure flat-patch noise sigma and mid-grey SNR on the 8-step grey wedge."""
    means, stds = _grey_wedge_patch_stats(canon_img)
    mean_sigma = float(np.mean(stds))
    # Mid-grey step (index 3 or 4)
    mid_sig = float(np.mean(stds[3:5]))
    mid_val = float(np.mean(means[3:5]))
    snr_db = 20.0 * math.log10(max(mid_val, 1e-6) / max(mid_sig, 1e-8))
    return {
        "flat_noise_sigma": round(mean_sigma, 6),
        "mid_grey_snr_db": round(snr_db, 2),
    }


def evaluate_chart_only(
    canon_img: np.ndarray,
    lr_px_per_chart_h: float,
    is_single_frame: bool = False,
) -> dict[str, Any]:
    """Run all Protocol B (chart-only) measurements on a canonical chart image."""
    return {
        "lr_px_per_chart_h": round(lr_px_per_chart_h, 2),
        "star_sin": measure_siemens_star(canon_img, spec.STAR_SIN_CENTER, lr_px_per_chart_h),
        "star_bin": measure_siemens_star(canon_img, spec.STAR_BIN_CENTER, lr_px_per_chart_h),
        "edge_high": measure_slanted_edge_mtf(canon_img, "edge_high", lr_px_per_chart_h, is_single_frame),
        "edge_low": measure_slanted_edge_mtf(canon_img, "edge_low", lr_px_per_chart_h, is_single_frame),
        "wedge_hyp": measure_wedge(canon_img, "wedge_hyp", lr_px_per_chart_h),
        "wedge_lin": measure_wedge(canon_img, "wedge_lin", lr_px_per_chart_h),
        "noise": measure_flat_noise(canon_img),
    }


# --------------------------------------------------------------------------- #
# 5. Summary panel builder (single | SR [| GT])
# --------------------------------------------------------------------------- #
def _to_srgb_u8(img: np.ndarray) -> np.ndarray:
    """Convert linear float32 RGB/gray image to displayable uint8 BGR for OpenCV."""
    f = np.clip(img.astype(np.float32), 0.0, 1.0)
    if f.ndim == 2:
        f = np.stack([f, f, f], axis=2)
    # Apply mild display gamma if linear
    g = np.power(f, 1.0 / 2.2)
    u8 = np.clip(np.round(g * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)


def save_summary_panel(
    images: list[tuple[str, np.ndarray]],
    out_path: str,
    crop_size: int = 360,
) -> None:
    """Save a side-by-side PNG panel of Siemens star center + hyperbolic wedge crops."""
    if not images:
        return
    h, w = images[0][1].shape[:2]

    # Star center crop (around sinusoidal star center)
    scx = int(round(spec.STAR_SIN_CENTER[0] * w))
    scy = int(round(spec.STAR_SIN_CENTER[1] * h))
    half = int(round(spec.STAR_RADIUS_H * h * 0.75))

    # Wedge crop (high-frequency half of hyperbolic wedge)
    wx0, wy0, wx1, wy1 = spec.ROIS_NORM["wedge_hyp"].to_px(w, h)

    cols = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    header_h = 34
    for label, img in images:
        bgr = _to_srgb_u8(img)
        star_crop = bgr[max(0, scy - half) : min(h, scy + half), max(0, scx - half) : min(w, scx + half)]
        star_tile = cv2.resize(star_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

        wedge_crop = bgr[wy0:wy1, wx0:wx1]
        wedge_h = int(round(crop_size * (wy1 - wy0) / max(1, wx1 - wx0)))
        wedge_tile = cv2.resize(wedge_crop, (crop_size, max(80, wedge_h)), interpolation=cv2.INTER_NEAREST)

        hdr = np.full((header_h, crop_size, 3), 24, dtype=np.uint8)
        cv2.putText(hdr, label, (10, 23), font, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
        sep = np.full((6, crop_size, 3), 60, dtype=np.uint8)
        cols.append(np.vstack([hdr, star_tile, sep, wedge_tile]))

    vsep = np.full((cols[0].shape[0], 6, 3), 60, dtype=np.uint8)
    panel = cols[0]
    for col in cols[1:]:
        panel = np.hstack([panel, vsep, col])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# --------------------------------------------------------------------------- #
# 6. High-level measurement entry point + CLI
# --------------------------------------------------------------------------- #
def measure_triplet(
    single_img: np.ndarray,
    sr_img: np.ndarray,
    ratio: float,
    gt_img: np.ndarray | None = None,
    out_json: str | None = None,
    out_panel: str | None = None,
) -> dict[str, Any]:
    """Run Protocol B (and Protocol A if ``gt_img`` is provided) on in-memory images."""
    H_s, px_h_s, _ = detect_fiducials_and_homography(single_img)
    H_sr, px_h_sr, _ = detect_fiducials_and_homography(sr_img)

    canon_single = warp_to_canonical(single_img, H_s)
    canon_sr = warp_to_canonical(sr_img, H_sr)

    # Effective LR chart height (single_img is already upsampled by ratio if bicubic)
    lr_px_h_sr = px_h_sr / float(ratio)
    lr_px_h_s = px_h_s / float(ratio)

    report: dict[str, Any] = {
        "ratio": ratio,
        "protocol": "zoom_gt+chart_only" if gt_img is not None else "chart_only",
        "chart_only": {
            "single": evaluate_chart_only(canon_single, lr_px_h_s, is_single_frame=True),
            "sr": evaluate_chart_only(canon_sr, lr_px_h_sr, is_single_frame=False),
        },
    }

    panel_items: list[tuple[str, np.ndarray]] = [
        (f"Single ({report['chart_only']['single']['star_sin']['f50_cyc_lr_px']:.2f} c/px)", canon_single),
        (f"SR {ratio:.0f}x ({report['chart_only']['sr']['star_sin']['f50_cyc_lr_px']:.2f} c/px)", canon_sr),
    ]

    if gt_img is not None:
        H_gt, px_h_gt, _ = detect_fiducials_and_homography(gt_img)
        canon_gt = warp_to_canonical(gt_img, H_gt)
        # Sub-pixel ECC refinement of single & SR onto GT in canonical space
        canon_single_ref = refine_canonical_ecc(canon_single, canon_gt)
        canon_sr_ref = refine_canonical_ecc(canon_sr, canon_gt)
        report["chart_only"]["gt"] = evaluate_chart_only(canon_gt, lr_px_h_sr, is_single_frame=False)
        report["zoom_gt"] = {
            "single_vs_gt": evaluate_zoom_gt(canon_single_ref, canon_gt),
            "sr_vs_gt": evaluate_zoom_gt(canon_sr_ref, canon_gt),
        }
        panel_items = [
            (f"Single ({report['zoom_gt']['single_vs_gt']['overall']['psnr_db']:.2f} dB)", photometric_match_grey_wedge(canon_single_ref, canon_gt)),
            (f"SR {ratio:.0f}x ({report['zoom_gt']['sr_vs_gt']['overall']['psnr_db']:.2f} dB)", photometric_match_grey_wedge(canon_sr_ref, canon_gt)),
            ("Zoom GT", canon_gt),
        ]

    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if out_panel:
        save_summary_panel(panel_items, out_panel)
    return report


def _load_dng_dir(directory: str) -> list[str]:
    exts = ("*.dng", "*.DNG", "*.cr2", "*.CR2", "*.nef", "*.NEF", "*.arw", "*.ARW")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def run_from_directories(
    burst_dir: str,
    gt_dir: str | None,
    ratio: int,
    out_dir: str,
    fusion: str = "demosaic_drizzle",
) -> dict[str, Any]:
    """Run raw_sr on a real DNG burst directory and measure against optional GT dir."""
    from raw_sr.core.superres import SuperResolution, load_raw_linear

    burst_files = _load_dng_dir(burst_dir)
    if len(burst_files) < 2:
        raise RuntimeError(f"Need at least 2 RAW/DNG files in {burst_dir}, found {len(burst_files)}")

    frames = [load_raw_linear(p) for p in burst_files]
    sr_engine = SuperResolution(ratio=float(ratio), fusion=fusion)
    res = sr_engine.run(frames)
    sr_img = res.hr.astype(np.float32)

    ref_frame = frames[res.reference_index]
    h_hr, w_hr = sr_img.shape[:2]
    single_img = cv2.resize(ref_frame.rgb.astype(np.float32), (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)

    gt_img: np.ndarray | None = None
    if gt_dir:
        gt_files = _load_dng_dir(gt_dir)
        if not gt_files:
            raise RuntimeError(f"No RAW/DNG files found in GT directory: {gt_dir}")
        # Median-stack registered GT frames in canonical space, or use single frame
        gt_frames = [load_raw_linear(p).rgb.astype(np.float32) for p in gt_files]
        if len(gt_frames) == 1:
            gt_img = gt_frames[0]
        else:
            # Align all GT frames to the first GT frame via fiducial homography
            H0, _, _ = detect_fiducials_and_homography(gt_frames[0])
            warped = [warp_to_canonical(gt_frames[0], H0)]
            for gf in gt_frames[1:]:
                Hg, _, _ = detect_fiducials_and_homography(gf)
                warped.append(refine_canonical_ecc(warp_to_canonical(gf, Hg), warped[0]))
            gt_img = np.median(np.stack(warped, axis=0), axis=0).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    return measure_triplet(
        single_img=single_img,
        sr_img=sr_img,
        ratio=float(ratio),
        gt_img=gt_img,
        out_json=os.path.join(out_dir, "chart_metrics.json"),
        out_panel=os.path.join(out_dir, "chart_panel.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure WP10 resolution test chart (Zoom GT and/or Chart-only).")
    parser.add_argument("--burst", required=True, help="Directory containing burst RAW/DNG frames at distance d.")
    parser.add_argument("--gt", default=None, help="Optional directory containing steady RAW/DNG GT frames at d/2 or d/3.")
    parser.add_argument("--ratio", type=int, default=2, choices=[2, 3, 4], help="Super-resolution scale factor.")
    parser.add_argument("--fusion", default="demosaic_drizzle", choices=["demosaic_drizzle", "cfa_drizzle"], help="Fusion mode.")
    parser.add_argument("--out-dir", default="/tmp/rawsr/wp10/measure", help="Output directory for JSON and panel PNG.")
    args = parser.parse_args()

    report = run_from_directories(args.burst, args.gt, args.ratio, args.out_dir, args.fusion)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
