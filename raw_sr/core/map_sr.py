"""WP8: Full forward-model MAP super-resolution (joint demosaic + SR + deconvolution).

Classical (non-neural) inverse-problem solver following Farsiu, Robinson, Elad &
Milanfar (2004, 2006):

    y_k = D * W_k * B * x + n_k

where:
- x is the HR linear RGB image (H_hr, W_hr, 3), initialised from demosaic_drizzle;
- B is the separable 2D PSF (pixel aperture box * Gaussian lens sigma);
- W_k is the bilinear warp from HR to frame k's LR grid (exact inverse of
  ``to_drizzle_matrix``);
- D is the Bayer CFA channel sampling operator;
- y_k are the calibrated raw Bayer mosaics.

Objective:
    J(x) = sum_k w_k * sum_p rho_delta((A_k x)_p - y_{k,p})
           + lam_btv * BTV(Y(x))
           + lam_chroma * 0.5 * (||grad(R - G)||_2^2 + ||grad(B - G)||_2^2)

Solved via Barzilai-Borwein projected gradient descent with Armijo backtracking
line search and Morozov/relative-change stopping. Supports GPU-resident CuPy
execution with transparent CPU fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cfa_drizzle import channel_map, pattern_array
from .superres import SENSOR_PSF_SIGMA_PER_RATIO, to_drizzle_matrix


@dataclass
class MAPSRResult:
    """Output of :func:`map_super_resolution`."""
    hr: np.ndarray
    metrics: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Noise estimation & PSF kernel construction
# --------------------------------------------------------------------------- #

def immerkaer_sigma(img2d: np.ndarray) -> float:
    """Fast Immerkaer (1996) noise standard deviation estimator on a 2D plane."""
    a = np.asarray(img2d, dtype=np.float64)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 1e-4
    # 3x3 Immerkaer Laplacian mask: [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    resp = (
        a[:-2, :-2] - 2.0 * a[:-2, 1:-1] + a[:-2, 2:]
        - 2.0 * a[1:-1, :-2] + 4.0 * a[1:-1, 1:-1] - 2.0 * a[1:-1, 2:]
        + a[2:, :-2] - 2.0 * a[2:, 1:-1] + a[2:, 2:]
    )
    sigma = float(np.mean(np.abs(resp)) * np.sqrt(0.5 * np.pi) / 6.0)
    return max(sigma, 1e-5)


def build_psf_kernel_1d(ratio: float, lens_sigma_lr: Optional[float] = None) -> np.ndarray:
    """Build a symmetric, unit-sum 1D PSF kernel in HR pixels.

    - ``lens_sigma_lr is None``: uses the existing ``raw_sr`` sensor model
      ``sigma_hr = SENSOR_PSF_SIGMA_PER_RATIO * ratio`` (Gaussian approximation
      to the 1-LR-pixel integration window).
    - ``lens_sigma_lr >= 0``: convolves the 1D pixel aperture box of width
      ``ratio`` HR px with a Gaussian of std ``lens_sigma_lr * ratio`` HR px.
    """
    ratio = float(ratio)
    if lens_sigma_lr is None:
        sigma_hr = SENSOR_PSF_SIGMA_PER_RATIO * ratio
        r = max(1, int(np.ceil(3.0 * sigma_hr)))
        x = np.arange(-r, r + 1, dtype=np.float64)
        k = np.exp(-0.5 * (x / max(sigma_hr, 1e-6)) ** 2)
        k /= k.sum()
        return k.astype(np.float32)

    # Pixel aperture box of width `ratio` HR pixels (symmetric)
    half_box = 0.5 * ratio
    r_box = int(np.ceil(half_box - 0.5))
    xs = np.arange(-r_box, r_box + 1, dtype=np.float64)
    # Overlap of pixel interval [x - 0.5, x + 0.5] with [-half_box, +half_box]
    lo = np.maximum(xs - 0.5, -half_box)
    hi = np.minimum(xs + 0.5, half_box)
    box = np.maximum(0.0, hi - lo)
    if box.sum() <= 0:
        box = np.array([1.0], dtype=np.float64)
    else:
        box /= box.sum()

    sigma_hr = float(lens_sigma_lr) * ratio
    if sigma_hr <= 1e-4:
        k = box
    else:
        r_g = max(1, int(np.ceil(3.0 * sigma_hr)))
        xg = np.arange(-r_g, r_g + 1, dtype=np.float64)
        g = np.exp(-0.5 * (xg / sigma_hr) ** 2)
        g /= g.sum()
        k = np.convolve(box, g, mode="full")
        k /= k.sum()
    # Enforce exact symmetry
    k = 0.5 * (k + k[::-1])
    k /= k.sum()
    return k.astype(np.float32)


def build_psf_kernel_1d_matched(ratio: float, lens_sigma_lr: float) -> np.ndarray:
    """Build 1D PSF kernel in HR px whose convolution with bilinear warp W_k matches
    the physical pixel-aperture box (var = ratio^2 / 12) * lens Gaussian (var = (sigma*ratio)^2).

    Bilinear gather W_k adds ~1/6 HR px^2 variance on average; subtracting it prevents
    systematic PSF over-estimation and edge ringing during deconvolution.
    """
    r = float(ratio)
    var_phys = (r * r) / 12.0 + (float(lens_sigma_lr) * r) ** 2
    var_k = max(var_phys - 1.0 / 6.0, 0.01)
    sigma_hr = float(np.sqrt(var_k))
    rad = max(1, int(np.ceil(3.0 * sigma_hr)))
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_hr) ** 2)
    k = 0.5 * (k + k[::-1])
    k /= k.sum()
    return k.astype(np.float32)


def build_psf_kernels_rgb(
    ratio: float,
    lens_sigma_lr: Optional[Union[float, Sequence[float]]] = None,
    *,
    compensate_warp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-channel (R, G, B) symmetric unit-sum 1D PSF kernels in HR pixels.

    Accepts ``None`` (legacy sensor-only Gaussian), a scalar ``float`` (shared
    box * Gaussian across RGB), or a 3-element sequence ``(sigma_r, sigma_g, sigma_b)``
    in LR pixels to model lateral chromatic blur.
    """
    if lens_sigma_lr is None:
        k = build_psf_kernel_1d(ratio, lens_sigma_lr=None)
        return (k, k, k)
    builder = build_psf_kernel_1d_matched if compensate_warp else build_psf_kernel_1d
    if isinstance(lens_sigma_lr, (int, float, np.floating, np.integer)):
        k = builder(ratio, float(lens_sigma_lr))
        return (k, k, k)
    vals = list(lens_sigma_lr)
    if len(vals) != 3:
        raise ValueError(f"lens_sigma_lr sequence must have length 3, got {len(vals)}")
    return (
        builder(ratio, float(vals[0])),
        builder(ratio, float(vals[1])),
        builder(ratio, float(vals[2])),
    )


def _correct_psf_rgb(
    raw_rgb: Tuple[float, float, float],
    bias_lr: float,
    min_sigma_lr: float,
    max_sigma_lr: float,
) -> List[float]:
    bias = float(max(bias_lr, 0.0))
    bias2 = bias * bias
    # On Bayer demosaic / drizzle images, R and B share identical 2x sparser
    # sampling geometry so demosaic adds a common variance excess to R and B
    # over G. Subtracting only min(var_R, var_B) - var_G removes demosaic blur
    # while preserving true dispersive lateral chromatic blur (R > G > B or B > G > R).
    rb_excess2 = max(min(raw_rgb[0] ** 2, raw_rgb[2] ** 2) - raw_rgb[1] ** 2, 0.0) if bias > 0.0 else 0.0
    sub2 = (bias2 + rb_excess2, bias2, bias2 + rb_excess2)
    return [
        float(np.clip(np.sqrt(max(s * s - d2, min_sigma_lr * min_sigma_lr)), min_sigma_lr, max_sigma_lr))
        for s, d2 in zip(raw_rgb, sub2)
    ]


def estimate_burst_psf_rgb(
    init_hr: np.ndarray,
    ratio: float,
    *,
    ref_rgb: Optional[np.ndarray] = None,
    drizzle_bias_lr: float = 0.10,
    ref_demosaic_bias_lr: float = 0.40,
    min_edges: int = 15,
    min_sigma_lr: float = 0.15,
    max_sigma_lr: float = 0.90,
    max_channel_spread: float = 0.15,
    default_sigma_lr: float = 0.35,
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    """Robustly estimate per-channel (R, G, B) lens PSF sigma (LR px).

    Uses WP6 structure-tensor slanted-edge ESF fitting (:func:`estimate_psf_sigma_edges`).
    When ``ref_rgb`` (the single demosaiced reference LR frame) is supplied, it
    estimates the PSF on both the single-frame LR image (free of multi-frame
    parallax/drizzle smear, corrected for 1x demosaic blur) and ``init_hr``
    (corrected for ``drizzle_bias_lr``), requires at least ``min_edges`` clean
    edges for reliability, and takes the per-channel minimum so neither parallax
    smear nor 1x demosaic blur can inflate the deconvolution kernel.
    """
    from .enhance import estimate_psf_sigma_edges

    hr_info = estimate_psf_sigma_edges(
        init_hr,
        ratio=float(ratio),
        default_sigma_lr=float(default_sigma_lr),
        min_sigma_lr=float(min_sigma_lr),
        max_sigma_lr=1.35,
    )
    hr_raw = tuple(float(v) for v in hr_info.sigma_rgb)
    hr_corr = (
        _correct_psf_rgb(hr_raw, drizzle_bias_lr, min_sigma_lr, max_sigma_lr)
        if not hr_info.used_fallback else None
    )

    ref_corr: Optional[List[float]] = None
    ref_raw: Optional[Tuple[float, float, float]] = None
    ref_n_edges = 0
    if ref_rgb is not None:
        ref_info = estimate_psf_sigma_edges(
            ref_rgb,
            ratio=1.0,
            default_sigma_lr=float(default_sigma_lr),
            min_sigma_lr=float(min_sigma_lr),
            max_sigma_lr=1.35,
        )
        ref_n_edges = int(ref_info.n_edges)
        ref_raw = tuple(float(v) for v in ref_info.sigma_rgb)
        if not ref_info.used_fallback and ref_n_edges >= int(min_edges):
            ref_corr = _correct_psf_rgb(ref_raw, ref_demosaic_bias_lr, min_sigma_lr, max_sigma_lr)

    if ref_rgb is not None:
        hr_reliable = (hr_corr is not None) and (int(hr_info.n_edges) >= int(min_edges))
        if ref_corr is not None and hr_reliable:
            corr = [min(r, h) for r, h in zip(ref_corr, hr_corr)]  # type: ignore[arg-type]
            source = "wp15_ref_hr_min"
        elif ref_corr is not None:
            corr = ref_corr
            source = "wp15_ref_edges_rgb"
        elif hr_reliable:
            corr = hr_corr  # type: ignore[assignment]
            source = "wp15_hr_edges_rgb"
        else:
            s = float(np.clip(default_sigma_lr, min_sigma_lr, max_sigma_lr))
            return (s, s, s), {
                "source": "wp15_low_edge_fallback",
                "n_edges": int(hr_info.n_edges),
                "ref_n_edges": ref_n_edges,
                "raw_rgb": [round(v, 4) for v in hr_raw],
                "hr_sigma_rgb": [round(v, 4) for v in hr_corr] if hr_corr is not None else None,
                "ref_sigma_rgb": [round(v, 4) for v in ref_raw] if ref_raw is not None else None,
                "sigma_rgb": [round(s, 4)] * 3,
            }
    else:
        if hr_corr is None:
            s = float(np.clip(default_sigma_lr, min_sigma_lr, max_sigma_lr))
            return (s, s, s), {
                "source": "fallback",
                "n_edges": int(hr_info.n_edges),
                "raw_rgb": [round(v, 4) for v in hr_raw],
                "sigma_rgb": [round(s, 4)] * 3,
            }
        corr = hr_corr
        source = "wp13_edges_rgb"

    # Anchor R and B within +-max_channel_spread of G to prevent chroma instability
    sg = corr[1]
    spread = float(max(max_channel_spread, 0.0))
    sr = float(np.clip(corr[0], max(min_sigma_lr, sg - spread), min(max_sigma_lr, sg + spread)))
    sb = float(np.clip(corr[2], max(min_sigma_lr, sg - spread), min(max_sigma_lr, sg + spread)))
    sigma_rgb = (sr, sg, sb)
    meta: Dict[str, Any] = {
        "source": source,
        "n_edges": int(hr_info.n_edges),
        "raw_rgb": [round(v, 4) for v in hr_raw],
        "sigma_rgb": [round(v, 4) for v in sigma_rgb],
    }
    if ref_rgb is not None:
        meta["ref_n_edges"] = ref_n_edges
        meta["hr_sigma_rgb"] = [round(v, 4) for v in hr_corr] if hr_corr is not None else None
        meta["ref_sigma_rgb"] = [round(v, 4) for v in ref_corr] if ref_corr is not None else (
            [round(v, 4) for v in ref_raw] if ref_raw is not None else None
        )
    return sigma_rgb, meta


def _compute_local_envelope(
    x0: np.ndarray,
    sigma_med: float,
    env_tol: Optional[float],
    env_radius: int,
    mismatch_hr: Optional[np.ndarray] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute per-pixel [x_lo, x_hi] adaptive local min/max envelope from ``x0``.

    When ``mismatch_hr`` (per-HR-pixel inter-frame standardized residual std) is
    provided, the tight envelope ``[local_min - m, local_max + m]`` is activated
    proportionally to local model-mismatch risk (parallax/misregistration where
    ``mismatch_hr > 1.5``) and near-clipped highlight risk (``local_max > 0.60``),
    while staying unconstrained ``[0, 1]`` on clean, static, well-exposed regions.
    """
    if env_tol is None or float(env_tol) < 0.0 or int(env_radius) <= 0:
        return None
    import cv2
    k = 2 * int(env_radius) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    lo_ch = [cv2.erode(x0[..., c], se, borderType=cv2.BORDER_REPLICATE) for c in range(3)]
    hi_ch = [cv2.dilate(x0[..., c], se, borderType=cv2.BORDER_REPLICATE) for c in range(3)]
    local_min = np.stack(lo_ch, axis=2).astype(np.float32)
    local_max = np.stack(hi_ch, axis=2).astype(np.float32)
    local_range = np.maximum(local_max - local_min, np.float32(0.0))
    margin = np.float32(env_tol) * local_range + np.float32(2.0 * max(float(sigma_med), 1e-4))
    env_lo = np.clip(local_min - margin, np.float32(0.0), np.float32(1.0))
    env_hi = np.clip(local_max + margin, np.float32(0.0), np.float32(1.0))

    if mismatch_hr is None:
        return env_lo, env_hi

    mis_smooth = cv2.GaussianBlur(
        np.asarray(mismatch_hr, dtype=np.float32), (7, 7), 1.5,
        borderType=cv2.BORDER_REPLICATE,
    )
    mis_risk = np.clip((mis_smooth - np.float32(2.0)) / np.float32(2.0), np.float32(0.0), np.float32(1.0))
    hi_max = np.max(local_max, axis=2)
    hi_risk = np.clip((hi_max - np.float32(0.72)) / np.float32(0.20), np.float32(0.0), np.float32(1.0))
    risk = np.maximum(mis_risk, hi_risk)[..., None]

    x_lo = (risk * env_lo).astype(np.float32)
    x_hi = ((np.float32(1.0) - risk) + risk * env_hi).astype(np.float32)
    return x_lo, x_hi


def estimate_poisson_gaussian_noise(
    ref_mosaic: np.ndarray,
    pattern: Any,
    *,
    lo: float = 0.01,
    hi: float = 0.90,
) -> Tuple[float, float, float]:
    """Estimate Poisson-Gaussian noise model ``var(y) = a + b * max(y, 0)`` from a Bayer mosaic.

    Extracts one green sub-grid (clean single-channel 2D plane), computes the
    3x3 Immerkaer Laplacian, rejects high-gradient edge/texture pixels, and fits
    ``var ~ a + b * y`` across intensity quantile bins using robust MAD.

    Returns ``(read_var_a, shot_gain_b, sigma_median)``.
    """
    pat = pattern_array(pattern)
    g_sites = [(r, c) for r in range(2) for c in range(2) if int(pat[r, c]) in (1, 3)]
    if g_sites:
        r0, c0 = g_sites[0]
        plane = np.asarray(ref_mosaic[r0::2, c0::2], dtype=np.float64)
    else:
        plane = np.asarray(ref_mosaic, dtype=np.float64)

    if plane.shape[0] < 12 or plane.shape[1] < 12:
        s = max(immerkaer_sigma(plane), 5e-4)
        return float(s * s), 0.0, float(s)

    # 3x3 Immerkaer Laplacian on interior
    resp = (
        plane[:-2, :-2] - 2.0 * plane[:-2, 1:-1] + plane[:-2, 2:]
        - 2.0 * plane[1:-1, :-2] + 4.0 * plane[1:-1, 1:-1] - 2.0 * plane[1:-1, 2:]
        + plane[2:, :-2] - 2.0 * plane[2:, 1:-1] + plane[2:, 2:]
    )
    center = plane[1:-1, 1:-1]
    # Local gradient magnitude to exclude edges & fine textures
    gx = 0.5 * (plane[1:-1, 2:] - plane[1:-1, :-2])
    gy = 0.5 * (plane[2:, 1:-1] - plane[:-2, 1:-1])
    gmag = np.hypot(gx, gy)

    valid = (center > lo) & (center < hi) & np.isfinite(center) & np.isfinite(resp)
    if int(np.sum(valid)) < 256:
        mad = float(np.median(np.abs(resp))) if resp.size else 1e-3
        s = max(mad / (6.0 * 0.67448975), 5e-4)
        return float(s * s), 0.0, float(s)

    # Keep quieter 60% of valid pixels (rejects strong edges and periodic textures)
    g_thr = float(np.percentile(gmag[valid], 60.0))
    quiet = valid & (gmag <= max(g_thr, 1e-6))
    vals = center[quiet]
    abs_r = np.abs(resp[quiet])

    # Overall robust sigma
    sigma_all = max(float(np.median(abs_r)) / (6.0 * 0.67448975), 3e-4)
    med_y = float(np.median(vals)) if vals.size else 0.2

    # Intensity quantile bins for heteroscedastic fit
    qs = np.linspace(0.0, 100.0, 6)
    edges = np.percentile(vals, qs)
    mus: List[float] = []
    vars_b: List[float] = []
    for i in range(len(edges) - 1):
        m = (vals >= edges[i]) & (vals <= edges[i + 1])
        if int(np.sum(m)) >= 64:
            sb = max(float(np.median(abs_r[m])) / (6.0 * 0.67448975), 2e-4)
            mus.append(float(np.median(vals[m])))
            vars_b.append(sb * sb)

    if len(mus) >= 3 and (max(mus) - min(mus)) >= 0.04:
        mu_arr = np.asarray(mus, dtype=np.float64)
        var_arr = np.asarray(vars_b, dtype=np.float64)
        # Ordinary least squares var = a + b * mu, clamped to physical a >= 1e-8, b >= 0
        X = np.stack([np.ones_like(mu_arr), mu_arr], axis=1)
        coef, *_ = np.linalg.lstsq(X, var_arr, rcond=None)
        a = float(coef[0])
        b = float(coef[1])
        if b < 0.0:
            b = 0.0
            a = float(np.median(var_arr))
        if a < 1e-8:
            # Refit b through origin + floor
            a = 1e-8
            b = float(max(0.0, np.dot(mu_arr, np.maximum(var_arr - a, 0.0)) / max(np.dot(mu_arr, mu_arr), 1e-12)))
        # Sanity cap: variance at med_y should stay consistent with sigma_all^2
        sigma_med = max(float(np.sqrt(a + b * max(med_y, 0.0))), 3e-4)
        return a, b, sigma_med

    return float(sigma_all * sigma_all), 0.0, float(sigma_all)


# --------------------------------------------------------------------------- #
# Exact forward / adjoint 1D replicate-boundary convolution (CPU)
# --------------------------------------------------------------------------- #

def _conv1d_replicate_axis(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """1D convolution along ``axis`` with edge replication (forward B)."""
    r = len(k) // 2
    if r == 0:
        return arr * k[0]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode="edge")
    n = arr.shape[axis]
    out = np.zeros_like(arr)
    for d in range(-r, r + 1):
        w = k[d + r]
        if w == 0:
            continue
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(r + d, r + d + n)
        out += w * padded[tuple(sl)]
    return out


def _conv1d_replicate_adjoint_axis(arr: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Exact adjoint of :func:`_conv1d_replicate_axis` along ``axis``."""
    r = len(k) // 2
    if r == 0:
        return arr * k[0]
    n = arr.shape[axis]
    out = np.zeros_like(arr)
    for d in range(-r, r + 1):
        w = k[d + r]
        if w == 0:
            continue
        # Forward: out[i] += w * in[clip(i + d, 0, n - 1)]
        # Adjoint: in_adj[clip(i + d, 0, n - 1)] += w * out[i]
        if d == 0:
            out += w * arr
        elif d > 0:
            # i in [0, n - 1 - d]: clip(i+d) = i+d
            sl_dst = [slice(None)] * arr.ndim
            sl_src = [slice(None)] * arr.ndim
            sl_dst[axis] = slice(d, n)
            sl_src[axis] = slice(0, n - d)
            out[tuple(sl_dst)] += w * arr[tuple(sl_src)]
            # i in [n - d, n - 1]: clip(i+d) = n - 1
            sl_end = [slice(None)] * arr.ndim
            sl_tail = [slice(None)] * arr.ndim
            sl_end[axis] = n - 1
            sl_tail[axis] = slice(n - d, n)
            out[tuple(sl_end)] += w * np.sum(arr[tuple(sl_tail)], axis=axis)
        else:
            # d < 0, let m = -d > 0
            m = -d
            # i in [m, n - 1]: clip(i - m) = i - m
            sl_dst = [slice(None)] * arr.ndim
            sl_src = [slice(None)] * arr.ndim
            sl_dst[axis] = slice(0, n - m)
            sl_src[axis] = slice(m, n)
            out[tuple(sl_dst)] += w * arr[tuple(sl_src)]
            # i in [0, m - 1]: clip(i - m) = 0
            sl_beg = [slice(None)] * arr.ndim
            sl_head = [slice(None)] * arr.ndim
            sl_beg[axis] = 0
            sl_head[axis] = slice(0, m)
            out[tuple(sl_beg)] += w * np.sum(arr[tuple(sl_head)], axis=axis)
    return out


def blur_forward_cpu(x: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """Separable 2D PSF blur B(x) on (H, W, 3) with exact replicate boundaries."""
    tmp = _conv1d_replicate_axis(x, k1d, axis=1)
    return _conv1d_replicate_axis(tmp, k1d, axis=0)


def blur_adjoint_cpu(y: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """Exact adjoint B^T(y) of :func:`blur_forward_cpu`."""
    tmp = _conv1d_replicate_adjoint_axis(y, k1d, axis=0)
    return _conv1d_replicate_adjoint_axis(tmp, k1d, axis=1)


def _kernels_equal(k1d_rgb: Sequence[np.ndarray]) -> bool:
    return (
        len(k1d_rgb[0]) == len(k1d_rgb[1]) == len(k1d_rgb[2])
        and np.allclose(k1d_rgb[0], k1d_rgb[1], atol=1e-7)
        and np.allclose(k1d_rgb[0], k1d_rgb[2], atol=1e-7)
    )


def blur_forward_rgb_cpu(x: np.ndarray, k1d_rgb: Sequence[np.ndarray]) -> np.ndarray:
    """Per-channel separable 2D PSF blur B(x) on (H, W, 3) with exact replicate boundaries."""
    if _kernels_equal(k1d_rgb):
        return blur_forward_cpu(x, k1d_rgb[0].astype(x.dtype, copy=False))
    out = np.empty_like(x)
    for c in range(3):
        kc = k1d_rgb[c].astype(x.dtype, copy=False)
        tmp = _conv1d_replicate_axis(x[..., c], kc, axis=1)
        out[..., c] = _conv1d_replicate_axis(tmp, kc, axis=0)
    return out


def blur_adjoint_rgb_cpu(y: np.ndarray, k1d_rgb: Sequence[np.ndarray]) -> np.ndarray:
    """Exact adjoint B^T(y) of :func:`blur_forward_rgb_cpu`."""
    if _kernels_equal(k1d_rgb):
        return blur_adjoint_cpu(y, k1d_rgb[0].astype(y.dtype, copy=False))
    out = np.empty_like(y)
    for c in range(3):
        kc = k1d_rgb[c].astype(y.dtype, copy=False)
        tmp = _conv1d_replicate_adjoint_axis(y[..., c], kc, axis=0)
        out[..., c] = _conv1d_replicate_adjoint_axis(tmp, kc, axis=1)
    return out


# --------------------------------------------------------------------------- #
# Exact forward / adjoint CFA warp operator A_k = D * W_k * B (CPU)
# --------------------------------------------------------------------------- #

@dataclass
class _FrameWarpCache:
    """Precomputed bilinear gather/scatter indices and weights for one LR frame."""
    idx00: np.ndarray  # int32 flat indices into (H_hr * W_hr * 3)
    idx10: np.ndarray
    idx01: np.ndarray
    idx11: np.ndarray
    w00: np.ndarray    # float32
    w10: np.ndarray
    w01: np.ndarray
    w11: np.ndarray
    valid: np.ndarray  # bool (H_lr, W_lr)
    valid_flat: np.ndarray  # int32 indices into (H_lr * W_lr)


def _precompute_warp_cache(
    M: np.ndarray,
    lr_shape: Tuple[int, int],
    hr_shape: Tuple[int, int],
    ch_map: np.ndarray,
) -> _FrameWarpCache:
    h_lr, w_lr = int(lr_shape[0]), int(lr_shape[1])
    h_hr, w_hr = int(hr_shape[0]), int(hr_shape[1])
    M64 = np.asarray(M, dtype=np.float64)

    xs, ys = np.meshgrid(
        np.arange(w_lr, dtype=np.float64),
        np.arange(h_lr, dtype=np.float64),
    )
    denom = M64[2, 0] * xs + M64[2, 1] * ys + M64[2, 2]
    ok_d = np.abs(denom) > 1e-12
    safe_d = np.where(ok_d, denom, 1.0)
    u = (M64[0, 0] * xs + M64[0, 1] * ys + M64[0, 2]) / safe_d
    v = (M64[1, 0] * xs + M64[1, 1] * ys + M64[1, 2]) / safe_d

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    valid = ok_d & (u0 >= 0) & (u0 + 1 < w_hr) & (v0 >= 0) & (v0 + 1 < h_hr)
    valid_flat = np.flatnonzero(valid.ravel()).astype(np.int32)

    uv_u0 = u0.ravel()[valid_flat]
    uv_v0 = v0.ravel()[valid_flat]
    fu = (u.ravel()[valid_flat] - uv_u0).astype(np.float32)
    fv = (v.ravel()[valid_flat] - uv_v0).astype(np.float32)
    ch = ch_map.ravel()[valid_flat].astype(np.int64)

    w00 = (1.0 - fu) * (1.0 - fv)
    w10 = fu * (1.0 - fv)
    w01 = (1.0 - fu) * fv
    w11 = fu * fv

    stride_y = w_hr * 3
    idx00 = (uv_v0 * stride_y + uv_u0 * 3 + ch).astype(np.int32)
    idx10 = (uv_v0 * stride_y + (uv_u0 + 1) * 3 + ch).astype(np.int32)
    idx01 = ((uv_v0 + 1) * stride_y + uv_u0 * 3 + ch).astype(np.int32)
    idx11 = ((uv_v0 + 1) * stride_y + (uv_u0 + 1) * 3 + ch).astype(np.int32)

    return _FrameWarpCache(
        idx00=idx00, idx10=idx10, idx01=idx01, idx11=idx11,
        w00=w00, w10=w10, w01=w01, w11=w11,
        valid=valid, valid_flat=valid_flat,
    )


def cfa_warp_forward_cpu(z: np.ndarray, cache: _FrameWarpCache, lr_shape: Tuple[int, int]) -> np.ndarray:
    """Compute D * W_k(z) on CPU using precomputed bilinear taps."""
    zf = z.ravel()
    vals = (
        cache.w00 * zf[cache.idx00]
        + cache.w10 * zf[cache.idx10]
        + cache.w01 * zf[cache.idx01]
        + cache.w11 * zf[cache.idx11]
    )
    out = np.zeros(lr_shape[0] * lr_shape[1], dtype=z.dtype)
    out[cache.valid_flat] = vals
    return out.reshape(lr_shape)


def cfa_warp_adjoint_cpu(r: np.ndarray, cache: _FrameWarpCache, hr_shape: Tuple[int, int]) -> np.ndarray:
    """Exact adjoint W_k^T * D^T(r) on CPU."""
    rf = r.ravel()[cache.valid_flat].astype(np.float64)
    n_hr = hr_shape[0] * hr_shape[1] * 3
    acc = np.bincount(cache.idx00, weights=rf * cache.w00, minlength=n_hr)
    acc += np.bincount(cache.idx10, weights=rf * cache.w10, minlength=n_hr)
    acc += np.bincount(cache.idx01, weights=rf * cache.w01, minlength=n_hr)
    acc += np.bincount(cache.idx11, weights=rf * cache.w11, minlength=n_hr)
    return acc.reshape(hr_shape[0], hr_shape[1], 3).astype(r.dtype)


class MAPForwardModel:
    """Consistent forward/adjoint operator A = [D * W_k * B * C]_k for testing and CPU solve."""

    def __init__(
        self,
        transforms: Sequence[Any],
        ratio: float,
        lr_shape: Tuple[int, int],
        hr_shape: Tuple[int, int],
        pattern: Any,
        lens_sigma_lr: Optional[Union[float, Sequence[float]]] = None,
        colour_matrix: Optional[np.ndarray] = None,
    ):
        self.ratio = float(ratio)
        self.lr_shape = (int(lr_shape[0]), int(lr_shape[1]))
        self.hr_shape = (int(hr_shape[0]), int(hr_shape[1]))
        self.pattern = pattern_array(pattern)
        self.ch_map = channel_map(self.pattern, self.lr_shape)
        self.k1d_rgb = build_psf_kernels_rgb(self.ratio, lens_sigma_lr=lens_sigma_lr)
        self.k1d = self.k1d_rgb[1]
        if colour_matrix is None:
            self.colour_matrix = np.eye(3, dtype=np.float64)
        else:
            C = np.asarray(colour_matrix, dtype=np.float64)
            if C.shape != (3, 3):
                raise ValueError(f"colour_matrix must be (3, 3), got {C.shape}")
            self.colour_matrix = C.copy()
        self.matrices = [to_drizzle_matrix(W, self.ratio).astype(np.float64) for W in transforms]
        self.caches = [
            _precompute_warp_cache(M, self.lr_shape, self.hr_shape, self.ch_map)
            for M in self.matrices
        ]

    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        """Apply A(x) = [D * W_k * C * B_rgb(x)]_k -> list of (H_lr, W_lr) Bayer mosaics."""
        z = blur_forward_rgb_cpu(x, self.k1d_rgb)
        z_cam = np.matmul(z, self.colour_matrix.T.astype(x.dtype))
        return [cfa_warp_forward_cpu(z_cam, c, self.lr_shape) for c in self.caches]

    def adjoint(self, ys: Sequence[np.ndarray]) -> np.ndarray:
        """Apply A^T(ys) = B_rgb^T * C^T * sum_k W_k^T * D^T(y_k) -> (H_hr, W_hr, 3)."""
        dtype = ys[0].dtype
        acc = np.zeros((self.hr_shape[0], self.hr_shape[1], 3), dtype=dtype)
        for y, c in zip(ys, self.caches):
            acc += cfa_warp_adjoint_cpu(y, c, self.hr_shape)
        acc_rgb = np.matmul(acc, self.colour_matrix.astype(dtype))
        return blur_adjoint_rgb_cpu(acc_rgb, self.k1d_rgb)


# --------------------------------------------------------------------------- #
# Colour matrix estimation & Gr/Gb green-imbalance equalization (WP11)
# --------------------------------------------------------------------------- #

def estimate_colour_matrix(
    ref_rgb: np.ndarray,
    ref_mosaic: np.ndarray,
    pattern: Any,
    *,
    lo: float = 0.005,
    hi: float = 0.90,
    border: int = 16,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Estimate the 3x3 matrix C mapping output linear RGB -> WB-scaled camera mosaic space.

    Fits ``ref_mosaic[site_c] ~ ref_rgb[site_c] @ C[c]`` on interior unclipped pixels
    for each colour channel ``c in (0, 1, 2)``. Falls back to ``I_3`` if the ROI has
    too few unclipped pixels or the fit is ill-conditioned.
    """
    rgb = np.asarray(ref_rgb, dtype=np.float64)
    mos = np.asarray(ref_mosaic, dtype=np.float64)
    pat = pattern_array(pattern)
    h, w = mos.shape
    ch = channel_map(pat, (h, w))

    b = min(border, h // 8, w // 8)
    mask = np.zeros((h, w), dtype=bool)
    mask[b:h - b, b:w - b] = True
    unclipped = (
        mask
        & np.all(rgb > lo, axis=2)
        & np.all(rgb < hi, axis=2)
        & (mos > lo)
        & (mos < hi)
    )

    C = np.eye(3, dtype=np.float64)
    rms_before: List[float] = []
    rms_after: List[float] = []
    counts: List[int] = []
    for c in range(3):
        sel = (ch == c) & unclipped
        n_c = int(np.sum(sel))
        counts.append(n_c)
        if n_c < 200:
            return np.eye(3, dtype=np.float64), {
                "fallback": True,
                "reason": f"insufficient_pixels_ch{c}:{n_c}",
                "cond": 1.0,
            }
        X = rgb[sel]
        y = mos[sel]
        w_c, *_ = np.linalg.lstsq(X, y, rcond=None)
        C[c] = w_c
        rms_before.append(float(np.sqrt(np.mean((X[:, c] - y) ** 2))))
        rms_after.append(float(np.sqrt(np.mean((X @ w_c - y) ** 2))))

    cond = float(np.linalg.cond(C))
    if not np.all(np.isfinite(C)) or cond > 20.0:
        return np.eye(3, dtype=np.float64), {
            "fallback": True,
            "reason": f"ill_conditioned:{cond:.2f}",
            "cond": cond,
        }

    return C, {
        "fallback": False,
        "cond": round(cond, 3),
        "rms_before": [round(v, 6) for v in rms_before],
        "rms_after": [round(v, 6) for v in rms_after],
        "counts": counts,
    }


def equalize_green_imbalance(
    mosaics: Sequence[np.ndarray],
    pattern: Any,
    ref_index: int = 0,
    *,
    lo: float = 0.02,
    hi: float = 0.90,
    min_diff: float = 0.002,
    max_diff: float = 0.05,
) -> Tuple[List[np.ndarray], float]:
    """Equalize Gr/Gb green-channel gain imbalance across a Bayer mosaic burst.

    Measures the mean ratio ``G0 / G1`` on interior unclipped green sites of the
    reference mosaic. If the imbalance is between ``min_diff`` and ``max_diff``,
    scales ``G0`` by ``1/sqrt(r)`` and ``G1`` by ``sqrt(r)`` on copies of all frames.
    """
    pat = pattern_array(pattern)
    g_sites = [(r, c) for r in range(2) for c in range(2) if int(pat[r, c]) in (1, 3)]
    if len(g_sites) != 2 or not mosaics:
        return [np.asarray(m, dtype=np.float32) for m in mosaics], 1.0

    ref = np.asarray(mosaics[ref_index], dtype=np.float64)
    h, w = ref.shape
    b = min(16, h // 8, w // 8)
    b = b - (b % 2)
    r0, c0 = g_sites[0]
    r1, c1 = g_sites[1]
    g0 = ref[b + r0:h - b:2, b + c0:w - b:2]
    g1 = ref[b + r1:h - b:2, b + c1:w - b:2]
    hm = min(g0.shape[0], g1.shape[0])
    wm = min(g0.shape[1], g1.shape[1])
    g0 = g0[:hm, :wm]
    g1 = g1[:hm, :wm]
    valid = (g0 > lo) & (g0 < hi) & (g1 > lo) & (g1 < hi)
    if int(np.sum(valid)) < 200:
        return [np.asarray(m, dtype=np.float32) for m in mosaics], 1.0

    ratio = float(np.mean(g0[valid]) / max(np.mean(g1[valid]), 1e-8))
    if abs(ratio - 1.0) <= min_diff or abs(ratio - 1.0) > max_diff:
        return [np.asarray(m, dtype=np.float32) for m in mosaics], ratio

    s0 = np.float32(1.0 / np.sqrt(ratio))
    s1 = np.float32(np.sqrt(ratio))
    out: List[np.ndarray] = []
    for m in mosaics:
        mc = np.asarray(m, dtype=np.float32).copy()
        mc[r0::2, c0::2] *= s0
        mc[r1::2, c1::2] *= s1
        out.append(mc)
    return out, ratio


# --------------------------------------------------------------------------- #
# Priors: BTV on luminance + Tikhonov/Laplacian on chroma (CPU)
# --------------------------------------------------------------------------- #

_BTV_OFFSETS = [
    (dl, dm, 0.6 ** (abs(dl) + abs(dm)))
    for dl in range(0, 3)
    for dm in range(-2, 3)
    if (dl > 0) or (dl == 0 and dm > 0)
]


def _btv_value_and_grad_cpu(Y: np.ndarray, eps: float = 1e-3) -> Tuple[float, np.ndarray]:
    """Smooth Bilateral Total Variation on 2D luminance Y."""
    h, w = Y.shape
    val = 0.0
    grad = np.zeros_like(Y, dtype=np.float32)
    eps2 = np.float32(eps * eps)
    for dy, dx, alpha in _BTV_OFFSETS:
        # Slice overlapping region where (y, x) and (y + dy, x + dx) are both in bounds
        y0_s = max(0, -dy)
        y1_s = h - max(0, dy)
        x0_s = max(0, -dx)
        x1_s = w - max(0, dx)
        a = Y[y0_s:y1_s, x0_s:x1_s]
        b = Y[y0_s + dy:y1_s + dy, x0_s + dx:x1_s + dx]
        diff = a - b
        norm = np.sqrt(diff * diff + eps2)
        val += float(alpha) * float(np.sum(norm - eps, dtype=np.float64))
        g = (np.float32(alpha) * diff) / norm
        grad[y0_s:y1_s, x0_s:x1_s] += g
        grad[y0_s + dy:y1_s + dy, x0_s + dx:x1_s + dx] -= g
    return val, grad


def _chroma_tikhonov_value_and_grad_cpu(C: np.ndarray) -> Tuple[float, np.ndarray]:
    """0.5 * ||grad(C)||_2^2 and its gradient (-Laplacian(C)) on a 2D chroma plane."""
    dx = C[:, 1:] - C[:, :-1]
    dy = C[1:, :] - C[:-1, :]
    val = 0.5 * (float(np.sum(dx * dx, dtype=np.float64)) + float(np.sum(dy * dy, dtype=np.float64)))
    grad = np.zeros_like(C, dtype=np.float32)
    grad[:, 1:] += dx
    grad[:, :-1] -= dx
    grad[1:, :] += dy
    grad[:-1, :] -= dy
    return val, grad


def _huber_tv_value_and_grad_cpu(Y: np.ndarray, eps: float = 1e-3) -> Tuple[float, np.ndarray]:
    """Smooth isotropic Huber/Charbonnier TV: sum(sqrt(gx^2 + gy^2 + eps^2) - eps)."""
    gx = np.zeros_like(Y, dtype=np.float32)
    gy = np.zeros_like(Y, dtype=np.float32)
    gx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    gy[:-1, :] = Y[1:, :] - Y[:-1, :]
    eps32 = np.float32(max(eps, 1e-6))
    norm = np.sqrt(gx * gx + gy * gy + eps32 * eps32)
    val = float(np.sum(norm - eps32, dtype=np.float64))
    inv_n = np.float32(1.0) / norm
    px = gx * inv_n
    py = gy * inv_n
    grad = np.zeros_like(Y, dtype=np.float32)
    grad[:, 1:] += px[:, :-1]
    grad[:, :-1] -= px[:, :-1]
    grad[1:, :] += py[:-1, :]
    grad[:-1, :] -= py[:-1, :]
    return val, grad


def _hyper_lap_value_and_grad_cpu(
    Y: np.ndarray,
    eps: float = 1e-3,
    p: float = 0.8,
) -> Tuple[float, np.ndarray]:
    """Smooth isotropic hyper-Laplacian prior: (1/p) * sum((gx^2 + gy^2 + eps^2)^(p/2) - eps^p)."""
    gx = np.zeros_like(Y, dtype=np.float32)
    gy = np.zeros_like(Y, dtype=np.float32)
    gx[:, :-1] = Y[:, 1:] - Y[:, :-1]
    gy[:-1, :] = Y[1:, :] - Y[:-1, :]
    eps2 = np.float32(max(eps, 1e-6) ** 2)
    g2 = gx * gx + gy * gy + eps2
    p32 = np.float32(p)
    half_p = np.float32(0.5 * p)
    val = float(np.sum(np.power(g2, half_p) - np.float32(eps2 ** (0.5 * p)), dtype=np.float64)) / float(p)
    w = np.power(g2, np.float32(0.5 * p - 1.0))
    px = gx * w
    py = gy * w
    grad = np.zeros_like(Y, dtype=np.float32)
    grad[:, 1:] += px[:, :-1]
    grad[:, :-1] -= px[:, :-1]
    grad[1:, :] += py[:-1, :]
    grad[:-1, :] -= py[:-1, :]
    return val, grad


def _edge_chroma_value_and_grad_cpu(
    C: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Edge-weighted quadratic chroma smoothness: 0.5 * sum(wx*dx^2 + wy*dy^2)."""
    dx = C[:, 1:] - C[:, :-1]
    dy = C[1:, :] - C[:-1, :]
    wdx = wx * dx
    wdy = wy * dy
    val = 0.5 * (float(np.sum(dx * wdx, dtype=np.float64)) + float(np.sum(dy * wdy, dtype=np.float64)))
    grad = np.zeros_like(C, dtype=np.float32)
    grad[:, 1:] += wdx
    grad[:, :-1] -= wdx
    grad[1:, :] += wdy
    grad[:-1, :] -= wdy
    return val, grad


def _nyquist_penalty_value_and_grad_cpu(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """Period-2 checkerboard (mixed derivative) regularizer: 0.5 * ||0.25*[1,-1;-1,1] * x||_2^2.

    Identically zero on constants, linear ramps, and all pure horizontal or vertical
    edges; maximal on period-2 Nyquist checkerboard modes (-1)^(i+j).
    """
    d = np.float32(0.25) * (x[:-1, :-1] - x[:-1, 1:] - x[1:, :-1] + x[1:, 1:])
    val = 0.5 * float(np.sum(d * d, dtype=np.float64))
    g = np.float32(0.25) * d
    grad = np.zeros_like(x, dtype=np.float32)
    grad[:-1, :-1] += g
    grad[:-1, 1:] -= g
    grad[1:, :-1] -= g
    grad[1:, 1:] += g
    return val, grad


# --------------------------------------------------------------------------- #
# CuPy GPU kernels (exact match to CPU forward/adjoint)
# --------------------------------------------------------------------------- #

_CFA_FWD_KERNEL = None
_CFA_ADJ_KERNEL = None


def _get_cfa_kernels(cp):
    global _CFA_FWD_KERNEL, _CFA_ADJ_KERNEL
    if _CFA_FWD_KERNEL is None:
        _CFA_FWD_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void cfa_warp_fwd(
            const float* __restrict__ z,
            int h_hr, int w_hr,
            const signed char* __restrict__ ch_map,
            int h_lr, int w_lr,
            double m00, double m01, double m02,
            double m10, double m11, double m12,
            double m20, double m21, double m22,
            float* __restrict__ out,
            unsigned char* __restrict__ valid
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= w_lr || y >= h_lr) return;
            int idx = y * w_lr + x;
            double denom = m20 * (double)x + m21 * (double)y + m22;
            if (fabs(denom) < 1e-12) {
                out[idx] = 0.0f;
                valid[idx] = 0;
                return;
            }
            double u = (m00 * (double)x + m01 * (double)y + m02) / denom;
            double v = (m10 * (double)x + m11 * (double)y + m12) / denom;
            int u0 = (int)floor(u);
            int v0 = (int)floor(v);
            if (u0 < 0 || u0 + 1 >= w_hr || v0 < 0 || v0 + 1 >= h_hr) {
                out[idx] = 0.0f;
                valid[idx] = 0;
                return;
            }
            float fu = (float)(u - (double)u0);
            float fv = (float)(v - (double)v0);
            int c = (int)ch_map[idx];
            int s = w_hr * 3;
            float v00 = z[v0 * s + u0 * 3 + c];
            float v10 = z[v0 * s + (u0 + 1) * 3 + c];
            float v01 = z[(v0 + 1) * s + u0 * 3 + c];
            float v11 = z[(v0 + 1) * s + (u0 + 1) * 3 + c];
            out[idx] = ((1.0f - fu) * (1.0f - fv) * v00
                      + fu * (1.0f - fv) * v10
                      + (1.0f - fu) * fv * v01
                      + fu * fv * v11);
            valid[idx] = 1;
        }
        ''', 'cfa_warp_fwd')
    if _CFA_ADJ_KERNEL is None:
        _CFA_ADJ_KERNEL = cp.RawKernel(r'''
        extern "C" __global__
        void cfa_warp_adj(
            const float* __restrict__ r,
            const unsigned char* __restrict__ valid,
            const signed char* __restrict__ ch_map,
            int h_lr, int w_lr,
            double m00, double m01, double m02,
            double m10, double m11, double m12,
            double m20, double m21, double m22,
            float* __restrict__ acc,
            int h_hr, int w_hr
        ) {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= w_lr || y >= h_lr) return;
            int idx = y * w_lr + x;
            if (!valid[idx]) return;
            float rv = r[idx];
            if (rv == 0.0f) return;
            double denom = m20 * (double)x + m21 * (double)y + m22;
            double u = (m00 * (double)x + m01 * (double)y + m02) / denom;
            double v = (m10 * (double)x + m11 * (double)y + m12) / denom;
            int u0 = (int)floor(u);
            int v0 = (int)floor(v);
            float fu = (float)(u - (double)u0);
            float fv = (float)(v - (double)v0);
            int c = (int)ch_map[idx];
            int s = w_hr * 3;
            atomicAdd(&acc[v0 * s + u0 * 3 + c], rv * (1.0f - fu) * (1.0f - fv));
            atomicAdd(&acc[v0 * s + (u0 + 1) * 3 + c], rv * fu * (1.0f - fv));
            atomicAdd(&acc[(v0 + 1) * s + u0 * 3 + c], rv * (1.0f - fu) * fv);
            atomicAdd(&acc[(v0 + 1) * s + (u0 + 1) * 3 + c], rv * fu * fv);
        }
        ''', 'cfa_warp_adj')
    return _CFA_FWD_KERNEL, _CFA_ADJ_KERNEL


def _blur_forward_gpu(cp, x_g, k1d_np: np.ndarray):
    """Exact replicate-boundary separable 2D blur on GPU."""
    r = len(k1d_np) // 2
    if r == 0:
        return x_g * float(k1d_np[0])
    # Along axis=1 (width)
    w = x_g.shape[1]
    pad_x = cp.pad(x_g, ((0, 0), (r, r), (0, 0)), mode="edge")
    tmp = cp.zeros_like(x_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt != 0.0:
            tmp += wt * pad_x[:, r + d:r + d + w, :]
    # Along axis=0 (height)
    h = x_g.shape[0]
    pad_y = cp.pad(tmp, ((r, r), (0, 0), (0, 0)), mode="edge")
    out = cp.zeros_like(x_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt != 0.0:
            out += wt * pad_y[r + d:r + d + h, :, :]
    return out


def _blur_adjoint_gpu(cp, y_g, k1d_np: np.ndarray):
    """Exact adjoint of :func:`_blur_forward_gpu` on GPU."""
    r = len(k1d_np) // 2
    if r == 0:
        return y_g * float(k1d_np[0])
    # Adjoint along axis=0 (height)
    h = y_g.shape[0]
    tmp = cp.zeros_like(y_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt == 0.0:
            continue
        if d == 0:
            tmp += wt * y_g
        elif d > 0:
            tmp[d:h, :, :] += wt * y_g[0:h - d, :, :]
            tmp[h - 1, :, :] += wt * cp.sum(y_g[h - d:h, :, :], axis=0)
        else:
            m = -d
            tmp[0:h - m, :, :] += wt * y_g[m:h, :, :]
            tmp[0, :, :] += wt * cp.sum(y_g[0:m, :, :], axis=0)
    # Adjoint along axis=1 (width)
    w = tmp.shape[1]
    out = cp.zeros_like(tmp)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt == 0.0:
            continue
        if d == 0:
            out += wt * tmp
        elif d > 0:
            out[:, d:w, :] += wt * tmp[:, 0:w - d, :]
            out[:, w - 1, :] += wt * cp.sum(tmp[:, w - d:w, :], axis=1)
        else:
            m = -d
            out[:, 0:w - m, :] += wt * tmp[:, m:w, :]
            out[:, 0, :] += wt * cp.sum(tmp[:, 0:m, :], axis=1)
    return out


def _blur_forward_2d_gpu(cp, plane_g, k1d_np: np.ndarray):
    r = len(k1d_np) // 2
    if r == 0:
        return plane_g * float(k1d_np[0])
    w = plane_g.shape[1]
    pad_x = cp.pad(plane_g, ((0, 0), (r, r)), mode="edge")
    tmp = cp.zeros_like(plane_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt != 0.0:
            tmp += wt * pad_x[:, r + d:r + d + w]
    h = plane_g.shape[0]
    pad_y = cp.pad(tmp, ((r, r), (0, 0)), mode="edge")
    out = cp.zeros_like(plane_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt != 0.0:
            out += wt * pad_y[r + d:r + d + h, :]
    return out


def _blur_adjoint_2d_gpu(cp, plane_g, k1d_np: np.ndarray):
    r = len(k1d_np) // 2
    if r == 0:
        return plane_g * float(k1d_np[0])
    h = plane_g.shape[0]
    tmp = cp.zeros_like(plane_g)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt == 0.0:
            continue
        if d == 0:
            tmp += wt * plane_g
        elif d > 0:
            tmp[d:h, :] += wt * plane_g[0:h - d, :]
            tmp[h - 1, :] += wt * cp.sum(plane_g[h - d:h, :], axis=0)
        else:
            m = -d
            tmp[0:h - m, :] += wt * plane_g[m:h, :]
            tmp[0, :] += wt * cp.sum(plane_g[0:m, :], axis=0)
    w = tmp.shape[1]
    out = cp.zeros_like(tmp)
    for d in range(-r, r + 1):
        wt = float(k1d_np[d + r])
        if wt == 0.0:
            continue
        if d == 0:
            out += wt * tmp
        elif d > 0:
            out[:, d:w] += wt * tmp[:, 0:w - d]
            out[:, w - 1] += wt * cp.sum(tmp[:, w - d:w], axis=1)
        else:
            m = -d
            out[:, 0:w - m] += wt * tmp[:, m:w]
            out[:, 0] += wt * cp.sum(tmp[:, 0:m], axis=1)
    return out


def _blur_forward_rgb_gpu(cp, x_g, k1d_rgb: Sequence[np.ndarray]):
    if _kernels_equal(k1d_rgb):
        return _blur_forward_gpu(cp, x_g, k1d_rgb[0])
    out = cp.empty_like(x_g)
    for c in range(3):
        out[..., c] = _blur_forward_2d_gpu(cp, x_g[..., c], k1d_rgb[c])
    return out


def _blur_adjoint_rgb_gpu(cp, y_g, k1d_rgb: Sequence[np.ndarray]):
    if _kernels_equal(k1d_rgb):
        return _blur_adjoint_gpu(cp, y_g, k1d_rgb[0])
    out = cp.empty_like(y_g)
    for c in range(3):
        out[..., c] = _blur_adjoint_2d_gpu(cp, y_g[..., c], k1d_rgb[c])
    return out


def _btv_value_and_grad_gpu(cp, Y_g, eps: float = 1e-3):
    h, w = Y_g.shape
    val = 0.0
    grad = cp.zeros_like(Y_g)
    eps2 = cp.float32(eps * eps)
    for dy, dx, alpha in _BTV_OFFSETS:
        y0_s = max(0, -dy)
        y1_s = h - max(0, dy)
        x0_s = max(0, -dx)
        x1_s = w - max(0, dx)
        a = Y_g[y0_s:y1_s, x0_s:x1_s]
        b = Y_g[y0_s + dy:y1_s + dy, x0_s + dx:x1_s + dx]
        diff = a - b
        norm = cp.sqrt(diff * diff + eps2)
        val += float(alpha) * float(cp.sum(norm - eps, dtype=cp.float64).get())
        g = (cp.float32(alpha) * diff) / norm
        grad[y0_s:y1_s, x0_s:x1_s] += g
        grad[y0_s + dy:y1_s + dy, x0_s + dx:x1_s + dx] -= g
    return val, grad


def _chroma_tikhonov_value_and_grad_gpu(cp, C_g):
    dx = C_g[:, 1:] - C_g[:, :-1]
    dy = C_g[1:, :] - C_g[:-1, :]
    val = 0.5 * (
        float(cp.sum(dx * dx, dtype=cp.float64).get())
        + float(cp.sum(dy * dy, dtype=cp.float64).get())
    )
    grad = cp.zeros_like(C_g)
    grad[:, 1:] += dx
    grad[:, :-1] -= dx
    grad[1:, :] += dy
    grad[:-1, :] -= dy
    return val, grad


def _huber_tv_value_and_grad_gpu(cp, Y_g, eps: float = 1e-3):
    gx = cp.zeros_like(Y_g)
    gy = cp.zeros_like(Y_g)
    gx[:, :-1] = Y_g[:, 1:] - Y_g[:, :-1]
    gy[:-1, :] = Y_g[1:, :] - Y_g[:-1, :]
    eps32 = cp.float32(max(eps, 1e-6))
    norm = cp.sqrt(gx * gx + gy * gy + eps32 * eps32)
    val = float(cp.sum(norm - eps32, dtype=cp.float64).get())
    inv_n = cp.float32(1.0) / norm
    px = gx * inv_n
    py = gy * inv_n
    grad = cp.zeros_like(Y_g)
    grad[:, 1:] += px[:, :-1]
    grad[:, :-1] -= px[:, :-1]
    grad[1:, :] += py[:-1, :]
    grad[:-1, :] -= py[:-1, :]
    return val, grad


def _hyper_lap_value_and_grad_gpu(cp, Y_g, eps: float = 1e-3, p: float = 0.8):
    gx = cp.zeros_like(Y_g)
    gy = cp.zeros_like(Y_g)
    gx[:, :-1] = Y_g[:, 1:] - Y_g[:, :-1]
    gy[:-1, :] = Y_g[1:, :] - Y_g[:-1, :]
    eps2 = cp.float32(max(eps, 1e-6) ** 2)
    g2 = gx * gx + gy * gy + eps2
    half_p = cp.float32(0.5 * p)
    val = float(cp.sum(cp.power(g2, half_p) - cp.float32(float(eps2) ** (0.5 * p)), dtype=cp.float64).get()) / float(p)
    w = cp.power(g2, cp.float32(0.5 * p - 1.0))
    px = gx * w
    py = gy * w
    grad = cp.zeros_like(Y_g)
    grad[:, 1:] += px[:, :-1]
    grad[:, :-1] -= px[:, :-1]
    grad[1:, :] += py[:-1, :]
    grad[:-1, :] -= py[:-1, :]
    return val, grad


def _edge_chroma_value_and_grad_gpu(cp, C_g, wx_g, wy_g):
    dx = C_g[:, 1:] - C_g[:, :-1]
    dy = C_g[1:, :] - C_g[:-1, :]
    wdx = wx_g * dx
    wdy = wy_g * dy
    val = 0.5 * (
        float(cp.sum(dx * wdx, dtype=cp.float64).get())
        + float(cp.sum(dy * wdy, dtype=cp.float64).get())
    )
    grad = cp.zeros_like(C_g)
    grad[:, 1:] += wdx
    grad[:, :-1] -= wdx
    grad[1:, :] += wdy
    grad[:-1, :] -= wdy
    return val, grad


def _nyquist_penalty_value_and_grad_gpu(cp, x_g):
    d = cp.float32(0.25) * (x_g[:-1, :-1] - x_g[:-1, 1:] - x_g[1:, :-1] + x_g[1:, 1:])
    val = 0.5 * float(cp.sum(d * d, dtype=cp.float64).get())
    g = cp.float32(0.25) * d
    grad = cp.zeros_like(x_g)
    grad[:-1, :-1] += g
    grad[:-1, 1:] -= g
    grad[1:, :-1] -= g
    grad[1:, 1:] += g
    return val, grad


# --------------------------------------------------------------------------- #
# Main MAP-SR solver
# --------------------------------------------------------------------------- #

def map_super_resolution(
    init_hr: np.ndarray,
    mosaics: Sequence[np.ndarray],
    transforms: Sequence[Any],
    ratio: float,
    pattern: Any,
    weights: Optional[Sequence[float]] = None,
    *,
    colour_matrix: Optional[np.ndarray] = None,
    equalize_green: bool = True,
    ref_index: int = 0,
    max_iter: int = 35,
    lens_sigma_lr: Optional[Union[float, Sequence[float]]] = None,
    estimate_psf: bool = False,
    btv_weight: float = 0.10,
    chroma_weight: float = 4.0,
    huber_k: float = 2.0,
    tol: float = 1e-4,
    use_gpu: Optional[bool] = None,
    mode: str = "legacy",
    prior: str = "huber_tv",
    luma_weight: float = 0.25,
    chroma_weight_wp13: float = 0.04,
    nyquist_weight: float = 0.06,
    huber_kappa: float = 3.0,
    drizzle_bias_lr: float = 0.10,
    discrepancy_adapt: bool = True,
    lbfgs_m: int = 5,
    ref_rgb: Optional[np.ndarray] = None,
    env_tol: Optional[float] = 0.08,
    env_radius: int = 2,
) -> MAPSRResult:
    """Solve joint demosaic + SR + deconvolution MAP inverse problem.

    Parameters
    ----------
    init_hr : (H_hr, W_hr, 3) float32
        Initial HR RGB estimate (from ``demosaic_drizzle``).
    mosaics : sequence of (H_lr, W_lr) float32
        Calibrated Bayer mosaics for the fused frames.
    transforms : sequence of 3x3 homographies
        ``ref -> frame`` transforms from ``SuperResolution``.
    ratio : float
        Upscaling factor (e.g. 2.0, 3.0).
    pattern : 2x2 array or str
        Bayer CFA pattern.
    weights : optional sequence of float
        Per-frame quality weights (default uniform).
    max_iter : int
        Maximum solver iterations (default 35 for legacy, 70 for wp13 if <= 35).
    lens_sigma_lr : optional float or 3-tuple of float
        Lens Gaussian PSF sigma in LR pixels (scalar or per-channel R, G, B).
        ``None`` uses the default sensor aperture model in legacy mode, or
        auto-estimates per-channel PSF in ``mode='wp13'``.
    estimate_psf : bool
        If True and ``lens_sigma_lr is None``, estimate lens sigma from
        ``init_hr`` via WP6 ``estimate_psf_sigma_edges``.
    btv_weight : float
        Dimensionless BTV prior weight in legacy mode.
    chroma_weight : float
        Dimensionless chroma Tikhonov prior weight in legacy mode.
    huber_k : float
        Huber transition threshold in units of measured noise sigma (legacy).
    tol : float
        Relative step-size stopping tolerance.
    use_gpu : None | True | False
        CuPy GPU acceleration control (falls back to CPU transparently).
    mode : str
        ``'legacy'`` (WP8/WP11 default) or ``'wp13'`` (per-channel PSF +
        Poisson-Gaussian noise model + Coverage-Preconditioned Projected L-BFGS
        + hyper-Laplacian/Huber-TV luma + edge-weighted chroma + Nyquist prior).
    """
    t0 = time.perf_counter()
    x0 = np.clip(np.asarray(init_hr, dtype=np.float32), 0.0, 1.0)
    hr_shape = (int(x0.shape[0]), int(x0.shape[1]))
    lr_shape = (int(mosaics[0].shape[0]), int(mosaics[0].shape[1]))
    n_frames = len(mosaics)

    if weights is None:
        w_arr = np.ones(n_frames, dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        w_mean = float(np.mean(w_arr)) if w_arr.size else 1.0
        if w_mean > 0:
            w_arr = w_arr / w_mean

    if colour_matrix is None:
        C_mat = np.eye(3, dtype=np.float64)
    else:
        C_mat = np.asarray(colour_matrix, dtype=np.float64)
        if C_mat.shape != (3, 3):
            raise ValueError(f"colour_matrix must be (3, 3), got {C_mat.shape}")

    if equalize_green:
        mosaics_use, gr_gb_ratio = equalize_green_imbalance(mosaics, pattern, ref_index=ref_index)
    else:
        mosaics_use = [np.asarray(m, dtype=np.float32) for m in mosaics]
        gr_gb_ratio = 1.0

    # WP13/WP15 mode dispatch
    if mode == "wp13":
        psf_meta: Dict[str, Any] = {}
        if lens_sigma_lr is None:
            psf_rgb, psf_meta = estimate_burst_psf_rgb(
                x0, ratio=float(ratio), ref_rgb=ref_rgb, drizzle_bias_lr=drizzle_bias_lr,
            )
            psf_source = psf_meta["source"]
        elif isinstance(lens_sigma_lr, (int, float, np.floating, np.integer)):
            s = float(lens_sigma_lr)
            psf_rgb = (s, s, s)
            psf_source = "explicit_scalar"
        else:
            psf_rgb = tuple(float(v) for v in lens_sigma_lr)
            psf_source = "explicit_rgb"

        k1d_rgb = build_psf_kernels_rgb(ratio, lens_sigma_lr=psf_rgb, compensate_warp=True)
        wp13_max_iter = max(int(max_iter), 70) if max_iter == 35 else int(max_iter)

        want_gpu = False
        if use_gpu is not False:
            try:
                from .gpu import gpu_available, gpu_fits_vram
                if gpu_available():
                    hr_bytes = hr_shape[0] * hr_shape[1] * 3 * 4
                    lr_bytes = lr_shape[0] * lr_shape[1] * 4
                    req_bytes = hr_bytes * (10 + 2 * lbfgs_m) + lr_bytes * (2 * n_frames + 4)
                    if gpu_fits_vram(req_bytes):
                        want_gpu = True
            except Exception:
                want_gpu = False

        if want_gpu:
            try:
                return _solve_map_wp13_gpu(
                    x0, mosaics_use, transforms, ratio, pattern, w_arr, k1d_rgb, C_mat,
                    ref_index=ref_index, prior=prior,
                    luma_weight=luma_weight, chroma_weight=chroma_weight_wp13,
                    nyquist_weight=nyquist_weight, huber_kappa=huber_kappa,
                    discrepancy_adapt=discrepancy_adapt, lbfgs_m=lbfgs_m,
                    max_iter=wp13_max_iter, tol=tol,
                    psf_source=psf_source, psf_sigma_rgb=psf_rgb,
                    psf_meta=psf_meta, env_tol=env_tol, env_radius=env_radius,
                    gr_gb_ratio=gr_gb_ratio, t0=t0,
                )
            except Exception:
                try:
                    from .gpu import free_gpu_memory
                    free_gpu_memory()
                except Exception:
                    pass

        return _solve_map_wp13_cpu(
            x0, mosaics_use, transforms, ratio, pattern, w_arr, k1d_rgb, C_mat,
            ref_index=ref_index, prior=prior,
            luma_weight=luma_weight, chroma_weight=chroma_weight_wp13,
            nyquist_weight=nyquist_weight, huber_kappa=huber_kappa,
            discrepancy_adapt=discrepancy_adapt, lbfgs_m=lbfgs_m,
            max_iter=wp13_max_iter, tol=tol,
            psf_source=psf_source, psf_sigma_rgb=psf_rgb,
            psf_meta=psf_meta, env_tol=env_tol, env_radius=env_radius,
            gr_gb_ratio=gr_gb_ratio, t0=t0,
        )

    # Legacy mode (WP8/WP11): fix the estimate_psf attribute-access bug!
    psf_source = "default_sensor"
    if estimate_psf and lens_sigma_lr is None:
        try:
            from .enhance import estimate_psf_sigma_edges
            psf_info = estimate_psf_sigma_edges(x0, ratio=float(ratio))
            if not psf_info.used_fallback:
                lens_sigma_lr = float(np.clip(psf_info.sigma_lr, 0.15, 1.2))
                psf_source = "wp6_edges"
        except Exception:
            pass
    elif lens_sigma_lr is not None:
        psf_source = "explicit_box_gauss"

    scalar_sigma = (
        float(lens_sigma_lr[1]) if isinstance(lens_sigma_lr, (tuple, list)) else lens_sigma_lr
    )
    k1d = build_psf_kernel_1d(ratio, lens_sigma_lr=scalar_sigma)

    # Estimate noise sigma from init_hr luminance (Immerkaer)
    y_init = 0.299 * x0[..., 0] + 0.587 * x0[..., 1] + 0.114 * x0[..., 2]
    sigma_n = max(immerkaer_sigma(y_init), 5e-4)
    delta = max(float(huber_k) * sigma_n, 1e-3)

    # Scale regularization weights to match per-HR-pixel data term density
    # Data term has ~ (sum(w_k) / ratio^2) LR samples per HR pixel; Huber slope is 1/delta
    n_eff = float(np.sum(w_arr)) / max(float(ratio) ** 2, 1.0)
    lam_btv = float(btv_weight) * n_eff * (sigma_n / delta)
    lam_chroma = float(chroma_weight) * n_eff / max(delta, 1e-3) * sigma_n

    # Try GPU path if requested / available
    want_gpu = False
    if use_gpu is not False:
        try:
            from .gpu import gpu_available, gpu_fits_vram, is_oom_error, free_gpu_memory, _import_cupy
            if gpu_available():
                hr_bytes = hr_shape[0] * hr_shape[1] * 3 * 4
                lr_bytes = lr_shape[0] * lr_shape[1] * 4
                req_bytes = hr_bytes * 10 + lr_bytes * (n_frames + 3)
                if gpu_fits_vram(req_bytes):
                    want_gpu = True
        except Exception:
            want_gpu = False

    if want_gpu:
        try:
            return _solve_map_gpu(
                x0, mosaics_use, transforms, ratio, pattern, w_arr, k1d, C_mat,
                delta=delta, lam_btv=lam_btv, lam_chroma=lam_chroma,
                sigma_n=sigma_n, max_iter=max_iter, tol=tol,
                psf_source=psf_source, lens_sigma_lr=lens_sigma_lr,
                gr_gb_ratio=gr_gb_ratio, t0=t0,
            )
        except Exception as exc:
            try:
                from .gpu import free_gpu_memory
                free_gpu_memory()
            except Exception:
                pass
            if use_gpu is True:
                # Still fall back gracefully to CPU
                pass

    return _solve_map_cpu(
        x0, mosaics_use, transforms, ratio, pattern, w_arr, k1d, C_mat,
        delta=delta, lam_btv=lam_btv, lam_chroma=lam_chroma,
        sigma_n=sigma_n, max_iter=max_iter, tol=tol,
        psf_source=psf_source, lens_sigma_lr=lens_sigma_lr,
        gr_gb_ratio=gr_gb_ratio, t0=t0,
    )


def _solve_map_cpu(
    x0: np.ndarray,
    mosaics: Sequence[np.ndarray],
    transforms: Sequence[Any],
    ratio: float,
    pattern: Any,
    w_arr: np.ndarray,
    k1d: np.ndarray,
    colour_matrix: np.ndarray,
    *,
    delta: float,
    lam_btv: float,
    lam_chroma: float,
    sigma_n: float,
    max_iter: int,
    tol: float,
    psf_source: str,
    lens_sigma_lr: Optional[float],
    gr_gb_ratio: float,
    t0: float,
) -> MAPSRResult:
    hr_shape = (int(x0.shape[0]), int(x0.shape[1]))
    lr_shape = (int(mosaics[0].shape[0]), int(mosaics[0].shape[1]))
    fm = MAPForwardModel(
        transforms, ratio, lr_shape, hr_shape, pattern,
        lens_sigma_lr=lens_sigma_lr, colour_matrix=colour_matrix,
    )
    fm.k1d = k1d
    C32 = colour_matrix.astype(np.float32)
    CT32 = np.ascontiguousarray(C32.T)
    y_list = [np.asarray(m, dtype=np.float32) for m in mosaics]
    total_valid = sum(int(c.valid_flat.size) for c in fm.caches)

    def eval_obj_grad(x: np.ndarray) -> Tuple[float, np.ndarray, float]:
        z = blur_forward_cpu(x, k1d)
        z_cam = np.matmul(z, CT32)
        data_val = 0.0
        sq_res = 0.0
        acc_z = np.zeros_like(x)
        inv_delta = np.float32(1.0 / delta)
        half_delta = np.float32(0.5 * delta)
        for yk, wk, cache in zip(y_list, w_arr, fm.caches):
            pred = cfa_warp_forward_cpu(z_cam, cache, lr_shape)
            r = pred - yk
            r = np.where(cache.valid, r, np.float32(0.0))
            ar = np.abs(r)
            quad = ar <= delta
            rho = np.where(quad, np.float32(0.5) * inv_delta * r * r, ar - half_delta)
            data_val += float(wk) * float(np.sum(rho, dtype=np.float64))
            sq_res += float(np.sum(r * r, dtype=np.float64))
            psi = np.clip(r * inv_delta, -1.0, 1.0) * np.float32(wk)
            acc_z += cfa_warp_adjoint_cpu(psi, cache, hr_shape)
        acc_rgb = np.matmul(acc_z, C32)
        grad = blur_adjoint_cpu(acc_rgb, k1d)

        # Priors: BTV(Y) + Tikhonov(R-G, B-G)
        Y = np.float32(0.299) * x[..., 0] + np.float32(0.587) * x[..., 1] + np.float32(0.114) * x[..., 2]
        Cr = x[..., 0] - x[..., 1]
        Cb = x[..., 2] - x[..., 1]

        btv_val, gY = _btv_value_and_grad_cpu(Y)
        cr_val, gCr = _chroma_tikhonov_value_and_grad_cpu(Cr)
        cb_val, gCb = _chroma_tikhonov_value_and_grad_cpu(Cb)

        lb = np.float32(lam_btv)
        lc = np.float32(lam_chroma)
        grad[..., 0] += lb * np.float32(0.299) * gY + lc * gCr
        grad[..., 1] += lb * np.float32(0.587) * gY - lc * (gCr + gCb)
        grad[..., 2] += lb * np.float32(0.114) * gY + lc * gCb

        total_obj = data_val + lam_btv * btv_val + lam_chroma * (cr_val + cb_val)
        rms_res = float(np.sqrt(sq_res / max(total_valid, 1)))
        return total_obj, grad, rms_res

    x = x0.copy()
    obj, grad, rms0 = eval_obj_grad(x)
    obj0 = obj
    step = float(delta / max(float(np.sum(w_arr)) / max(ratio * ratio, 1.0), 0.25))
    iters_run = 0
    stop_reason = "max_iter"
    rms = rms0

    for it in range(max_iter):
        iters_run = it + 1
        # Armijo backtracking line search
        g_norm2 = float(np.sum(grad.astype(np.float64) ** 2))
        if g_norm2 < 1e-18:
            stop_reason = "zero_grad"
            break
        accepted = False
        cur_step = step
        for _ in range(8):
            x_new = np.clip(x - np.float32(cur_step) * grad, 0.0, 1.0)
            obj_new, grad_new, rms_new = eval_obj_grad(x_new)
            if obj_new <= obj + 1e-12:
                accepted = True
                break
            cur_step *= 0.4
        if not accepted:
            stop_reason = "line_search_stop"
            break

        s = (x_new - x).astype(np.float64)
        y_diff = (grad_new - grad).astype(np.float64)
        s_norm = float(np.sqrt(np.sum(s * s)))
        x_norm = float(np.sqrt(np.sum(x.astype(np.float64) ** 2))) + 1e-8
        sy = float(np.sum(s * y_diff))
        if sy > 1e-12:
            step = float(np.clip((s_norm * s_norm) / sy, 1e-4, 5.0))
        else:
            step = cur_step

        x, obj, grad, rms = x_new, obj_new, grad_new, rms_new

        if s_norm / x_norm < tol:
            stop_reason = "rel_step_tol"
            break
        if it >= 5 and rms < 0.95 * sigma_n:
            stop_reason = "morozov_discrepancy"
            break

    elapsed = float(time.perf_counter() - t0)
    return MAPSRResult(
        hr=x.astype(np.float32),
        metrics={
            "engine": "cpu",
            "iterations": iters_run,
            "stop_reason": stop_reason,
            "obj_init": round(obj0, 4),
            "obj_final": round(obj, 4),
            "rms_init": round(rms0, 6),
            "rms_final": round(rms, 6),
            "sigma_n": round(sigma_n, 6),
            "delta": round(delta, 6),
            "lam_btv": round(lam_btv, 6),
            "lam_chroma": round(lam_chroma, 6),
            "psf_source": psf_source,
            "gr_gb_ratio": round(float(gr_gb_ratio), 5),
            "seconds": round(elapsed, 3),
        },
    )


def _solve_map_gpu(
    x0: np.ndarray,
    mosaics: Sequence[np.ndarray],
    transforms: Sequence[Any],
    ratio: float,
    pattern: Any,
    w_arr: np.ndarray,
    k1d: np.ndarray,
    colour_matrix: np.ndarray,
    *,
    delta: float,
    lam_btv: float,
    lam_chroma: float,
    sigma_n: float,
    max_iter: int,
    tol: float,
    psf_source: str,
    lens_sigma_lr: Optional[float],
    gr_gb_ratio: float,
    t0: float,
) -> MAPSRResult:
    from .gpu import _import_cupy
    cp, _ = _import_cupy()
    fwd_kern, adj_kern = _get_cfa_kernels(cp)

    hr_shape = (int(x0.shape[0]), int(x0.shape[1]))
    lr_shape = (int(mosaics[0].shape[0]), int(mosaics[0].shape[1]))
    h_hr, w_hr = hr_shape
    h_lr, w_lr = lr_shape

    C_g = cp.asarray(colour_matrix, dtype=cp.float32)
    CT_g = cp.ascontiguousarray(C_g.T)
    ch_map_np = channel_map(pattern_array(pattern), lr_shape)
    ch_map_g = cp.asarray(ch_map_np, dtype=cp.int8)
    y_list_g = [cp.asarray(m, dtype=cp.float32) for m in mosaics]
    matrices = [to_drizzle_matrix(W, ratio).astype(np.float64) for W in transforms]

    block = (16, 16)
    grid = ((w_lr + block[0] - 1) // block[0], (h_lr + block[1] - 1) // block[1])
    pred_buf = cp.empty((h_lr, w_lr), dtype=cp.float32)
    valid_buf = cp.empty((h_lr, w_lr), dtype=cp.uint8)

    def eval_obj_grad_gpu(x_g):
        z_g = _blur_forward_gpu(cp, x_g, k1d)
        z_cam_g = cp.ascontiguousarray(cp.matmul(z_g, CT_g))
        data_val = 0.0
        sq_res = 0.0
        total_valid = 0
        acc_z = cp.zeros_like(x_g)
        inv_delta = cp.float32(1.0 / delta)
        half_delta = cp.float32(0.5 * delta)
        for yk_g, wk, M in zip(y_list_g, w_arr, matrices):
            fwd_kern(
                grid, block,
                (z_cam_g, h_hr, w_hr, ch_map_g, h_lr, w_lr,
                 float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                 float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
                 float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
                 pred_buf, valid_buf),
            )
            r = cp.where(valid_buf != 0, pred_buf - yk_g, cp.float32(0.0))
            ar = cp.abs(r)
            rho = cp.where(ar <= delta, cp.float32(0.5) * inv_delta * r * r, ar - half_delta)
            data_val += float(wk) * float(cp.sum(rho, dtype=cp.float64).get())
            sq_res += float(cp.sum(r * r, dtype=cp.float64).get())
            total_valid += int(cp.sum(valid_buf, dtype=cp.int64).get())
            psi = cp.clip(r * inv_delta, -1.0, 1.0) * cp.float32(wk)
            adj_kern(
                grid, block,
                (psi, valid_buf, ch_map_g, h_lr, w_lr,
                 float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                 float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
                 float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
                 acc_z, h_hr, w_hr),
            )
        acc_rgb_g = cp.matmul(acc_z, C_g)
        grad_g = _blur_adjoint_gpu(cp, acc_rgb_g, k1d)

        Y_g = cp.float32(0.299) * x_g[..., 0] + cp.float32(0.587) * x_g[..., 1] + cp.float32(0.114) * x_g[..., 2]
        Cr_g = x_g[..., 0] - x_g[..., 1]
        Cb_g = x_g[..., 2] - x_g[..., 1]

        btv_val, gY = _btv_value_and_grad_gpu(cp, Y_g)
        cr_val, gCr = _chroma_tikhonov_value_and_grad_gpu(cp, Cr_g)
        cb_val, gCb = _chroma_tikhonov_value_and_grad_gpu(cp, Cb_g)

        lb = cp.float32(lam_btv)
        lc = cp.float32(lam_chroma)
        grad_g[..., 0] += lb * cp.float32(0.299) * gY + lc * gCr
        grad_g[..., 1] += lb * cp.float32(0.587) * gY - lc * (gCr + gCb)
        grad_g[..., 2] += lb * cp.float32(0.114) * gY + lc * gCb

        total_obj = data_val + lam_btv * btv_val + lam_chroma * (cr_val + cb_val)
        rms_res = float(np.sqrt(sq_res / max(total_valid, 1)))
        return total_obj, grad_g, rms_res

    x_g = cp.asarray(x0, dtype=cp.float32)
    obj, grad_g, rms0 = eval_obj_grad_gpu(x_g)
    obj0 = obj
    step = float(delta / max(float(np.sum(w_arr)) / max(ratio * ratio, 1.0), 0.25))
    iters_run = 0
    stop_reason = "max_iter"
    rms = rms0

    for it in range(max_iter):
        iters_run = it + 1
        g_norm2 = float(cp.sum(grad_g.astype(cp.float64) ** 2).get())
        if g_norm2 < 1e-18:
            stop_reason = "zero_grad"
            break
        accepted = False
        cur_step = step
        for _ in range(8):
            x_new = cp.clip(x_g - cp.float32(cur_step) * grad_g, 0.0, 1.0)
            obj_new, grad_new, rms_new = eval_obj_grad_gpu(x_new)
            if obj_new <= obj + 1e-12:
                accepted = True
                break
            cur_step *= 0.4
        if not accepted:
            stop_reason = "line_search_stop"
            break

        s = x_new - x_g
        y_diff = grad_new - grad_g
        s_norm2 = float(cp.sum(s.astype(cp.float64) ** 2).get())
        x_norm = float(np.sqrt(float(cp.sum(x_g.astype(cp.float64) ** 2).get()))) + 1e-8
        sy = float(cp.sum(s.astype(cp.float64) * y_diff.astype(cp.float64)).get())
        if sy > 1e-12:
            step = float(np.clip(s_norm2 / sy, 1e-4, 5.0))
        else:
            step = cur_step

        x_g, obj, grad_g, rms = x_new, obj_new, grad_new, rms_new

        if np.sqrt(s_norm2) / x_norm < tol:
            stop_reason = "rel_step_tol"
            break
        if it >= 5 and rms < 0.95 * sigma_n:
            stop_reason = "morozov_discrepancy"
            break

    hr_out = cp.asnumpy(x_g).astype(np.float32)
    elapsed = float(time.perf_counter() - t0)
    return MAPSRResult(
        hr=hr_out,
        metrics={
            "engine": "cupy",
            "iterations": iters_run,
            "stop_reason": stop_reason,
            "obj_init": round(obj0, 4),
            "obj_final": round(obj, 4),
            "rms_init": round(rms0, 6),
            "rms_final": round(rms, 6),
            "sigma_n": round(sigma_n, 6),
            "delta": round(delta, 6),
            "lam_btv": round(lam_btv, 6),
            "lam_chroma": round(lam_chroma, 6),
            "psf_source": psf_source,
            "gr_gb_ratio": round(float(gr_gb_ratio), 5),
            "seconds": round(elapsed, 3),
        },
    )


# --------------------------------------------------------------------------- #
# WP13: Noise-aware, per-channel PSF, Coverage-Preconditioned Projected L-BFGS
# --------------------------------------------------------------------------- #

def _wp13_setup_weights_and_priors(
    x0: np.ndarray,
    mosaics: Sequence[np.ndarray],
    pattern: Any,
    w_arr: np.ndarray,
    ratio: float,
    ref_index: int,
    prior: str,
    luma_weight: float,
    chroma_weight: float,
    nyquist_weight: float,
) -> Dict[str, Any]:
    """Compute Poisson-Gaussian noise model, edge-chroma weights, and dimensionless lambdas."""
    a_var, b_gain, sigma_med = estimate_poisson_gaussian_noise(mosaics[ref_index], pattern)

    # Edge-preserving weights for chroma smoothness computed once from init_hr luma
    Y0 = (
        np.float32(0.2126) * x0[..., 0]
        + np.float32(0.7152) * x0[..., 1]
        + np.float32(0.0722) * x0[..., 2]
    )
    tau = np.float32(max(3.0 * sigma_med, 0.015))
    inv_tau2 = np.float32(1.0 / (float(tau) * float(tau)))
    dy0_x = Y0[:, 1:] - Y0[:, :-1]
    dy0_y = Y0[1:, :] - Y0[:-1, :]
    wx = (np.float32(1.0) / (np.float32(1.0) + dy0_x * dy0_x * inv_tau2)).astype(np.float32)
    wy = (np.float32(1.0) / (np.float32(1.0) + dy0_y * dy0_y * inv_tau2)).astype(np.float32)

    n_eff = float(np.sum(w_arr)) / max(float(ratio) ** 2, 1.0)
    eps_y = max(0.5 * sigma_med, 5e-4)
    p_hl = 0.8
    if prior == "hyper_lap":
        scale_p = 0.02 ** (1.0 - p_hl)
        lam_luma = float(luma_weight) * scale_p * np.sqrt(max(n_eff, 0.25)) / sigma_med
    else:
        lam_luma = float(luma_weight) * np.sqrt(max(n_eff, 0.25)) / sigma_med
    lam_chroma = float(chroma_weight) * n_eff / (sigma_med * sigma_med)
    lam_nyq = float(nyquist_weight) * n_eff / (sigma_med * sigma_med)

    return {
        "a_var": a_var,
        "b_gain": b_gain,
        "sigma_med": sigma_med,
        "wx": wx,
        "wy": wy,
        "eps_y": eps_y,
        "p_hl": p_hl,
        "lam_luma": lam_luma,
        "lam_chroma": lam_chroma,
        "lam_nyq": lam_nyq,
    }


def _refine_robust_frame_weights(
    med_abs_u: Sequence[float],
    w_arr: np.ndarray,
    ref_index: int,
) -> np.ndarray:
    """Lorentzian per-frame robust weight update based on initial standardized residuals."""
    if len(med_abs_u) <= 2:
        return w_arr.copy()
    med_burst = float(np.median(med_abs_u))
    if med_burst <= 1e-6:
        return w_arr.copy()
    w_out = w_arr.copy()
    for k, mk in enumerate(med_abs_u):
        if k == ref_index:
            continue
        rk = float(mk) / med_burst
        if rk > 1.25:
            w_out[k] *= 1.0 / (1.0 + (rk - 1.25) ** 2)
    w_mean = float(np.mean(w_out))
    if w_mean > 1e-8:
        w_out = w_out / w_mean
    return w_out


def _solve_map_wp13_cpu(
    x0: np.ndarray,
    mosaics: Sequence[np.ndarray],
    transforms: Sequence[Any],
    ratio: float,
    pattern: Any,
    w_arr: np.ndarray,
    k1d_rgb: Tuple[np.ndarray, np.ndarray, np.ndarray],
    colour_matrix: np.ndarray,
    *,
    ref_index: int,
    prior: str,
    luma_weight: float,
    chroma_weight: float,
    nyquist_weight: float,
    huber_kappa: float,
    discrepancy_adapt: bool,
    lbfgs_m: int,
    max_iter: int,
    tol: float,
    psf_source: str,
    psf_sigma_rgb: Tuple[float, float, float],
    psf_meta: Optional[Dict[str, Any]] = None,
    env_tol: Optional[float] = 0.08,
    env_radius: int = 2,
    gr_gb_ratio: float,
    t0: float,
) -> MAPSRResult:
    hr_shape = (int(x0.shape[0]), int(x0.shape[1]))
    lr_shape = (int(mosaics[0].shape[0]), int(mosaics[0].shape[1]))
    fm = MAPForwardModel(
        transforms, ratio, lr_shape, hr_shape, pattern,
        lens_sigma_lr=psf_sigma_rgb, colour_matrix=colour_matrix,
    )
    fm.k1d_rgb = k1d_rgb
    C32 = colour_matrix.astype(np.float32)
    CT32 = np.ascontiguousarray(C32.T)
    y_list = [np.asarray(m, dtype=np.float32) for m in mosaics]
    total_valid = sum(int(c.valid_flat.size) for c in fm.caches)

    cfg = _wp13_setup_weights_and_priors(
        x0, y_list, pattern, w_arr, ratio, ref_index,
        prior, luma_weight, chroma_weight, nyquist_weight,
    )
    a_var, b_gain, sigma_med = cfg["a_var"], cfg["b_gain"], cfg["sigma_med"]
    wx, wy = cfg["wx"], cfg["wy"]
    eps_y, p_hl = cfg["eps_y"], cfg["p_hl"]
    lam_luma = cfg["lam_luma"]
    lam_chroma = cfg["lam_chroma"]
    lam_nyq = cfg["lam_nyq"]
    kappa = np.float32(max(huber_kappa, 0.5))
    inv_kappa2 = np.float32(1.0 / (float(kappa) * float(kappa)))
    kappa2 = np.float32(float(kappa) * float(kappa))

    # Compute unbiased Poisson-Gaussian variance from expected signal mu_k = A_k(x0)
    # (never from noisy y_k, which biases dark pixels upward in weight -> darker mean!)
    sig_lo = np.float32(max(0.75 * sigma_med, 1e-4))
    sig_hi = np.float32(max(1.50 * sigma_med, 2e-4))
    z0 = blur_forward_rgb_cpu(x0, k1d_rgb)
    z0_cam = np.matmul(z0, CT32)
    inv_sig_list: List[np.ndarray] = []
    inv_var_list: List[np.ndarray] = []
    med_u: List[float] = []
    acc_mw = np.zeros_like(x0, dtype=np.float32)
    acc_mu = np.zeros_like(x0, dtype=np.float32)
    acc_mu2 = np.zeros_like(x0, dtype=np.float32)
    for yk, cache in zip(y_list, fm.caches):
        pred = cfa_warp_forward_cpu(z0_cam, cache, lr_shape)
        var = np.float32(a_var) + np.float32(b_gain) * np.clip(pred, np.float32(0.0), np.float32(1.0))
        sig_k = np.clip(np.sqrt(np.maximum(var, np.float32(1e-8))), sig_lo, sig_hi)
        isig = (np.float32(1.0) / sig_k).astype(np.float32)
        inv_sig_list.append(isig)
        inv_var_list.append((isig * isig).astype(np.float32))
        u_map = np.where(cache.valid, (pred - yk) * isig, np.float32(0.0))
        v_map = np.where(cache.valid, np.float32(1.0), np.float32(0.0))
        acc_mw += cfa_warp_adjoint_cpu(v_map, cache, hr_shape)
        acc_mu += cfa_warp_adjoint_cpu(u_map, cache, hr_shape)
        acc_mu2 += cfa_warp_adjoint_cpu(u_map * u_map, cache, hr_shape)
        if cache.valid_flat.size > 0:
            u_v = np.abs(u_map.ravel()[cache.valid_flat])
            med_u.append(float(np.median(u_v)))
        else:
            med_u.append(1.0)
    w_use = _refine_robust_frame_weights(med_u, w_arr, ref_index)
    sum_mw = np.maximum(np.sum(acc_mw, axis=2), np.float32(1e-6))
    mean_u = np.sum(acc_mu, axis=2) / sum_mw
    mean_u2 = np.sum(acc_mu2, axis=2) / sum_mw
    mismatch_hr = np.sqrt(np.maximum(mean_u2 - mean_u * mean_u, np.float32(0.0)))

    # Diagonal coverage-weighted preconditioner P^{-1} = 1 / (d_data + d_prior)
    acc_d = np.zeros_like(x0, dtype=np.float32)
    for wk, ivar, cache in zip(w_use, inv_var_list, fm.caches):
        acc_d += cfa_warp_adjoint_cpu(np.where(cache.valid, ivar * np.float32(wk), np.float32(0.0)), cache, hr_shape)
    d_rgb = np.matmul(acc_d, np.abs(C32))
    d_data = blur_adjoint_rgb_cpu(d_rgb, k1d_rgb)

    def make_inv_diag(ll: float) -> np.ndarray:
        d_prior = np.float32(ll / max(eps_y, 1e-4) + 4.0 * lam_chroma + 0.25 * lam_nyq)
        return (np.float32(1.0) / np.maximum(d_data + d_prior, np.float32(1.0))).astype(np.float32)

    inv_diag = make_inv_diag(lam_luma)

    def eval_obj_grad(x: np.ndarray, ll: float) -> Tuple[float, np.ndarray, float]:
        z = blur_forward_rgb_cpu(x, k1d_rgb)
        z_cam = np.matmul(z, CT32)
        data_val = 0.0
        sq_u = 0.0
        acc_z = np.zeros_like(x)
        for yk, wk, isig, cache in zip(y_list, w_use, inv_sig_list, fm.caches):
            pred = cfa_warp_forward_cpu(z_cam, cache, lr_shape)
            u = np.where(cache.valid, (pred - yk) * isig, np.float32(0.0))
            u2 = u * u
            sq_u += float(np.sum(u2, dtype=np.float64))
            rt = np.sqrt(np.float32(1.0) + u2 * inv_kappa2)
            rho = kappa2 * (rt - np.float32(1.0))
            data_val += float(wk) * float(np.sum(rho, dtype=np.float64))
            psi = (u / rt) * isig * np.float32(wk)
            acc_z += cfa_warp_adjoint_cpu(psi, cache, hr_shape)
        acc_rgb = np.matmul(acc_z, C32)
        grad = blur_adjoint_rgb_cpu(acc_rgb, k1d_rgb)

        # Luminance prior
        Y = (
            np.float32(0.2126) * x[..., 0]
            + np.float32(0.7152) * x[..., 1]
            + np.float32(0.0722) * x[..., 2]
        )
        if prior == "btv":
            luma_val, gY = _btv_value_and_grad_cpu(Y, eps=eps_y)
        elif prior == "huber_tv":
            luma_val, gY = _huber_tv_value_and_grad_cpu(Y, eps=eps_y)
        else:
            luma_val, gY = _hyper_lap_value_and_grad_cpu(Y, eps=eps_y, p=p_hl)

        # Edge-weighted chroma smoothness (DC-invariant)
        Cr = x[..., 0] - x[..., 1]
        Cb = x[..., 2] - x[..., 1]
        cr_val, gCr = _edge_chroma_value_and_grad_cpu(Cr, wx, wy)
        cb_val, gCb = _edge_chroma_value_and_grad_cpu(Cb, wx, wy)

        ll32 = np.float32(ll)
        lc32 = np.float32(lam_chroma)
        grad[..., 0] += ll32 * np.float32(0.2126) * gY + lc32 * gCr
        grad[..., 1] += ll32 * np.float32(0.7152) * gY - lc32 * (gCr + gCb)
        grad[..., 2] += ll32 * np.float32(0.0722) * gY + lc32 * gCb

        # Period-2 Nyquist subspace regularizer (DC-invariant, zero on H/V edges)
        nyq_val = 0.0
        if lam_nyq > 0.0:
            nyq_val, g_nyq = _nyquist_penalty_value_and_grad_cpu(x)
            grad += np.float32(lam_nyq) * g_nyq

        total_obj = data_val + ll * luma_val + lam_chroma * (cr_val + cb_val) + lam_nyq * nyq_val
        rms_u = float(np.sqrt(sq_u / max(total_valid, 1)))
        return total_obj, grad, rms_u

    env = _compute_local_envelope(x0, sigma_med, env_tol, env_radius, mismatch_hr=mismatch_hr)
    if env is not None:
        x_lo, x_hi = env
    else:
        x_lo = np.float32(0.0)
        x_hi = np.float32(1.0)

    x = x0.copy()
    obj, grad, rms_u0 = eval_obj_grad(x, lam_luma)
    obj0 = obj
    rms_u = rms_u0
    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []
    rho_hist: List[float] = []
    adapted = False
    iters_run = 0
    stop_reason = "max_iter"
    rel_step = 1.0
    rel_obj = 1.0

    for it in range(max_iter):
        iters_run = it + 1
        # Projected L-BFGS two-loop recursion with diagonal coverage preconditioner
        q = grad.copy()
        alphas: List[float] = []
        for s_i, y_i, rho_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a_i = rho_i * float(np.sum(s_i.astype(np.float64) * q.astype(np.float64)))
            alphas.append(a_i)
            q = q - np.float32(a_i) * y_i
        if s_hist:
            sy_last = float(np.sum(s_hist[-1].astype(np.float64) * y_hist[-1].astype(np.float64)))
            yHy = float(np.sum((y_hist[-1].astype(np.float64) ** 2) * inv_diag.astype(np.float64)))
            gamma = float(np.clip(sy_last / max(yHy, 1e-12), 1e-3, 1e3))
        else:
            gamma = 1.0
        r_vec = (np.float32(gamma) * inv_diag) * q
        for s_i, y_i, rho_i, a_i in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            b_i = rho_i * float(np.sum(y_i.astype(np.float64) * r_vec.astype(np.float64)))
            r_vec = r_vec + np.float32(a_i - b_i) * s_i
        p = -r_vec

        # Active-set box projection on direction
        active_lo = (x <= x_lo) & (grad > np.float32(0.0))
        active_hi = (x >= x_hi) & (grad < np.float32(0.0))
        p = np.where(active_lo | active_hi, np.float32(0.0), p)

        pg = float(np.sum(p.astype(np.float64) * grad.astype(np.float64)))
        if pg >= 0.0:
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            p = -inv_diag * grad
            p = np.where(active_lo | active_hi, np.float32(0.0), p)
            pg = float(np.sum(p.astype(np.float64) * grad.astype(np.float64)))
            if pg >= -1e-18:
                stop_reason = "zero_proj_grad"
                break

        # Armijo backtracking line search
        alpha = 1.0
        accepted = False
        for _ in range(8):
            x_new = np.clip(x + np.float32(alpha) * p, x_lo, x_hi)
            obj_new, grad_new, rms_u_new = eval_obj_grad(x_new, lam_luma)
            dx_actual = (x_new - x).astype(np.float64)
            dg = float(np.sum(grad.astype(np.float64) * dx_actual))
            if obj_new <= obj + 1e-4 * min(dg, 0.0) + 1e-12:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if s_hist:
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                continue
            stop_reason = "line_search_stop"
            break

        s = x_new - x
        y_diff = grad_new - grad
        s_norm = float(np.sqrt(np.sum(s.astype(np.float64) ** 2)))
        x_norm = float(np.sqrt(np.sum(x.astype(np.float64) ** 2))) + 1e-8
        rel_step = s_norm / x_norm
        rel_obj = abs(obj - obj_new) / max(abs(obj), 1e-12)

        sy = float(np.sum(s.astype(np.float64) * y_diff.astype(np.float64)))
        y_norm = float(np.sqrt(np.sum(y_diff.astype(np.float64) ** 2)))
        if sy > 1e-10 * s_norm * y_norm:
            if len(s_hist) >= lbfgs_m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y_diff)
            rho_hist.append(1.0 / sy)

        x, obj, grad, rms_u = x_new, obj_new, grad_new, rms_u_new

        # Discrepancy-principle check after warm-up: if overfitting noise (rms_u < 0.90), raise lam_luma
        if discrepancy_adapt and not adapted and it == 12:
            adapted = True
            if rms_u < 0.90:
                scale_up = min((0.95 / max(rms_u, 0.35)) ** 2, 4.0)
                lam_luma *= scale_up
                inv_diag = make_inv_diag(lam_luma)
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                obj, grad, rms_u = eval_obj_grad(x, lam_luma)
                continue

        if it >= 5 and (rel_step < tol or rel_obj < 1e-5):
            stop_reason = "rel_step_tol" if rel_step < tol else "rel_obj_tol"
            break

    elapsed = float(time.perf_counter() - t0)
    return MAPSRResult(
        hr=x.astype(np.float32),
        metrics={
            "engine": "cpu",
            "mode": "wp13",
            "prior": prior,
            "iterations": iters_run,
            "stop_reason": stop_reason,
            "obj_init": round(obj0, 4),
            "obj_final": round(obj, 4),
            "rel_step_final": round(rel_step, 6),
            "rel_obj_final": round(rel_obj, 7),
            "rms_u_init": round(rms_u0, 4),
            "rms_u_final": round(rms_u, 4),
            "sigma_n": round(cfg["sigma_med"], 6),
            "noise_a": round(cfg["a_var"], 8),
            "noise_b": round(cfg["b_gain"], 6),
            "lam_luma": round(lam_luma, 4),
            "lam_chroma": round(lam_chroma, 4),
            "lam_nyq": round(lam_nyq, 4),
            "psf_source": psf_source,
            "psf_sigma_rgb": [round(float(v), 4) for v in psf_sigma_rgb],
            "psf_ref_rgb": (psf_meta or {}).get("ref_sigma_rgb"),
            "psf_hr_rgb": (psf_meta or {}).get("hr_sigma_rgb"),
            "env_tol": env_tol,
            "env_radius": int(env_radius),
            "gr_gb_ratio": round(float(gr_gb_ratio), 5),
            "seconds": round(elapsed, 3),
        },
    )


def _solve_map_wp13_gpu(
    x0: np.ndarray,
    mosaics: Sequence[np.ndarray],
    transforms: Sequence[Any],
    ratio: float,
    pattern: Any,
    w_arr: np.ndarray,
    k1d_rgb: Tuple[np.ndarray, np.ndarray, np.ndarray],
    colour_matrix: np.ndarray,
    *,
    ref_index: int,
    prior: str,
    luma_weight: float,
    chroma_weight: float,
    nyquist_weight: float,
    huber_kappa: float,
    discrepancy_adapt: bool,
    lbfgs_m: int,
    max_iter: int,
    tol: float,
    psf_source: str,
    psf_sigma_rgb: Tuple[float, float, float],
    psf_meta: Optional[Dict[str, Any]] = None,
    env_tol: Optional[float] = 0.08,
    env_radius: int = 2,
    gr_gb_ratio: float,
    t0: float,
) -> MAPSRResult:
    from .gpu import _import_cupy
    cp, _ = _import_cupy()
    fwd_kern, adj_kern = _get_cfa_kernels(cp)

    hr_shape = (int(x0.shape[0]), int(x0.shape[1]))
    lr_shape = (int(mosaics[0].shape[0]), int(mosaics[0].shape[1]))
    h_hr, w_hr = hr_shape
    h_lr, w_lr = lr_shape

    y_list_np = [np.asarray(m, dtype=np.float32) for m in mosaics]
    cfg = _wp13_setup_weights_and_priors(
        x0, y_list_np, pattern, w_arr, ratio, ref_index,
        prior, luma_weight, chroma_weight, nyquist_weight,
    )
    eps_y, p_hl = cfg["eps_y"], cfg["p_hl"]
    lam_luma = cfg["lam_luma"]
    lam_chroma = cfg["lam_chroma"]
    lam_nyq = cfg["lam_nyq"]
    kappa = cp.float32(max(huber_kappa, 0.5))
    inv_kappa2 = cp.float32(1.0 / (float(huber_kappa) ** 2))
    kappa2 = cp.float32(float(huber_kappa) ** 2)

    a_var, b_gain, sigma_med = cfg["a_var"], cfg["b_gain"], cfg["sigma_med"]
    C_g = cp.asarray(colour_matrix, dtype=cp.float32)
    CT_g = cp.ascontiguousarray(C_g.T)
    C_abs_g = cp.abs(C_g)
    ch_map_g = cp.asarray(channel_map(pattern_array(pattern), lr_shape), dtype=cp.int8)
    y_list_g = [cp.asarray(m, dtype=cp.float32) for m in y_list_np]
    wx_g = cp.asarray(cfg["wx"], dtype=cp.float32)
    wy_g = cp.asarray(cfg["wy"], dtype=cp.float32)
    matrices = [to_drizzle_matrix(W, ratio).astype(np.float64) for W in transforms]

    block = (16, 16)
    grid = ((w_lr + block[0] - 1) // block[0], (h_lr + block[1] - 1) // block[1])
    pred_buf = cp.empty((h_lr, w_lr), dtype=cp.float32)
    valid_buf = cp.empty((h_lr, w_lr), dtype=cp.uint8)

    # Unbiased Poisson-Gaussian variance from expected signal mu_k = A_k(x0)
    sig_lo = cp.float32(max(0.75 * sigma_med, 1e-4))
    sig_hi = cp.float32(max(1.50 * sigma_med, 2e-4))
    x_g = cp.asarray(x0, dtype=cp.float32)
    z0_g = _blur_forward_rgb_gpu(cp, x_g, k1d_rgb)
    z0_cam_g = cp.ascontiguousarray(cp.matmul(z0_g, CT_g))
    med_u: List[float] = []
    valid_masks_g = []
    inv_sig_g = []
    inv_var_g = []
    acc_mw_g = cp.zeros((h_hr, w_hr, 3), dtype=cp.float32)
    acc_mu_g = cp.zeros((h_hr, w_hr, 3), dtype=cp.float32)
    acc_mu2_g = cp.zeros((h_hr, w_hr, 3), dtype=cp.float32)
    for yk_g, M in zip(y_list_g, matrices):
        fwd_kern(
            grid, block,
            (z0_cam_g, h_hr, w_hr, ch_map_g, h_lr, w_lr,
             float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
             float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
             float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
             pred_buf, valid_buf),
        )
        vm = valid_buf.copy()
        valid_masks_g.append(vm)
        var_g = cp.float32(a_var) + cp.float32(b_gain) * cp.clip(pred_buf, cp.float32(0.0), cp.float32(1.0))
        sig_g = cp.clip(cp.sqrt(cp.maximum(var_g, cp.float32(1e-8))), sig_lo, sig_hi)
        isig_g = cp.float32(1.0) / sig_g
        inv_sig_g.append(isig_g)
        inv_var_g.append(isig_g * isig_g)
        u_map_g = cp.where(vm != 0, (pred_buf - yk_g) * isig_g, cp.float32(0.0))
        v_map_g = cp.where(vm != 0, cp.float32(1.0), cp.float32(0.0))
        m_args = (
            float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
            float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
            float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
        )
        adj_kern(grid, block, (v_map_g, vm, ch_map_g, h_lr, w_lr, *m_args, acc_mw_g, h_hr, w_hr))
        adj_kern(grid, block, (u_map_g, vm, ch_map_g, h_lr, w_lr, *m_args, acc_mu_g, h_hr, w_hr))
        adj_kern(grid, block, (u_map_g * u_map_g, vm, ch_map_g, h_lr, w_lr, *m_args, acc_mu2_g, h_hr, w_hr))
        u_v = cp.abs(u_map_g)[vm != 0]
        med_u.append(float(cp.median(u_v).get()) if u_v.size > 0 else 1.0)
    w_use = _refine_robust_frame_weights(med_u, w_arr, ref_index)
    sum_mw_g = cp.maximum(cp.sum(acc_mw_g, axis=2), cp.float32(1e-6))
    mean_u_g = cp.sum(acc_mu_g, axis=2) / sum_mw_g
    mean_u2_g = cp.sum(acc_mu2_g, axis=2) / sum_mw_g
    mismatch_hr = cp.asnumpy(cp.sqrt(cp.maximum(mean_u2_g - mean_u_g * mean_u_g, cp.float32(0.0)))).astype(np.float32)
    del acc_mw_g, acc_mu_g, acc_mu2_g, sum_mw_g, mean_u_g, mean_u2_g
    total_valid = sum(int(cp.sum(vm, dtype=cp.int64).get()) for vm in valid_masks_g)

    acc_d = cp.zeros_like(x_g)
    for wk, ivar_g, vm, M in zip(w_use, inv_var_g, valid_masks_g, matrices):
        w_ivar = cp.where(vm != 0, ivar_g * cp.float32(wk), cp.float32(0.0))
        adj_kern(
            grid, block,
            (w_ivar, vm, ch_map_g, h_lr, w_lr,
             float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
             float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
             float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
             acc_d, h_hr, w_hr),
        )
    d_data_g = _blur_adjoint_rgb_gpu(cp, cp.matmul(acc_d, C_abs_g), k1d_rgb)

    def make_inv_diag_g(ll: float):
        d_prior = cp.float32(ll / max(eps_y, 1e-4) + 4.0 * lam_chroma + 0.25 * lam_nyq)
        return cp.float32(1.0) / cp.maximum(d_data_g + d_prior, cp.float32(1.0))

    inv_diag_g = make_inv_diag_g(lam_luma)

    def eval_obj_grad_gpu(xg, ll: float):
        z_g = _blur_forward_rgb_gpu(cp, xg, k1d_rgb)
        z_cam_g = cp.ascontiguousarray(cp.matmul(z_g, CT_g))
        data_val = 0.0
        sq_u = 0.0
        acc_z = cp.zeros_like(xg)
        for yk_g, wk, isig_g, vm, M in zip(y_list_g, w_use, inv_sig_g, valid_masks_g, matrices):
            fwd_kern(
                grid, block,
                (z_cam_g, h_hr, w_hr, ch_map_g, h_lr, w_lr,
                 float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                 float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
                 float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
                 pred_buf, valid_buf),
            )
            u = cp.where(vm != 0, (pred_buf - yk_g) * isig_g, cp.float32(0.0))
            u2 = u * u
            sq_u += float(cp.sum(u2, dtype=cp.float64).get())
            rt = cp.sqrt(cp.float32(1.0) + u2 * inv_kappa2)
            rho = kappa2 * (rt - cp.float32(1.0))
            data_val += float(wk) * float(cp.sum(rho, dtype=cp.float64).get())
            psi = (u / rt) * isig_g * cp.float32(wk)
            adj_kern(
                grid, block,
                (psi, vm, ch_map_g, h_lr, w_lr,
                 float(M[0, 0]), float(M[0, 1]), float(M[0, 2]),
                 float(M[1, 0]), float(M[1, 1]), float(M[1, 2]),
                 float(M[2, 0]), float(M[2, 1]), float(M[2, 2]),
                 acc_z, h_hr, w_hr),
            )
        grad_g = _blur_adjoint_rgb_gpu(cp, cp.matmul(acc_z, C_g), k1d_rgb)

        Y_g = (
            cp.float32(0.2126) * xg[..., 0]
            + cp.float32(0.7152) * xg[..., 1]
            + cp.float32(0.0722) * xg[..., 2]
        )
        if prior == "btv":
            luma_val, gY = _btv_value_and_grad_gpu(cp, Y_g, eps=eps_y)
        elif prior == "huber_tv":
            luma_val, gY = _huber_tv_value_and_grad_gpu(cp, Y_g, eps=eps_y)
        else:
            luma_val, gY = _hyper_lap_value_and_grad_gpu(cp, Y_g, eps=eps_y, p=p_hl)

        Cr_g = xg[..., 0] - xg[..., 1]
        Cb_g = xg[..., 2] - xg[..., 1]
        cr_val, gCr = _edge_chroma_value_and_grad_gpu(cp, Cr_g, wx_g, wy_g)
        cb_val, gCb = _edge_chroma_value_and_grad_gpu(cp, Cb_g, wx_g, wy_g)

        ll32 = cp.float32(ll)
        lc32 = cp.float32(lam_chroma)
        grad_g[..., 0] += ll32 * cp.float32(0.2126) * gY + lc32 * gCr
        grad_g[..., 1] += ll32 * cp.float32(0.7152) * gY - lc32 * (gCr + gCb)
        grad_g[..., 2] += ll32 * cp.float32(0.0722) * gY + lc32 * gCb

        nyq_val = 0.0
        if lam_nyq > 0.0:
            nyq_val, g_nyq = _nyquist_penalty_value_and_grad_gpu(cp, xg)
            grad_g += cp.float32(lam_nyq) * g_nyq

        total_obj = data_val + ll * luma_val + lam_chroma * (cr_val + cb_val) + lam_nyq * nyq_val
        rms_u = float(np.sqrt(sq_u / max(total_valid, 1)))
        return total_obj, grad_g, rms_u

    env = _compute_local_envelope(x0, sigma_med, env_tol, env_radius, mismatch_hr=mismatch_hr)
    if env is not None:
        x_lo_g = cp.asarray(env[0], dtype=cp.float32)
        x_hi_g = cp.asarray(env[1], dtype=cp.float32)
    else:
        x_lo_g = cp.float32(0.0)
        x_hi_g = cp.float32(1.0)

    obj, grad_g, rms_u0 = eval_obj_grad_gpu(x_g, lam_luma)
    obj0 = obj
    rms_u = rms_u0
    s_hist: list = []
    y_hist: list = []
    rho_hist: List[float] = []
    adapted = False
    iters_run = 0
    stop_reason = "max_iter"
    rel_step = 1.0
    rel_obj = 1.0

    for it in range(max_iter):
        iters_run = it + 1
        q = grad_g.copy()
        alphas: List[float] = []
        for s_i, y_i, rho_i in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a_i = rho_i * float(cp.sum(s_i.astype(cp.float64) * q.astype(cp.float64)).get())
            alphas.append(a_i)
            q = q - cp.float32(a_i) * y_i
        if s_hist:
            sy_last = float(cp.sum(s_hist[-1].astype(cp.float64) * y_hist[-1].astype(cp.float64)).get())
            yHy = float(cp.sum((y_hist[-1].astype(cp.float64) ** 2) * inv_diag_g.astype(cp.float64)).get())
            gamma = float(np.clip(sy_last / max(yHy, 1e-12), 1e-3, 1e3))
        else:
            gamma = 1.0
        r_vec = (cp.float32(gamma) * inv_diag_g) * q
        for s_i, y_i, rho_i, a_i in zip(s_hist, y_hist, rho_hist, reversed(alphas)):
            b_i = rho_i * float(cp.sum(y_i.astype(cp.float64) * r_vec.astype(cp.float64)).get())
            r_vec = r_vec + cp.float32(a_i - b_i) * s_i
        p_g = -r_vec

        active_lo = (x_g <= x_lo_g) & (grad_g > cp.float32(0.0))
        active_hi = (x_g >= x_hi_g) & (grad_g < cp.float32(0.0))
        p_g = cp.where(active_lo | active_hi, cp.float32(0.0), p_g)

        pg = float(cp.sum(p_g.astype(cp.float64) * grad_g.astype(cp.float64)).get())
        if pg >= 0.0:
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            p_g = -inv_diag_g * grad_g
            p_g = cp.where(active_lo | active_hi, cp.float32(0.0), p_g)
            pg = float(cp.sum(p_g.astype(cp.float64) * grad_g.astype(cp.float64)).get())
            if pg >= -1e-18:
                stop_reason = "zero_proj_grad"
                break

        alpha = 1.0
        accepted = False
        for _ in range(8):
            x_new = cp.clip(x_g + cp.float32(alpha) * p_g, x_lo_g, x_hi_g)
            obj_new, grad_new, rms_u_new = eval_obj_grad_gpu(x_new, lam_luma)
            dx_actual = x_new - x_g
            dg = float(cp.sum(grad_g.astype(cp.float64) * dx_actual.astype(cp.float64)).get())
            if obj_new <= obj + 1e-4 * min(dg, 0.0) + 1e-12:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            if s_hist:
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                continue
            stop_reason = "line_search_stop"
            break

        s = x_new - x_g
        y_diff = grad_new - grad_g
        s_norm = float(np.sqrt(float(cp.sum(s.astype(cp.float64) ** 2).get())))
        x_norm = float(np.sqrt(float(cp.sum(x_g.astype(cp.float64) ** 2).get()))) + 1e-8
        rel_step = s_norm / x_norm
        rel_obj = abs(obj - obj_new) / max(abs(obj), 1e-12)

        sy = float(cp.sum(s.astype(cp.float64) * y_diff.astype(cp.float64)).get())
        y_norm = float(np.sqrt(float(cp.sum(y_diff.astype(cp.float64) ** 2).get())))
        if sy > 1e-10 * s_norm * y_norm:
            if len(s_hist) >= lbfgs_m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y_diff)
            rho_hist.append(1.0 / sy)

        x_g, obj, grad_g, rms_u = x_new, obj_new, grad_new, rms_u_new

        if discrepancy_adapt and not adapted and it == 12:
            adapted = True
            if rms_u < 0.90:
                scale_up = min((0.95 / max(rms_u, 0.35)) ** 2, 4.0)
                lam_luma *= scale_up
                inv_diag_g = make_inv_diag_g(lam_luma)
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                obj, grad_g, rms_u = eval_obj_grad_gpu(x_g, lam_luma)
                continue

        if it >= 5 and (rel_step < tol or rel_obj < 1e-5):
            stop_reason = "rel_step_tol" if rel_step < tol else "rel_obj_tol"
            break

    hr_out = cp.asnumpy(x_g).astype(np.float32)
    elapsed = float(time.perf_counter() - t0)
    return MAPSRResult(
        hr=hr_out,
        metrics={
            "engine": "cupy",
            "mode": "wp13",
            "prior": prior,
            "iterations": iters_run,
            "stop_reason": stop_reason,
            "obj_init": round(obj0, 4),
            "obj_final": round(obj, 4),
            "rel_step_final": round(rel_step, 6),
            "rel_obj_final": round(rel_obj, 7),
            "rms_u_init": round(rms_u0, 4),
            "rms_u_final": round(rms_u, 4),
            "sigma_n": round(cfg["sigma_med"], 6),
            "noise_a": round(cfg["a_var"], 8),
            "noise_b": round(cfg["b_gain"], 6),
            "lam_luma": round(lam_luma, 4),
            "lam_chroma": round(lam_chroma, 4),
            "lam_nyq": round(lam_nyq, 4),
            "psf_source": psf_source,
            "psf_sigma_rgb": [round(float(v), 4) for v in psf_sigma_rgb],
            "psf_ref_rgb": (psf_meta or {}).get("ref_sigma_rgb"),
            "psf_hr_rgb": (psf_meta or {}).get("hr_sigma_rgb"),
            "env_tol": env_tol,
            "env_radius": int(env_radius),
            "gr_gb_ratio": round(float(gr_gb_ratio), 5),
            "seconds": round(elapsed, 3),
        },
    )
