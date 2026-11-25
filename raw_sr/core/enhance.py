"""Classical (non-neural) resolution enhancement on top of linear SR masters.

This module implements astrophotography-grade mathematical enhancement for
linear multi-frame super-resolution outputs:

1. Forward-model PSF construction (sensor integration box + optics Gaussian)
   with slanted-edge ESF->LSF optics sigma estimation.
2. Luminance-only TV-regularized Richardson-Lucy deconvolution with noise-aware
   stopping (discrepancy principle) and local-support deringing.
3. A-trous starlet (B3 spline) multiscale sharpening with MAD k-sigma noise
   thresholding and highlight/shadow protection.

The measured linear SR master is never modified in place; this module always
produces a separate "Enhanced" image.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

# Rec. 709 linear luminance weights
_LUMA_R = np.float32(0.2126)
_LUMA_G = np.float32(0.7152)
_LUMA_B = np.float32(0.0722)

# B3 spline 1D kernel for a-trous starlet transform
_B3_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0

# Theoretical white-noise standard deviation factors for 2D B3 starlet layers
# (Starck & Murtagh, Astronomical Image and Data Analysis)
_STARLET_NOISE_FACTORS = (0.8907, 0.2007, 0.0856, 0.0413, 0.0205)

# Safe default optics blur sigma in LR pixels
DEFAULT_OPTICS_SIGMA_LR = 0.35

# Named presets mapping to a single strength parameter in [0, 1]
ENHANCE_PRESETS: Dict[str, float] = {
    "off": 0.0,
    "subtle": 0.35,
    "balanced": 0.65,
    "astro": 0.90,
}

DEFAULT_PRESET = "balanced"
DEFAULT_STRENGTH = ENHANCE_PRESETS[DEFAULT_PRESET]
DEFAULT_ENHANCE_MODE = "wp2"  # "wp2" (legacy) or "adaptive" (WP6 PSF-estimated, scene-adaptive)


@dataclass
class PSFEstimate:
    """Estimated optical PSF sigma (in LR pixels) from image edges."""

    sigma_lr: float
    sigma_rgb: Tuple[float, float, float]
    tile_sigmas: np.ndarray  # (ny, nx) float32 array, NaN where no edges
    n_edges: int
    used_fallback: bool


@dataclass
class EnhanceConfig:
    """Configuration for classical resolution enhancement."""

    strength: float = DEFAULT_STRENGTH
    mode: str = DEFAULT_ENHANCE_MODE         # "wp2" or "adaptive"
    optics_sigma_lr: Optional[float] = None  # None -> auto-estimate from edges
    max_rl_iters: Optional[int] = None       # None -> derived from strength
    tv_weight: Optional[float] = None        # None -> derived from strength & noise
    dering_tolerance: float = 0.025          # Fraction of local min/max range
    dering_radius: int = 2                   # Local support radius in HR px
    starlet_scales: int = 3                  # Number of a-trous detail scales
    starlet_k_sigma: float = 3.0             # Noise threshold in MAD sigmas
    starlet_gains: Optional[Tuple[float, ...]] = None  # None -> derived from strength
    shadow_protect: float = 0.03             # Linear luma knee for shadow protection
    highlight_protect: float = 0.06          # Linear luma knee for highlight protection
    highlight_clip_knee: float = 0.80        # Adaptive mode: attenuate above this linear level
    psf_tile_grid: Tuple[int, int] = (4, 4)  # Adaptive mode: spatial tile grid for PSF
    safety_check: bool = True                # Adaptive mode: auto back-off if MTF drops or noise > +5%
    safety_max_noise_ratio: float = 1.05     # Max allowed flat-noise ratio vs plain SR

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "EnhanceConfig":
        key = preset.strip().lower()
        if key not in ENHANCE_PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Expected one of {sorted(ENHANCE_PRESETS)}."
            )
        return cls(strength=ENHANCE_PRESETS[key], **overrides)


@dataclass
class EnhanceResult:
    """Output of :func:`enhance_image` with diagnostic metadata."""

    image: np.ndarray
    optics_sigma_lr: float
    noise_sigma: float
    rl_iterations_run: int
    rl_stopped_reason: str
    strength: float
    psf_1d: np.ndarray
    details: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Luminance / Chroma & Noise helpers
# --------------------------------------------------------------------------- #

def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """Compute linear Rec. 709 luminance from an (H, W, 3) float32 RGB image."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32, copy=False)
    return (
        _LUMA_R * rgb[..., 0]
        + _LUMA_G * rgb[..., 1]
        + _LUMA_B * rgb[..., 2]
    ).astype(np.float32, copy=False)


def apply_luma_to_rgb(
    rgb: np.ndarray,
    luma_orig: np.ndarray,
    luma_enh: np.ndarray,
    eps: float = 1e-4,
    max_scale: float = 4.0,
) -> np.ndarray:
    """Recombine enhanced luminance with original chroma (in linear light)."""
    if rgb.ndim == 2:
        return np.clip(luma_enh, 0.0, 1.0).astype(np.float32)
    denom = np.maximum(luma_orig, np.float32(eps))
    scale = np.clip(luma_enh / denom, np.float32(0.0), np.float32(max_scale))
    # For very dark pixels below eps, add the luma delta directly to avoid
    # amplifying chroma noise by dividing by ~0.
    dark = luma_orig < eps
    out = rgb * scale[..., None]
    if np.any(dark):
        delta = (luma_enh - luma_orig)[..., None]
        out = np.where(dark[..., None], rgb + delta, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Estimate additive noise sigma via Immerkaer's fast Laplacian operator."""
    gray = rgb_to_luma(img).astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    r = cv2.filter2D(gray, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
    # Use MAD rather than mean(|r|) so strong edges/textures do not inflate sigma
    mad = float(np.median(np.abs(r)))
    # For N(0, sigma^2), std(r) = 6*sigma, median(|r|) = 0.67448975 * 6 * sigma
    return max(mad / (6.0 * 0.67448975), 1e-7)


def effective_stack_noise(
    frame_noise_sigma: Optional[Sequence[float]],
    effective_frames: Optional[float] = None,
) -> Optional[float]:
    """Compute expected linear noise sigma of a stacked master."""
    if not frame_noise_sigma:
        return None
    arr = np.asarray(frame_noise_sigma, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    n_eff = float(effective_frames) if effective_frames is not None else float(arr.size)
    n_eff = max(n_eff, 1.0)
    return float(np.median(arr) / np.sqrt(n_eff))


# --------------------------------------------------------------------------- #
# Forward-model PSF & Optics Sigma Estimation
# --------------------------------------------------------------------------- #

def build_psf_1d(ratio: float, optics_sigma_lr: float = DEFAULT_OPTICS_SIGMA_LR) -> np.ndarray:
    """Build separable 1D forward-model PSF (sensor box * optics Gaussian).

    Args:
        ratio: Super-resolution scale factor (>= 1.0).
        optics_sigma_lr: Lens optical blur sigma in LR pixels (>= 0).

    Returns:
        1D float32 kernel of odd length summing to 1.0.
    """
    r = max(float(ratio), 1.0)
    sigma_hr = max(float(optics_sigma_lr), 0.0) * r

    # 1. Sensor pixel integration box of width `r` HR pixels (symmetric)
    half_box = int(np.ceil(r * 0.5))
    bx = np.arange(-half_box, half_box + 1, dtype=np.float64)
    # Fraction of each HR pixel covered by [-r/2, +r/2]
    box = np.clip(np.minimum(bx + 0.5, r * 0.5) - np.maximum(bx - 0.5, -r * 0.5), 0.0, 1.0)
    if box.sum() <= 0:
        box = np.array([1.0], dtype=np.float64)
    else:
        box /= box.sum()

    # 2. Optics Gaussian in HR pixels
    if sigma_hr > 1e-4:
        half_g = max(1, int(np.ceil(3.0 * sigma_hr)))
        gx = np.arange(-half_g, half_g + 1, dtype=np.float64)
        gauss = np.exp(-0.5 * (gx / sigma_hr) ** 2)
        gauss /= gauss.sum()
        k1d = np.convolve(box, gauss)
    else:
        k1d = box

    # Trim negligible tails (< 1e-4 of peak) while preserving odd symmetry
    peak = k1d.max()
    nz = np.where(k1d >= peak * 1e-4)[0]
    if nz.size > 0:
        pad = max(len(k1d) - 1 - int(nz[-1]), int(nz[0]))
        if pad > 0 and 2 * pad < len(k1d):
            k1d = k1d[pad:-pad]

    # Ensure odd length and exact unit sum
    if len(k1d) % 2 == 0:
        k1d = np.pad(k1d, (0, 1), mode="constant")
    k1d = k1d / k1d.sum()
    return k1d.astype(np.float32)


def build_psf(ratio: float, optics_sigma_lr: float = DEFAULT_OPTICS_SIGMA_LR) -> np.ndarray:
    """Build normalized 2D forward-model PSF (sensor box * optics Gaussian)."""
    k1d = build_psf_1d(ratio, optics_sigma_lr)
    psf2d = np.outer(k1d, k1d).astype(np.float32)
    psf2d /= psf2d.sum()
    return psf2d


def estimate_optics_sigma(
    img: np.ndarray,
    ratio: float,
    default_sigma_lr: float = DEFAULT_OPTICS_SIGMA_LR,
    min_sigma_lr: float = 0.15,
    max_sigma_lr: float = 0.70,
) -> float:
    """Estimate optics Gaussian sigma (in LR px) from strongest edges via ESF->LSF.

    Samples high-gradient edge patches, projects pixels onto the local edge
    normal to build an oversampled ESF, differentiates to LSF, measures its
    Gaussian-equivalent sigma, and subtracts the sensor box variance
    ``(ratio^2 / 12)`` in quadrature. Falls back to ``default_sigma_lr`` if
    no clean high-contrast edges are found.
    """
    luma = rgb_to_luma(img)
    H, W = luma.shape
    r = max(float(ratio), 1.0)
    patch_r = max(8, int(round(4.0 * r)))
    if H < 2 * patch_r + 8 or W < 2 * patch_r + 8:
        return float(default_sigma_lr)

    # Smooth slightly to find stable edge normals without noise bias
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)

    # Exclude image border
    margin = patch_r + 4
    interior = gmag[margin:-margin, margin:-margin]
    if interior.size == 0:
        return float(default_sigma_lr)

    p99 = float(np.percentile(interior, 99.5))
    if p99 < 1e-3:
        return float(default_sigma_lr)

    # Pick up to 16 well-separated strong edge locations via coarse grid max
    grid = max(patch_r * 2, min(H, W) // 8)
    candidates = []
    for y0 in range(margin, H - margin, grid):
        for x0 in range(margin, W - margin, grid):
            y1 = min(y0 + grid, H - margin)
            x1 = min(x0 + grid, W - margin)
            sub = gmag[y0:y1, x0:x1]
            idx = int(np.argmax(sub))
            val = float(sub.ravel()[idx])
            if val >= 0.35 * p99:
                ly, lx = divmod(idx, sub.shape[1])
                candidates.append((val, y0 + ly, x0 + lx))

    if not candidates:
        return float(default_sigma_lr)
    candidates.sort(reverse=True)
    candidates = candidates[:16]

    sigmas_lr = []
    oversamp = 2
    bin_w = 1.0 / oversamp
    d_max = float(patch_r)
    bins = np.arange(-d_max, d_max + bin_w, bin_w)
    n_bins = len(bins) - 1
    yy_p, xx_p = np.mgrid[-patch_r:patch_r + 1, -patch_r:patch_r + 1].astype(np.float64)

    for _, cy, cx in candidates:
        patch = luma[cy - patch_r:cy + patch_r + 1, cx - patch_r:cx + patch_r + 1].astype(np.float64)
        # Local dominant gradient direction
        gx_loc = float(gx[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        gy_loc = float(gy[cy - 1:cy + 2, cx - 1:cx + 2].mean())
        gnorm = np.hypot(gx_loc, gy_loc)
        if gnorm < 1e-5:
            continue
        nx, ny = gx_loc / gnorm, gy_loc / gnorm

        # Signed distance along edge normal
        dist = (xx_p * nx + yy_p * ny).ravel()
        vals = patch.ravel()

        idx = np.digitize(dist, bins) - 1
        ok = (idx >= 0) & (idx < n_bins)
        esf = np.zeros(n_bins, dtype=np.float64)
        cnt = np.zeros(n_bins, dtype=np.float64)
        np.add.at(esf, idx[ok], vals[ok])
        np.add.at(cnt, idx[ok], 1.0)
        good = cnt > 0
        if good.sum() < 12:
            continue
        esf[good] /= cnt[good]
        esf = np.interp(np.arange(n_bins), np.where(good)[0], esf[good])

        lsf = np.diff(esf)
        # Keep positive lobe around peak
        pk = int(np.argmax(lsf))
        pk_val = float(lsf[pk])
        if pk_val <= 1e-4 or pk < 3 or pk > len(lsf) - 4:
            continue

        # Measure FWHM around peak (robust against distant texture)
        half_max = 0.5 * pk_val
        left = pk
        while left > 0 and lsf[left] > half_max:
            left -= 1
        right = pk
        while right < len(lsf) - 1 and lsf[right] > half_max:
            right += 1
        if right <= left:
            continue
        # Linear interpolation at half-max crossings
        t_l = (half_max - lsf[left]) / max(lsf[left + 1] - lsf[left], 1e-12)
        t_r = (half_max - lsf[right]) / max(lsf[right - 1] - lsf[right], 1e-12)
        fwhm_bins = (right - t_r) - (left + t_l)
        fwhm_hr = fwhm_bins * bin_w
        sigma_total_hr = fwhm_hr / 2.354820045

        # Deconvolve sensor box variance (ratio^2 / 12) in quadrature
        var_optics_hr = sigma_total_hr ** 2 - (r * r) / 12.0
        if var_optics_hr > 0.0:
            s_lr = float(np.sqrt(var_optics_hr) / r)
            if 0.05 <= s_lr <= 1.5:
                sigmas_lr.append(s_lr)

    if len(sigmas_lr) < 2:
        return float(default_sigma_lr)

    # Use lower quartile (sharpest in-focus edges) to avoid out-of-focus bias
    est = float(np.percentile(sigmas_lr, 30))
    return float(np.clip(est, min_sigma_lr, max_sigma_lr))


# --------------------------------------------------------------------------- #
# WP6: Structure-Tensor PSF Estimation (Per-Channel & Per-Tile)
# --------------------------------------------------------------------------- #

def _model_esf_on_grid(ratio: float, sigma_lr: float, d_grid: np.ndarray) -> np.ndarray:
    """Evaluate the analytical forward-model ESF (box * Gaussian) on signed HR distances."""
    r = max(float(ratio), 1.0)
    k1d = build_psf_1d(r, sigma_lr).astype(np.float64)
    half = len(k1d) // 2
    kx = np.arange(-half, half + 1, dtype=np.float64)
    # Cumulative integral of PSF (piecewise linear between integer HR samples)
    cdf = np.cumsum(k1d)
    cdf = np.concatenate([[0.0], cdf])
    edges = np.concatenate([[kx[0] - 0.5], kx + 0.5])
    return np.interp(d_grid, edges, cdf, left=0.0, right=1.0)


def _estimate_channel_psf_tiles(
    chan: np.ndarray,
    ratio: float,
    tile_grid: Tuple[int, int] = (4, 4),
    min_sigma_lr: float = 0.15,
    max_sigma_lr: float = 1.25,
    coherence_thr: float = 0.65,
    max_edges_per_tile: int = 3,
) -> Tuple[list[float], np.ndarray]:
    """Detect strong straight edges via structure tensor coherence and fit ESF->sigma."""
    H, W = chan.shape
    ny, nx = tile_grid
    tile_sigmas = np.full((ny, nx), np.nan, dtype=np.float32)
    r = max(float(ratio), 1.0)
    patch_r = max(10, int(round(5.0 * r)))
    if H < 2 * patch_r + 8 or W < 2 * patch_r + 8:
        return [], tile_sigmas

    # Structure tensor on slightly smoothed image to suppress pixel noise
    sm = cv2.GaussianBlur(chan, (3, 3), 0.7, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Scharr(sm, cv2.CV_32F, 1, 0) * np.float32(1.0 / 32.0)
    gy = cv2.Scharr(sm, cv2.CV_32F, 0, 1) * np.float32(1.0 / 32.0)

    win_s = max(1.5, 0.8 * r)
    k_st = int(np.ceil(3.0 * win_s)) * 2 + 1
    jxx = cv2.GaussianBlur(gx * gx, (k_st, k_st), win_s, borderType=cv2.BORDER_REPLICATE)
    jxy = cv2.GaussianBlur(gx * gy, (k_st, k_st), win_s, borderType=cv2.BORDER_REPLICATE)
    jyy = cv2.GaussianBlur(gy * gy, (k_st, k_st), win_s, borderType=cv2.BORDER_REPLICATE)

    trace = jxx + jyy
    diff = jxx - jyy
    disc = np.sqrt(np.maximum(diff * diff + np.float32(4.0) * jxy * jxy, np.float32(0.0)))
    coherence = (disc / np.maximum(trace, np.float32(1e-10))) ** 2
    strength = np.sqrt(np.maximum(trace, np.float32(0.0)))

    margin = patch_r + 4
    interior_s = strength[margin:-margin, margin:-margin]
    if interior_s.size == 0:
        return [], tile_sigmas
    p99 = float(np.percentile(interior_s, 99.0))
    if p99 < 1e-3:
        return [], tile_sigmas

    # Precompute candidate ESF model bank
    oversamp = 4
    bin_w = 1.0 / oversamp
    fit_half = min(float(patch_r) - 1.5, max(4.0 * r, 8.0))
    d_grid = np.arange(-fit_half, fit_half + 0.5 * bin_w, bin_w, dtype=np.float64)
    n_fit = len(d_grid)

    cand_sigmas = np.linspace(min_sigma_lr, max_sigma_lr, 45)
    models = np.stack([_model_esf_on_grid(r, float(s), d_grid) for s in cand_sigmas], axis=0)
    # Precompute centered models for fast closed-form OLS
    m_mean = models.mean(axis=1, keepdims=True)
    m_c = models - m_mean
    m_norm2 = np.sum(m_c * m_c, axis=1) + 1e-12

    yy_p, xx_p = np.mgrid[-patch_r:patch_r + 1, -patch_r:patch_r + 1].astype(np.float64)
    bins_full = np.arange(-float(patch_r), float(patch_r) + bin_w, bin_w)
    n_bins_full = len(bins_full) - 1
    bin_centers = 0.5 * (bins_full[:-1] + bins_full[1:])

    all_sigmas: list[float] = []
    tile_h = max(1, (H - 2 * margin) // ny)
    tile_w = max(1, (W - 2 * margin) // nx)

    for ty in range(ny):
        y0 = margin + ty * tile_h
        y1 = (H - margin) if ty == ny - 1 else min(H - margin, y0 + tile_h)
        if y1 <= y0:
            continue
        for tx in range(nx):
            x0 = margin + tx * tile_w
            x1 = (W - margin) if tx == nx - 1 else min(W - margin, x0 + tile_w)
            if x1 <= x0:
                continue

            sub_s = strength[y0:y1, x0:x1]
            sub_c = coherence[y0:y1, x0:x1]
            # Score favours strong, straight edges
            score = np.where(sub_c >= coherence_thr, sub_s * sub_c, np.float32(0.0))

            # Greedily pick up to max_edges_per_tile well-separated peaks
            tile_vals: list[float] = []
            score_work = score.copy()
            for _ in range(max_edges_per_tile):
                idx = int(np.argmax(score_work))
                best_val = float(score_work.ravel()[idx])
                if best_val < 0.20 * p99:
                    break
                ly, lx = divmod(idx, score_work.shape[1])
                cy, cx = y0 + ly, x0 + lx
                # Suppress neighbourhood so we don't pick the same edge twice
                sy0 = max(0, ly - patch_r)
                sy1 = min(score_work.shape[0], ly + patch_r + 1)
                sx0 = max(0, lx - patch_r)
                sx1 = min(score_work.shape[1], lx + patch_r + 1)
                score_work[sy0:sy1, sx0:sx1] = 0.0

                patch = chan[cy - patch_r:cy + patch_r + 1, cx - patch_r:cx + patch_r + 1].astype(np.float64)
                # Reject clipped / specular patches
                if float(patch.max()) > 0.95 or float(patch.max() - patch.min()) < 0.03:
                    continue

                # Structure tensor dominant eigenvector at (cy, cx)
                angle = 0.5 * np.arctan2(2.0 * float(jxy[cy, cx]), float(jxx[cy, cx] - jyy[cy, cx]))
                enx, eny = float(np.cos(angle)), float(np.sin(angle))
                # Orient normal along local gradient so ESF is increasing
                gx_loc = float(gx[cy - 1:cy + 2, cx - 1:cx + 2].mean())
                gy_loc = float(gy[cy - 1:cy + 2, cx - 1:cx + 2].mean())
                if enx * gx_loc + eny * gy_loc < 0.0:
                    enx, eny = -enx, -eny

                dist = (xx_p * enx + yy_p * eny).ravel()
                vals = patch.ravel()

                bidx = np.digitize(dist, bins_full) - 1
                ok = (bidx >= 0) & (bidx < n_bins_full)
                esf = np.zeros(n_bins_full, dtype=np.float64)
                cnt = np.zeros(n_bins_full, dtype=np.float64)
                np.add.at(esf, bidx[ok], vals[ok])
                np.add.at(cnt, bidx[ok], 1.0)
                good = cnt > 0
                if good.sum() < 16:
                    continue
                esf[good] /= cnt[good]
                esf = np.interp(np.arange(n_bins_full), np.where(good)[0], esf[good])

                # Find sub-bin edge centre from LSF peak centroid
                lsf = np.diff(esf)
                pk = int(np.argmax(lsf))
                if pk < 4 or pk > len(lsf) - 5 or lsf[pk] <= 1e-4:
                    continue
                w0 = max(0, pk - oversamp * 2)
                w1 = min(len(lsf), pk + oversamp * 2 + 1)
                w_lsf = np.maximum(lsf[w0:w1], 0.0)
                w_sum = float(w_lsf.sum())
                if w_sum <= 1e-8:
                    continue
                lsf_centers = 0.5 * (bin_centers[w0:w1] + bin_centers[w0 + 1:w1 + 1])
                d0 = float(np.dot(w_lsf, lsf_centers) / w_sum)

                # Resample ESF onto centered d_grid
                y_esf = np.interp(d_grid + d0, bin_centers, esf)
                y_mean = float(y_esf.mean())
                y_c = y_esf - y_mean
                y_var = float(np.dot(y_c, y_c))
                if y_var <= 1e-10:
                    continue

                # Closed-form OLS fit for all candidate sigmas simultaneously
                dots = m_c @ y_c
                a_hat = dots / m_norm2
                valid_m = a_hat > 0.0
                if not np.any(valid_m):
                    continue
                sse = y_var - np.where(valid_m, a_hat * dots, 0.0)
                best_i = int(np.argmin(sse))
                r2 = 1.0 - float(sse[best_i]) / y_var
                if r2 < 0.90:
                    continue

                s_est = float(cand_sigmas[best_i])
                tile_vals.append(s_est)
                all_sigmas.append(s_est)

            if tile_vals:
                tile_sigmas[ty, tx] = float(np.median(tile_vals))

    return all_sigmas, tile_sigmas


def estimate_psf_sigma_edges(
    img: np.ndarray,
    ratio: float,
    default_sigma_lr: float = DEFAULT_OPTICS_SIGMA_LR,
    min_sigma_lr: float = 0.15,
    max_sigma_lr: float = 1.25,
    tile_grid: Tuple[int, int] = (4, 4),
) -> PSFEstimate:
    """Estimate optical PSF sigma (in LR px) per channel and per spatial tile.

    Uses structure-tensor coherence to select strong straight edges, fits the
    exact forward-model ESF (sensor box * optics Gaussian), and aggregates
    robustly (lower quartile to avoid out-of-focus background edges). Falls
    back to ``default_sigma_lr`` when fewer than 2 clean edges are found.
    """
    if img.ndim == 2:
        sigmas, tiles = _estimate_channel_psf_tiles(
            img.astype(np.float32, copy=False),
            ratio=ratio,
            tile_grid=tile_grid,
            min_sigma_lr=min_sigma_lr,
            max_sigma_lr=max_sigma_lr,
        )
        if len(sigmas) < 2:
            s = float(default_sigma_lr)
            return PSFEstimate(s, (s, s, s), tiles, len(sigmas), True)
        s = float(np.clip(np.percentile(sigmas, 30), min_sigma_lr, max_sigma_lr))
        return PSFEstimate(s, (s, s, s), tiles, len(sigmas), False)

    ch_sigmas: list[float] = []
    luma_tiles = np.full(tile_grid, np.nan, dtype=np.float32)
    total_edges = 0
    for c in range(3):
        s_list, tiles = _estimate_channel_psf_tiles(
            img[..., c].astype(np.float32, copy=False),
            ratio=ratio,
            tile_grid=tile_grid,
            min_sigma_lr=min_sigma_lr,
            max_sigma_lr=max_sigma_lr,
        )
        total_edges += len(s_list)
        if c == 1:
            luma_tiles = tiles
        if len(s_list) >= 2:
            ch_sigmas.append(float(np.clip(np.percentile(s_list, 30), min_sigma_lr, max_sigma_lr)))
        else:
            ch_sigmas.append(float("nan"))

    valid = [s for s in ch_sigmas if np.isfinite(s)]
    if not valid:
        s = float(default_sigma_lr)
        return PSFEstimate(s, (s, s, s), luma_tiles, total_edges, True)

    # Prefer green channel (highest SNR / spatial density), or Rec.709 weighted average
    if all(np.isfinite(ch_sigmas)):
        s_global = float(0.2126 * ch_sigmas[0] + 0.7152 * ch_sigmas[1] + 0.0722 * ch_sigmas[2])
    else:
        s_global = float(np.median(valid))
    s_rgb = tuple(float(s) if np.isfinite(s) else s_global for s in ch_sigmas)
    return PSFEstimate(
        sigma_lr=float(np.clip(s_global, min_sigma_lr, max_sigma_lr)),
        sigma_rgb=(s_rgb[0], s_rgb[1], s_rgb[2]),
        tile_sigmas=luma_tiles,
        n_edges=total_edges,
        used_fallback=False,
    )


# --------------------------------------------------------------------------- #
# WP6: Scene-Adaptive Mask & Safety Check
# --------------------------------------------------------------------------- #

def compute_scene_adaptive_mask(
    luma: np.ndarray,
    noise_sigma: float,
    highlight_clip_knee: float = 0.80,
    shadow_protect: float = 0.03,
) -> np.ndarray:
    """Build a [0, 1] local enhancement mask:
    - 0 in flat areas (below ~3*noise_sigma gradient) so flat noise never grows
    - 0 on near-clipped specular highlights (protects glass/metal reflections)
    - 1 on well-exposed edges and texture
    """
    obs = luma.astype(np.float32, copy=False)
    gx = cv2.Sobel(obs, cv2.CV_32F, 1, 0, ksize=3) * np.float32(0.125)
    gy = cv2.Sobel(obs, cv2.CV_32F, 0, 1, ksize=3) * np.float32(0.125)
    gmag = np.hypot(gx, gy)

    # Flat-area suppression: ramp 0 -> 1 between 1.5*sigma and 4.5*sigma
    g_lo = np.float32(max(1.5 * noise_sigma, 1e-5))
    g_hi = np.float32(max(4.5 * noise_sigma, 3e-5))
    edge_w = np.clip((gmag - g_lo) / np.maximum(g_hi - g_lo, np.float32(1e-6)), np.float32(0.0), np.float32(1.0))
    edge_w = cv2.GaussianBlur(edge_w, (5, 5), 1.0, borderType=cv2.BORDER_REPLICATE)

    # Specular / near-clipped highlight protection using local maximum so
    # ringing cannot bleed out of clipped glass/specular edges
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    local_max = cv2.dilate(obs, se, borderType=cv2.BORDER_REPLICATE)
    knee = np.float32(np.clip(highlight_clip_knee, 0.5, 0.98))
    hi_w = np.clip((np.float32(0.96) - local_max) / np.maximum(np.float32(0.96) - knee, np.float32(0.02)), np.float32(0.0), np.float32(1.0))

    # Shadow protection
    if shadow_protect > 0.0:
        sh_w = np.clip(obs / np.float32(shadow_protect), np.float32(0.0), np.float32(1.0))
    else:
        sh_w = np.ones_like(obs)

    return (edge_w * hi_w * sh_w).astype(np.float32, copy=False)


def _find_best_edge_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Find (x0, y0, win) of the strongest vertical-ish edge patch in gray."""
    H, W = gray.shape
    win = min(160, H // 2, W // 2)
    if win < 32:
        return None
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    score = np.maximum(gx - np.float32(0.5) * gy, np.float32(0.0))
    margin = min(win, H // 6, W // 6)
    best = (0.0, W // 2 - win // 2, H // 2 - win // 2)
    step = max(32, win // 4)
    for y in range(margin, max(margin + 1, H - margin - win), step):
        for x in range(margin, max(margin + 1, W - margin - win), step):
            s = float(score[y:y + win, x:x + win].mean())
            if s > best[0]:
                best = (s, x, y)
    if best[0] < 1e-4:
        return None
    return (best[1], best[2], win)


def _quick_edge_mtf50(
    gray: np.ndarray,
    ratio: float,
    roi: Optional[Tuple[int, int, int]] = None,
) -> Optional[float]:
    """Fast slanted-edge MTF50 on a given (x0, y0, win) or strongest edge patch."""
    if roi is None:
        roi = _find_best_edge_roi(gray)
    if roi is None:
        return None
    x0, y0, win = roi
    patch = gray[y0:y0 + win, x0:x0 + win].astype(np.float64)
    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)
    dgx = np.diff(patch, axis=1)
    w_g = dgx * dgx
    row_sum = w_g.sum(axis=1)
    valid = row_sum > np.percentile(row_sum, 20)
    if valid.sum() < 8:
        return None
    xs_g = np.arange(pw - 1, dtype=np.float64) + 0.5
    centers = (w_g[valid] @ xs_g) / np.maximum(row_sum[valid], 1e-12)
    ys_v = np.arange(ph, dtype=np.float64)[valid]
    a, b = np.polyfit(ys_v, centers, 1)
    norm = np.sqrt(1.0 + a * a)
    dist = ((xx - (a * yy + b)) / norm).ravel()
    vals = patch.ravel()

    oversamp = 4
    bin_w = 1.0 / oversamp
    d_max = min(16.0 * ratio, pw * 0.4)
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
        return None
    esf[good] /= cnt[good]
    esf = np.interp(np.arange(n_bins), np.where(good)[0], esf[good])
    lsf = np.diff(esf)
    pk = int(np.argmax(np.abs(lsf)))
    win_half = min(pk, len(lsf) - 1 - pk, int(8 * ratio * oversamp))
    if win_half < 8:
        return None
    lsf_w = lsf[pk - win_half:pk + win_half + 1] * np.hamming(2 * win_half + 1)
    n_fft = max(1024, 1 << int(np.ceil(np.log2(len(lsf_w) * 4))))
    mtf = np.abs(np.fft.rfft(lsf_w, n=n_fft))
    if mtf[0] <= 1e-12:
        return None
    mtf /= mtf[0]
    freqs_lr = np.fft.rfftfreq(n_fft, d=bin_w) * float(ratio)
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0:
        return float(freqs_lr[-1])
    i1 = int(below[0])
    if i1 == 0:
        return 0.0
    i0 = i1 - 1
    t = (0.5 - mtf[i0]) / (mtf[i1] - mtf[i0] + 1e-12)
    return float(freqs_lr[i0] + t * (freqs_lr[i1] - freqs_lr[i0]))


def _find_flat_roi(gray: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """Find (x0, y0, win) of the quietest interior patch in gray."""
    H, W = gray.shape
    win = min(96, H // 3, W // 3)
    if win < 16:
        return None
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    margin = min(win, H // 6, W // 6)
    best = (float("inf"), W // 2 - win // 2, H // 2 - win // 2)
    step = max(24, win // 2)
    for y in range(margin, max(margin + 1, H - margin - win), step):
        for x in range(margin, max(margin + 1, W - margin - win), step):
            patch = gray[y:y + win, x:x + win]
            if float(patch.mean()) < 0.02 or float(patch.max()) > 0.95:
                continue
            s = float(gmag[y:y + win, x:x + win].mean())
            if s < best[0]:
                best = (s, x, y)
    return (best[1], best[2], win)


def _quick_flat_noise(
    gray: np.ndarray,
    roi: Optional[Tuple[int, int, int]] = None,
) -> float:
    """Measure Immerkaer noise sigma on a given (x0, y0, win) or quietest patch."""
    if roi is None:
        roi = _find_flat_roi(gray)
    if roi is None:
        return estimate_noise_sigma(gray)
    x0, y0, win = roi
    return estimate_noise_sigma(gray[y0:y0 + win, x0:x0 + win])


def evaluate_enhance_safety(
    luma_orig: np.ndarray,
    luma_enh: np.ndarray,
    ratio: float,
    max_noise_ratio: float = 1.05,
) -> Tuple[bool, Dict[str, float]]:
    """Check whether enhanced luma degrades edge MTF50 or raises flat noise > 5%.

    Both images are measured on the exact same edge ROI and flat ROI chosen on
    ``luma_orig``.
    """
    edge_roi = _find_best_edge_roi(luma_orig)
    flat_roi = _find_flat_roi(luma_orig)
    mtf_orig = _quick_edge_mtf50(luma_orig, ratio, roi=edge_roi)
    mtf_enh = _quick_edge_mtf50(luma_enh, ratio, roi=edge_roi)
    noise_orig = _quick_flat_noise(luma_orig, roi=flat_roi)
    noise_enh = _quick_flat_noise(luma_enh, roi=flat_roi)

    noise_ratio = noise_enh / max(noise_orig, 1e-8)
    mtf_ok = True
    if mtf_orig is not None and mtf_enh is not None and mtf_orig > 1e-3:
        mtf_ok = mtf_enh >= mtf_orig * 0.985
    noise_ok = noise_ratio <= max_noise_ratio

    return (mtf_ok and noise_ok), {
        "safety_mtf_orig": float(mtf_orig) if mtf_orig is not None else -1.0,
        "safety_mtf_enh": float(mtf_enh) if mtf_enh is not None else -1.0,
        "safety_noise_ratio": float(noise_ratio),
    }


# --------------------------------------------------------------------------- #
# TV-Regularized Richardson-Lucy with Deringing
# --------------------------------------------------------------------------- #

def _tv_divergence(u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Compute div(grad(u) / sqrt(|grad(u)|^2 + eps^2)) in float32."""
    gx = np.empty_like(u)
    gy = np.empty_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gx[:, -1] = 0.0
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    gy[-1, :] = 0.0

    norm = np.sqrt(gx * gx + gy * gy + np.float32(eps * eps))
    gx /= norm
    gy /= norm

    div = np.empty_like(u)
    div[:, 0] = gx[:, 0]
    div[:, 1:] = gx[:, 1:] - gx[:, :-1]
    div[0, :] += gy[0, :]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    return div


def richardson_lucy_tv(
    luma: np.ndarray,
    psf_1d: np.ndarray,
    max_iters: int = 10,
    tv_weight: float = 0.002,
    noise_sigma: Optional[float] = None,
    discrepancy_tau: float = 1.05,
    dering_tolerance: float = 0.04,
    dering_radius: int = 2,
    max_overshoot_frac: Optional[float] = None,
    use_gpu: Optional[bool] = None,
) -> Tuple[np.ndarray, int, str]:
    """TV-regularized Richardson-Lucy deconvolution on linear luminance.

    Args:
        luma: 2D float32 linear luminance in [0, 1].
        psf_1d: Separable symmetric 1D PSF kernel summing to 1.0.
        max_iters: Maximum number of RL-TV iterations.
        tv_weight: Total-Variation regularization weight (0 = standard RL).
        noise_sigma: Expected noise std of ``luma`` for discrepancy-principle
            early stopping (None disables early stopping).
        discrepancy_tau: Stopping multiplier on ``noise_sigma``.
        dering_tolerance: Overshoot tolerance as a fraction of local min/max
            range. Set <= 0 to disable deringing.
        dering_radius: Local support radius (HR px) for min/max deringing.
        max_overshoot_frac: Optional ringing monitor; if 99.5th-percentile
            local overshoot exceeds this fraction of local range, stop early.
        use_gpu: Optional GPU override (None=auto, True=GPU with CPU fallback,
            False=CPU).

    Returns:
        ``(deconvolved_luma, iters_run, stop_reason)``.
    """
    if use_gpu is not False:
        from .gpu import resolve_use_gpu, richardson_lucy_tv_gpu
        if resolve_use_gpu(use_gpu, "enhance", int(luma.size)):
            gpu_res = richardson_lucy_tv_gpu(
                luma,
                psf_1d=psf_1d,
                max_iters=max_iters,
                tv_weight=tv_weight,
                noise_sigma=noise_sigma,
                discrepancy_tau=discrepancy_tau,
                dering_tolerance=dering_tolerance,
                dering_radius=dering_radius,
                max_overshoot_frac=max_overshoot_frac,
            )
            if gpu_res is not None:
                return gpu_res

    eps = np.float32(1e-5)
    obs = np.clip(luma.astype(np.float32, copy=False), eps, np.float32(1.0))
    est = obs.copy()
    k = psf_1d.astype(np.float32, copy=False)
    tv_w = np.float32(max(tv_weight, 0.0))

    local_min = local_max = local_range = None
    if (dering_tolerance > 0.0 and dering_radius > 0) or (max_overshoot_frac is not None and max_overshoot_frac > 0.0):
        ksz = 2 * max(int(dering_radius), 1) + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        local_min = cv2.erode(obs, se, borderType=cv2.BORDER_REPLICATE)
        local_max = cv2.dilate(obs, se, borderType=cv2.BORDER_REPLICATE)
        local_range = np.maximum(local_max - local_min, np.float32(1e-4))

    iters_run = 0
    stop_reason = "max_iters"
    prev_rms = float("inf")

    for it in range(1, max_iters + 1):
        conv = cv2.sepFilter2D(est, cv2.CV_32F, k, k, borderType=cv2.BORDER_REPLICATE)

        # Noise-aware stopping (Morozov discrepancy principle)
        if noise_sigma is not None and noise_sigma > 0.0 and it > 1:
            diff = obs - conv
            rms = float(np.sqrt(np.mean(diff * diff, dtype=np.float64)))
            if rms <= discrepancy_tau * noise_sigma:
                stop_reason = "discrepancy"
                break
            if it >= 3 and (prev_rms - rms) < 1e-4 * prev_rms:
                stop_reason = "converged"
                break
            prev_rms = rms

        rel = obs / np.maximum(conv, eps)
        corr = cv2.sepFilter2D(rel, cv2.CV_32F, k, k, borderType=cv2.BORDER_REPLICATE)

        if tv_w > 0.0:
            div = _tv_divergence(est)
            denom = np.maximum(np.float32(1.0) - tv_w * div, np.float32(0.5))
            est_next = est * (corr / denom)
        else:
            est_next = est * corr

        np.clip(est_next, np.float32(0.0), np.float32(1.0), out=est_next)

        # Optional overshoot / ringing monitor
        if max_overshoot_frac is not None and max_overshoot_frac > 0.0 and it >= 3 and local_max is not None:
            over = np.maximum(est_next - local_max, local_min - est_next)
            strong = local_range > np.float32(0.05)
            if np.any(strong):
                ov_frac = float(np.percentile((over[strong] / local_range[strong]), 98.0))
                if ov_frac > max_overshoot_frac:
                    stop_reason = "overshoot_monitor"
                    break

        est = est_next
        iters_run = it

    # Local-support deringing: clamp to [local_min - tol*range, local_max + tol*range]
    if dering_tolerance > 0.0 and dering_radius > 0 and iters_run > 0 and local_min is not None:
        tol = np.float32(dering_tolerance)
        lo = np.maximum(np.float32(0.0), local_min - tol * (local_max - local_min))
        hi = np.minimum(np.float32(1.0), local_max + tol * (local_max - local_min))
        np.clip(est, lo, hi, out=est)

    # Edge-aware blending: suppress RL noise amplification in flat regions
    if noise_sigma is not None and noise_sigma > 0.0 and iters_run > 0:
        gx = cv2.Sobel(obs, cv2.CV_32F, 1, 0, ksize=3) * np.float32(0.125)
        gy = cv2.Sobel(obs, cv2.CV_32F, 0, 1, ksize=3) * np.float32(0.125)
        gmag = np.hypot(gx, gy)
        # Smoothly ramp from 0 in flat noise to 1 on real structure (>= 3*sigma)
        g_thr = np.float32(max(2.5 * noise_sigma, 1e-5))
        weight = np.clip((gmag - np.float32(0.5) * g_thr) / g_thr, np.float32(0.0), np.float32(1.0))
        weight = cv2.GaussianBlur(weight, (5, 5), 1.0, borderType=cv2.BORDER_REPLICATE)
        est = obs + weight * (est - obs)

    return est, iters_run, stop_reason


# --------------------------------------------------------------------------- #
# A-Trous Starlet (B3 Spline) Multiscale Sharpening
# --------------------------------------------------------------------------- #

def _dilated_b3_kernel(scale: int) -> np.ndarray:
    """Return 1D B3 spline kernel dilated by 2^scale."""
    step = 1 << scale
    if step == 1:
        return _B3_KERNEL
    k = np.zeros(4 * step + 1, dtype=np.float32)
    k[0::step] = _B3_KERNEL
    return k


def starlet_decompose(
    img: np.ndarray,
    n_scales: int = 3,
) -> Tuple[list[np.ndarray], np.ndarray]:
    """Undecimated a-trous B3-spline starlet decomposition.

    Returns ``(details, residual)`` such that
    ``sum(details) + residual == img`` up to float32 rounding.
    """
    c = img.astype(np.float32, copy=True)
    details: list[np.ndarray] = []
    for j in range(n_scales):
        k = _dilated_b3_kernel(j)
        c_next = cv2.sepFilter2D(c, cv2.CV_32F, k, k, borderType=cv2.BORDER_REPLICATE)
        details.append(c - c_next)
        c = c_next
    return details, c


def starlet_sharpen(
    luma: np.ndarray,
    gains: Sequence[float],
    k_sigma: float = 3.0,
    noise_sigma: Optional[float] = None,
    shadow_protect: float = 0.03,
    highlight_protect: float = 0.06,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """Multiscale a-trous starlet sharpening with MAD k-sigma noise gating.

    When all ``gains`` are 1.0, returns an exact reconstruction of ``luma``.
    """
    n_scales = len(gains)
    if n_scales == 0 or all(abs(g - 1.0) < 1e-6 for g in gains):
        return luma.astype(np.float32, copy=True)

    if use_gpu is not False:
        from .gpu import resolve_use_gpu, starlet_sharpen_gpu
        if resolve_use_gpu(use_gpu, "enhance", int(luma.size)):
            gpu_out = starlet_sharpen_gpu(
                luma,
                gains=gains,
                k_sigma=k_sigma,
                noise_sigma=noise_sigma,
                shadow_protect=shadow_protect,
                highlight_protect=highlight_protect,
            )
            if gpu_out is not None:
                return gpu_out

    details, residual = starlet_decompose(luma, n_scales=n_scales)

    # Highlight & shadow protection mask
    prot = np.ones_like(luma, dtype=np.float32)
    if shadow_protect > 0.0:
        prot *= np.clip(luma / np.float32(shadow_protect), np.float32(0.0), np.float32(1.0))
    if highlight_protect > 0.0:
        prot *= np.clip(
            (np.float32(1.0) - luma) / np.float32(highlight_protect),
            np.float32(0.0),
            np.float32(1.0),
        )

    out = residual.copy()
    for j, (w, g) in enumerate(zip(details, gains)):
        if abs(g - 1.0) < 1e-6:
            out += w
            continue
        # Per-layer robust noise estimate (MAD), floored by theoretical noise
        mad_sigma = float(np.median(np.abs(w))) * 1.4826
        if noise_sigma is not None and noise_sigma > 0.0:
            nf = _STARLET_NOISE_FACTORS[min(j, len(_STARLET_NOISE_FACTORS) - 1)]
            layer_sigma = max(mad_sigma, float(noise_sigma) * nf)
        else:
            layer_sigma = mad_sigma

        thr = np.float32(max(k_sigma * layer_sigma, 0.0))
        if thr > 0.0:
            abs_w = np.abs(w)
            excess = np.sign(w) * np.maximum(abs_w - thr, np.float32(0.0))
        else:
            excess = w

        out += w + np.float32(g - 1.0) * prot * excess

    return np.clip(out, np.float32(0.0), np.float32(1.0))


# --------------------------------------------------------------------------- #
# Main Entry Point & Fairness Helper
# --------------------------------------------------------------------------- #

def enhance_image(
    linear_rgb: np.ndarray,
    ratio: float = 2.0,
    noise_sigma: Optional[float] = None,
    config: Optional[EnhanceConfig] = None,
    strength: Optional[float] = None,
    preset: Optional[str] = None,
    mode: Optional[str] = None,
    use_gpu: Optional[bool] = None,
) -> EnhanceResult:
    """Enhance a linear HR RGB image using classical deconvolution + starlets.

    Works on ANY linear HR RGB image (current SR, WP1 CFA fusion, or single
    bicubic baseline) so it composes cleanly with any upstream fusion engine.

    Args:
        linear_rgb: (H, W, 3) or (H, W) float32 linear image in [0, 1].
        ratio: Super-resolution scale factor (>= 1.0).
        noise_sigma: Optional linear noise std of the image. If None, estimated
            directly from the image via Immerkaer MAD.
        config: Optional :class:`EnhanceConfig`.
        strength: Optional override for ``config.strength`` in [0, 1].
        preset: Optional preset name (``off``, ``subtle``, ``balanced``, ``astro``).
        mode: Optional override for ``config.mode`` (``wp2`` or ``adaptive``).

    Returns:
        :class:`EnhanceResult` containing the enhanced linear RGB image and
        diagnostics.
    """
    if config is None:
        if preset is not None:
            cfg = EnhanceConfig.from_preset(preset)
        else:
            cfg = EnhanceConfig()
    else:
        cfg = config

    eff_mode = (mode if mode is not None else cfg.mode).strip().lower()
    s = float(np.clip(strength if strength is not None else cfg.strength, 0.0, 1.0))
    sigma_n = float(noise_sigma) if (noise_sigma is not None and noise_sigma > 0) else estimate_noise_sigma(linear_rgb)

    if s <= 1e-6:
        psf_1d = build_psf_1d(ratio, cfg.optics_sigma_lr or DEFAULT_OPTICS_SIGMA_LR)
        return EnhanceResult(
            image=np.clip(linear_rgb, 0.0, 1.0).astype(np.float32, copy=True),
            optics_sigma_lr=float(cfg.optics_sigma_lr or DEFAULT_OPTICS_SIGMA_LR),
            noise_sigma=sigma_n,
            rl_iterations_run=0,
            rl_stopped_reason="strength_zero",
            strength=0.0,
            psf_1d=psf_1d,
        )

    luma_orig = rgb_to_luma(linear_rgb)
    extra_details: Dict[str, float] = {}

    # 1. PSF estimation & construction
    if cfg.optics_sigma_lr is not None:
        optics_sigma_lr = float(cfg.optics_sigma_lr)
    elif eff_mode == "adaptive":
        psf_est = estimate_psf_sigma_edges(
            linear_rgb, ratio=ratio, tile_grid=cfg.psf_tile_grid
        )
        optics_sigma_lr = psf_est.sigma_lr
        extra_details["psf_sigma_r"] = psf_est.sigma_rgb[0]
        extra_details["psf_sigma_g"] = psf_est.sigma_rgb[1]
        extra_details["psf_sigma_b"] = psf_est.sigma_rgb[2]
        extra_details["psf_n_edges"] = float(psf_est.n_edges)
        extra_details["psf_used_fallback"] = 1.0 if psf_est.used_fallback else 0.0
    else:
        optics_sigma_lr = estimate_optics_sigma(luma_orig, ratio)
    psf_1d = build_psf_1d(ratio, optics_sigma_lr)

    # 2. Map single strength control (0..1) -> RL iterations, TV weight, starlet gains
    max_iters = cfg.max_rl_iters if cfg.max_rl_iters is not None else max(2, int(round(2 + 12 * s)))
    if cfg.tv_weight is not None:
        tv_weight = float(cfg.tv_weight)
    else:
        # Scale TV regularization with noise level so noisier inputs stay clean
        tv_weight = float(np.clip(0.001 + 1.5 * sigma_n, 0.001, 0.025))

    want_gpu = False
    if use_gpu is not False:
        from .gpu import resolve_use_gpu
        want_gpu = resolve_use_gpu(use_gpu, "enhance", int(luma_orig.size))

    luma_rl, iters_run, stop_reason = richardson_lucy_tv(
        luma_orig,
        psf_1d=psf_1d,
        max_iters=max_iters,
        tv_weight=tv_weight,
        noise_sigma=sigma_n,
        dering_tolerance=cfg.dering_tolerance,
        dering_radius=max(cfg.dering_radius, int(round(ratio * 0.75))),
        max_overshoot_frac=0.18 if eff_mode == "adaptive" else None,
        use_gpu=want_gpu,
    )

    # 3. Multiscale a-trous starlet sharpening
    if cfg.starlet_gains is not None:
        gains = tuple(cfg.starlet_gains)
    else:
        # Gentle micro-contrast boost on fine & medium scales, scaled by strength
        g1 = 1.0 + 0.35 * s
        g2 = 1.0 + 0.25 * s
        g3 = 1.0 + 0.12 * s
        gains = (g1, g2, g3)[: cfg.starlet_scales]

    luma_enh = starlet_sharpen(
        luma_rl,
        gains=gains,
        k_sigma=cfg.starlet_k_sigma,
        noise_sigma=sigma_n,
        shadow_protect=cfg.shadow_protect,
        highlight_protect=cfg.highlight_protect,
        use_gpu=want_gpu,
    )
    extra_details["gpu_active"] = 1.0 if want_gpu else 0.0

    # WP6: Scene-adaptive local mask (protect flat areas & specular highlights)
    if eff_mode == "adaptive":
        scene_mask = compute_scene_adaptive_mask(
            luma_orig,
            noise_sigma=sigma_n,
            highlight_clip_knee=cfg.highlight_clip_knee,
            shadow_protect=cfg.shadow_protect,
        )
        luma_enh = luma_orig + scene_mask * (luma_enh - luma_orig)

    # Final local-support deringing clamp against luma_orig so combined
    # RL + starlet sharpening cannot exceed the overshoot bound.
    if cfg.dering_tolerance > 0.0 and cfg.dering_radius > 0:
        dr = max(cfg.dering_radius, int(round(ratio * 0.75)))
        ksz = 2 * int(dr) + 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        local_min = cv2.erode(luma_orig, se, borderType=cv2.BORDER_REPLICATE)
        local_max = cv2.dilate(luma_orig, se, borderType=cv2.BORDER_REPLICATE)
        local_range = local_max - local_min
        tol = np.float32(cfg.dering_tolerance)
        lo = np.maximum(np.float32(0.0), local_min - tol * local_range)
        hi = np.minimum(np.float32(1.0), local_max + tol * local_range)
        np.clip(luma_enh, lo, hi, out=luma_enh)

    # WP6: Automatic safety check & back-off if MTF50 drops or flat noise > +5%
    backoff_factor = 1.0
    if eff_mode == "adaptive" and cfg.safety_check:
        ok, safety_info = evaluate_enhance_safety(
            luma_orig, luma_enh, ratio=ratio, max_noise_ratio=cfg.safety_max_noise_ratio
        )
        if not ok:
            delta = luma_enh - luma_orig
            for factor in (0.5, 0.25, 0.0):
                cand = np.clip(luma_orig + np.float32(factor) * delta, np.float32(0.0), np.float32(1.0))
                if factor == 0.0:
                    luma_enh = cand
                    backoff_factor = 0.0
                    _, safety_info = evaluate_enhance_safety(
                        luma_orig, luma_enh, ratio=ratio, max_noise_ratio=cfg.safety_max_noise_ratio
                    )
                    break
                ok_c, safety_info = evaluate_enhance_safety(
                    luma_orig, cand, ratio=ratio, max_noise_ratio=cfg.safety_max_noise_ratio
                )
                if ok_c:
                    luma_enh = cand
                    backoff_factor = factor
                    break
        extra_details.update(safety_info)
        extra_details["safety_backoff_factor"] = backoff_factor

    # 4. Recombine enhanced luminance with original chroma
    rgb_enh = apply_luma_to_rgb(linear_rgb, luma_orig, luma_enh)

    details = {
        "tv_weight": tv_weight,
        "max_iters": float(max_iters),
        "starlet_g1": float(gains[0]) if len(gains) > 0 else 1.0,
        "starlet_g2": float(gains[1]) if len(gains) > 1 else 1.0,
    }
    details.update(extra_details)

    return EnhanceResult(
        image=rgb_enh,
        optics_sigma_lr=optics_sigma_lr,
        noise_sigma=sigma_n,
        rl_iterations_run=iters_run,
        rl_stopped_reason=stop_reason,
        strength=s * backoff_factor,
        psf_1d=psf_1d,
        details=details,
    )


def enhance_fair_pair(
    sr_hr: np.ndarray,
    single_bicubic_hr: np.ndarray,
    ratio: float,
    sr_noise_sigma: Optional[float] = None,
    single_noise_sigma: Optional[float] = None,
    config: Optional[EnhanceConfig] = None,
    strength: Optional[float] = None,
    preset: Optional[str] = None,
    mode: Optional[str] = None,
    use_gpu: Optional[bool] = None,
) -> Tuple[EnhanceResult, EnhanceResult]:
    """Apply the exact same enhancement configuration to SR and single-frame baseline.

    Each image uses its own noise estimate (either passed explicitly or
    estimated from the image itself) so the comparison is strictly fair.
    """
    res_sr = enhance_image(
        sr_hr,
        ratio=ratio,
        noise_sigma=sr_noise_sigma,
        config=config,
        strength=strength,
        preset=preset,
        mode=mode,
        use_gpu=use_gpu,
    )
    res_single = enhance_image(
        single_bicubic_hr,
        ratio=ratio,
        noise_sigma=single_noise_sigma,
        config=config,
        strength=strength,
        preset=preset,
        mode=mode,
        use_gpu=use_gpu,
    )
    return res_sr, res_single
