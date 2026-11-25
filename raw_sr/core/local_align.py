"""Tile-based local residual alignment and per-tile robustness / lucky selection.

Classical (non-neural) local alignment inspired by:
  * Wronski et al. (2019), "Handheld Multi-Frame Super-Resolution" — coarse-to-fine
    tile alignment pyramid + per-tile consistency robustness weights.
  * Astrophotography lucky-imaging stackers (AutoStakkert) — multi-alignment-point
    local shifts + optional per-tile top-X% frame selection.

Coordinate convention (matches ``raw_sr.core.superres``):
  * Global homography ``W`` maps ``ref -> mov`` in LR pixel coordinates.
  * ``aligned(x_ref, y_ref) = mov(W(x_ref, y_ref))`` is rendered via
    ``cv2.warpPerspective(mov, inv(W), (w, h))``.
  * The local residual field ``(dx_ref, dy_ref)`` is in reference LR pixels such
    that ``aligned(x, y) ~= ref(x + dx_ref, y + dy_ref)`` — i.e., a sample at
    ``(x, y)`` in ``aligned`` belongs at ``(x + dx_ref, y + dy_ref)`` in ``ref``.
  * Warping ``(dx_ref, dy_ref)`` into ``mov`` coordinates via
    ``cv2.warpPerspective(field, W, (w, h))`` gives ``(dx_mov, dy_mov)`` indexed
    by ``mov`` LR pixel ``(y_mov, x_mov)``; during HR scatter deposition, adding
    ``(ratio * dx_mov, ratio * dy_mov)`` to ``(U, V)`` places each sample at its
    locally aligned HR coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import time

import cv2
import numpy as np


@dataclass
class LocalAlignResult:
    """Per-frame local residual displacement fields and per-pixel weights."""
    # Per-frame (H, W) float32 displacement in mov LR coords (units: ref LR px).
    dx_mov: List[np.ndarray]
    dy_mov: List[np.ndarray]
    # Per-frame (H, W) float32 per-pixel robustness/selection weight in mov coords.
    weights_mov: List[np.ndarray]
    # Summary diagnostics.
    metrics: dict


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer 3x3 Laplacian MAD noise estimate on a float32 image."""
    g = np.asarray(gray, dtype=np.float32)
    k = np.array([[1.0, -2.0, 1.0],
                  [-2.0, 4.0, -2.0],
                  [1.0, -2.0, 1.0]], dtype=np.float32)
    lap = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
    if lap.shape[0] > 4 and lap.shape[1] > 4:
        lap = lap[2:-2, 2:-2]
    mad = float(np.median(np.abs(lap)))
    # Immerkaer scaling: sigma = sqrt(pi/2) / 6 * mean(|L|), MAD->mean ~ 1.4826 * sqrt(2/pi)
    sigma = mad * 1.4826 / 6.0
    return max(sigma, 1e-5)


def _normalize_h(W: np.ndarray) -> np.ndarray:
    arr = np.asarray(W, dtype=np.float64)
    if arr.shape == (3, 3):
        H = arr
    elif arr.shape == (2, 3):
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = arr
    elif arr.shape == (2,):
        H = np.array([[1.0, 0.0, float(arr[0])],
                      [0.0, 1.0, float(arr[1])],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
    else:
        raise ValueError(f"cannot interpret transform of shape {arr.shape}")
    if abs(H[2, 2]) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return H / H[2, 2]


def _render_in_ref(ref_shape: Tuple[int, int], mov: np.ndarray,
                   W: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        Mi = np.linalg.inv(_normalize_h(W)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None, None
    h, w = int(ref_shape[0]), int(ref_shape[1])
    src = np.asarray(mov, dtype=np.float32)
    aligned = cv2.warpPerspective(src, Mi, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT)
    ones = np.ones((h, w), dtype=np.float32)
    valid = cv2.warpPerspective(ones, Mi, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    return aligned, valid


def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _lk_refine_tile(ref_t: np.ndarray, aligned_t: np.ndarray,
                    gx_t: np.ndarray, gy_t: np.ndarray,
                    dx0: float, dy0: float, max_shift: float,
                    iters: int = 3) -> Tuple[float, float, float]:
    """Sub-pixel Lucas-Kanade refinement on one tile.

    Finds ``(dx, dy)`` minimizing ``sum((aligned(x - dx, y - dy) - ref(x, y))^2)``
    (equivalently ``aligned(x, y) ~= ref(x + dx, y + dy)``).
    Returns ``(dx, dy, rms_after)``.
    """
    th, tw = ref_t.shape
    # Interior crop avoids border artifacts during sub-pixel shift.
    m = 2 if min(th, tw) >= 16 else 1
    ref_in = ref_t[m:th - m, m:tw - m]
    gx = gx_t[m:th - m, m:tw - m]
    gy = gy_t[m:th - m, m:tw - m]

    h00 = float(np.sum(gx * gx))
    h01 = float(np.sum(gx * gy))
    h11 = float(np.sum(gy * gy))
    tr = h00 + h11
    if tr < 1e-10:
        diff = aligned_t[m:th - m, m:tw - m] - ref_in
        return float(dx0), float(dy0), float(np.sqrt(np.mean(diff * diff)))

    # Small Tikhonov damping stabilizes 1D-edge (aperture-problem) tiles along
    # the edge tangent while keeping the normal shift unbiased.
    reg = 1e-3 * tr
    a00 = h00 + reg
    a11 = h11 + reg
    det = a00 * a11 - h01 * h01
    if det < 1e-14:
        diff = aligned_t[m:th - m, m:tw - m] - ref_in
        return float(dx0), float(dy0), float(np.sqrt(np.mean(diff * diff)))

    dx, dy = float(dx0), float(dy0)
    # Linearize: aligned(x - dx, y - dy) ~= ref(x, y).
    # Using cv2.warpAffine with M = [[1, 0, dx], [0, 1, dy]] computes
    # dst(x, y) = src(x - dx, y - dy).
    for _ in range(iters):
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(aligned_t, M, (tw, th),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
        err = warped[m:th - m, m:tw - m] - ref_in
        # d/d(dx) [aligned(x - dx, y - dy)] = -gx, so Gauss-Newton step is:
        # H * [ddx, ddy]^T = [sum(gx * err), sum(gy * err)]^T
        b0 = float(np.sum(gx * err))
        b1 = float(np.sum(gy * err))
        ddx = (a11 * b0 - h01 * b1) / det
        ddy = (a00 * b1 - h01 * b0) / det
        dx = float(np.clip(dx + ddx, -max_shift, max_shift))
        dy = float(np.clip(dy + ddy, -max_shift, max_shift))
        if ddx * ddx + ddy * ddy < 1e-5:
            break

    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped = cv2.warpAffine(aligned_t, M, (tw, th),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    err = warped[m:th - m, m:tw - m] - ref_in
    rms = float(np.sqrt(np.mean(err * err)))
    return dx, dy, rms


def _estimate_level_tiles(ref: np.ndarray, aligned: np.ndarray,
                          valid: np.ndarray, tile_size: int,
                          max_shift: float, noise_sigma: float,
                          init_dx: Optional[np.ndarray] = None,
                          init_dy: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate tile-grid ``(dx, dy, conf)`` at one pyramid level."""
    h, w = ref.shape
    t = int(min(tile_size, h, w))
    stride = max(8, t // 2)
    ys = _tile_starts(h, t, stride)
    xs = _tile_starts(w, t, stride)
    ny, nx = len(ys), len(xs)

    dx_grid = np.zeros((ny, nx), dtype=np.float32)
    dy_grid = np.zeros((ny, nx), dtype=np.float32)
    conf_grid = np.zeros((ny, nx), dtype=np.float32)

    gx_full = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / 8.0)
    gy_full = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / 8.0)
    win = cv2.createHanningWindow((t, t), cv2.CV_32F)

    # Minimum tile std to be considered textured (above sensor noise).
    min_std = max(1e-4, 2.5 * noise_sigma)

    for iy, y0 in enumerate(ys):
        y1 = y0 + t
        for ix, x0 in enumerate(xs):
            x1 = x0 + t
            if float(valid[y0:y1, x0:x1].mean()) < 0.95:
                continue
            ref_t = ref[y0:y1, x0:x1]
            std_r = float(ref_t.std())
            if std_r < min_std:
                continue
            ali_t = aligned[y0:y1, x0:x1]
            # Gain-normalize tile mean to resist local brightness drift.
            mr = float(ref_t.mean())
            ma = float(ali_t.mean())
            if ma > 1e-5 and mr > 1e-5:
                gain = float(np.clip(mr / ma, 0.85, 1.18))
                ali_n = ali_t * np.float32(gain)
            else:
                ali_n = ali_t

            # Initial shift from coarse level (if provided)
            dx0 = float(init_dx[iy, ix]) if init_dx is not None else 0.0
            dy0 = float(init_dy[iy, ix]) if init_dy is not None else 0.0
            dx0 = float(np.clip(dx0, -max_shift, max_shift))
            dy0 = float(np.clip(dy0, -max_shift, max_shift))

            # Phase correlation residual around (dx0, dy0)
            if abs(dx0) > 1e-3 or abs(dy0) > 1e-3:
                M0 = np.array([[1.0, 0.0, dx0], [0.0, 1.0, dy0]], dtype=np.float32)
                ali_s = cv2.warpAffine(ali_n, M0, (t, t),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
            else:
                ali_s = ali_n

            (pdx, pdy), resp = cv2.phaseCorrelate(
                ref_t.astype(np.float32, copy=True),
                ali_s.astype(np.float32, copy=True),
                win,
            )
            # Sign: phaseCorrelate(ref, mov) returns +shift of mov relative to ref,
            # so deposit correction is (-pdx, -pdy).
            if float(resp) >= 0.08 and np.hypot(pdx, pdy) <= max_shift:
                dx_cand = float(np.clip(dx0 - pdx, -max_shift, max_shift))
                dy_cand = float(np.clip(dy0 - pdy, -max_shift, max_shift))
            else:
                dx_cand, dy_cand = dx0, dy0

            # Lucas-Kanade sub-pixel refinement
            m = 2 if t >= 16 else 1
            diff0 = ali_n[m:t - m, m:t - m] - ref_t[m:t - m, m:t - m]
            rms_before = float(np.sqrt(np.mean(diff0 * diff0)))
            dx_f, dy_f, rms_after = _lk_refine_tile(
                ref_t, ali_n, gx_full[y0:y1, x0:x1], gy_full[y0:y1, x0:x1],
                dx_cand, dy_cand, max_shift=max_shift, iters=3,
            )

            # Require a genuine residual improvement so well-aligned rigid tiles
            # do not get perturbed by sub-pixel estimation noise.
            if rms_after > rms_before * 0.97:
                # Try starting from (0, 0) directly with LK
                dx_z, dy_z, rms_z = _lk_refine_tile(
                    ref_t, ali_n, gx_full[y0:y1, x0:x1], gy_full[y0:y1, x0:x1],
                    0.0, 0.0, max_shift=max_shift, iters=3,
                )
                if rms_z < rms_after and rms_z <= rms_before * 0.97:
                    dx_f, dy_f, rms_after = dx_z, dy_z, rms_z
                else:
                    continue

            if np.hypot(dx_f, dy_f) > max_shift or np.hypot(dx_f, dy_f) < 0.04:
                continue

            # Confidence from texture SNR and residual quality
            tex_conf = float(np.clip((std_r - min_std) / (3.0 * min_std), 0.0, 1.0))
            res_conf = float(np.clip(1.0 - rms_after / max(std_r, 1e-5), 0.0, 1.0))
            conf = tex_conf * max(res_conf, float(np.clip(resp, 0.0, 1.0)))
            if conf > 0.05:
                dx_grid[iy, ix] = dx_f
                dy_grid[iy, ix] = dy_f
                conf_grid[iy, ix] = conf

    return dx_grid, dy_grid, conf_grid


def _upsample_tile_grid(grid: np.ndarray, ys: List[int], xs: List[int],
                        tile_size: int, h: int, w: int) -> np.ndarray:
    """Bilinearly interpolate a ``(ny, nx)`` tile-center grid to ``(h, w)``."""
    ny, nx = grid.shape
    half = 0.5 * (tile_size - 1)
    yc = np.array([y0 + half for y0 in ys], dtype=np.float32)
    xc = np.array([x0 + half for x0 in xs], dtype=np.float32)
    # Map pixel coordinates (0..h-1, 0..w-1) into fractional tile-grid indices
    yi = np.interp(np.arange(h, dtype=np.float32), yc, np.arange(ny, dtype=np.float32))
    xi = np.interp(np.arange(w, dtype=np.float32), xc, np.arange(nx, dtype=np.float32))
    map_x, map_y = np.meshgrid(xi.astype(np.float32), yi.astype(np.float32))
    return cv2.remap(grid.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def estimate_local_displacement(ref_green: np.ndarray,
                                mov_green: np.ndarray,
                                W: np.ndarray,
                                tile_size: int = 48,
                                max_shift: float = 2.0,
                                noise_sigma: Optional[float] = None
                                ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Estimate a smooth dense residual displacement field ``(dx_ref, dy_ref)``.

    Args:
        ref_green: ``(H, W)`` float32 reference green/luma channel.
        mov_green: ``(H, W)`` float32 moving frame green/luma channel.
        W: 3x3 global homography ``ref -> mov`` in LR pixels.
        tile_size: fine-level tile size in LR pixels (50% overlap).
        max_shift: maximum allowed residual displacement in LR pixels.
        noise_sigma: optional noise sigma estimate on ``ref_green``.

    Returns:
        ``(dx_ref, dy_ref, info)`` where ``dx_ref, dy_ref`` are ``(H, W)``
        float32 arrays in reference LR coordinates.
    """
    ref = np.asarray(ref_green, dtype=np.float32)
    h, w = ref.shape
    aligned, valid = _render_in_ref((h, w), mov_green, W)
    if aligned is None:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z.copy(), {"valid_tile_frac": 0.0, "rms_lr": 0.0, "max_lr": 0.0}

    if noise_sigma is None:
        noise_sigma = _estimate_noise_sigma(ref)

    t_fine = int(max(16, min(tile_size, min(h, w))))
    init_dx = init_dy = None

    # Coarse level (2x downsampled) when frame is large enough
    if min(h, w) >= 2 * t_fine:
        hc, wc = h // 2, w // 2
        ref_c = cv2.resize(ref, (wc, hc), interpolation=cv2.INTER_AREA)
        ali_c = cv2.resize(aligned, (wc, hc), interpolation=cv2.INTER_AREA)
        val_c = cv2.resize(valid, (wc, hc), interpolation=cv2.INTER_AREA)
        dx_c, dy_c, conf_c = _estimate_level_tiles(
            ref_c, ali_c, val_c, tile_size=t_fine,
            max_shift=max_shift * 0.5, noise_sigma=noise_sigma * 0.5,
        )
        # Smooth coarse field before upsampling to fine tile grid
        if conf_c.size > 0 and float(conf_c.max()) > 0.0:
            k3 = np.ones((3, 3), dtype=np.float32) / 9.0
            c_s = cv2.filter2D(conf_c, cv2.CV_32F, k3, borderType=cv2.BORDER_REPLICATE)
            dx_c = cv2.filter2D(dx_c * conf_c, cv2.CV_32F, k3,
                                borderType=cv2.BORDER_REPLICATE) / np.maximum(c_s + 0.1, 1e-6)
            dy_c = cv2.filter2D(dy_c * conf_c, cv2.CV_32F, k3,
                                borderType=cv2.BORDER_REPLICATE) / np.maximum(c_s + 0.1, 1e-6)
            stride_f = max(8, t_fine // 2)
            ny_f = len(_tile_starts(h, t_fine, stride_f))
            nx_f = len(_tile_starts(w, t_fine, stride_f))
            init_dx = cv2.resize(dx_c * 2.0, (nx_f, ny_f), interpolation=cv2.INTER_LINEAR)
            init_dy = cv2.resize(dy_c * 2.0, (nx_f, ny_f), interpolation=cv2.INTER_LINEAR)

    # Fine level
    dx_g, dy_g, conf_g = _estimate_level_tiles(
        ref, aligned, valid, tile_size=t_fine,
        max_shift=max_shift, noise_sigma=noise_sigma,
        init_dx=init_dx, init_dy=init_dy,
    )

    valid_frac = float((conf_g > 0.05).mean()) if conf_g.size else 0.0

    # Confidence-weighted spatial smoothing on tile grid; low-confidence /
    # textureless tiles are pulled smoothly back to 0 (global homography),
    # while confident tiles have zero shrinkage bias.
    if conf_g.size > 0 and float(conf_g.max()) > 0.0:
        ker = np.array([[1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]], dtype=np.float32) / 16.0
        c_sm = cv2.filter2D(conf_g, cv2.CV_32F, ker, borderType=cv2.BORDER_REPLICATE)
        pull = np.clip(c_sm / 0.20, 0.0, 1.0)
        denom = np.maximum(c_sm, 1e-6)
        dx_sm = (cv2.filter2D(dx_g * conf_g, cv2.CV_32F, ker,
                              borderType=cv2.BORDER_REPLICATE) / denom) * pull
        dy_sm = (cv2.filter2D(dy_g * conf_g, cv2.CV_32F, ker,
                              borderType=cv2.BORDER_REPLICATE) / denom) * pull
    else:
        dx_sm = np.zeros_like(dx_g)
        dy_sm = np.zeros_like(dy_g)

    # Clamp magnitude
    mag = np.hypot(dx_sm, dy_sm)
    over = mag > max_shift
    if np.any(over):
        scale = np.float32(max_shift) / np.maximum(mag[over], 1e-6)
        dx_sm[over] *= scale
        dy_sm[over] *= scale

    # Upsample tile grid to full LR resolution smoothly at true tile centers
    stride_f = max(8, t_fine // 2)
    ys_f = _tile_starts(h, t_fine, stride_f)
    xs_f = _tile_starts(w, t_fine, stride_f)
    dx_ref = _upsample_tile_grid(dx_sm, ys_f, xs_f, t_fine, h, w)
    dy_ref = _upsample_tile_grid(dy_sm, ys_f, xs_f, t_fine, h, w)

    mag_full = np.hypot(dx_ref, dy_ref)
    info = {
        "valid_tile_frac": valid_frac,
        "rms_lr": float(np.sqrt(np.mean(mag_full * mag_full))),
        "max_lr": float(mag_full.max()) if mag_full.size else 0.0,
    }
    return dx_ref, dy_ref, info


def estimate_burst_local_alignment(
    frames: Sequence,
    transforms: Sequence[np.ndarray],
    reference_pos: int,
    tile_size: int = 48,
    max_shift: float = 2.0,
    local_robust: bool = True,
    local_select: Optional[float] = None,
) -> LocalAlignResult:
    """Estimate local residual fields and per-tile robustness weights for a burst.

    Args:
        frames: sequence of ``RawFrame`` (only ``.green`` is read).
        transforms: per-frame ``ref -> mov`` 3x3 homographies.
        reference_pos: index of the reference frame inside ``frames``.
        tile_size: tile size in LR pixels (50% overlap).
        max_shift: clamp on local residual displacement in LR pixels.
        local_robust: compute Wronski-style per-tile consistency weights.
        local_select: optional float in ``(0, 1)`` for AutoStakkert-style
            per-tile lucky selection (keep best ``local_select`` fraction of
            frames per tile; reference is always kept).

    Returns:
        :class:`LocalAlignResult` with ``dx_mov``, ``dy_mov``, ``weights_mov``
        all in each frame's own LR coordinates.
    """
    t0 = time.perf_counter()
    n = len(frames)
    ref_green = np.asarray(frames[reference_pos].green, dtype=np.float32)
    h, w = ref_green.shape
    noise_sigma = _estimate_noise_sigma(ref_green)

    dx_refs: List[np.ndarray] = []
    dy_refs: List[np.ndarray] = []
    dx_movs: List[np.ndarray] = []
    dy_movs: List[np.ndarray] = []
    infos: List[dict] = []

    # Tile grid for robustness / lucky selection
    t_fine = int(max(16, min(tile_size, min(h, w))))
    stride = max(8, t_fine // 2)
    ys = _tile_starts(h, t_fine, stride)
    xs = _tile_starts(w, t_fine, stride)
    ny, nx = len(ys), len(xs)

    # Precompute reference gradient magnitude for edge-aware tolerance
    gx_r = cv2.Sobel(ref_green, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / 8.0)
    gy_r = cv2.Sobel(ref_green, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / 8.0)
    gmag_r = np.hypot(gx_r, gy_r)

    tile_rms = np.zeros((n, ny, nx), dtype=np.float32)
    tile_tol = np.zeros((ny, nx), dtype=np.float32)
    for iy, y0 in enumerate(ys):
        y1 = y0 + t_fine
        for ix, x0 in enumerate(xs):
            x1 = x0 + t_fine
            # Expected noise + sub-pixel aliasing tolerance on edges
            tile_tol[iy, ix] = max(
                1e-4,
                2.5 * noise_sigma + 0.35 * float(gmag_r[y0:y1, x0:x1].mean()),
            )

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))

    for k in range(n):
        if k == reference_pos:
            z = np.zeros((h, w), dtype=np.float32)
            dx_refs.append(z)
            dy_refs.append(z.copy())
            dx_movs.append(z.copy())
            dy_movs.append(z.copy())
            infos.append({"valid_tile_frac": 1.0, "rms_lr": 0.0, "max_lr": 0.0})
            continue

        W = _normalize_h(transforms[k])
        dx_r, dy_r, info = estimate_local_displacement(
            ref_green, frames[k].green, W,
            tile_size=t_fine, max_shift=max_shift, noise_sigma=noise_sigma,
        )
        dx_refs.append(dx_r)
        dy_refs.append(dy_r)
        infos.append(info)

        # Warp (dx_ref, dy_ref) into mov LR coordinates:
        # warpPerspective(src, W, (w, h)) evaluates src(inv(W)(x_mov, y_mov)).
        W32 = W.astype(np.float32)
        dx_m = cv2.warpPerspective(dx_r, W32, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        dy_m = cv2.warpPerspective(dy_r, W32, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        dx_movs.append(dx_m)
        dy_movs.append(dy_m)

        # Compute post-local-alignment residual per tile for robustness/selection
        if local_robust or (local_select is not None):
            aligned, valid = _render_in_ref((h, w), frames[k].green, W)
            if aligned is not None:
                # Apply local shift: locally_aligned(x, y) = aligned(x - dx_r, y - dy_r)
                map_x = grid_x - dx_r
                map_y = grid_y - dy_r
                loc_ali = cv2.remap(aligned, map_x, map_y,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
                # Global brightness normalization (handles slight exposure drift)
                vm = valid > 0.95
                if np.any(vm):
                    mr = float(np.median(ref_green[vm]))
                    ma = float(np.median(loc_ali[vm]))
                    if ma > 1e-5 and mr > 1e-5:
                        gain = float(np.clip(mr / ma, 0.85, 1.18))
                        loc_ali = loc_ali * np.float32(gain)
                diff = loc_ali - ref_green
                for iy, y0 in enumerate(ys):
                    y1 = y0 + t_fine
                    for ix, x0 in enumerate(xs):
                        x1 = x0 + t_fine
                        if float(valid[y0:y1, x0:x1].mean()) < 0.90:
                            tile_rms[k, iy, ix] = 1e3
                        else:
                            d = diff[y0:y1, x0:x1]
                            tile_rms[k, iy, ix] = float(np.sqrt(np.mean(d * d)))
            else:
                tile_rms[k, :, :] = 1e3

    # Build per-tile weights (Wronski-style robustness + optional AutoStakkert top-X%)
    tile_w = np.ones((n, ny, nx), dtype=np.float32)
    if local_robust and n >= 2:
        for k in range(n):
            if k == reference_pos:
                tile_w[k] = 1.0
                continue
            ratio_r = tile_rms[k] / np.maximum(tile_tol, 1e-5)
            # Smooth Gaussian roll-off above 2.5 * tol; zero for moving objects (>5.5 * tol)
            excess = np.maximum(0.0, ratio_r - 2.5)
            tile_w[k] = np.exp(-0.5 * (excess * 1.2) ** 2).astype(np.float32)
            tile_w[k][ratio_r > 5.5] = 0.0

    if local_select is not None and 0.0 < float(local_select) < 1.0 and n >= 3:
        keep_k = max(2, int(round(float(local_select) * n)))
        for iy in range(ny):
            for ix in range(nx):
                order = np.argsort(tile_rms[:, iy, ix])
                keep_set = set(int(i) for i in order[:keep_k])
                keep_set.add(int(reference_pos))
                for k in range(n):
                    if k not in keep_set:
                        tile_w[k, iy, ix] = 0.0

    # Upsample per-tile weights to mov LR coordinates
    weights_mov: List[np.ndarray] = []
    for k in range(n):
        if k == reference_pos or (not local_robust and local_select is None):
            weights_mov.append(np.ones((h, w), dtype=np.float32))
            continue
        w_ref = _upsample_tile_grid(tile_w[k], ys, xs, t_fine, h, w)
        w_ref = np.clip(w_ref, 0.0, 1.0).astype(np.float32)
        W32 = _normalize_h(transforms[k]).astype(np.float32)
        w_mov = cv2.warpPerspective(w_ref, W32, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        weights_mov.append(np.clip(w_mov, 0.0, 1.0).astype(np.float32))

    non_ref = [inf for k, inf in enumerate(infos) if k != reference_pos]
    non_ref_w = [float(weights_mov[k].mean()) for k in range(n) if k != reference_pos]
    metrics = {
        "local_align_active": True,
        "local_tile_size": int(t_fine),
        "local_max_shift": float(max_shift),
        "local_robust": bool(local_robust),
        "local_select": (None if local_select is None else float(local_select)),
        "local_shift_rms_lr": (float(np.mean([i["rms_lr"] for i in non_ref]))
                               if non_ref else 0.0),
        "local_shift_max_lr": (float(np.max([i["max_lr"] for i in non_ref]))
                               if non_ref else 0.0),
        "local_valid_tile_frac": (float(np.mean([i["valid_tile_frac"] for i in non_ref]))
                                  if non_ref else 1.0),
        "local_mean_weight": (float(np.mean(non_ref_w)) if non_ref_w else 1.0),
        "local_align_seconds": float(time.perf_counter() - t0),
    }
    return LocalAlignResult(dx_mov=dx_movs, dy_mov=dy_movs,
                            weights_mov=weights_mov, metrics=metrics)
