"""WP7: Joint registration refinement against the HR estimate (classical, non-neural).

Given an HR estimate ``hr`` on the reference frame's pixel-centre aligned grid
and an initial per-frame homography ``W`` (ref LR -> mov LR), refines the 8
homography parameters and per-frame photometric gain/offset ``(a, b)`` by
minimising the robust (Huber) residual between each frame's observed LR
samples (raw Bayer CFA mosaic when available, otherwise green channel) and
the forward-model prediction from ``hr`` (sensor PSF blur + perspective
sampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


SENSOR_PSF_SIGMA_PER_RATIO = 1.0 / np.sqrt(12.0)


@dataclass
class FrameRefineResult:
    """Refinement result for a single frame."""

    W: np.ndarray
    gain: float
    offset: float
    corner_delta_lr_px: float
    rms_before: float
    rms_after: float
    iterations: int
    accepted: bool


@dataclass
class BurstRefineResult:
    """Refinement result for one outer iteration across a burst."""

    transforms: List[np.ndarray]
    gains: List[float]
    offsets: List[float]
    max_corner_delta_lr_px: float
    mean_corner_delta_lr_px: float
    accepted_count: int
    per_frame: List[FrameRefineResult]


def _normalize_h(W: np.ndarray) -> np.ndarray:
    W64 = np.asarray(W, dtype=np.float64)
    if W64.shape == (2, 3):
        W64 = np.vstack([W64, [0.0, 0.0, 1.0]])
    s = W64[2, 2]
    if abs(s) > 1e-12:
        W64 = W64 / s
    return W64


def corner_displacement_lr_px(W1: np.ndarray, W2: np.ndarray,
                              lr_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Return (mean, max) displacement in LR px of the 4 image corners under W1 vs W2."""
    h, w = int(lr_shape[0]), int(lr_shape[1])
    W1 = _normalize_h(W1)
    W2 = _normalize_h(W2)
    corners = np.array(
        [[0.0, 0.0, 1.0],
         [w - 1.0, 0.0, 1.0],
         [0.0, h - 1.0, 1.0],
         [w - 1.0, h - 1.0, 1.0]],
        dtype=np.float64,
    ).T  # (3, 4)
    p1 = W1 @ corners
    p1 = p1[:2] / np.where(np.abs(p1[2:3]) > 1e-12, p1[2:3], 1.0)
    p2 = W2 @ corners
    p2 = p2[:2] / np.where(np.abs(p2[2:3]) > 1e-12, p2[2:3], 1.0)
    d = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    return float(np.mean(d)), float(np.max(d))


def _norm_matrix(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalization matrix N mapping LR (x, y) to [-1, 1] and its inverse."""
    cx = 0.5 * (w - 1.0)
    cy = 0.5 * (h - 1.0)
    s = max(0.5 * max(h, w), 1.0)
    N = np.array(
        [[1.0 / s, 0.0, -cx / s],
         [0.0, 1.0 / s, -cy / s],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    N_inv = np.array(
        [[s, 0.0, cx],
         [0.0, s, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return N, N_inv, s


def _to_drizzle_matrix_f64(W: np.ndarray, ratio: float) -> np.ndarray:
    """mov LR -> HR pixel-centre aligned matrix (float64)."""
    W64 = _normalize_h(W)
    r = float(ratio)
    off = 0.5 * (r - 1.0)
    TS = np.array(
        [[r, 0.0, off],
         [0.0, r, off],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    M = TS @ np.linalg.inv(W64)
    return _normalize_h(M)


def _bayer_channel_map(shape: Tuple[int, int], pattern: np.ndarray) -> np.ndarray:
    """Return uint8 (H, W) array with values in {0: R, 1: G, 2: B} for each CFA site."""
    h, w = int(shape[0]), int(shape[1])
    pat = np.asarray(pattern, dtype=np.int32)
    cmap = np.empty((h, w), dtype=np.uint8)
    for r in range(2):
        for c in range(2):
            ci = int(pat[r, c])
            ch = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
            cmap[r::2, c::2] = ch
    return cmap


def _fit_hr_to_ref_mosaic_color(hr_blur: np.ndarray,
                                ref_mosaic: np.ndarray,
                                pattern: np.ndarray,
                                ratio: float) -> np.ndarray:
    """Fit a global 3x3 + offset linear color map from hr_blur to ref_mosaic.

    On synthetic bursts or cfa_kr this is ~identity; on real bursts with
    demosaic_drizzle it bridges LibRaw's sRGB color matrix to camera-RGB CFA
    space so R/B edges do not introduce color-matrix bias into the residual.
    """
    h, w = ref_mosaic.shape[:2]
    M_inv = np.linalg.inv(_to_drizzle_matrix_f64(np.eye(3), ratio)).astype(np.float32)
    pred_rgb = cv2.warpPerspective(
        hr_blur, M_inv, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    cmap = _bayer_channel_map((h, w), pattern)
    pad = max(4, min(16, min(h, w) // 16))
    interior = np.zeros((h, w), dtype=bool)
    interior[pad:h - pad, pad:w - pad] = True

    out = np.empty_like(hr_blur, dtype=np.float32)
    for ch in range(3):
        m = interior & (cmap == ch)
        if np.count_nonzero(m) < 64:
            out[..., ch] = hr_blur[..., ch]
            continue
        X = np.concatenate(
            [pred_rgb[m].astype(np.float64),
             np.ones((int(np.count_nonzero(m)), 1), dtype=np.float64)],
            axis=1,
        )
        y = ref_mosaic[m].astype(np.float64)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        # Apply coef to full hr_blur
        out[..., ch] = (
            hr_blur[..., 0] * np.float32(coef[0])
            + hr_blur[..., 1] * np.float32(coef[1])
            + hr_blur[..., 2] * np.float32(coef[2])
            + np.float32(coef[3])
        )
    return out


def _blur_hr(hr: np.ndarray, forward_sigma: float) -> np.ndarray:
    if forward_sigma > 0.0:
        k = int(np.ceil(3.0 * float(forward_sigma))) | 1
        return cv2.GaussianBlur(
            hr.astype(np.float32, copy=False), (k, k), float(forward_sigma),
            borderType=cv2.BORDER_REPLICATE,
        )
    return hr.astype(np.float32, copy=False)


def _hr_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference spatial gradients (per HR pixel)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REPLICATE)
    return gx, gy


def _sample_cfa_from_rgb(rgb_lr: np.ndarray, cmap: np.ndarray) -> np.ndarray:
    """Gather the single Bayer channel per pixel from an (H, W, 3) array."""
    h, w = cmap.shape
    out = np.empty((h, w), dtype=np.float32)
    for ch in range(3):
        m = (cmap == ch)
        out[m] = rgb_lr[..., ch][m]
    return out


def _gauss_newton_level(
    hr_blur: np.ndarray,
    hr_gx: np.ndarray,
    hr_gy: np.ndarray,
    obs: np.ndarray,
    W_init: np.ndarray,
    ratio: float,
    cmap: Optional[np.ndarray],
    gain_init: float,
    offset_init: float,
    max_iters: int,
    tol_lr_px: float,
    pad_hr: float,
    damping_persp: float = 0.2,
    damping_affine: float = 0.02,
    damping_base: float = 1e-5,
) -> Tuple[np.ndarray, float, float, float, float, int]:
    """Run Huber-weighted Gauss-Newton on 8 homography + 2 exposure params."""
    h, w = obs.shape[:2]
    hh, wh = hr_blur.shape[:2]
    N, N_inv, s = _norm_matrix(h, w)
    W = _normalize_h(W_init)
    gain = float(gain_init)
    offset = float(offset_init)

    # Precompute LR pixel coordinates
    xs_grid, ys_grid = np.meshgrid(
        np.arange(w, dtype=np.float64),
        np.arange(h, dtype=np.float64),
    )
    obs64 = obs.astype(np.float64, copy=False)
    is_cfa = cmap is not None

    rms_before = float("inf")
    rms_after = float("inf")
    iters_done = 0
    scale_rs = float(ratio) * float(s)

    for it in range(max_iters):
        M = _to_drizzle_matrix_f64(W, ratio)
        M_inv = np.linalg.inv(M).astype(np.float32)

        # Warp blurred HR and its gradients onto frame's LR grid
        pred_full = cv2.warpPerspective(
            hr_blur, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        gx_full = cv2.warpPerspective(
            hr_gx, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        gy_full = cv2.warpPerspective(
            hr_gy, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        if is_cfa:
            pred = _sample_cfa_from_rgb(pred_full, cmap)
            gx_s = _sample_cfa_from_rgb(gx_full, cmap)
            gy_s = _sample_cfa_from_rgb(gy_full, cmap)
        else:
            pred = pred_full
            gx_s = gx_full
            gy_s = gy_full

        # Valid interior mask: HR sample point inside [pad_hr, wh-1-pad_hr]
        W_inv = np.linalg.inv(W)
        # Map LR (x, y) -> ref LR -> normalized (u, v) and HR (qx, qy)
        denom = W_inv[2, 0] * xs_grid + W_inv[2, 1] * ys_grid + W_inv[2, 2]
        ok = np.abs(denom) > 1e-12
        denom_safe = np.where(ok, denom, 1.0)
        rx = (W_inv[0, 0] * xs_grid + W_inv[0, 1] * ys_grid + W_inv[0, 2]) / denom_safe
        ry = (W_inv[1, 0] * xs_grid + W_inv[1, 1] * ys_grid + W_inv[1, 2]) / denom_safe

        off_hr = 0.5 * (float(ratio) - 1.0)
        qx = float(ratio) * rx + off_hr
        qy = float(ratio) * ry + off_hr
        valid = (
            ok
            & (qx >= pad_hr) & (qx <= wh - 1.0 - pad_hr)
            & (qy >= pad_hr) & (qy <= hh - 1.0 - pad_hr)
        )
        n_valid = int(np.count_nonzero(valid))
        if n_valid < 256:
            break

        p_v = pred[valid].astype(np.float64)
        o_v = obs64[valid]
        res = gain * p_v + offset - o_v
        rms = float(np.sqrt(np.mean(res * res)))
        if it == 0:
            rms_before = rms
        rms_after = rms

        # Huber IRLS weights (scale-invariant via MAD)
        mad = float(np.median(np.abs(res)))
        sigma = max(1.4826 * mad, 1e-4)
        tau = 1.345 * sigma
        abs_r = np.abs(res)
        w_rob = np.where(abs_r <= tau, 1.0, tau / np.maximum(abs_r, 1e-12))

        # Normalized reference coordinates (u, v) in [-1, 1]
        u = (rx[valid] - N_inv[0, 2]) / s
        v = (ry[valid] - N_inv[1, 2]) / s
        gx_v = (gain * scale_rs) * gx_s[valid].astype(np.float64)
        gy_v = (gain * scale_rs) * gy_s[valid].astype(np.float64)

        # Build 10-column Jacobian: [d0..d7, da, db]
        J = np.empty((n_valid, 10), dtype=np.float64)
        J[:, 0] = gx_v * u
        J[:, 1] = gx_v * v
        J[:, 2] = gx_v
        J[:, 3] = gy_v * u
        J[:, 4] = gy_v * v
        J[:, 5] = gy_v
        J[:, 6] = -(gx_v * u * u + gy_v * u * v)
        J[:, 7] = -(gx_v * u * v + gy_v * v * v)
        J[:, 8] = p_v
        J[:, 9] = 1.0

        # Weighted normal equations: (J^T W J + damp) delta = -J^T W r
        Jw = J * w_rob[:, None]
        H_mat = Jw.T @ J
        g_vec = -(Jw.T @ res)

        # Scale-aware Levenberg-Marquardt damping (stronger on affine/perspective
        # so flat image corners cannot drift)
        diag_h = np.maximum(np.diag(H_mat), 1e-6)
        trans_scale = max(float(diag_h[2]), float(diag_h[5]), 1e-6)
        damp = np.full(10, damping_base, dtype=np.float64)
        damp[0] = damping_affine
        damp[1] = damping_affine
        damp[3] = damping_affine
        damp[4] = damping_affine
        damp[6] = damping_persp
        damp[7] = damping_persp
        # Floor affine/perspective damping by translation Hessian scale so
        # weakly-constrained corners stay anchored to W_init.
        diag_ref = diag_h.copy()
        diag_ref[[0, 1, 3, 4]] = np.maximum(diag_ref[[0, 1, 3, 4]], 0.25 * trans_scale)
        diag_ref[[6, 7]] = np.maximum(diag_ref[[6, 7]], 0.25 * trans_scale)
        H_mat[np.arange(10), np.arange(10)] += damp * diag_ref + 1e-9

        try:
            delta = np.linalg.solve(H_mat, g_vec)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(delta)):
            break

        dH = np.array(
            [[1.0 + delta[0], delta[1], delta[2]],
             [delta[3], 1.0 + delta[4], delta[5]],
             [delta[6], delta[7], 1.0]],
            dtype=np.float64,
        )
        if abs(np.linalg.det(dH)) < 1e-4:
            break

        W_cand = _normalize_h(W @ N_inv @ np.linalg.inv(dH) @ N)
        gain_cand = float(np.clip(gain + delta[8], 0.5, 2.0))
        offset_cand = float(np.clip(offset + delta[9], -0.1, 0.1))

        _, max_step_px = corner_displacement_lr_px(W, W_cand, (h, w))
        # Guard against wild divergence (> 5 LR px in a single GN step)
        if max_step_px > 5.0:
            break

        W = W_cand
        gain = gain_cand
        offset = offset_cand
        iters_done = it + 1

        if max_step_px < tol_lr_px:
            break

    # Final RMS check after last update
    if iters_done > 0:
        M = _to_drizzle_matrix_f64(W, ratio)
        M_inv = np.linalg.inv(M).astype(np.float32)
        pred_full = cv2.warpPerspective(
            hr_blur, M_inv, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        pred = _sample_cfa_from_rgb(pred_full, cmap) if is_cfa else pred_full
        W_inv = np.linalg.inv(W)
        denom = W_inv[2, 0] * xs_grid + W_inv[2, 1] * ys_grid + W_inv[2, 2]
        ok = np.abs(denom) > 1e-12
        denom_safe = np.where(ok, denom, 1.0)
        rx = (W_inv[0, 0] * xs_grid + W_inv[0, 1] * ys_grid + W_inv[0, 2]) / denom_safe
        ry = (W_inv[1, 0] * xs_grid + W_inv[1, 1] * ys_grid + W_inv[1, 2]) / denom_safe
        off_hr = 0.5 * (float(ratio) - 1.0)
        qx = float(ratio) * rx + off_hr
        qy = float(ratio) * ry + off_hr
        valid = (
            ok
            & (qx >= pad_hr) & (qx <= wh - 1.0 - pad_hr)
            & (qy >= pad_hr) & (qy <= hh - 1.0 - pad_hr)
        )
        if np.count_nonzero(valid) >= 256:
            res_f = gain * pred[valid].astype(np.float64) + offset - obs64[valid]
            rms_after = float(np.sqrt(np.mean(res_f * res_f)))

    return W, gain, offset, rms_before, rms_after, iters_done


def refine_frame_registration(
    hr_blur: np.ndarray,
    hr_gx: np.ndarray,
    hr_gy: np.ndarray,
    obs: np.ndarray,
    W_init: np.ndarray,
    ratio: float,
    cmap: Optional[np.ndarray] = None,
    forward_sigma: Optional[float] = None,
    max_iters: int = 8,
    tol_lr_px: float = 0.002,
    coarse_to_fine: bool = True,
    max_total_shift_px: float = 3.0,
) -> FrameRefineResult:
    """Refine a single frame's homography and exposure against blurred HR."""
    h, w = obs.shape[:2]
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    pad_hr = 2.0 + (np.ceil(3.0 * float(forward_sigma)) if forward_sigma > 0 else 0.0)

    W_cur = _normalize_h(W_init)
    gain = 1.0
    offset = 0.0
    total_iters = 0

    # Optional 2x coarse level for larger images (>= 512 px) when not using CFA
    # (CFA downsampling by 2x would mix Bayer phases; on CFA we can run directly
    # at full res since initial ECC is already within < 1 LR px).
    if coarse_to_fine and cmap is None and min(h, w) >= 512:
        h2, w2 = h // 2, w // 2
        hh2, wh2 = hr_blur.shape[0] // 2, hr_blur.shape[1] // 2
        obs2 = cv2.resize(obs, (w2, h2), interpolation=cv2.INTER_AREA)
        hr2 = cv2.resize(hr_blur, (wh2, hh2), interpolation=cv2.INTER_AREA)
        gx2, gy2 = _hr_gradients(hr2)
        S_down = np.diag([0.5, 0.5, 1.0])
        S_up = np.diag([2.0, 2.0, 1.0])
        W2 = _normalize_h(S_down @ W_cur @ S_up)
        W2_ref, gain, offset, _, _, it_c = _gauss_newton_level(
            hr2, gx2, gy2, obs2, W2, ratio, None,
            gain, offset, max_iters=4, tol_lr_px=0.01,
            pad_hr=max(2.0, pad_hr * 0.5),
        )
        W_up = _normalize_h(S_up @ W2_ref @ S_down)
        _, step_c = corner_displacement_lr_px(W_cur, W_up, (h, w))
        if step_c <= max_total_shift_px:
            W_cur = W_up
        total_iters += it_c

    W_ref, gain, offset, rms_before, rms_after, it_f = _gauss_newton_level(
        hr_blur, hr_gx, hr_gy, obs, W_cur, ratio, cmap,
        gain, offset, max_iters=max_iters, tol_lr_px=tol_lr_px,
        pad_hr=pad_hr,
    )
    total_iters += it_f

    _, delta_px = corner_displacement_lr_px(W_init, W_ref, (h, w))
    # Accept only if residual did not worsen and total shift is bounded
    accepted = (
        np.isfinite(rms_after)
        and rms_after <= rms_before * 1.001
        and delta_px <= max_total_shift_px
    )
    if not accepted:
        return FrameRefineResult(
            W=_normalize_h(W_init),
            gain=1.0,
            offset=0.0,
            corner_delta_lr_px=0.0,
            rms_before=rms_before,
            rms_after=rms_before,
            iterations=total_iters,
            accepted=False,
        )

    return FrameRefineResult(
        W=W_ref,
        gain=gain,
        offset=offset,
        corner_delta_lr_px=delta_px,
        rms_before=rms_before,
        rms_after=rms_after,
        iterations=total_iters,
        accepted=True,
    )


def refine_burst_registration(
    hr: np.ndarray,
    frames: Sequence,
    transforms: Sequence[np.ndarray],
    reference_pos: int,
    ratio: float,
    forward_sigma: Optional[float] = None,
    use_cfa: bool = True,
    max_iters: int = 8,
    tol_lr_px: float = 0.002,
) -> BurstRefineResult:
    """Refine all non-reference frame homographies and exposure against ``hr``.

    Args:
        hr: Current HR estimate (H_hr, W_hr, 3) float32 on the reference grid.
        frames: Sequence of RawFrame objects (must have ``.green``, optionally
            ``.mosaic`` and ``.pattern``).
        transforms: Sequence of 3x3 ref->mov homographies.
        reference_pos: Index of the reference frame in ``frames`` (kept fixed).
        ratio: SR scale factor.
        forward_sigma: Forward-model PSF sigma in HR px (default 0.289 * ratio).
        use_cfa: If True and all frames have ``.mosaic`` and ``.pattern``,
            refine directly against raw Bayer CFA samples. Otherwise use green.
        max_iters: Max Gauss-Newton iterations per frame.
        tol_lr_px: Inner stopping threshold in LR px.
    """
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)

    has_cfa = (
        use_cfa
        and len(frames) > 0
        and all(getattr(f, "mosaic", None) is not None
                and getattr(f, "pattern", None) is not None for f in frames)
    )

    if has_cfa:
        hr_blur_rgb = _blur_hr(hr, forward_sigma)
        ref_f = frames[reference_pos]
        hr_cam = _fit_hr_to_ref_mosaic_color(
            hr_blur_rgb, ref_f.mosaic, ref_f.pattern, ratio,
        )
        hr_gx, hr_gy = _hr_gradients(hr_cam)
        hr_src = hr_cam
        cmap = _bayer_channel_map(ref_f.mosaic.shape[:2], ref_f.pattern)
    else:
        hr_g = hr[..., 1] if hr.ndim == 3 else hr
        hr_src = _blur_hr(hr_g, forward_sigma)
        hr_gx, hr_gy = _hr_gradients(hr_src)
        cmap = None

    new_transforms: List[np.ndarray] = []
    gains: List[float] = []
    offsets: List[float] = []
    per_frame: List[FrameRefineResult] = []
    deltas: List[float] = []
    accepted_count = 0

    for idx, (f, W) in enumerate(zip(frames, transforms)):
        if idx == reference_pos:
            W_id = _normalize_h(W)
            new_transforms.append(W_id)
            gains.append(1.0)
            offsets.append(0.0)
            per_frame.append(FrameRefineResult(
                W=W_id, gain=1.0, offset=0.0, corner_delta_lr_px=0.0,
                rms_before=0.0, rms_after=0.0, iterations=0, accepted=True,
            ))
            continue

        obs = f.mosaic if has_cfa else f.green
        res = refine_frame_registration(
            hr_src, hr_gx, hr_gy, obs, W, ratio,
            cmap=cmap, forward_sigma=forward_sigma,
            max_iters=max_iters, tol_lr_px=tol_lr_px,
        )
        new_transforms.append(res.W)
        gains.append(res.gain)
        offsets.append(res.offset)
        per_frame.append(res)
        if res.accepted:
            accepted_count += 1
            deltas.append(res.corner_delta_lr_px)

    max_delta = float(max(deltas)) if deltas else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return BurstRefineResult(
        transforms=new_transforms,
        gains=gains,
        offsets=offsets,
        max_corner_delta_lr_px=max_delta,
        mean_corner_delta_lr_px=mean_delta,
        accepted_count=accepted_count,
        per_frame=per_frame,
    )
