"""RAW session helpers: validation, previews, thumbnails, blur flags, analysis.

Owned by AGENT-A. Baseline implementation by the orchestrator.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from desktop.backend.imaging import encode_png, fit_max_dim, to_uint8_view
RAW_EXTENSIONS = {
    'dng', 'cr2', 'cr3', 'nef', 'nrw', 'arw', 'srf', 'sr2',
    'raf', 'rw2', 'orf', 'pef', 'raw', 'tif', 'tiff',
}

BLUR_THRESHOLD_RATIO = 0.5

# Canonical preview resolution: every consumer (thumbnails, session previews,
# ROI analysis) decodes a file once at this size and downscales from there, so
# repeated previews/analyze of the same burst reuse one RAW decode.
CANONICAL_PREVIEW_DIM = 1024
ANALYZE_PREVIEW_DIM = 512

# ROI-awareness thresholds.
ROI_MIN_OVERLAP = 0.5          # fraction of the ROI a frame's shifted view must cover
MAX_SHIFT_FRACTION = 0.5       # reject a frame shifted out by > this fraction of the frame

# Small thread-safe decode cache (functools.lru_cache is safe for concurrent
# use in CPython). Keyed by (path, mtime, max_dim) so edits invalidate it.
_CACHE_MAXSIZE = 16


def is_raw(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    return ext in RAW_EXTENSIONS


def load_preview_linear(path: str, half_size: bool = True
                        ) -> Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]:
    """Load a RAW file to linear float32 RGB (fast half-size preview) + metadata.

    Returns (rgb float32 0..1, meta, (full_height, full_width)).
    """
    import rawpy

    with rawpy.imread(path) as raw:
        full_h = int(raw.sizes.height)
        full_w = int(raw.sizes.width)
        rgb16 = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            half_size=half_size,
        )
        meta = {
            'black_level': list(raw.black_level_per_channel)
            if raw.black_level_per_channel is not None else None,
            'white_level': int(raw.white_level) if raw.white_level is not None else None,
            'camera_whitebalance': list(raw.camera_whitebalance)
            if raw.camera_whitebalance is not None else None,
        }

    rgb = rgb16.astype(np.float32) / 65535.0
    return rgb, meta, (full_h, full_w)


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _decode_scaled(path: str, mtime: float, max_dim: int
                   ) -> Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]:
    """Cached half-size decode, downscaled to ``max_dim`` (see load_scaled_linear)."""
    del mtime  # part of the cache key only; invalidates when the file changes.
    rgb, meta, full_hw = load_preview_linear(path, half_size=True)
    return fit_max_dim(rgb, max_dim), meta, full_hw


def load_scaled_linear(path: str, max_dim: int = CANONICAL_PREVIEW_DIM
                       ) -> Tuple[np.ndarray, Dict[str, Any], Tuple[int, int]]:
    """Decode ``path`` (cached) to linear RGB whose longest side is ``max_dim``.

    Repeated calls for an unchanged file (same path + mtime + max_dim) return
    the cached array without re-running the RAW demosaic. The returned array is
    a copy the caller may freely read; treat it as read-only.
    """
    st = os.stat(path)
    rgb, meta, full_hw = _decode_scaled(path, float(st.st_mtime), int(max_dim))
    return rgb.copy(), dict(meta), full_hw


def clear_cache() -> None:
    """Drop the decode cache (used by tests)."""
    _decode_scaled.cache_clear()


# Frame id -> path, recorded by build_frame_infos (i.e. POST /api/session/files)
# so that later requests which carry frame *ids* can resolve them to paths.
_FRAME_ID_MAP: Dict[str, str] = {}


def frame_id_map() -> Dict[str, str]:
    """Snapshot of the id -> path mapping from the most recent session loads."""
    return dict(_FRAME_ID_MAP)


def clear_frame_ids() -> None:
    """Forget recorded frame ids (used by tests)."""
    _FRAME_ID_MAP.clear()


def _sharpness(linear_rgb: np.ndarray) -> float:
    view = to_uint8_view(linear_rgb)
    gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_frame_infos(paths: List[str]
                      ) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Validate and describe selected files. Never fails the whole batch."""
    loaded: List[Tuple[int, str, np.ndarray, Tuple[int, int]]] = []
    rejected: List[Dict[str, str]] = []

    for i, path in enumerate(paths):
        if not os.path.exists(path):
            rejected.append({'path': path, 'reason': 'File not found'})
            continue
        if not is_raw(path):
            rejected.append({
                'path': path,
                'reason': 'Not a RAW file — this pipeline requires RAW sensor data.',
            })
            continue
        try:
            rgb, _meta, full_hw = load_scaled_linear(path, CANONICAL_PREVIEW_DIM)
        except Exception as exc:  # noqa: BLE001
            rejected.append({'path': path, 'reason': f'Could not decode RAW: {exc}'})
            continue
        loaded.append((i, path, rgb, full_hw))

    scores = [_sharpness(rgb) for _, _, rgb, _ in loaded]
    best = max(scores) if scores else 0.0

    frames: List[Dict[str, Any]] = []
    for idx, (i, path, rgb, full_hw) in enumerate(loaded):
        blurry = best > 0.0 and scores[idx] < BLUR_THRESHOLD_RATIO * best
        _FRAME_ID_MAP[str(i)] = path
        frames.append({
            'id': str(i),
            'path': path,
            'name': os.path.basename(path),
            'width': int(full_hw[1]),
            'height': int(full_hw[0]),
            'thumbnail': encode_png(fit_max_dim(to_uint8_view(rgb), 160)),
            'preview': encode_png(fit_max_dim(to_uint8_view(rgb), 1024)),
            'sharpness': float(scores[idx]),
            'blurry': bool(blurry),
        })
    return frames, rejected


def preview_data_url(path: str, max_dim: int = CANONICAL_PREVIEW_DIM) -> Tuple[str, int, int]:
    rgb, _meta, full_hw = load_scaled_linear(path, max_dim)
    view = fit_max_dim(to_uint8_view(rgb), max_dim)
    return encode_png(view), int(full_hw[1]), int(full_hw[0])


def load_reference_linear(path: str,
                          roi: Optional[Tuple[int, int, int, int]] = None
                          ) -> np.ndarray:
    """Linear RGB of one frame, decoded exactly like ``FrameInfo.preview``.

    This is the honest source for the result's ``before`` image: it reuses the
    *preview* decode path (:func:`load_preview_linear` — rawpy's default
    demosaic, ``gamma=(1, 1)``, ``no_auto_bright``, camera white balance, no
    ``demosaic_algorithm`` override) instead of the SR core's DCB-demosaiced
    ``load_raw_linear`` frame. DCB is a sharp directional demosaic, so a
    ``before`` built from it looked like a processed derivative rather than the
    original RAW frame; the default demosaic matches what the frame preview
    shows. No extra sharpening is applied here.

    ``half_size=False`` decodes at full sensor resolution (the previews use the
    half-size decode purely as a speed optimisation) so the ROI keeps its detail
    when it is later upscaled to the HR display size. The optional ``roi`` is in
    full-resolution source pixels; it is clamped to the decoded frame.
    """
    rgb, _meta, _full_hw = load_preview_linear(path, half_size=False)
    if roi is not None:
        x, y, w, h = (int(v) for v in roi)
        x = max(0, min(x, rgb.shape[1] - 1))
        y = max(0, min(y, rgb.shape[0] - 1))
        w = max(1, min(w, rgb.shape[1] - x))
        h = max(1, min(h, rgb.shape[0] - y))
        rgb = rgb[y:y + h, x:x + w]
    return np.ascontiguousarray(rgb)


# --------------------------------------------------------------------------- #
# ROI helpers
# --------------------------------------------------------------------------- #

def roi_overlap_fraction(roi: Tuple[float, float, float, float],
                         frame_shape: Tuple[int, int],
                         shift: Tuple[float, float]) -> float:
    """Fraction of ``roi`` (reference coords) covered by a frame shifted by ``shift``.

    ``estimate_translation`` returns ``(dx, dy)`` such that
    ``frame(x, y) ~= ref(x + dx, y + dy)`` (the core convention). Frame pixel
    ``u`` therefore samples the reference at ``p = u + dx``; the frame
    contributes where ``u = p - dx`` is inside ``[0, width)``, i.e.
    ``p in [dx, width + dx)``. The same applies vertically. Returns a value in
    ``[0, 1]``.
    """
    rx, ry, rw, rh = (float(v) for v in roi)
    fh, fw = float(frame_shape[0]), float(frame_shape[1])
    dx, dy = float(shift[0]), float(shift[1])

    vx0, vx1 = max(0.0, dx), min(fw, fw + dx)
    vy0, vy1 = max(0.0, dy), min(fh, fh + dy)

    ix0, ix1 = max(rx, vx0), min(rx + rw, vx1)
    iy0, iy1 = max(ry, vy0), min(ry + rh, vy1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = rw * rh
    return float(inter / area) if area > 0 else 0.0


def _roi_to_preview(roi: Optional[Tuple[int, int, int, int]],
                    full_hw: Tuple[int, int],
                    preview_hw: Tuple[int, int]
                    ) -> Optional[Tuple[float, float, float, float]]:
    """Scale a source-pixel (x, y, w, h) ROI into preview pixel coordinates."""
    if roi is None:
        return None
    fh, fw = float(full_hw[0]), float(full_hw[1])
    ph, pw = float(preview_hw[0]), float(preview_hw[1])
    sx = pw / fw if fw > 0 else 1.0
    sy = ph / fh if fh > 0 else 1.0
    x, y, w, h = (float(v) for v in roi)
    return (x * sx, y * sy, w * sx, h * sy)


def resolve_exclude(paths: List[str], exclude: Optional[List[str]],
                    id_map: Optional[Dict[str, str]] = None) -> List[str]:
    """Resolve ``exclude`` frame ids to the subset of ``paths`` to keep.

    Ids are the strings returned by ``POST /api/session/files``. When an
    ``id_map`` (id -> original path) is available it is authoritative: the id
    is resolved to its path and only dropped if that exact path is among
    ``paths`` (ids whose path is absent are ignored). Otherwise the id is
    treated as an in-range index into ``paths``, falling back to a literal
    path match. Unknown or malformed ids are ignored and never raise.
    """
    if not exclude:
        return list(paths)
    n = len(paths)
    drop: set[int] = set()
    for raw_id in exclude:
        if raw_id is None:
            continue
        token = str(raw_id)
        if id_map is not None and token in id_map:
            target = id_map[token]
            if target in paths:
                drop.add(paths.index(target))
            continue
        try:
            idx = int(token)
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < n:
            drop.add(idx)
            continue
        for i, p in enumerate(paths):
            if p == token:
                drop.add(i)
    return [p for i, p in enumerate(paths) if i not in drop]


def analyze(paths: List[str],
            target_roi: Optional[Tuple[int, int, int, int]],
            search_area: Optional[Tuple[int, int, int, int]],
            sr_ratio: float,
            k_threshold: float = 3.0) -> Dict[str, Any]:
    """Fast motion + achievable-SR feedback on downscaled previews.

    ``per_frame_usable`` / ``usable_frames`` are ROI-aware: a frame counts only
    when its (shifted) view still covers the target ROI — and, when supplied,
    the search area used for registration. A global shift-magnitude cap alone
    is not sufficient, though an extreme shift is still rejected.
    """
    from raw_sr.core.superres import estimate_translation, translation_matrix
    from raw_sr.core.motion_analyzer import MotionAnalyzer
    from raw_sr.core.sr_ratio_analyzer import SRRatioAnalyzer

    decoded: List[Optional[Tuple[np.ndarray, Tuple[int, int]]]] = []
    for path in paths:
        try:
            rgb, _m, full_hw = load_scaled_linear(path, CANONICAL_PREVIEW_DIM)
            decoded.append((fit_max_dim(rgb, ANALYZE_PREVIEW_DIM), full_hw))
        except Exception:  # noqa: BLE001
            decoded.append(None)

    total = len(paths)
    usable_idx = [i for i, d in enumerate(decoded) if d is not None]
    if not usable_idx:
        return {
            'motion': {'avg_shift': 0.0, 'max_shift': 0.0, 'sufficient': False,
                       'message': 'No decodable frames.',
                       'shifts': [[0.0, 0.0] for _ in range(total)]},
            'sr_ratio': {'requested': float(sr_ratio), 'achievable': 1.0,
                         'coverage_fraction': 0.0, 'message': 'No frames.'},
            'usable_frames': 0, 'total_frames': total,
            'per_frame_usable': [False] * total,
        }

    ref_small, ref_full = decoded[usable_idx[0]]  # type: ignore[misc]
    ref_green = ref_small[:, :, 1]

    shifts: Dict[int, Tuple[float, float]] = {usable_idx[0]: (0.0, 0.0)}
    for i in usable_idx[1:]:
        dx, dy = estimate_translation(ref_green, decoded[i][0][:, :, 1])  # type: ignore[index]
        shifts[i] = (float(dx), float(dy))

    shift_list = [shifts[i] for i in usable_idx]
    homographies = [translation_matrix(dx, dy) for dx, dy in shift_list]
    usable_previews = [decoded[i][0] for i in usable_idx]  # type: ignore[index]

    # Internal feasibility feedback stays in the analysis-preview domain:
    # MotionAnalyzer's thresholds (0.1..5 px) and SRRatioAnalyzer's subpixel
    # binning were tuned for the downscaled arrays, so `sufficient` and
    # `sr_ratio` must be computed on the preview shifts, not the source ones.
    analyzer = MotionAnalyzer()
    motion = analyzer.analyze(usable_previews, homographies=homographies)
    ratio_info = SRRatioAnalyzer().analyze(shift_list, float(sr_ratio))

    ph, pw = ref_small.shape[:2]
    roi_px = _roi_to_preview(target_roi, ref_full, (ph, pw))
    if roi_px is None:
        roi_px = (0.0, 0.0, float(pw), float(ph))
    search_px = _roi_to_preview(search_area, ref_full, (ph, pw))

    per_frame = [False] * total
    for i, d in enumerate(decoded):
        if d is None or i not in shifts:
            continue
        small, _full = d
        frame_hw = (int(small.shape[0]), int(small.shape[1]))
        dx, dy = shifts[i]
        magnitude = float(np.hypot(dx, dy))

        frac = roi_overlap_fraction(roi_px, frame_hw, (dx, dy))
        usable = frac >= ROI_MIN_OVERLAP
        if search_px is not None:
            search_frac = roi_overlap_fraction(search_px, frame_hw, (dx, dy))
            usable = usable and search_frac >= ROI_MIN_OVERLAP
        usable = usable and magnitude <= MAX_SHIFT_FRACTION * min(frame_hw)
        per_frame[i] = bool(usable)

    # The API reports motion in SOURCE pixels (frozen contract), so the ROI
    # panel can feed the shifts straight into its source-space geometry. The
    # analysis preview is fit_max_dim's downscale of the full sensor, so invert
    # that per axis (source_dim / preview_dim) — the same ratio
    # ``_roi_to_preview`` applies in the other direction. The reference frame
    # (the first usable frame) stays at (0, 0).
    sx = float(ref_full[1]) / float(pw) if pw else 1.0
    sy = float(ref_full[0]) / float(ph) if ph else 1.0
    source_shifts = {
        i: (float(shifts[i][0]) * sx, float(shifts[i][1]) * sy)
        for i in usable_idx
    }
    # Statistics skip the reference frame, matching MotionAnalyzer (frame 0).
    magnitudes = [float(np.hypot(*source_shifts[i])) for i in usable_idx[1:]]
    avg_shift = float(np.mean(magnitudes)) if magnitudes else 0.0
    max_shift = float(np.max(magnitudes)) if magnitudes else 0.0

    # Keep the preview-domain sufficiency verdict (the branch decision) but
    # phrase the message with the source-pixel magnitude the caller sees.
    preview_avg = float(motion['avg_shift'])
    if preview_avg < analyzer.min_useful_shift:
        suffix = f'too small for subpixel SR (need ≥{analyzer.min_useful_shift:g}px)'
    elif preview_avg > analyzer.max_usable_shift:
        suffix = 'too large, frames may be misaligned'
    else:
        suffix = 'suitable for subpixel SR'
    message = f'Average inter-frame shift: {avg_shift:.2f}px — {suffix}'

    shifts_out = [
        [source_shifts[i][0], source_shifts[i][1]] if i in source_shifts else [0.0, 0.0]
        for i in range(total)
    ]
    return {
        'motion': {
            'avg_shift': avg_shift,
            'max_shift': max_shift,
            'sufficient': bool(motion['sufficient']),
            'message': message,
            'shifts': shifts_out,
        },
        'sr_ratio': ratio_info,
        'usable_frames': int(sum(per_frame)),
        'total_frames': total,
        'per_frame_usable': [bool(v) for v in per_frame],
    }


def quality(paths: List[str], threshold_ratio: float = BLUR_THRESHOLD_RATIO
            ) -> Dict[str, Any]:
    """Recompute blur flags/scores for a set of paths."""
    scores: List[float] = []
    for path in paths:
        try:
            rgb, _m, _s = load_scaled_linear(path, CANONICAL_PREVIEW_DIM)
            scores.append(_sharpness(rgb))
        except Exception:  # noqa: BLE001
            scores.append(0.0)
    best = max(scores) if scores else 0.0
    flags = [best > 0 and s < threshold_ratio * best for s in scores]
    return {'flags': flags, 'scores': scores}
