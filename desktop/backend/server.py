"""FastAPI backend exposing the raw_sr multi-frame SR engine as HTTP.

Owned by AGENT-A. Baseline implementation by the orchestrator.
Run:  python -m desktop.backend.server --port 8765
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from desktop.backend import session
from desktop.backend.imaging import apply_view, encode_png, fit_max_dim, heatmap
from desktop.backend.jobs import Job, JobManager

# Make the repository root importable so `raw_sr` resolves when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --------------------------------------------------------------------------- #
# OpenCL device pinning — must run before ANY cv2 UMat use in this process.
#
# OpenCV reads OPENCV_OPENCL_DEVICE exactly once, when it initialises its
# default OpenCL context on first UMat/T-API use (process-wide). On a
# hybrid-graphics laptop that lazy default is nondeterministic across machines
# (whichever ICD registers first). When the NVIDIA proprietary stack is
# detected we pin its GPU explicitly so the *same* device is selected here and
# in every spawned core worker (spawn children inherit os.environ). A user-set
# OPENCV_OPENCL_DEVICE always wins; on machines without the NVIDIA runtime we
# change nothing, keeping the brand-agnostic default. An *invalid* pin is NOT
# harmless — OpenCV's selectOpenCLDevice then disables OpenCL entirely (silent
# CPU fallback) — which is why detection (nvidia-smi answers AND the ICD is
# registered) gates the pin instead of setting the env unconditionally.
NVIDIA_OPENCL_DEVICE = 'NVIDIA CUDA:GPU:0'


def _nvidia_gpu_present() -> bool:
    """Cheap one-shot probe: NVIDIA driver answers AND its OpenCL ICD is registered."""
    import subprocess
    try:
        if subprocess.run(['nvidia-smi', '-L'], capture_output=True,
                          timeout=5).returncode != 0:
            return False
    except Exception:  # noqa: BLE001 - no driver/binary -> not present
        return False
    import platform
    if platform.system() == 'Linux':  # ICD must be registered for the loader
        return os.path.exists('/etc/OpenCL/vendors/nvidia.icd')
    return True


def _pin_nvidia_opencl_device(env: Optional[Dict[str, str]] = None,
                              has_nvidia=None) -> Optional[str]:
    """Set OPENCV_OPENCL_DEVICE to the NVIDIA GPU iff warranted; return its value.

    Pure enough to unit-test: pass a dict as `env` and a `has_nvidia` callable
    to stub the platform probe. Never overrides a user-supplied value.
    """
    env = os.environ if env is None else env
    if env.get('OPENCV_OPENCL_DEVICE'):
        return None  # explicit user/platform choice always wins
    if not (has_nvidia or _nvidia_gpu_present)():
        return None
    env['OPENCV_OPENCL_DEVICE'] = NVIDIA_OPENCL_DEVICE
    return NVIDIA_OPENCL_DEVICE


# Runs at module import, before the first UMat can initialise the context. The
# returned value (None when the pin was not written) is reported by the engine
# diagnostics below.
_PINNED_DEVICE = _pin_nvidia_opencl_device()

DEFAULT_VIEW = {'gamma': 2.2, 'brightness': 0.0, 'contrast': 1.0}
MAX_ROI_AREA = 2048 * 2048
MAX_HR_AREA = 4096 * 4096
MAX_DISPLAY_DIM = 1024
# Product bursts are 2-20 frames; 24 keeps honest headroom while capping the
# real memory a single request can pin (each decoded frame holds a full-res
# base until the core's ROI detach lands). Enforced on every `paths` endpoint
# via _require_non_empty -> 400.
MAX_FRAMES = 24
MAX_SR_RATIO = 8.0
REGISTRATION_MODES = ('homography', 'translation')

app = FastAPI(title='raw_sr desktop backend', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
manager = JobManager()

# Single-flight guard for POST /api/process: the SR run decodes every frame
# into RAM, so two concurrent process jobs would double the peak. A second
# request while one is in flight gets 409 (see ARCHITECTURE.md). Released by
# the job's on_finish hook on every terminal path (done/error/cancelled).
_PROCESS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Consistent error handling
# --------------------------------------------------------------------------- #

@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_request: Request, exc: RequestValidationError
                                    ) -> JSONResponse:
    """Normalize pydantic validation failures to the contract's error body.

    The frozen contract only specifies ``{ "detail": "..." }``; FastAPI's
    default validation response uses a list detail, so collapse it to a string
    and use 400 (a client error) rather than 422.
    """
    parts = []
    for err in exc.errors():
        loc = '.'.join(str(p) for p in err.get('loc', ()) if p not in ('body', 'query'))
        msg = str(err.get('msg', 'invalid value'))
        parts.append(f'{loc}: {msg}' if loc else msg)
    return JSONResponse(status_code=400, content={'detail': '; '.join(parts) or 'Invalid request'})


def _require_non_empty(paths: List[str]) -> None:
    if not paths:
        raise HTTPException(status_code=400, detail='No paths provided')
    if len(paths) > MAX_FRAMES:
        raise HTTPException(status_code=400,
                            detail=f'Too many frames ({len(paths)} > {MAX_FRAMES})')


def _require_readable_raw(paths: List[str]) -> None:
    """Raise 404/400 when a required file is missing or not a supported RAW."""
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise HTTPException(status_code=404, detail=f'File not found: {missing[0]}')
    non_raw = [p for p in paths if not session.is_raw(p)]
    if non_raw:
        raise HTTPException(status_code=400,
                            detail=f'Not a RAW file: {non_raw[0]}')


def _require_registration(mode: str) -> None:
    if mode not in REGISTRATION_MODES:
        raise HTTPException(status_code=400,
                            detail=f'Unknown registration mode: {mode}')


# --------------------------------------------------------------------------- #
# HR-size memory guard (mirrors SuperResolution._check_hr_size)
#
# Peak memory scales with the HR pixel count, not with the LR ROI, so a
# ratio/ROI combination can ask for gigabytes before any kernel runs. The core
# raises MemoryError for it; refusing in the endpoint as well means the user
# gets a clean, actionable 413 *before* the burst is decoded (and before the
# single-flight lock is taken by a doomed job). The numbers come from the core
# when the installed build exposes them, so the two guards cannot disagree.
# --------------------------------------------------------------------------- #

_FALLBACK_BYTES_PER_HR_PIXEL = 160
_FALLBACK_WARN_FRACTION = 0.75


def _memory_guard(path: str, target_roi: Optional[Tuple[int, int, int, int]],
                  ratio: float, budget_bytes: Optional[int],
                  n_frames: int = 1) -> Optional[str]:
    """Return a non-fatal warning string; raise 413 when the run cannot fit.

    ``budget_bytes`` is the request override; None means the core's own budget
    (60 % of ``MemAvailable``, clamped). ``n_frames`` is the number of fused
    frames: the core's peak grows with the per-frame gate stack, so the
    per-pixel estimate is frame-count aware too. Any failure while probing
    degrades to "no pre-check" rather than blocking a legitimate run — the
    core's own guard still applies at run time.
    """
    try:
        from raw_sr.core import superres as core_sr
        per_px_fn = getattr(core_sr, '_hr_guard_bytes_per_pixel', None)
        per_px = (int(per_px_fn(n_frames)) if callable(per_px_fn)
                  else int(getattr(core_sr, '_HR_GUARD_BYTES_PER_PIXEL',
                                   _FALLBACK_BYTES_PER_HR_PIXEL)))
        warn_frac = float(getattr(core_sr, '_HR_GUARD_WARN_FRACTION',
                                  _FALLBACK_WARN_FRACTION))
        estimate = getattr(core_sr, 'estimate_hr_peak_bytes', None)
        (_rx, _ry, rw, rh), _clipped = resolve_roi(path, target_roi, ratio=ratio)
        oh, ow = int(round(rh * ratio)), int(round(rw * ratio))
        pixels = max(1, oh * ow)
        budget = (int(budget_bytes) if budget_bytes is not None
                  else int(core_sr._hr_guard_budget_bytes()))
        limit = max(1, budget // per_px)
        if callable(estimate):
            try:
                est = int(estimate(pixels, n_frames=n_frames))
            except TypeError:  # older core without frame-count awareness
                est = int(estimate(pixels))
        else:
            est = pixels * per_px
        detail = (f'HR output {ow}x{oh} = {pixels:,} px at ratio {ratio:g} '
                  f'(ROI {rw}x{rh}, {n_frames} frames); estimated peak '
                  f'{est / 2 ** 30:.1f} GiB, budget {budget / 2 ** 30:.1f} GiB')
        if pixels > limit:
            raise HTTPException(
                status_code=413,
                detail=('Not enough memory for this request: ' + detail
                        + '. Reduce the SR ratio or the ROI and try again, or '
                          'raise the memory budget (memory_budget_bytes).'))
        if est > warn_frac * budget:
            return detail + ' — close to the memory limit.'
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - a probe must never block a valid run
        return None
    return None


def _run_with_memory_note(sr: Any, frames: List[Any]) -> Tuple[Any, Optional[str]]:
    """Run the core, capturing its HR-size RuntimeWarning as a non-fatal note.

    The core warns (does not raise) between 75 % and 100 % of the memory budget.
    A warning normally goes to stderr and is lost with the job; capturing it
    lets the report carry ``memory_warning`` so the UI can say "this ran, but it
    was close to the limit" instead of silently finishing.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        res = sr.run(frames)
    for entry in caught:
        if issubclass(entry.category, RuntimeWarning):
            note = str(entry.message).strip()
            if note:
                return res, note
    return res, None


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class FilesRequest(BaseModel):
    paths: List[str]


class PreviewRequest(BaseModel):
    path: str = Field(min_length=1)
    # The UI never asks above 1024; 2048 bounds one decode-cache entry to a
    # few MB (8192 let a single entry reach hundreds of MB). See ARCHITECTURE.md.
    max_dim: int = Field(default=1024, ge=16, le=2048)


class AnalyzeRequest(BaseModel):
    paths: List[str]
    target_roi: Optional[Tuple[int, int, int, int]] = None
    search_area: Optional[Tuple[int, int, int, int]] = None
    sr_ratio: float = Field(default=2.0, gt=0.0, le=MAX_SR_RATIO)
    k_threshold: float = Field(default=3.0, ge=0.0)

    @field_validator('target_roi', 'search_area')
    @classmethod
    def _roi_positive(cls, v):
        if v is not None and (v[2] <= 0 or v[3] <= 0):
            raise ValueError('ROI width and height must be positive')
        return v


class ProcessRequest(BaseModel):
    paths: List[str]
    target_roi: Optional[Tuple[int, int, int, int]] = None
    sr_ratio: float = Field(default=2.0, gt=0.0, le=MAX_SR_RATIO)
    registration: str = 'homography'
    exclude: List[str] = Field(default_factory=list)
    # Brand-agnostic GPU acceleration via OpenCL (see ARCHITECTURE.md).
    # None (default) = auto: the core enables OpenCL iff a device exists.
    use_opencl: Optional[bool] = None
    # Robust (Tukey) fusion that suppresses moving-subject ghost trails.
    # None (default) = auto: the core detects a moving subject / parallax and
    # decides; True = always robust weights; False = plain fusion.
    robust_fusion: Optional[bool] = None
    # HR-size memory guard (see docs in raw_sr/core/superres.py): the core
    # refuses an HR output above this budget and warns above 75 % of it. None
    # (default) = the core's own budget (60 % of MemAvailable, clamped 2-6 GiB).
    # Exposed so a workstation owner can raise the ceiling instead of guessing.
    memory_budget_bytes: Optional[int] = Field(default=None, ge=1)

    @field_validator('target_roi')
    @classmethod
    def _roi_positive(cls, v):
        if v is not None and (v[2] <= 0 or v[3] <= 0):
            raise ValueError('ROI width and height must be positive')
        return v


class RenderRequest(BaseModel):
    gamma: float = Field(default=2.2, gt=0.0, le=10.0)
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=4.0)


class ViewTransformModel(BaseModel):
    gamma: float = Field(default=2.2, gt=0.0, le=10.0)
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=4.0)


class ExportRequest(BaseModel):
    job_id: str = Field(min_length=1)
    kind: str
    output_path: str = Field(min_length=1)
    view_transform: Optional[ViewTransformModel] = None
    enhance_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enhance_preset: Optional[str] = None


class QualityRequest(BaseModel):
    paths: List[str]
    threshold_ratio: float = Field(default=0.5, gt=0.0, le=1.0)


class SyntheticRequest(BaseModel):
    ratio: float = Field(default=2.0, gt=0.0, le=MAX_SR_RATIO)
    n_frames: int = Field(default=5, ge=2, le=32)


# --------------------------------------------------------------------------- #
# ROI resolution (mirrors raw_sr.ui.worker without the PySide6 dependency)
# --------------------------------------------------------------------------- #

def _frame_size(path: str) -> Tuple[int, int]:
    import rawpy
    with rawpy.imread(path) as raw:
        return int(raw.sizes.height), int(raw.sizes.width)


def resolve_roi(path: str, target_roi: Optional[Tuple[int, int, int, int]],
                ratio: float = 1.0,
                max_roi_area: int = MAX_ROI_AREA,
                max_hr_area: int = MAX_HR_AREA) -> Tuple[Tuple[int, int, int, int], bool]:
    h, w = _frame_size(path)
    if target_roi is None:
        x, y, rw, rh = 0, 0, w, h
    else:
        x, y, rw, rh = (int(v) for v in target_roi)
        rw, rh = max(rw, 1), max(rh, 1)
    clipped = False
    cap = min(max_roi_area, max_hr_area / max(float(ratio), 1e-6) ** 2)
    cap = max(1.0, cap)
    if rw * rh > cap:
        # Shrink the requested ROI to fit the cap *around its own centre*, so
        # the user's subject stays in the frame instead of being replaced by a
        # centred square of the whole sensor.
        scale = math.sqrt(cap / float(rw * rh))
        nw = max(1, int(rw * scale))
        nh = max(1, int(rh * scale))
        while nw * nh > cap:
            if nw >= nh:
                nw -= 1
            else:
                nh -= 1
        x = x + (rw - nw) // 2
        y = y + (rh - nh) // 2
        rw, rh = max(1, nw), max(1, nh)
        clipped = True
    rw = min(rw, w)
    rh = min(rh, h)
    x = min(max(x, 0), w - rw)
    y = min(max(y, 0), h - rh)
    return (x, y, rw, rh), clipped


# --------------------------------------------------------------------------- #
# SR core construction
# --------------------------------------------------------------------------- #

def _make_superres(**kwargs: Any) -> Any:
    """Build a ``SuperResolution`` passing only arguments the core accepts.

    The SR core evolves in parallel, so we never hard-code a constructor
    argument that may not exist. Keyword arguments unknown to the installed
    ``SuperResolution`` are dropped (with ``inspect.signature``), letting the
    core's own defaults win. ``None`` values are dropped for the same reason:
    e.g. ``use_opencl=None`` simply means "not requested" and the core
    auto-detects (GPU via OpenCL when a device exists).

    In particular the desktop path deliberately does **not** override
    ``ecc_scale``: the legacy PySide worker pinned ``ecc_scale=0.25``, which
    locked less reliably than the core's improved default of ``0.5``. We rely
    on the core defaults (``ecc_scale=0.5``, ``registration='homography'``)
    rather than re-introducing the worse value.
    """
    import inspect
    from raw_sr.core.superres import SuperResolution

    params = inspect.signature(SuperResolution.__init__).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return SuperResolution(**accepted)


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #

def _alignment_verdict(mean_zncc: float, mean_residual: float) -> Tuple[float, str, str]:
    """Map mean ZNCC + mean residual (px) to (score, level, message).

    ``score`` is a monotonic [0, 1] "how well did the frames align" measure that
    blends both signals::

        score = 0.6 * mean_zncc + 0.4 * exp(-mean_residual / 1px)

    It rises with the mean zero-normalized cross-correlation and falls with the
    mean alignment residual in pixels (0 px contributes 0.4, 1 px ~0.15, large
    residuals ~0). ``level`` thresholds are::

        ok   score >= 0.75
        warn 0.50 <= score < 0.75
        bad  score <  0.50
    """
    z = float(np.clip(mean_zncc, 0.0, 1.0))
    r = float(mean_residual)
    if not math.isfinite(r):
        r = 1e9
    r = max(0.0, r)
    score = float(np.clip(0.6 * z + 0.4 * math.exp(-r), 0.0, 1.0))
    if score >= 0.75:
        level, verdict = 'ok', 'Excellent alignment'
    elif score >= 0.5:
        level, verdict = 'warn', 'Fair alignment'
    else:
        level, verdict = 'bad', 'Poor alignment'
    return score, level, f'{verdict} (mean ZNCC {z:.2f}, {r:.2f} px residual)'


def _alignment_quality_map(greens: List[np.ndarray], transforms: List[Any],
                           ratio: float, max_dim: int = MAX_DISPLAY_DIM) -> str:
    """One low-resolution PNG data URL of per-pixel alignment quality.

    Each kept frame's green/luma channel is warped onto the HR grid with the
    same drizzle transform used for fusion (:func:`to_drizzle_matrix`), then the
    per-pixel agreement across the warped frames is ``1 - std/scale``, where
    ``scale`` is the 95th percentile of the per-pixel inter-frame std over the
    covered pixels. Low inter-frame disagreement therefore reads as good
    alignment. Pixels no frame covers are zeroed. The stack is computed at a
    reduced LR size so the HR working grid stays within ``max_dim`` — cheap and
    exactly what the low-res heatmap needs.
    """
    from raw_sr.core.superres import to_drizzle_matrix

    if not greens or not transforms:
        return heatmap(np.zeros((2, 2), dtype=np.float32), max_dim)

    h, w = greens[0].shape[:2]
    hr_h, hr_w = max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))
    scale = min(1.0, float(max_dim) / float(max(hr_h, hr_w)))

    S = np.diag([scale, scale, 1.0])
    S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
    ww, wh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    oh, ow = max(1, int(round(wh * ratio))), max(1, int(round(ww * ratio)))

    ones_lr = np.ones((wh, ww), dtype=np.float32)
    warped: List[np.ndarray] = []
    valid = np.zeros((oh, ow), dtype=np.float32)
    for g, W in zip(greens, transforms):
        Wn = np.asarray(W, dtype=np.float64)
        if Wn.shape != (3, 3):
            Wn = np.eye(3, dtype=np.float64)
        if abs(float(Wn[2, 2])) > 1e-12:
            Wn = Wn / Wn[2, 2]
        Ww = S @ Wn @ S_inv
        if abs(float(Ww[2, 2])) > 1e-12:
            Ww = Ww / Ww[2, 2]
        mat = to_drizzle_matrix(Ww, float(ratio))
        small = cv2.resize(np.asarray(g, dtype=np.float32), (ww, wh),
                           interpolation=cv2.INTER_AREA)
        warped.append(cv2.warpPerspective(small, mat, (ow, oh),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT))
        valid += cv2.warpPerspective(ones_lr, mat, (ow, oh),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)

    stack = np.stack(warped, axis=0)
    std = stack.std(axis=0).astype(np.float32)
    covered = valid > 0.5
    quality = np.zeros((oh, ow), dtype=np.float32)
    if np.any(covered):
        span = float(np.percentile(std[covered], 95))
        if span <= 1e-9:
            span = float(std[covered].max()) + 1e-6
        quality[covered] = np.clip(1.0 - std[covered] / span, 0.0, 1.0)
    return heatmap(quality, max_dim)


def _build_alignment(res: Any, paths: List[str],
                     greens: List[np.ndarray]) -> Dict[str, Any]:
    """Assemble the ``alignment`` object (see ARCHITECTURE.md).

    ``per_frame`` has one entry per frame fed to the core, in ``paths`` order,
    with the session frame id when the id-map recorded it at
    ``/api/session/files`` time (else the index into ``paths`` as a string).
    ZNCC/residual/shift come from the core for kept frames and are zero for
    dropped ones.
    """
    kept = [int(i) for i in getattr(res, 'kept_indices', [])]
    scores = [float(v) for v in getattr(res, 'align_scores', [])]
    residuals = [float(v) for v in getattr(res, 'align_residuals', [])]
    shifts = list(getattr(res, 'shifts', []))

    id_map = session.frame_id_map()
    path_to_id: Dict[str, str] = {}
    for fid, p in id_map.items():
        path_to_id.setdefault(p, fid)

    kept_pos = {idx: k for k, idx in enumerate(kept)}
    per_frame: List[Dict[str, Any]] = []
    for i, path in enumerate(paths):
        pos = kept_pos.get(i)
        used = pos is not None
        zncc = float(np.clip(scores[pos], 0.0, 1.0)) if used and pos < len(scores) else 0.0
        residual = float(residuals[pos]) if used and pos < len(residuals) else 0.0
        if not math.isfinite(residual):
            residual = 0.0
        if used and pos < len(shifts):
            shift = [float(shifts[pos][0]), float(shifts[pos][1])]
        else:
            shift = [0.0, 0.0]
        per_frame.append({
            'id': path_to_id.get(path, str(i)),
            'shift': shift,
            'zncc': zncc,
            'residual': residual,
            'used': bool(used),
        })

    kept_scores = [float(np.clip(s, 0.0, 1.0)) for s in scores]
    finite_res = [float(r) for r in residuals if math.isfinite(float(r))]
    mean_zncc = float(np.mean(kept_scores)) if kept_scores else 0.0
    min_zncc = float(np.min(kept_scores)) if kept_scores else 0.0
    mean_residual = float(np.mean(finite_res)) if finite_res else 0.0

    score, level, message = _alignment_verdict(mean_zncc, mean_residual)

    kept_greens = [np.asarray(greens[i]) for i in kept if 0 <= i < len(greens)]
    quality_map = _alignment_quality_map(kept_greens,
                                         list(getattr(res, 'transforms', [])),
                                         float(getattr(res, 'ratio', 1.0)))

    return {
        'score': score,
        'level': level,
        'message': message,
        'mean_zncc': mean_zncc,
        'min_zncc': min_zncc,
        'per_frame': per_frame,
        'map': quality_map,
    }


def _is_number(v: Any) -> bool:
    """True for a real JSON number (bools are ints in Python, but not here)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _normalize_refinement_metrics(metrics: Dict[str, Any]) -> None:
    """Additive, backward-compatible refinement/deposit keys (in place).

    Current core builds report ``ibp_*`` / ``drizzle_pixfrac`` /
    ``drizzle_subsamples`` / ``reference_index`` (see
    docs/IBP_AND_HELDOUT.md). Older builds may omit some or all: never invent
    a configuration, only derive the sweep count from ``ibp_iterations`` when
    that key is present, and otherwise fall back to ``''`` / ``None`` so the UI
    can omit the line.
    """
    iterations = metrics.get('ibp_iterations')
    if not (isinstance(iterations, int) and not isinstance(iterations, bool)
            and iterations >= 0):
        iterations = None
    metrics['ibp_iterations'] = iterations

    summary = metrics.get('ibp_summary')
    if not isinstance(summary, str):
        summary = (f'{iterations} sweep{"" if iterations == 1 else "s"}'
                   if iterations else '')
    metrics['ibp_summary'] = summary.strip()

    pixfrac = metrics.get('drizzle_pixfrac')
    metrics['drizzle_pixfrac'] = (float(pixfrac)
                                  if _is_number(pixfrac) and math.isfinite(float(pixfrac))
                                  and float(pixfrac) > 0 else None)
    subsamples = metrics.get('drizzle_subsamples')
    metrics['drizzle_subsamples'] = (int(subsamples)
                                     if isinstance(subsamples, int)
                                     and not isinstance(subsamples, bool)
                                     and subsamples >= 1 else None)
    ref = metrics.get('reference_index')
    metrics['reference_index'] = (int(ref) if isinstance(ref, int)
                                  and not isinstance(ref, bool) and ref >= 0
                                  else None)


def _normalize_fusion_metrics(metrics: Dict[str, Any]) -> None:
    """Additive fusion-weight diagnostics, type-checked in place.

    ``frame_noise_sigma`` is the per-kept-frame noise estimate the fusion
    weight used (``_fusion_weight`` = sharpness / noise variance) and
    ``fusion_weight_spread`` is max/min of the normalised weights (1.0 =
    uniform). Both describe *this* run's weighting and are what makes a mixed
    noise burst readable; a core that predates them yields ``None`` so the UI
    omits the line instead of inventing a value.
    """
    sigma = metrics.get('frame_noise_sigma')
    if isinstance(sigma, (list, tuple)):
        values = [float(v) for v in sigma
                  if _is_number(v) and math.isfinite(float(v)) and float(v) >= 0.0]
        metrics['frame_noise_sigma'] = values or None
    else:
        metrics['frame_noise_sigma'] = None

    spread = metrics.get('fusion_weight_spread')
    metrics['fusion_weight_spread'] = (
        float(spread) if _is_number(spread) and math.isfinite(float(spread))
        and float(spread) >= 0.0 else None)


def _add_honest_single_frame_metrics(metrics: Dict[str, Any], hr: np.ndarray,
                                     before: np.ndarray) -> None:
    """Additive detail/acutance numbers measured against a *single* frame.

    ``hf_gain_db`` (from the core) compares the HR result with a bicubic
    upscale of the denoised LR **stack** — an already-averaged baseline, so it
    reads as acutance, not as new resolution. The honest resolution figure is
    the same band measured against one frame upscaled (the
    ``scripts/sr_acceptance.py`` convention, ``hf_gain_db_vs_single``): real
    multi-frame SR puts energy above the single-frame Nyquist there, a
    bicubic-upsampled single frame has essentially none. Both keys are
    nullable: a core/build without the helpers leaves them out.
    """
    try:
        from raw_sr.core.superres import (high_frequency_gain_db,
                                          laplacian_sharpness, to_view_float)
        hf = high_frequency_gain_db(to_view_float(hr), to_view_float(before))
        sharp = (laplacian_sharpness(hr)
                 / max(laplacian_sharpness(before), 1e-9))
        metrics['hf_gain_db_vs_single'] = float(hf) if math.isfinite(hf) else None
        metrics['sharpness_gain_vs_single'] = (float(sharp)
                                               if math.isfinite(sharp) else None)
    except Exception:  # noqa: BLE001 - metrics must never break a finished job
        metrics['hf_gain_db_vs_single'] = None
        metrics['sharpness_gain_vs_single'] = None


def _store_result(job: Job, res: Any, before_rgb: np.ndarray, roi: Tuple[int, int, int, int],
                  clipped: bool, paths: Optional[List[str]] = None,
                  greens: Optional[List[np.ndarray]] = None,
                  opencl_reason: Optional[str] = None,
                  robust_reason: Optional[str] = None,
                  memory_warning: Optional[str] = None) -> None:
    hr = res.hr
    disp = fit_max_dim(hr, MAX_DISPLAY_DIM).astype(np.float32)
    # `before` is the reference frame upscaled to the HR size. `before_rgb` is
    # already the reference rendered through the app's preview pipeline (see
    # session.load_reference_linear) so it carries no DCB sharpening; bicubic
    # upscaling to the HR grid is the baseline the fused result must beat.
    before = cv2.resize(before_rgb, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    before_disp = fit_max_dim(before, MAX_DISPLAY_DIM).astype(np.float32)

    # Engine visibility (result-metrics contract, see desktop/ARCHITECTURE.md):
    # the core's metrics pass through untouched, plus exactly two additive
    # keys so the UI can say plainly which engine fused this burst.
    # None-safe: a core build that predates the keys reports 'unavailable' /
    # null instead of leaving the UI guessing. No other field is added or
    # altered here.
    metrics: Dict[str, Any] = dict(getattr(res, 'metrics', None) or {})
    if metrics.get('opencl_scatter') not in ('cpu', 'gpu', 'unavailable'):
        metrics['opencl_scatter'] = 'unavailable'
    if not isinstance(metrics.get('opencl_device'), str) or not metrics['opencl_device']:
        metrics['opencl_device'] = None
    # Additive: one line naming *why* the fusion engine is on the GPU or the
    # CPU (see _job_opencl_reason / _engine_status). Older core metrics stay
    # untouched; a pre-existing opencl_reason (future core) is preserved.
    if opencl_reason and not metrics.get('opencl_reason'):
        metrics['opencl_reason'] = opencl_reason
    # Additive: the ghost-suppression verdict. The core supplies
    # robust_active/robust_reason (True/False + a one-line why for auto/on/off);
    # a core that predates the keys reports plain fusion / 'unavailable' rather
    # than leaving the UI guessing. No other field is altered here.
    if not isinstance(metrics.get('robust_active'), bool):
        metrics['robust_active'] = False
    if robust_reason and not metrics.get('robust_reason'):
        metrics['robust_reason'] = robust_reason
    if not isinstance(metrics.get('robust_reason'), str) or not metrics['robust_reason']:
        metrics['robust_reason'] = 'unavailable'
    # Additive: the refinement (IBP) and drizzle-deposit configuration the core
    # actually ran (see docs/IBP_AND_HELDOUT.md). `ibp_iterations` changed
    # default 3 -> 1, so these keys are what makes two runs comparable. A core
    # that predates them reports honest null/'' fallbacks; the panel then hides
    # the line instead of printing defaults that did not run.
    _normalize_refinement_metrics(metrics)
    # Additive: the fusion-weight diagnostics of this run (frame noise sigma
    # and weight spread) and the honest detail figure measured against a
    # single frame instead of the denoised stack.
    _normalize_fusion_metrics(metrics)
    _add_honest_single_frame_metrics(metrics, hr, before)

    report = {
        'used_frames': int(res.used_frames),
        'total_frames': int(res.total_frames),
        'ratio': float(res.ratio),
        'hr_shape': [int(v) for v in hr.shape],
        'roi': [int(v) for v in roi],
        'roi_clipped': bool(clipped),
        'metrics': metrics,
        'low_coverage_warning': bool(res.used_frames < 0.2 * max(res.total_frames, 1)),
        # Non-fatal HR-size memory heads-up from the core's guard (None when the
        # run was comfortably inside the budget). The job still succeeded.
        'memory_warning': (str(memory_warning).strip() if memory_warning else None),
    }

    job.data.update({
        'hr': hr,
        'disp_linear': disp,
        'before_disp': before_disp,
        'coverage': res.coverage,
        'transforms': res.transforms,
        'weights': np.asarray(res.weights, dtype=np.float64).tolist(),
        'before_rgb': before_rgb,
        'ratio': float(res.ratio),
        'report': report,
        'view_transform': dict(DEFAULT_VIEW),
    })
    job.data['alignment'] = _build_alignment(res, list(paths or []),
                                             list(greens or []))


def _render_result(job: Job, vt: Dict[str, float]) -> Dict[str, Any]:
    disp: np.ndarray = job.data['disp_linear']
    before: np.ndarray = job.data['before_disp']

    fused_u8 = apply_view(disp, vt['gamma'], vt['brightness'], vt['contrast'])
    before_u8 = apply_view(before, vt['gamma'], vt['brightness'], vt['contrast'])
    return {
        'fused': encode_png(fused_u8),
        'before': encode_png(before_u8),
        'width': int(disp.shape[1]),
        'height': int(disp.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get('/api/health')
def health() -> Dict[str, Any]:
    return {'status': 'ok', 'version': '1.0.0'}


_capabilities_cache: Optional[Dict[str, Any]] = None


def _opencl_device_info() -> Tuple[Optional[str], Optional[str]]:
    """(name, vendor) of the OpenCL device OpenCV T-API selected, else (None, None).

    ``cv2.ocl.Device.getDefault()`` initialises (or reuses) OpenCV's default
    context, so this reports the device the *server actually resolved* under
    the OPENCV_OPENCL_DEVICE pin above — not a re-probe. OpenCV 4.x bindings
    without the Device class degrade to (None, None).
    """
    try:
        dev_cls = getattr(cv2.ocl, 'Device', None)
        if dev_cls is None or not cv2.ocl.haveOpenCL():
            return None, None
        dev = dev_cls.getDefault()

        def _text(attr: str) -> Optional[str]:
            val = getattr(dev, attr, None)
            if callable(val):  # OpenCV 5 exposes these as methods
                val = val()
            return val if isinstance(val, str) and val else None

        return _text('name'), _text('vendorName')
    except Exception:  # noqa: BLE001 - probe must never break the API
        return None, None


# --------------------------------------------------------------------------- #
# Engine diagnostics: *why* the fusion engine is on the GPU or the CPU
#
# "GPU is fast for me but slow for the user" is almost always a silent gate:
# pyopencl missing in the interpreter that actually runs, no OpenCL device, or
# the one-time auto-calibration measuring the CPU faster (the documented
# outcome on an RTX 3050 Laptop). These helpers turn that invisible decision
# into a readable status dict so the UI — and a bug report — can name the
# cause instead of guessing.
# --------------------------------------------------------------------------- #

_engine_status_cache: Optional[Dict[str, Any]] = None
_pyopencl_probe_cache: Optional[Dict[str, Any]] = None


def _pyopencl_probe() -> Dict[str, Any]:
    """Inspect pyopencl once: importability, platforms and the fastest GPU.

    Never raises. A machine without a GPU/ICD reports ``pyopencl=False`` (or a
    populated ``pyopencl_error``) and every caller degrades to the CPU path.
    """
    global _pyopencl_probe_cache
    if _pyopencl_probe_cache is not None:
        return dict(_pyopencl_probe_cache)
    info: Dict[str, Any] = {
        'pyopencl': False,
        'pyopencl_error': None,
        'opencl_platforms': [],
        'gpu_device': None,
    }
    try:
        import pyopencl as cl
    except Exception as exc:  # noqa: BLE001 - absence is a normal CPU state
        info['pyopencl_error'] = f'{type(exc).__name__}: {exc}'
        _pyopencl_probe_cache = info
        return dict(info)
    info['pyopencl'] = True
    platforms: List[Any] = []
    try:
        platforms = list(cl.get_platforms())
    except Exception as exc:  # noqa: BLE001
        info['pyopencl_error'] = f'{type(exc).__name__}: {exc}'
    info['opencl_platforms'] = [str(getattr(p, 'name', p)) for p in platforms]
    devices: List[Any] = []
    for plat in platforms:
        try:
            devices.extend(plat.get_devices(device_type=cl.device_type.GPU))
        except Exception:  # noqa: BLE001 - a platform may expose no GPU
            continue
    if devices:
        # Same pick as the core's scatter engine: the device with most memory.
        best = max(devices, key=lambda d: int(d.global_mem_size))
        info['gpu_device'] = str(best.name).strip() or None
    _pyopencl_probe_cache = info
    return dict(info)


def _opencl_gate(use_opencl: Optional[bool]) -> Tuple[str, str]:
    """The fusion-engine gate for a request: ``(decision, reason)``.

    ``decision`` is ``'gpu'`` when the scatter deposition will run on the GPU
    and ``'cpu'`` when it declines; ``reason`` is a one-line human explanation
    (``'pyopencl not importable in <exe>'``, ``'no OpenCL device'``,
    ``'calibration: CPU faster'``, ``'forced by request'``). Mirrors the core's
    own tri-state gate in ``SuperResolution.run``.
    """
    probe = _pyopencl_probe()
    if use_opencl is False:
        return 'cpu', 'forced off by request'
    if not probe['pyopencl']:
        detail = probe['pyopencl_error'] or 'import failed'
        return 'cpu', f'pyopencl not importable in {sys.executable} ({detail})'
    if probe['gpu_device'] is None:
        plats = ', '.join(probe['opencl_platforms']) or 'none'
        return 'cpu', f'no OpenCL GPU device (platforms: {plats})'
    if use_opencl is True:
        return 'gpu', 'forced by request'
    try:
        from raw_sr.core import superres as core_sr
        if core_sr.gpu_scatter_beneficial():
            return 'gpu', f'GPU scatter faster on {probe["gpu_device"]}'
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break a run
        return 'cpu', f'calibration failed ({type(exc).__name__}: {exc})'
    return 'cpu', 'calibration: CPU faster'


def _job_opencl_reason(use_opencl: Optional[bool], res: Any) -> str:
    """Gate reason recorded in a job's result metrics as ``opencl_reason``.

    Reports what actually happened when an engine ran (the core's
    ``opencl_scatter`` / ``opencl_active``), falling back to the pre-run gate
    reason when the fusion declined to use the GPU.
    """
    metrics = getattr(res, 'metrics', None) or {}
    used_gpu = (metrics.get('opencl_scatter') == 'gpu'
                or bool(metrics.get('opencl_active')))
    if used_gpu:
        if use_opencl is True:
            return 'forced by request'
        dev = (metrics.get('opencl_device')
               or _pyopencl_probe()['gpu_device'] or 'GPU')
        return f'GPU accelerated on {dev}'
    return _opencl_gate(use_opencl)[1]


def _job_robust_reason(robust_fusion: Optional[bool], res: Any) -> str:
    """Ghost-suppression reason recorded in a job's metrics as ``robust_reason``.

    The core reports its own ``robust_reason`` when it made the call (the auto
    path that detects a moving subject / parallax); this only fills a fallback
    for a core build that predates the key, naming the request override when
    one was sent.
    """
    metrics = getattr(res, 'metrics', None) or {}
    reason = metrics.get('robust_reason')
    if isinstance(reason, str) and reason:
        return reason
    if robust_fusion is True:
        return 'forced on by request'
    if robust_fusion is False:
        return 'forced off by request'
    return 'unavailable'


def _engine_status() -> Dict[str, Any]:
    """Startup snapshot of the fusion-engine capability surface.

    Additive and non-breaking; cached after the first call. ``gate_decision`` /
    ``gate_reason`` describe the *auto* (no override) gate, which is what a
    fresh launch with the default GPU mode uses.
    """
    global _engine_status_cache
    if _engine_status_cache is not None:
        return dict(_engine_status_cache)
    probe = _pyopencl_probe()
    decision, reason = _opencl_gate(None)
    _engine_status_cache = {
        'python_executable': sys.executable,
        'pyopencl': bool(probe['pyopencl']),
        'pyopencl_error': probe['pyopencl_error'],
        'opencl_platforms': list(probe['opencl_platforms']),
        'gpu_device': probe['gpu_device'],
        'nvidia_smi': bool(_nvidia_gpu_present()),
        'pinned_device': _PINNED_DEVICE,
        'gate_decision': decision,
        'gate_reason': reason,
    }
    return dict(_engine_status_cache)


# Compute once at startup so /api/capabilities and the launch log can name the
# gate immediately. Never fatal: a probe failure leaves the CPU defaults.
try:  # pragma: no cover - exercised via /api/capabilities
    _engine_status()
except Exception:  # noqa: BLE001
    pass


@app.get('/api/capabilities')
def capabilities() -> Dict[str, Any]:
    """Runtime hardware capabilities (additive, non-breaking).

    ``opencl``: whether an OpenCL device is available to OpenCV T-API
    (brand-agnostic GPU path); cached because ``cv2.ocl.haveOpenCL()`` probes
    the device layer. ``cpu_count``: host logical cores.
    ``opencl_device``/``opencl_vendor``: name/vendor of the device OpenCV
    actually selected (see :func:`_opencl_device_info`), null when none.
    ``engine_status``: the fusion-engine gate snapshot (see
    :func:`_engine_status`) so a client can explain a CPU verdict.
    """
    global _capabilities_cache
    if _capabilities_cache is None:
        try:
            opencl = bool(cv2.ocl.haveOpenCL())
        except Exception:  # noqa: BLE001 - probe must never break the API
            opencl = False
        device, vendor = _opencl_device_info() if opencl else (None, None)
        _capabilities_cache = {'opencl': opencl, 'cpu_count': os.cpu_count() or 1,
                               'opencl_device': device, 'opencl_vendor': vendor,
                               'engine_status': _engine_status()}
    return dict(_capabilities_cache)


@app.post('/api/session/files')
def session_files(req: FilesRequest) -> Dict[str, Any]:
    _require_non_empty(req.paths)
    frames, rejected = session.build_frame_infos(req.paths)
    return {'frames': frames, 'rejected': rejected}


@app.post('/api/session/preview')
def session_preview(req: PreviewRequest) -> Dict[str, Any]:
    if not os.path.isfile(req.path):
        raise HTTPException(status_code=404, detail='File not found')
    if not session.is_raw(req.path):
        raise HTTPException(status_code=400, detail='Not a RAW file')
    try:
        url, w, h = session.preview_data_url(req.path, req.max_dim)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'Could not decode RAW: {exc}') from exc
    return {'preview': url, 'width': w, 'height': h}


@app.post('/api/analyze')
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    _require_non_empty(req.paths)
    return session.analyze(req.paths, req.target_roi, req.search_area, req.sr_ratio,
                           req.k_threshold)


@app.post('/api/quality')
def quality(req: QualityRequest) -> Dict[str, Any]:
    _require_non_empty(req.paths)
    return session.quality(req.paths, req.threshold_ratio)


@app.post('/api/process')
def process(req: ProcessRequest) -> Dict[str, Any]:
    _require_non_empty(req.paths)
    _require_registration(req.registration)
    _require_readable_raw(req.paths)

    # `exclude` holds frame ids from POST /api/session/files; resolve them to a
    # subset of the provided paths. Unknown/malformed ids are ignored.
    effective_paths = session.resolve_exclude(req.paths, req.exclude, session.frame_id_map())
    excluded = len(req.paths) - len(effective_paths)
    if not effective_paths:
        raise HTTPException(status_code=400, detail='All frames were excluded')

    # HR-size memory guard: refuse an impossible ratio/ROI combination with a
    # clean 413 before decoding, and remember a near-limit run as a note.
    memory_warning = _memory_guard(effective_paths[0], req.target_roi,
                                   req.sr_ratio, req.memory_budget_bytes,
                                   n_frames=len(effective_paths))

    # Single flight: the SR run pins every decoded frame in RAM, so a second
    # concurrent process job would double the peak. Release happens exactly
    # once via the job's on_finish hook (done / error / cancelled alike).
    if not _PROCESS_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='A process job is already running')

    from raw_sr.core.superres import load_raw_linear

    def run(job: Job) -> None:
        roi, clipped = resolve_roi(effective_paths[0], req.target_roi, ratio=req.sr_ratio)
        n = len(effective_paths)
        frames = []
        for i, path in enumerate(effective_paths):
            if job.cancel_requested:
                return
            job.progress = 5 + int(45 * i / max(n, 1))
            job.message = f'Loading frame {i + 1}/{n}…'
            # NOTE: a single rawpy decode of a large RAW is not interruptible.
            frames.append(load_raw_linear(path, roi=roi))
        if job.cancel_requested:
            return
        job.progress = 55
        job.message = 'Registering frames…'
        # NOTE: the SR core's registration + fusion run is not interruptible
        # mid-call; cancellation is honored once it returns.
        # `use_opencl` is forwarded only when the request explicitly set it;
        # `_make_superres` additionally drops it when the installed core does
        # not accept the argument yet (None = core auto-detects).
        sr_kwargs: Dict[str, Any] = {'ratio': req.sr_ratio, 'registration': req.registration}
        if req.use_opencl is not None:
            sr_kwargs['use_opencl'] = req.use_opencl
        if req.robust_fusion is not None:
            sr_kwargs['robust_fusion'] = req.robust_fusion
        # Same ceiling in the core as the endpoint pre-check, so a run cannot
        # pass the pre-check and then exceed the guard (or vice versa).
        if req.memory_budget_bytes is not None:
            sr_kwargs['memory_budget_bytes'] = req.memory_budget_bytes
        sr = _make_superres(**sr_kwargs)
        res, run_warning = _run_with_memory_note(sr, frames)
        if job.cancel_requested:
            return
        job.progress = 90
        job.message = 'Building report…'
        # Memory: release the N full-res frames before building the result —
        # holding them through the before/report phases would keep the whole
        # decoded burst pinned while `hr` + display arrays are allocated on
        # top. The later steps need only: the greens (``RawFrame.green`` is
        # already a detached per-frame copy) and `before_rgb` below.
        greens = [f.green for f in frames]
        # The honest `before` is the **reference frame** — the frame whose LR
        # grid the HR output lives on (``res.reference_index``) — decoded
        # through the preview pipeline (default demosaic, no DCB sharpening)
        # rather than the DCB SR-core frame. Using frames[0] here left the
        # before/after panels offset by that frame's registration shift.
        # Fall back to the loaded frame if the re-decode fails — as an explicit
        # COPY, so the stored result never pins a full frame.
        ref_i = getattr(res, 'reference_index', None)
        if ref_i is None or not (0 <= int(ref_i) < len(effective_paths)):
            ref_i = 0
        ref_i = int(ref_i)
        try:
            before_rgb = session.load_reference_linear(effective_paths[ref_i],
                                                       roi=roi)
        except Exception:  # noqa: BLE001
            before_rgb = np.array(frames[ref_i].rgb)
        frames.clear()
        _store_result(job, res, before_rgb, roi, clipped,
                      paths=effective_paths, greens=greens,
                      opencl_reason=_job_opencl_reason(req.use_opencl, res),
                      robust_reason=_job_robust_reason(req.robust_fusion, res),
                      memory_warning=memory_warning or run_warning)

    try:
        job = manager.create(run, on_finish=_PROCESS_LOCK.release)
    except Exception:
        _PROCESS_LOCK.release()
        raise
    if excluded:
        job.message = f'Queued ({excluded} frame(s) excluded)'
    return {'job_id': job.id}


@app.post('/api/dev/synthetic')
def dev_synthetic(req: SyntheticRequest) -> Dict[str, Any]:
    """Run the real SR core on an in-memory synthetic burst (no RAW files).

    Used by the end-to-end smoke test when no DNG samples are present.
    """
    from raw_sr.core.superres import RawFrame

    def run(job: Job) -> None:
        if job.cancel_requested:
            return
        job.progress = 20
        job.message = 'Building synthetic burst…'
        truth = _bandpass_truth(192, 192)
        h, w = truth.shape[0] // 2, truth.shape[1] // 2
        shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75)]
        frames = []
        for i, (dx, dy) in enumerate(shifts[:max(req.n_frames, 2)]):
            lr = _sample_lr(truth, dx, dy, (h, w))
            frames.append(RawFrame(name=f'syn{i}', rgb=lr, green=lr[:, :, 1].copy()))
        if job.cancel_requested:
            return
        job.progress = 55
        job.message = 'Running SR core…'
        res, run_warning = _run_with_memory_note(
            _make_superres(ratio=req.ratio, blur_threshold_ratio=0.0), frames)
        if job.cancel_requested:
            return
        _store_result(job, res, frames[0].rgb, (0, 0, w, h), False,
                      paths=[f.name for f in frames], greens=[f.green for f in frames],
                      opencl_reason=_job_opencl_reason(None, res),
                      robust_reason=_job_robust_reason(None, res),
                      memory_warning=run_warning)

    job = manager.create(run)
    return {'job_id': job.id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str) -> Dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Unknown job')
    return {'state': job.state, 'progress': job.progress, 'message': job.message,
            'error': job.error,
            # Additive: 'memory' when the run was refused for an HR size the
            # machine cannot hold, so a client can word the failure as an
            # out-of-memory (not a crash) and point at ratio/ROI.
            'error_kind': job.error_kind}


@app.post('/api/jobs/{job_id}/cancel')
def job_cancel(job_id: str) -> Dict[str, Any]:
    if not manager.cancel(job_id):
        raise HTTPException(status_code=404, detail='Unknown job')
    return {'cancelled': True}


@app.get('/api/jobs/{job_id}/result')
def job_result(job_id: str) -> Dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Unknown job')
    if job.state != 'done':
        raise HTTPException(status_code=409, detail=f'Job is {job.state}')
    rendered = _render_result(job, job.data['view_transform'])
    return {
        'report': job.data['report'],
        'images': {
            'fused': rendered['fused'],
            'before': rendered['before'],
            'width': rendered['width'],
            'height': rendered['height'],
        },
        'alignment': job.data['alignment'],
        'view_transform': job.data['view_transform'],
    }


@app.post('/api/jobs/{job_id}/render')
def job_render(job_id: str, req: RenderRequest) -> Dict[str, Any]:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Unknown job')
    if job.state != 'done':
        raise HTTPException(status_code=409, detail=f'Job is {job.state}')
    vt = {'gamma': req.gamma, 'brightness': req.brightness, 'contrast': req.contrast}
    job.data['view_transform'] = vt
    rendered = _render_result(job, vt)
    # Return BOTH images under the same view transform so the before/after
    # comparison stays consistent while a slider moves.
    return {
        'fused': rendered['fused'],
        'before': rendered['before'],
        'width': rendered['width'],
        'height': rendered['height'],
    }


@app.post('/api/export')
def export(req: ExportRequest) -> Dict[str, Any]:
    if req.kind not in ('master', 'baked', 'enhanced', 'enhanced_baked'):
        raise HTTPException(status_code=400, detail=f'Unknown export kind: {req.kind}')
    job = manager.get(req.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Unknown job')
    if job.state != 'done':
        raise HTTPException(status_code=409, detail=f'Job is {job.state}')
    from raw_sr.core.export import Exporter
    from raw_sr.core.view_transform import ViewTransform

    # Heavy arrays (hr/coverage/before_rgb) are retained only for the most
    # recent completed process job (JobManager LRU). An evicted job still
    # renders result/render, but exporting it needs the dropped `hr`.
    hr: Optional[np.ndarray] = job.data.get('hr')
    if hr is None:
        raise HTTPException(
            status_code=409,
            detail='The full-resolution result for this job was evicted from '
                   'memory (only the most recent processed result is exportable). '
                   'Re-run processing to export.')
    exporter = Exporter()
    enhance_note: Optional[str] = None
    try:
        if req.kind == 'master':
            exporter.export_linear(hr, req.output_path)
        elif req.kind == 'baked':
            vt = req.view_transform or ViewTransformModel()
            exporter.export_baked(hr, req.output_path,
                                  view_transform=ViewTransform(vt.gamma, vt.brightness, vt.contrast))
        else:
            from raw_sr.core.enhance import effective_stack_noise
            # (was job.result, which Job never had: every enhanced export raised
            # AttributeError -> HTTP 500 since WP2.)
            res_meta = job.data.get('report') or {}
            metrics = res_meta.get('metrics') or {}
            ratio = float(res_meta.get('ratio', 2.0))
            used = res_meta.get('used_frames')
            noise_sigma = effective_stack_noise(metrics.get('frame_noise_sigma'), effective_frames=used)
            vt_obj = None
            if req.kind == 'enhanced_baked' or req.output_path.lower().endswith('.png'):
                vt = req.view_transform or ViewTransformModel()
                vt_obj = ViewTransform(vt.gamma, vt.brightness, vt.contrast)
            # MAP (fusion='map', the 2x default via 'auto' since WP12) already
            # inverts the optics blur in its forward model; sharpening on top
            # double-deconvolves (synthetic 2x GT: map 37.2 dB, +wp2 balanced
            # 30.2 dB, even strength 0.05 32.9 dB). Unless the caller asks for
            # an explicit strength, the enhanced export of a MAP master is the
            # master itself (strength 0 is an exact identity).
            strength, preset = req.enhance_strength, req.enhance_preset
            if metrics.get('fusion') == 'map' and strength is None:
                strength, preset = 0.0, None
                enhance_note = ('map master: already deconvolved, enhancement '
                                'skipped (pass enhance_strength to force)')
            exporter.export_enhanced(
                hr,
                req.output_path,
                ratio=ratio,
                noise_sigma=noise_sigma,
                strength=strength,
                preset=preset,
                view_transform=vt_obj,
                baked=(req.kind == 'enhanced_baked'),
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f'Could not write output: {exc}') from exc
    out: Dict[str, Any] = {'output_path': req.output_path}
    if enhance_note:
        out['enhance_note'] = enhance_note
    return out


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #

def _bandpass_truth(h: int = 192, w: int = 192, lo: float = 0.30, hi: float = 0.50,
                    seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.random((h, w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    F[(r < lo) | (r > hi)] = 0
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    return np.stack([g, g, g], axis=2)


def _sample_lr(truth: np.ndarray, dx: float, dy: float,
               lr_shape: Tuple[int, int]) -> np.ndarray:
    ratio = truth.shape[0] // lr_shape[0]
    xs, ys = np.meshgrid(np.arange(lr_shape[1], dtype=np.float32),
                         np.arange(lr_shape[0], dtype=np.float32))
    return cv2.remap(truth, ratio * (xs + dx), ratio * (ys + dy), cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
