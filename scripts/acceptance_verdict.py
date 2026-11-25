#!/usr/bin/env python3
"""Explicit PASS / FAIL / N-A verdict for the project's acceptance criteria.

``scripts/sr_acceptance.py`` measures the SR core on a real burst and prints
metrics, but until now nothing turned those numbers into a verdict: the reader
had to remember the SPEC thresholds and look the values up by hand.  This
module holds the criteria table (each row cites the SPEC lines it comes from),
the measurement helpers that produce the values, and the formatter that prints
the table and writes it into the acceptance JSON.

Criteria sources (the only authority):
  * ``docs/SPEC.md`` §7 (version 0.2.0 acceptance, lines 219-221) and §11
    (1.0.0 acceptance, lines 307-317);
  * the project's own truth-free proof (held-out PSNR gain vs the denoised LR
    stack of the same frames), recorded in ``state.md`` and
    ``docs/IBP_AND_HELDOUT.md`` -- marked as project, not SPEC.

Honesty rules this module enforces:

  * A criterion whose data does not exist is **N/A**, never PASS.  The
    flat-field 15+ frame case is the important one: the available burst has
    9 frames of real scene, not a flat target, so the SPEC row cannot be
    tested.  What *is* measurable -- the noise drop in the flattest region of
    the 9-frame burst -- is reported as a substitute with its scaling
    assumption written out, and the row stays N/A.
  * Every row carries the measured value, the threshold, the unit and the
    SPEC line, so a FAIL cannot be read as a rounding artefact.
  * Values are reported on the grid/space they were measured in; the JSON key
    each value comes from is listed as ``source``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_NA = "N/A"

# The threshold below which the desktop backend raises its low-coverage
# warning (desktop/backend/server.py: ``low_coverage_warning``).  Kept here as
# a constant so the acceptance row and the UI cannot drift apart silently.
LOW_COVERAGE_FRACTION = 0.2


# --------------------------------------------------------------------------- #
# Criteria table
# --------------------------------------------------------------------------- #

def _criterion(cid, title, requirement, spec, threshold, unit, kind):
    return {
        "id": cid,
        "title": title,
        "requirement": requirement,
        "spec": list(spec),
        "threshold": threshold,
        "unit": unit,
        "kind": kind,          # "data" (needs this run's data) | "selfcheck"
    }


CRITERIA = [
    _criterion(
        "S1_snr_flat_15f",
        "SNR improvement on a flat surface with 15+ RAW frames",
        "measurable SNR improvement of at least 3-5 dB",
        ["docs/SPEC.md §7 L219-L220", "docs/SPEC.md §11 L307-L308"],
        ">= 3.0 dB (SPEC 3-5 dB band)", "dB", "data"),
    _criterion(
        "S2_hf_vs_any_single",
        "Genuine HF content gain vs ANY single input frame",
        "frequency-domain analysis shows non-aliased HF gain in the HR "
        "output vs. any single input frame",
        ["docs/SPEC.md §11 L309-L310"],
        "min over frames > 0 dB", "dB", "data"),
    _criterion(
        "S3_time_roi500",
        "Processing time for a ~500x500 source ROI",
        "seconds to ~1 minute, hardware-dependent",
        ["docs/SPEC.md §11 L311-L313"],
        "<= 60 s", "s", "data"),
    _criterion(
        "S4_parallax_no_crash",
        "Parallax region does not crash; low confidence is indicated/warned",
        "either a low-confidence result with clear indication, or a clear "
        "warning",
        ["docs/SPEC.md §11 L314-L317"],
        "no exception + non-empty robust_reason", "-", "selfcheck"),
    _criterion(
        "S5_alignment_heatmap",
        "Alignment-quality heatmap reflects excluded/low-confidence regions",
        "qualitative: heatmap visibly reflects excluded regions",
        ["docs/SPEC.md §7 L221"],
        "qualitative", "-", "selfcheck"),
    _criterion(
        "S6_low_coverage_warning",
        "Low-coverage warning below ~20% usable coverage; never aborts",
        "proceed anyway, don't abort",
        ["docs/SPEC.md §7 L218"],
        "warning fires when used/total < 0.2; no abort", "-", "selfcheck"),
    _criterion(
        "S7_ratio_reported",
        "Requested vs measurement-supported achievable SR ratio is reported",
        "free-form SR ratio input + requested-vs-achieved feedback",
        ["docs/SPEC.md §7 L213"],
        "requested and achievable both reported", "x", "selfcheck"),
    _criterion(
        "S8_threshold_scaling",
        "Residual threshold scaled by the requested SR ratio (stricter)",
        "higher ratio => smaller threshold: (median + k*MAD) / sqrt(ratio)",
        ["docs/SPEC.md §7 L211-L212"],
        "t(1x) > t(2x) > t(4x) > 0", "px", "selfcheck"),
    _criterion(
        "P1_heldout_gain",
        "Held-out PSNR gain vs the denoised LR stack of the same frames",
        "project's truth-free SR proof: predict a frame that was not fused",
        ["state.md (project criterion, not SPEC)",
         "docs/IBP_AND_HELDOUT.md §5"],
        "min held-out gain > 0 dB", "dB", "data"),
]


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #

def _fmt(value, unit):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if unit == "x":
        return f"{float(value):.2f}x"
    if unit == "s":
        return f"{float(value):.2f} s"
    if unit == "dB":
        return f"{float(value):+.2f} dB"
    if unit == "px":
        return f"{float(value):.3g} px"
    return f"{float(value):.4g}"


def format_table(result, width=None):
    """Render one run's verdict as a plain-text table (no external deps)."""
    rows = [("ID", "STATUS", "MEASURED", "THRESHOLD", "SPEC")]
    for c in result["criteria"]:
        spec = c["spec"][0] if c["spec"] else ""
        rows.append((c["id"], c["status"],
                     _fmt(c.get("value"), c.get("unit")),
                     c["threshold"], spec))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))
             for r in rows]
    counts = result["counts"]
    lines.append("")
    lines.append("verdict: %s  (%d PASS, %d FAIL, %d N/A)"
                 % (result["overall"], counts[STATUS_PASS],
                    counts[STATUS_FAIL], counts[STATUS_NA]))
    return "\n".join(lines)


def _result(criteria, label=""):
    counts = {STATUS_PASS: 0, STATUS_FAIL: 0, STATUS_NA: 0}
    for c in criteria:
        counts[c["status"]] += 1
    if counts[STATUS_FAIL]:
        overall = STATUS_FAIL
    elif counts[STATUS_NA]:
        overall = "INCOMPLETE"
    else:
        overall = STATUS_PASS
    return {"label": label, "criteria": criteria, "counts": counts,
            "overall": overall}


def _row(cid, status, value=None, detail="", source="", substitute=None,
         unit=None):
    base = next(c for c in CRITERIA if c["id"] == cid)
    out = {
        "id": cid,
        "title": base["title"],
        "requirement": base["requirement"],
        "spec": base["spec"],
        "threshold": base["threshold"],
        "unit": base["unit"] if unit is None else unit,
        "status": status,
        "value": value,
        "detail": detail,
        "source": source,
    }
    if substitute is not None:
        out["substitute"] = substitute
    return out


# --------------------------------------------------------------------------- #
# Measurement helpers (pure functions; the acceptance script calls them)
# --------------------------------------------------------------------------- #

def flattest_block(gray, block=64, stride=16, hp_sigma=1.5):
    """Locate the flattest ``block`` x ``block`` region and its noise sigma.

    Used by the flat-surface SNR substitute: the SPEC's flat target is absent,
    so the quietest real region of the burst stands in for it.  "Flat" means
    lowest high-pass energy, measured as the std of ``gray - GaussianBlur``;
    a printed chart/sky/field patch wins over texture.

    Returns ``(y, x, sigma)`` in the input's pixel coordinates.
    """
    import cv2

    g = np.asarray(gray, dtype=np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), float(hp_sigma),
                              borderType=cv2.BORDER_REPLICATE)
    h, w = g.shape
    b = int(block)
    if h < b or w < b:
        raise ValueError("flattest_block: block larger than the image")
    best = (0.0, 0.0, float("inf"))
    for y in range(0, h - b + 1, max(int(stride), 1)):
        for x in range(0, w - b + 1, max(int(stride), 1)):
            s = float(np.std(hp[y:y + b, x:x + b]))
            if s < best[2]:
                best = (float(y), float(x), s)
    return best


def _flat_sigma(img, patch=None):
    """Robust noise sigma (MAD) of a luminance patch, tolerant of a few edges."""
    import cv2

    g = np.asarray(img, dtype=np.float32)
    if g.ndim == 3:
        g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    if patch is not None:
        y0, x0, b = patch
        g = g[y0:y0 + b, x0:x0 + b]
    if g.size < 16:
        raise ValueError("flat_region_snr: patch is empty")
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.5,
                              borderType=cv2.BORDER_REPLICATE)
    mad = float(np.median(np.abs(hp - np.median(hp))))
    return max(1.4826 * mad, 1e-12)


def flat_region_snr(hr, naive, reference_gray, ratio, block_lr=64,
                    pitch="lr"):
    """Noise-only SNR gain in the flattest region, single frame -> SR.

    Both images are on the HR grid: ``naive`` is the reference frame
    bicubically upscaled, ``hr`` the SR output.  Inside the flattest LR block
    the signal is ~constant, so "SNR" reduces to noise power:
    ``20 log10(sigma_naive / sigma_sr)``.

    ``pitch`` selects the scale the two are compared at, and it matters:

    * ``"lr"`` (default, the honest one): both images are box-averaged by the
      ratio first, i.e. read at one sample per *input* pixel -- the pitch the
      sensor measured at, and the pitch at which "averaging N frames" is
      supposed to reduce noise.  This is the physically correct comparison.
    * ``"hr"``: compare per HR pixel.  Reported as a diagnostic only: bicubic
      upsampling smooths a single frame's noise, so this baseline looks
      quieter as the ratio grows (measured: 3.96e-4 / 2.37e-4 / 1.54e-4 at
      2x/3x/4x for the same sensor noise), and the number then falls with the
      ratio for reasons that have nothing to do with the fusion.

    This is a *substitute* for the SPEC's 15-frame flat-field measurement, not
    that measurement: it uses 9 real frames of an ordinary scene, one region,
    and it does not test the flat-target capture chain end to end.
    """
    import cv2

    ratio = float(ratio)
    b = int(block_lr)
    y, x, _ = flattest_block(reference_gray, block=b,
                             stride=max(b // 4, 1))
    r = ratio
    y0, x0 = int(round(y * r)), int(round(x * r))

    def sample(img):
        g = img[..., :3] if np.ndim(img) == 3 else img
        p = g[y0:y0 + int(round(b * r)), x0:x0 + int(round(b * r))]
        if p.size == 0 or p.shape[0] < b or p.shape[1] < b:
            raise ValueError("flat_region_snr: mapped block is empty")
        if pitch == "hr":
            return _flat_sigma(p)
        return _flat_sigma(cv2.resize(p, (b, b),
                                      interpolation=cv2.INTER_AREA))

    out = {
        "snr_gain_db": float(20.0 * np.log10(sample(naive) / sample(hr))),
        "pitch": pitch,
        "flat_block_lr": [int(y), int(x), b],
    }
    sig_n = sample(naive)
    sig_s = sample(hr)
    out["noise_sigma_single"] = float(sig_n)
    out["noise_sigma_sr"] = float(sig_s)
    if pitch == "lr":
        # Same region, same images, per HR pixel: kept beside the headline so
        # the bicubic-smoothing effect on the baseline is visible rather than
        # hidden.
        hr_pitch = flat_region_snr(hr, naive, reference_gray, ratio,
                                   block_lr=b, pitch="hr")
        out["snr_gain_db_hr_pitch"] = hr_pitch["snr_gain_db"]
        out["noise_sigma_single_hr_pitch"] = hr_pitch["noise_sigma_single"]
        out["noise_sigma_sr_hr_pitch"] = hr_pitch["noise_sigma_sr"]
    return out


def single_frame_hf_gains(hr, frames, gain_fn, view_fn, *,
                          max_frames=None):
    """HF-band gain (dB) of the SR output vs EVERY input frame, upscaled.

    The SPEC says "vs. any single input frame", so the verdict uses the
    minimum over all frames -- not the reference frame alone (which is what
    ``hf_gain_db_vs_single`` reports and is the most favourable single frame
    on a burst whose reference is the sharpest one).

    The comparison upscale is bicubic with no extra alignment: the HF band
    energy above the LR Nyquist is translation invariant, so a frame's
    registration offset does not change the band total; only its border does,
    and the reference frame is included in the list, so the strongest case is
    measured too.

    ``gain_fn`` is ``high_frequency_gain_db`` and ``view_fn`` is
    ``to_view_float`` (passed in so this module needs no core import).
    """
    import cv2

    hh, ww = hr.shape[:2]
    sr_view = view_fn(hr)
    out = []
    for i, fr in enumerate(frames):
        if max_frames is not None and i >= int(max_frames):
            break
        naive = cv2.resize(fr.rgb, (ww, hh), interpolation=cv2.INTER_CUBIC)
        out.append(float(gain_fn(sr_view, view_fn(naive))))
        del naive
    del sr_view
    return out


def ratio_feedback(shifts, requested_ratio):
    """Requested vs achievable ratio, via the shipped analyzer."""
    from raw_sr.core.sr_ratio_analyzer import SRRatioAnalyzer
    return SRRatioAnalyzer().analyze(
        [tuple(map(float, s)) for s in shifts], float(requested_ratio))


def threshold_scaling(residuals=None, k=3.0):
    """Residual threshold at 1x/2x/4x for the same residuals (SPEC §7 L211)."""
    from raw_sr.core.sr_ratio_analyzer import SRRatioAnalyzer
    if residuals is None:
        # Deterministic, representative residual set; the point is the scaling
        # law, not the absolute value.
        residuals = [0.02, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2, 0.3, 0.45]
    an = SRRatioAnalyzer()
    arr = np.asarray(residuals, dtype=np.float64)
    return {f"{r:g}x": float(an.compute_residual_threshold(arr, k, r))
            for r in (1.0, 2.0, 4.0)}


# --------------------------------------------------------------------------- #
# Self-checks (cheap synthetic runs; no real data needed)
# --------------------------------------------------------------------------- #

def _as_frame(rgb, name):
    from raw_sr.core.superres import RawFrame
    rgb = np.asarray(rgb, dtype=np.float32)
    return RawFrame(name=name, rgb=rgb, green=rgb[:, :, 1].copy())


def _smooth_scene(h, w, seed=7):
    import cv2
    rng = np.random.default_rng(seed)
    t = rng.random((h, w, 3)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), 2.0)
    lo, hi = float(t.min()), float(t.max())
    return ((t - lo) / max(hi - lo, 1e-9) * 0.8 + 0.1).astype(np.float32)


def _sample_scene(scene, dx, dy, out_shape, ratio=2.0):
    """Sample ``scene`` at LR sites shifted by (dx, dy) LR px."""
    import cv2
    h, w = out_shape
    off = 0.5 * (ratio - 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(scene, ratio * (xs + dx) + off, ratio * (ys + dy) + off,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def selfcheck_parallax(frames=5, roi=160, ratio=2.0):
    """Run a synthetic parallax burst; report crash/indication (SPEC §11)."""
    from raw_sr.core.superres import SuperResolution

    big = _smooth_scene(roi * 3, roi * 3)
    fg_scale = 1.9  # foreground apparent motion differs from the background
    shifts = [(0.0, 0.0), (-1.2, 0.5), (0.9, -1.1), (-0.4, -1.6), (1.5, 1.2)]
    burst = []
    for i, (dx, dy) in enumerate(shifts[:frames]):
        bg = _sample_scene(big, dx, dy, (roi, roi), ratio)
        fg_src = big[roi:2 * roi, roi:2 * roi]
        fg = _sample_scene(fg_src, dx * fg_scale, dy * fg_scale,
                           (roi, roi), ratio)
        m = max(1, roi // 3)
        y0, x0 = roi // 3, roi // 2
        bg[y0:y0 + m, x0:x0 + m] = fg[y0:y0 + m, x0:x0 + m]
        burst.append(_as_frame(bg, f"parallax{i}"))
    try:
        res = SuperResolution(ratio=ratio).run(burst)
    except Exception as exc:  # noqa: BLE001 - the criterion is "does not crash"
        return {"crashed": True, "error": f"{type(exc).__name__}: {exc}",
                "robust_active": None, "robust_reason": "",
                "frames": len(burst)}
    reason = str(res.metrics.get("robust_reason", "") or "")
    return {
        "crashed": False,
        "frames": len(burst),
        "used_frames": int(res.used_frames),
        "robust_active": bool(res.metrics.get("robust_active", False)),
        "robust_reason": reason,
        "ghost_energy_norm": res.metrics.get("ghost_energy_norm"),
        "finite_hr": bool(np.isfinite(res.hr).all()),
    }


def selfcheck_low_coverage(frames=9, roi=128, ratio=2.0):
    """One sharp frame + blurred ones: warning path must fire and not abort."""
    import cv2
    from raw_sr.core.superres import SuperResolution

    scene = _smooth_scene(roi * 2, roi * 2, seed=11)
    sharp = _sample_scene(scene, 0.0, 0.0, (roi, roi), ratio)
    burst = [_as_frame(sharp, "sharp0")]
    for i in range(1, frames):
        blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
        burst.append(_as_frame(blurred, f"blurred{i}"))
    try:
        res = SuperResolution(ratio=ratio).run(burst)
    except Exception as exc:  # noqa: BLE001 - "never abort"
        return {"crashed": True, "error": f"{type(exc).__name__}: {exc}",
                "used": 0, "total": len(burst), "warning": None}
    used, total = int(res.used_frames), int(res.total_frames)
    return {
        "crashed": False,
        "used": used,
        "total": total,
        "fraction": used / max(total, 1),
        "warning": bool(used < LOW_COVERAGE_FRACTION * max(total, 1)),
    }


# The corrupted rectangle of the heatmap fixture, in fractions of the frame
# (y0, x0, h, w). One frame disagrees there and nowhere else, so a working
# alignment-quality map must read low inside it and high outside.
_HEATMAP_REGION = (0.30, 0.35, 0.40, 0.40)


def _decode_heatmap(data_url):
    """Decode the shipped heatmap PNG data URL to an HxWx3 uint8 array."""
    import base64
    import cv2

    payload = data_url.split(",", 1)[-1]
    raw = base64.b64decode(payload)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("heatmap data URL did not decode as an image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def selfcheck_heatmap(frames=9, roi=192, ratio=2.0):
    """Does the SHIPPED alignment-quality map mark the disagreeing region?

    SPEC §7 L221 ("the alignment-quality heatmap visibly reflects
    excluded/low-confidence regions") is qualitative for a human looking at the
    app, but the map itself is a deterministic function
    (``desktop.backend.server._alignment_quality_map``).  This check builds a
    burst in which exactly one frame disagrees inside one rectangle, asks the
    shipped function for the map, and compares the TURBO-red level inside that
    rectangle against the rest of the covered area.

    The disagreement is a +0.25 linear level step confined to the rectangle
    (an "object that moved in one frame"), not noise: a noise patch fails
    registration and the gate drops that frame, which would leave a
    one-frame stack whose std is identically zero (measured: ratio 1.00,
    i.e. an uninformative map).

    Returns a dict with the measurement, ``available: False`` when the desktop
    backend cannot be imported (the check then degrades to N/A, never PASS).
    """
    from raw_sr.core.superres import SuperResolution

    try:
        from desktop.backend.server import _alignment_quality_map
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"available": False,
                "error": f"desktop backend unavailable: {type(exc).__name__}"}

    scene = _smooth_scene(roi * 2, roi * 2, seed=17)
    shifts = [(0.0, 0.0), (-1.1, 0.6), (0.8, -1.0), (-0.5, -1.4), (1.3, 1.1),
              (0.2, -0.7), (-0.9, 1.3), (1.7, 0.3), (-1.5, -0.4)]
    burst = []
    for i, (dx, dy) in enumerate(shifts[:frames]):
        rgb = _sample_scene(scene, dx, dy, (roi, roi), ratio)
        if i == frames // 2:
            fy, fx, fh, fw = _HEATMAP_REGION
            y0, x0 = int(fy * roi), int(fx * roi)
            y1 = min(y0 + int(fh * roi), roi)
            x1 = min(x0 + int(fw * roi), roi)
            rgb = rgb.copy()
            rgb[y0:y1, x0:x1] = np.clip(rgb[y0:y1, x0:x1] + 0.25, 0.0, 1.0)
        burst.append(_as_frame(rgb, f"heat{i}"))

    res = SuperResolution(ratio=ratio).run(burst)
    kept = [int(i) for i in res.kept_indices]
    greens = [burst[i].green for i in kept]
    url = _alignment_quality_map(greens, list(res.transforms), float(ratio))
    img = _decode_heatmap(url)
    # TURBO: quality rises from dark blue to red, so the red channel tracks it.
    red = img[..., 0].astype(np.float32)
    h, w = red.shape
    fy, fx, fh, fw = _HEATMAP_REGION
    y0, x0 = int(fy * h), int(fx * w)
    y1, x1 = min(y0 + int(fh * h), h), min(x0 + int(fw * w), w)
    inside = red[y0:y1, x0:x1]
    mask = np.ones_like(red, dtype=bool)
    mask[y0:y1, x0:x1] = False
    outside = red[mask]
    inside_mean = float(inside.mean())
    outside_mean = float(outside.mean())
    return {
        "available": True,
        "shape": [int(h), int(w)],
        "region": [int(y0), int(x0), int(y1 - y0), int(x1 - x0)],
        "inside_red_mean": inside_mean,
        "outside_red_mean": outside_mean,
        "ratio": inside_mean / max(outside_mean, 1e-9),
        # The disagreeing rectangle must read clearly lower; 0.8 is a loose
        # bar (the measured margin is far larger) so the check does not pin the
        # colormap's exact scaling.
        "low_inside": bool(inside_mean < 0.8 * outside_mean),
        "frames_used": int(res.used_frames),
    }


def run_self_checks(ratio=2.0):
    """All synthetic checks, each isolated so one failure cannot hide others."""
    out = {"ratio": float(ratio)}
    for key, fn in (("parallax", selfcheck_parallax),
                    ("low_coverage", selfcheck_low_coverage),
                    ("heatmap", selfcheck_heatmap)):
        try:
            out[key] = fn(ratio=float(ratio))
        except Exception as exc:  # noqa: BLE001
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# Criterion evaluation
# --------------------------------------------------------------------------- #

def _c1_snr(metrics, ctx):
    sub = (ctx or {}).get("flat_snr") or metrics.get("flat_snr")
    if not sub:
        return _row("S1_snr_flat_15f", STATUS_NA,
                    detail="flat-field 15+ frame measurement not run",
                    source="flat_snr")
    frames = int(sub.get("frames_used") or metrics.get("frames_used") or 0)
    gain = float(sub["snr_gain_db"])
    if frames >= 15:
        return _row("S1_snr_flat_15f",
                    STATUS_PASS if gain >= 3.0 else STATUS_FAIL,
                    value=gain, source="flat_snr.snr_gain_db",
                    detail="%d frames fused" % frames)
    # Honest substitute: the SPEC burst does not exist here (9 real frames, no
    # flat target). Report the measured noise-only gain and the sqrt(N) scaling
    # expectation, and keep the row N/A.
    scaled = gain + 20.0 * np.log10(np.sqrt(15.0 / max(frames, 1)))
    block = sub.get("flat_block_lr", [0, 0, 64])[2]
    hr_pitch = sub.get("snr_gain_db_hr_pitch")
    detail = ("SPEC burst absent (9 real frames of ordinary scene, no flat "
              "target). Substitute, measured at LR (input-pixel) pitch in the "
              "flattest %dx%d LR block: noise-only SNR gain %+.2f dB "
              "(sigma %.2e -> %.2e); sqrt(N) scaling to 15 frames would be "
              "%+.2f dB. NOT the SPEC measurement."
              % (block, block, gain,
                 float(sub.get("noise_sigma_single") or 0.0),
                 float(sub.get("noise_sigma_sr") or 0.0), scaled))
    if hr_pitch is not None:
        detail += (" Per-HR-pixel value %+.2f dB (bicubic smooths the "
                   "single-frame baseline, so it falls with ratio; diagnostic "
                   "only)." % float(hr_pitch))
    return _row(
        "S1_snr_flat_15f", STATUS_NA, value=gain,
        source="flat_snr.snr_gain_db",
        detail=detail,
        substitute={"snr_gain_db": gain, "frames": frames,
                    "scaled_15f_db": float(scaled),
                    "snr_gain_db_hr_pitch": hr_pitch,
                    "method": "20*log10(sigma_single/sigma_sr) on the "
                              "flattest LR block, both read at one sample per "
                              "input pixel (box-averaged by the ratio), "
                              "sensor-noise only",
                    "passes_scaled_expectation": bool(scaled >= 3.0)})


def _c2_hf_single(metrics, ctx):
    gains = metrics.get("hf_gain_db_vs_single_frames")
    if gains:
        vals = [float(g) for g in gains]
        lo, hi = min(vals), max(vals)
        return _row("S2_hf_vs_any_single",
                    STATUS_PASS if lo > 0.0 else STATUS_FAIL, value=lo,
                    source="hf_gain_db_vs_single_frames (min over %d frames)"
                           % len(vals),
                    detail="range %+.2f .. %+.2f dB over every input frame"
                           % (lo, hi))
    if "hf_gain_db_vs_single" in metrics:
        v = float(metrics["hf_gain_db_vs_single"])
        return _row("S2_hf_vs_any_single",
                    STATUS_PASS if v > 0.0 else STATUS_FAIL, value=v,
                    source="hf_gain_db_vs_single (reference frame only)",
                    detail="per-frame list missing; reference frame is the "
                           "sharpest, so this is the most favourable frame")
    return _row("S2_hf_vs_any_single", STATUS_NA,
                detail="no HF-vs-single measurement in this JSON",
                source="hf_gain_db_vs_single_frames")


def _c3_time(metrics, ctx):
    roi = metrics.get("roi")
    if roi is None and ctx:
        roi = ctx.get("roi")
    seconds = metrics.get("seconds")
    if seconds is None:
        return _row("S3_time_roi500", STATUS_NA, detail="no timing recorded",
                    source="seconds")
    size = None
    if roi:
        try:
            size = min(float(roi[2]), float(roi[3]))
        except (TypeError, IndexError):
            size = None
    if size is not None and size <= 560:
        return _row("S3_time_roi500",
                    STATUS_PASS if float(seconds) <= 60.0 else STATUS_FAIL,
                    value=float(seconds), source="seconds",
                    detail="ROI %gx%g source px (SR call wall time)"
                           % (float(roi[2]), float(roi[3])))
    return _row("S3_time_roi500", STATUS_NA, value=float(seconds),
                source="seconds",
                detail="this run's ROI is %s px, not the SPEC's ~500x500 case; "
                       "see the ROI-512 timing run"
                       % ("%gx%g" % (float(roi[2]), float(roi[3]))
                          if roi else "unknown"))


def _c4_parallax(metrics, ctx):
    chk = (ctx or {}).get("parallax")
    if not chk:
        return _row("S4_parallax_no_crash", STATUS_NA,
                    detail="parallax self-check did not run",
                    source="self_checks.parallax")
    if chk.get("crashed"):
        return _row("S4_parallax_no_crash", STATUS_FAIL, value="crash",
                    source="self_checks.parallax",
                    detail=chk.get("error", ""))
    if "error" in chk:
        return _row("S4_parallax_no_crash", STATUS_NA,
                    detail="parallax self-check could not run: %s"
                           % chk["error"], source="self_checks.parallax")
    reason = chk.get("robust_reason") or ""
    ok = bool(reason) and bool(chk.get("finite_hr", True))
    return _row("S4_parallax_no_crash", STATUS_PASS if ok else STATUS_FAIL,
                value="no crash", source="self_checks.parallax",
                detail="%d-frame synthetic parallax layer: robust_active=%s, "
                       "reason=%r" % (chk.get("frames", 0),
                                      chk.get("robust_active"), reason))


def _c5_heatmap(metrics, ctx):
    chk = (ctx or {}).get("heatmap")
    scores = ("%d-frame run: mean ZNCC %.4f, mean residual %.3f px"
              % (int(metrics.get("frames_used") or 0),
                 float(metrics.get("mean_align_score") or 0.0),
                 float(metrics.get("align_residual_mean_px") or 0.0)))
    if not chk:
        return _row("S5_alignment_heatmap", STATUS_NA,
                    value=metrics.get("mean_align_score"),
                    source="self_checks.heatmap",
                    detail="heatmap self-check did not run; " + scores)
    if not chk.get("available", True):
        return _row("S5_alignment_heatmap", STATUS_NA,
                    value=metrics.get("mean_align_score"),
                    source="self_checks.heatmap",
                    detail=chk.get("error", "heatmap check unavailable"))
    if "error" in chk:
        return _row("S5_alignment_heatmap", STATUS_NA,
                    source="self_checks.heatmap",
                    detail="heatmap self-check could not run: %s"
                           % chk["error"])
    ok = bool(chk.get("low_inside"))
    return _row("S5_alignment_heatmap", STATUS_PASS if ok else STATUS_FAIL,
                value=float(chk.get("ratio", 0.0)), unit="-",
                source="self_checks.heatmap "
                       "(desktop.backend.server._alignment_quality_map)",
                detail="one frame disagrees inside a %dx%d px rectangle: map "
                       "red level inside/outside = %.2f (%.0f vs %.0f); a "
                       "working map marks the region low. %s"
                       % (chk["region"][2], chk["region"][3],
                          float(chk.get("ratio", 0.0)),
                          float(chk.get("inside_red_mean", 0.0)),
                          float(chk.get("outside_red_mean", 0.0)), scores))


def _c6_low_coverage(metrics, ctx):
    chk = (ctx or {}).get("low_coverage")
    if not chk:
        return _row("S6_low_coverage_warning", STATUS_NA,
                    detail="low-coverage self-check did not run",
                    source="self_checks.low_coverage")
    if chk.get("crashed"):
        return _row("S6_low_coverage_warning", STATUS_FAIL, value="abort",
                    source="self_checks.low_coverage",
                    detail=chk.get("error", ""))
    if "error" in chk:
        return _row("S6_low_coverage_warning", STATUS_NA,
                    detail="low-coverage self-check could not run: %s"
                           % chk["error"], source="self_checks.low_coverage")
    frac = float(chk.get("fraction", 1.0))
    warning = chk.get("warning")
    if frac >= LOW_COVERAGE_FRACTION:
        return _row("S6_low_coverage_warning", STATUS_NA,
                    value=frac, unit="-", source="self_checks.low_coverage",
                    detail="self-check kept %.0f%% of frames, so the <20%% "
                           "warning path was not exercised" % (100 * frac))
    ok = bool(warning)
    return _row("S6_low_coverage_warning", STATUS_PASS if ok else STATUS_FAIL,
                value=frac, unit="-", source="self_checks.low_coverage",
                detail="%d/%d frames usable (%.0f%% < %.0f%%) -> warning=%s, "
                       "run completed (no abort)"
                       % (chk.get("used", 0), chk.get("total", 0),
                          100 * frac, 100 * LOW_COVERAGE_FRACTION, warning))


def _c7_ratio(metrics, ctx):
    info = (ctx or {}).get("ratio_feedback")
    if not info:
        return _row("S7_ratio_reported", STATUS_NA,
                    detail="ratio feedback not computed (run's shifts absent?)",
                    source="sr_ratio_analyzer")
    ok = all(k in info for k in ("requested", "achievable", "message"))
    return _row("S7_ratio_reported", STATUS_PASS if ok else STATUS_FAIL,
                value=float(info["achievable"]),
                source="sr_ratio_analyzer.analyze(shifts, requested)",
                detail="requested %.2fx -> achievable ~%.2fx (%s)"
                       % (float(info["requested"]), float(info["achievable"]),
                          "message includes both"
                          if ok else "message missing"))


def _c8_threshold(metrics, ctx):
    info = (ctx or {}).get("threshold_scaling")
    if not info:
        return _row("S8_threshold_scaling", STATUS_NA,
                    detail="threshold scaling not computed",
                    source="SRRatioAnalyzer.compute_residual_threshold")
    t1, t2, t4 = (info.get("1x"), info.get("2x"), info.get("4x"))
    ok = (t1 is not None and t2 is not None and t4 is not None
          and t1 > t2 > t4 > 0.0)
    return _row("S8_threshold_scaling", STATUS_PASS if ok else STATUS_FAIL,
                value=t4, source="SRRatioAnalyzer.compute_residual_threshold",
                detail="t(1x)=%.4g px, t(2x)=%.4g px, t(4x)=%.4g px (stricter "
                       "with ratio)" % (t1 or 0.0, t2 or 0.0, t4 or 0.0))


def _p1_heldout(metrics, ctx):
    if "heldout_gain_mean_db" in metrics:
        mean = float(metrics["heldout_gain_mean_db"])
        lo = float(metrics.get("heldout_gain_min_db", mean))
        return _row("P1_heldout_gain", STATUS_PASS if lo > 0.0 else STATUS_FAIL,
                    value=lo, source="heldout_gain_min_db / _mean_db",
                    detail="mean %+.3f dB, min %+.3f dB over %s held-out "
                           "frames" % (mean, lo,
                                       len(metrics.get("heldout_sweep", []))
                                       or "?"))
    if "heldout_gain_db" in metrics:
        v = float(metrics["heldout_gain_db"])
        return _row("P1_heldout_gain", STATUS_PASS if v > 0.0 else STATUS_FAIL,
                    value=v, source="heldout_gain_db (single held-out frame)",
                    detail="single held-out frame only")
    return _row("P1_heldout_gain", STATUS_NA,
                detail="no held-out measurement (run with --heldout/--heldout-sweep)",
                source="heldout_gain_db")


_EVALUATORS = {
    "S1_snr_flat_15f": _c1_snr,
    "S2_hf_vs_any_single": _c2_hf_single,
    "S3_time_roi500": _c3_time,
    "S4_parallax_no_crash": _c4_parallax,
    "S5_alignment_heatmap": _c5_heatmap,
    "S6_low_coverage_warning": _c6_low_coverage,
    "S7_ratio_reported": _c7_ratio,
    "S8_threshold_scaling": _c8_threshold,
    "P1_heldout_gain": _p1_heldout,
}


def evaluate(metrics, ctx=None, label=""):
    """Evaluate every criterion for one run; returns the JSON-ready verdict."""
    ctx = dict(ctx or {})
    # Self-checks are stored in the same JSON so a re-evaluation (or the CLI on
    # an archived file) reaches the same verdict without re-running them.
    stored_checks = metrics.get("self_checks") or {}
    for key in ("parallax", "low_coverage", "heatmap"):
        if ctx.get(key) is None:
            ctx[key] = stored_checks.get(key)
    if "ratio_feedback" not in ctx and metrics.get("shifts"):
        try:
            ctx["ratio_feedback"] = ratio_feedback(metrics["shifts"],
                                                   metrics.get("ratio", 2.0))
        except Exception:  # noqa: BLE001 - a missing row is honest, a crash is not
            pass
    if "threshold_scaling" not in ctx:
        ctx["threshold_scaling"] = threshold_scaling()
    rows = [_EVALUATORS[c["id"]](metrics, ctx) for c in CRITERIA]
    return _result(rows, label=label)


# --------------------------------------------------------------------------- #
# Multi-run aggregation (the "verdict table")
# --------------------------------------------------------------------------- #

def evaluate_set(runs, self_checks=None, flat_snr=None, timing=None):
    """Verdict for a set of runs: per-ratio rows + one GLOBAL evidence row.

    ``runs`` is a list of ``(label, metrics, ctx)``.  Criteria that are
    properties of the code (parallax, coverage, ratio, scaling) are evaluated
    once from ``self_checks``; the per-run rows keep their own status for the
    data criteria.
    """
    base_ctx = {
        "parallax": (self_checks or {}).get("parallax"),
        "low_coverage": (self_checks or {}).get("low_coverage"),
        "flat_snr": flat_snr,
        "threshold_scaling": threshold_scaling(),
    }
    out = []
    for label, metrics, ctx in runs:
        merged = dict(base_ctx)
        merged.update(ctx or {})
        out.append(evaluate(metrics, merged, label=label))
    if timing is not None:
        # Timing has its own run (ROI 512); attach it to the set-level table.
        trow = _c3_time(timing, None)
        out.append(_result([trow], label="GLOBAL_ROI512"))
    return out


def format_set(tables):
    """Render every run's table plus a one-line summary per run."""
    blocks = []
    for t in tables:
        head = t.get("label") or "run"
        blocks.append("%s\n%s\n%s" % (head, "-" * len(head),
                                      format_table(t)))
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    ap = argparse.ArgumentParser(
        description="Print the PASS/FAIL/N-A verdict stored in acceptance "
                    "JSONs (or recompute what the JSON allows).")
    ap.add_argument("json", nargs="+", help="sr_acceptance JSON file(s)")
    ap.add_argument("--json-out", default=None,
                    help="write the combined verdict tables here")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when any criterion FAILs")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    tables = []
    for path in args.json:
        with open(path) as fh:
            data = json.load(fh)
        label = os.path.basename(path)
        stored = data.get("acceptance")
        if stored and stored.get("criteria"):
            # File name first (two runs can share a ratio, e.g. ROI 1024 and
            # ROI 512), the stored run label after it.
            tag = stored.get("label")
            tables.append(dict(stored,
                               label=f"{label} ({tag})" if tag else label))
        else:
            tables.append(evaluate(data, label=label))
    text = format_set(tables)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"generated_from": args.json, "tables": tables}, fh,
                      indent=2)
        print("\nwrote:", args.json_out)
    if args.strict and any(t["counts"][STATUS_FAIL] for t in tables):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
