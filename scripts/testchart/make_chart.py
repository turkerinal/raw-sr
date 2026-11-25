#!/usr/bin/env python3
"""Generate printable A4 and A3 resolution test charts (vector PDF + 600 dpi PNG).

Chart elements (all coordinates from ``chart_spec.py``):
  - 4 corner ArUco fiducials (DICT_4X4_50, IDs 0..3) for automatic registration
  - Two 144-spoke Siemens stars: sinusoidal and binary
  - Two ISO 12233-style 5-degree slanted edges: high-contrast B/W and low-contrast grey
  - Hyperbolic (horizontal) and linear (vertical) line-pair wedges with frequency marks
  - Deterministic fine random-noise texture patch (for texture / aliasing evaluation)
  - 8-step linear grey wedge (photometric matching + flat-patch noise)
  - Print-size scale bar (50 mm on A4, 100 mm on A3) to verify 100% print scaling
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import zlib

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.testchart import chart_spec as spec
else:
    from . import chart_spec as spec


def _aruco_marker_img(marker_id: int, side_px: int) -> np.ndarray:
    """Render a single ArUco marker as float32 in [0, 1]."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.ARUCO_DICT_NAME))
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, max(16, side_px))
    return (marker.astype(np.float32) / 255.0)


def _render_siemens_star(
    h: int,
    w: int,
    cx: float,
    cy: float,
    r_outer: float,
    r_inner: float,
    n_spokes: int,
    sinusoidal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a Siemens star patch and its circular mask in float32 [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    s = np.sin(n_spokes * ang)
    if sinusoidal:
        # 0.05..0.95 sinusoidal modulation
        val = 0.50 + 0.45 * s
    else:
        # Crisp binary spokes (anti-aliased only at 1-px sub-spoke boundary)
        val = np.where(s >= 0.0, 0.95, 0.05)
    mask = (r <= r_outer) & (r >= r_inner)
    core = r < r_inner
    out = np.full((h, w), 0.96, dtype=np.float32)
    out[mask] = val[mask].astype(np.float32)
    out[core] = 0.50
    return out, (r <= r_outer)


def _render_slanted_edge(
    h: int, w: int, low: float, high: float, tilt_deg: float = spec.EDGE_TILT_DEG
) -> np.ndarray:
    """Render an ISO 12233-style slanted edge patch in float32 [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = math.radians(tilt_deg)
    cx, cy = 0.5 * w, 0.5 * h
    dist = (xx - cx) * math.cos(theta) + (yy - cy) * math.sin(theta)
    # 1-px analytical anti-aliasing
    t = np.clip(0.5 + dist, 0.0, 1.0)
    return (low + (high - low) * t).astype(np.float32)


def _render_wedge_hyperbolic(
    h: int, w: int, chart_h: float, f_min: float, f_max: float
) -> np.ndarray:
    """Horizontal chirp with hyperbolic spatial frequency sweep (vertical bars).

    Local spatial frequency along x varies linearly in period (1/f) from f_min
    at x=0 to f_max at x=w, so frequency f(u) = 1 / ((1/f_min)*(1-u) + (1/f_max)*u)
    in cycles per chart height.
    """
    xs = np.linspace(0.0, 1.0, w, dtype=np.float64)
    # Physical x span in units of chart_h
    span_h = w / float(chart_h)
    # Period p(u) = p0 + (p1 - p0)*u  (in chart_h / cycle)
    p0 = 1.0 / f_min
    p1 = 1.0 / f_max
    # Integral of (1 / p(u)) * span_h du = span_h / (p1 - p0) * ln(p(u) / p0)
    p_u = p0 + (p1 - p0) * xs
    phase_cyc = (span_h / (p1 - p0)) * np.log(p_u / p0)
    row = (0.50 + 0.45 * np.cos(2.0 * np.pi * phase_cyc)).astype(np.float32)
    return np.tile(row[None, :], (h, 1))


def _render_wedge_linear(
    h: int, w: int, chart_h: float, f_min: float, f_max: float
) -> np.ndarray:
    """Vertical chirp with linear spatial frequency sweep (horizontal bars).

    Local spatial frequency along y increases linearly from f_min at y=0 to
    f_max at y=h in cycles per chart height.
    """
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    span_h = h / float(chart_h)
    # Integral of (f_min + (f_max - f_min)*v) * span_h dv
    phase_cyc = span_h * (f_min * ys + 0.5 * (f_max - f_min) * ys * ys)
    col = (0.50 + 0.45 * np.cos(2.0 * np.pi * phase_cyc)).astype(np.float32)
    return np.tile(col[:, None], (1, w))


def _render_noise_patch(h: int, w: int, seed: int = spec.NOISE_PATCH_SEED) -> np.ndarray:
    """Deterministic multi-scale fine random-noise texture patch in [0.10, 0.90]."""
    rng = np.random.default_rng(seed)
    # Generate at a fixed base grid so the physical texture spectrum is identical across DPIs
    base_h, base_w = 256, 256
    white = rng.standard_normal((base_h, base_w)).astype(np.float32)
    # Mildly band-limited pink-ish / white mix for rich texture + aliasing test
    smooth = cv2.GaussianBlur(white, (0, 0), 1.0)
    mix = 0.65 * white + 0.35 * smooth
    mix = mix / (3.0 * float(np.std(mix)) + 1e-8)
    patch = np.clip(0.50 + 0.38 * mix, 0.08, 0.92).astype(np.float32)
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_CUBIC)


def render_chart_raster(paper: str = "A4", dpi: int = 600) -> np.ndarray:
    """Render the complete test chart as a uint8 greyscale image (H, W)."""
    w_mm, h_mm = spec.PAPER_MM[paper]
    w_px = int(round(w_mm * dpi / 25.4))
    h_px = int(round(h_mm * dpi / 25.4))
    aspect = w_mm / h_mm

    # White background (0.96 to avoid hard printer clipping)
    canvas = np.full((h_px, w_px), 0.96, dtype=np.float32)

    # 1. Four corner ArUco fiducials
    for fid, rect in spec.fiducial_rects_norm(aspect).items():
        x0, y0, x1, y1 = rect.to_px(w_px, h_px)
        side = min(x1 - x0, y1 - y0)
        marker = _aruco_marker_img(fid, side)
        canvas[y0 : y0 + side, x0 : x0 + side] = marker

    # 2. Two 144-spoke Siemens stars (sinusoidal & binary)
    r_outer = spec.STAR_RADIUS_H * h_px
    r_inner = spec.STAR_INNER_RADIUS_H * h_px
    for key, center, is_sin in (
        ("star_sin", spec.STAR_SIN_CENTER, True),
        ("star_bin", spec.STAR_BIN_CENTER, False),
    ):
        x0, y0, x1, y1 = spec.ROIS_NORM[key].to_px(w_px, h_px)
        cx_local = center[0] * w_px - x0
        cy_local = center[1] * h_px - y0
        star, mask = _render_siemens_star(
            y1 - y0, x1 - x0, cx_local, cy_local, r_outer, r_inner, spec.STAR_N_SPOKES, is_sin
        )
        sub = canvas[y0:y1, x0:x1]
        sub[mask] = star[mask]

    # 3. High-contrast and low-contrast 5-deg slanted edges
    for key, (lo, hi) in (
        ("edge_high", spec.EDGE_HIGH_CONTRAST),
        ("edge_low", spec.EDGE_LOW_CONTRAST),
    ):
        x0, y0, x1, y1 = spec.ROIS_NORM[key].to_px(w_px, h_px)
        canvas[y0:y1, x0:x1] = _render_slanted_edge(y1 - y0, x1 - x0, lo, hi, spec.EDGE_TILT_DEG)

    # 4. Hyperbolic & linear line-pair wedges
    x0, y0, x1, y1 = spec.ROIS_NORM["wedge_hyp"].to_px(w_px, h_px)
    canvas[y0:y1, x0:x1] = _render_wedge_hyperbolic(
        y1 - y0, x1 - x0, h_px, spec.WEDGE_FREQ_MIN_CYC_H, spec.WEDGE_FREQ_MAX_CYC_H
    )
    x0, y0, x1, y1 = spec.ROIS_NORM["wedge_lin"].to_px(w_px, h_px)
    canvas[y0:y1, x0:x1] = _render_wedge_linear(
        y1 - y0, x1 - x0, h_px, spec.WEDGE_FREQ_MIN_CYC_H, spec.WEDGE_FREQ_MAX_CYC_H
    )

    # 5. Fine random-noise texture patch
    x0, y0, x1, y1 = spec.ROIS_NORM["noise_patch"].to_px(w_px, h_px)
    canvas[y0:y1, x0:x1] = _render_noise_patch(y1 - y0, x1 - x0)

    # 6. 8-step linear grey wedge
    gx0, gy0, gx1, gy1 = spec.ROIS_NORM["grey_wedge"].to_px(w_px, h_px)
    n_steps = len(spec.GREY_STEPS)
    step_w = (gx1 - gx0) / n_steps
    for idx, val in enumerate(spec.GREY_STEPS):
        sx0 = int(round(gx0 + idx * step_w))
        sx1 = int(round(gx0 + (idx + 1) * step_w))
        canvas[gy0:gy1, sx0:sx1] = val

    # Convert to uint8 for crisp borders, scale bar, and text labels
    img = np.clip(np.round(canvas * 255.0), 0, 255).astype(np.uint8)

    # Draw thin frames around rectangular ROIs
    thick = max(1, int(round(h_px / 900.0)))
    for key in ("edge_high", "edge_low", "wedge_hyp", "wedge_lin", "noise_patch", "grey_wedge"):
        x0, y0, x1, y1 = spec.ROIS_NORM[key].to_px(w_px, h_px)
        cv2.rectangle(img, (x0, y0), (x1, y1), 30, thick)

    # 7. Print-size scale bar (50 mm on A4, 100 mm on A3)
    bar_mm = 50.0 if paper == "A4" else 100.0
    bar_px = int(round(bar_mm * dpi / 25.4))
    sb_x0, sb_y0, sb_x1, sb_y1 = spec.ROIS_NORM["scale_bar"].to_px(w_px, h_px)
    bar_x0 = (sb_x0 + sb_x1 - bar_px) // 2
    bar_x1 = bar_x0 + bar_px
    bar_y = int(round(0.55 * sb_y0 + 0.45 * sb_y1))
    bar_h = max(3, int(round(0.008 * h_px)))
    # Alternating 10 mm black/white segments
    n_seg = int(round(bar_mm / 10.0))
    for s_i in range(n_seg):
        sx0 = bar_x0 + int(round(s_i * bar_px / n_seg))
        sx1 = bar_x0 + int(round((s_i + 1) * bar_px / n_seg))
        col = 15 if (s_i % 2 == 0) else 245
        cv2.rectangle(img, (sx0, bar_y), (sx1, bar_y + bar_h), col, -1)
    cv2.rectangle(img, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), 15, thick)

    # Text annotations (OpenCV Hershey font, scaled by chart height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = h_px / 1400.0
    f_thick = max(1, int(round(f_scale * 2.0)))
    f_small = f_scale * 0.55
    f_small_thick = max(1, int(round(f_small * 2.0)))

    title = f"raw_sr WP10 Resolution Chart ({paper} {int(w_mm)}x{int(h_mm)} mm, 144-spoke)"
    cv2.putText(img, title, (sb_x0, sb_y0 + int(0.012 * h_px)), font, f_small * 1.15, 20, f_small_thick, cv2.LINE_AA)
    scale_lbl = f"Scale bar: {int(bar_mm)} mm (verify 100% print scale)"
    cv2.putText(img, scale_lbl, (bar_x0, bar_y + bar_h + int(0.022 * h_px)), font, f_small, 20, f_small_thick, cv2.LINE_AA)

    # Labels under stars & edges
    labels = [
        ("Sinusoidal 144-spoke", spec.ROIS_NORM["star_sin"]),
        ("Binary 144-spoke", spec.ROIS_NORM["star_bin"]),
        ("Edge 5 deg (HC)", spec.ROIS_NORM["edge_high"]),
        ("Edge 5 deg (LC)", spec.ROIS_NORM["edge_low"]),
        (f"Hyperbolic wedge ({int(spec.WEDGE_FREQ_MIN_CYC_H)}-{int(spec.WEDGE_FREQ_MAX_CYC_H)} cyc/H)", spec.ROIS_NORM["wedge_hyp"]),
        (f"Linear wedge ({int(spec.WEDGE_FREQ_MIN_CYC_H)}-{int(spec.WEDGE_FREQ_MAX_CYC_H)} cyc/H)", spec.ROIS_NORM["wedge_lin"]),
        ("Fine noise texture", spec.ROIS_NORM["noise_patch"]),
    ]
    for text, rect in labels:
        rx0, ry0, rx1, ry1 = rect.to_px(w_px, h_px)
        cv2.putText(img, text, (rx0, ry1 + int(0.020 * h_px)), font, f_small, 25, f_small_thick, cv2.LINE_AA)

    # Frequency marks along hyperbolic wedge
    wx0, wy0, wx1, wy1 = spec.ROIS_NORM["wedge_hyp"].to_px(w_px, h_px)
    p0 = 1.0 / spec.WEDGE_FREQ_MIN_CYC_H
    p1 = 1.0 / spec.WEDGE_FREQ_MAX_CYC_H
    for freq in (60, 100, 150, 200, 300, 450, 600):
        u = ((1.0 / freq) - p0) / (p1 - p0)
        if 0.0 <= u <= 1.0:
            tx = int(round(wx0 + u * (wx1 - wx0)))
            cv2.line(img, (tx, wy0 - int(0.008 * h_px)), (tx, wy0), 20, thick)
            cv2.putText(img, str(freq), (tx - int(0.010 * h_px), wy0 - int(0.012 * h_px)), font, f_small * 0.75, 25, f_small_thick, cv2.LINE_AA)

    return img


def write_vector_pdf(paper: str, out_path: str, raster_dpi: int = 300) -> None:
    """Write a self-contained PDF 1.4 with vector geometry + FlateDecode raster patches.

    Vector elements: ArUco fiducials, binary 144-spoke Siemens star, slanted edges,
    8-step grey wedge, scale bar, borders, and Helvetica text labels.
    Embedded lossless greyscale raster patches: sinusoidal Siemens star, hyperbolic
    and linear wedges, and fine random-noise texture patch.
    """
    w_mm, h_mm = spec.PAPER_MM[paper]
    pt_per_mm = 72.0 / 25.4
    w_pt = w_mm * pt_per_mm
    h_pt = h_mm * pt_per_mm
    aspect = w_mm / h_mm

    def pt_x(nx: float) -> float:
        return nx * w_pt

    def pt_y(ny: float) -> float:
        # PDF origin is bottom-left; chart_spec origin is top-left
        return (1.0 - ny) * h_pt

    # Helper to prepare FlateDecode greyscale XObject images
    xobjects: list[tuple[str, int, int, bytes]] = []

    def add_gray_xobj(name: str, arr_f32: np.ndarray) -> str:
        u8 = np.clip(np.round(arr_f32 * 255.0), 0, 255).astype(np.uint8)
        ih, iw = u8.shape
        comp = zlib.compress(u8.tobytes(), level=6)
        xobjects.append((name, iw, ih, comp))
        return name

    # Build embedded raster patches at raster_dpi
    px_per_mm = raster_dpi / 25.4
    w_px = int(round(w_mm * px_per_mm))
    h_px = int(round(h_mm * px_per_mm))

    # Sinusoidal star raster
    sx0, sy0, sx1, sy1 = spec.ROIS_NORM["star_sin"].to_px(w_px, h_px)
    star_sin, _ = _render_siemens_star(
        sy1 - sy0,
        sx1 - sx0,
        spec.STAR_SIN_CENTER[0] * w_px - sx0,
        spec.STAR_SIN_CENTER[1] * h_px - sy0,
        spec.STAR_RADIUS_H * h_px,
        spec.STAR_INNER_RADIUS_H * h_px,
        spec.STAR_N_SPOKES,
        sinusoidal=True,
    )
    add_gray_xobj("ImStarSin", star_sin)

    # Wedges + noise patch
    wx0, wy0, wx1, wy1 = spec.ROIS_NORM["wedge_hyp"].to_px(w_px, h_px)
    add_gray_xobj(
        "ImWedgeHyp",
        _render_wedge_hyperbolic(wy1 - wy0, wx1 - wx0, h_px, spec.WEDGE_FREQ_MIN_CYC_H, spec.WEDGE_FREQ_MAX_CYC_H),
    )
    lx0, ly0, lx1, ly1 = spec.ROIS_NORM["wedge_lin"].to_px(w_px, h_px)
    add_gray_xobj(
        "ImWedgeLin",
        _render_wedge_linear(ly1 - ly0, lx1 - lx0, h_px, spec.WEDGE_FREQ_MIN_CYC_H, spec.WEDGE_FREQ_MAX_CYC_H),
    )
    nx0, ny0, nx1, ny1 = spec.ROIS_NORM["noise_patch"].to_px(w_px, h_px)
    add_gray_xobj("ImNoise", _render_noise_patch(ny1 - ny0, nx1 - nx0))

    # Build PDF content stream
    ops: list[str] = []
    # White background
    ops.append(f"0.96 g 0 0 {w_pt:.2f} {h_pt:.2f} re f")

    # Place embedded raster XObjects
    for name, key in (
        ("ImStarSin", "star_sin"),
        ("ImWedgeHyp", "wedge_hyp"),
        ("ImWedgeLin", "wedge_lin"),
        ("ImNoise", "noise_patch"),
    ):
        r = spec.ROIS_NORM[key]
        x = pt_x(r.x0)
        y = pt_y(r.y1)
        rw = pt_x(r.x1) - x
        rh = pt_y(r.y0) - y
        ops.append(f"q {rw:.2f} 0 0 {rh:.2f} {x:.2f} {y:.2f} cm /{name} Do Q")

    # Vector ArUco fiducials (6x6 grid including 1-cell black border)
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.ARUCO_DICT_NAME))
    for fid, rect in spec.fiducial_rects_norm(aspect).items():
        # Generate 6x6 marker (4x4 bits + 1-bit border)
        m6 = cv2.aruco.generateImageMarker(aruco_dict, fid, 6)
        x = pt_x(rect.x0)
        y_top = pt_y(rect.y0)
        side = pt_x(rect.x1) - x
        cell = side / 6.0
        ops.append("0 g")
        for r_i in range(6):
            for c_i in range(6):
                if m6[r_i, c_i] < 128:
                    cx = x + c_i * cell
                    cy = y_top - (r_i + 1) * cell
                    ops.append(f"{cx:.2f} {cy:.2f} {cell:.2f} {cell:.2f} re f")

    # Vector binary 144-spoke Siemens star
    cx = pt_x(spec.STAR_BIN_CENTER[0])
    cy = pt_y(spec.STAR_BIN_CENTER[1])
    r_out = spec.STAR_RADIUS_H * h_pt
    r_in = spec.STAR_INNER_RADIUS_H * h_pt
    # White disk background
    ops.append("0.95 g")
    n_circ = 180
    pts = [
        f"{cx + r_out * math.cos(2 * math.pi * k / n_circ):.2f} {cy + r_out * math.sin(2 * math.pi * k / n_circ):.2f}"
        for k in range(n_circ)
    ]
    ops.append(f"{pts[0]} m " + " ".join(f"{p} l" for p in pts[1:]) + " h f")
    # 144 dark spokes
    ops.append("0.05 g")
    dtheta = 2.0 * math.pi / (2.0 * spec.STAR_N_SPOKES)
    for k in range(spec.STAR_N_SPOKES):
        a0 = (2 * k + 1) * dtheta
        a1 = (2 * k + 2) * dtheta
        x0_i, y0_i = cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)
        x0_o, y0_o = cx + r_out * math.cos(a0), cy + r_out * math.sin(a0)
        x1_o, y1_o = cx + r_out * math.cos(a1), cy + r_out * math.sin(a1)
        x1_i, y1_i = cx + r_in * math.cos(a1), cy + r_in * math.sin(a1)
        ops.append(
            f"{x0_i:.2f} {y0_i:.2f} m {x0_o:.2f} {y0_o:.2f} l {x1_o:.2f} {y1_o:.2f} l {x1_i:.2f} {y1_i:.2f} l h f"
        )
    # Neutral grey center core
    ops.append("0.50 g")
    pts_in = [
        f"{cx + r_in * math.cos(2 * math.pi * k / 36):.2f} {cy + r_in * math.sin(2 * math.pi * k / 36):.2f}"
        for k in range(36)
    ]
    ops.append(f"{pts_in[0]} m " + " ".join(f"{p} l" for p in pts_in[1:]) + " h f")

    # Vector 5-deg slanted edges
    tan_t = math.tan(math.radians(spec.EDGE_TILT_DEG))
    for key, (lo, hi) in (
        ("edge_high", spec.EDGE_HIGH_CONTRAST),
        ("edge_low", spec.EDGE_LOW_CONTRAST),
    ):
        r = spec.ROIS_NORM[key]
        x0, y0 = pt_x(r.x0), pt_y(r.y1)
        rw, rh = pt_x(r.x1) - x0, pt_y(r.y0) - y0
        ops.append(f"{lo:.3f} g {x0:.2f} {y0:.2f} {rw:.2f} {rh:.2f} re f")
        xm = x0 + 0.5 * rw
        dx = 0.5 * rh * tan_t
        ops.append(
            f"{hi:.3f} g {xm + dx:.2f} {y0:.2f} m {x0 + rw:.2f} {y0:.2f} l "
            f"{x0 + rw:.2f} {y0 + rh:.2f} l {xm - dx:.2f} {y0 + rh:.2f} l h f"
        )

    # Vector 8-step grey wedge
    gw = spec.ROIS_NORM["grey_wedge"]
    gx0, gy0 = pt_x(gw.x0), pt_y(gw.y1)
    g_w, g_h = pt_x(gw.x1) - gx0, pt_y(gw.y0) - gy0
    sw = g_w / len(spec.GREY_STEPS)
    for idx, val in enumerate(spec.GREY_STEPS):
        ops.append(f"{val:.3f} g {gx0 + idx * sw:.2f} {gy0:.2f} {sw:.2f} {g_h:.2f} re f")

    # Thin borders around rectangular ROIs
    ops.append("0.12 G 0.6 w")
    for key in ("edge_high", "edge_low", "wedge_hyp", "wedge_lin", "noise_patch", "grey_wedge"):
        r = spec.ROIS_NORM[key]
        x0, y0 = pt_x(r.x0), pt_y(r.y1)
        ops.append(f"{x0:.2f} {y0:.2f} {pt_x(r.x1) - x0:.2f} {pt_y(r.y0) - y0:.2f} re S")

    # Vector scale bar
    bar_mm = 50.0 if paper == "A4" else 100.0
    bar_pt = bar_mm * pt_per_mm
    sb = spec.ROIS_NORM["scale_bar"]
    sb_cx = 0.5 * (pt_x(sb.x0) + pt_x(sb.x1))
    bx0 = sb_cx - 0.5 * bar_pt
    by0 = pt_y(0.55 * sb.y0 + 0.45 * sb.y1) - 4.0
    bh = 5.0
    n_seg = int(round(bar_mm / 10.0))
    for s_i in range(n_seg):
        col = "0.05" if (s_i % 2 == 0) else "0.96"
        ops.append(f"{col} g {bx0 + s_i * (bar_pt / n_seg):.2f} {by0:.2f} {bar_pt / n_seg:.2f} {bh:.2f} re f")
    ops.append(f"0.05 G 0.6 w {bx0:.2f} {by0:.2f} {bar_pt:.2f} {bh:.2f} re S")

    # Helvetica text labels
    def pdf_text(x: float, y: float, size: float, txt: str) -> str:
        safe = txt.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 {size:.1f} Tf 0.08 g {x:.2f} {y:.2f} Td ({safe}) Tj ET"

    title_size = 10.0 if paper == "A4" else 14.0
    lbl_size = 7.5 if paper == "A4" else 10.5
    ops.append(pdf_text(pt_x(sb.x0), pt_y(sb.y0) - 8.0, title_size, f"raw_sr WP10 Resolution Chart ({paper} {int(w_mm)}x{int(h_mm)} mm, 144-spoke)"))
    ops.append(pdf_text(bx0, by0 - lbl_size - 2.0, lbl_size, f"Scale bar: {int(bar_mm)} mm (verify 100% print scale)"))
    for text, key in (
        ("Sinusoidal 144-spoke", "star_sin"),
        ("Binary 144-spoke", "star_bin"),
        ("Edge 5 deg (HC)", "edge_high"),
        ("Edge 5 deg (LC)", "edge_low"),
        (f"Hyperbolic wedge ({int(spec.WEDGE_FREQ_MIN_CYC_H)}-{int(spec.WEDGE_FREQ_MAX_CYC_H)} cyc/H)", "wedge_hyp"),
        (f"Linear wedge ({int(spec.WEDGE_FREQ_MIN_CYC_H)}-{int(spec.WEDGE_FREQ_MAX_CYC_H)} cyc/H)", "wedge_lin"),
        ("Fine noise texture", "noise_patch"),
    ):
        r = spec.ROIS_NORM[key]
        ops.append(pdf_text(pt_x(r.x0), pt_y(r.y1) - lbl_size - 2.0, lbl_size, text))

    content_bytes = zlib.compress("\n".join(ops).encode("ascii"), level=6)

    # Assemble PDF objects
    objs: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Page, 4: Font Helvetica, 5: Content stream, 6..: XObjects
    xobj_refs = " ".join(f"/{name} {6 + i} 0 R" for i, (name, _, _, _) in enumerate(xobjects))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.2f} {h_pt:.2f}] "
        f"/Resources << /Font << /F1 4 0 R >> /XObject << {xobj_refs} >> >> "
        f"/Contents 5 0 R >>".encode("ascii")
    )
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(
        f"<< /Length {len(content_bytes)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
        + content_bytes
        + b"\nendstream"
    )
    for _, iw, ih, comp in xobjects:
        objs.append(
            f"<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
            f"/Length {len(comp)} >>\nstream\n".encode("ascii")
            + comp
            + b"\nendstream"
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = [0]
        for idx, body in enumerate(objs, start=1):
            offsets.append(f.tell())
            f.write(f"{idx} 0 obj\n".encode("ascii"))
            f.write(body)
            f.write(b"\nendobj\n")
        xref_pos = f.tell()
        f.write(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            f.write(f"{off:010d} 00000 n \n".encode("ascii"))
        f.write(
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )


def generate_charts(
    out_dir: str,
    sizes: tuple[str, ...] = ("A4", "A3"),
    dpi: int = 600,
    preview_path: str | None = None,
) -> dict[str, str]:
    """Generate PDF and PNG charts for the requested paper sizes."""
    os.makedirs(out_dir, exist_ok=True)
    outputs: dict[str, str] = {}
    for paper in sizes:
        png_path = os.path.join(out_dir, f"raw_sr_chart_{paper.lower()}_{dpi}dpi.png")
        pdf_path = os.path.join(out_dir, f"raw_sr_chart_{paper.lower()}.pdf")
        img = render_chart_raster(paper=paper, dpi=dpi)
        cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        write_vector_pdf(paper=paper, out_path=pdf_path)
        outputs[f"{paper}_png"] = png_path
        outputs[f"{paper}_pdf"] = pdf_path
        if preview_path and paper == "A4":
            os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)
            prev = cv2.resize(img, (1600, int(round(1600 * 210.0 / 297.0))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(preview_path, prev, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            outputs["preview_png"] = preview_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WP10 printable resolution test charts.")
    parser.add_argument("--out-dir", default="/tmp/rawsr/wp10/charts", help="Output directory for PDF/PNG charts.")
    parser.add_argument("--sizes", nargs="+", default=["A4", "A3"], choices=["A4", "A3"], help="Paper sizes.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster PNG resolution in DPI.")
    parser.add_argument("--preview", default=None, help="Optional path for a compact preview PNG.")
    args = parser.parse_args()
    outputs = generate_charts(args.out_dir, tuple(args.sizes), dpi=args.dpi, preview_path=args.preview)
    for k, p in outputs.items():
        sz = os.path.getsize(p) / (1024.0 * 1024.0)
        print(f"{k}: {p} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
