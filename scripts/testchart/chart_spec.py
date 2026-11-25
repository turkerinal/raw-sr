"""Canonical geometry specification for the WP10 resolution test chart.

All coordinates are defined in normalized chart units [0, 1] x [0, 1] on a
landscape sqrt(2):1 canvas (A4 = 297 x 210 mm, A3 = 420 x 297 mm).
Both ``make_chart.py`` and ``measure_chart.py`` import this module so that
rendering and ROI extraction share the exact same geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

# Paper sizes in millimetres (landscape: width x height)
PAPER_MM: dict[str, tuple[float, float]] = {
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
}

# ArUco dictionary and corner marker IDs (TL, TR, BR, BL)
ARUCO_DICT_NAME = "DICT_4X4_50"
FIDUCIAL_IDS: tuple[int, int, int, int] = (0, 1, 2, 3)

# Fiducial square size and margin as fraction of chart height (H)
# Keeping fiducials square in physical mm: width_norm = size_h / aspect
FIDUCIAL_MARGIN_H = 0.045
FIDUCIAL_SIZE_H = 0.105

# Canonical warped image height in pixels for measurement (width = round(H * sqrt(2)))
CANONICAL_H = 1400
CANONICAL_W = int(round(CANONICAL_H * (297.0 / 210.0)))  # 1980

# Siemens star parameters
STAR_N_SPOKES = 144  # 144 full cycles around 360 degrees
STAR_RADIUS_H = 0.175  # outer radius as fraction of chart height
STAR_INNER_RADIUS_H = 0.012  # blank neutral core radius

# Slanted edge parameters
EDGE_TILT_DEG = 5.0
EDGE_HIGH_CONTRAST = (0.05, 0.95)
EDGE_LOW_CONTRAST = (0.30, 0.70)

# Grey step wedge: 8 linear reflectance levels
GREY_STEPS: tuple[float, ...] = (0.05, 0.15, 0.28, 0.42, 0.56, 0.70, 0.83, 0.95)

# Random noise patch seed (deterministic)
NOISE_PATCH_SEED = 20261124


@dataclass(frozen=True)
class RectNorm:
    """Axis-aligned rectangle in normalized [0, 1] x [0, 1] chart coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float

    def to_px(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return integer (x0, y0, x1, y1) pixel coordinates."""
        return (
            int(round(self.x0 * width)),
            int(round(self.y0 * height)),
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
        )


def fiducial_rects_norm(aspect: float = 297.0 / 210.0) -> dict[int, RectNorm]:
    """Return normalized bounding boxes for the 4 corner ArUco markers (IDs 0..3)."""
    mh = FIDUCIAL_MARGIN_H
    sh = FIDUCIAL_SIZE_H
    mw = mh / aspect
    sw = sh / aspect
    return {
        0: RectNorm(mw, mh, mw + sw, mh + sh),  # Top-Left
        1: RectNorm(1.0 - mw - sw, mh, 1.0 - mw, mh + sh),  # Top-Right
        2: RectNorm(1.0 - mw - sw, 1.0 - mh - sh, 1.0 - mw, 1.0 - mh),  # Bottom-Right
        3: RectNorm(mw, 1.0 - mh - sh, mw + sw, 1.0 - mh),  # Bottom-Left
    }


def fiducial_centers_norm(aspect: float = 297.0 / 210.0) -> dict[int, tuple[float, float]]:
    """Return normalized (cx, cy) centers of the 4 corner ArUco markers."""
    rects = fiducial_rects_norm(aspect)
    return {
        fid: (0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1))
        for fid, r in rects.items()
    }


# Layout of all measurement regions in normalized [0, 1] x [0, 1] coordinates.
# Canvas aspect is ~1.414 (landscape A4/A3).
# Top band (y: 0.04..0.15): TL/TR fiducials + title + scale bar
# Upper-mid band (y: 0.17..0.54):
#   - Left: Sinusoidal 144-spoke Siemens star
#   - Center: Binary 144-spoke Siemens star
#   - Right: High-contrast & low-contrast 5-deg slanted edges
# Lower-mid band (y: 0.57..0.82):
#   - Left: Hyperbolic line-pair wedge (horizontal sweep, vertical bars)
#   - Center: Linear line-pair wedge (vertical sweep, horizontal bars)
#   - Right: Fine random-noise texture patch
# Bottom band (y: 0.85..0.955): BL/BR fiducials + 8-step grey wedge

STAR_SIN_CENTER = (0.23, 0.355)
STAR_BIN_CENTER = (0.53, 0.355)

ROIS_NORM: dict[str, RectNorm] = {
    "star_sin": RectNorm(
        STAR_SIN_CENTER[0] - STAR_RADIUS_H / (297.0 / 210.0) - 0.01,
        STAR_SIN_CENTER[1] - STAR_RADIUS_H - 0.01,
        STAR_SIN_CENTER[0] + STAR_RADIUS_H / (297.0 / 210.0) + 0.01,
        STAR_SIN_CENTER[1] + STAR_RADIUS_H + 0.01,
    ),
    "star_bin": RectNorm(
        STAR_BIN_CENTER[0] - STAR_RADIUS_H / (297.0 / 210.0) - 0.01,
        STAR_BIN_CENTER[1] - STAR_RADIUS_H - 0.01,
        STAR_BIN_CENTER[0] + STAR_RADIUS_H / (297.0 / 210.0) + 0.01,
        STAR_BIN_CENTER[1] + STAR_RADIUS_H + 0.01,
    ),
    "edge_high": RectNorm(0.70, 0.18, 0.82, 0.52),
    "edge_low": RectNorm(0.84, 0.18, 0.96, 0.52),
    "wedge_hyp": RectNorm(0.10, 0.58, 0.44, 0.80),
    "wedge_lin": RectNorm(0.47, 0.58, 0.72, 0.80),
    "noise_patch": RectNorm(0.75, 0.58, 0.95, 0.80),
    "grey_wedge": RectNorm(0.15, 0.855, 0.85, 0.945),
    "scale_bar": RectNorm(0.28, 0.065, 0.72, 0.125),
}

# Frequency range of the line-pair wedges in cycles per chart height (cyc/H)
# On A4 (H=210 mm): 40..420 cyc/H corresponds to ~0.19..2.0 lp/mm.
# When the chart fills a 3000 px tall sensor, 1500 cyc/H = Nyquist (0.5 cyc/px).
# For a 1000-2000 px chart height in sensor pixels, 60..600 cyc/H spans 0.06..0.6 cyc/LR px.
WEDGE_FREQ_MIN_CYC_H = 60.0
WEDGE_FREQ_MAX_CYC_H = 600.0
