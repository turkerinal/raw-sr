"""Image helpers: ndarray -> PNG data URL, resizing, heatmaps, view transform.

Owned by AGENT-A. Baseline implementation by the orchestrator.
"""
from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np


def fit_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale so the longest side is <= max_dim (no upscaling)."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def to_uint8_view(linear: np.ndarray, exposure: float = 4.0, gamma: float = 2.2) -> np.ndarray:
    """Non-destructive display transform of a linear float image to uint8 RGB."""
    v = np.clip(np.asarray(linear, dtype=np.float32) * float(exposure), 0.0, 1.0)
    v = np.power(v, 1.0 / max(float(gamma), 1e-6))
    return (v * 255.0 + 0.5).astype(np.uint8)


def apply_view(linear: np.ndarray, gamma: float, brightness: float, contrast: float) -> np.ndarray:
    """Apply a ViewTransform-equivalent curve to a linear float image -> uint8."""
    d = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0).copy()
    d = np.power(d, 1.0 / max(float(gamma), 1e-6))
    d = (d - 0.5) * float(contrast) + 0.5
    d = d + float(brightness)
    d = np.clip(d, 0.0, 1.0)
    return (d * 255.0 + 0.5).astype(np.uint8)


def encode_png(rgb: np.ndarray) -> str:
    """Encode an RGB uint8 array as a `data:image/png;base64,...` URL."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def heatmap(data: np.ndarray, max_dim: int) -> str:
    """Normalize a scalar map and render it as a TURBO heatmap data URL."""
    d = np.asarray(data, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    d = d - float(np.nanmin(d)) if np.isfinite(d).any() else d
    span = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    if span > 1e-9:
        d = d / span
    else:
        d = np.zeros_like(d)
    d = np.clip(d, 0.0, 1.0)
    small = fit_max_dim(d, max_dim)
    u8 = (small * 255.0 + 0.5).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return encode_png(rgb)


def png_bytes_from_rgb(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return buf.tobytes()


__all__ = [
    'fit_max_dim',
    'to_uint8_view',
    'apply_view',
    'encode_png',
    'heatmap',
    'png_bytes_from_rgb',
]
