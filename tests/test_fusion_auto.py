"""fusion='auto' (default since WP12): MAP at 2x with a CUDA GPU, else demosaic_drizzle."""
from types import SimpleNamespace

import numpy as np
import pytest

import raw_sr.core.gpu as gpu
from raw_sr.core.superres import SuperResolution


def _frames(n=4, size=64, mosaic=True):
    out = []
    for _ in range(n):
        out.append(SimpleNamespace(
            rgb=np.zeros((size, size, 3), np.float32),
            mosaic=np.zeros((size, size), np.float32) if mosaic else None,
            pattern=np.array([[0, 1], [3, 2]], np.int8) if mosaic else None,
        ))
    return out


@pytest.fixture
def with_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_available", lambda: True)


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "gpu_available", lambda: False)


def test_default_is_auto():
    sr = SuperResolution(ratio=2.0)
    assert sr.fusion_requested == "auto"


def test_auto_picks_map_at_2x_with_gpu(with_gpu):
    assert SuperResolution(ratio=2.0)._resolve_fusion(_frames())[0] == "map"


def test_auto_falls_back_without_gpu(no_gpu):
    f, why = SuperResolution(ratio=2.0)._resolve_fusion(_frames())
    assert f == "demosaic_drizzle" and "GPU" in why


def test_auto_respects_use_gpu_false(with_gpu):
    f, _ = SuperResolution(ratio=2.0, use_gpu=False)._resolve_fusion(_frames())
    assert f == "demosaic_drizzle"


@pytest.mark.parametrize("ratio", [1.5, 3.0, 4.0])
def test_auto_keeps_default_off_2x(with_gpu, ratio):
    assert SuperResolution(ratio=ratio)._resolve_fusion(_frames())[0] == "demosaic_drizzle"


def test_auto_needs_mosaic(with_gpu):
    assert SuperResolution(ratio=2.0)._resolve_fusion(_frames(mosaic=False))[0] == "demosaic_drizzle"


def test_auto_memory_guard(with_gpu):
    # 2048^2 ROI x 9 frames at 2x exceeded 6 GB in measurement -> must not pick map.
    f, why = SuperResolution(ratio=2.0)._resolve_fusion(_frames(n=9, size=2048))
    assert f == "demosaic_drizzle" and "memory" in why
    assert SuperResolution(ratio=2.0)._resolve_fusion(_frames(n=9, size=1024))[0] == "map"


@pytest.mark.parametrize("name", ["demosaic_drizzle", "cfa_drizzle", "cfa_kr", "map"])
def test_explicit_fusion_unchanged(with_gpu, name):
    f, why = SuperResolution(ratio=3.0, fusion=name)._resolve_fusion(_frames())
    assert (f, why) == (name, "explicit")
