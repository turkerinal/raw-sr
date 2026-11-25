"""Backend tests. Baseline by the orchestrator; extended by AGENT-A.

These exercise the real SR core through the HTTP surface using the synthetic
dev endpoint, so they need no RAW sample files. Pure helpers (ROI overlap,
exclude resolution, the decode cache) are unit-tested directly.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from desktop.backend import session
from desktop.backend.server import app, manager

client = TestClient(app)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _wait(job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f'/api/jobs/{job_id}').json()
        if last['state'] in ('done', 'error', 'cancelled'):
            return last
        time.sleep(0.1)
    raise TimeoutError(f'job {job_id} did not finish; last={last}')


def _synthetic(ratio: float = 2.0, n_frames: int = 5) -> str:
    r = client.post('/api/dev/synthetic', json={'ratio': ratio, 'n_frames': n_frames})
    assert r.status_code == 200, r.text
    return r.json()['job_id']


def _done_synthetic(**kwargs) -> tuple[str, dict]:
    job_id = _synthetic(**kwargs)
    status = _wait(job_id)
    assert status['state'] == 'done', status
    return job_id, status


# --------------------------------------------------------------------------- #
# Health & session validation
# --------------------------------------------------------------------------- #

def test_health():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok', 'version': '1.0.0'}


def test_capabilities_shape():
    r = client.get('/api/capabilities')
    assert r.status_code == 200
    body = r.json()
    assert {'opencl', 'cpu_count'} <= set(body)  # additive surface (opencl_device/vendor)
    assert isinstance(body['opencl'], bool)
    assert isinstance(body['cpu_count'], int) and body['cpu_count'] >= 1
    # Cached probe: repeated calls are stable and return a copy (non-breaking
    # surface — callers must never see a mutating dict).
    again = client.get('/api/capabilities')
    assert again.status_code == 200 and again.json() == body


def test_files_rejects_non_raw(tmp_path):
    stray = tmp_path / 'not_a_raw.png'
    stray.write_bytes(b'not an image')
    r = client.post('/api/session/files', json={'paths': [str(stray)]})
    assert r.status_code == 200
    body = r.json()
    assert body['frames'] == []
    assert len(body['rejected']) == 1
    assert body['rejected'][0]['path'] == str(stray)
    assert body['rejected'][0]['reason']


def test_files_rejects_missing_file():
    r = client.post('/api/session/files', json={'paths': ['/tmp/raw_sr_definitely_missing.dng']})
    assert r.status_code == 200
    body = r.json()
    assert body['frames'] == []
    assert 'not found' in body['rejected'][0]['reason'].lower()


def test_files_empty_400():
    r = client.post('/api/session/files', json={'paths': []})
    assert r.status_code == 400
    assert isinstance(r.json()['detail'], str)


def test_session_preview_missing_file_404():
    r = client.post('/api/session/preview', json={'path': '/tmp/raw_sr_nope.dng', 'max_dim': 256})
    assert r.status_code == 404
    assert r.json()['detail']


def test_session_preview_non_raw_400(tmp_path):
    stray = tmp_path / 'x.png'
    stray.write_bytes(b'x')
    r = client.post('/api/session/preview', json={'path': str(stray), 'max_dim': 256})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Analyze & quality (tolerant of undecodable paths)
# --------------------------------------------------------------------------- #

def test_analyze_shape_with_missing_paths():
    r = client.post('/api/analyze', json={
        'paths': ['/tmp/raw_sr_missing_a.dng', '/tmp/raw_sr_missing_b.dng'],
        'target_roi': [0, 0, 100, 100],
        'search_area': [0, 0, 150, 150],
        'sr_ratio': 2.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body['total_frames'] == 2
    assert body['usable_frames'] == 0
    assert body['per_frame_usable'] == [False, False]
    assert set(body['motion']) == {'avg_shift', 'max_shift', 'sufficient', 'message', 'shifts'}
    assert len(body['motion']['shifts']) == 2
    assert set(body['sr_ratio']) == {'requested', 'achievable', 'coverage_fraction', 'message'}


def test_analyze_empty_400():
    r = client.post('/api/analyze', json={'paths': [], 'sr_ratio': 2.0})
    assert r.status_code == 400


def test_analyze_invalid_ratio_400():
    r = client.post('/api/analyze', json={'paths': ['x.dng'], 'sr_ratio': 0})
    assert r.status_code == 400
    assert isinstance(r.json()['detail'], str)


def test_analyze_invalid_roi_400():
    r = client.post('/api/analyze', json={'paths': ['x.dng'], 'sr_ratio': 2.0,
                                          'target_roi': [0, 0, 0, 10]})
    assert r.status_code == 400


def test_quality_shape():
    r = client.post('/api/quality', json={'paths': ['/tmp/raw_sr_missing.dng']})
    assert r.status_code == 200
    body = r.json()
    assert body['flags'] == [False]
    assert len(body['scores']) == 1


def test_quality_empty_400():
    r = client.post('/api/quality', json={'paths': []})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# ROI overlap helper (unit)
# --------------------------------------------------------------------------- #

def test_roi_overlap_fraction():
    # Core convention: frame(x) ~= ref(x + d), so a frame shifted by +d covers
    # reference coordinates [d, W + d) and misses the left edge instead.
    # No shift, ROI == full frame -> fully covered.
    assert session.roi_overlap_fraction((0, 0, 100, 100), (100, 100), (0, 0)) == pytest.approx(1.0)
    # Frame shifted right by 10px in a 100px frame covers ref [10, 110) ∩ [0, 100).
    assert session.roi_overlap_fraction((0, 0, 100, 100), (100, 100), (10, 0)) == pytest.approx(0.9)
    # ROI on the left edge, outside the shifted frame's valid region -> 0.
    assert session.roi_overlap_fraction((0, 0, 5, 100), (100, 100), (10, 0)) == pytest.approx(0.0)
    # ROI on the right edge is fully covered by that +10px shift.
    assert session.roi_overlap_fraction((95, 0, 5, 100), (100, 100), (10, 0)) == pytest.approx(1.0)
    # ROI fully inside the valid region -> 1.
    assert session.roi_overlap_fraction((10, 10, 20, 20), (100, 100), (0, 0)) == pytest.approx(1.0)
    # Vertical axis uses the same convention.
    assert session.roi_overlap_fraction((0, 95, 100, 5), (100, 100), (0, 10)) == pytest.approx(1.0)
    assert session.roi_overlap_fraction((0, 0, 100, 5), (100, 100), (0, 10)) == pytest.approx(0.0)


def test_roi_overlap_fraction_matches_real_core_shift():
    """Pin the sign against the real estimate_translation, not a stub."""
    import cv2
    from raw_sr.core.superres import estimate_translation

    rng = np.random.default_rng(5)
    ref = cv2.GaussianBlur(rng.random((160, 200), dtype=np.float32), (0, 0), 2.0)
    tx, ty = 12.5, -7.25
    mov = cv2.warpAffine(ref, np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]],
                                       dtype=np.float64), (200, 160),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    dx, dy = estimate_translation(ref, mov)
    # Content moved right/down by (tx, ty): the frame covers ref [d, W+d).
    assert dx == pytest.approx(-tx, abs=0.5)
    assert dy == pytest.approx(-ty, abs=0.5)
    # The frame misses the right edge (content moved right) and the top edge
    # (content moved up); the opposite edges stay covered.
    assert session.roi_overlap_fraction((195, 20, 5, 120), (160, 200), (dx, dy)) == pytest.approx(0.0, abs=0.05)
    assert session.roi_overlap_fraction((20, 0, 160, 5), (160, 200), (dx, dy)) == pytest.approx(0.0, abs=0.05)
    assert session.roi_overlap_fraction((0, 20, 5, 120), (160, 200), (dx, dy)) == pytest.approx(1.0)
    assert session.roi_overlap_fraction((20, 155, 160, 5), (160, 200), (dx, dy)) == pytest.approx(1.0)


def test_resolve_roi_clamps_around_the_user_position(monkeypatch):
    """An oversized ROI shrinks around its own centre, not the sensor centre."""
    import desktop.backend.server as server

    monkeypatch.setattr(server, '_frame_size', lambda path: (3000, 4000))
    # 4000x1000 -> 800x200 (area 160,000 == cap), centred on (2300, 600).
    roi, clipped = server.resolve_roi('x.dng', (300, 100, 4000, 1000),
                                      ratio=1.0, max_roi_area=160000,
                                      max_hr_area=10 ** 12)
    assert clipped is True
    x, y, w, h = roi
    assert (w, h) == (800, 200)
    assert x + w / 2 == pytest.approx(2300)
    assert y + h / 2 == pytest.approx(600)
    # The old centred-square fallback would have used the sensor centre.
    assert abs(x - (4000 - 800) // 2) > 100


def test_resolve_roi_clamped_area_never_exceeds_cap(monkeypatch):
    import desktop.backend.server as server

    monkeypatch.setattr(server, '_frame_size', lambda path: (3000, 4000))
    for req in ((0, 0, 4000, 3000), (300, 200, 2001, 1001), (3900, 2900, 400, 400)):
        roi, clipped = server.resolve_roi('x.dng', req, ratio=1.0,
                                          max_roi_area=123457, max_hr_area=10 ** 12)
        x, y, w, h = roi
        assert clipped is True
        assert w * h <= 123457
        assert 0 <= x <= 4000 - w and 0 <= y <= 3000 - h


def test_memory_guard_is_frame_count_aware(monkeypatch):
    """A 30-frame burst is refused on a budget where 2 frames fit."""
    from fastapi import HTTPException
    import desktop.backend.server as server

    monkeypatch.setattr(server, 'resolve_roi',
                        lambda *a, **k: ((0, 0, 8, 8), False))
    budget = 230_000  # 8x8 LR at 4x = 1,024 HR px
    assert server._memory_guard('x', None, 4.0, budget, n_frames=2) is None
    with pytest.raises(HTTPException) as ei:
        server._memory_guard('x', None, 4.0, budget, n_frames=30)
    assert ei.value.status_code == 413
    assert '30 frames' in ei.value.detail


def test_analyze_usability_is_roi_aware(monkeypatch):
    """Same shifts, different ROI -> different per_frame_usable (not a global cap)."""
    from raw_sr.core import superres as core_sr

    frame = np.full((100, 100, 3), 0.5, dtype=np.float32)

    def fake_load(path, max_dim=1024):
        return frame.copy(), {}, (100, 100)

    def patch_shift(sx, sy):
        # analyze hardcodes the reference at (0, 0) and only estimates the rest.
        monkeypatch.setattr(core_sr, 'estimate_translation',
                            lambda ref, mov, sx=sx, sy=sy: (sx, sy))

    monkeypatch.setattr(session, 'load_scaled_linear', fake_load)

    # Core convention: the +30px frame covers ref [30, 130), so an ROI on the
    # LEFT strip is no longer reached.
    patch_shift(30.0, 0.0)
    left = session.analyze(['a.dng', 'b.dng'], (0, 0, 30, 100), None, 2.0)
    assert left['per_frame_usable'] == [True, False]
    assert left['usable_frames'] == 1

    # A ROI the shifted frame does cover: now both frames are usable.
    patch_shift(30.0, 0.0)
    inside = session.analyze(['a.dng', 'b.dng'], (40, 0, 30, 100), None, 2.0)
    assert inside['per_frame_usable'] == [True, True]
    assert inside['usable_frames'] == 2

    # A search area the shifted frame cannot cover makes it unusable even
    # though the target ROI itself is covered.
    patch_shift(30.0, 0.0)
    search = session.analyze(['a.dng', 'b.dng'], (40, 0, 30, 100), (0, 0, 20, 100), 2.0)
    assert search['per_frame_usable'] == [True, False]

    # The reported shift list stays aligned with `paths`.
    assert left['motion']['shifts'] == [[0.0, 0.0], [30.0, 0.0]]


def test_analyze_reports_motion_in_source_pixels(monkeypatch):
    """`/api/analyze` motion shifts are SOURCE pixels, not preview pixels.

    The analysis runs on a downscaled preview (`fit_max_dim(rgb, 512)`), so the
    raw per-frame shifts come out in preview pixels. The API must scale them
    back up by source/preview per axis — the ROI panel consumes them as source
    pixels — while leaving the reference frame at (0, 0).
    """
    from raw_sr.core import superres as core_sr

    # A 100x100 analysis preview standing in for a 1000x1000 source frame: a
    # known 10x downscale. `fit_max_dim(..., 512)` leaves the preview untouched.
    preview = np.full((100, 100, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr(session, 'load_scaled_linear',
                        lambda path, max_dim=1024: (preview.copy(), {}, (1000, 1000)))
    # estimate_translation runs on the preview and returns preview pixels.
    monkeypatch.setattr(core_sr, 'estimate_translation',
                        lambda ref, mov: (5.0, 3.0))

    out = session.analyze(['a.dng', 'b.dng'], None, None, 2.0)

    # Reference frame stays (0, 0); the other frame is scaled 10x per axis.
    assert out['motion']['shifts'][0] == [0.0, 0.0]
    assert out['motion']['shifts'][1] == pytest.approx([50.0, 30.0], rel=1e-6)
    expected = float(np.hypot(50.0, 30.0))   # source-pixel magnitude
    assert out['motion']['avg_shift'] == pytest.approx(expected, rel=1e-6)
    assert out['motion']['max_shift'] == pytest.approx(expected, rel=1e-6)
    # The message quotes the source-pixel value the panel displays.
    assert f'{expected:.2f}px' in out['motion']['message']
    # Scaling is per axis, so an anisotropic preview maps correctly too.
    non_square = np.full((100, 200, 3), 0.5, dtype=np.float32)
    monkeypatch.setattr(session, 'load_scaled_linear',
                        lambda path, max_dim=1024: (non_square.copy(), {}, (1000, 1000)))
    out2 = session.analyze(['a.dng', 'b.dng'], None, None, 2.0)
    # sx = 1000/200 = 5, sy = 1000/100 = 10.
    assert out2['motion']['shifts'][1] == pytest.approx([25.0, 30.0], rel=1e-6)


# --------------------------------------------------------------------------- #
# exclude resolution (unit)
# --------------------------------------------------------------------------- #

def test_resolve_exclude_ids_and_fallbacks():
    session.clear_frame_ids()
    paths = ['a.dng', 'b.dng', 'c.dng']
    assert session.resolve_exclude(paths, []) == paths
    assert session.resolve_exclude(paths, ['1']) == ['a.dng', 'c.dng']
    assert session.resolve_exclude(paths, ['0', '2']) == ['b.dng']
    # Unknown, malformed and out-of-range ids are ignored, never raise.
    assert session.resolve_exclude(paths, ['99', 'x', '-1', '']) == paths
    # An id given as a path string also resolves.
    assert session.resolve_exclude(paths, ['b.dng']) == ['a.dng', 'c.dng']
    # Duplicates are harmless.
    assert session.resolve_exclude(paths, ['1', '1', 'nope']) == ['a.dng', 'c.dng']


def test_resolve_exclude_with_id_map():
    # The id -> path map from /api/session/files is authoritative. Here frame
    # id "1" (/x/b.dng) was already filtered out of `paths`, so it must be
    # ignored rather than being mistaken for the new index 1.
    paths = ['/x/a.dng', '/x/c.dng', '/x/d.dng']
    id_map = {'0': '/x/a.dng', '1': '/x/b.dng', '2': '/x/c.dng', '3': '/x/d.dng'}
    assert session.resolve_exclude(paths, ['1'], id_map) == paths
    assert session.resolve_exclude(paths, ['0', '2'], id_map) == ['/x/d.dng']
    # Ids absent from the map still fall back to index resolution.
    assert session.resolve_exclude(paths, ['1'], {}) == ['/x/a.dng', '/x/d.dng']


# --------------------------------------------------------------------------- #
# Decode cache (unit)
# --------------------------------------------------------------------------- #

def test_decode_cache_reuses_decode(tmp_path, monkeypatch):
    p = tmp_path / 'frame.dng'
    p.write_bytes(b'fake raw bytes')
    calls = {'n': 0}

    def fake_load(path, half_size=True):
        calls['n'] += 1
        return np.zeros((30, 40, 3), dtype=np.float32), {'ok': True}, (60, 80)

    session.clear_cache()
    monkeypatch.setattr(session, 'load_preview_linear', fake_load)
    try:
        a, meta, hw = session.load_scaled_linear(str(p))
        b, _, _ = session.load_scaled_linear(str(p))
        assert calls['n'] == 1                 # second call hit the cache
        assert a.shape == b.shape
        assert hw == (60, 80) and meta['ok'] is True
        # A different max_dim is a different cache key -> one new decode.
        session.load_scaled_linear(str(p), 512)
        assert calls['n'] == 2
    finally:
        session.clear_cache()


# --------------------------------------------------------------------------- #
# Process validation (no decode needed)
# --------------------------------------------------------------------------- #

def test_process_missing_file_404(tmp_path):
    r = client.post('/api/process', json={'paths': [str(tmp_path / 'nope.dng')], 'sr_ratio': 2.0})
    assert r.status_code == 404


def test_process_non_raw_400(tmp_path):
    stray = tmp_path / 'x.png'
    stray.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(stray)], 'sr_ratio': 2.0})
    assert r.status_code == 400


def test_process_invalid_registration_400(tmp_path):
    raw = tmp_path / 'a.dng'
    raw.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(raw)], 'sr_ratio': 2.0,
                                          'registration': 'bogus'})
    assert r.status_code == 400


def test_process_all_excluded_400(tmp_path):
    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a), str(b)], 'sr_ratio': 2.0,
                                          'exclude': ['0', '1']})
    assert r.status_code == 400


def test_process_empty_paths_400():
    r = client.post('/api/process', json={'paths': [], 'sr_ratio': 2.0})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# use_opencl plumbing: request -> _make_superres kwargs
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('sent, forwarded', [
    ('absent', None),   # field omitted -> never passed, core auto-detects
    (None, None),       # explicit null -> never passed, core auto-detects
    (True, True),       # force GPU on
    (False, False),     # force CPU (False must survive, unlike None)
])
def test_process_forwards_use_opencl_only_when_set(tmp_path, monkeypatch, sent, forwarded):
    """The request accepts use_opencl in {absent, null, true, false} and reaches
    _make_superres with the kwarg ONLY when the request set it non-None."""
    import desktop.backend.server as server

    _mock_process_core(monkeypatch)
    real_make = server._make_superres
    captured = {}

    def recorder(**kw):
        captured.update(kw)
        # Run the REAL builder too, so an unsupported `use_opencl` on the
        # installed core would crash the job if the signature guard regressed.
        real_make(**kw)
        return type('S', (), {'run': lambda self, frames: _FakeSRResult()})()

    monkeypatch.setattr(server, '_make_superres', recorder)

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    body = {'paths': [str(a), str(b)], 'sr_ratio': 2.0}
    if sent != 'absent':
        body['use_opencl'] = sent

    r = client.post('/api/process', json=body)
    assert r.status_code == 200, r.text
    assert _wait(r.json()['job_id'])['state'] == 'done'

    assert ('use_opencl' in captured) == (forwarded is not None), captured
    if forwarded is not None:
        assert captured['use_opencl'] is forwarded


@pytest.mark.parametrize('sent, forwarded', [
    ('absent', None),   # field omitted -> never passed, core auto-detects
    (None, None),       # explicit null -> never passed, core auto-detects
    (True, True),       # force robust (Tukey) weights on
    (False, False),     # force plain fusion (False must survive, unlike None)
])
def test_process_forwards_robust_fusion_only_when_set(tmp_path, monkeypatch, sent, forwarded):
    """The request accepts robust_fusion in {absent, null, true, false} and
    reaches _make_superres with the kwarg ONLY when the request set it non-None."""
    import desktop.backend.server as server

    _mock_process_core(monkeypatch)
    real_make = server._make_superres
    captured = {}

    def recorder(**kw):
        captured.update(kw)
        # Run the REAL builder too, so an unsupported `robust_fusion` on the
        # installed core would crash the job if the signature guard regressed.
        real_make(**kw)
        return type('S', (), {'run': lambda self, frames: _FakeSRResult()})()

    monkeypatch.setattr(server, '_make_superres', recorder)

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    body = {'paths': [str(a), str(b)], 'sr_ratio': 2.0}
    if sent != 'absent':
        body['robust_fusion'] = sent

    r = client.post('/api/process', json=body)
    assert r.status_code == 200, r.text
    assert _wait(r.json()['job_id'])['state'] == 'done'

    assert ('robust_fusion' in captured) == (forwarded is not None), captured
    if forwarded is not None:
        assert captured['robust_fusion'] is forwarded


def test_process_rejects_non_boolean_robust_fusion(tmp_path):
    """robust_fusion is strictly bool|null; a non-boolean is rejected."""
    a = tmp_path / 'a.dng'
    a.write_bytes(b'x')
    r = client.post('/api/process',
                    json={'paths': [str(a)], 'sr_ratio': 2.0, 'robust_fusion': 'maybe'})
    assert r.status_code == 400, r.text
    assert 'robust_fusion' in str(r.json()['detail'])


def test_make_superres_signature_guard_filters_unknown_and_none():
    """_make_superres never forwards kwargs the installed core cannot take."""
    from desktop.backend.server import _make_superres
    from raw_sr.core.superres import SuperResolution

    # Unknown kwargs are dropped (core evolves in parallel).
    assert isinstance(_make_superres(ratio=2.0, totally_unknown_kw=5), SuperResolution)
    # None values are dropped: use_opencl=None means "not requested".
    assert isinstance(_make_superres(ratio=2.0, use_opencl=None), SuperResolution)
    # Non-None use_opencl must work whether or not the installed core accepts
    # the argument yet (True forces GPU, False forces CPU when supported).
    assert isinstance(_make_superres(ratio=2.0, use_opencl=True), SuperResolution)
    assert isinstance(_make_superres(ratio=2.0, use_opencl=False), SuperResolution)


# --------------------------------------------------------------------------- #
# Jobs: polling, result shape, render, export
# --------------------------------------------------------------------------- #

def test_job_polling_reaches_done():
    job_id = _synthetic()
    seen_running_or_done = False
    deadline = time.time() + 120
    while time.time() < deadline:
        body = client.get(f'/api/jobs/{job_id}').json()
        assert body['state'] in ('queued', 'running', 'done', 'error', 'cancelled')
        if body['state'] in ('running', 'done'):
            seen_running_or_done = True
        if body['state'] == 'done':
            assert body['progress'] == 100
            assert body['error'] is None
            break
        time.sleep(0.05)
    else:
        raise TimeoutError('job never completed')
    assert seen_running_or_done


def test_result_alignment_shape():
    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    res = client.get(f'/api/jobs/{job_id}/result')
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {'report', 'images', 'alignment', 'view_transform'}

    report = body['report']
    for key in ('used_frames', 'total_frames', 'ratio', 'hr_shape', 'roi',
                'roi_clipped', 'metrics', 'low_coverage_warning'):
        assert key in report
    assert len(report['hr_shape']) == 3
    assert len(report['roi']) == 4
    assert report['used_frames'] >= 2
    for key in ('hf_gain_db', 'hf_gain_db_linear', 'sharpness_ratio',
                'mean_shift_px', 'max_shift_px', 'registration',
                'opencl_scatter', 'opencl_device'):
        assert key in report['metrics']

    images = body['images']
    for key in ('fused', 'before', 'width', 'height'):
        assert key in images
    assert images['fused'].startswith('data:image/png;base64,')
    assert images['before'].startswith('data:image/png;base64,')
    assert images['width'] > 0 and images['height'] > 0

    al = body['alignment']
    assert set(al) == {'score', 'level', 'message', 'mean_zncc', 'min_zncc',
                       'per_frame', 'map'}
    assert 0.0 <= al['score'] <= 1.0
    assert al['level'] in ('ok', 'warn', 'bad')
    assert isinstance(al['message'], str) and al['message']
    assert 0.0 <= al['min_zncc'] <= al['mean_zncc'] <= 1.0
    assert al['map'].startswith('data:image/png;base64,')
    for f in al['per_frame']:
        assert set(f) == {'id', 'shift', 'zncc', 'residual', 'used'}
        assert isinstance(f['id'], str)
        assert len(f['shift']) == 2
        assert 0.0 <= f['zncc'] <= 1.0
        assert f['residual'] >= 0.0
        assert isinstance(f['used'], bool)

    for key in ('gamma', 'brightness', 'contrast'):
        assert key in body['view_transform']


def test_result_metrics_report_fusion_engine():
    """The real-core result payload names the fusion engine (additive
    contract, see desktop/ARCHITECTURE.md): opencl_scatter is always one of
    cpu/gpu/unavailable, opencl_device is str|null, and both survive JSON
    serialization to the client. Older metrics fields stay untouched."""
    import json

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']

    assert m['opencl_scatter'] in ('cpu', 'gpu', 'unavailable')
    assert m['opencl_device'] is None or isinstance(m['opencl_device'], str)
    # Pre-existing metrics are not disturbed by the additive keys.
    for key in ('hf_gain_db', 'sharpness_ratio', 'registration'):
        assert key in m

    # The stored report survives a raw JSON round-trip too (plain str/None,
    # no numpy artifacts leaked into the payload).
    stored = manager.get(job_id).data['report']['metrics']
    revived = json.loads(json.dumps(stored))
    assert revived['opencl_scatter'] == m['opencl_scatter']
    assert revived['opencl_device'] == m['opencl_device']


def test_result_metrics_name_the_fusion():
    """The result panel labels a MAP run by ``metrics.fusion`` (its
    opencl_scatter reads 'unavailable' because MAP never uses the OpenCL
    drizzle scatter), so the payload must carry fusion + fusion_reason."""
    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    m = client.get(f'/api/jobs/{job_id}/result').json()['report']['metrics']
    assert isinstance(m.get('fusion'), str) and m['fusion']
    assert isinstance(m.get('fusion_reason'), str) and m['fusion_reason']


def test_result_metrics_include_opencl_reason():
    """Every job records why its fusion engine ran on the GPU or the CPU as
    ``opencl_reason`` in the result metrics (additive; string, JSON-safe)."""
    import json

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']
    assert 'opencl_reason' in m
    assert isinstance(m['opencl_reason'], str) and m['opencl_reason']
    stored = manager.get(job_id).data['report']['metrics']
    assert stored['opencl_reason'] == m['opencl_reason']
    # Plain str survives a raw JSON round-trip (no numpy artifacts).
    assert json.loads(json.dumps(stored))['opencl_reason'] == m['opencl_reason']


def test_result_metrics_include_ghost_suppression():
    """Every job records the ghost-suppression verdict as additive
    ``robust_active`` (bool) + ``robust_reason`` (non-empty str) in the result
    metrics, JSON-safe, so the panel can render one 'Ghost suppression:' line."""
    import json

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']
    assert isinstance(m['robust_active'], bool)
    assert isinstance(m['robust_reason'], str) and m['robust_reason']
    stored = manager.get(job_id).data['report']['metrics']
    assert stored['robust_active'] is m['robust_active']
    assert stored['robust_reason'] == m['robust_reason']
    # Plain bool/str survive a raw JSON round-trip (no numpy artifacts).
    revived = json.loads(json.dumps(stored))
    assert revived['robust_active'] == m['robust_active']
    assert revived['robust_reason'] == m['robust_reason']


def test_result_metrics_include_refinement_and_deposit():
    """Every job records the IBP configuration and drizzle deposit it ran as
    additive metric keys (docs/IBP_AND_HELDOUT.md), JSON-safe, so the panel can
    render the refinement/deposit lines without knowing the core defaults."""
    import json

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']
    assert m['ibp_iterations'] == 1  # shipped default (3 -> 1 in fix/heldout2)
    assert isinstance(m['ibp_summary'], str) and 'sweep' in m['ibp_summary']
    assert isinstance(m['drizzle_subsamples'], int) and m['drizzle_subsamples'] >= 1
    assert isinstance(m['drizzle_pixfrac'], float) and m['drizzle_pixfrac'] > 0
    assert isinstance(m['reference_index'], int)
    assert 0 <= m['reference_index'] < body['report']['total_frames']
    stored = manager.get(job_id).data['report']['metrics']
    revived = json.loads(json.dumps(stored))
    for key in ('ibp_iterations', 'ibp_summary', 'drizzle_subsamples',
                'drizzle_pixfrac', 'reference_index'):
        assert revived[key] == stored[key]


def test_result_metrics_backfill_missing_refinement_fields(tmp_path, monkeypatch):
    """A core that predates the refinement/deposit keys still yields honest,
    JSON-safe fallbacks ('' / None) instead of crashing or inventing values."""
    _mock_process_core(monkeypatch)
    import desktop.backend.server as server

    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((4, 4, 3), 0.5, dtype=np.float32))
    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a), str(b)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    m = manager.get(job_id).data['report']['metrics']
    assert m['ibp_iterations'] is None
    assert m['ibp_summary'] == ''
    assert m['drizzle_pixfrac'] is None
    assert m['drizzle_subsamples'] is None
    assert m['reference_index'] is None


def test_result_metrics_include_fusion_noise_and_weight_spread():
    """The real core reports the per-frame noise sigma it weighted by and how
    far the normalized weights spread (1.0 = uniform) as additive, JSON-safe
    keys, so the panel can show them in its advanced section."""
    import json

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']
    sigmas = m['frame_noise_sigma']
    assert isinstance(sigmas, list) and sigmas
    assert all(isinstance(s, float) and s >= 0.0 for s in sigmas)
    assert len(sigmas) == body['report']['used_frames']
    assert isinstance(m['fusion_weight_spread'], float)
    assert m['fusion_weight_spread'] >= 1.0 - 1e-9
    stored = manager.get(job_id).data['report']['metrics']
    revived = json.loads(json.dumps(stored))
    assert revived['frame_noise_sigma'] == sigmas
    assert revived['fusion_weight_spread'] == m['fusion_weight_spread']


def test_result_metrics_include_single_frame_detail():
    """The honest resolution figure is measured against ONE frame upscaled, not
    against the denoised stack the core's ``hf_gain_db`` uses. The backend adds
    it with the ``scripts/sr_acceptance.py`` convention, recomputed here from
    the stored HR result and reference frame so the contract is pinned."""
    import json

    import cv2
    from raw_sr.core.superres import high_frequency_gain_db, to_view_float

    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    body = client.get(f'/api/jobs/{job_id}/result').json()
    m = body['report']['metrics']
    job = manager.get(job_id)
    hr = job.data['hr']
    naive = cv2.resize(job.data['before_rgb'], (hr.shape[1], hr.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    expected = high_frequency_gain_db(to_view_float(hr), to_view_float(naive))
    assert isinstance(m['hf_gain_db_vs_single'], float)
    assert m['hf_gain_db_vs_single'] == pytest.approx(expected, abs=1e-9)
    assert isinstance(m['sharpness_gain_vs_single'], float)
    assert m['sharpness_gain_vs_single'] > 0.0
    stored = job.data['report']['metrics']
    revived = json.loads(json.dumps(stored))
    assert revived['hf_gain_db_vs_single'] == m['hf_gain_db_vs_single']
    assert revived['sharpness_gain_vs_single'] == m['sharpness_gain_vs_single']


def test_missing_fusion_metrics_backfill_to_none(tmp_path, monkeypatch):
    """A core predating frame_noise_sigma / fusion_weight_spread yields honest
    ``None`` values (the panel then omits the line) rather than a guess."""
    _mock_process_core(monkeypatch)
    import desktop.backend.server as server

    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((4, 4, 3), 0.5, dtype=np.float32))
    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a), str(b)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    m = manager.get(job_id).data['report']['metrics']
    assert m['frame_noise_sigma'] is None
    assert m['fusion_weight_spread'] is None
    assert manager.get(job_id).data['report']['memory_warning'] is None


# --------------------------------------------------------------------------- #
# Memory guard: refuse an impossible request cleanly, keep a near-limit run
# non-fatal (runtime warning -> report note), and never show a traceback.
# --------------------------------------------------------------------------- #

def test_process_memory_guard_refuses_oversize_with_413(tmp_path, monkeypatch):
    """A ratio/ROI whose HR output cannot fit the budget is refused BEFORE any
    frame is decoded, with an actionable 413 (not a crashed job)."""
    import desktop.backend.server as server
    from raw_sr.core import superres as core_sr

    loaded = []

    def fake_load(path, roi=None, demosaic_algorithm=None):
        loaded.append(path)
        return core_sr.RawFrame(name='f', rgb=np.full((4, 4, 3), 0.5, dtype=np.float32),
                                green=np.zeros((4, 4), dtype=np.float32))

    monkeypatch.setattr(core_sr, 'load_raw_linear', fake_load)
    # 1024x1024 LR ROI at 4x = 4096x4096 HR = 16.7 Mpx, > a 64 MB budget.
    monkeypatch.setattr(server, 'resolve_roi',
                        lambda path, target, ratio=1.0, **kw: ((0, 0, 1024, 1024), False))
    a = tmp_path / 'a.dng'
    a.write_bytes(b'x')
    r = client.post('/api/process',
                    json={'paths': [str(a)], 'sr_ratio': 4.0,
                          'memory_budget_bytes': 64 * 1024 * 1024})
    assert r.status_code == 413, r.text
    detail = r.json()['detail']
    assert 'Not enough memory' in detail
    assert 'Reduce the SR ratio or the ROI' in detail
    assert 'budget 0.1 GiB' in detail
    assert loaded == []  # refused before decoding anything


def test_process_memory_guard_passes_a_fitting_request(tmp_path, monkeypatch):
    """The guard does not refuse a run that fits its budget."""
    _mock_process_core(monkeypatch)
    import desktop.backend.server as server

    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((4, 4, 3), 0.5, dtype=np.float32))
    monkeypatch.setattr(server, 'resolve_roi',
                        lambda path, target, ratio=1.0, **kw: ((0, 0, 512, 512), False))
    a = tmp_path / 'a.dng'
    a.write_bytes(b'x')
    r = client.post('/api/process',
                    json={'paths': [str(a)], 'sr_ratio': 2.0,
                          'memory_budget_bytes': 4 * 1024 ** 3})
    assert r.status_code == 200, r.text
    assert _wait(r.json()['job_id'])['state'] == 'done'


def test_job_memory_error_is_actionable_and_kind_is_set():
    """A MemoryError from the core surfaces as a readable message with
    error_kind='memory' — never a bare traceback."""
    def target(job):
        raise MemoryError('HR output 8192x8192 at ratio 4 — reduce the ratio')

    job = manager.create(target)
    status = _wait(job.id)
    assert status['state'] == 'error'
    assert status['error_kind'] == 'memory'
    assert status['message'] == 'Not enough memory.'
    assert 'Not enough memory' in status['error']
    assert '8192x8192' in status['error']
    # A generic error keeps error_kind None.
    other = manager.create(lambda j: (_ for _ in ()).throw(RuntimeError('boom')))
    status = _wait(other.id)
    assert status['error_kind'] is None
    assert 'boom' in status['error']


def test_process_runtime_warning_becomes_a_nonfatal_note(tmp_path, monkeypatch):
    """A near-limit run (RuntimeWarning, no raise) still completes and carries
    a non-fatal ``memory_warning`` note in the report."""
    import warnings

    import desktop.backend.server as server
    from raw_sr.core import superres as core_sr

    def fake_load(path, roi=None, demosaic_algorithm=None):
        return core_sr.RawFrame(name='f', rgb=np.full((4, 4, 3), 0.5, dtype=np.float32),
                                green=np.zeros((4, 4), dtype=np.float32))

    class _WarnThenFinish:
        def run(self, frames):
            warnings.warn('HR output close to the memory limit.', RuntimeWarning,
                          stacklevel=2)
            return _FakeSRResult()

    monkeypatch.setattr(core_sr, 'load_raw_linear', fake_load)
    monkeypatch.setattr(server, 'resolve_roi',
                        lambda path, target, ratio=1.0, **kw: ((0, 0, 4, 4), False))
    monkeypatch.setattr(server, '_make_superres', lambda **kw: _WarnThenFinish())
    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((4, 4, 3), 0.5, dtype=np.float32))
    a = tmp_path / 'a.dng'
    a.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    body = client.get(f'/api/jobs/{job_id}/result').json()
    assert body['report']['memory_warning'] == 'HR output close to the memory limit.'


def test_alignment_per_frame_covers_all_frames():
    """per_frame has one entry per processed frame, in order, with used flags."""
    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    job = manager.get(job_id)
    al = job.data['alignment']
    report = job.data['report']
    n = report['total_frames']
    assert len(al['per_frame']) == n == 5
    # Synthetic path has no session id-map, so ids fall back to path indices.
    assert [f['id'] for f in al['per_frame']] == [str(i) for i in range(n)]
    used_flags = [f['used'] for f in al['per_frame']]
    assert sum(used_flags) == report['used_frames'] >= 2
    # The core picks the reference frame (the improved core selects the sharpest
    # frame near the burst centroid, not necessarily frame 0) and always keeps
    # it, perfectly aligned. Older cores used frame 0.
    ref_idx = report['metrics'].get('reference_index', 0)
    assert 0 <= ref_idx < n
    ref_entry = al['per_frame'][ref_idx]
    assert ref_entry['used'] is True
    assert ref_entry['zncc'] == pytest.approx(1.0)
    assert ref_entry['residual'] == pytest.approx(0.0)
    # Filed, dropped frames carry zeroed zncc/residual/shift.
    for f in al['per_frame']:
        if f['used']:
            assert 0.0 <= f['zncc'] <= 1.0
            assert f['residual'] >= 0.0
        else:
            assert f['zncc'] == 0.0
            assert f['residual'] == 0.0
            assert f['shift'] == [0.0, 0.0]


def test_alignment_verdict_monotonic_and_bounded():
    from desktop.backend.server import _alignment_verdict

    # score is always within [0, 1].
    for z, r in [(0.0, 0.0), (1.0, 0.0), (0.5, 2.0), (0.99, 0.1), (0.0, 50.0)]:
        score, level, msg = _alignment_verdict(z, r)
        assert 0.0 <= score <= 1.0
        assert level in ('ok', 'warn', 'bad')
        assert isinstance(msg, str) and msg

    # Lower residual at fixed ZNCC -> strictly higher (monotonic in residual).
    assert _alignment_verdict(0.8, 0.0)[0] > _alignment_verdict(0.8, 1.0)[0]
    assert _alignment_verdict(0.8, 1.0)[0] > _alignment_verdict(0.8, 5.0)[0]
    # Higher ZNCC at fixed residual -> strictly higher (monotonic in ZNCC).
    assert _alignment_verdict(0.9, 0.5)[0] > _alignment_verdict(0.5, 0.5)[0]
    # Documented level thresholds.
    assert _alignment_verdict(0.99, 0.0)[1] == 'ok'
    assert _alignment_verdict(0.85, 100.0)[1] == 'warn'
    assert _alignment_verdict(0.0, 100.0)[1] == 'bad'


def test_alignment_map_is_single_png_and_quality_bounded():
    job_id, _ = _done_synthetic(ratio=2.0, n_frames=5)
    job = manager.get(job_id)
    quality = job.data['alignment']['map']
    assert quality.startswith('data:image/png;base64,')
    assert quality == job.data['alignment']['map']  # one map, computed once
    # The map is also served through the endpoint exactly as stored.
    body = client.get(f'/api/jobs/{job_id}/result').json()
    assert body['alignment']['map'] == quality
    assert 'confidence' not in body


def test_render_applies_view_transform_and_persists():
    job_id, _ = _done_synthetic()
    r = client.post(f'/api/jobs/{job_id}/render',
                    json={'gamma': 1.8, 'brightness': 0.1, 'contrast': 1.2})
    assert r.status_code == 200
    body = r.json()
    # /render must return BOTH images under the same view transform so the
    # before/after comparison stays consistent when a slider moves.
    assert set(body) == {'fused', 'before', 'width', 'height'}
    assert body['fused'].startswith('data:image/png;base64,')
    assert body['before'].startswith('data:image/png;base64,')
    assert body['width'] > 0 and body['height'] > 0
    # The transform is remembered for subsequent result fetches.
    res = client.get(f'/api/jobs/{job_id}/result').json()
    assert res['view_transform']['gamma'] == pytest.approx(1.8)
    assert res['view_transform']['brightness'] == pytest.approx(0.1)
    # The rendered before matches the result's before (same view transform).
    assert res['images']['before'] == body['before']
    assert res['images']['fused'] == body['fused']


# --------------------------------------------------------------------------- #
# `before` honesty: it must come from the preview pipeline, not DCB
# --------------------------------------------------------------------------- #

class _FakeSRResult:
    """Minimal SRResult stand-in for the mocked process path."""

    def __init__(self, hr_shape=(8, 8, 3)):
        self.hr = np.full(hr_shape, 0.5, dtype=np.float32)
        self.ratio = 2.0
        self.shifts = [(0.0, 0.0)]
        self.weights = np.array([1.0])
        self.coverage = np.ones(hr_shape[:2], dtype=np.float32)
        self.used_frames = 1
        self.total_frames = 1
        self.metrics = {}
        self.transforms = []
        self.kept_indices = [0]
        self.align_scores = [1.0]
        self.align_residuals = [0.0]


def _mock_process_core(monkeypatch):
    """Make POST /api/process run without touching real RAW files or the core."""
    from raw_sr.core import superres as core_sr
    import desktop.backend.server as server

    def fake_load(path, roi=None, demosaic_algorithm=None):
        return core_sr.RawFrame(name='f', rgb=np.full((4, 4, 3), 0.5, dtype=np.float32),
                                green=np.zeros((4, 4), dtype=np.float32))

    monkeypatch.setattr(core_sr, 'load_raw_linear', fake_load)
    monkeypatch.setattr(server, 'resolve_roi',
                        lambda path, target, ratio=1.0, **kw: ((0, 0, 4, 4), False))
    monkeypatch.setattr(server, '_make_superres',
                        lambda **kw: type('S', (), {'run': lambda self, frames: _FakeSRResult()})())


def test_reference_linear_uses_preview_pipeline(monkeypatch):
    """load_reference_linear reuses the preview decode (full res, default demosaic)."""
    calls = {}

    def fake_load(path, half_size=True):
        calls['path'] = path
        calls['half_size'] = half_size
        return np.full((100, 120, 3), 0.25, dtype=np.float32), {'m': 1}, (100, 120)

    monkeypatch.setattr(session, 'load_preview_linear', fake_load)

    out = session.load_reference_linear('x.dng', roi=(10, 20, 30, 40))
    assert calls['half_size'] is False           # full sensor resolution
    assert out.shape == (40, 30, 3)
    assert np.allclose(out, 0.25)
    assert out.flags['C_CONTIGUOUS']

    # A ROI past the frame edge is clamped rather than raising.
    clamped = session.load_reference_linear('x.dng', roi=(110, 90, 50, 50))
    assert clamped.shape[0] <= 10 and clamped.shape[1] <= 10


def test_process_before_uses_preview_reference(tmp_path, monkeypatch):
    """The job's before_rgb comes from session.load_reference_linear, not DCB."""
    _mock_process_core(monkeypatch)
    import desktop.backend.server as server

    marker = np.full((4, 4, 3), 0.123, dtype=np.float32)
    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: marker.copy())

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a), str(b)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    job = manager.get(job_id)
    assert np.allclose(job.data['before_rgb'], 0.123)
    # Display size stays consistent with the fused preview.
    assert job.data['before_disp'].shape[:2] == job.data['disp_linear'].shape[:2]


def test_process_before_uses_the_reference_frame(tmp_path, monkeypatch):
    """The before image must be the frame the HR grid is aligned to.

    Using frames[0] left the before/after panels offset by that frame's
    registration shift whenever the auto-selected reference was not index 0
    (normally it is the sharpest central frame).
    """
    _mock_process_core(monkeypatch)
    import desktop.backend.server as server

    seen = {}

    def fake_reference(path, roi=None):
        seen['path'] = path
        return np.full((4, 4, 3), 0.2, dtype=np.float32)

    monkeypatch.setattr(server.session, 'load_reference_linear',
                        fake_reference)

    class RefResult(_FakeSRResult):
        reference_index = 1

    monkeypatch.setattr(
        server, '_make_superres',
        lambda **kw: type('S', (), {'run': lambda self, frames: RefResult()})())

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    r = client.post('/api/process',
                    json={'paths': [str(a), str(b)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    assert seen['path'] == str(b)
    assert np.allclose(manager.get(job_id).data['before_rgb'], 0.2)


def test_alignment_ids_match_frame_ids(tmp_path, monkeypatch):
    """per_frame[].id joins to the FilePanel frame ids (FrameInfo.id)."""
    session.clear_frame_ids()
    _mock_process_core(monkeypatch)
    monkeypatch.setattr(session, 'load_scaled_linear',
                        lambda path, max_dim=1024: (np.full((20, 20, 3), 0.5, dtype=np.float32),
                                                    {}, (20, 20)))

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    a.write_bytes(b'x')
    b.write_bytes(b'x')
    files = client.post('/api/session/files', json={'paths': [str(a), str(b)]}).json()
    frame_ids = [f['id'] for f in files['frames']]
    assert frame_ids == ['0', '1']

    r = client.post('/api/process', json={'paths': [str(a), str(b)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    job = manager.get(job_id)
    assert [f['id'] for f in job.data['alignment']['per_frame']] == frame_ids


def test_alignment_ids_after_exclude_still_join(tmp_path, monkeypatch):
    """Excluding a frame keeps the remaining per-frame ids matching the panel."""
    session.clear_frame_ids()
    _mock_process_core(monkeypatch)
    monkeypatch.setattr(session, 'load_scaled_linear',
                        lambda path, max_dim=1024: (np.full((20, 20, 3), 0.5, dtype=np.float32),
                                                    {}, (20, 20)))

    a = tmp_path / 'a.dng'
    b = tmp_path / 'b.dng'
    c = tmp_path / 'c.dng'
    for p in (a, b, c):
        p.write_bytes(b'x')
    files = client.post('/api/session/files',
                        json={'paths': [str(a), str(b), str(c)]}).json()
    frame_ids = [f['id'] for f in files['frames']]

    # Exclude the middle frame by its session id.
    r = client.post('/api/process', json={'paths': [str(a), str(b), str(c)],
                                          'sr_ratio': 2.0, 'exclude': [frame_ids[1]]})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'
    job = manager.get(job_id)
    got = [f['id'] for f in job.data['alignment']['per_frame']]
    assert got == [frame_ids[0], frame_ids[2]]


def test_export_master_and_baked(tmp_path):
    job_id, _ = _done_synthetic()
    master = tmp_path / 'master.tiff'
    baked = tmp_path / 'baked.tiff'

    r = client.post('/api/export', json={'job_id': job_id, 'kind': 'master',
                                         'output_path': str(master)})
    assert r.status_code == 200 and master.exists() and master.stat().st_size > 0
    assert r.json()['output_path'] == str(master)

    r = client.post('/api/export', json={'job_id': job_id, 'kind': 'baked',
                                         'output_path': str(baked),
                                         'view_transform': {'gamma': 2.2, 'brightness': 0.0,
                                                            'contrast': 1.0}})
    assert r.status_code == 200 and baked.exists() and baked.stat().st_size > 0


def test_export_unknown_kind_400(tmp_path):
    job_id, _ = _done_synthetic()
    r = client.post('/api/export', json={'job_id': job_id, 'kind': 'bogus',
                                         'output_path': str(tmp_path / 'x.tiff')})
    assert r.status_code == 400
    assert 'kind' in r.json()['detail'].lower()


def test_export_unknown_job_404(tmp_path):
    r = client.post('/api/export', json={'job_id': 'nope', 'kind': 'master',
                                         'output_path': str(tmp_path / 'x.tiff')})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# Unknown jobs & error paths
# --------------------------------------------------------------------------- #

def test_unknown_job_404_everywhere():
    assert client.get('/api/jobs/nope').status_code == 404
    assert client.post('/api/jobs/nope/cancel').status_code == 404
    assert client.get('/api/jobs/nope/result').status_code == 404
    assert client.post('/api/jobs/nope/render', json={}).status_code == 404


def test_result_and_render_409_before_done():
    release = threading.Event()

    def target(job):
        release.wait(5.0)

    job = manager.create(target)
    try:
        assert client.get(f'/api/jobs/{job.id}/result').status_code == 409
        assert client.post(f'/api/jobs/{job.id}/render', json={}).status_code == 409
    finally:
        release.set()
    _wait(job.id)


def test_cancel_running_job():
    started = threading.Event()

    def target(job):
        started.set()
        while not job.cancel_requested:
            time.sleep(0.005)

    job = manager.create(target)
    assert started.wait(5.0)
    r = client.post(f'/api/jobs/{job.id}/cancel')
    assert r.status_code == 200 and r.json() == {'cancelled': True}
    status = _wait(job.id)
    assert status['state'] == 'cancelled'
    assert manager.get(job.id).state == 'cancelled'


def test_job_error_path_reports_message():
    def target(job):
        raise RuntimeError('boom')

    job = manager.create(target)
    status = _wait(job.id)
    assert status['state'] == 'error'
    assert status['error'] is not None and 'boom' in status['error']


# --------------------------------------------------------------------------- #
# Memory: the decoded burst is released once the core returns (fix: free
# frames after run). The fake loader mirrors load_raw_linear's ROI-view
# behaviour: frame.rgb is a VIEW into a big decoded base (the F-pin the real
# core keeps until the ROI detach lands), so ANY lingering reference to a
# frame — or a before_rgb aliasing one — keeps that base alive forever.
# --------------------------------------------------------------------------- #

def test_process_releases_frames_after_run(tmp_path, monkeypatch):
    """No decoded frame base stays pinned after a process job finishes."""
    import gc
    import weakref
    from raw_sr.core import superres as core_sr
    import desktop.backend.server as server

    bases: list = []  # weakrefs ONLY — a strong ref here would defeat the test

    def fake_load(path, roi=None, demosaic_algorithm=None):
        base = np.full((64, 64, 3), 0.5, dtype=np.float32)  # the decoded frame
        bases.append(weakref.ref(base))
        rgb = base[8:16, 8:16]  # ROI view: shares memory with the base
        return core_sr.RawFrame(name='f', rgb=rgb,
                                green=np.zeros((8, 8), dtype=np.float32))

    _mock_process_core(monkeypatch)
    monkeypatch.setattr(core_sr, 'load_raw_linear', fake_load)
    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((8, 8, 3), 0.4, dtype=np.float32))

    paths = []
    for i in range(3):
        p = tmp_path / f'{i}.dng'
        p.write_bytes(b'x')
        paths.append(str(p))
    r = client.post('/api/process', json={'paths': paths, 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    assert _wait(r.json()['job_id'])['state'] == 'done'
    assert len(bases) == 3

    gc.collect()
    pinned = [w for w in bases if w() is not None]
    assert pinned == [], f'{len(pinned)}/{len(bases)} decoded bases still pinned'


def test_process_fallback_before_rgb_is_a_copy(tmp_path, monkeypatch):
    """If the reference re-decode fails, before_rgb must be a COPY, not a view.

    The old code stored `frames[0].rgb` directly: a base-pinned view that kept
    ~one full frame alive inside job.data for the job's whole lifetime.
    """
    from raw_sr.core import superres as core_sr
    import desktop.backend.server as server

    holder: dict = {}

    def fake_load(path, roi=None, demosaic_algorithm=None):
        base = np.full((64, 64, 3), 0.7, dtype=np.float32)
        holder['base'] = base
        return core_sr.RawFrame(name='f', rgb=base[8:16, 8:16],
                                green=np.zeros((8, 8), dtype=np.float32))

    def broken_reference(path, roi=None):
        raise RuntimeError('re-decode failed')

    _mock_process_core(monkeypatch)
    monkeypatch.setattr(core_sr, 'load_raw_linear', fake_load)
    monkeypatch.setattr(server.session, 'load_reference_linear', broken_reference)

    a = tmp_path / 'a.dng'
    a.write_bytes(b'x')
    r = client.post('/api/process', json={'paths': [str(a)], 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    assert _wait(job_id)['state'] == 'done'

    before = manager.get(job_id).data['before_rgb']
    assert before.base is None                          # standalone, not a view
    assert not np.shares_memory(before, holder['base'])  # pins nothing of the frame
    assert before.shape == (8, 8, 3) and np.allclose(before, 0.7)


# --------------------------------------------------------------------------- #
# Single-flight /api/process: 409 while a process job is in flight
# --------------------------------------------------------------------------- #

def _process_body(tmp_path, n=2):
    paths = []
    for i in range(n):
        p = tmp_path / f'{i}.dng'
        p.write_bytes(b'x')
        paths.append(str(p))
    return {'paths': paths, 'sr_ratio': 2.0}


def test_process_second_request_409_while_one_runs(tmp_path, monkeypatch):
    """POST /api/process is single-flight: overlapping request -> 409."""
    import desktop.backend.server as server

    _mock_process_core(monkeypatch)
    monkeypatch.setattr(server.session, 'load_reference_linear',
                        lambda path, roi=None: np.full((4, 4, 3), 0.5, dtype=np.float32))

    entered = threading.Event()
    release = threading.Event()

    class BlockingSR:
        def run(self, frames):
            entered.set()
            assert release.wait(15.0), 'test deadlock: release never set'
            return _FakeSRResult()

    monkeypatch.setattr(server, '_make_superres', lambda **kw: BlockingSR())
    body = _process_body(tmp_path)

    r1 = client.post('/api/process', json=body)
    assert r1.status_code == 200, r1.text
    job1 = r1.json()['job_id']
    try:
        assert entered.wait(10.0), 'first job never reached the core'
        r2 = client.post('/api/process', json=body)
        assert r2.status_code == 409
        detail = r2.json()['detail']
        assert isinstance(detail, str) and 'already running' in detail.lower()
        # The running job stays cancel-able while the guard is held.
        assert client.post(f'/api/jobs/{job1}/cancel').status_code == 200
    finally:
        release.set()
    assert _wait(job1)['state'] == 'cancelled'

    # A finished job releases the guard: a new request is accepted again.
    r3 = client.post('/api/process', json=body)
    assert r3.status_code == 200, r3.text
    assert _wait(r3.json()['job_id'])['state'] == 'done'


def test_process_lock_released_when_job_errors(tmp_path, monkeypatch):
    """An errored process job must not wedge the single-flight guard."""
    import desktop.backend.server as server

    _mock_process_core(monkeypatch)

    class BoomSR:
        def run(self, frames):
            raise RuntimeError('core exploded')

    monkeypatch.setattr(server, '_make_superres', lambda **kw: BoomSR())
    body = _process_body(tmp_path)

    r1 = client.post('/api/process', json=body)
    assert r1.status_code == 200, r1.text
    assert _wait(r1.json()['job_id'])['state'] == 'error'

    r2 = client.post('/api/process', json=body)
    assert r2.status_code == 200, f'lock leaked after error: {r2.text}'
    _wait(r2.json()['job_id'])  # terminal (still BoomSR), just drain it


def test_process_lock_released_when_cancelled_before_start(monkeypatch):
    """Cancel racing thread startup still runs on_finish — no lock leak.

    Exercises the JobManager hook directly with the server's real lock: the
    target never executes, yet the guard must come back.
    """
    import desktop.backend.server as server

    assert server._PROCESS_LOCK.acquire(blocking=False)
    job = manager.create(lambda j: None, on_finish=server._PROCESS_LOCK.release)
    job.request_cancel()  # cancel before the thread's _run starts its target
    _wait(job.id)
    assert server._PROCESS_LOCK.acquire(blocking=False), 'single-flight lock leaked'
    server._PROCESS_LOCK.release()


# --------------------------------------------------------------------------- #
# Heavy-result eviction: only the K newest completed jobs keep hr/coverage/
# before_rgb; older jobs still render, only export reports a graceful error.
# --------------------------------------------------------------------------- #

def test_result_eviction_keeps_only_latest_heavy_result(tmp_path):
    job1, _ = _done_synthetic()
    assert 'hr' in manager.get(job1).data
    assert manager.held_heavy_ids() == [job1]

    job2, _ = _done_synthetic()
    j1, j2 = manager.get(job1), manager.get(job2)
    for key in ('hr', 'coverage', 'before_rgb'):
        assert key not in j1.data, f'{key} should have been evicted'
        assert key in j2.data
    assert manager.held_heavy_ids() == [job2]

    # The evicted job still renders a full result (report + display images +
    # alignment) — exactly what GET /api/jobs/{id}/result serves.
    r = client.get(f'/api/jobs/{job1}/result')
    assert r.status_code == 200
    body = r.json()
    assert body['report']['total_frames'] >= 2
    assert body['images']['fused'].startswith('data:image/png;base64,')
    assert body['images']['before'].startswith('data:image/png;base64,')
    assert body['alignment']['map'].startswith('data:image/png;base64,')
    assert client.post(f'/api/jobs/{job1}/render', json={}).status_code == 200

    # Re-requesting the heavy EXPORT path on an evicted job: graceful 409.
    evicted_out = tmp_path / 'evicted.tiff'
    r = client.post('/api/export', json={'job_id': job1, 'kind': 'master',
                                         'output_path': str(evicted_out)})
    assert r.status_code == 409
    assert 'evict' in r.json()['detail'].lower()
    assert not evicted_out.exists()

    # The retained (most recent) job still exports normally.
    fresh_out = tmp_path / 'fresh.tiff'
    r = client.post('/api/export', json={'job_id': job2, 'kind': 'master',
                                         'output_path': str(fresh_out)})
    assert r.status_code == 200 and fresh_out.stat().st_size > 0


def _mgr_wait(mgr, job, timeout: float = 10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if job.is_terminal:
            return
        time.sleep(0.01)
    raise TimeoutError(f'job {job.id} did not finish')


def _heavy_data(job, tag):
    job.data.update({
        'hr': np.full((4, 4, 3), tag, dtype=np.float32),
        'coverage': np.ones((4, 4), dtype=np.float32),
        'before_rgb': np.zeros((4, 4, 3), dtype=np.float32),
        'disp_linear': np.zeros((2, 2, 3), dtype=np.float32),
        'report': {'tag': tag},
    })


def test_job_manager_heavy_lru_is_configurable():
    from desktop.backend.jobs import JobManager

    mgr = JobManager(keep_heavy_results=2)
    ids = []
    for i in range(4):
        job = mgr.create(lambda j, i=i: _heavy_data(j, i))
        _mgr_wait(mgr, job)
        assert job.state == 'done'
        ids.append(job.id)

    assert mgr.held_heavy_ids() == ids[2:]  # newest K retained
    for jid in ids[:2]:
        d = mgr.get(jid).data
        assert 'hr' not in d and 'coverage' not in d and 'before_rgb' not in d
        assert d['report']['tag'] in (0, 1)          # report kept
        assert 'disp_linear' in d                    # display arrays kept
    for jid in ids[2:]:
        assert 'hr' in mgr.get(jid).data


def test_heavy_lru_ignores_jobs_without_results():
    """Plain completed jobs (no result data) never evict a real result."""
    from desktop.backend.jobs import JobManager

    mgr = JobManager(keep_heavy_results=1)
    heavy = mgr.create(lambda j: _heavy_data(j, 7))
    _mgr_wait(mgr, heavy)
    plain = mgr.create(lambda j: None)
    _mgr_wait(mgr, plain)

    assert mgr.held_heavy_ids() == [heavy.id]
    assert 'hr' in mgr.get(heavy.id).data


def test_cancelled_and_errored_jobs_drop_heavy_arrays():
    from desktop.backend.jobs import JobManager

    mgr = JobManager(keep_heavy_results=4)

    def cancel_after_store(job):
        _heavy_data(job, 1)
        job.request_cancel()

    cancelled = mgr.create(cancel_after_store)
    _mgr_wait(mgr, cancelled)
    assert cancelled.state == 'cancelled'
    assert 'hr' not in cancelled.data
    assert mgr.held_heavy_ids() == []

    def error_after_store(job):
        _heavy_data(job, 2)
        raise RuntimeError('boom after store')

    errored = mgr.create(error_after_store)
    _mgr_wait(mgr, errored)
    assert errored.state == 'error'
    assert 'hr' not in errored.data
    assert mgr.held_heavy_ids() == []


def test_on_finish_hook_runs_once_even_when_target_raises():
    from desktop.backend.jobs import JobManager

    calls = []
    mgr = JobManager()
    job = mgr.create(_boom_target, on_finish=lambda: calls.append(1))
    _mgr_wait(mgr, job)
    assert job.state == 'error'
    assert calls == [1]


def _boom_target(job):
    raise RuntimeError('hook test')


# --------------------------------------------------------------------------- #
# API caps: MAX_FRAMES per paths endpoint, preview max_dim bound
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('endpoint', ['/api/session/files', '/api/analyze',
                                      '/api/quality', '/api/process'])
def test_frame_cap_rejected_per_endpoint(endpoint):
    """Every `paths` endpoint rejects > MAX_FRAMES with 400 (one helper)."""
    from desktop.backend.server import MAX_FRAMES

    too_many = [f'/tmp/raw_sr_cap_{i:03d}.dng' for i in range(MAX_FRAMES + 1)]
    assert len(too_many) == 25 and MAX_FRAMES == 24
    r = client.post(endpoint, json={'paths': too_many, 'sr_ratio': 2.0})
    assert r.status_code == 400
    detail = r.json()['detail']
    assert 'Too many frames' in detail and str(MAX_FRAMES) in detail


def test_frame_cap_boundary_allows_max():
    """Exactly MAX_FRAMES paths pass validation (product hint is 2-20)."""
    from desktop.backend.server import MAX_FRAMES

    paths = [f'/tmp/raw_sr_cap_ok_{i:03d}.dng' for i in range(MAX_FRAMES)]
    r = client.post('/api/quality', json={'paths': paths})
    assert r.status_code == 200            # quality tolerates missing files
    assert len(r.json()['flags']) == MAX_FRAMES


def test_preview_max_dim_capped_at_2048():
    """max_dim > 2048 is a validation 400; the bound keeps cache entries small."""
    missing = '/tmp/raw_sr_nope.dng'
    r = client.post('/api/session/preview', json={'path': missing, 'max_dim': 4096})
    assert r.status_code == 400
    assert 'max_dim' in r.json()['detail']

    # 2048 still validates — the 404 proves it passed model validation.
    r = client.post('/api/session/preview', json={'path': missing, 'max_dim': 2048})
    assert r.status_code == 404
    r = client.post('/api/session/preview', json={'path': missing, 'max_dim': 8192})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# GPU device layer: selected-device capabilities + NVIDIA OpenCL pin policy
# --------------------------------------------------------------------------- #

def test_capabilities_report_selected_device():
    """opencl_device/opencl_vendor are additive keys: str|null naming the
    device OpenCV resolved (null-safe on machines/bindings without one)."""
    r = client.get('/api/capabilities')
    body = r.json()
    assert {'opencl', 'cpu_count', 'opencl_device', 'opencl_vendor',
            'engine_status'} == set(body)
    for key in ('opencl_device', 'opencl_vendor'):
        assert body[key] is None or (isinstance(body[key], str) and body[key])


def test_capabilities_device_matches_cv2_default():
    """The reported device is exactly cv2's resolved default device — i.e.
    the one routed warps execute on, under the startup pin."""
    import cv2

    dev_cls = getattr(cv2.ocl, 'Device', None)
    if dev_cls is None or not cv2.ocl.haveOpenCL():
        pytest.skip('no cv2.ocl.Device introspection / no OpenCL here')
    dev = dev_cls.getDefault()
    name = getattr(dev, 'name')
    name = name() if callable(name) else name
    body = client.get('/api/capabilities').json()
    assert body['opencl_device'] == (name or None)


def test_capabilities_engine_status_shape():
    """engine_status names the interpreter, pyopencl/OpenCL probes and the
    auto gate verdict/reason (additive; see desktop/ARCHITECTURE.md)."""
    body = client.get('/api/capabilities').json()
    assert 'engine_status' in body
    st = body['engine_status']
    assert set(st) == {
        'python_executable', 'pyopencl', 'pyopencl_error', 'opencl_platforms',
        'gpu_device', 'nvidia_smi', 'pinned_device', 'gate_decision', 'gate_reason',
    }
    assert isinstance(st['python_executable'], str) and st['python_executable']
    assert isinstance(st['pyopencl'], bool)
    assert st['pyopencl_error'] is None or isinstance(st['pyopencl_error'], str)
    assert isinstance(st['opencl_platforms'], list)
    assert all(isinstance(p, str) for p in st['opencl_platforms'])
    assert st['gpu_device'] is None or isinstance(st['gpu_device'], str)
    assert isinstance(st['nvidia_smi'], bool)
    assert st['pinned_device'] is None or isinstance(st['pinned_device'], str)
    assert st['gate_decision'] in ('gpu', 'cpu')
    assert isinstance(st['gate_reason'], str) and st['gate_reason']
    # A device-less CPU verdict names the cause rather than leaving it blank.
    if st['gate_decision'] == 'cpu' and not st['gpu_device']:
        assert st['gate_reason']


def test_opencl_gate_reasons_are_specific():
    """The gate helper always returns a decision plus a human reason, and a
    forced choice is reported as such (unit; no GPU required)."""
    from desktop.backend.server import _opencl_gate

    for use_opencl in (None, True, False):
        decision, reason = _opencl_gate(use_opencl)
        assert decision in ('gpu', 'cpu')
        assert isinstance(reason, str) and reason
    assert _opencl_gate(False) == ('cpu', 'forced off by request')
    decision, reason = _opencl_gate(True)
    assert decision in ('gpu', 'cpu')
    if decision == 'gpu':
        assert reason == 'forced by request'
    else:
        assert isinstance(reason, str) and reason


def test_opencl_pin_respects_user_override():
    """A user-set OPENCV_OPENCL_DEVICE always wins — the pin never touches it."""
    from desktop.backend.server import _pin_nvidia_opencl_device

    env = {'OPENCV_OPENCL_DEVICE': 'Some Vendor:GPU:1'}
    assert _pin_nvidia_opencl_device(env=env, has_nvidia=lambda: True) is None
    assert env == {'OPENCV_OPENCL_DEVICE': 'Some Vendor:GPU:1'}


def test_opencl_pin_sets_nvidia_when_present():
    """Detected NVIDIA runtime + unset env -> deterministic NVIDIA GPU pin."""
    from desktop.backend.server import NVIDIA_OPENCL_DEVICE, _pin_nvidia_opencl_device

    assert NVIDIA_OPENCL_DEVICE == 'NVIDIA CUDA:GPU:0'
    env: dict = {}
    assert _pin_nvidia_opencl_device(env=env, has_nvidia=lambda: True) \
        == NVIDIA_OPENCL_DEVICE
    assert env == {'OPENCV_OPENCL_DEVICE': NVIDIA_OPENCL_DEVICE}


def test_opencl_pin_noop_without_nvidia():
    """No NVIDIA runtime -> env untouched: brand-agnostic default preserved,
    and the never-set-invalid-pin path stays reachable."""
    from desktop.backend.server import _pin_nvidia_opencl_device

    env: dict = {}
    assert _pin_nvidia_opencl_device(env=env, has_nvidia=lambda: False) is None
    assert env == {}


# --------------------------------------------------------------------------- #
# run.sh launch summary (engine diagnostics, scripted env)
# --------------------------------------------------------------------------- #

def test_run_sh_emits_launch_summary():
    """`RAW_SR_PREFLIGHT_ONLY=1 bash run.sh` prints the engine summary (python,
    pyopencl, OpenCL device) and exits without launching Electron.

    Exercises the real script in a scripted environment; preflight-only stops
    before the dependency-install and launch steps, so it is fast and never
    triggers pip/npm.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    run_sh = repo / 'run.sh'
    if not run_sh.exists():
        pytest.skip('run.sh not found')
    env = {**os.environ, 'RAW_SR_PREFLIGHT_ONLY': '1', 'RAW_SR_SKIP_PREFLIGHT': '1'}
    proc = subprocess.run(['bash', str(run_sh)], cwd=str(repo), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert 'Launch summary' in out
    assert 'python:' in out
    assert 'pyopencl:' in out
    assert 'opencl:' in out
    # Preflight-only must not reach the launch step.
    assert 'Launching raw_sr desktop' not in out


# --------------------------------------------------------------------------- #
# Enhanced export of a MAP master (fusion='auto' -> map since WP12)
# --------------------------------------------------------------------------- #

def test_enhanced_export_of_map_master_is_identity_unless_forced(tmp_path):
    import tifffile
    job_id, _ = _done_synthetic()
    job = manager.get(job_id)
    job.data['report'].setdefault('metrics', {})['fusion'] = 'map'
    base = {'job_id': job_id}
    master = tmp_path / 'm.tiff'
    enh = tmp_path / 'e.tiff'
    forced = tmp_path / 'f.tiff'
    assert client.post('/api/export', json={**base, 'kind': 'master',
                                            'output_path': str(master)}).status_code == 200
    r = client.post('/api/export', json={**base, 'kind': 'enhanced', 'output_path': str(enh)})
    assert r.status_code == 200 and 'map master' in r.json()['enhance_note']
    assert np.array_equal(tifffile.imread(master), tifffile.imread(enh))
    r = client.post('/api/export', json={**base, 'kind': 'enhanced', 'output_path': str(forced),
                                         'enhance_strength': 0.65})
    assert r.status_code == 200 and 'enhance_note' not in r.json()
    assert not np.array_equal(tifffile.imread(master), tifffile.imread(forced))


def test_enhanced_export_default_master_runs(tmp_path):
    # Regression: the enhanced export read a non-existent job.result (HTTP 500).
    job_id, _ = _done_synthetic()
    manager.get(job_id).data['report'].setdefault('metrics', {})['fusion'] = 'demosaic_drizzle'
    r = client.post('/api/export', json={'job_id': job_id, 'kind': 'enhanced',
                                         'output_path': str(tmp_path / 'e.png')})
    assert r.status_code == 200, r.text
    assert 'enhance_note' not in r.json()
