#!/usr/bin/env python3
"""Verify the desktop backend against a real RAW burst (optional, needs DNGs).

Drives the HTTP surface end to end through the real SR core:
  session/files -> analyze -> process -> result -> render -> export.

Usage (from the repository root, using the venv):
    venv/bin/python desktop/scripts/verify_real_raw.py [TEST_DATA_DIR]

TEST_DATA_DIR defaults to <repo>/test_data. Exits 0 on PASS, 1 on FAIL.
Skips (exit 0) with a notice when fewer than 2 RAW files are found.
"""
from __future__ import annotations

import glob
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from desktop.backend.server import app  # noqa: E402

EXTENSIONS = ('dng', 'cr2', 'cr3', 'nef', 'arw', 'raf', 'rw2', 'orf', 'pef')


def main() -> int:
    test_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, 'test_data')
    files = []
    for ext in EXTENSIONS:
        files += glob.glob(os.path.join(test_dir, f'*.{ext}'))
        files += glob.glob(os.path.join(test_dir, f'*.{ext.upper()}'))
    files = sorted(set(files))

    if len(files) < 2:
        print(f'No RAW burst found in {test_dir} (found {len(files)}). Skipping.')
        return 0

    client = TestClient(app)
    print(f'real RAW files: {len(files)}')

    r = client.post('/api/session/files', json={'paths': files})
    assert r.status_code == 200, r.text
    body = r.json()
    frames = body['frames']
    assert len(frames) == len(files), body
    assert all(f['thumbnail'].startswith('data:image/png') for f in frames)
    print(f"[files] {len(frames)} frames, {len(body['rejected'])} rejected")

    # A modest centered ROI keeps the real-data run fast.
    side = min(512, frames[0]['width'], frames[0]['height'])
    roi = [(frames[0]['width'] - side) // 2, (frames[0]['height'] - side) // 2, side, side]

    r = client.post('/api/analyze', json={'paths': files, 'target_roi': roi, 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    an = r.json()
    print(f"[analyze] avg_shift={an['motion']['avg_shift']:.2f}px "
          f"achievable={an['sr_ratio']['achievable']:.2f}x "
          f"usable={an['usable_frames']}/{an['total_frames']}")

    r = client.post('/api/process', json={'paths': files, 'target_roi': roi, 'sr_ratio': 2.0})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    status = {}
    for _ in range(2400):
        status = client.get(f'/api/jobs/{job_id}').json()
        if status['state'] in ('done', 'error', 'cancelled'):
            break
        time.sleep(0.25)
    assert status.get('state') == 'done', status
    print(f"[process] {status['state']}")

    res = client.get(f'/api/jobs/{job_id}/result').json()
    rep = res['report']
    assert 'confidence' not in res, 'confidence layers must be removed'
    images = res['images']
    assert images['fused'].startswith('data:image/png'), images['fused'][:32]
    assert images['before'].startswith('data:image/png'), images['before'][:32]
    assert images['width'] > 0 and images['height'] > 0
    al = res['alignment']
    assert 0.0 <= al['score'] <= 1.0, al['score']
    assert al['level'] in ('ok', 'warn', 'bad'), al['level']
    assert al['map'].startswith('data:image/png'), al['map'][:32]
    assert isinstance(al['message'], str) and al['message'], al['message']
    assert len(al['per_frame']) == rep['total_frames'], (len(al['per_frame']), rep['total_frames'])
    assert all(isinstance(f['used'], bool) for f in al['per_frame'])
    assert sum(1 for f in al['per_frame'] if f['used']) == rep['used_frames']
    assert 0.0 <= al['min_zncc'] <= al['mean_zncc'] <= 1.0
    det = rep['metrics'].get('hf_gain_db_vs_single')
    det_txt = f'{det:+.2f}dB' if isinstance(det, (int, float)) else 'n/a'
    print(f"[result] hr_shape={rep['hr_shape']} used={rep['used_frames']}/{rep['total_frames']} "
          f"detail_vs_single={det_txt} "
          f"acutance_vs_stack={rep['metrics']['hf_gain_db']:+.2f}dB")
    print(f"[alignment] score={al['score']:.2f} level={al['level']} "
          f"mean_zncc={al['mean_zncc']:.3f} min_zncc={al['min_zncc']:.3f} "
          f"({al['message']})")

    r = client.post(f'/api/jobs/{job_id}/render',
                    json={'gamma': 1.8, 'brightness': 0.05, 'contrast': 1.15})
    assert r.status_code == 200, r.text
    rendered = r.json()
    assert rendered['fused'].startswith('data:image/png'), rendered['fused'][:32]
    assert rendered['before'].startswith('data:image/png'), rendered['before'][:32]
    assert rendered['width'] == images['width'] and rendered['height'] == images['height']
    print(f"[render] fused+before {rendered['width']}x{rendered['height']}")

    out_dir = os.path.join(REPO_ROOT, 'output')
    os.makedirs(out_dir, exist_ok=True)
    for kind in ('master', 'baked'):
        out = os.path.join(out_dir, f'desktop_verify_{kind}.tiff')
        r = client.post('/api/export', json={'job_id': job_id, 'kind': kind, 'output_path': out})
        assert r.status_code == 200 and os.path.getsize(out) > 0, r.text
        print(f'[export] {kind}: {out}')

    print('\nREAL-RAW VERIFICATION: ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
