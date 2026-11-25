#!/usr/bin/env python3
"""One-command headless smoke test for raw-sr.

Proves, with no display and no user interaction, that the shipped app works
end to end:

  1. The genuine multi-frame SR core recovers resolution on a synthetic burst
     whose subpixel shifts are known (output shape is ratio x input and the
     high-frequency gain is positive).
  2. The pipeline worker runs the same core off the UI thread and reports HR
     dimensions + metrics (the real app code path, not a re-implementation).
  3. The MainWindow instantiates under the offscreen Qt platform (the app
     launches headless).

With ``--real`` it additionally runs the core on a small ROI of the real DNG
burst when ``test_data/`` is present.

Usage:
    venv/bin/python scripts/smoke.py
    venv/bin/python scripts/smoke.py --real
Exit code is 0 on PASS, 1 on FAIL.
"""

import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_sr.core.superres import RawFrame, SuperResolution  # noqa: E402

RESULT = []


def check(name, ok, detail=""):
    RESULT.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def _synthetic_truth(h=192, w=192, lo=0.02, hi=0.45, seed=7):
    """Broadband synthetic HR scene with both below-Nyquist structure (so
    registration can recover subpixel shifts) and above-Nyquist detail
    (0.25..0.45 cyc/HR px) that single-frame bicubic cannot reconstruct."""
    rng = np.random.default_rng(seed)
    g = rng.random((h, w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    F[(r < lo) | (r > hi)] = 0
    mask = (r >= lo) & (r <= hi)
    F[mask] /= np.sqrt(r[mask])
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    return np.stack([g, g, g], axis=2)


def _sample_lr(truth, dx, dy, lr_shape):
    ratio = truth.shape[0] // lr_shape[0]
    xs, ys = np.meshgrid(np.arange(lr_shape[1], dtype=np.float32),
                         np.arange(lr_shape[0], dtype=np.float32))
    off = 0.5 * (float(ratio) - 1.0)   # pixel-centre aligned HR grid
    return cv2.remap(truth, ratio * (xs + dx) + off,
                     ratio * (ys + dy) + off, cv2.INTER_LINEAR)


_SYNTHETIC_SHIFTS = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75)]


def synthetic_burst():
    truth = _synthetic_truth()
    h, w = truth.shape[0] // 2, truth.shape[1] // 2
    frames = []
    for i, (dx, dy) in enumerate(_SYNTHETIC_SHIFTS):
        lr = _sample_lr(truth, dx, dy, (h, w))
        frames.append(RawFrame(name=f"syn{i}", rgb=lr, green=lr[:, :, 1].copy()))
    return frames, truth


def smoke_core():
    frames, truth = synthetic_burst()
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    h, w = frames[0].rgb.shape[:2]
    check("core: HR shape is ratio x input",
          res.hr.shape == (h * 2, w * 2, 3),
          f"{res.hr.shape} from {(h, w)}")
    check("core: used all frames", res.used_frames == len(frames),
          f"{res.used_frames}/{res.total_frames}")
    # The mission, checked against known truth: SR must reconstruct the HR scene
    # better than bicubic-upsampling a single frame. Align truth to whichever
    # frame SuperResolution selected as reference (auto-ref may pick != 0).
    ref_idx = int(res.metrics.get("reference_index", 0))
    rdx, rdy = _SYNTHETIC_SHIFTS[ref_idx]
    hr_h, hr_w = res.hr.shape[:2]
    xs, ys = np.meshgrid(np.arange(hr_w, dtype=np.float32),
                         np.arange(hr_h, dtype=np.float32))
    truth_ref = cv2.remap(truth, xs + 2.0 * rdx, ys + 2.0 * rdy, cv2.INTER_CUBIC)
    naive = cv2.resize(frames[ref_idx].rgb, (hr_w, hr_h),
                       interpolation=cv2.INTER_CUBIC)
    sl = (slice(8, -8), slice(8, -8))
    err_sr = float(np.mean((res.hr[sl] - truth_ref[sl]) ** 2))
    err_naive = float(np.mean((naive[sl] - truth_ref[sl]) ** 2))
    check("core: beats bicubic single-frame upscale (MSE vs truth)",
          err_sr < err_naive,
          f"SR={err_sr:.5f} < naive={err_naive:.5f}")
    return res


def smoke_worker():
    """Run the real app worker path off the UI thread (no Qt loop needed)."""
    from raw_sr.ui.worker import PipelineWorker
    import raw_sr.ui.worker as worker_mod

    frames, _ = synthetic_burst()
    # The worker loads from disk; feed a synthetic burst through the same core
    # by short-circuiting the loader, keeping the rest of the app path real.
    worker_mod._frame_size = lambda path: (96, 96)
    worker_mod.load_raw_linear = lambda path, roi=None: frames.pop(0)
    w = PipelineWorker(["a", "b", "c", "d", "e"], target_roi=None,
                       requested_sr_ratio=2.0)
    captured = {}
    w.finished.connect(lambda d: captured.update(d))
    w.run()
    ok = bool(captured) and "fused" in captured
    check("worker: produced a result on the app path", ok)
    if ok:
        report = captured["report"]
        check("worker: HR shape reported",
              tuple(report["hr_shape"])[0] == 192,
              f"{report['hr_shape']}")


def smoke_window():
    from PySide6.QtWidgets import QApplication
    from raw_sr.ui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    check("ui: MainWindow instantiates headless", win is not None,
          f"title={win.windowTitle()!r}")
    win.deleteLater()


def smoke_real():
    import glob
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.environ.get("RAW_SR_TEST_DATA") or os.path.join(here, "test_data")
    files = sorted(glob.glob(os.path.join(data_dir, "*.dng")))
    if len(files) < 2:
        check("real: DNG burst present", False,
              "no DNG burst (set RAW_SR_TEST_DATA or add test_data/*.dng)")
        return
    from raw_sr.core.superres import load_raw_linear
    roi = (1650, 1240, 256, 256)  # small ROI keeps the smoke fast
    frames = [load_raw_linear(f, roi=roi) for f in files[:5]]
    res = SuperResolution(ratio=2.0).run(frames)
    check("real: 2x HR from DNG burst", res.hr.shape[:2] == (512, 512),
          f"{res.hr.shape}")
    check("real: positive HF gain vs denoised stack",
          res.metrics["hf_gain_db"] > 0.0,
          f"{res.metrics['hf_gain_db']:+.2f} dB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="also run the core on a small real-DNG ROI")
    args = ap.parse_args()

    print("raw-sr headless smoke")
    smoke_core()
    smoke_worker()
    smoke_window()
    if args.real:
        smoke_real()

    passed, total = sum(RESULT), len(RESULT)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
