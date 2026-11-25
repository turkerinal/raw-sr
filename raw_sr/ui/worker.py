"""
Background super-resolution worker.

Loads the selected RAW files as full-resolution linear frames and runs the
genuine multi-frame :class:`~raw_sr.core.superres.SuperResolution` core in a
QThread so the UI stays responsive. Emits a ``result_data`` dict with the HR
linear image, a coverage map, confidence layers, a metrics report and a view
transform.

The ROI is resolved here (not in the UI) so a full-frame request cannot blow
the HR buffer: requests whose HR output would exceed ``max_hr_area`` pixels
(or whose input exceeds ``max_roi_area``) are replaced by a centred square that
fits and flagged as ``roi_clipped`` in the report.
"""

import math

from PySide6.QtCore import QObject, Signal, Slot

from raw_sr.core.superres import load_raw_linear, SuperResolution
from raw_sr.core.view_transform import ViewTransform

# Input ROI budget. At ratio 2 this caps the HR buffer near 17 MP (~200 MB
# float32 RGB); the full 3472x4624 sensor at 2x would be ~770 MB.
MAX_ROI_AREA = 2048 * 2048

# Output (HR) pixel budget. The HR buffer is ratio^2 x the input, so the input
# cap alone is not enough: a 2048x2048 input at 8x would be a 16384x16384 float
# RGB buffer (~3 GB, more with the accumulator) and could exhaust memory. This
# bounds the OUTPUT instead, which is what actually costs memory. 4096x4096 is
# proven feasible by the 4x real-data acceptance run.
MAX_HR_AREA = 4096 * 4096


def _frame_size(path):
    """(height, width) of a RAW file's postprocessed output, cheaply."""
    import rawpy
    with rawpy.imread(path) as raw:
        return int(raw.sizes.height), int(raw.sizes.width)


def resolve_roi(path, target_roi, max_roi_area=MAX_ROI_AREA, ratio=1.0,
                max_hr_area=MAX_HR_AREA):
    """Clamp a requested (x, y, w, h) ROI to the frame and the memory budget.

    The effective input cap is ``min(max_roi_area, max_hr_area / ratio^2)`` so
    the HR buffer stays within ``max_hr_area`` pixels for the requested ratio.

    Returns ``(roi, clipped)`` where ``roi`` is a valid in-bounds crop.
    """
    h, w = _frame_size(path)
    if target_roi is None:
        x, y, rw, rh = 0, 0, w, h
    else:
        x, y, rw, rh = (int(v) for v in target_roi)
        rw, rh = max(rw, 1), max(rh, 1)
    clipped = False
    cap = min(max_roi_area, max_hr_area / max(float(ratio), 1e-6) ** 2)
    if rw * rh > cap:
        side = max(1, int(math.sqrt(cap)))
        x = (w - side) // 2
        y = (h - side) // 2
        rw = rh = side
        clipped = True
    # Keep the crop in bounds.
    rw = min(rw, w)
    rh = min(rh, h)
    x = min(max(x, 0), w - rw)
    y = min(max(y, 0), h - rh)
    return (x, y, rw, rh), clipped


class PipelineWorker(QObject):
    """QThread worker that runs the multi-frame SR core off the UI thread."""

    progress = Signal(int, str)   # percentage, message
    finished = Signal(dict)       # result_data dict
    error = Signal(str)           # error message

    def __init__(self, files, target_roi=None, requested_sr_ratio=2.0,
                 registration="homography", ecc_scale=0.25,
                 max_roi_area=MAX_ROI_AREA):
        super().__init__()
        self.files = list(files)
        self.target_roi = target_roi
        self.requested_sr_ratio = float(requested_sr_ratio)
        self.registration = registration
        self.ecc_scale = ecc_scale
        self.max_roi_area = max_roi_area
        self._cancelled = False

    @Slot()
    def run(self):
        """Load frames, run super-resolution, emit the result."""
        try:
            if not self.files:
                self.error.emit("No RAW files selected.")
                return

            self.progress.emit(2, "Resolving region of interest...")
            roi, clipped = resolve_roi(self.files[0], self.target_roi,
                                       self.max_roi_area,
                                       ratio=self.requested_sr_ratio)

            frames = []
            n = len(self.files)
            for i, path in enumerate(self.files):
                if self._cancelled:
                    self.error.emit("Processing cancelled by user")
                    return
                self.progress.emit(
                    5 + int(45 * i / n),
                    f"Loading frame {i + 1}/{n}...",
                )
                frames.append(load_raw_linear(path, roi=roi))

            if self._cancelled:
                self.error.emit("Processing cancelled by user")
                return

            self.progress.emit(55, "Registering frames (homography)...")
            sr = SuperResolution(ratio=self.requested_sr_ratio,
                                 registration=self.registration,
                                 ecc_scale=self.ecc_scale)
            res = sr.run(frames)
            if self._cancelled:
                self.error.emit("Processing cancelled by user")
                return

            self.progress.emit(90, "Building report...")
            report = {
                "used_frames": res.used_frames,
                "total_frames": res.total_frames,
                "ratio": res.ratio,
                "hr_shape": res.hr.shape,
                "roi": roi,
                "roi_clipped": clipped,
                "metrics": res.metrics,
            }
            result_data = {
                "fused": res.hr,
                "deposit_weight": res.coverage,  # drizzle weight-sum map, not an SNR map
                "confidence_layers": {"coverage": res.coverage},
                "report": report,
                "view_transform": ViewTransform(),
                "metrics": res.metrics,
                "ratio": res.ratio,
            }
            self.progress.emit(100, "Done!")
            self.finished.emit(result_data)

        except Exception as e:  # noqa: BLE001 - surface any failure to the UI
            self.error.emit(f"{type(e).__name__}: {e}")

    def cancel(self):
        """Request cancellation of the current processing."""
        self._cancelled = True
