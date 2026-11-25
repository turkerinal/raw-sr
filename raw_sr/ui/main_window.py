"""
Main application window (v1.0.0).

RAW multi-frame super-resolution: file selection with SR ratio → ROI selection
→ process → view the HR result + coverage/confidence → export a 16-bit linear
master or a baked (view-transformed) TIFF.

The processing backend is the genuine multi-frame SR core
(:mod:`raw_sr.core.superres`); the UI only collects inputs and displays results.
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QSplitter,
)
from PySide6.QtCore import Qt, QThread

from raw_sr.ui.file_panel import FilePanel
from raw_sr.ui.roi_selector import ROISelector
from raw_sr.ui.result_panel import ResultPanel
from raw_sr.ui.file_dialog import RawFileDialog
from raw_sr.ui.worker import PipelineWorker
from raw_sr.core.export import Exporter
from raw_sr.core.superres import load_raw_linear, to_uint8_view


class MainWindow(QMainWindow):
    """Main window for raw_sr — RAW multi-frame super-resolution."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("raw_sr — RAW Multi-Frame Super-Resolution")
        self.resize(900, 700)

        self._worker = None
        self._thread = None
        self._raw_frames = []          # selected file paths
        self._current_result_data = None

        self._setup_ui()

    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("RAW Multi-Frame Super-Resolution")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(title)

        # Warning (always visible — required by spec Section 10)
        warning = QLabel(
            "⚠ This tool is designed for static scenes photographed\n"
            "with a moving camera. It does not produce reliable\n"
            "results for moving subjects (vehicles, people, animals)."
        )
        warning.setWordWrap(True)
        warning.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(warning)

        # Top row: File panel + ROI selector (side by side)
        top_row = QSplitter(Qt.Horizontal)

        self._file_panel = FilePanel()
        self._file_panel.files_selected.connect(self._on_files_selected)
        self._file_panel.sr_ratio_changed.connect(self._on_sr_ratio_changed)
        top_row.addWidget(self._file_panel)

        self._roi_selector = ROISelector()
        top_row.addWidget(self._roi_selector)

        main_layout.addWidget(top_row)

        # Process + progress row
        process_row = QHBoxLayout()
        self._process_btn = QPushButton("Process")
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._on_process)
        process_row.addWidget(self._process_btn)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        process_row.addWidget(self._progress)

        self._status_label = QLabel("Ready.")
        self._status_label.setAlignment(Qt.AlignCenter)
        process_row.addWidget(self._status_label)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        process_row.addWidget(self._cancel_btn)

        main_layout.addLayout(process_row)

        # Bottom row: Result panel
        self._result_panel = ResultPanel()
        self._result_panel.export_master.connect(self._on_export_master)
        self._result_panel.export_baked.connect(self._on_export_baked)
        self._result_panel.gamma_changed.connect(self._on_gamma_changed)
        self._result_panel.brightness_changed.connect(self._on_brightness_changed)
        self._result_panel.contrast_changed.connect(self._on_contrast_changed)
        main_layout.addWidget(self._result_panel)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def _on_files_selected(self, files):
        """Handle file selection from FilePanel."""
        self._raw_frames = files
        self._status_label.setText(f"{len(files)} files selected. Loading preview...")
        self._load_preview()

    def _load_preview(self):
        """Load the first frame as a display preview for the ROI selector."""
        try:
            frame = load_raw_linear(self._raw_frames[0])
            preview = to_uint8_view(frame.rgb, gamma=2.2, exposure=4.0)
            self._roi_selector.set_frame(preview)
            h, w = frame.rgb.shape[:2]
            self._process_btn.setEnabled(True)
            self._status_label.setText(
                f"{len(self._raw_frames)} frames ready. Source {w}x{h}.")
        except Exception as e:  # noqa: BLE001
            self._status_label.setText(f"Error loading RAW files: {e}")

    def _on_sr_ratio_changed(self, ratio):
        """Update the requested SR ratio for the next process run."""
        self._status_label.setText(f"SR ratio set to {ratio:.1f}x")

    def _on_process(self):
        """Handle process button click."""
        if not self._raw_frames:
            return

        self._progress.setValue(0)
        self._process_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Processing...")

        self._worker = PipelineWorker(
            self._raw_frames,
            target_roi=self._roi_selector.target_roi,
            requested_sr_ratio=self._file_panel.sr_ratio,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def _on_progress(self, percentage, message):
        """Update progress bar and status."""
        self._progress.setValue(percentage)
        self._status_label.setText(message)

    def _on_finished(self, result_data):
        """Handle pipeline completion — display results in panels."""
        self._status_label.setText("Processing complete.")

        fused = result_data["fused"]
        self._result_panel.set_fused_image(
            to_uint8_view(fused, gamma=2.2, exposure=4.0))
        self._result_panel.set_deposit_weight(result_data["deposit_weight"])
        self._result_panel.set_confidence_layers(result_data["confidence_layers"])
        self._result_panel.set_report(self._format_report(result_data["report"]))

        self._result_panel._export_master_btn.setEnabled(True)
        self._result_panel._export_baked_btn.setEnabled(True)
        self._current_result_data = result_data

        report = result_data["report"]
        if report.get("roi_clipped"):
            self._status_label.setText(
                "⚠ ROI reduced to fit memory — see report for the crop region.")

        self._cleanup_worker()

    def _format_report(self, report):
        """Format the metadata report into a readable string for the UI."""
        metrics = report.get("metrics", {})
        shape = report.get("hr_shape")
        parts = [
            f"Frames: {report.get('used_frames')}/{report.get('total_frames')}",
            f"SR: {report.get('ratio'):g}x",
        ]
        if shape is not None:
            parts.append(f"Output: {shape[1]}x{shape[0]}")
        parts += [
            f"HF gain: {metrics.get('hf_gain_db', 0.0):+.1f} dB",
            f"Sharpness: {metrics.get('sharpness_ratio', 0.0):.2f}x",
            f"Mean shift: {metrics.get('mean_shift_px', 0.0):.1f} px",
        ]
        if report.get("roi_clipped"):
            parts.append(f"ROI: {report.get('roi')} (clipped)")
        return " | ".join(parts)

    def _on_export_master(self):
        """Export the linear master TIFF (the measured HR data)."""
        result_data = self._current_result_data
        if result_data is None:
            return
        output_path = RawFileDialog.select_output_path(self, "sr_master.tiff")
        if output_path:
            Exporter().export_linear(result_data["fused"], output_path)
            self._status_label.setText(f"Master saved to: {output_path}")

    def _on_export_baked(self):
        """Export the baked (view-transformed) TIFF."""
        result_data = self._current_result_data
        if result_data is None:
            return
        output_path = RawFileDialog.select_output_path(self, "sr_baked.tiff")
        if output_path:
            vt = result_data["view_transform"]
            vt.gamma = self._result_panel.current_gamma
            vt.brightness = self._result_panel.current_brightness
            vt.contrast = self._result_panel.current_contrast
            Exporter().export_baked(result_data["fused"], output_path,
                                    view_transform=vt)
            self._status_label.setText(f"Baked saved to: {output_path}")

    def _on_gamma_changed(self, gamma):
        """Handle gamma slider change from result panel (applied at export)."""
        pass

    def _on_brightness_changed(self, brightness):
        """Handle brightness slider change (applied at export)."""
        pass

    def _on_contrast_changed(self, contrast):
        """Handle contrast slider change (applied at export)."""
        pass

    def _on_error(self, message):
        """Handle pipeline error."""
        self._status_label.setText(f"Error: {message}")
        self._cleanup_worker()

    def _on_cancel(self):
        """Handle cancel button click."""
        if self._worker:
            self._worker.cancel()
        self._status_label.setText("Cancelling...")

    def _cleanup_worker(self):
        """Clean up worker and thread."""
        self._process_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setValue(100)

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
