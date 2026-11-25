"""
File selection panel (v0.2.0).

Provides a reusable widget for selecting RAW files and displaying
the file count, with SR ratio input and process/export controls.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
)
from PySide6.QtCore import Qt, Signal

from raw_sr.ui.file_dialog import RawFileDialog


class FilePanel(QWidget):
    """File selection and SR ratio input panel."""

    files_selected = Signal(list)       # emits list of file paths
    sr_ratio_changed = Signal(float)   # emits requested SR ratio

    def __init__(self):
        super().__init__()
        self._raw_frames = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # File selection button
        self._file_btn = QPushButton("Select RAW Files...")
        self._file_btn.clicked.connect(self._on_select_files)
        layout.addWidget(self._file_btn)

        # File count label
        self._file_count_label = QLabel("No files selected")
        self._file_count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._file_count_label)

        # SR ratio input
        sr_row = QHBoxLayout()
        sr_row.addWidget(QLabel("SR Ratio:"))
        self._sr_ratio_input = QDoubleSpinBox()
        self._sr_ratio_input.setRange(1.0, 8.0)
        self._sr_ratio_input.setSingleStep(0.5)
        # Default to 2x: a 1x run is a no-op upscale and would make the app
        # look like it does nothing. 2x is the acceptance-tested sweet spot.
        self._sr_ratio_input.setValue(2.0)
        sr_row.addWidget(self._sr_ratio_input)
        sr_row.addStretch()
        self._sr_ratio_input.valueChanged.connect(self._on_sr_ratio_changed)
        layout.addLayout(sr_row)

        self.setLayout(layout)

    def _on_select_files(self):
        files = RawFileDialog.select_raw_files(self)
        if files:
            self._raw_frames = files
            self._file_count_label.setText(f"{len(files)} files selected")
            self.files_selected.emit(files)
        else:
            self._file_count_label.setText("No files selected")

    def _on_sr_ratio_changed(self):
        ratio = self._sr_ratio_input.value()
        self.sr_ratio_changed.emit(ratio)

    @property
    def raw_frames(self):
        return self._raw_frames

    @property
    def sr_ratio(self):
        return self._sr_ratio_input.value()
