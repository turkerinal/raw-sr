"""
ROI selector widget (v0.2.0).

Provides interactive Target ROI and Search Area selection with
visual overlay on the reference frame. Supports reset to full-frame
defaults.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QGroupBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class ROISelector(QWidget):
    """Interactive ROI selection widget with visual preview."""

    # Signals
    target_roi_changed = Signal(tuple)  # (x, y, w, h)
    search_area_changed = Signal(tuple)  # (x, y, w, h)
    apply_roi = Signal(tuple, tuple)  # (target_roi, search_area)

    def __init__(self, frame_shape=(256, 256)):
        super().__init__()
        self._frame_h, self._frame_w = frame_shape
        self._target_roi = (0, 0, self._frame_w, self._frame_h)  # default full
        self._search_area = (0, 0, self._frame_w, self._frame_h)  # default full
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("ROI & Search Area")
        title.setStyleSheet("font-size: 12px; font-weight: bold;")
        layout.addWidget(title)

        # Preview image
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setStyleSheet("border: 1px solid #888; background: #222;")
        layout.addWidget(self._preview)

        # Target ROI inputs
        target_group = QGroupBox("Target ROI (x, y, w, h)")
        target_layout = QVBoxLayout()
        self._target_inputs = self._make_coord_inputs(target_group, target_layout)
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # Search Area inputs
        search_group = QGroupBox("Search Area (x, y, w, h)")
        search_layout = QVBoxLayout()
        self._search_inputs = self._make_coord_inputs(search_group, search_layout)
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)

        # Buttons
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset to Full Frame")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)
        self._update_inputs()

    def _make_coord_inputs(self, group, layout):
        """Create 4 spin boxes for x, y, w, h."""
        inputs = {}
        for label_text, key in [("X:", "x"), ("Y:", "y"), ("W:", "w"), ("H:", "h")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            spin = QSpinBox()
            spin.setRange(0, max(self._frame_w, self._frame_h))
            spin.setSingleStep(10)
            row.addWidget(spin)
            layout.addLayout(row)
            inputs[key] = spin
        for spin in inputs.values():
            spin.valueChanged.connect(self._on_input_changed)
        return inputs

    def _update_inputs(self):
        """Sync spin boxes with current ROI / search area values."""
        tx, ty, tw, th = self._target_roi
        self._target_inputs["x"].setValue(tx)
        self._target_inputs["y"].setValue(ty)
        self._target_inputs["w"].setValue(tw)
        self._target_inputs["h"].setValue(th)

        sx, sy, sw, sh = self._search_area
        self._search_inputs["x"].setValue(sx)
        self._search_inputs["y"].setValue(sy)
        self._search_inputs["w"].setValue(sw)
        self._search_inputs["h"].setValue(sh)

    def _on_input_changed(self):
        """Handle spin box changes — emit coordinate updates."""
        tx = self._target_inputs["x"].value()
        ty = self._target_inputs["y"].value()
        tw = self._target_inputs["w"].value()
        th = self._target_inputs["h"].value()
        self._target_roi = (tx, ty, tw, th)
        self.target_roi_changed.emit(self._target_roi)

        sx = self._search_inputs["x"].value()
        sy = self._search_inputs["y"].value()
        sw = self._search_inputs["w"].value()
        sh = self._search_inputs["h"].value()
        self._search_area = (sx, sy, sw, sh)
        self.search_area_changed.emit(self._search_area)

    def _on_reset(self):
        """Reset both ROI and search area to full frame."""
        self._target_roi = (0, 0, self._frame_w, self._frame_h)
        self._search_area = (0, 0, self._frame_w, self._frame_h)
        self._update_inputs()
        self.target_roi_changed.emit(self._target_roi)
        self.search_area_changed.emit(self._search_area)

    def _on_apply(self):
        """Emit the current ROI pair for processing."""
        self.apply_roi.emit(self._target_roi, self._search_area)

    def set_frame(self, rgb_array):
        """Set the reference frame for preview (RGB uint8).

        Also adopts the frame's real dimensions: the ROI inputs are widened to
        the full sensor size and the selection is reset to the full frame.
        """
        h, w = rgb_array.shape[:2]
        if (h, w) != (self._frame_h, self._frame_w):
            self._frame_h, self._frame_w = h, w
            for inputs in (self._target_inputs, self._search_inputs):
                for spin in inputs.values():
                    spin.setRange(0, max(w, h))
            self._target_roi = (0, 0, w, h)
            self._search_area = (0, 0, w, h)
            self._update_inputs()
        bytes_per_line = w * 3
        qimg = QImage(rgb_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Scale to fit while keeping aspect ratio
        self._preview.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    @property
    def target_roi(self):
        return self._target_roi

    @property
    def search_area(self):
        return self._search_area
