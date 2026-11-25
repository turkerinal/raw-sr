"""
Result panel widget (v0.2.0).

Displays the fused output image, drizzle deposit-weight map, and confidence layers.
Provides controls for view transform adjustments (gamma, brightness,
contrast) and export buttons for master (linear) and baked TIFF.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QGroupBox,
    QTabBar,
    QStackedWidget,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage


class ResultPanel(QWidget):
    """Result display panel with multi-layer tabs and view controls."""

    # Signals
    export_master = Signal()
    export_baked = Signal()
    gamma_changed = Signal(float)
    brightness_changed = Signal(float)
    contrast_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self._fused_image = None
        self._deposit_weight = None
        self._confidence_layers = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # Layer tabs
        self._tab_bar = QTabBar()
        self._tab_bar.addTab("Fused")
        self._tab_bar.addTab("Deposit Weight")
        self._tab_bar.addTab("Confidence")

        self._stack = QStackedWidget()
        self._fused_label = self._make_image_label()
        self._deposit_label = self._make_image_label()
        self._conf_label = self._make_image_label()
        self._stack.addWidget(self._fused_label)
        self._stack.addWidget(self._deposit_label)
        self._stack.addWidget(self._conf_label)

        self._tab_bar.currentChanged.connect(self._stack.setCurrentIndex)

        layout.addWidget(self._tab_bar)
        layout.addWidget(self._stack)

        # View transform controls
        vt_group = QGroupBox("View Transform")
        vt_layout = QVBoxLayout()

        # Gamma
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Gamma:"))
        self._gamma_slider = self._make_slider(10, 40, 22)  # 1.0 to 4.0, default 2.2
        gamma_row.addWidget(self._gamma_slider)
        self._gamma_value = QLabel("2.2")
        gamma_row.addWidget(self._gamma_value)
        self._gamma_slider.valueChanged.connect(self._on_gamma_changed)
        vt_layout.addLayout(gamma_row)

        # Brightness
        bright_row = QHBoxLayout()
        bright_row.addWidget(QLabel("Brightness:"))
        self._brightness_slider = self._make_slider(0, 200, 100)  # -1.0 to +1.0 mapped
        bright_row.addWidget(self._brightness_slider)
        self._brightness_value = QLabel("0.0")
        bright_row.addWidget(self._brightness_value)
        self._brightness_slider.valueChanged.connect(self._on_brightness_changed)
        vt_layout.addLayout(bright_row)

        # Contrast
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast:"))
        self._contrast_slider = self._make_slider(0, 200, 100)  # 0.0 to 2.0 mapped
        contrast_row.addWidget(self._contrast_slider)
        self._contrast_value = QLabel("1.0")
        contrast_row.addWidget(self._contrast_value)
        self._contrast_slider.valueChanged.connect(self._on_contrast_changed)
        vt_layout.addLayout(contrast_row)

        vt_group.setLayout(vt_layout)
        layout.addWidget(vt_group)

        # Export buttons
        btn_row = QHBoxLayout()
        self._export_master_btn = QPushButton("Export Master (Linear)")
        self._export_baked_btn = QPushButton("Export Baked")
        self._export_master_btn.clicked.connect(self.export_master)
        self._export_baked_btn.clicked.connect(self.export_baked)
        btn_row.addWidget(self._export_master_btn)
        btn_row.addWidget(self._export_baked_btn)
        layout.addLayout(btn_row)

        # Status / report
        self._report_label = QLabel("Ready.")
        self._report_label.setWordWrap(True)
        layout.addWidget(self._report_label)

        self.setLayout(layout)

    def _make_image_label(self):
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("border: 1px solid #888; background: #222;")
        label.setMinimumSize(200, 200)
        return label

    def _make_slider(self, min_val, max_val, default):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        return slider

    def _on_gamma_changed(self, val):
        gamma = val / 10.0
        self._gamma_value.setText(f"{gamma:.1f}")
        self.gamma_changed.emit(gamma)

    def _on_brightness_changed(self, val):
        brightness = (val - 100) / 100.0
        self._brightness_value.setText(f"{brightness:.1f}")
        self.brightness_changed.emit(brightness)

    def _on_contrast_changed(self, val):
        contrast = val / 100.0
        self._contrast_value.setText(f"{contrast:.1f}")
        self.contrast_changed.emit(contrast)

    def set_fused_image(self, rgb_array):
        """Set the fused RGB image for display (uint8 or float32)."""
        self._fused_image = rgb_array
        self._display_image(self._fused_label, rgb_array)

    def set_deposit_weight(self, weight_map):
        """Set the drizzle deposit-weight map for display (float array).

        This is the per-HR-pixel sum of deposition weights scaled by the frame
        coverage, not an SNR estimate: its absolute scale depends on ratio,
        sub-sampling and frame count, so only relative structure is meaningful.
        """
        self._deposit_weight = weight_map
        self._display_heatmap(self._deposit_label, weight_map)

    def set_confidence_layers(self, layers):
        """Set confidence layers dict for display."""
        self._confidence_layers = layers

    def set_report(self, report_text):
        """Update the report status label."""
        self._report_label.setText(report_text)

    def _display_image(self, label, rgb_array):
        """Display an RGB array as a pixmap."""
        if rgb_array is None:
            label.setText("No image")
            return
        if rgb_array.dtype != 'uint8':
            rgb = (rgb_array * 255).clip(0, 255).astype('uint8')
        else:
            rgb = rgb_array
        h, w = rgb.shape[:2]
        bytes_per_line = w * 3
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def _display_heatmap(self, label, data):
        """Display a scalar array as a grayscale heatmap."""
        if data is None:
            label.setText("No data")
            return
        if data.ndim == 3:
            data = data[:, :, 0]
        norm = data / max(data.max(), 1e-10)
        rgb = (norm * 255).astype('uint8')
        rgb3 = np.stack([rgb, rgb, rgb], axis=-1)
        self._display_image(label, rgb3)

    @property
    def current_gamma(self):
        return self._gamma_slider.value() / 10.0

    @property
    def current_brightness(self):
        return (self._brightness_slider.value() - 100) / 100.0

    @property
    def current_contrast(self):
        return self._contrast_slider.value() / 100.0
