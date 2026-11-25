"""Tests for UI modules."""
from PySide6.QtWidgets import QApplication
import pytest

# Use a single QApplication instance for all tests
_app = None


@pytest.fixture(scope="module")
def qapp():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def test_main_window_importable():
    from raw_sr.ui.main_window import MainWindow
    assert MainWindow is not None


def test_file_dialog_importable():
    from raw_sr.ui.file_dialog import RawFileDialog
    assert RawFileDialog is not None


def test_worker_importable():
    from raw_sr.ui.worker import PipelineWorker
    assert PipelineWorker is not None


def test_result_panel_names_deposit_weight_not_snr(qapp):
    """The heat-map tab shows the drizzle weight sum, not an SNR map (A3 #10)."""
    from raw_sr.ui.result_panel import ResultPanel
    panel = ResultPanel()
    tabs = [panel._tab_bar.tabText(i) for i in range(panel._tab_bar.count())]
    assert "SNR Map" not in tabs
    assert "Deposit Weight" in tabs
    assert hasattr(panel, "set_deposit_weight")
    assert not hasattr(panel, "set_snr_map")


def test_main_entry_point():
    from raw_sr.main import main
    assert callable(main)


def test_main_window_can_be_created(qapp):
    """MainWindow should instantiate without errors."""
    from raw_sr.ui.main_window import MainWindow
    window = MainWindow()
    assert window is not None
    window.setWindowTitle("Test")
    assert window.windowTitle() == "Test"


def test_pipeline_worker_signals_exist(qapp):
    """PipelineWorker should have progress, finished, error signals."""
    from raw_sr.ui.worker import PipelineWorker
    worker = PipelineWorker([], requested_sr_ratio=2.0)
    assert hasattr(worker, 'progress')
    assert hasattr(worker, 'finished')
    assert hasattr(worker, 'error')


def test_resolve_roi_clamps_to_budget(monkeypatch):
    """A full-frame request beyond the memory budget becomes a centred crop."""
    from raw_sr.ui import worker as worker_mod
    monkeypatch.setattr(worker_mod, "_frame_size", lambda path: (4624, 3472))
    roi, clipped = worker_mod.resolve_roi("x.dng", None, max_roi_area=1024 * 1024)
    x, y, w, h = roi
    assert clipped is True
    assert w == h == 1024
    assert x == (3472 - 1024) // 2 and y == (4624 - 1024) // 2


def test_resolve_roi_caps_output_area_at_high_ratio(monkeypatch):
    """At a high ratio the input ROI shrinks so the HR buffer stays bounded."""
    from raw_sr.ui import worker as worker_mod
    monkeypatch.setattr(worker_mod, "_frame_size", lambda path: (4624, 3472))
    # Huge input budget: only the output budget may apply.
    roi, clipped = worker_mod.resolve_roi("x.dng", None,
                                          max_roi_area=10**9, ratio=8.0)
    x, y, w, h = roi
    assert clipped is True
    # 4096^2 / 8^2 = 512^2 input -> 4096^2 output (the HR budget).
    assert w == h == 512
    assert (w * 8) * (h * 8) <= worker_mod.MAX_HR_AREA


def test_resolve_roi_keeps_small_request(monkeypatch):
    from raw_sr.ui import worker as worker_mod
    monkeypatch.setattr(worker_mod, "_frame_size", lambda path: (4624, 3472))
    roi, clipped = worker_mod.resolve_roi("x.dng", (100, 120, 800, 600))
    assert clipped is False
    assert roi == (100, 120, 800, 600)


def test_file_panel_defaults_to_2x_sr(qapp):
    """The app must upscale by default; a 1x default is a no-op."""
    from raw_sr.ui.file_panel import FilePanel
    panel = FilePanel()
    assert panel.sr_ratio >= 2.0, f"default SR ratio {panel.sr_ratio} < 2x"


def test_raw_file_dialog_filters_include_raw():
    """RawFileDialog should have RAW filters."""
    from raw_sr.ui.file_dialog import RawFileDialog
    # The class should exist and have the filter attributes
    assert hasattr(RawFileDialog, 'RAW_FILTERS')