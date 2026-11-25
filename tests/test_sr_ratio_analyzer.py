"""Tests for SR Ratio Analyzer."""
import numpy as np
from raw_sr.core.sr_ratio_analyzer import SRRatioAnalyzer


class TestSRRatioAnalyzer:
    def test_requested_2x_with_sufficient_shifts_reports_achievable_2x(self):
        """With diverse subpixel shifts, achievable SR ≈ requested."""
        analyzer = SRRatioAnalyzer()
        # Shifts spread across both axes to cover all 4 subpixel bins at 2x
        shifts = [(0.1, 0.1), (0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
        result = analyzer.analyze(shifts, requested_ratio=2.0)
        assert result['requested'] == 2.0
        assert result['achievable'] >= 1.5  # should be close to 2

    def test_zero_shift_reports_achievable_1x(self):
        """With no sub-pixel diversity, achievable SR drops to 1x (denoise only)."""
        analyzer = SRRatioAnalyzer()
        shifts = [(0, 0), (0, 0), (0, 0)]
        result = analyzer.analyze(shifts, requested_ratio=4.0)
        assert result['achievable'] <= 1.5  # no SR possible
        assert result['requested'] == 4.0

    def test_achievable_capped_by_measured_diversity(self):
        """Achievable SR should not exceed what the measured shifts support."""
        analyzer = SRRatioAnalyzer()
        # 4 distinct subpixel offsets → can support ~2x at most
        shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
        result = analyzer.analyze(shifts, requested_ratio=10.0)
        assert result['achievable'] < 10.0  # capped by measured diversity

    def test_threshold_scales_with_ratio(self):
        """Validation residual threshold should scale with requested SR ratio."""
        analyzer = SRRatioAnalyzer()
        res_array = np.array([0.5, 0.6, 0.7, 0.5, 0.6])
        threshold_1x = analyzer.compute_residual_threshold(res_array, k=3.0, sr_ratio=1.0)
        threshold_4x = analyzer.compute_residual_threshold(res_array, k=3.0, sr_ratio=4.0)
        assert threshold_4x < threshold_1x  # higher SR ratio → stricter threshold

    def test_threshold_returns_positive_value(self):
        """Threshold should always be a positive number."""
        analyzer = SRRatioAnalyzer()
        res_array = np.array([1.0, 1.5, 2.0, 0.5, 3.0])
        threshold = analyzer.compute_residual_threshold(res_array, k=3.0, sr_ratio=2.0)
        assert threshold > 0
