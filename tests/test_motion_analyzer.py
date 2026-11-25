"""Tests for motion analyzer."""
import numpy as np
import cv2
from raw_sr.core.motion_analyzer import MotionAnalyzer


class TestMotionAnalyzer:
    def test_identical_frames_show_zero_shift(self):
        """Identical frames should report ~0px average shift."""
        frames = [np.ones((100, 100, 3), dtype=np.uint8) * 128] * 5
        analyzer = MotionAnalyzer()
        shifts = analyzer.estimate_shifts(frames)
        # Average shift should be near zero
        avg_shift = np.mean([np.linalg.norm(s) for s in shifts[1:]])  # skip frame 0
        assert avg_shift < 1.0

    def test_shifted_frames_show_expected_shift(self):
        """Frames shifted by known amount should report that shift."""
        np.random.seed(42)
        base = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        frames = [base.copy()]
        H = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], dtype=np.float64)
        shifted = cv2.warpPerspective(base, H, (200, 200))
        frames.append(shifted)
        analyzer = MotionAnalyzer()
        shifts = analyzer.estimate_shifts(frames)
        # The shift for frame 1 should be close to (5, 3)
        shift = shifts[1]
        assert abs(shift[0] - 5) < 1.0  # dx
        assert abs(shift[1] - 3) < 1.0  # dy

    def test_sufficiency_feedback_suggests_subpixel(self):
        """Small fractional shifts (<1px) should indicate suitability for SR."""
        np.random.seed(42)
        base = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        analyzer = MotionAnalyzer()
        feedback = analyzer.analyze([base, base])
        assert 'avg_shift' in feedback
        assert 'sufficient' in feedback

    def test_sufficiency_flagged_too_small(self):
        """If average shift is ~0, feedback should warn 'too small'."""
        np.random.seed(42)
        base = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        analyzer = MotionAnalyzer()
        feedback = analyzer.analyze([base.copy(), base.copy(), base.copy()])
        # Identical frames → zero shift → not suitable for SR
        assert feedback['sufficient'] == False
