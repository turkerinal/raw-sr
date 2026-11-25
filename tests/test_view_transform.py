"""Tests for view transform module."""
import numpy as np
from raw_sr.core.view_transform import ViewTransform


class TestViewTransform:
    def test_linear_mid_gray_maps_to_mid_display(self):
        """Linear 0.5 should map to a displayable mid-value after gamma."""
        vt = ViewTransform()
        linear = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        display = vt.apply(linear)
        assert display.shape == (1, 1, 3)
        assert 0.0 <= display[0, 0, 0] <= 1.0

    def test_gamma_applied(self):
        """Applying gamma 2.2 should brighten mid-tones (inverse gamma)."""
        vt = ViewTransform(gamma=2.2)
        linear = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        display = vt.apply(linear)
        # With gamma 2.2, 0.5 linear → 0.5^(1/2.2) ≈ 0.73 display
        assert display[0, 0, 0] > 0.5  # should be brightened (inverse gamma)

    def test_brightness_adjustment(self):
        """Brightness +0.1 should increase all values by ~0.1."""
        vt = ViewTransform(brightness=0.1)
        linear = np.array([[[0.4, 0.4, 0.4]]], dtype=np.float32)
        display_brightened = vt.apply(linear)
        vt_none = ViewTransform(brightness=0.0)
        display_none = vt_none.apply(linear)
        # Brightness adds offset after gamma; the difference should be ~0.1
        diff = display_brightened[0, 0, 0] - display_none[0, 0, 0]
        assert abs(diff - 0.1) < 0.05

    def test_contrast_adjustment(self):
        """Contrast adjustment should amplify values away from 0.5 midpoint.

        Note: gamma is applied first, then contrast around the post-gamma midpoint.
        """
        vt_high = ViewTransform(contrast=1.5)
        vt_none = ViewTransform(contrast=1.0)
        linear = np.array([[[0.2, 0.8]]], dtype=np.float32)
        display_high = vt_high.apply(linear)
        display_none = vt_none.apply(linear)
        # Higher contrast should push 0.8 higher and 0.2 lower
        assert display_high[0, 0, 1] > display_none[0, 0, 1]
        assert display_high[0, 0, 0] < display_none[0, 0, 0]

    def test_view_transform_does_not_modify_input(self):
        """The original linear array must not be mutated."""
        vt = ViewTransform(brightness=0.2, contrast=1.1, gamma=2.2)
        original = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)
        original_copy = original.copy()
        _ = vt.apply(original)
        np.testing.assert_array_equal(original, original_copy)

    def test_output_in_valid_range(self):
        """All output values must be in [0, 1]."""
        vt = ViewTransform(gamma=2.2, brightness=0.3, contrast=1.5)
        linear = np.random.RandomState(42).rand(50, 50, 3).astype(np.float32)
        display = vt.apply(linear)
        assert display.min() >= 0.0
        assert display.max() <= 1.0
