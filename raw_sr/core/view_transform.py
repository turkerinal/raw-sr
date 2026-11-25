"""View Transform — non-destructive linear → display transform (v0.2.0).

Applies gamma correction, brightness, and contrast adjustments to the
linear master image for on-screen display and baked export.
The linear master itself is never altered.
"""
import numpy as np
from typing import Optional


class ViewTransform:
    """Non-destructive view transform for display and baked export.

    Args:
        gamma: Display gamma (default 2.2). Applied as inverse — pow(val, 1/gamma).
        brightness: Offset added after gamma [-1, 1].
        contrast: Multiplier around 0.5 midpoint [0, 2].
    """

    def __init__(self, gamma: float = 2.2, brightness: float = 0.0, contrast: float = 1.0):
        self.gamma = max(gamma, 0.1)
        self.brightness = np.clip(brightness, -1.0, 1.0)
        self.contrast = max(contrast, 0.0)

    def apply(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Apply view transform to a linear RGB image.

        Args:
            linear_rgb: 3D float32 RGB array in [0, 1] range (linear).

        Returns:
            3D float32 RGB array in [0, 1] range (display-encoded).
        """
        # Work on a copy to avoid mutating the input
        display = np.clip(linear_rgb, 0.0, 1.0).astype(np.float32).copy()

        # Apply inverse gamma for display
        display = np.power(display, 1.0 / self.gamma)

        # Apply contrast around midpoint (0.5)
        display = (display - 0.5) * self.contrast + 0.5

        # Apply brightness offset
        display = display + self.brightness

        # Clamp to valid range
        return np.clip(display, 0.0, 1.0).astype(np.float32)

    def to_uint8(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Apply view transform and return uint8 [0, 255]."""
        display = self.apply(linear_rgb)
        return np.clip(display * 255.0, 0, 255).astype(np.uint8)
