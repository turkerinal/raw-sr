"""Motion analyzer — inter-frame shift estimation + sufficiency feedback (v0.2.0)."""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any


class MotionAnalyzer:
    """Estimates inter-frame shifts and provides sufficiency feedback.

    Uses ORB + RANSAC homography to compute per-frame shift relative to
    the reference frame, then reports statistics and whether the motion
    is suitable for subpixel SR.
    """

    def __init__(self, min_useful_shift: float = 0.1, max_usable_shift: float = 5.0):
        """
        Args:
            min_useful_shift: Below this avg shift, motion is "too small" for SR.
            max_usable_shift: Above this avg shift, motion may cause misalignment.
        """
        self.min_useful_shift = min_useful_shift
        self.max_usable_shift = max_usable_shift

    def estimate_shifts(self, frames: List[np.ndarray], homographies: List[np.ndarray] = None) -> List[Tuple[float, float]]:
        """Estimate (dx, dy) shift of each frame relative to reference.

        Args:
            frames: List of RGB/uint8 images.
            homographies: Optional pre-computed homographies. If None, estimated internally.

        Returns:
            List of (dx, dy) shifts per frame (frame 0 is (0, 0)).
        """
        if homographies is None:
            # The live registration path is phase correlation + ECC in
            # ``superres``; the old ORB+RANSAC RegistrationEngine belonged to
            # the removed v0.1 pipeline. Estimate translations the same way the
            # desktop analysis does, so this fallback matches the product.
            from raw_sr.core.superres import estimate_translation, translation_matrix
            ref_green = np.asarray(frames[0])
            if ref_green.ndim == 3:
                ref_green = ref_green[:, :, 1]
            homographies = []
            for frame in frames:
                g = np.asarray(frame)
                if g.ndim == 3:
                    g = g[:, :, 1]
                # ``estimate_translation`` returns the resampling offset; this
                # class reports the frame's displacement in image coordinates,
                # so flip the sign (a content shift of +5 px reads as +5).
                dx, dy = estimate_translation(ref_green, g)
                homographies.append(translation_matrix(-dx, -dy))

        shifts = []
        for H in homographies:
            # Extract translation from homography (assuming small rotation/skew)
            # H maps reference→target. The translation components are H[0,2], H[1,2].
            dx = float(H[0, 2])
            dy = float(H[1, 2])
            shifts.append((dx, dy))

        return shifts

    def analyze(self, frames: List[np.ndarray], homographies: List[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze motion sufficiency across a frame set.

        Returns a dict with:
            - avg_shift: average Euclidean shift magnitude (pixels)
            - max_shift: maximum shift magnitude
            - shifts: per-frame (dx, dy) list
            - sufficient: whether motion is suitable for subpixel SR
            - message: human-readable feedback string
        """
        shifts = self.estimate_shifts(frames, homographies)
        # Skip frame 0 (identity) for statistics
        non_zero_shifts = shifts[1:] if len(shifts) > 1 else shifts
        magnitudes = [np.sqrt(dx**2 + dy**2) for dx, dy in non_zero_shifts]

        avg_shift = float(np.mean(magnitudes)) if magnitudes else 0.0
        max_shift = float(np.max(magnitudes)) if magnitudes else 0.0

        sufficient = self.min_useful_shift <= avg_shift <= self.max_usable_shift

        if avg_shift < self.min_useful_shift:
            message = (f"Average inter-frame shift: {avg_shift:.2f}px — "
                       f"too small for subpixel SR (need ≥{self.min_useful_shift}px)")
        elif avg_shift > self.max_usable_shift:
            message = (f"Average inter-frame shift: {avg_shift:.2f}px — "
                       f"too large, frames may be misaligned")
        else:
            message = (f"Average inter-frame shift: {avg_shift:.2f}px — "
                       f"suitable for subpixel SR")

        return {
            'avg_shift': avg_shift,
            'max_shift': max_shift,
            'shifts': shifts,
            'sufficient': sufficient,
            'message': message,
        }
