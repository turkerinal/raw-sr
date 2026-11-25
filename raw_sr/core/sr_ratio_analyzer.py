"""SR Ratio Analyzer — requested vs. measurement-supported achievable SR (v0.2.0)."""
import numpy as np
from typing import List, Tuple, Dict, Any


class SRRatioAnalyzer:
    """Computes measurement-supported achievable SR ratio.

    The achievable SR is informed by subpixel phase diversity in the
    measured inter-frame shifts. The achievable ratio is capped by
    the actual diversity of fractional pixel offsets observed.
    """

    def __init__(self, base_scale: int = 1):
        """
        Args:
            base_scale: Base sampling grid (1 means original pixel spacing).
        """
        self.base_scale = base_scale

    def _fractional_bin_coverage(self, shifts: List[Tuple[float, float]], sr_ratio: float) -> float:
        """Estimate what fraction of subpixel bins are covered by the shifts.

        For an Nx SR ratio, there are N^2 subpixel positions per pixel.
        We bin the fractional parts of the shifts and count coverage.
        """
        if not shifts:
            return 0.0

        n_bins = int(np.ceil(sr_ratio))
        bin_size = 1.0 / n_bins
        covered = set()

        for dx, dy in shifts:
            fx = dx % 1.0  # fractional part
            fy = dy % 1.0
            bx = min(int(fx / bin_size), n_bins - 1)
            by = min(int(fy / bin_size), n_bins - 1)
            covered.add((bx, by))

        total_bins = n_bins * n_bins
        return len(covered) / total_bins if total_bins > 0 else 0.0

    def analyze(self, shifts: List[Tuple[float, float]], requested_ratio: float) -> Dict[str, Any]:
        """Analyze achievable SR given shifts and requested ratio.

        Args:
            shifts: List of (dx, dy) shifts per frame (including frame 0 = (0,0)).
            requested_ratio: User-requested SR ratio (e.g. 2.0 for 2x).

        Returns:
            Dict with:
                - requested: the requested SR ratio
                - achievable: measurement-supported achievable SR ratio
                - coverage_fraction: fraction of subpixel bins covered
                - message: human-readable feedback
        """
        n = len(shifts)
        if n < 2:
            achievable = 1.0
            coverage = 0.0
        else:
            # Estimate achievable based on shift diversity and frame count
            # More frames + more diverse shifts → higher achievable SR
            # Cap by sqrt(n) (Nyquist-like limit on independent subpixel samples)
            max_by_count = min(np.sqrt(n), requested_ratio)
            coverage = self._fractional_bin_coverage(shifts, requested_ratio)
            # Achievable is the minimum of what diversity supports and what frames allow
            diversity_limited = coverage * requested_ratio
            achievable = min(max_by_count, diversity_limited, requested_ratio)
            achievable = max(1.0, achievable)  # at least 1x (denoise-only)

        if achievable >= requested_ratio * 0.8:
            message = f"Requested SR: {requested_ratio:.1f}x — measurement-supported achievable: ~{achievable:.1f}x"
        else:
            message = (f"Requested SR: {requested_ratio:.1f}x — measurement-supported "
                       f"achievable: ~{achievable:.1f}x (insufficient subpixel diversity)")

        return {
            'requested': float(requested_ratio),
            'achievable': float(achievable),
            'coverage_fraction': float(coverage),
            'message': message,
        }

    def compute_residual_threshold(self, residuals: np.ndarray, k: float, sr_ratio: float) -> float:
        """Compute adaptive residual threshold scaled by SR ratio.

        threshold = median(residuals) + k * MAD(residuals)

        The threshold is tightened (reduced) for higher SR ratios because
        more subpixel precision is demanded.

        Args:
            residuals: Array of reprojection residuals.
            k: User-adjustable MAD multiplier (strict↔tolerant).
            sr_ratio: Requested SR ratio (scales the threshold).

        Returns:
            Adaptive threshold value (pixels).
        """
        median_res = float(np.median(residuals))
        mad_res = float(np.median(np.abs(residuals - median_res)))

        # Lower MAD scale factor for higher SR ratios (stricter)
        # At 1x, full tolerance; at higher ratios, scaled down
        # e.g. 1x → 1.0, 2x → 0.707, 4x → 0.5
        scale = 1.0 / np.sqrt(max(sr_ratio, 1.0))

        threshold = (median_res + k * mad_res) * scale
        return threshold
