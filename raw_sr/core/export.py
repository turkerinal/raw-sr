"""
Export module (v0.1.0 + v0.2.0).

v0.1.0: 16-bit linear TIFF master export.
v0.2.0: Baked (view-transformed, display-ready) 16-bit TIFF export.
"""

import numpy as np
import tifffile


class Exporter:
    """Export results as 16-bit linear TIFF (master file)."""

    def __init__(self):
        pass

    def export_linear(self, linear_rgb, output_path):
        """
        Export a linear RGB float32 image as 16-bit TIFF.

        The image is expected to be in [0, 1] range (linear, post-demosaic).
        Values outside [0, 1] are clamped. No gamma or tone mapping applied
        — this is the linear "master" file.

        Args:
            linear_rgb: 3D float32 RGB array in [0, 1] range.
            output_path: Output file path (should end in .tiff or .tif).

        Raises:
            ValueError: If the file extension is not TIFF.
        """
        # Validate output format — TIFF only for v0.1.0
        lower = output_path.lower()
        if not (lower.endswith('.tiff') or lower.endswith('.tif')):
            raise ValueError(
                f"Export requires TIFF format. Got: {output_path}"
            )

        # Clamp to [0, 1] range before converting
        clamped = np.clip(linear_rgb, 0.0, 1.0)

        # Convert float32 [0, 1] → uint16 [0, 65535]. Round to nearest: a
        # plain astype truncates and biases every code value down by up to
        # one LSB (0.5 LSB on average).
        uint16_data = np.rint(clamped * 65535.0).astype(np.uint16)

        # Write as 16-bit TIFF
        tifffile.imwrite(output_path, uint16_data, dtype=np.uint16)

    def export_baked(self, linear_rgb, output_path, view_transform=None):
        """
        Export a linear RGB image with view transform applied (baked/display-ready).

        Applies gamma, brightness, and contrast adjustments before export.
        This produces a "baked" 16-bit TIFF suitable for direct viewing.

        Args:
            linear_rgb: 3D float32 RGB array in [0, 1] range (linear).
            output_path: Output file path (should end in .tiff or .tif).
            view_transform: Optional ViewTransform instance. If None, uses default
                (gamma=2.2, brightness=0.0, contrast=1.0).

        Raises:
            ValueError: If the file extension is not TIFF.
        """
        # Validate output format — TIFF only
        lower = output_path.lower()
        if not (lower.endswith('.tiff') or lower.endswith('.tif')):
            raise ValueError(
                f"Export requires TIFF format. Got: {output_path}"
            )

        # Apply view transform (non-destructive — operates on a copy)
        if view_transform is None:
            from raw_sr.core.view_transform import ViewTransform
            view_transform = ViewTransform()
        baked_display = view_transform.apply(linear_rgb)

        # Scale float32 [0, 1] → uint16 [0, 65535], rounding to nearest (no
        # truncation bias; see export_linear).
        uint16_data = np.rint(np.clip(baked_display, 0.0, 1.0)
                              * 65535.0).astype(np.uint16)

        tifffile.imwrite(output_path, uint16_data, dtype=np.uint16)

    def export_enhanced(
        self,
        linear_rgb,
        output_path,
        ratio=2.0,
        noise_sigma=None,
        strength=None,
        preset=None,
        view_transform=None,
        baked=False,
    ):
        """
        Compute and export a classically enhanced image on top of a linear master.

        The linear master ``linear_rgb`` is never modified. Supports 16-bit
        TIFF (linear or view-transformed) and 8-bit PNG (always display-encoded).
        """
        lower = output_path.lower()
        is_tiff = lower.endswith('.tiff') or lower.endswith('.tif')
        is_png = lower.endswith('.png')
        if not (is_tiff or is_png):
            raise ValueError(
                f"Enhanced export requires TIFF or PNG format. Got: {output_path}"
            )

        from raw_sr.core.enhance import enhance_image
        res = enhance_image(
            linear_rgb,
            ratio=ratio,
            noise_sigma=noise_sigma,
            strength=strength,
            preset=preset,
        )
        enh = res.image

        if is_png or baked:
            if view_transform is None:
                from raw_sr.core.view_transform import ViewTransform
                view_transform = ViewTransform()
            display = view_transform.apply(enh)
            if is_png:
                import cv2
                u8 = np.rint(np.clip(display, 0.0, 1.0) * 255.0).astype(np.uint8)
                cv2.imwrite(output_path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
            else:
                u16 = np.rint(np.clip(display, 0.0, 1.0) * 65535.0).astype(np.uint16)
                tifffile.imwrite(output_path, u16, dtype=np.uint16)
        else:
            u16 = np.rint(np.clip(enh, 0.0, 1.0) * 65535.0).astype(np.uint16)
            tifffile.imwrite(output_path, u16, dtype=np.uint16)

        return res
