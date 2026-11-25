"""Tests for export module."""
import numpy as np
import pytest
import tifffile
from raw_sr.core.export import Exporter


class TestExporter:
    def test_export_linear_tiff_16bit(self, tmp_path):
        # Create a linear float32 RGB image (0 to 1 range)
        img = np.random.RandomState(42).rand(50, 50, 3).astype(np.float32)

        exporter = Exporter()
        output_path = tmp_path / "test_output.tiff"
        exporter.export_linear(img, str(output_path))

        # Read back and verify
        with tifffile.TiffFile(str(output_path)) as tif:
            saved = tif.pages[0]
            assert saved.dtype == np.uint16
            assert saved.shape == (50, 50, 3)

    def test_export_clamps_out_of_range(self, tmp_path):
        # Image with values outside [0, 1]
        img = np.array([[-0.5, 2.0, 0.5]], dtype=np.float32).reshape(1, 1, 3)
        img = np.tile(img, (10, 10, 1))

        exporter = Exporter()
        output_path = tmp_path / "clamped.tiff"
        exporter.export_linear(img, str(output_path))

        data = tifffile.imread(str(output_path))
        # Should be clamped to [0, 65535]
        assert data.min() == 0
        assert data.max() == 65535

    def test_export_preserves_linearity(self, tmp_path):
        # A known linear value should map linearly to uint16
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5  # mid-gray

        exporter = Exporter()
        output_path = tmp_path / "linear.tiff"
        exporter.export_linear(img, str(output_path))

        data = tifffile.imread(str(output_path))
        # 0.5 in float32 → 32767 or 32768 in uint16 (0.5 * 65535)
        np.testing.assert_allclose(data[0, 0, 0], 32767.5, atol=1)

    def test_export_rounds_instead_of_truncating(self, tmp_path):
        """Quantisation must be nearest, not a -0.5 LSB truncation bias."""
        from raw_sr.core.superres import to_uint16_linear
        # Codes just above an integer truncate down; nearest rounds up.
        codes = np.array([0.6, 1.6, 10.6, 32767.6, 65534.6])
        img = np.broadcast_to((codes / 65535.0).astype(np.float32)[:, None, None],
                              (5, 1, 3))
        out = to_uint16_linear(img)[:, 0, 0]
        assert np.array_equal(out, np.array([1, 2, 11, 32768, 65535]))

    def test_export_rejects_jpeg_extension(self, tmp_path):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        exporter = Exporter()
        output_path = tmp_path / "test.jpg"

        with pytest.raises(ValueError, match="TIFF"):
            exporter.export_linear(img, str(output_path))

    def test_export_baked_applies_view_transform(self, tmp_path):
        """Baked export should apply view transform (gamma, brightness, contrast)."""
        from raw_sr.core.view_transform import ViewTransform

        img = np.full((10, 10, 3), 0.5, dtype=np.float32)
        vt = ViewTransform(gamma=2.2, brightness=0.0, contrast=1.0)

        exporter = Exporter()
        output_path = tmp_path / "baked.tiff"
        exporter.export_baked(img, str(output_path), view_transform=vt)

        data = tifffile.imread(str(output_path))
        assert data.dtype == np.uint16
        # 0.5 linear → ~0.73 display (inverse gamma 2.2) → ~47891 uint16
        np.testing.assert_allclose(data[0, 0, 0], 0.7306 * 65535, atol=500)

    def test_export_baked_default_view_transform(self, tmp_path):
        """Baked export with no view_transform should use defaults (gamma=2.2)."""
        img = np.full((10, 10, 3), 0.8, dtype=np.float32)

        exporter = Exporter()
        output_path = tmp_path / "baked_default.tiff"
        exporter.export_baked(img, str(output_path))

        data = tifffile.imread(str(output_path))
        assert data.dtype == np.uint16
        # Should be brighter than linear (inverse gamma brightens midtones)
        linear_val = int(0.8 * 65535)
        baked_val = data[0, 0, 0]
        assert baked_val != linear_val  # gamma changed it

    def test_export_baked_rejects_jpeg(self, tmp_path):
        """Baked export should also validate TIFF format."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        exporter = Exporter()
        output_path = tmp_path / "baked.jpg"

        with pytest.raises(ValueError, match="TIFF"):
            exporter.export_baked(img, str(output_path))