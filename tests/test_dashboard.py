"""Tests for the Streamlit dashboard application."""

import zipfile
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.dashboard.app import (
    SUPPORTED_FORMATS,
    _preprocess_standard,
    create_batch_zip,
    mask_to_png_bytes,
    preprocess_uploaded_image,
    process_batch,
    run_segmentation,
)


def _create_test_png_bytes(h: int = 64, w: int = 64) -> bytes:
    """Create test PNG image bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_test_jpg_bytes(h: int = 64, w: int = 64) -> bytes:
    """Create test JPEG image bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestSupportedFormats:
    """Tests for supported format constants."""

    def test_formats_defined(self) -> None:
        """Supported formats list is not empty."""
        assert len(SUPPORTED_FORMATS) > 0
        assert "jpg" in SUPPORTED_FORMATS
        assert "png" in SUPPORTED_FORMATS
        assert "dcm" in SUPPORTED_FORMATS


class TestPreprocessUploadedImage:
    """Tests for preprocess_uploaded_image function."""

    def test_png_preprocessing(self) -> None:
        """Preprocess a PNG image."""
        png_bytes = _create_test_png_bytes(100, 80)
        result = preprocess_uploaded_image(png_bytes, "test.png", target_size=64)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_jpg_preprocessing(self) -> None:
        """Preprocess a JPEG image."""
        jpg_bytes = _create_test_jpg_bytes(100, 80)
        result = preprocess_uploaded_image(jpg_bytes, "test.jpg", target_size=32)
        assert result.shape == (32, 32, 3)

    def test_jpeg_extension(self) -> None:
        """Handle .jpeg extension."""
        jpg_bytes = _create_test_jpg_bytes()
        result = preprocess_uploaded_image(jpg_bytes, "test.jpeg", target_size=32)
        assert result.shape == (32, 32, 3)

    def test_unsupported_format(self) -> None:
        """Raise ValueError for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            preprocess_uploaded_image(b"data", "test.bmp", target_size=64)


class TestPreprocessStandard:
    """Tests for _preprocess_standard function."""

    def test_rgb_output(self) -> None:
        """Output should be 3-channel float32."""
        png_bytes = _create_test_png_bytes()
        result = _preprocess_standard(png_bytes, target_size=32)
        assert result.ndim == 3
        assert result.shape[-1] == 3
        assert result.dtype == np.float32


class TestRunSegmentation:
    """Tests for run_segmentation function."""

    def test_returns_expected_keys(self) -> None:
        """Return prediction, binary_mask, and overlay."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 64, 64, 1).astype(
            np.float32
        )

        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = run_segmentation(mock_model, image)

        assert "prediction" in result
        assert "binary_mask" in result
        assert "overlay" in result

    def test_prediction_shape(self) -> None:
        """Prediction should be (H, W)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 32, 32, 1).astype(
            np.float32
        )

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = run_segmentation(mock_model, image)
        assert result["prediction"].shape == (32, 32)

    def test_binary_mask_values(self) -> None:
        """Binary mask should contain only 0 and 1."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 32, 32, 1).astype(
            np.float32
        )

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = run_segmentation(mock_model, image)
        unique = set(np.unique(result["binary_mask"]))
        assert unique.issubset({0.0, 1.0})

    def test_overlay_shape(self) -> None:
        """Overlay should be (H, W, 3) uint8."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 64, 64, 1).astype(
            np.float32
        )

        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = run_segmentation(mock_model, image)
        assert result["overlay"].shape == (64, 64, 3)
        assert result["overlay"].dtype == np.uint8

    def test_custom_threshold(self) -> None:
        """Custom threshold affects binary mask."""
        mock_model = MagicMock()
        pred = np.full((1, 32, 32, 1), 0.6, dtype=np.float32)
        mock_model.predict.return_value = pred

        image = np.random.rand(32, 32, 3).astype(np.float32)

        result_low = run_segmentation(mock_model, image, threshold=0.3)
        result_high = run_segmentation(mock_model, image, threshold=0.9)

        assert result_low["binary_mask"].sum() >= result_high["binary_mask"].sum()


class TestMaskToPngBytes:
    """Tests for mask_to_png_bytes function."""

    def test_returns_valid_png(self) -> None:
        """Return valid PNG bytes from a mask."""
        mask = np.random.randint(0, 2, (32, 32)).astype(np.float32)
        png_bytes = mask_to_png_bytes(mask)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Verify it's valid PNG
        img = Image.open(BytesIO(png_bytes))
        assert img.size == (32, 32)

    def test_binary_mask_values(self) -> None:
        """PNG from all-ones mask should be white."""
        mask = np.ones((16, 16), dtype=np.float32)
        png_bytes = mask_to_png_bytes(mask)
        img = Image.open(BytesIO(png_bytes))
        arr = np.array(img)
        assert arr.max() == 255


class TestCreateBatchZip:
    """Tests for create_batch_zip function."""

    def test_creates_valid_zip(self) -> None:
        """Create a ZIP with mask files."""
        results = [
            {"filename": "img1.png", "mask": np.ones((32, 32), dtype=np.float32)},
            {"filename": "img2.jpg", "mask": np.zeros((32, 32), dtype=np.float32)},
        ]
        zip_bytes = create_batch_zip(results)
        assert isinstance(zip_bytes, bytes)

        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            assert "img1_mask.png" in names
            assert "img2_mask.png" in names
            assert len(names) == 2


class TestProcessBatch:
    """Tests for process_batch function."""

    def test_batch_processing(self) -> None:
        """Process a batch of mock files."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 32, 32, 1).astype(
            np.float32
        )

        mock_file = MagicMock()
        mock_file.read.return_value = _create_test_png_bytes(32, 32)
        mock_file.name = "test.png"

        results = process_batch(mock_model, [mock_file], target_size=32, threshold=0.5)
        assert len(results) == 1
        assert "mask" in results[0]
        assert results[0]["filename"] == "test.png"

    def test_batch_handles_errors(self) -> None:
        """Handle failing files in batch."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 32, 32, 1).astype(
            np.float32
        )

        mock_file = MagicMock()
        mock_file.read.side_effect = Exception("read error")
        mock_file.name = "bad.png"

        results = process_batch(mock_model, [mock_file], target_size=32, threshold=0.5)
        assert len(results) == 1
        assert "error" in results[0]
