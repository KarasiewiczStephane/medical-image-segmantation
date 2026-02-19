"""Tests for the DICOM file handler."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.dicom_handler import (
    convert_to_rgb,
    dicom_to_numpy,
    extract_metadata,
    extract_pixel_array,
    read_dicom,
)


def _make_mock_dataset(
    pixel_array: np.ndarray | None = None,
    photometric: str = "MONOCHROME2",
    rows: int = 64,
    columns: int = 64,
) -> SimpleNamespace:
    """Create a mock DICOM dataset for testing."""
    ds = SimpleNamespace()
    ds.PatientID = "P001"
    ds.PatientName = "Test Patient"
    ds.StudyDate = "20240101"
    ds.Modality = "CT"
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.PhotometricInterpretation = photometric
    ds.PixelSpacing = [0.5, 0.5]
    if pixel_array is not None:
        ds.pixel_array = pixel_array
    return ds


class TestReadDicom:
    """Tests for read_dicom function."""

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="DICOM file not found"):
            read_dicom("/nonexistent/file.dcm")

    def test_read_valid_file_with_mock_pydicom(self, tmp_path: Path) -> None:
        """Read a valid DICOM file with mocked pydicom."""
        dcm_file = tmp_path / "test.dcm"
        dcm_file.write_bytes(b"fake dicom")

        mock_ds = _make_mock_dataset()
        mock_pydicom = MagicMock()
        mock_pydicom.dcmread.return_value = mock_ds

        with patch.dict(sys.modules, {"pydicom": mock_pydicom}):
            result = read_dicom(str(dcm_file))
            assert result.PatientID == "P001"
            mock_pydicom.dcmread.assert_called_once_with(str(dcm_file))

    def test_import_error_without_pydicom(self, tmp_path: Path) -> None:
        """Raise ImportError when pydicom is not available."""
        dcm_file = tmp_path / "test.dcm"
        dcm_file.write_bytes(b"fake dicom")

        with patch.dict(sys.modules, {"pydicom": None}):
            with pytest.raises(ImportError, match="pydicom is required"):
                read_dicom(str(dcm_file))


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_all_fields(self) -> None:
        """Extract all standard metadata fields."""
        ds = _make_mock_dataset()
        meta = extract_metadata(ds)
        assert meta["patient_id"] == "P001"
        assert meta["modality"] == "CT"
        assert meta["rows"] == 64
        assert meta["bits_allocated"] == 16

    def test_missing_fields_return_none(self) -> None:
        """Return None for missing DICOM fields."""
        ds = SimpleNamespace()
        ds.PatientID = "P002"
        meta = extract_metadata(ds)
        assert meta["patient_id"] == "P002"
        assert meta["modality"] is None
        assert meta["rows"] is None

    def test_pixel_spacing_as_list(self) -> None:
        """Handle PixelSpacing as a list of floats."""
        ds = _make_mock_dataset()
        meta = extract_metadata(ds)
        assert meta["pixel_spacing"] == [0.5, 0.5]


class TestExtractPixelArray:
    """Tests for extract_pixel_array function."""

    def test_normalize_grayscale(self) -> None:
        """Normalize grayscale pixel values to [0, 1]."""
        pixels = np.array([[0, 100], [200, 400]], dtype=np.uint16)
        ds = _make_mock_dataset(pixel_array=pixels)
        result = extract_pixel_array(ds)
        assert result.dtype == np.float32
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_monochrome1_inversion(self) -> None:
        """Invert values for MONOCHROME1 photometric interpretation."""
        pixels = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        ds = _make_mock_dataset(pixel_array=pixels, photometric="MONOCHROME1")
        result = extract_pixel_array(ds)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 1] == pytest.approx(0.0)

    def test_constant_image(self) -> None:
        """Handle constant-value images (avoid division by zero)."""
        pixels = np.full((4, 4), 100, dtype=np.uint16)
        ds = _make_mock_dataset(pixel_array=pixels)
        result = extract_pixel_array(ds)
        assert np.all(result == 0.0)

    def test_no_pixel_data(self) -> None:
        """Raise AttributeError if no pixel data present."""
        ds = SimpleNamespace()
        with pytest.raises(AttributeError, match="does not contain pixel data"):
            extract_pixel_array(ds)


class TestConvertToRgb:
    """Tests for convert_to_rgb function."""

    def test_grayscale_2d(self) -> None:
        """Convert 2D grayscale to 3-channel RGB."""
        arr = np.random.rand(64, 64).astype(np.float32)
        result = convert_to_rgb(arr)
        assert result.shape == (64, 64, 3)
        np.testing.assert_array_equal(result[:, :, 0], arr)
        np.testing.assert_array_equal(result[:, :, 1], arr)

    def test_single_channel_3d(self) -> None:
        """Convert (H, W, 1) to (H, W, 3)."""
        arr = np.random.rand(32, 32, 1).astype(np.float32)
        result = convert_to_rgb(arr)
        assert result.shape == (32, 32, 3)

    def test_already_rgb(self) -> None:
        """Return unchanged if already 3-channel."""
        arr = np.random.rand(32, 32, 3).astype(np.float32)
        result = convert_to_rgb(arr)
        np.testing.assert_array_equal(result, arr)

    def test_invalid_shape(self) -> None:
        """Raise ValueError for unexpected shapes."""
        arr = np.random.rand(32, 32, 4).astype(np.float32)
        with pytest.raises(ValueError, match="Unexpected pixel array shape"):
            convert_to_rgb(arr)


class TestDicomToNumpy:
    """Tests for dicom_to_numpy convenience function."""

    @patch("src.data.dicom_handler.read_dicom")
    def test_returns_array_and_metadata(self, mock_read: MagicMock) -> None:
        """Return both pixel array and metadata."""
        pixels = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        ds = _make_mock_dataset(pixel_array=pixels)
        mock_read.return_value = ds

        arr, meta = dicom_to_numpy("/fake/path.dcm")
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        assert meta["patient_id"] == "P001"
