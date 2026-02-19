"""DICOM file handler for medical image reading and metadata extraction.

Provides utilities to read DICOM files, extract metadata, convert
pixel data to numpy arrays, and handle common DICOM edge cases.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def read_dicom(file_path: str | Path) -> Any:
    """Read a DICOM file and return the dataset object.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        pydicom Dataset object.

    Raises:
        FileNotFoundError: If the DICOM file does not exist.
        ImportError: If pydicom is not installed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    try:
        import pydicom
    except ImportError as e:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install it with: pip install pydicom"
        ) from e

    logger.debug("Reading DICOM file: %s", file_path)
    ds = pydicom.dcmread(str(file_path))
    return ds


def extract_metadata(ds: Any) -> dict[str, Any]:
    """Extract key metadata fields from a DICOM dataset.

    Args:
        ds: pydicom Dataset object.

    Returns:
        Dictionary containing extracted metadata fields.
    """
    metadata: dict[str, Any] = {}

    fields = {
        "PatientID": "patient_id",
        "PatientName": "patient_name",
        "StudyDate": "study_date",
        "Modality": "modality",
        "Rows": "rows",
        "Columns": "columns",
        "BitsAllocated": "bits_allocated",
        "BitsStored": "bits_stored",
        "PhotometricInterpretation": "photometric_interpretation",
        "PixelSpacing": "pixel_spacing",
        "ImageType": "image_type",
        "StudyDescription": "study_description",
        "SeriesDescription": "series_description",
        "Manufacturer": "manufacturer",
    }

    for dicom_field, key in fields.items():
        if hasattr(ds, dicom_field):
            value = getattr(ds, dicom_field)
            if hasattr(value, "original_string"):
                value = str(value)
            elif isinstance(value, (list, tuple)):
                value = [
                    float(v) if isinstance(v, (int, float)) else str(v) for v in value
                ]
            elif hasattr(value, "value"):
                value = value.value
            metadata[key] = value
        else:
            metadata[key] = None

    return metadata


def extract_pixel_array(ds: Any) -> np.ndarray:
    """Extract the pixel data from a DICOM dataset as a numpy array.

    Handles different photometric interpretations and bit depths.
    The output is always a float32 array with values in [0, 1].

    Args:
        ds: pydicom Dataset object with pixel data.

    Returns:
        Numpy array of shape (H, W) or (H, W, C) with float32 values in [0, 1].

    Raises:
        AttributeError: If the dataset does not contain pixel data.
    """
    if not hasattr(ds, "pixel_array"):
        raise AttributeError("DICOM dataset does not contain pixel data")

    pixel_array = ds.pixel_array.astype(np.float32)

    # Normalize to [0, 1]
    pmin = pixel_array.min()
    pmax = pixel_array.max()
    if pmax > pmin:
        pixel_array = (pixel_array - pmin) / (pmax - pmin)
    else:
        pixel_array = np.zeros_like(pixel_array)

    # Handle MONOCHROME1 (inverted)
    photometric = getattr(ds, "PhotometricInterpretation", "")
    if photometric == "MONOCHROME1":
        pixel_array = 1.0 - pixel_array

    logger.debug(
        "Extracted pixel array: shape=%s, dtype=%s",
        pixel_array.shape,
        pixel_array.dtype,
    )
    return pixel_array


def dicom_to_numpy(
    file_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a DICOM file and return both the pixel array and metadata.

    This is a convenience function that combines read_dicom,
    extract_metadata, and extract_pixel_array.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        Tuple of (pixel_array, metadata_dict).
    """
    ds = read_dicom(file_path)
    metadata = extract_metadata(ds)
    pixel_array = extract_pixel_array(ds)
    return pixel_array, metadata


def convert_to_rgb(pixel_array: np.ndarray) -> np.ndarray:
    """Convert a grayscale pixel array to RGB by repeating channels.

    Args:
        pixel_array: Array of shape (H, W) or (H, W, 1).

    Returns:
        Array of shape (H, W, 3) with the same values across channels.
        If already 3-channel, returns unchanged.
    """
    if pixel_array.ndim == 2:
        return np.stack([pixel_array] * 3, axis=-1)
    if pixel_array.ndim == 3 and pixel_array.shape[-1] == 1:
        return np.concatenate([pixel_array] * 3, axis=-1)
    if pixel_array.ndim == 3 and pixel_array.shape[-1] == 3:
        return pixel_array
    raise ValueError(
        f"Unexpected pixel array shape: {pixel_array.shape}. "
        "Expected (H, W), (H, W, 1), or (H, W, 3)."
    )
