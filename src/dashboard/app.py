"""Streamlit dashboard for medical image segmentation.

Provides an interactive web interface for uploading images (JPEG, PNG,
DICOM), running segmentation inference, and displaying results with
mask overlays, uncertainty maps, and Grad-CAM visualizations.
"""

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.dicom_handler import convert_to_rgb, extract_pixel_array, read_dicom
from src.data.preprocessor import normalize_image, resize_image
from src.models.evaluator import create_overlay
from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "dcm", "dicom"]


def preprocess_uploaded_image(
    uploaded_bytes: bytes,
    filename: str,
    target_size: int = 256,
) -> np.ndarray:
    """Preprocess an uploaded image for model inference.

    Args:
        uploaded_bytes: Raw bytes of the uploaded file.
        filename: Original filename to determine format.
        target_size: Target spatial dimension.

    Returns:
        Preprocessed image array, shape (H, W, 3), float32 in [0, 1].

    Raises:
        ValueError: If the file format is not supported.
    """
    ext = Path(filename).suffix.lower().lstrip(".")

    if ext in ("dcm", "dicom"):
        return _preprocess_dicom(uploaded_bytes, target_size)
    elif ext in ("jpg", "jpeg", "png"):
        return _preprocess_standard(uploaded_bytes, target_size)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _preprocess_dicom(uploaded_bytes: bytes, target_size: int) -> np.ndarray:
    """Preprocess a DICOM file from bytes.

    Args:
        uploaded_bytes: Raw DICOM file bytes.
        target_size: Target spatial dimension.

    Returns:
        Preprocessed image array.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as f:
        f.write(uploaded_bytes)
        tmp_path = f.name

    try:
        ds = read_dicom(tmp_path)
        pixel_array = extract_pixel_array(ds)
        image = convert_to_rgb(pixel_array)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def _preprocess_standard(uploaded_bytes: bytes, target_size: int) -> np.ndarray:
    """Preprocess a standard image (JPEG/PNG) from bytes.

    Args:
        uploaded_bytes: Raw image bytes.
        target_size: Target spatial dimension.

    Returns:
        Preprocessed image array.
    """
    pil_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    image = np.array(pil_image)
    image = resize_image(image, target_size)
    image = normalize_image(image)
    return image


def run_segmentation(
    model: Any,
    image: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Run segmentation inference on a preprocessed image.

    Args:
        model: Loaded Keras model.
        image: Preprocessed image, shape (H, W, 3), float32 in [0, 1].
        threshold: Binarization threshold for the prediction.

    Returns:
        Dictionary with keys:
        - 'prediction': Raw model output, shape (H, W).
        - 'binary_mask': Thresholded binary mask, shape (H, W).
        - 'overlay': Overlay visualization, shape (H, W, 3).
    """
    input_tensor = np.expand_dims(image, axis=0)
    prediction = model.predict(input_tensor, verbose=0)
    pred_map = prediction[0].squeeze()

    binary_mask = (pred_map > threshold).astype(np.float32)

    dummy_gt = np.zeros_like(pred_map)
    overlay = create_overlay(image, dummy_gt, pred_map, threshold)

    return {
        "prediction": pred_map,
        "binary_mask": binary_mask,
        "overlay": overlay,
    }


def create_streamlit_app() -> None:
    """Create and run the Streamlit dashboard application."""
    try:
        import streamlit as st
    except ImportError:
        logger.error("Streamlit is required. Install with: pip install streamlit")
        return

    config = get_config()
    app_title = get_nested(
        config, "dashboard", "title", default="Medical Image Segmentation"
    )
    target_size = get_nested(config, "data", "image_size", default=256)
    default_threshold = get_nested(
        config, "evaluation", "confidence_threshold", default=0.5
    )

    st.set_page_config(page_title=app_title, layout="wide")
    st.title(app_title)

    # Sidebar controls
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.05,
    )

    # Model loading
    model = _load_model_cached(config)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a medical image",
        type=SUPPORTED_FORMATS,
        help="Supported formats: JPEG, PNG, DICOM",
    )

    if uploaded_file is not None:
        _process_single_upload(st, uploaded_file, model, target_size, threshold)


def _load_model_cached(config: dict[str, Any]) -> Any:
    """Load the segmentation model with caching."""
    try:
        import streamlit as st
    except ImportError:
        return None

    checkpoint_dir = Path(
        get_nested(config, "paths", "checkpoint_dir", default="models/checkpoints")
    )
    checkpoint_path = checkpoint_dir / "best_model.keras"

    if not checkpoint_path.exists():
        st.warning(
            "No trained model found. Please train a model first with: "
            "`python -m src.main train`"
        )
        return None

    try:
        from src.models.trainer import load_trained_model

        model = load_trained_model(checkpoint_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        logger.error("Model loading failed: %s", e)
        return None


def _process_single_upload(
    st: Any,
    uploaded_file: Any,
    model: Any,
    target_size: int,
    threshold: float,
) -> None:
    """Process a single uploaded file and display results.

    Args:
        st: Streamlit module.
        uploaded_file: Streamlit UploadedFile object.
        model: Loaded Keras model (or None).
        target_size: Target image size.
        threshold: Confidence threshold.
    """
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name

    try:
        image = preprocess_uploaded_image(file_bytes, filename, target_size)
    except (ValueError, Exception) as e:
        st.error(f"Failed to process image: {e}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        display_img = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        st.image(display_img, use_container_width=True)

    if model is not None:
        with st.spinner("Running segmentation..."):
            results = run_segmentation(model, image, threshold)

        with col2:
            st.subheader("Segmentation Result")
            st.image(results["overlay"], use_container_width=True)

        st.subheader("Predicted Mask")
        mask_display = (results["binary_mask"] * 255).astype(np.uint8)
        st.image(mask_display, use_container_width=True)
    else:
        with col2:
            st.info("Upload a trained model to run segmentation.")


if __name__ == "__main__":
    create_streamlit_app()
