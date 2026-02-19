"""Streamlit dashboard for medical image segmentation.

Provides an interactive web interface for uploading images (JPEG, PNG,
DICOM), running segmentation inference, and displaying results with
mask overlays, uncertainty maps, and Grad-CAM visualizations.
Supports batch processing and mask export.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import cv2
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


def run_uncertainty(
    model: Any,
    image: np.ndarray,
    n_passes: int = 20,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Run MC Dropout uncertainty estimation.

    Args:
        model: Keras model with dropout layers.
        image: Preprocessed image, shape (H, W, 3).
        n_passes: Number of MC Dropout forward passes.
        threshold: Confidence threshold.

    Returns:
        Dictionary with uncertainty maps and confidence mask.
    """
    from src.models.uncertainty import run_uncertainty_estimation

    config = {
        "evaluation": {
            "mc_dropout_passes": n_passes,
            "confidence_threshold": threshold,
        }
    }
    return run_uncertainty_estimation(model, image, config)


def run_grad_cam(model: Any, image: np.ndarray) -> np.ndarray:
    """Generate Grad-CAM heatmap for an image.

    Args:
        model: Keras model.
        image: Preprocessed image, shape (H, W, 3).

    Returns:
        Grad-CAM heatmap, shape (H, W), values in [0, 1].
    """
    from src.models.grad_cam import compute_grad_cam

    return compute_grad_cam(model, image)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Convert a binary mask to PNG bytes for download.

    Args:
        mask: Binary mask, shape (H, W), values in {0, 1}.

    Returns:
        PNG image bytes.
    """
    mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    pil_image = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def create_batch_zip(
    results: list[dict[str, Any]],
) -> bytes:
    """Create a ZIP archive containing all segmentation masks.

    Args:
        results: List of result dictionaries, each with 'filename' and 'mask'.

    Returns:
        ZIP file bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in results:
            filename = result["filename"]
            mask_bytes = mask_to_png_bytes(result["mask"])
            stem = Path(filename).stem
            zf.writestr(f"{stem}_mask.png", mask_bytes)
    return buf.getvalue()


def process_batch(
    model: Any,
    files: list[Any],
    target_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Process a batch of uploaded files.

    Args:
        model: Loaded Keras model.
        files: List of uploaded file objects.
        target_size: Target image size.
        threshold: Confidence threshold.

    Returns:
        List of result dictionaries with filename, image, mask, and overlay.
    """
    results = []
    for uploaded_file in files:
        try:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            image = preprocess_uploaded_image(file_bytes, filename, target_size)
            seg_result = run_segmentation(model, image, threshold)
            results.append(
                {
                    "filename": filename,
                    "image": image,
                    "mask": seg_result["binary_mask"],
                    "overlay": seg_result["overlay"],
                    "prediction": seg_result["prediction"],
                }
            )
        except Exception as e:
            logger.warning("Failed to process %s: %s", uploaded_file.name, e)
            results.append(
                {
                    "filename": uploaded_file.name,
                    "error": str(e),
                }
            )
    return results


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
    mc_passes = get_nested(config, "evaluation", "mc_dropout_passes", default=20)

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
    show_uncertainty = st.sidebar.checkbox("Show Uncertainty Map", value=False)
    show_grad_cam = st.sidebar.checkbox("Show Grad-CAM", value=False)

    # Mode selection
    mode = st.sidebar.radio("Mode", ["Single Image", "Batch Processing"])

    # Model loading
    model = _load_model_cached(config)

    if mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Upload a medical image",
            type=SUPPORTED_FORMATS,
            help="Supported formats: JPEG, PNG, DICOM",
        )
        if uploaded_file is not None:
            _process_single_upload(
                st,
                uploaded_file,
                model,
                target_size,
                threshold,
                show_uncertainty,
                show_grad_cam,
                mc_passes,
            )
    else:
        uploaded_files = st.file_uploader(
            "Upload medical images",
            type=SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help="Select multiple files for batch processing",
        )
        if uploaded_files and model is not None:
            _process_batch_upload(st, uploaded_files, model, target_size, threshold)


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
    show_uncertainty: bool = False,
    show_grad_cam: bool = False,
    mc_passes: int = 20,
) -> None:
    """Process a single uploaded file and display results.

    Args:
        st: Streamlit module.
        uploaded_file: Streamlit UploadedFile object.
        model: Loaded Keras model (or None).
        target_size: Target image size.
        threshold: Confidence threshold.
        show_uncertainty: Whether to display uncertainty maps.
        show_grad_cam: Whether to display Grad-CAM.
        mc_passes: Number of MC Dropout passes.
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

        # Download mask
        mask_bytes = mask_to_png_bytes(results["binary_mask"])
        st.download_button(
            "Download Mask (PNG)",
            data=mask_bytes,
            file_name=f"{Path(filename).stem}_mask.png",
            mime="image/png",
        )

        # Uncertainty maps
        if show_uncertainty:
            st.subheader("Uncertainty Analysis")
            with st.spinner("Running MC Dropout..."):
                unc_results = run_uncertainty(model, image, mc_passes, threshold)

            unc_col1, unc_col2 = st.columns(2)
            with unc_col1:
                st.caption("Uncertainty Map (Std)")
                std_display = (
                    np.clip(
                        unc_results["std"] / max(unc_results["std"].max(), 1e-6), 0, 1
                    )
                    * 255
                ).astype(np.uint8)
                std_colored = cv2.applyColorMap(std_display, cv2.COLORMAP_HOT)
                std_colored = cv2.cvtColor(std_colored, cv2.COLOR_BGR2RGB)
                st.image(std_colored, use_container_width=True)

            with unc_col2:
                st.caption("Confidence Mask")
                conf_display = (unc_results["confidence_mask"] * 255).astype(np.uint8)
                st.image(conf_display, use_container_width=True)

        # Grad-CAM
        if show_grad_cam:
            st.subheader("Grad-CAM Visualization")
            with st.spinner("Computing Grad-CAM..."):
                from src.models.grad_cam import overlay_grad_cam

                heatmap = run_grad_cam(model, image)
                cam_overlay = overlay_grad_cam(image, heatmap)
            st.image(cam_overlay, use_container_width=True)
    else:
        with col2:
            st.info("Upload a trained model to run segmentation.")


def _process_batch_upload(
    st: Any,
    uploaded_files: list[Any],
    model: Any,
    target_size: int,
    threshold: float,
) -> None:
    """Process batch of uploaded files and display results.

    Args:
        st: Streamlit module.
        uploaded_files: List of uploaded file objects.
        model: Loaded Keras model.
        target_size: Target image size.
        threshold: Confidence threshold.
    """
    st.subheader(f"Batch Processing: {len(uploaded_files)} files")

    with st.spinner(f"Processing {len(uploaded_files)} images..."):
        results = process_batch(model, uploaded_files, target_size, threshold)

    # Display results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if failed:
        st.warning(f"{len(failed)} files failed to process")
        for r in failed:
            st.text(f"  - {r['filename']}: {r['error']}")

    st.success(f"Successfully processed {len(successful)} images")

    for result in successful:
        with st.expander(result["filename"]):
            col1, col2 = st.columns(2)
            with col1:
                display_img = (np.clip(result["image"], 0, 1) * 255).astype(np.uint8)
                st.image(display_img, caption="Original", use_container_width=True)
            with col2:
                st.image(
                    result["overlay"], caption="Segmentation", use_container_width=True
                )

    # Batch download
    if successful:
        zip_bytes = create_batch_zip(successful)
        st.download_button(
            "Download All Masks (ZIP)",
            data=zip_bytes,
            file_name="segmentation_masks.zip",
            mime="application/zip",
        )


if __name__ == "__main__":
    create_streamlit_app()
