"""Grad-CAM visualization for U-Net segmentation model.

Generates class activation maps highlighting regions the model
focuses on for its segmentation predictions.
"""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def find_last_conv_layer(model: Any) -> str:
    """Find the name of the last convolutional layer in the model.

    Args:
        model: Keras model to inspect.

    Returns:
        Name of the last Conv2D layer.

    Raises:
        ValueError: If no Conv2D layer is found.
    """
    last_conv = None
    for layer in model.layers:
        if "conv2d" in layer.name.lower() and hasattr(layer, "filters"):
            last_conv = layer.name
    if last_conv is None:
        raise ValueError("No Conv2D layer found in the model")
    return last_conv


def compute_grad_cam(
    model: Any,
    image: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the given image.

    Args:
        model: Keras model.
        image: Input image, shape (H, W, C) or (1, H, W, C).
        layer_name: Name of the target convolutional layer.
            If None, uses the last Conv2D layer.

    Returns:
        Grad-CAM heatmap, shape (H, W), values in [0, 1].
    """
    import tensorflow as tf

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    image_tensor = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image_tensor)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        logger.warning("Gradients are None for layer %s", layer_name)
        h, w = image.shape[1], image.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_output[0] * weights, axis=-1).numpy()

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    h, w = image.shape[1], image.shape[2]
    cam = cv2.resize(cam, (w, h))

    return cam.astype(np.float32)


def overlay_grad_cam(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image, shape (H, W, 3), values in [0, 1].
        heatmap: Grad-CAM heatmap, shape (H, W), values in [0, 1].
        alpha: Transparency for the overlay.
        colormap: OpenCV colormap for the heatmap.

    Returns:
        Overlay image, shape (H, W, 3), uint8.
    """
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    heatmap_uint8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay
