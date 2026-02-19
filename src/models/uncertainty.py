"""MC Dropout uncertainty estimation for segmentation predictions.

Runs multiple forward passes with dropout enabled at inference time
to produce mean prediction maps, uncertainty (std) maps, and
confidence-weighted segmentation masks.
"""

import logging
from typing import Any

import numpy as np

from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def mc_dropout_predict(
    model: Any,
    image: np.ndarray,
    n_passes: int = 20,
) -> np.ndarray:
    """Run multiple stochastic forward passes with dropout enabled.

    Args:
        model: Keras model with Dropout layers (training=True mode).
        image: Input image, shape (H, W, C) or (1, H, W, C).
        n_passes: Number of forward passes.

    Returns:
        Array of predictions, shape (n_passes, H, W, 1).
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    predictions = []
    for i in range(n_passes):
        pred = model(image, training=True)
        predictions.append(pred.numpy() if hasattr(pred, "numpy") else np.array(pred))

    stacked = np.concatenate(predictions, axis=0)
    logger.debug("MC Dropout: %d passes, output shape=%s", n_passes, stacked.shape)
    return stacked


def compute_uncertainty_maps(
    predictions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute mean prediction and uncertainty maps from MC predictions.

    Args:
        predictions: Array of shape (N, H, W, 1) from N forward passes.

    Returns:
        Dictionary with keys:
        - 'mean': Mean prediction map, shape (H, W).
        - 'std': Per-pixel standard deviation (uncertainty), shape (H, W).
        - 'entropy': Predictive entropy map, shape (H, W).
    """
    preds = predictions.squeeze(axis=-1) if predictions.ndim == 4 else predictions

    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)

    eps = 1e-7
    mean_clipped = np.clip(mean_pred, eps, 1 - eps)
    entropy = -(
        mean_clipped * np.log2(mean_clipped)
        + (1 - mean_clipped) * np.log2(1 - mean_clipped)
    )

    return {
        "mean": mean_pred,
        "std": std_pred,
        "entropy": entropy,
    }


def confidence_weighted_segmentation(
    mean_prediction: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Produce a confidence-weighted segmentation mask.

    Pixels with high uncertainty are given lower confidence, making
    the segmentation boundary smoother and more conservative.

    Args:
        mean_prediction: Mean prediction map, shape (H, W).
        uncertainty: Uncertainty (std) map, shape (H, W).
        threshold: Base threshold for binarization.

    Returns:
        Confidence-weighted binary mask, shape (H, W).
    """
    max_uncertainty = uncertainty.max()
    if max_uncertainty > 0:
        confidence = 1.0 - (uncertainty / max_uncertainty)
    else:
        confidence = np.ones_like(uncertainty)

    weighted = mean_prediction * confidence
    mask = (weighted > threshold).astype(np.float32)
    return mask


def run_uncertainty_estimation(
    model: Any,
    image: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run full uncertainty estimation pipeline on a single image.

    Args:
        model: Keras model with MC Dropout.
        image: Input image, shape (H, W, C).
        config: Application configuration. If None, loads default.

    Returns:
        Dictionary with keys: 'mean', 'std', 'entropy',
        'confidence_mask', 'predictions'.
    """
    if config is None:
        config = get_config()

    n_passes = get_nested(config, "evaluation", "mc_dropout_passes", default=20)
    threshold = get_nested(config, "evaluation", "confidence_threshold", default=0.5)

    predictions = mc_dropout_predict(model, image, n_passes)
    maps = compute_uncertainty_maps(predictions)
    conf_mask = confidence_weighted_segmentation(maps["mean"], maps["std"], threshold)

    return {
        "mean": maps["mean"],
        "std": maps["std"],
        "entropy": maps["entropy"],
        "confidence_mask": conf_mask,
        "predictions": predictions,
    }
