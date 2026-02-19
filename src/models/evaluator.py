"""Evaluation suite for segmentation model performance.

Computes Dice coefficient, IoU (Jaccard), pixel accuracy, sensitivity,
and specificity. Supports per-image evaluation with overlay visualization
and aggregate metrics with confidence intervals.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from scipy import stats


logger = logging.getLogger(__name__)


def compute_dice(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute Dice coefficient between prediction and ground truth.

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        Dice coefficient in [0, 1].
    """
    smooth = 1e-7
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return float(
        (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    )


def compute_iou(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute Intersection over Union (Jaccard Index).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        IoU score in [0, 1].
    """
    smooth = 1e-7
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred_bin.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_pixel_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute pixel-level accuracy.

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        Pixel accuracy in [0, 1].
    """
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    correct = np.sum(y_true.flatten() == y_pred_bin.flatten())
    total = y_true.size
    return float(correct / total) if total > 0 else 0.0


def compute_sensitivity(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute sensitivity (true positive rate / recall).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        Sensitivity in [0, 1].
    """
    smooth = 1e-7
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    tp = np.sum(y_true.flatten() * y_pred_bin.flatten())
    fn = np.sum(y_true.flatten() * (1 - y_pred_bin.flatten()))
    return float((tp + smooth) / (tp + fn + smooth))


def compute_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute specificity (true negative rate).

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        Specificity in [0, 1].
    """
    smooth = 1e-7
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    tn = np.sum((1 - y_true.flatten()) * (1 - y_pred_bin.flatten()))
    fp = np.sum((1 - y_true.flatten()) * y_pred_bin.flatten())
    return float((tn + smooth) / (tn + fp + smooth))


def evaluate_single(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute all metrics for a single image.

    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted probability or binary mask.
        threshold: Threshold for binarizing predictions.

    Returns:
        Dictionary with all metric values.
    """
    return {
        "dice_coefficient": compute_dice(y_true, y_pred, threshold),
        "iou": compute_iou(y_true, y_pred, threshold),
        "pixel_accuracy": compute_pixel_accuracy(y_true, y_pred, threshold),
        "sensitivity": compute_sensitivity(y_true, y_pred, threshold),
        "specificity": compute_specificity(y_true, y_pred, threshold),
    }


def evaluate_batch(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Compute metrics for a batch of images.

    Args:
        y_true_batch: Batch of ground truth masks, shape (N, H, W) or (N, H, W, 1).
        y_pred_batch: Batch of predictions, shape (N, H, W) or (N, H, W, 1).
        threshold: Threshold for binarizing predictions.

    Returns:
        List of metric dictionaries, one per image.
    """
    results = []
    for i in range(len(y_true_batch)):
        y_true = y_true_batch[i].squeeze()
        y_pred = y_pred_batch[i].squeeze()
        results.append(evaluate_single(y_true, y_pred, threshold))
    return results


def aggregate_metrics(
    per_image_metrics: list[dict[str, float]],
    confidence_level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Compute aggregate statistics with confidence intervals.

    Args:
        per_image_metrics: List of per-image metric dictionaries.
        confidence_level: Confidence level for intervals (e.g., 0.95).

    Returns:
        Dictionary mapping metric names to dicts with keys:
        'mean', 'std', 'ci_lower', 'ci_upper'.
    """
    if not per_image_metrics:
        return {}

    metric_names = per_image_metrics[0].keys()
    aggregated: dict[str, dict[str, float]] = {}

    for name in metric_names:
        values = np.array([m[name] for m in per_image_metrics])
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        if len(values) > 1 and std > 0:
            ci = stats.t.interval(
                confidence_level,
                df=len(values) - 1,
                loc=mean,
                scale=std / np.sqrt(len(values)),
            )
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        aggregated[name] = {
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return aggregated


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a visualization overlay of prediction on the original image.

    Ground truth is shown in green, prediction in red, overlap in yellow.

    Args:
        image: Original image, shape (H, W, 3), values in [0, 1].
        mask: Ground truth mask, shape (H, W).
        prediction: Predicted mask, shape (H, W).
        threshold: Threshold for binarizing the prediction.
        alpha: Transparency for the overlay.

    Returns:
        Overlay image, shape (H, W, 3), uint8.
    """
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    overlay = img_uint8.copy()
    pred_bin = (prediction > threshold).astype(bool)
    mask_bin = mask.astype(bool)

    # Green for ground truth only
    gt_only = mask_bin & ~pred_bin
    overlay[gt_only] = [0, 255, 0]

    # Red for prediction only
    pred_only = pred_bin & ~mask_bin
    overlay[pred_only] = [255, 0, 0]

    # Yellow for overlap
    overlap = mask_bin & pred_bin
    overlay[overlap] = [255, 255, 0]

    result = cv2.addWeighted(img_uint8, 1 - alpha, overlay, alpha, 0)
    return result


def save_evaluation_results(
    results: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    """Save aggregate evaluation results to a text file.

    Args:
        results: Aggregated metrics dictionary.
        output_path: Path to save the results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["Evaluation Results", "=" * 50]
    for metric_name, values in results.items():
        lines.append(
            f"{metric_name}: {values['mean']:.4f} +/- {values['std']:.4f} "
            f"(95% CI: [{values['ci_lower']:.4f}, {values['ci_upper']:.4f}])"
        )

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved evaluation results to %s", output_path)
