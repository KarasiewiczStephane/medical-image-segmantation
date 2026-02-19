"""Tests for the evaluation suite."""

from pathlib import Path

import numpy as np
import pytest

from src.models.evaluator import (
    aggregate_metrics,
    compute_dice,
    compute_iou,
    compute_pixel_accuracy,
    compute_sensitivity,
    compute_specificity,
    create_overlay,
    evaluate_batch,
    evaluate_single,
    save_evaluation_results,
)


class TestComputeDice:
    """Tests for compute_dice function."""

    def test_perfect_match(self) -> None:
        """Dice of identical masks should be ~1."""
        mask = np.ones((64, 64), dtype=np.float32)
        assert compute_dice(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        """Dice of non-overlapping masks should be ~0."""
        y_true = np.ones((64, 64), dtype=np.float32)
        y_pred = np.zeros((64, 64), dtype=np.float32)
        assert compute_dice(y_true, y_pred) < 0.01

    def test_partial_overlap(self) -> None:
        """Dice of partially overlapping masks between 0 and 1."""
        y_true = np.zeros((4, 4), dtype=np.float32)
        y_true[:2, :] = 1.0
        y_pred = np.zeros((4, 4), dtype=np.float32)
        y_pred[:, :2] = 1.0
        dice = compute_dice(y_true, y_pred)
        assert 0.0 < dice < 1.0

    def test_both_empty(self) -> None:
        """Dice of two empty masks should be ~1 (smoothing)."""
        mask = np.zeros((64, 64), dtype=np.float32)
        assert compute_dice(mask, mask) == pytest.approx(1.0, abs=0.01)


class TestComputeIoU:
    """Tests for compute_iou function."""

    def test_perfect_match(self) -> None:
        """IoU of identical masks should be ~1."""
        mask = np.ones((64, 64), dtype=np.float32)
        assert compute_iou(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        """IoU of non-overlapping masks should be ~0."""
        y_true = np.ones((64, 64), dtype=np.float32)
        y_pred = np.zeros((64, 64), dtype=np.float32)
        assert compute_iou(y_true, y_pred) < 0.01

    def test_iou_less_than_dice(self) -> None:
        """IoU should be <= Dice for the same masks."""
        y_true = np.zeros((8, 8), dtype=np.float32)
        y_true[:4, :] = 1.0
        y_pred = np.zeros((8, 8), dtype=np.float32)
        y_pred[:, :4] = 1.0
        assert compute_iou(y_true, y_pred) <= compute_dice(y_true, y_pred)


class TestComputePixelAccuracy:
    """Tests for compute_pixel_accuracy function."""

    def test_perfect_accuracy(self) -> None:
        """Perfect predictions should give 100% accuracy."""
        mask = np.ones((32, 32), dtype=np.float32)
        assert compute_pixel_accuracy(mask, mask) == pytest.approx(1.0)

    def test_half_accuracy(self) -> None:
        """Half-correct predictions."""
        y_true = np.zeros((2, 2), dtype=np.float32)
        y_true[0, 0] = 1.0
        y_true[0, 1] = 1.0
        y_pred = np.zeros((2, 2), dtype=np.float32)
        y_pred[0, 0] = 1.0
        y_pred[1, 0] = 1.0
        assert compute_pixel_accuracy(y_true, y_pred) == pytest.approx(0.5)


class TestComputeSensitivity:
    """Tests for compute_sensitivity function."""

    def test_perfect_sensitivity(self) -> None:
        """All positives detected."""
        y_true = np.ones((32, 32), dtype=np.float32)
        y_pred = np.ones((32, 32), dtype=np.float32)
        assert compute_sensitivity(y_true, y_pred) == pytest.approx(1.0, abs=1e-5)

    def test_zero_sensitivity(self) -> None:
        """No positives detected."""
        y_true = np.ones((32, 32), dtype=np.float32)
        y_pred = np.zeros((32, 32), dtype=np.float32)
        assert compute_sensitivity(y_true, y_pred) < 0.01


class TestComputeSpecificity:
    """Tests for compute_specificity function."""

    def test_perfect_specificity(self) -> None:
        """All negatives correctly identified."""
        y_true = np.zeros((32, 32), dtype=np.float32)
        y_pred = np.zeros((32, 32), dtype=np.float32)
        assert compute_specificity(y_true, y_pred) == pytest.approx(1.0, abs=1e-5)

    def test_zero_specificity(self) -> None:
        """All negatives incorrectly classified as positive."""
        y_true = np.zeros((32, 32), dtype=np.float32)
        y_pred = np.ones((32, 32), dtype=np.float32)
        assert compute_specificity(y_true, y_pred) < 0.01


class TestEvaluateSingle:
    """Tests for evaluate_single function."""

    def test_returns_all_metrics(self) -> None:
        """Return all five metrics."""
        y_true = np.random.randint(0, 2, (32, 32)).astype(np.float32)
        y_pred = np.random.rand(32, 32).astype(np.float32)
        result = evaluate_single(y_true, y_pred)
        assert "dice_coefficient" in result
        assert "iou" in result
        assert "pixel_accuracy" in result
        assert "sensitivity" in result
        assert "specificity" in result

    def test_metrics_in_range(self) -> None:
        """All metrics should be in [0, 1]."""
        y_true = np.random.randint(0, 2, (32, 32)).astype(np.float32)
        y_pred = np.random.rand(32, 32).astype(np.float32)
        result = evaluate_single(y_true, y_pred)
        for value in result.values():
            assert 0.0 <= value <= 1.0


class TestEvaluateBatch:
    """Tests for evaluate_batch function."""

    def test_batch_length(self) -> None:
        """Return one result per image in batch."""
        y_true = np.random.randint(0, 2, (5, 32, 32)).astype(np.float32)
        y_pred = np.random.rand(5, 32, 32).astype(np.float32)
        results = evaluate_batch(y_true, y_pred)
        assert len(results) == 5

    def test_handles_4d_input(self) -> None:
        """Handle (N, H, W, 1) input shape."""
        y_true = np.random.randint(0, 2, (3, 32, 32, 1)).astype(np.float32)
        y_pred = np.random.rand(3, 32, 32, 1).astype(np.float32)
        results = evaluate_batch(y_true, y_pred)
        assert len(results) == 3


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_basic_aggregation(self) -> None:
        """Compute mean, std, and CI."""
        metrics = [
            {"dice_coefficient": 0.8, "iou": 0.7},
            {"dice_coefficient": 0.9, "iou": 0.8},
            {"dice_coefficient": 0.85, "iou": 0.75},
        ]
        agg = aggregate_metrics(metrics)
        assert "dice_coefficient" in agg
        assert "mean" in agg["dice_coefficient"]
        assert "std" in agg["dice_coefficient"]
        assert agg["dice_coefficient"]["mean"] == pytest.approx(0.85, abs=0.01)

    def test_confidence_interval(self) -> None:
        """CI should contain the mean."""
        metrics = [{"dice_coefficient": v} for v in [0.8, 0.85, 0.9, 0.87, 0.82]]
        agg = aggregate_metrics(metrics)
        ci_lower = agg["dice_coefficient"]["ci_lower"]
        ci_upper = agg["dice_coefficient"]["ci_upper"]
        mean = agg["dice_coefficient"]["mean"]
        assert ci_lower <= mean <= ci_upper

    def test_empty_metrics(self) -> None:
        """Return empty dict for empty input."""
        assert aggregate_metrics([]) == {}

    def test_single_sample(self) -> None:
        """Handle single sample without error."""
        metrics = [{"dice_coefficient": 0.9}]
        agg = aggregate_metrics(metrics)
        assert agg["dice_coefficient"]["mean"] == pytest.approx(0.9)


class TestCreateOverlay:
    """Tests for create_overlay function."""

    def test_output_shape(self) -> None:
        """Output should be (H, W, 3) uint8."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (64, 64)).astype(np.float32)
        pred = np.random.rand(64, 64).astype(np.float32)
        overlay = create_overlay(image, mask, pred)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_grayscale_input(self) -> None:
        """Handle grayscale (H, W) image input."""
        image = np.random.rand(64, 64).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        pred = np.zeros((64, 64), dtype=np.float32)
        overlay = create_overlay(image, mask, pred)
        assert overlay.shape == (64, 64, 3)


class TestSaveEvaluationResults:
    """Tests for save_evaluation_results function."""

    def test_save_to_file(self, tmp_path: Path) -> None:
        """Save results to a text file."""
        results = {
            "dice_coefficient": {
                "mean": 0.85,
                "std": 0.05,
                "ci_lower": 0.80,
                "ci_upper": 0.90,
            },
            "iou": {"mean": 0.75, "std": 0.06, "ci_lower": 0.70, "ci_upper": 0.80},
        }
        output_path = tmp_path / "results.txt"
        save_evaluation_results(results, output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "dice_coefficient" in content
        assert "0.8500" in content
