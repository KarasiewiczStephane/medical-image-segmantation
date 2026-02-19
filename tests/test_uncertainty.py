"""Tests for MC Dropout uncertainty estimation."""

import numpy as np

from src.models.uncertainty import (
    compute_uncertainty_maps,
    confidence_weighted_segmentation,
)


class TestComputeUncertaintyMaps:
    """Tests for compute_uncertainty_maps function."""

    def test_output_keys(self) -> None:
        """Return mean, std, and entropy maps."""
        preds = np.random.rand(10, 64, 64, 1).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert "mean" in result
        assert "std" in result
        assert "entropy" in result

    def test_output_shapes(self) -> None:
        """Output maps have shape (H, W)."""
        preds = np.random.rand(5, 32, 32, 1).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert result["mean"].shape == (32, 32)
        assert result["std"].shape == (32, 32)
        assert result["entropy"].shape == (32, 32)

    def test_mean_in_range(self) -> None:
        """Mean predictions should be in [0, 1]."""
        preds = np.random.rand(20, 16, 16, 1).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert result["mean"].min() >= 0.0
        assert result["mean"].max() <= 1.0

    def test_std_non_negative(self) -> None:
        """Standard deviation should be non-negative."""
        preds = np.random.rand(10, 16, 16, 1).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert result["std"].min() >= 0.0

    def test_identical_predictions_zero_std(self) -> None:
        """Identical predictions should give zero std."""
        single = np.random.rand(1, 16, 16, 1).astype(np.float32)
        preds = np.repeat(single, 10, axis=0)
        result = compute_uncertainty_maps(preds)
        np.testing.assert_allclose(result["std"], 0.0, atol=1e-6)

    def test_entropy_in_range(self) -> None:
        """Entropy should be in [0, 1] for binary predictions."""
        preds = np.random.rand(10, 16, 16, 1).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert result["entropy"].min() >= 0.0
        assert result["entropy"].max() <= 1.0 + 1e-6

    def test_3d_input(self) -> None:
        """Handle (N, H, W) input without channel dimension."""
        preds = np.random.rand(5, 32, 32).astype(np.float32)
        result = compute_uncertainty_maps(preds)
        assert result["mean"].shape == (32, 32)


class TestConfidenceWeightedSegmentation:
    """Tests for confidence_weighted_segmentation function."""

    def test_output_binary(self) -> None:
        """Output should be binary {0, 1}."""
        mean_pred = np.random.rand(32, 32).astype(np.float32)
        uncertainty = np.random.rand(32, 32).astype(np.float32) * 0.3
        mask = confidence_weighted_segmentation(mean_pred, uncertainty)
        unique = set(np.unique(mask))
        assert unique.issubset({0.0, 1.0})

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        mean_pred = np.random.rand(64, 64).astype(np.float32)
        uncertainty = np.random.rand(64, 64).astype(np.float32) * 0.1
        mask = confidence_weighted_segmentation(mean_pred, uncertainty)
        assert mask.shape == (64, 64)

    def test_zero_uncertainty(self) -> None:
        """Zero uncertainty means pure thresholding."""
        mean_pred = np.array([[0.8, 0.3], [0.6, 0.1]], dtype=np.float32)
        uncertainty = np.zeros((2, 2), dtype=np.float32)
        mask = confidence_weighted_segmentation(mean_pred, uncertainty, threshold=0.5)
        expected = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(mask, expected)

    def test_high_uncertainty_reduces_positives(self) -> None:
        """High uncertainty should reduce the number of positive predictions."""
        mean_pred = np.full((32, 32), 0.6, dtype=np.float32)
        low_unc = np.full((32, 32), 0.01, dtype=np.float32)
        high_unc = np.full((32, 32), 0.5, dtype=np.float32)

        mask_low = confidence_weighted_segmentation(mean_pred, low_unc, threshold=0.5)
        mask_high = confidence_weighted_segmentation(mean_pred, high_unc, threshold=0.5)
        assert mask_low.sum() >= mask_high.sum()
