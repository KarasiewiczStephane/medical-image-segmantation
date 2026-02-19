"""Tests for Grad-CAM visualization."""

import numpy as np
import pytest

from src.models.grad_cam import find_last_conv_layer, overlay_grad_cam


def _tf_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


tf_required = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")


@tf_required
class TestFindLastConvLayer:
    """Tests for find_last_conv_layer function."""

    def test_find_in_unet(self) -> None:
        """Find the last conv layer in a U-Net model."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        layer_name = find_last_conv_layer(model)
        assert "conv2d" in layer_name.lower()

    def test_no_conv_layer(self) -> None:
        """Raise ValueError when no Conv2D layer exists."""
        from types import SimpleNamespace

        mock_model = SimpleNamespace()
        mock_model.layers = []
        with pytest.raises(ValueError, match="No Conv2D layer"):
            find_last_conv_layer(mock_model)


@tf_required
class TestComputeGradCam:
    """Tests for compute_grad_cam function."""

    def test_output_shape(self) -> None:
        """Grad-CAM output shape matches input spatial dims."""
        from src.models.grad_cam import compute_grad_cam
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        image = np.random.rand(32, 32, 3).astype(np.float32)
        cam = compute_grad_cam(model, image)
        assert cam.shape == (32, 32)

    def test_output_range(self) -> None:
        """Grad-CAM values should be in [0, 1]."""
        from src.models.grad_cam import compute_grad_cam
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        image = np.random.rand(32, 32, 3).astype(np.float32)
        cam = compute_grad_cam(model, image)
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0


class TestOverlayGradCam:
    """Tests for overlay_grad_cam function."""

    def test_output_shape(self) -> None:
        """Overlay should be (H, W, 3) uint8."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = overlay_grad_cam(image, heatmap)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_grayscale_input(self) -> None:
        """Handle grayscale image input."""
        image = np.random.rand(64, 64).astype(np.float32)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = overlay_grad_cam(image, heatmap)
        assert overlay.shape == (64, 64, 3)

    def test_custom_alpha(self) -> None:
        """Apply custom alpha transparency."""
        image = np.random.rand(32, 32, 3).astype(np.float32)
        heatmap = np.random.rand(32, 32).astype(np.float32)
        overlay = overlay_grad_cam(image, heatmap, alpha=0.7)
        assert overlay.shape == (32, 32, 3)


class TestGradCamImports:
    """Tests that work without TensorFlow."""

    def test_module_imports(self) -> None:
        """Module can be imported without TensorFlow."""
        from src.models import grad_cam  # noqa: F401
