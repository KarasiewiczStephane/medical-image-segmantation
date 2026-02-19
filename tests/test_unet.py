"""Tests for U-Net model architecture."""

import numpy as np
import pytest


def _tf_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


tf_required = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")


@tf_required
class TestBuildUnet:
    """Tests for build_unet function."""

    def test_default_architecture(self) -> None:
        """Build U-Net with default parameters."""
        from src.models.unet import build_unet

        model = build_unet(input_shape=(64, 64, 3))
        assert model.input_shape == (None, 64, 64, 3)
        assert model.output_shape == (None, 64, 64, 1)

    def test_custom_channels(self) -> None:
        """Build U-Net with custom encoder channels."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[16, 32],
            bottleneck_channels=64,
        )
        assert model.output_shape == (None, 32, 32, 1)

    def test_multi_class_output(self) -> None:
        """Build U-Net with multiple output classes."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            num_classes=4,
            encoder_channels=[16, 32],
            bottleneck_channels=64,
            output_activation="softmax",
        )
        assert model.output_shape == (None, 32, 32, 4)

    def test_no_batch_norm(self) -> None:
        """Build U-Net without batch normalization."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[16, 32],
            bottleneck_channels=64,
            use_batch_norm=False,
        )
        assert model is not None

    def test_forward_pass(self) -> None:
        """Run a forward pass through the model."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8, 16],
            bottleneck_channels=32,
        )
        dummy_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (2, 32, 32, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_model_name(self) -> None:
        """Model should be named 'unet'."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        assert model.name == "unet"


@tf_required
class TestBuildUnetFromConfig:
    """Tests for build_unet_from_config function."""

    def test_from_default_config(self) -> None:
        """Build model from default configuration."""
        from src.models.unet import build_unet_from_config

        model = build_unet_from_config()
        assert model.input_shape == (None, 256, 256, 3)
        assert model.output_shape == (None, 256, 256, 1)

    def test_from_custom_config(self) -> None:
        """Build model from custom configuration."""
        from src.models.unet import build_unet_from_config

        config = {
            "data": {
                "image_size": 64,
                "num_channels": 1,
                "num_classes": 1,
            },
            "model": {
                "encoder_channels": [8, 16],
                "bottleneck_channels": 32,
                "dropout_rate": 0.3,
                "activation": "relu",
                "output_activation": "sigmoid",
                "use_batch_norm": True,
            },
        }
        model = build_unet_from_config(config)
        assert model.input_shape == (None, 64, 64, 1)


@tf_required
class TestModelUtilities:
    """Tests for model utility functions."""

    def test_get_model_summary(self) -> None:
        """Get a non-empty model summary string."""
        from src.models.unet import build_unet, get_model_summary

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        summary = get_model_summary(model)
        assert isinstance(summary, str)
        assert "unet" in summary.lower() or "Model" in summary

    def test_count_parameters(self) -> None:
        """Count model parameters."""
        from src.models.unet import build_unet, count_parameters

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        params = count_parameters(model)
        assert params["trainable"] > 0
        assert params["total"] == params["trainable"] + params["non_trainable"]


class TestUnetWithoutTF:
    """Tests for U-Net that work without TensorFlow."""

    def test_module_imports(self) -> None:
        """Module can be imported without TensorFlow."""
        from src.models import unet  # noqa: F401

        assert hasattr(unet, "build_unet")
        assert hasattr(unet, "build_unet_from_config")
        assert hasattr(unet, "count_parameters")
