"""Tests for the training pipeline."""

import pytest

from src.models.trainer import (
    build_callbacks,
    combined_loss,
    compile_model,
    dice_coefficient,
    dice_loss,
    load_trained_model,
)


def _tf_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


tf_required = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")


@tf_required
class TestDiceCoefficient:
    """Tests for dice_coefficient function."""

    def test_perfect_prediction(self) -> None:
        """Dice of identical tensors should be ~1."""
        import numpy as np
        import tensorflow as tf

        y = tf.constant(np.ones((4, 4)), dtype=tf.float32)
        result = dice_coefficient(y, y).numpy()
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self) -> None:
        """Dice of non-overlapping tensors should be ~0."""
        import numpy as np
        import tensorflow as tf

        y_true = tf.constant(np.ones((4, 4)), dtype=tf.float32)
        y_pred = tf.constant(np.zeros((4, 4)), dtype=tf.float32)
        result = dice_coefficient(y_true, y_pred).numpy()
        assert result < 0.01

    def test_partial_overlap(self) -> None:
        """Dice of partially overlapping tensors between 0 and 1."""
        import numpy as np
        import tensorflow as tf

        y_true = tf.constant(np.array([[1, 1], [0, 0]]), dtype=tf.float32)
        y_pred = tf.constant(np.array([[1, 0], [0, 0]]), dtype=tf.float32)
        result = dice_coefficient(y_true, y_pred).numpy()
        assert 0.0 < result < 1.0


@tf_required
class TestDiceLoss:
    """Tests for dice_loss function."""

    def test_perfect_prediction_zero_loss(self) -> None:
        """Dice loss of identical tensors should be ~0."""
        import numpy as np
        import tensorflow as tf

        y = tf.constant(np.ones((4, 4)), dtype=tf.float32)
        result = dice_loss(y, y).numpy()
        assert result == pytest.approx(0.0, abs=1e-5)


@tf_required
class TestCombinedLoss:
    """Tests for combined_loss function."""

    def test_returns_callable(self) -> None:
        """Return a callable loss function."""
        loss_fn = combined_loss()
        assert callable(loss_fn)
        assert loss_fn.__name__ == "dice_bce_loss"

    def test_loss_value(self) -> None:
        """Loss value is a positive scalar."""
        import numpy as np
        import tensorflow as tf

        loss_fn = combined_loss(dice_weight=0.5, bce_weight=0.5)
        y_true = tf.constant(np.array([[1, 0], [0, 1]]), dtype=tf.float32)
        y_pred = tf.constant(np.array([[0.9, 0.1], [0.2, 0.8]]), dtype=tf.float32)
        result = loss_fn(y_true, y_pred).numpy()
        assert result > 0.0


@tf_required
class TestBuildCallbacks:
    """Tests for build_callbacks function."""

    def test_default_callbacks(self) -> None:
        """Build default set of callbacks."""
        callbacks = build_callbacks()
        assert len(callbacks) == 3  # EarlyStopping, ReduceLR, Checkpoint

    def test_custom_config(self) -> None:
        """Build callbacks from custom config."""
        config = {
            "training": {
                "early_stopping": {"patience": 5, "min_delta": 0.01},
                "reduce_lr": {"factor": 0.2, "patience": 3, "min_lr": 1e-7},
            },
            "paths": {"checkpoint_dir": "/tmp/test_ckpt"},
        }
        callbacks = build_callbacks(config)
        assert len(callbacks) == 3


@tf_required
class TestCompileModel:
    """Tests for compile_model function."""

    def test_compile_unet(self) -> None:
        """Compile a U-Net model with loss and metrics."""
        from src.models.unet import build_unet

        model = build_unet(
            input_shape=(32, 32, 3),
            encoder_channels=[8],
            bottleneck_channels=16,
        )
        compiled = compile_model(model)
        assert compiled.optimizer is not None


class TestLoadTrainedModel:
    """Tests for load_trained_model function."""

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing checkpoint."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_trained_model("/nonexistent/model.keras")


class TestTrainerImports:
    """Tests that work without TensorFlow."""

    def test_module_imports(self) -> None:
        """Module can be imported without TensorFlow."""
        from src.models import trainer  # noqa: F401
