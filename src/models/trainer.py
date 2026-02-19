"""Training pipeline for the U-Net segmentation model.

Includes combined Dice+BCE loss, training/validation loops with
logging, ReduceLROnPlateau scheduler, early stopping, and model
checkpointing.
"""

import logging
from pathlib import Path
from typing import Any


from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def dice_coefficient(y_true: Any, y_pred: Any) -> Any:
    """Compute the Dice coefficient between predictions and ground truth.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.

    Returns:
        Scalar Dice coefficient tensor.
    """
    import tensorflow as tf

    smooth = 1e-7
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true: Any, y_pred: Any) -> Any:
    """Compute Dice loss (1 - Dice coefficient).

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.

    Returns:
        Scalar Dice loss tensor.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(dice_weight: float = 0.5, bce_weight: float = 0.5) -> Any:
    """Create a combined Dice + Binary Cross-Entropy loss function.

    Args:
        dice_weight: Weight for the Dice loss component.
        bce_weight: Weight for the BCE loss component.

    Returns:
        A loss function that computes the weighted sum.
    """
    import tensorflow as tf

    def loss_fn(y_true: Any, y_pred: Any) -> Any:
        """Compute weighted Dice + BCE loss."""
        d_loss = dice_loss(y_true, y_pred)
        b_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        b_loss = tf.reduce_mean(b_loss)
        return dice_weight * d_loss + bce_weight * b_loss

    loss_fn.__name__ = "dice_bce_loss"
    return loss_fn


def build_callbacks(config: dict[str, Any] | None = None) -> list[Any]:
    """Build training callbacks from configuration.

    Args:
        config: Application configuration. If None, loads default.

    Returns:
        List of Keras callback instances.
    """
    import tensorflow as tf

    if config is None:
        config = get_config()

    callbacks = []

    # Early stopping
    es_config = get_nested(config, "training", "early_stopping", default={})
    es_patience = es_config.get("patience", 15)
    es_min_delta = es_config.get("min_delta", 0.001)
    es_monitor = es_config.get("monitor", "val_dice_coefficient")

    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            patience=es_patience,
            min_delta=es_min_delta,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        )
    )

    # ReduceLROnPlateau
    lr_config = get_nested(config, "training", "reduce_lr", default={})
    lr_factor = lr_config.get("factor", 0.5)
    lr_patience = lr_config.get("patience", 7)
    lr_min = lr_config.get("min_lr", 1e-6)
    lr_monitor = lr_config.get("monitor", "val_dice_coefficient")

    callbacks.append(
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=lr_monitor,
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            mode="max",
            verbose=1,
        )
    )

    # Model checkpoint
    checkpoint_dir = Path(
        get_nested(config, "paths", "checkpoint_dir", default="models/checkpoints")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.keras"

    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
    )

    logger.info("Built %d training callbacks", len(callbacks))
    return callbacks


def compile_model(
    model: Any,
    config: dict[str, Any] | None = None,
) -> Any:
    """Compile the model with optimizer and loss function.

    Args:
        model: Keras model to compile.
        config: Application configuration. If None, loads default.

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf

    if config is None:
        config = get_config()

    lr = get_nested(config, "training", "learning_rate", default=0.001)
    loss_weights = get_nested(config, "training", "loss_weights", default={})
    d_weight = loss_weights.get("dice", 0.5)
    b_weight = loss_weights.get("bce", 0.5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = combined_loss(dice_weight=d_weight, bce_weight=b_weight)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[dice_coefficient],
    )

    logger.info(
        "Compiled model: lr=%.6f, dice_weight=%.2f, bce_weight=%.2f",
        lr,
        d_weight,
        b_weight,
    )
    return model


def train_model(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    config: dict[str, Any] | None = None,
) -> Any:
    """Train the model with the specified datasets and configuration.

    Args:
        model: Compiled Keras model.
        train_dataset: Training tf.data.Dataset.
        val_dataset: Validation tf.data.Dataset.
        config: Application configuration. If None, loads default.

    Returns:
        Keras History object with training metrics.
    """
    if config is None:
        config = get_config()

    epochs = get_nested(config, "training", "epochs", default=100)
    callbacks = build_callbacks(config)

    logger.info("Starting training for %d epochs", epochs)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info(
        "Training complete. Final val_dice: %.4f",
        history.history.get("val_dice_coefficient", [0])[-1],
    )
    return history


def load_trained_model(checkpoint_path: str | Path) -> Any:
    """Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the model checkpoint.

    Returns:
        Loaded Keras model with custom objects.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    import tensorflow as tf

    custom_objects = {
        "dice_coefficient": dice_coefficient,
        "dice_bce_loss": combined_loss(),
    }

    model = tf.keras.models.load_model(
        str(checkpoint_path), custom_objects=custom_objects
    )
    logger.info("Loaded model from %s", checkpoint_path)
    return model
