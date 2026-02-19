"""U-Net architecture for medical image segmentation.

Implements the standard U-Net encoder-decoder architecture with
skip connections, batch normalization, and MC Dropout support
for uncertainty estimation.
"""

import logging
from typing import Any

import numpy as np

from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def _conv_block(
    x: Any,
    filters: int,
    activation: str = "relu",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
) -> Any:
    """Apply two Conv2D-BN-ReLU layers with optional dropout.

    Args:
        x: Input tensor.
        filters: Number of convolution filters.
        activation: Activation function name.
        use_batch_norm: Whether to apply batch normalization.
        dropout_rate: Dropout rate (0 to disable).

    Returns:
        Output tensor after the convolutional block.
    """
    from tensorflow import keras

    for _ in range(2):
        x = keras.layers.Conv2D(
            filters, (3, 3), padding="same", kernel_initializer="he_normal"
        )(x)
        if use_batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation)(x)

    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate)(x, training=True)

    return x


def _encoder_block(
    x: Any,
    filters: int,
    activation: str = "relu",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
) -> tuple[Any, Any]:
    """Apply a convolution block followed by max pooling.

    Args:
        x: Input tensor.
        filters: Number of convolution filters.
        activation: Activation function name.
        use_batch_norm: Whether to apply batch normalization.
        dropout_rate: Dropout rate for MC Dropout.

    Returns:
        Tuple of (skip_connection, pooled_output).
    """
    from tensorflow import keras

    skip = _conv_block(x, filters, activation, use_batch_norm, dropout_rate)
    pooled = keras.layers.MaxPooling2D((2, 2))(skip)
    return skip, pooled


def _decoder_block(
    x: Any,
    skip: Any,
    filters: int,
    activation: str = "relu",
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
) -> Any:
    """Apply up-convolution, concatenate skip connection, and conv block.

    Args:
        x: Input tensor from the previous decoder layer.
        skip: Skip connection tensor from the encoder.
        filters: Number of convolution filters.
        activation: Activation function name.
        use_batch_norm: Whether to apply batch normalization.
        dropout_rate: Dropout rate for MC Dropout.

    Returns:
        Output tensor after the decoder block.
    """
    from tensorflow import keras

    x = keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)
    x = keras.layers.Concatenate()([x, skip])
    x = _conv_block(x, filters, activation, use_batch_norm, dropout_rate)
    return x


def build_unet(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 1,
    encoder_channels: list[int] | None = None,
    bottleneck_channels: int = 1024,
    dropout_rate: float = 0.5,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    use_batch_norm: bool = True,
) -> Any:
    """Build a U-Net model with the specified architecture.

    Args:
        input_shape: Shape of input images (H, W, C).
        num_classes: Number of output segmentation classes.
        encoder_channels: List of filter counts for encoder blocks.
            Defaults to [64, 128, 256, 512].
        bottleneck_channels: Filter count for the bottleneck layer.
        dropout_rate: Dropout rate for MC Dropout layers.
        activation: Activation function for intermediate layers.
        output_activation: Activation function for the output layer.
        use_batch_norm: Whether to use batch normalization.

    Returns:
        A compiled Keras Model.
    """
    from tensorflow import keras

    if encoder_channels is None:
        encoder_channels = [64, 128, 256, 512]

    inputs = keras.layers.Input(shape=input_shape)

    # Encoder path
    skips = []
    x = inputs
    for filters in encoder_channels:
        skip, x = _encoder_block(
            x, filters, activation, use_batch_norm, dropout_rate=0.0
        )
        skips.append(skip)

    # Bottleneck
    x = _conv_block(x, bottleneck_channels, activation, use_batch_norm, dropout_rate)

    # Decoder path
    for filters, skip in zip(reversed(encoder_channels), reversed(skips)):
        x = _decoder_block(x, skip, filters, activation, use_batch_norm, dropout_rate)

    # Output layer
    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="unet")

    logger.info(
        "Built U-Net: input=%s, encoder=%s, bottleneck=%d, output=%s",
        input_shape,
        encoder_channels,
        bottleneck_channels,
        output_activation,
    )
    return model


def build_unet_from_config(config: dict[str, Any] | None = None) -> Any:
    """Build a U-Net model from configuration.

    Args:
        config: Application configuration. If None, loads default.

    Returns:
        A Keras Model built from config parameters.
    """
    if config is None:
        config = get_config()

    image_size = get_nested(config, "data", "image_size", default=256)
    num_channels = get_nested(config, "data", "num_channels", default=3)
    num_classes = get_nested(config, "data", "num_classes", default=1)

    encoder_channels = get_nested(
        config, "model", "encoder_channels", default=[64, 128, 256, 512]
    )
    bottleneck_channels = get_nested(
        config, "model", "bottleneck_channels", default=1024
    )
    dropout_rate = get_nested(config, "model", "dropout_rate", default=0.5)
    activation = get_nested(config, "model", "activation", default="relu")
    output_activation = get_nested(
        config, "model", "output_activation", default="sigmoid"
    )
    use_batch_norm = get_nested(config, "model", "use_batch_norm", default=True)

    return build_unet(
        input_shape=(image_size, image_size, num_channels),
        num_classes=num_classes,
        encoder_channels=encoder_channels,
        bottleneck_channels=bottleneck_channels,
        dropout_rate=dropout_rate,
        activation=activation,
        output_activation=output_activation,
        use_batch_norm=use_batch_norm,
    )


def get_model_summary(model: Any) -> str:
    """Get a string summary of a Keras model.

    Args:
        model: A Keras Model.

    Returns:
        String containing the model summary.
    """
    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)


def count_parameters(model: Any) -> dict[str, int]:
    """Count trainable and non-trainable parameters.

    Args:
        model: A Keras Model.

    Returns:
        Dictionary with keys 'trainable', 'non_trainable', and 'total'.
    """
    trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
