"""ONNX model export, benchmarking, and quantization.

Converts trained TensorFlow/Keras models to ONNX format using tf2onnx,
performs inference benchmarking, model size comparison, and optional
INT8 quantization.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: Any,
    output_path: str | Path,
    opset_version: int = 13,
    input_shape: tuple[int, ...] | None = None,
) -> Path:
    """Export a Keras model to ONNX format.

    Args:
        model: Trained Keras model.
        output_path: Path for the output ONNX file.
        opset_version: ONNX opset version.
        input_shape: Input shape including batch dimension.
            If None, inferred from model.

    Returns:
        Path to the saved ONNX file.
    """
    import tf2onnx
    import tensorflow as tf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_shape is None:
        model_input = model.input_shape
        input_shape = (1,) + tuple(d if d is not None else 1 for d in model_input[1:])

    input_spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]

    logger.info("Exporting model to ONNX (opset=%d): %s", opset_version, output_path)

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_spec,
        opset=opset_version,
        output_path=str(output_path),
    )

    logger.info("ONNX export complete: %s", output_path)
    return output_path


def load_onnx_model(model_path: str | Path) -> Any:
    """Load an ONNX model for inference.

    Args:
        model_path: Path to the ONNX model file.

    Returns:
        ONNX Runtime InferenceSession.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    logger.info("Loaded ONNX model: %s", model_path)
    return session


def onnx_predict(
    session: Any,
    image: np.ndarray,
) -> np.ndarray:
    """Run inference with an ONNX model.

    Args:
        session: ONNX Runtime InferenceSession.
        image: Input image, shape (1, H, W, C) or (H, W, C).

    Returns:
        Model predictions as numpy array.
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    image = image.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: image})
    return result[0]


def benchmark_inference(
    model: Any,
    onnx_session: Any,
    input_shape: tuple[int, ...],
    n_runs: int = 50,
    warmup: int = 5,
) -> dict[str, dict[str, float]]:
    """Benchmark inference speed for TensorFlow vs ONNX.

    Args:
        model: Keras model.
        onnx_session: ONNX Runtime InferenceSession.
        input_shape: Shape of test input (without batch dim).
        n_runs: Number of inference runs for timing.
        warmup: Number of warmup runs to exclude.

    Returns:
        Dictionary with 'tensorflow' and 'onnx' timing stats.
    """
    dummy_input = np.random.rand(1, *input_shape).astype(np.float32)

    # Warmup TF
    for _ in range(warmup):
        model.predict(dummy_input, verbose=0)

    # Benchmark TF
    tf_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy_input, verbose=0)
        tf_times.append(time.perf_counter() - start)

    # Warmup ONNX
    for _ in range(warmup):
        onnx_predict(onnx_session, dummy_input)

    # Benchmark ONNX
    onnx_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        onnx_predict(onnx_session, dummy_input)
        onnx_times.append(time.perf_counter() - start)

    results = {
        "tensorflow": {
            "mean_ms": float(np.mean(tf_times) * 1000),
            "std_ms": float(np.std(tf_times) * 1000),
            "min_ms": float(np.min(tf_times) * 1000),
            "max_ms": float(np.max(tf_times) * 1000),
        },
        "onnx": {
            "mean_ms": float(np.mean(onnx_times) * 1000),
            "std_ms": float(np.std(onnx_times) * 1000),
            "min_ms": float(np.min(onnx_times) * 1000),
            "max_ms": float(np.max(onnx_times) * 1000),
        },
    }

    speedup = results["tensorflow"]["mean_ms"] / max(results["onnx"]["mean_ms"], 1e-6)
    logger.info(
        "Benchmark: TF=%.2fms, ONNX=%.2fms, speedup=%.2fx",
        results["tensorflow"]["mean_ms"],
        results["onnx"]["mean_ms"],
        speedup,
    )
    return results


def compare_model_sizes(
    tf_model_path: str | Path,
    onnx_model_path: str | Path,
) -> dict[str, float]:
    """Compare file sizes of TensorFlow and ONNX models.

    Args:
        tf_model_path: Path to the TensorFlow saved model directory or file.
        onnx_model_path: Path to the ONNX model file.

    Returns:
        Dictionary with sizes in MB and compression ratio.
    """
    tf_path = Path(tf_model_path)
    onnx_path = Path(onnx_model_path)

    if tf_path.is_dir():
        tf_size = sum(f.stat().st_size for f in tf_path.rglob("*") if f.is_file())
    elif tf_path.exists():
        tf_size = tf_path.stat().st_size
    else:
        tf_size = 0

    onnx_size = onnx_path.stat().st_size if onnx_path.exists() else 0

    tf_mb = tf_size / (1024 * 1024)
    onnx_mb = onnx_size / (1024 * 1024)
    ratio = tf_mb / max(onnx_mb, 1e-6)

    logger.info(
        "Model sizes: TF=%.2fMB, ONNX=%.2fMB, ratio=%.2f",
        tf_mb,
        onnx_mb,
        ratio,
    )
    return {
        "tf_size_mb": tf_mb,
        "onnx_size_mb": onnx_mb,
        "compression_ratio": ratio,
    }


def export_from_config(
    model: Any,
    config: dict[str, Any] | None = None,
) -> Path:
    """Export model to ONNX using configuration settings.

    Args:
        model: Trained Keras model.
        config: Application configuration. If None, loads default.

    Returns:
        Path to the exported ONNX file.
    """
    if config is None:
        config = get_config()

    export_dir = Path(
        get_nested(config, "paths", "export_dir", default="models/export")
    )
    opset = get_nested(config, "export", "onnx", "opset_version", default=13)

    output_path = export_dir / "model.onnx"
    return export_to_onnx(model, output_path, opset_version=opset)
