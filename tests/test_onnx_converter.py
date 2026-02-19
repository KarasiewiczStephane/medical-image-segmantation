"""Tests for ONNX model export and benchmarking."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.export.onnx_converter import (
    compare_model_sizes,
    load_onnx_model,
    onnx_predict,
)


def _tf_available() -> bool:
    """Check if TensorFlow is available."""
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False


def _onnxruntime_available() -> bool:
    """Check if ONNX Runtime is available."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


tf_required = pytest.mark.skipif(not _tf_available(), reason="TensorFlow not available")
ort_required = pytest.mark.skipif(
    not _onnxruntime_available(), reason="ONNX Runtime not available"
)


class TestLoadOnnxModel:
    """Tests for load_onnx_model function."""

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            load_onnx_model("/nonexistent/model.onnx")


class TestOnnxPredict:
    """Tests for onnx_predict function."""

    def test_predict_with_mock_session(self) -> None:
        """Run prediction with a mocked ONNX session."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [
            np.random.rand(1, 32, 32, 1).astype(np.float32)
        ]

        image = np.random.rand(1, 32, 32, 3).astype(np.float32)
        result = onnx_predict(mock_session, image)
        assert result.shape == (1, 32, 32, 1)

    def test_predict_3d_input(self) -> None:
        """Handle 3D input by adding batch dimension."""
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [
            np.random.rand(1, 32, 32, 1).astype(np.float32)
        ]

        image = np.random.rand(32, 32, 3).astype(np.float32)
        result = onnx_predict(mock_session, image)
        assert result.shape == (1, 32, 32, 1)


class TestCompareModelSizes:
    """Tests for compare_model_sizes function."""

    def test_compare_files(self, tmp_path: Path) -> None:
        """Compare sizes of two model files."""
        tf_file = tmp_path / "model.h5"
        onnx_file = tmp_path / "model.onnx"
        tf_file.write_bytes(b"x" * 1000)
        onnx_file.write_bytes(b"x" * 500)

        result = compare_model_sizes(tf_file, onnx_file)
        assert result["tf_size_mb"] > 0
        assert result["onnx_size_mb"] > 0
        assert result["compression_ratio"] > 1.0

    def test_compare_directory(self, tmp_path: Path) -> None:
        """Compare directory-based TF model with ONNX file."""
        tf_dir = tmp_path / "saved_model"
        tf_dir.mkdir()
        (tf_dir / "model.pb").write_bytes(b"x" * 2000)
        (tf_dir / "variables.data").write_bytes(b"x" * 3000)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"x" * 1000)

        result = compare_model_sizes(tf_dir, onnx_file)
        assert result["tf_size_mb"] > result["onnx_size_mb"]

    def test_missing_files(self, tmp_path: Path) -> None:
        """Handle missing files gracefully."""
        result = compare_model_sizes(
            tmp_path / "missing_tf",
            tmp_path / "missing.onnx",
        )
        assert result["tf_size_mb"] == 0.0
        assert result["onnx_size_mb"] == 0.0


class TestOnnxConverterImports:
    """Tests that work without TF or ONNX Runtime."""

    def test_module_imports(self) -> None:
        """Module can be imported without TF or ONNX Runtime."""
        from src.export import onnx_converter  # noqa: F401
