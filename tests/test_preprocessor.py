"""Tests for image preprocessing and TFRecord generation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preprocessor import (
    extract_patient_id,
    load_image,
    load_mask,
    normalize_image,
    preprocess_pair,
    resize_image,
    split_by_patient,
)


def _create_test_image(
    path: Path, shape: tuple = (100, 80, 3), value: int = 128
) -> Path:
    """Create a test image file on disk."""
    img = np.full(shape, value, dtype=np.uint8)
    if len(shape) == 3:
        cv2.imwrite(str(path), img)
    else:
        cv2.imwrite(str(path), img)
    return path


def _create_test_mask(path: Path, shape: tuple = (100, 80), value: int = 255) -> Path:
    """Create a test mask file on disk."""
    mask = np.full(shape, value, dtype=np.uint8)
    cv2.imwrite(str(path), mask)
    return path


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_rgb(self) -> None:
        """Resize a 3-channel image."""
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = resize_image(img, target_size=64)
        assert result.shape == (64, 64, 3)

    def test_resize_grayscale(self) -> None:
        """Resize a grayscale image."""
        img = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        result = resize_image(img, target_size=32)
        assert result.shape == (32, 32)

    def test_resize_preserves_dtype(self) -> None:
        """Preserve dtype after resizing."""
        img = np.zeros((50, 50, 3), dtype=np.float32)
        result = resize_image(img, target_size=25)
        assert result.dtype == np.float32


class TestNormalizeImage:
    """Tests for normalize_image function."""

    def test_normalize_uint8(self) -> None:
        """Normalize uint8 image to [0, 1]."""
        img = np.array([[0, 128], [64, 255]], dtype=np.uint8)
        result = normalize_image(img)
        assert result.dtype == np.float32
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_normalize_constant(self) -> None:
        """Handle constant-value image without division by zero."""
        img = np.full((4, 4), 100, dtype=np.uint8)
        result = normalize_image(img)
        assert result.max() == pytest.approx(1.0)

    def test_normalize_zeros(self) -> None:
        """Handle all-zero image."""
        img = np.zeros((4, 4), dtype=np.uint8)
        result = normalize_image(img)
        assert np.all(result == 0.0)

    def test_already_float(self) -> None:
        """Handle float input with range [0, 1]."""
        img = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
        result = normalize_image(img)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)


class TestLoadImage:
    """Tests for load_image function."""

    def test_load_rgb(self, tmp_path: Path) -> None:
        """Load an RGB image."""
        img_path = _create_test_image(tmp_path / "test.png", (50, 40, 3))
        result = load_image(img_path)
        assert result.shape == (50, 40, 3)

    def test_load_grayscale(self, tmp_path: Path) -> None:
        """Load an image as grayscale."""
        img_path = _create_test_image(tmp_path / "test.png", (50, 40, 3))
        result = load_image(img_path, grayscale=True)
        assert result.ndim == 2

    def test_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.png")

    def test_invalid_image(self, tmp_path: Path) -> None:
        """Raise ValueError for unreadable file."""
        bad_file = tmp_path / "bad.png"
        bad_file.write_text("not an image")
        with pytest.raises(ValueError, match="Failed to read image"):
            load_image(bad_file)


class TestLoadMask:
    """Tests for load_mask function."""

    def test_load_binary_mask(self, tmp_path: Path) -> None:
        """Load and binarize a mask."""
        mask_path = _create_test_mask(tmp_path / "mask.png", (50, 40), 255)
        result = load_mask(mask_path)
        assert result.shape == (50, 40)
        assert np.all(result == 1.0)

    def test_load_zero_mask(self, tmp_path: Path) -> None:
        """Load an all-zero mask."""
        mask_path = _create_test_mask(tmp_path / "mask.png", (50, 40), 0)
        result = load_mask(mask_path)
        assert np.all(result == 0.0)

    def test_mask_file_not_found(self) -> None:
        """Raise FileNotFoundError for missing mask."""
        with pytest.raises(FileNotFoundError):
            load_mask("/nonexistent/mask.png")


class TestPreprocessPair:
    """Tests for preprocess_pair function."""

    def test_resize_and_normalize(self) -> None:
        """Resize and normalize image-mask pair."""
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (100, 80), dtype=np.uint8) * 255

        img_out, mask_out = preprocess_pair(img, mask, target_size=64)
        assert img_out.shape == (64, 64, 3)
        assert mask_out.shape == (64, 64)
        assert img_out.dtype == np.float32
        assert set(np.unique(mask_out)).issubset({0.0, 1.0})


class TestExtractPatientId:
    """Tests for extract_patient_id function."""

    def test_isic_format(self) -> None:
        """Extract ID from standard ISIC filename."""
        assert extract_patient_id("ISIC_0000001.jpg") == "0000001"

    def test_non_isic_format(self) -> None:
        """Generate hash-based ID for non-standard filenames."""
        pid = extract_patient_id("random_image_42.png")
        assert isinstance(pid, str)
        assert len(pid) == 8


class TestSplitByPatient:
    """Tests for split_by_patient function."""

    def test_basic_split(self, tmp_path: Path) -> None:
        """Split 10 patients into train/val/test."""
        images = [tmp_path / f"ISIC_{i:07d}.jpg" for i in range(10)]
        masks = [tmp_path / f"ISIC_{i:07d}_mask.png" for i in range(10)]

        splits = split_by_patient(images, masks, 0.7, 0.15, seed=42)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 10
        assert len(splits["train"]) >= 5

    def test_mismatched_lengths(self, tmp_path: Path) -> None:
        """Raise ValueError for mismatched image/mask counts."""
        images = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
        masks = [tmp_path / "a_mask.png"]
        with pytest.raises(ValueError, match="Mismatch"):
            split_by_patient(images, masks)

    def test_reproducible(self, tmp_path: Path) -> None:
        """Same seed produces same split."""
        images = [tmp_path / f"ISIC_{i:07d}.jpg" for i in range(20)]
        masks = [tmp_path / f"ISIC_{i:07d}_mask.png" for i in range(20)]

        split1 = split_by_patient(images, masks, seed=123)
        split2 = split_by_patient(images, masks, seed=123)
        assert split1["train"] == split2["train"]
        assert split1["val"] == split2["val"]

    def test_no_data_leakage(self, tmp_path: Path) -> None:
        """Ensure no patient appears in multiple splits."""
        images = [tmp_path / f"ISIC_{i:07d}.jpg" for i in range(10)]
        masks = [tmp_path / f"ISIC_{i:07d}_mask.png" for i in range(10)]

        splits = split_by_patient(images, masks)
        train_pids = {extract_patient_id(p.name) for p, _ in splits["train"]}
        val_pids = {extract_patient_id(p.name) for p, _ in splits["val"]}
        test_pids = {extract_patient_id(p.name) for p, _ in splits["test"]}

        assert train_pids.isdisjoint(val_pids)
        assert train_pids.isdisjoint(test_pids)
        assert val_pids.isdisjoint(test_pids)


class TestWriteTfrecords:
    """Tests for write_tfrecords function (mocked TF)."""

    def test_write_valid_pairs(self, tmp_path: Path) -> None:
        """Write TFRecords from valid image-mask pairs."""
        # Create test images and masks
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        pairs = []
        for i in range(3):
            img_path = images_dir / f"img_{i}.png"
            mask_path = masks_dir / f"mask_{i}.png"
            _create_test_image(img_path, (64, 64, 3))
            _create_test_mask(mask_path, (64, 64))
            pairs.append((img_path, mask_path))

        output_path = tmp_path / "output.tfrecord"

        try:
            from src.data.preprocessor import write_tfrecords

            count = write_tfrecords(pairs, output_path, target_size=32)
            assert count == 3
            assert output_path.exists()
        except ImportError:
            pytest.skip("TensorFlow not available")

    def test_write_skips_invalid(self, tmp_path: Path) -> None:
        """Skip pairs with missing files."""
        pairs = [
            (tmp_path / "missing.png", tmp_path / "missing_mask.png"),
        ]
        output_path = tmp_path / "output.tfrecord"

        try:
            from src.data.preprocessor import write_tfrecords

            count = write_tfrecords(pairs, output_path, target_size=32)
            assert count == 0
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestSerializeExample:
    """Tests for serialize_example function."""

    def test_serialize_returns_bytes(self) -> None:
        """Serialize returns a bytes object."""
        try:
            from src.data.preprocessor import serialize_example

            image = np.random.rand(32, 32, 3).astype(np.float32)
            mask = np.random.rand(32, 32).astype(np.float32)
            result = serialize_example(image, mask)
            assert isinstance(result, bytes)
            assert len(result) > 0
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestRunPreprocessingPipeline:
    """Tests for run_preprocessing_pipeline function."""

    def test_no_images_warns(self, tmp_path: Path) -> None:
        """Warn when no image-mask pairs are found."""
        config = {
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "tfrecord_dir": str(tmp_path / "tfrecords"),
            },
            "data": {"image_size": 64, "train_split": 0.7, "val_split": 0.15},
            "project": {"seed": 42},
        }
        (tmp_path / "raw" / "images").mkdir(parents=True)
        (tmp_path / "raw" / "masks").mkdir(parents=True)

        from src.data.preprocessor import run_preprocessing_pipeline

        # Should not raise, just log a warning
        run_preprocessing_pipeline(config)

    def test_pipeline_with_images(self, tmp_path: Path) -> None:
        """Run pipeline end-to-end with test images."""
        raw_dir = tmp_path / "raw"
        images_dir = raw_dir / "images"
        masks_dir = raw_dir / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        for i in range(5):
            _create_test_image(images_dir / f"ISIC_{i:07d}.jpg", (64, 64, 3))
            _create_test_mask(masks_dir / f"ISIC_{i:07d}_mask.png", (64, 64))

        config = {
            "paths": {
                "raw_dir": str(raw_dir),
                "tfrecord_dir": str(tmp_path / "tfrecords"),
            },
            "data": {"image_size": 32, "train_split": 0.6, "val_split": 0.2},
            "project": {"seed": 42},
        }

        try:
            from src.data.preprocessor import run_preprocessing_pipeline

            run_preprocessing_pipeline(config)
            tfrecord_dir = tmp_path / "tfrecords"
            assert tfrecord_dir.exists()
        except ImportError:
            pytest.skip("TensorFlow not available")
