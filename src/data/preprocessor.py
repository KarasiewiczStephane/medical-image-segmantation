"""Image preprocessing and TFRecord generation for the segmentation pipeline.

Handles image resizing, normalization, patient-ID-based train/val/test
splitting, and TFRecord serialization for efficient data loading.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def resize_image(
    image: np.ndarray, target_size: int = 256, interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """Resize an image to target_size x target_size.

    Args:
        image: Input image array of shape (H, W) or (H, W, C).
        target_size: Target height and width in pixels.
        interpolation: OpenCV interpolation method.

    Returns:
        Resized image array of shape (target_size, target_size, ...).
    """
    resized = cv2.resize(image, (target_size, target_size), interpolation=interpolation)
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to the range [0, 1].

    Args:
        image: Input image array with any numeric dtype.

    Returns:
        Float32 array with values in [0, 1].
    """
    image = image.astype(np.float32)
    pmin, pmax = image.min(), image.max()
    if pmax > pmin:
        image = (image - pmin) / (pmax - pmin)
    elif pmax > 0:
        image = image / pmax
    return image


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Load an image from disk using OpenCV.

    Args:
        path: Path to the image file.
        grayscale: If True, load as single-channel grayscale.

    Returns:
        Image array in RGB format (H, W, 3) or grayscale (H, W).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image could not be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")

    if not grayscale and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_mask(path: str | Path) -> np.ndarray:
    """Load a segmentation mask from disk.

    Args:
        path: Path to the mask image file.

    Returns:
        Binary mask array of shape (H, W) with values in {0, 1}.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the mask could not be read.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {path}")

    mask = (mask > 127).astype(np.float32)
    return mask


def preprocess_pair(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess an image-mask pair: resize and normalize.

    Args:
        image: Input image array.
        mask: Input binary mask array.
        target_size: Target spatial dimensions.

    Returns:
        Tuple of (preprocessed_image, preprocessed_mask).
    """
    image = resize_image(image, target_size, cv2.INTER_LINEAR)
    mask = resize_image(mask, target_size, cv2.INTER_NEAREST)

    image = normalize_image(image)
    mask = (mask > 0.5).astype(np.float32)

    return image, mask


def extract_patient_id(filename: str) -> str:
    """Extract a patient identifier from a filename for stratified splitting.

    Uses the ISIC image ID as a proxy for patient ID. For files without
    a clear patient identifier, a hash-based ID is generated.

    Args:
        filename: Name of the image file (e.g., 'ISIC_0000001.jpg').

    Returns:
        Patient identifier string.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "ISIC":
        return parts[1]
    return hashlib.md5(stem.encode()).hexdigest()[:8]


def split_by_patient(
    image_paths: list[Path],
    mask_paths: list[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Split image-mask pairs by patient ID to prevent data leakage.

    Args:
        image_paths: List of image file paths.
        mask_paths: List of corresponding mask file paths.
        train_ratio: Fraction of patients for training.
        val_ratio: Fraction of patients for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each mapping to
        a list of (image_path, mask_path) tuples.
    """
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
        )

    patient_map: dict[str, list[tuple[Path, Path]]] = {}
    for img, msk in zip(image_paths, mask_paths):
        pid = extract_patient_id(img.name)
        patient_map.setdefault(pid, []).append((img, msk))

    patient_ids = sorted(patient_map.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits: dict[str, list[tuple[Path, Path]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for pid in patient_ids[:train_end]:
        splits["train"].extend(patient_map[pid])
    for pid in patient_ids[train_end:val_end]:
        splits["val"].extend(patient_map[pid])
    for pid in patient_ids[val_end:]:
        splits["test"].extend(patient_map[pid])

    logger.info(
        "Split: train=%d, val=%d, test=%d (from %d patients)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        n,
    )
    return splits


def _bytes_feature(value: bytes) -> Any:
    """Create a TF bytes feature from raw bytes."""
    import tensorflow as tf

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> Any:
    """Create a TF int64 feature from an integer."""
    import tensorflow as tf

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image: np.ndarray, mask: np.ndarray) -> bytes:
    """Serialize an image-mask pair as a TFRecord example.

    Args:
        image: Preprocessed image array of shape (H, W, C).
        mask: Preprocessed mask array of shape (H, W).

    Returns:
        Serialized bytes representing the TFRecord example.
    """
    import tensorflow as tf

    feature = {
        "image": _bytes_feature(image.tobytes()),
        "mask": _bytes_feature(mask.tobytes()),
        "height": _int64_feature(image.shape[0]),
        "width": _int64_feature(image.shape[1]),
        "channels": _int64_feature(image.shape[2] if image.ndim == 3 else 1),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecords(
    pairs: list[tuple[Path, Path]],
    output_path: str | Path,
    target_size: int = 256,
) -> int:
    """Write image-mask pairs to a TFRecord file.

    Args:
        pairs: List of (image_path, mask_path) tuples.
        output_path: Path for the output TFRecord file.
        target_size: Target image dimensions.

    Returns:
        Number of examples written.
    """
    import tensorflow as tf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for img_path, msk_path in pairs:
            try:
                image = load_image(img_path)
                mask = load_mask(msk_path)
                image, mask = preprocess_pair(image, mask, target_size)
                serialized = serialize_example(image, mask)
                writer.write(serialized)
                count += 1
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Skipping %s: %s", img_path, e)

    logger.info("Wrote %d examples to %s", count, output_path)
    return count


def parse_tfrecord(serialized: bytes, image_size: int = 256, channels: int = 3) -> Any:
    """Parse a single TFRecord example back into image and mask tensors.

    Args:
        serialized: Serialized TFRecord example bytes.
        image_size: Expected spatial dimension of the image.
        channels: Number of image channels.

    Returns:
        Tuple of (image_tensor, mask_tensor).
    """
    import tensorflow as tf

    feature_desc = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(serialized, feature_desc)
    image = tf.io.decode_raw(parsed["image"], tf.float32)
    image = tf.reshape(image, [image_size, image_size, channels])

    mask = tf.io.decode_raw(parsed["mask"], tf.float32)
    mask = tf.reshape(mask, [image_size, image_size, 1])

    return image, mask


def load_tfrecord_dataset(
    tfrecord_path: str | Path,
    image_size: int = 256,
    channels: int = 3,
    batch_size: int = 16,
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
) -> Any:
    """Load a TFRecord file as a tf.data.Dataset.

    Args:
        tfrecord_path: Path to the TFRecord file.
        image_size: Expected spatial dimension.
        channels: Number of image channels.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        shuffle_buffer: Size of the shuffle buffer.

    Returns:
        A batched tf.data.Dataset yielding (image, mask) pairs.
    """
    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    def _parse(x: Any) -> Any:
        return parse_tfrecord(x, image_size, channels)

    dataset = dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def run_preprocessing_pipeline(config: dict[str, Any] | None = None) -> None:
    """Run the full preprocessing pipeline: load, split, write TFRecords.

    Args:
        config: Application configuration. If None, loads default.
    """
    if config is None:
        config = get_config()

    raw_dir = Path(get_nested(config, "paths", "raw_dir", default="data/raw"))
    tfrecord_dir = Path(
        get_nested(config, "paths", "tfrecord_dir", default="data/tfrecords")
    )
    target_size = get_nested(config, "data", "image_size", default=256)
    train_ratio = get_nested(config, "data", "train_split", default=0.70)
    val_ratio = get_nested(config, "data", "val_split", default=0.15)
    seed = get_nested(config, "project", "seed", default=42)

    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    mask_paths = []
    for img_path in image_paths:
        stem = img_path.stem
        mask_name = f"{stem}_mask.png"
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            mask_paths.append(mask_path)
        else:
            image_paths.remove(img_path)
            logger.warning("No mask found for %s, skipping", img_path.name)

    if not image_paths:
        logger.warning("No image-mask pairs found in %s", raw_dir)
        return

    splits = split_by_patient(image_paths, mask_paths, train_ratio, val_ratio, seed)

    for split_name, pairs in splits.items():
        if pairs:
            output_path = tfrecord_dir / f"{split_name}.tfrecord"
            write_tfrecords(pairs, output_path, target_size)
