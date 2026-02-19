"""Tests for the data augmentation pipeline."""

import albumentations as A
import numpy as np

from src.data.augmentation import (
    apply_augmentation,
    augment_batch,
    build_augmentation_pipeline,
    build_validation_pipeline,
)


def _random_image(h: int = 64, w: int = 64, c: int = 3) -> np.ndarray:
    """Create a random float32 image in [0, 1]."""
    return np.random.rand(h, w, c).astype(np.float32)


def _random_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a random binary mask."""
    return np.random.randint(0, 2, (h, w)).astype(np.float32)


class TestBuildAugmentationPipeline:
    """Tests for build_augmentation_pipeline function."""

    def test_returns_compose(self) -> None:
        """Return an albumentations Compose object."""
        pipeline = build_augmentation_pipeline()
        assert isinstance(pipeline, A.Compose)

    def test_has_transforms(self) -> None:
        """Pipeline contains multiple transforms."""
        pipeline = build_augmentation_pipeline()
        assert len(pipeline.transforms) > 0

    def test_custom_config(self) -> None:
        """Build pipeline from custom config values."""
        config = {
            "augmentation": {
                "enabled": True,
                "rotation_limit": 15,
                "horizontal_flip": True,
                "vertical_flip": False,
                "elastic_transform": {"alpha": 50, "sigma": 3},
                "grid_distortion": {"num_steps": 3, "distort_limit": 0.2},
                "brightness_contrast": {
                    "brightness_limit": 0.1,
                    "contrast_limit": 0.1,
                },
                "random_crop": {"min_scale": 0.9, "max_scale": 1.0},
            }
        }
        pipeline = build_augmentation_pipeline(config)
        assert isinstance(pipeline, A.Compose)

    def test_disabled_augmentation(self) -> None:
        """Return empty pipeline when augmentation is disabled."""
        config = {"augmentation": {"enabled": False}}
        pipeline = build_augmentation_pipeline(config)
        assert len(pipeline.transforms) == 0


class TestBuildValidationPipeline:
    """Tests for build_validation_pipeline function."""

    def test_returns_empty_compose(self) -> None:
        """Return a Compose with no transforms."""
        pipeline = build_validation_pipeline()
        assert isinstance(pipeline, A.Compose)
        assert len(pipeline.transforms) == 0


class TestApplyAugmentation:
    """Tests for apply_augmentation function."""

    def test_output_shapes(self) -> None:
        """Output shapes match input shapes."""
        image = _random_image(64, 64, 3)
        mask = _random_mask(64, 64)
        pipeline = build_validation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        assert aug_img.shape == image.shape
        assert aug_msk.shape == mask.shape

    def test_output_dtypes(self) -> None:
        """Outputs are float32."""
        image = _random_image()
        mask = _random_mask()
        pipeline = build_validation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        assert aug_img.dtype == np.float32
        assert aug_msk.dtype == np.float32

    def test_image_range(self) -> None:
        """Augmented image values stay in [0, 1]."""
        image = _random_image()
        mask = _random_mask()
        pipeline = build_augmentation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        assert aug_img.min() >= 0.0
        assert aug_img.max() <= 1.0

    def test_mask_binary(self) -> None:
        """Augmented mask values are binary {0, 1}."""
        image = _random_image()
        mask = _random_mask()
        pipeline = build_augmentation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        unique_vals = set(np.unique(aug_msk))
        assert unique_vals.issubset({0.0, 1.0})

    def test_uint8_input(self) -> None:
        """Handle uint8 input without errors."""
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        pipeline = build_validation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        assert aug_img.dtype == np.float32

    def test_identity_pipeline_preserves_content(self) -> None:
        """Identity (empty) pipeline should preserve the image content."""
        image = _random_image(32, 32, 3)
        mask = _random_mask(32, 32)
        pipeline = build_validation_pipeline()

        aug_img, aug_msk = apply_augmentation(image, mask, pipeline)
        np.testing.assert_allclose(aug_img, image, atol=2.0 / 255.0)


class TestAugmentBatch:
    """Tests for augment_batch function."""

    def test_batch_shapes(self) -> None:
        """Output batch shapes match input batch shapes."""
        images = np.random.rand(4, 64, 64, 3).astype(np.float32)
        masks = np.random.randint(0, 2, (4, 64, 64)).astype(np.float32)
        pipeline = build_validation_pipeline()

        aug_imgs, aug_msks = augment_batch(images, masks, pipeline)
        assert aug_imgs.shape == images.shape
        assert aug_msks.shape == masks.shape

    def test_batch_dtypes(self) -> None:
        """Batch outputs are float32."""
        images = np.random.rand(2, 32, 32, 3).astype(np.float32)
        masks = np.random.randint(0, 2, (2, 32, 32)).astype(np.float32)
        pipeline = build_validation_pipeline()

        aug_imgs, aug_msks = augment_batch(images, masks, pipeline)
        assert aug_imgs.dtype == np.float32
        assert aug_msks.dtype == np.float32

    def test_single_item_batch(self) -> None:
        """Handle batch of size 1."""
        images = np.random.rand(1, 64, 64, 3).astype(np.float32)
        masks = np.random.randint(0, 2, (1, 64, 64)).astype(np.float32)
        pipeline = build_augmentation_pipeline()

        aug_imgs, aug_msks = augment_batch(images, masks, pipeline)
        assert aug_imgs.shape[0] == 1
