"""Data augmentation pipeline using albumentations.

Provides configurable augmentation transforms for medical image
segmentation, applying identical spatial transforms to both
images and their corresponding segmentation masks.
"""

import logging
from typing import Any

import albumentations as A
import numpy as np

from src.utils.config import get_config

logger = logging.getLogger(__name__)


def build_augmentation_pipeline(config: dict[str, Any] | None = None) -> A.Compose:
    """Build an augmentation pipeline from configuration.

    All spatial transforms are applied identically to both the image
    and the mask to maintain correspondence.

    Args:
        config: Application configuration dictionary. If None, loads default.

    Returns:
        An albumentations Compose pipeline.
    """
    if config is None:
        config = get_config()

    aug_config = config.get("augmentation", {})

    if not aug_config.get("enabled", True):
        logger.info("Augmentation disabled, returning identity transform")
        return A.Compose([])

    rotation_limit = aug_config.get("rotation_limit", 30)
    h_flip = aug_config.get("horizontal_flip", True)
    v_flip = aug_config.get("vertical_flip", True)

    elastic_cfg = aug_config.get("elastic_transform", {})
    elastic_alpha = elastic_cfg.get("alpha", 120)
    elastic_sigma = elastic_cfg.get("sigma", 6)

    grid_cfg = aug_config.get("grid_distortion", {})
    grid_steps = grid_cfg.get("num_steps", 5)
    grid_limit = grid_cfg.get("distort_limit", 0.3)

    bc_cfg = aug_config.get("brightness_contrast", {})
    brightness_limit = bc_cfg.get("brightness_limit", 0.2)
    contrast_limit = bc_cfg.get("contrast_limit", 0.2)

    crop_cfg = aug_config.get("random_crop", {})
    min_scale = crop_cfg.get("min_scale", 0.8)
    max_scale = crop_cfg.get("max_scale", 1.0)

    transforms = [
        A.Rotate(limit=rotation_limit, border_mode=0, p=0.5),
    ]

    if h_flip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if v_flip:
        transforms.append(A.VerticalFlip(p=0.5))

    transforms.extend(
        [
            A.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, p=0.3),
            A.GridDistortion(num_steps=grid_steps, distort_limit=grid_limit, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(256, 256),
                scale=(min_scale, max_scale),
                ratio=(0.9, 1.1),
                p=0.3,
            ),
        ]
    )

    pipeline = A.Compose(transforms)
    logger.info("Built augmentation pipeline with %d transforms", len(transforms))
    return pipeline


def build_validation_pipeline() -> A.Compose:
    """Build a minimal pipeline for validation/test data (no augmentation).

    Returns:
        An albumentations Compose pipeline with no transforms.
    """
    return A.Compose([])


def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    pipeline: A.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an augmentation pipeline to an image-mask pair.

    Args:
        image: Input image array of shape (H, W, C), float32 in [0, 1].
        mask: Input mask array of shape (H, W), float32 with values {0, 1}.
        pipeline: Albumentations Compose pipeline.

    Returns:
        Tuple of (augmented_image, augmented_mask).
    """
    if image.dtype != np.uint8:
        image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image_uint8 = image

    if mask.dtype != np.uint8:
        mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    else:
        mask_uint8 = mask

    augmented = pipeline(image=image_uint8, mask=mask_uint8)
    aug_image = augmented["image"].astype(np.float32) / 255.0
    aug_mask = (augmented["mask"] > 127).astype(np.float32)

    return aug_image, aug_mask


def augment_batch(
    images: np.ndarray,
    masks: np.ndarray,
    pipeline: A.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation to a batch of image-mask pairs.

    Args:
        images: Batch of images, shape (N, H, W, C).
        masks: Batch of masks, shape (N, H, W).
        pipeline: Albumentations Compose pipeline.

    Returns:
        Tuple of (augmented_images, augmented_masks) with same shapes.
    """
    aug_images = []
    aug_masks = []
    for i in range(len(images)):
        aug_img, aug_msk = apply_augmentation(images[i], masks[i], pipeline)
        aug_images.append(aug_img)
        aug_masks.append(aug_msk)

    return np.stack(aug_images), np.stack(aug_masks)
