"""ISIC Skin Lesion dataset downloader.

Downloads images and segmentation masks from the ISIC Archive API
and saves them to the configured data directory.
"""

import logging
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from src.utils.config import get_config, get_nested

logger = logging.getLogger(__name__)


def get_image_list(
    api_base_url: str, limit: int = 100, offset: int = 0
) -> list[dict[str, Any]]:
    """Fetch a list of image metadata from the ISIC Archive API.

    Args:
        api_base_url: Base URL of the ISIC Archive API.
        limit: Maximum number of images to retrieve.
        offset: Number of images to skip from the start.

    Returns:
        List of image metadata dictionaries.

    Raises:
        requests.HTTPError: If the API request fails.
    """
    url = f"{api_base_url}/images"
    params = {"limit": limit, "offset": offset, "sort": "name"}
    logger.info("Fetching image list from %s (limit=%d, offset=%d)", url, limit, offset)

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    images = response.json()

    logger.info("Retrieved %d image records", len(images))
    return images


def download_file(url: str, dest_path: Path, timeout: int = 60) -> Path:
    """Download a file from a URL to a local path.

    Args:
        url: URL to download from.
        dest_path: Local filesystem path to save the file.
        timeout: Request timeout in seconds.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: If the download request fails.
    """
    if dest_path.exists():
        logger.debug("File already exists, skipping: %s", dest_path)
        return dest_path

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, disable=total_size == 0
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    logger.debug("Downloaded: %s", dest_path)
    return dest_path


def download_image_and_mask(
    image_meta: dict[str, Any],
    images_dir: Path,
    masks_dir: Path,
    api_base_url: str,
) -> tuple[Path | None, Path | None]:
    """Download an image and its segmentation mask from the ISIC Archive.

    Args:
        image_meta: Image metadata dictionary from the API.
        images_dir: Directory to save downloaded images.
        masks_dir: Directory to save downloaded masks.
        api_base_url: Base URL of the ISIC Archive API.

    Returns:
        Tuple of (image_path, mask_path). Either may be None if download fails.
    """
    isic_id = image_meta.get("isic_id", "")
    if not isic_id:
        logger.warning("Image metadata missing isic_id, skipping")
        return None, None

    image_url = f"{api_base_url}/images/{isic_id}/thumbnail"
    mask_url = f"{api_base_url}/segmentations/{isic_id}/thumbnail"

    image_path = images_dir / f"{isic_id}.jpg"
    mask_path = masks_dir / f"{isic_id}_mask.png"

    try:
        image_path = download_file(image_url, image_path)
    except requests.RequestException as e:
        logger.error("Failed to download image %s: %s", isic_id, e)
        return None, None

    try:
        mask_path = download_file(mask_url, mask_path)
    except requests.RequestException as e:
        logger.warning("Failed to download mask for %s: %s", isic_id, e)
        return image_path, None

    return image_path, mask_path


def download_dataset(
    config: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Path], list[Path]]:
    """Download the ISIC Skin Lesion dataset.

    Args:
        config: Application configuration dictionary. If None, loads default.
        limit: Override the number of images to download.

    Returns:
        Tuple of (image_paths, mask_paths) for successfully downloaded pairs.
    """
    if config is None:
        config = get_config()

    api_base_url = get_nested(config, "isic", "api_base_url")
    download_limit = limit or get_nested(config, "isic", "download_limit", default=100)
    raw_dir = Path(get_nested(config, "paths", "raw_dir", default="data/raw"))

    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading up to %d images from ISIC Archive to %s",
        download_limit,
        raw_dir,
    )

    image_list = get_image_list(api_base_url, limit=download_limit)

    image_paths: list[Path] = []
    mask_paths: list[Path] = []

    for image_meta in tqdm(image_list, desc="Downloading dataset"):
        img_path, msk_path = download_image_and_mask(
            image_meta, images_dir, masks_dir, api_base_url
        )
        if img_path is not None and msk_path is not None:
            image_paths.append(img_path)
            mask_paths.append(msk_path)

    logger.info("Download complete: %d image-mask pairs saved", len(image_paths))
    return image_paths, mask_paths
