"""Tests for the ISIC dataset downloader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.downloader import (
    download_dataset,
    download_file,
    download_image_and_mask,
    get_image_list,
)


class TestGetImageList:
    """Tests for get_image_list function."""

    @patch("src.data.downloader.requests.get")
    def test_returns_image_list(self, mock_get: MagicMock) -> None:
        """Return parsed JSON list from the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"isic_id": "ISIC_0000001"},
            {"isic_id": "ISIC_0000002"},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_image_list("https://api.example.com", limit=10)
        assert len(result) == 2
        assert result[0]["isic_id"] == "ISIC_0000001"

    @patch("src.data.downloader.requests.get")
    def test_passes_params(self, mock_get: MagicMock) -> None:
        """Pass limit and offset as query parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        get_image_list("https://api.example.com", limit=50, offset=10)
        _, kwargs = mock_get.call_args
        assert kwargs["params"]["limit"] == 50
        assert kwargs["params"]["offset"] == 10

    @patch("src.data.downloader.requests.get")
    def test_raises_on_http_error(self, mock_get: MagicMock) -> None:
        """Raise HTTPError on API failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_response

        with pytest.raises(requests.HTTPError):
            get_image_list("https://api.example.com")


class TestDownloadFile:
    """Tests for download_file function."""

    @patch("src.data.downloader.requests.get")
    def test_download_new_file(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Download and save a file that does not exist yet."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dest = tmp_path / "test.jpg"
        result = download_file("https://example.com/file.jpg", dest)
        assert result == dest
        assert dest.read_bytes() == b"data"

    def test_skip_existing_file(self, tmp_path: Path) -> None:
        """Skip download if file already exists."""
        dest = tmp_path / "existing.jpg"
        dest.write_bytes(b"existing")
        result = download_file("https://example.com/file.jpg", dest)
        assert result == dest
        assert dest.read_bytes() == b"existing"

    @patch("src.data.downloader.requests.get")
    def test_creates_parent_dirs(self, mock_get: MagicMock, tmp_path: Path) -> None:
        """Create parent directories if they don't exist."""
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "0"}
        mock_response.iter_content.return_value = [b"x"]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        dest = tmp_path / "a" / "b" / "file.jpg"
        download_file("https://example.com/file.jpg", dest)
        assert dest.exists()


class TestDownloadImageAndMask:
    """Tests for download_image_and_mask function."""

    @patch("src.data.downloader.download_file")
    def test_download_pair(self, mock_download: MagicMock, tmp_path: Path) -> None:
        """Download both image and mask for a valid record."""
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        images_dir.mkdir()
        masks_dir.mkdir()

        mock_download.side_effect = lambda url, dest, **kw: dest

        meta = {"isic_id": "ISIC_001"}
        img, msk = download_image_and_mask(
            meta, images_dir, masks_dir, "https://api.example.com"
        )
        assert img == images_dir / "ISIC_001.jpg"
        assert msk == masks_dir / "ISIC_001_mask.png"

    def test_skip_missing_isic_id(self, tmp_path: Path) -> None:
        """Return None for records without isic_id."""
        img, msk = download_image_and_mask(
            {}, tmp_path, tmp_path, "https://api.example.com"
        )
        assert img is None
        assert msk is None

    @patch("src.data.downloader.download_file")
    def test_image_download_failure(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        """Return None for both if image download fails."""
        mock_download.side_effect = requests.ConnectionError("timeout")

        meta = {"isic_id": "ISIC_001"}
        img, msk = download_image_and_mask(
            meta, tmp_path, tmp_path, "https://api.example.com"
        )
        assert img is None
        assert msk is None


class TestDownloadDataset:
    """Tests for download_dataset function."""

    @patch("src.data.downloader.download_image_and_mask")
    @patch("src.data.downloader.get_image_list")
    def test_download_dataset(
        self,
        mock_list: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Download dataset and return valid pairs."""
        mock_list.return_value = [
            {"isic_id": "ISIC_001"},
            {"isic_id": "ISIC_002"},
        ]
        mock_download.side_effect = [
            (tmp_path / "img1.jpg", tmp_path / "mask1.png"),
            (None, None),
        ]

        config = {
            "isic": {
                "api_base_url": "https://api.example.com",
                "download_limit": 2,
            },
            "paths": {"raw_dir": str(tmp_path / "raw")},
        }

        images, masks = download_dataset(config=config, limit=2)
        assert len(images) == 1
        assert len(masks) == 1
