"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    get_config,
    get_nested,
    load_config,
    reload_config,
)


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    """Create a temporary config YAML file for testing."""
    config_data = {
        "project": {"name": "test-project", "seed": 42},
        "data": {"image_size": 256, "batch_size": 16},
        "model": {"architecture": "unet", "encoder_channels": [64, 128]},
        "logging": {"level": "DEBUG"},
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file


@pytest.fixture()
def empty_config(tmp_path: Path) -> Path:
    """Create an empty config YAML file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    return config_file


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Reset the config cache before each test."""
    import src.utils.config as config_module

    config_module._config = None


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config: Path) -> None:
        """Load a valid YAML config file."""
        config = load_config(sample_config)
        assert config["project"]["name"] == "test-project"
        assert config["data"]["image_size"] == 256

    def test_load_default_config(self) -> None:
        """Load the default config.yaml from the project."""
        config = load_config()
        assert "project" in config
        assert "model" in config

    def test_load_nonexistent_file(self) -> None:
        """Raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_config(self, empty_config: Path) -> None:
        """Return empty dict for an empty YAML file."""
        config = load_config(empty_config)
        assert config == {}

    def test_load_from_env_variable(self, sample_config: Path) -> None:
        """Load config path from CONFIG_PATH environment variable."""
        with patch.dict(os.environ, {"CONFIG_PATH": str(sample_config)}):
            config = load_config()
            assert config["project"]["name"] == "test-project"

    def test_load_config_accepts_string_path(self, sample_config: Path) -> None:
        """Accept string path argument."""
        config = load_config(str(sample_config))
        assert config["project"]["seed"] == 42


class TestGetConfig:
    """Tests for get_config caching behavior."""

    def test_get_config_caches_result(self, sample_config: Path) -> None:
        """Return the same object on repeated calls (cached)."""
        config1 = get_config(sample_config)
        config2 = get_config()
        assert config1 is config2

    def test_reload_config(self, sample_config: Path) -> None:
        """Force reload replaces the cached config."""
        config1 = get_config(sample_config)
        config2 = reload_config(sample_config)
        assert config1 is not config2
        assert config2["project"]["name"] == "test-project"


class TestGetNested:
    """Tests for get_nested helper."""

    def test_get_existing_key(self) -> None:
        """Retrieve an existing nested key."""
        cfg = {"a": {"b": {"c": 42}}}
        assert get_nested(cfg, "a", "b", "c") == 42

    def test_get_missing_key_returns_default(self) -> None:
        """Return default for a missing key path."""
        cfg = {"a": {"b": 1}}
        assert get_nested(cfg, "a", "x", default="fallback") == "fallback"

    def test_get_top_level_key(self) -> None:
        """Retrieve a top-level key."""
        cfg = {"key": "value"}
        assert get_nested(cfg, "key") == "value"

    def test_default_is_none(self) -> None:
        """Default value is None when not specified."""
        cfg = {"a": 1}
        assert get_nested(cfg, "missing") is None

    def test_non_dict_intermediate(self) -> None:
        """Return default when an intermediate key is not a dict."""
        cfg = {"a": "string_value"}
        assert get_nested(cfg, "a", "b", default=0) == 0


class TestDefaultConfig:
    """Tests for the actual project config.yaml."""

    def test_default_config_path_exists(self) -> None:
        """The default config file exists on disk."""
        assert DEFAULT_CONFIG_PATH.exists()

    def test_default_config_has_required_sections(self) -> None:
        """The default config contains all required sections."""
        config = load_config()
        required_sections = [
            "project",
            "paths",
            "data",
            "augmentation",
            "model",
            "training",
            "evaluation",
            "export",
            "dashboard",
            "logging",
        ]
        for section in required_sections:
            assert section in config, f"Missing section: {section}"

    def test_default_config_values(self) -> None:
        """Spot-check important default values."""
        config = load_config()
        assert config["data"]["image_size"] == 256
        assert config["data"]["train_split"] == 0.70
        assert config["model"]["architecture"] == "unet"
        assert config["training"]["learning_rate"] == 0.001
