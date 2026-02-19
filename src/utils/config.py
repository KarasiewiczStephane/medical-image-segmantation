"""Configuration management for the medical image segmentation project.

Loads settings from configs/config.yaml and provides a unified interface
for accessing configuration values throughout the application.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_config: dict[str, Any] | None = None

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. If None, uses
            the default path at configs/config.yaml.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG_PATH))

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def get_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Get the application configuration, loading it if necessary.

    Uses a module-level cache to avoid re-reading the file on every call.
    Call reload_config() to force a fresh read.

    Args:
        config_path: Optional path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration values.
    """
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Force reload the configuration from disk.

    Args:
        config_path: Optional path to the YAML configuration file.

    Returns:
        Freshly loaded configuration dictionary.
    """
    global _config
    _config = load_config(config_path)
    return _config


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Retrieve a nested value from the configuration dictionary.

    Args:
        config: The configuration dictionary to search.
        *keys: Sequence of keys forming the path to the desired value.
        default: Value to return if the key path does not exist.

    Returns:
        The value at the specified key path, or default if not found.

    Example:
        >>> cfg = {"model": {"encoder_channels": [64, 128]}}
        >>> get_nested(cfg, "model", "encoder_channels")
        [64, 128]
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
