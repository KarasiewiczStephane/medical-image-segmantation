"""Structured logging setup for the medical image segmentation project.

Provides a configured logger with both console and optional file output.
Log level and format are controlled via configs/config.yaml.
"""

import logging
import sys
from pathlib import Path
from typing import Any


_loggers: dict[str, logging.Logger] = {}


def setup_logging(config: dict[str, Any] | None = None) -> None:
    """Configure the root logger based on application settings.

    Args:
        config: Full application configuration dictionary. If None,
            uses sensible defaults for console-only logging.
    """
    log_config = (config or {}).get("logging", {})

    level_name = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = log_config.get(
        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates on reload
    root_logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_config.get("file_logging", False):
        log_file = log_config.get("log_file", "logs/app.log")
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance.

    Args:
        name: Name for the logger, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]
