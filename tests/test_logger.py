"""Tests for structured logging setup."""

import logging
from pathlib import Path

from src.utils.logger import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_with_defaults(self) -> None:
        """Configure logging with default settings."""
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) >= 1

    def test_setup_with_config(self) -> None:
        """Configure logging from a config dictionary."""
        config = {
            "logging": {
                "level": "DEBUG",
                "format": "%(levelname)s - %(message)s",
                "file_logging": False,
            }
        }
        setup_logging(config)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_with_file_logging(self, tmp_path: Path) -> None:
        """Enable file logging to a temporary directory."""
        log_file = tmp_path / "test.log"
        config = {
            "logging": {
                "level": "INFO",
                "file_logging": True,
                "log_file": str(log_file),
            }
        }
        setup_logging(config)
        root = logging.getLogger()

        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == log_file

    def test_setup_clears_duplicate_handlers(self) -> None:
        """Calling setup_logging twice does not duplicate handlers."""
        setup_logging()
        handler_count_1 = len(logging.getLogger().handlers)
        setup_logging()
        handler_count_2 = len(logging.getLogger().handlers)
        assert handler_count_2 == handler_count_1

    def test_setup_with_none_config(self) -> None:
        """Passing None config uses defaults."""
        setup_logging(None)
        root = logging.getLogger()
        assert root.level == logging.INFO


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Return a logging.Logger instance."""
        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test.module"

    def test_get_logger_caches(self) -> None:
        """Return the same logger for the same name."""
        log1 = get_logger("test.cached")
        log2 = get_logger("test.cached")
        assert log1 is log2

    def test_get_logger_different_names(self) -> None:
        """Return different loggers for different names."""
        log1 = get_logger("module.a")
        log2 = get_logger("module.b")
        assert log1 is not log2
