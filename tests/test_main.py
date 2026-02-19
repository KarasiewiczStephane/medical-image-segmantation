"""Tests for the main entry point."""

from src.main import main, parse_args


class TestParseArgs:
    """Tests for argument parsing."""

    def test_no_args(self) -> None:
        """Parse empty arguments."""
        args = parse_args([])
        assert args.command is None
        assert args.config is None

    def test_train_command(self) -> None:
        """Parse train command."""
        args = parse_args(["train"])
        assert args.command == "train"

    def test_evaluate_command(self) -> None:
        """Parse evaluate command with checkpoint."""
        args = parse_args(["evaluate", "--checkpoint", "model.h5"])
        assert args.command == "evaluate"
        assert args.checkpoint == "model.h5"

    def test_export_command(self) -> None:
        """Parse export command."""
        args = parse_args(["export"])
        assert args.command == "export"

    def test_dashboard_command(self) -> None:
        """Parse dashboard command."""
        args = parse_args(["dashboard"])
        assert args.command == "dashboard"

    def test_custom_config(self) -> None:
        """Parse custom config path."""
        args = parse_args(["--config", "/path/to/config.yaml", "train"])
        assert args.config == "/path/to/config.yaml"
        assert args.command == "train"


class TestMain:
    """Tests for main function."""

    def test_no_command(self) -> None:
        """Return 0 when no command is specified."""
        result = main([])
        assert result == 0

    def test_train_command(self) -> None:
        """Return 0 for train command (placeholder)."""
        result = main(["train"])
        assert result == 0

    def test_evaluate_command(self) -> None:
        """Return 0 for evaluate command (placeholder)."""
        result = main(["evaluate"])
        assert result == 0

    def test_export_command(self) -> None:
        """Return 0 for export command (placeholder)."""
        result = main(["export"])
        assert result == 0

    def test_dashboard_command(self) -> None:
        """Return 0 for dashboard command (placeholder)."""
        result = main(["dashboard"])
        assert result == 0
