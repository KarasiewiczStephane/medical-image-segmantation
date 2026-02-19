"""Main entry point for the medical image segmentation application.

Supports running training, evaluation, export, and the dashboard
through command-line arguments.
"""

import argparse
import logging
import sys

from src.utils.config import get_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: List of arguments to parse. Defaults to sys.argv[1:].

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Medical Image Segmentation Tool",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    subparsers.add_parser("train", help="Train the segmentation model")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the trained model")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint"
    )

    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the application with the specified command.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    args = parse_args(argv)

    config = get_config(args.config)
    setup_logging(config)

    logger.info("Starting Medical Image Segmentation Tool")

    if args.command is None:
        logger.info("No command specified. Use --help to see available commands.")
        return 0

    logger.info("Running command: %s", args.command)

    if args.command == "train":
        logger.info("Training pipeline not yet implemented")
    elif args.command == "evaluate":
        logger.info("Evaluation pipeline not yet implemented")
    elif args.command == "export":
        logger.info("Export pipeline not yet implemented")
    elif args.command == "dashboard":
        logger.info("Dashboard not yet implemented")
    else:
        logger.error("Unknown command: %s", args.command)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
