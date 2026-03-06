"""Main entry point for the medical image segmentation application.

Supports running download, training, evaluation, export, and the dashboard
through command-line arguments.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.config import get_config, get_nested
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

    # Download command
    dl_parser = subparsers.add_parser("download", help="Download ISIC images")
    dl_parser.add_argument(
        "--limit", type=int, default=None, help="Number of images to download"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the segmentation model")
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    train_parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic masks for demo training (no real masks needed)",
    )

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


def _generate_synthetic_masks(images_dir: Path, masks_dir: Path) -> None:
    """Generate simple synthetic segmentation masks for demo training.

    Creates elliptical masks centered on each image to simulate lesion
    boundaries. This allows the model to learn basic segmentation without
    real annotations.
    """
    import cv2
    import numpy as np

    masks_dir.mkdir(parents=True, exist_ok=True)
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))

    for img_path in image_files:
        mask_path = masks_dir / f"{img_path.stem}_mask.png"
        if mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw an ellipse in the center (rough lesion approximation)
        center = (w // 2, h // 2)
        axes = (w // 4, h // 4)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        cv2.imwrite(str(mask_path), mask)

    logger.info("Generated synthetic masks for %d images", len(image_files))


def _run_train(config: dict, args: argparse.Namespace) -> int:
    """Run the training pipeline."""
    from src.data.preprocessor import run_preprocessing_pipeline
    from src.models.trainer import compile_model, train_model
    from src.models.unet import build_unet_from_config

    raw_dir = Path(get_nested(config, "paths", "raw_dir", default="data/raw"))
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "masks"

    # Check for images
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        logger.error(
            "No images found in %s. Run 'python -m src.main download' first.",
            images_dir,
        )
        return 1

    # Generate synthetic masks if --demo or no masks exist
    existing_masks = list(masks_dir.glob("*_mask.png")) if masks_dir.exists() else []
    if args.demo or not existing_masks:
        if not args.demo:
            logger.info("No masks found, generating synthetic masks for demo training")
        _generate_synthetic_masks(images_dir, masks_dir)

    # Preprocess: split and write TFRecords
    logger.info("Running preprocessing pipeline...")
    run_preprocessing_pipeline(config)

    # Check TFRecords were created
    from src.data.preprocessor import load_tfrecord_dataset

    tfrecord_dir = Path(
        get_nested(config, "paths", "tfrecord_dir", default="data/tfrecords")
    )
    train_tfrecord = tfrecord_dir / "train.tfrecord"
    val_tfrecord = tfrecord_dir / "val.tfrecord"

    if not train_tfrecord.exists():
        logger.error("Training TFRecord not found at %s", train_tfrecord)
        return 1

    image_size = get_nested(config, "data", "image_size", default=256)
    channels = get_nested(config, "data", "num_channels", default=3)
    batch_size = get_nested(config, "data", "batch_size", default=16)

    train_dataset = load_tfrecord_dataset(
        train_tfrecord, image_size, channels, batch_size, shuffle=True
    )
    val_dataset = (
        load_tfrecord_dataset(
            val_tfrecord, image_size, channels, batch_size, shuffle=False
        )
        if val_tfrecord.exists()
        else None
    )

    # Build and compile model
    logger.info("Building U-Net model...")
    model = build_unet_from_config(config)

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs

    model = compile_model(model, config)

    # Train
    logger.info("Starting training...")
    train_model(model, train_dataset, val_dataset, config)

    logger.info("Training complete! Model saved to models/checkpoints/best_model.keras")
    return 0


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

    if args.command == "download":
        from src.data.downloader import download_dataset

        image_paths, mask_paths = download_dataset(config, limit=args.limit)
        logger.info("Downloaded %d images", len(image_paths))
    elif args.command == "train":
        return _run_train(config, args)
    elif args.command == "evaluate":
        logger.info("Evaluation pipeline not yet implemented")
    elif args.command == "export":
        logger.info("Export pipeline not yet implemented")
    elif args.command == "dashboard":
        import subprocess

        subprocess.run(
            ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501"],
            check=True,
        )
    else:
        logger.error("Unknown command: %s", args.command)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
