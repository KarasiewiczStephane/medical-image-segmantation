# Medical Image Segmentation

> Part of my Data Science Portfolio — [Stephane Karasiewicz](https://github.com/KarasiewiczStephane)

End-to-end skin lesion segmentation system using a U-Net architecture trained on ISIC Archive dermoscopy images. Includes data pipeline, model training with combined Dice+BCE loss, MC Dropout uncertainty estimation, Grad-CAM interpretability, ONNX export, and an interactive Streamlit dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
│  ISIC API → DICOM/JPEG → Resize(256) → Normalize → Augment     │
│              │                                                  │
│              ▼                                                  │
│  Patient-ID Split (70/15/15) → TFRecord Serialization           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         U-Net Model                             │
│                                                                 │
│  Input (256×256×3)                                              │
│     │                                                           │
│     ├─► Encoder: Conv-BN-ReLU blocks [64,128,256,512]           │
│     │      │         ↓ MaxPool between stages                   │
│     │      │                                                    │
│     │      ├─► Bottleneck: 1024 channels + Dropout(0.5)         │
│     │      │                                                    │
│     │      ├─► Decoder: ConvTranspose + Skip Connections         │
│     │      │         [512,256,128,64]                            │
│     │                                                           │
│     └─► Output: Sigmoid → Binary Mask (256×256×1)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Evaluation & Export                        │
│                                                                 │
│  Metrics: Dice, IoU, Pixel Accuracy, Sensitivity, Specificity   │
│  Uncertainty: MC Dropout (N=20) → Mean, Std, Entropy maps       │
│  Interpretability: Grad-CAM heatmap overlay                     │
│  Export: ONNX (opset 13) with inference benchmarking            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                          │
│                                                                 │
│  Upload (JPEG/PNG/DICOM) → Segmentation Overlay                 │
│  Confidence Threshold Slider │ Uncertainty Maps                  │
│  Grad-CAM Toggle │ Batch Processing │ ZIP Export                 │
└─────────────────────────────────────────────────────────────────┘
```

## Evaluation Metrics

| Metric           | Description                                    |
|------------------|------------------------------------------------|
| Dice Coefficient | Overlap between predicted and ground-truth mask |
| IoU (Jaccard)    | Intersection over union of mask regions         |
| Pixel Accuracy   | Fraction of correctly classified pixels         |
| Sensitivity      | True positive rate (recall)                     |
| Specificity      | True negative rate                              |

All metrics are computed per-sample and aggregated with 95% confidence intervals using the t-distribution.

## Setup

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/medical-image-segmantation.git
cd medical-image-segmantation

# Install
pip install -r requirements.txt

# Run tests
make test
```

## Usage

### Training

```bash
python -m src.main train
```

Trains U-Net with combined Dice+BCE loss, early stopping (patience=15), and ReduceLROnPlateau scheduling.

### Evaluation

```bash
python -m src.main evaluate
```

Computes segmentation metrics on the test set with confidence intervals.

### ONNX Export

```bash
python -m src.main export
```

Exports the trained model to ONNX format with optional quantization and inference benchmarking.

### Dashboard

```bash
make dashboard
```

Launches the Streamlit web interface at `http://localhost:8501` with:
- Single image and batch processing modes
- DICOM, JPEG, and PNG support
- Adjustable confidence threshold
- MC Dropout uncertainty visualization
- Grad-CAM interpretability overlay
- Mask download (PNG) and batch ZIP export

### Docker

```bash
# Single container
make docker

# Docker Compose
make docker-compose
```

## Project Structure

```
medical-image-segmantation/
├── src/
│   ├── data/
│   │   ├── downloader.py       # ISIC Archive dataset downloader
│   │   ├── dicom_handler.py    # DICOM reading and conversion
│   │   ├── preprocessor.py     # Resize, normalize, TFRecord I/O
│   │   └── augmentation.py     # Albumentations augmentation pipeline
│   ├── models/
│   │   ├── unet.py             # U-Net architecture
│   │   ├── trainer.py          # Training loop with Dice+BCE loss
│   │   ├── evaluator.py        # Metrics computation and overlay
│   │   ├── uncertainty.py      # MC Dropout uncertainty estimation
│   │   └── grad_cam.py         # Grad-CAM visualization
│   ├── export/
│   │   └── onnx_converter.py   # ONNX export and benchmarking
│   ├── dashboard/
│   │   └── app.py              # Streamlit web interface
│   ├── utils/
│   │   ├── config.py           # YAML config loader
│   │   └── logger.py           # Structured logging setup
│   └── main.py                 # CLI entry point
├── tests/                      # Unit tests (189 tests)
├── configs/
│   └── config.yaml             # Central configuration
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Configuration

All hyperparameters and paths are centralized in `configs/config.yaml`. Override the config path with the `CONFIG_PATH` environment variable or the `--config` CLI flag.

## CI/CD

GitHub Actions runs on every push and PR to `main`:
- **Lint**: ruff check and format validation
- **Test**: pytest with coverage across Python 3.10, 3.11, 3.12
- **Docker**: image build validation (gates on lint + test)

## License

MIT
