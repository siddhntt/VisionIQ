# Image Quality Assessment

Full-stack app that evaluates image quality using a hybrid ML approach — MobileNetV2 (transfer learning) for overall quality scoring and classical classifiers on hand-crafted features for specific issue detection.

Detects: blur, underexposure, overexposure, noise, low contrast, JPEG corruption, and severe degradation.

## Setup

### Docker (easiest)

```bash
docker compose up --build
# open http://localhost:8000
```

### Manual

```bash
pip install -r requirements.txt

# generate dataset (needs Places365 images in data/places365_raw/)
# then run notebooks in order:
#   01_dataset_generation.ipynb
#   02_feature_engineering.ipynb
#   03_model_training.ipynb
#   04_evaluation.ipynb
#   05_inference_demo.ipynb

# start server
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

## How It Works

Two parallel ML pipelines run on each uploaded image:

1. **MobileNetV2** (pretrained on ImageNet, fine-tuned) predicts a quality score 0-100 and generates a Grad-CAM attention heatmap
2. **6 LogisticRegression classifiers** on 18 engineered features detect specific issues (blur, noise, etc.) with confidence scores and feature evidence

The final score blends both signals — CNN gives the base score, classical detections apply penalties for confirmed issues.

## Dataset

Uses Places365 validation images as clean sources. Download:
```
http://data.csail.mit.edu/places/places365/val_256.tar
```
Extract to `data/places365_raw/`. Configurable size via `DATASET_SIZE` env var (default: 5000).

### Data Leakage Prevention

Clean images are split into train/val/test **before** generating degradations. All versions of the same source image stay in the same split.

### Degradation Types

| Type | Method | Severity Levels |
|------|--------|-----------------|
| GOOD | Original | — |
| BLUR | Gaussian blur | σ: 1, 3, 5, 9, 15 |
| UNDEREXPOSURE | Gamma correction | γ: 0.3–0.8 |
| OVEREXPOSURE | Gamma correction | γ: 1.3–3.0 |
| NOISE | Gaussian + salt-pepper | σ: 10–70 |
| LOW_CONTRAST | Pull toward mean | factor: 0.2–0.8 |
| JPEG_CORRUPTION | Low-quality encode | quality: 3–35 |
| SEVERE_DEGRADATION | Combined | 2 degradations mixed |

## Engineered Features (18)

Sharpness (laplacian variance, high-freq energy), exposure (brightness, under/overexposed fractions, dynamic range), contrast (Michelson, RMS), noise (Immerkaer estimator, local variance), color (saturation, colorfulness), structure (edge density, block artifacts, entropy, texture energy).

## API

### POST /api/analyze
```bash
curl -F "file=@photo.jpg" http://localhost:8000/api/analyze
```
Returns:
```json
{
  "quality_score": 82.3,
  "quality_label": "ACCEPTABLE",
  "confidence": 0.9,
  "issues": [{"type": "noise", "severity": "low", "confidence": 0.71, "evidence": [...]}],
  "image_stats": {"width": 256, "height": 256, "mean_brightness": 128.5, ...},
  "model_signals": {"cnn_quality_score": 85.1, "model_type": "MobileNetV2"},
  "heatmap": "data:image/png;base64,..."
}
```

### GET /api/analyses
List past results. Params: `limit`, `offset`.

### GET /api/analyses/{id}
Get one result by ID.

### GET /api/health
Status check — model loaded, uptime, analysis count.

## Project Structure

```
├── ml/                     # core ML code
│   ├── features.py         # 18 engineered features
│   ├── model.py            # MobileNetV2 + QualityCNN
│   ├── synthetic_data.py   # dataset generation
│   └── inference.py        # analysis pipeline
├── backend/                # FastAPI server
│   ├── app.py              # endpoints
│   ├── database.py         # SQLite
│   └── schemas.py          # response models
├── frontend/               # web UI
│   ├── index.html
│   ├── style.css
│   └── script.js
├── notebooks/              # training pipeline (run in order)
│   ├── 01_dataset_generation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_inference_demo.ipynb
├── models/                 # trained artifacts (generated)
├── config.py               # all settings
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Configuration

Set via environment variables or edit `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| DATASET_SIZE | 5000 | Source images to use |
| RANDOM_SEED | 42 | For reproducibility |
| CNN_EPOCHS | 25 | Training epochs |
| CNN_BATCH_SIZE | 64 | Batch size |

## Tech Stack

PyTorch, torchvision, scikit-learn, OpenCV, FastAPI, SQLite, vanilla JS
