# Image Quality Assessment

> **Live Demo:** [https://visioniq-e7vk.onrender.com/](https://visioniq-e7vk.onrender.com/)

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

### Architecture & Model Selection
Two parallel ML pipelines evaluate each image:

1. **Deep Learning Quality Regressor (MobileNetV2)**:
   - **Why MobileNetV2?** Pretrained on ImageNet, lightweight (~3.4M parameters), fast inference on CPU, and effective transfer learning for general visual structure without the computational footprint of heavier backbones like ResNet-50.
   - **Baseline Comparison**: A custom 4-layer `QualityCNN` (~150K parameters) was trained from scratch as an experimental baseline.
   - **Training Strategy**: 2-phase training (Phase 1: warm-up head with frozen backbone; Phase 2: end-to-end fine-tuning with Cosine Annealing and Smooth L1 Loss).
   - Generates a holistic quality score (0–100) and a **Grad-CAM attention heatmap**.

2. **Classical Feature Extraction & Issue Classifiers**:
   - Computes **18 hand-crafted computer vision features** across 6 categories (sharpness, exposure, contrast, noise, color, structure).
   - **6 LogisticRegression classifiers** (one per issue type) predict specific defects with calibrated confidence scores and feature-level evidence.

3. **Hybrid Score Blending**:
   - The final score starts with the CNN prediction and applies bounded penalties for confirmed issues.
   - Categorized into standard decision buckets:
     - **ACCEPTABLE** (`score ≥ 70`)
     - **DEGRADED** (`40 ≤ score < 70`)
     - **DEFECTIVE** (`score < 40`)

### Explainability
Every prediction provides multi-level transparency:
- **Grad-CAM Heatmap**: Spatial visualization of image regions that influenced the CNN score most.
- **Feature Evidence**: Linear logistic regression weights identify the top 3 contributing features and their direction of influence for each detected issue.
- **Physical Image Statistics**: 10 interpretable metrics (brightness, sharpness, noise $\sigma$, contrast, saturation, entropy, etc.) displayed in real-time.

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

## Evaluation Summary

Evaluated on the held-out test split (unseen source images):

| Model / Classifier | Metric | Value |
|--------------------|--------|-------|
| MobileNetV2 (Regression) | MAE / R² | 9.60 / 0.690 |
| QualityCNN Baseline | MAE / R² | 9.65 / 0.706 |
| Low Contrast Classifier | ROC-AUC / F1 | 0.994 / 0.884 |
| Underexposure Classifier | ROC-AUC / F1 | 0.949 / 0.633 |
| Blur Classifier | ROC-AUC / F1 | 0.939 / 0.777 |
| Overexposure Classifier | ROC-AUC / F1 | 0.931 / 0.572 |
| Noise Classifier | ROC-AUC / F1 | 0.923 / 0.758 |
| Corruption Classifier | ROC-AUC / F1 | 0.803 / 0.585 |

*Full evaluation, confusion matrices, ROC curves, and failure analysis are in `notebooks/04_evaluation.ipynb`.*

## Database

Uses SQLite for storing analysis history (`quality_assessment.db`).
- **Setup:** Automatically created and migrated on application startup. No manual database setup required.
- **Configurable:** Override file path via `DATABASE_PATH` environment variable.

## Sample Images

The `samples/` folder includes 8 representative test images across all degradation types and severities (`GOOD`, `BLUR`, `NOISE`, `UNDEREXPOSURE`, `OVEREXPOSURE`, `LOW_CONTRAST`, `JPEG_CORRUPTION`, `SEVERE_DEGRADATION`) for immediate testing via the web UI or API.

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
├── models/                 # trained weights & scalers (.pt, .pkl)
├── samples/                # sample images for testing
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
| DATABASE_PATH | quality_assessment.db | SQLite database file |

## Tech Stack

PyTorch, torchvision, scikit-learn, OpenCV, FastAPI, SQLite, vanilla JS
