"""
Project configuration. Override with env vars where noted.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
PLACES365_RAW_DIR = DATA_DIR / "places365_raw"

# Dataset
DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 5000))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_SIZE = 128  # CNN input size

# Degradation parameters (severity 1-5)
BLUR_SIGMAS = {1: 1.0, 2: 3.0, 3: 5.0, 4: 9.0, 5: 15.0}
UNDEREXPOSURE_GAMMAS = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.4, 5: 0.3}
OVEREXPOSURE_GAMMAS = {1: 1.3, 2: 1.6, 3: 2.0, 4: 2.5, 5: 3.0}
NOISE_SIGMAS = {1: 10, 2: 20, 3: 35, 4: 50, 5: 70}
LOW_CONTRAST_FACTORS = {1: 0.8, 2: 0.65, 3: 0.5, 4: 0.35, 5: 0.2}
JPEG_QUALITIES = {1: 35, 2: 20, 3: 12, 4: 7, 5: 3}

# Quality score ranges per degradation + severity
QUALITY_SCORE_RANGES = {
    "GOOD":               (85, 100),
    "BLUR":               {1: (70, 80), 2: (55, 70), 3: (40, 55), 4: (25, 40), 5: (10, 25)},
    "UNDEREXPOSURE":      {1: (70, 80), 2: (55, 70), 3: (40, 55), 4: (25, 40), 5: (10, 25)},
    "OVEREXPOSURE":       {1: (70, 80), 2: (55, 70), 3: (40, 55), 4: (25, 40), 5: (10, 25)},
    "NOISE":              {1: (70, 80), 2: (55, 70), 3: (40, 55), 4: (25, 40), 5: (10, 25)},
    "LOW_CONTRAST":       {1: (65, 75), 2: (50, 65), 3: (35, 50), 4: (20, 35), 5: (10, 20)},
    "JPEG_CORRUPTION":    {1: (65, 75), 2: (50, 65), 3: (35, 50), 4: (20, 35), 5: (10, 20)},
    "SEVERE_DEGRADATION": {1: (50, 65), 2: (35, 50), 3: (20, 35), 4: (10, 20), 5: (5, 15)},
}

# Labels: >= 75 acceptable, >= 40 degraded, else defective
QUALITY_LABEL_THRESHOLDS = {"ACCEPTABLE": 75, "DEGRADED": 40}

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "low_contrast", "corruption"]
SEVERITY_THRESHOLDS = [(0.75, "high"), (0.45, "medium"), (0.0, "low")]

# Training
CNN_EPOCHS = int(os.environ.get("CNN_EPOCHS", 25))
CNN_BATCH_SIZE = int(os.environ.get("CNN_BATCH_SIZE", 64))
CNN_LR = float(os.environ.get("CNN_LR", 1.5e-3))

# Backend
DATABASE_PATH = PROJECT_ROOT / "backend" / "analyses.db"
MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
