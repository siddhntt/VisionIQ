"""
Dataset generation with controlled degradations from clean images.

Important: we split the original clean images into train/val/test FIRST,
then generate degraded versions. This prevents data leakage - no version
of the same source image ever appears in different splits.
"""
from __future__ import annotations
import csv
import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    BLUR_SIGMAS, UNDEREXPOSURE_GAMMAS, OVEREXPOSURE_GAMMAS,
    NOISE_SIGMAS, LOW_CONTRAST_FACTORS, JPEG_QUALITIES,
    QUALITY_SCORE_RANGES, RANDOM_SEED, DATASET_SIZE,
    TRAIN_RATIO, VAL_RATIO, IMG_SIZE, DATA_DIR, PLACES365_RAW_DIR,
)

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "low_contrast", "corruption"]

DEGRADATION_TYPES = [
    "GOOD", "BLUR", "UNDEREXPOSURE", "OVEREXPOSURE",
    "NOISE", "LOW_CONTRAST", "JPEG_CORRUPTION", "SEVERE_DEGRADATION"
]


@dataclass
class Sample:
    image: np.ndarray
    degradation_type: str
    severity: int           # 0 for GOOD, 1-5 for degraded
    quality_score: float
    source_image_id: str
    issues: dict = field(default_factory=dict)


# --- degradation functions ---

def apply_blur(img, severity):
    sigma = BLUR_SIGMAS[severity]
    ksize = int(sigma * 6) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def apply_underexposure(img, severity):
    gamma = UNDEREXPOSURE_GAMMAS[severity]
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_overexposure(img, severity):
    gamma = OVEREXPOSURE_GAMMAS[severity]
    table = np.array([min(255, ((i / 255.0) ** (1.0 / gamma)) * 255) for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_noise(img, severity, rng):
    sigma = NOISE_SIGMAS[severity]
    noise = rng.normal(0, sigma, img.shape).astype(np.float64)
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    # salt and pepper at higher severities
    if severity >= 3:
        sp = 0.005 * severity
        mask = rng.random(img.shape[:2])
        noisy[mask < sp / 2] = 0
        noisy[mask > 1 - sp / 2] = 255
    return noisy


def apply_low_contrast(img, severity):
    factor = LOW_CONTRAST_FACTORS[severity]
    mean = img.mean()
    return np.clip(mean + factor * (img.astype(np.float64) - mean), 0, 255).astype(np.uint8)


def apply_jpeg_corruption(img, severity):
    quality = JPEG_QUALITIES[severity]
    _, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_severe_degradation(img, severity, rng):
    """Combine 2 degradations randomly."""
    combos = [
        lambda im: apply_noise(apply_blur(im, min(severity, 4)), min(severity, 4), rng),
        lambda im: apply_noise(apply_underexposure(im, min(severity, 4)), min(severity, 3), rng),
        lambda im: apply_jpeg_corruption(apply_blur(im, min(severity, 4)), min(severity, 4)),
        lambda im: apply_noise(apply_overexposure(im, min(severity, 3)), min(severity, 4), rng),
        lambda im: apply_low_contrast(apply_blur(im, min(severity, 4)), min(severity, 4)),
        lambda im: apply_jpeg_corruption(apply_noise(im, min(severity, 3), rng), min(severity, 3)),
    ]
    return combos[rng.randint(len(combos))](img)


# --- label helpers ---

def derive_issues(degradation_type):
    issues = {t: {"present": False} for t in ISSUE_TYPES}
    mapping = {
        "BLUR": ["blur"], "UNDEREXPOSURE": ["underexposure"],
        "OVEREXPOSURE": ["overexposure"], "NOISE": ["noise"],
        "LOW_CONTRAST": ["low_contrast"], "JPEG_CORRUPTION": ["corruption"],
        "SEVERE_DEGRADATION": ["corruption", "blur", "noise"],
    }
    for issue in mapping.get(degradation_type, []):
        issues[issue]["present"] = True
    return issues


def assign_quality_score(degradation_type, severity, rng):
    ranges = QUALITY_SCORE_RANGES[degradation_type]
    lo, hi = ranges if isinstance(ranges, tuple) else ranges[severity]
    return round(rng.uniform(lo, hi), 1)


# --- dataset building ---

def find_source_images(raw_dir, max_images):
    all_images = sorted(raw_dir.glob("**/*.jpg")) + sorted(raw_dir.glob("**/*.png"))
    if not all_images:
        raise FileNotFoundError(f"No images in {raw_dir}. Download Places365 first.")
    if len(all_images) > max_images:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = sorted(rng.choice(len(all_images), max_images, replace=False))
        all_images = [all_images[i] for i in idx]
    return all_images


def split_source_images(image_paths):
    """Split originals into train/val/test BEFORE generating degradations."""
    rng = np.random.RandomState(RANDOM_SEED)
    indices = rng.permutation(len(image_paths))
    n_train = int(len(image_paths) * TRAIN_RATIO)
    n_val = int(len(image_paths) * VAL_RATIO)
    return {
        "train": [image_paths[i] for i in sorted(indices[:n_train])],
        "val": [image_paths[i] for i in sorted(indices[n_train:n_train + n_val])],
        "test": [image_paths[i] for i in sorted(indices[n_train + n_val:])],
    }


def generate_samples_for_image(img_path, source_id, rng, img_size=IMG_SIZE):
    """Generate GOOD + degraded versions of one source image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    samples = []

    # clean version
    samples.append(Sample(
        image=img.copy(), degradation_type="GOOD", severity=0,
        quality_score=assign_quality_score("GOOD", 0, rng),
        source_image_id=source_id, issues=derive_issues("GOOD"),
    ))

    # degraded versions
    deg_funcs = {
        "BLUR": lambda im, s: apply_blur(im, s),
        "UNDEREXPOSURE": lambda im, s: apply_underexposure(im, s),
        "OVEREXPOSURE": lambda im, s: apply_overexposure(im, s),
        "NOISE": lambda im, s: apply_noise(im, s, rng),
        "LOW_CONTRAST": lambda im, s: apply_low_contrast(im, s),
        "JPEG_CORRUPTION": lambda im, s: apply_jpeg_corruption(im, s),
        "SEVERE_DEGRADATION": lambda im, s: apply_severe_degradation(im, s, rng),
    }

    for deg_type, func in deg_funcs.items():
        n_sev = rng.choice([1, 2], p=[0.4, 0.6])
        severities = sorted(rng.choice([1, 2, 3, 4, 5], n_sev, replace=False))
        for sev in severities:
            try:
                degraded = func(img.copy(), sev)
                samples.append(Sample(
                    image=degraded, degradation_type=deg_type, severity=sev,
                    quality_score=assign_quality_score(deg_type, sev, rng),
                    source_image_id=source_id, issues=derive_issues(deg_type),
                ))
            except Exception:
                continue
    return samples


def build_dataset(source_dir=None, max_source_images=None, img_size=IMG_SIZE,
                  seed=RANDOM_SEED, save_dir=None):
    """Build dataset with leakage-safe splits."""
    source_dir = source_dir or PLACES365_RAW_DIR
    max_source_images = max_source_images or DATASET_SIZE

    print(f"Finding source images in {source_dir}...")
    source_images = find_source_images(source_dir, max_source_images)
    print(f"Found {len(source_images)} source images")

    # split originals first (data leakage prevention)
    print("Splitting source images into train/val/test...")
    splits = split_source_images(source_images)
    for name, imgs in splits.items():
        print(f"  {name}: {len(imgs)} sources")

    dataset = {}
    for split_name, split_images in splits.items():
        rng = np.random.RandomState(seed + hash(split_name) % 10000)
        samples = []
        for i, img_path in enumerate(split_images):
            source_id = f"{split_name}_{i:05d}_{img_path.stem}"
            samples.extend(generate_samples_for_image(img_path, source_id, rng, img_size))
            if (i + 1) % 500 == 0:
                print(f"  [{split_name}] {i+1}/{len(split_images)} ({len(samples)} samples)")
        dataset[split_name] = samples
        print(f"  [{split_name}] done: {len(samples)} samples")

    if save_dir:
        save_dataset(dataset, save_dir)
    return dataset


def save_dataset(dataset, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in dataset.items():
        split_dir = save_dir / split_name / "images"
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, s in enumerate(samples):
            fname = f"{i:06d}.jpg"
            cv2.imwrite(str(split_dir / fname), s.image)
            rows.append({
                "filename": fname, "degradation_type": s.degradation_type,
                "severity": s.severity, "quality_score": s.quality_score,
                "source_image_id": s.source_image_id,
                **{f"issue_{k}": int(v["present"]) for k, v in s.issues.items()},
            })

        csv_path = save_dir / split_name / "metadata.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved {len(samples)} to {split_dir}")
