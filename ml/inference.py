"""
Inference pipeline: image bytes -> quality analysis dict.

Combines CNN quality score with per-issue classical classifiers.
Also generates a Grad-CAM-style heatmap for explainability.
"""
from __future__ import annotations
import json
import pickle
import base64
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.features import extract_features, ImageFeatures
from ml.model import QualityCNN, MobileNetV2Quality
from ml.synthetic_data import ISSUE_TYPES

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import IMG_SIZE, MODELS_DIR, SEVERITY_THRESHOLDS, QUALITY_LABEL_THRESHOLDS


def _severity_bucket(confidence):
    for thresh, label in SEVERITY_THRESHOLDS:
        if confidence >= thresh:
            return label
    return "low"


class QualityAnalyzer:
    """Loads models once, reuse for every request."""

    def __init__(self, models_dir=MODELS_DIR):
        self.device = "cpu"

        # try mobilenet first, fall back to custom cnn
        mobilenet_path = models_dir / "mobilenet_quality.pt"
        cnn_path = models_dir / "cnn_quality.pt"

        if mobilenet_path.exists():
            self.cnn = MobileNetV2Quality(pretrained=False)
            self.cnn.load_state_dict(torch.load(mobilenet_path, map_location=self.device, weights_only=True))
            self.model_type = "MobileNetV2"
        elif cnn_path.exists():
            self.cnn = QualityCNN()
            self.cnn.load_state_dict(torch.load(cnn_path, map_location=self.device, weights_only=True))
            self.model_type = "QualityCNN"
        else:
            self.cnn = QualityCNN()
            self.model_type = "QualityCNN (untrained)"
        self.cnn.eval()

        # load sklearn classifiers
        clf_path = models_dir / "issue_classifiers.pkl"
        self.classifiers = pickle.loads(clf_path.read_bytes()) if clf_path.exists() else {}

        feat_path = models_dir / "feature_names.json"
        self.feature_names = json.loads(feat_path.read_text()) if feat_path.exists() else ImageFeatures.names()

        eval_path = models_dir / "eval_report.json"
        self.eval_report = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    @staticmethod
    def decode_image(image_bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _cnn_score_and_cam(self, bgr):
        resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)

        with torch.no_grad():
            score, feat = self.cnn.forward_with_activation(tensor)
        score = float(np.clip(score.item(), 0, 100))

        # simple attention map from last conv layer
        cam = feat.mean(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=(bgr.shape[0], bgr.shape[1]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return score, cam

    def _detect_issues(self, feats):
        x = feats.to_vector().reshape(1, -1)
        issues = []
        for issue_type in ISSUE_TYPES:
            if issue_type not in self.classifiers:
                continue
            scaler = self.classifiers[issue_type]["scaler"]
            clf = self.classifiers[issue_type]["model"]
            xs = scaler.transform(x)
            proba = float(clf.predict_proba(xs)[0, 1])

            if proba >= 0.5:
                # find which features drove this decision
                coefs = clf.coef_[0]
                contrib = coefs * xs[0]
                top_idx = np.argsort(-np.abs(contrib))[:3]
                evidence = [
                    {"feature": self.feature_names[i], "value": round(float(x[0, i]), 3),
                     "direction": "increases" if contrib[i] > 0 else "decreases"}
                    for i in top_idx
                ]
                issues.append({
                    "type": issue_type,
                    "severity": _severity_bucket(proba),
                    "confidence": round(proba, 3),
                    "evidence": evidence,
                })
        return issues

    @staticmethod
    def _quality_label(score):
        if score >= QUALITY_LABEL_THRESHOLDS["ACCEPTABLE"]:
            return "ACCEPTABLE"
        elif score >= QUALITY_LABEL_THRESHOLDS["DEGRADED"]:
            return "DEGRADED"
        return "DEFECTIVE"

    def analyze(self, image_bytes):
        bgr = self.decode_image(image_bytes)
        if bgr is None:
            raise ValueError("Could not decode image.")
        if bgr.shape[0] < 16 or bgr.shape[1] < 16:
            raise ValueError("Image too small (min 16x16).")

        feats = extract_features(bgr)
        cnn_score, cam = self._cnn_score_and_cam(bgr)
        issues = self._detect_issues(feats)

        # blend CNN score with issue penalties
        penalty = sum(i["confidence"] * 12 for i in issues if i["severity"] in ("high", "medium"))
        final_score = float(np.clip(cnn_score - penalty * 0.4, 0, 100))
        label = self._quality_label(final_score)

        cnn_label = self._quality_label(cnn_score)
        confidence = round(0.9 if cnn_label == label else 0.65, 2)

        return {
            "quality_score": round(final_score, 1),
            "quality_label": label,
            "confidence": confidence,
            "issues": issues,
            "image_stats": {
                "width": int(bgr.shape[1]),
                "height": int(bgr.shape[0]),
                "mean_brightness": round(feats.mean_brightness, 1),
                "sharpness_laplacian_var": round(feats.laplacian_variance, 1),
                "noise_sigma_estimate": round(feats.noise_sigma_estimate, 2),
                "contrast_rms": round(feats.rms_contrast, 3),
                "mean_saturation": round(feats.mean_saturation, 1),
                "colorfulness": round(feats.colorfulness, 1),
                "entropy": round(feats.entropy, 2),
                "dynamic_range": round(feats.dynamic_range, 1),
            },
            "model_signals": {
                "cnn_quality_score": round(cnn_score, 1),
                "cnn_implied_label": cnn_label,
                "model_type": self.model_type,
            },
            "original_image": self._encode_image(bgr),
            "heatmap": self._encode_heatmap(bgr, cam),
        }

    @staticmethod
    def _encode_image(bgr):
        h, w = bgr.shape[:2]
        max_dim = 600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"

    @staticmethod
    def _encode_heatmap(bgr, cam):
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.5, heat, 0.5, 0)
        ok, buf = cv2.imencode(".png", overlay)
        if not ok:
            return ""
        return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = QualityAnalyzer()
    return _analyzer
