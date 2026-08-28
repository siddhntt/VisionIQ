"""Pydantic response models."""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class IssueEvidence(BaseModel):
    feature: str
    value: float
    direction: str

class IssueDetail(BaseModel):
    type: str
    severity: str
    confidence: float
    evidence: list[IssueEvidence] = []

class ImageStats(BaseModel):
    width: int
    height: int
    mean_brightness: float
    sharpness_laplacian_var: float
    noise_sigma_estimate: float
    contrast_rms: float
    mean_saturation: float
    colorfulness: float
    entropy: float
    dynamic_range: float

class ModelSignals(BaseModel):
    cnn_quality_score: float
    cnn_implied_label: str
    model_type: str

class AnalysisResponse(BaseModel):
    id: Optional[int] = None
    filename: str = ""
    quality_score: float
    quality_label: str
    confidence: float
    issues: list[IssueDetail]
    image_stats: ImageStats
    model_signals: ModelSignals
    heatmap: str = ""
    created_at: Optional[str] = None

class AnalysisListItem(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    confidence: float
    issues: list[IssueDetail]
    image_stats: dict = {}
    model_signals: dict = {}
    heatmap: str = ""
    created_at: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    total_analyses: int
    uptime_seconds: float
