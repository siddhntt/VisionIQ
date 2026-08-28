"""
CNN models for quality score regression.

Two options:
  - QualityCNN: small custom CNN (~150K params), fast on CPU
  - MobileNetV2Quality: transfer learning from ImageNet, better accuracy
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class QualityCNN(nn.Module):
    """Lightweight CNN baseline. Input: (B, 3, 128, 128) -> scalar score."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.regressor(self.features(x)).squeeze(-1)

    def forward_with_activation(self, x):
        """Returns (score, last_conv_features) for Grad-CAM."""
        feat = self.features(x)
        score = self.regressor(feat)
        return score.squeeze(-1), feat


class MobileNetV2Quality(nn.Module):
    """MobileNetV2 pretrained on ImageNet, fine-tuned for quality regression."""

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights).features  # 1280 ch output

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.regressor(self.pool(self.backbone(x))).squeeze(-1)

    def forward_with_activation(self, x):
        feat = self.backbone(x)
        score = self.regressor(self.pool(feat))
        return score.squeeze(-1), feat

    def unfreeze_backbone(self, from_layer=-5):
        """Unfreeze the last N layers for fine-tuning."""
        for layer in list(self.backbone.children())[from_layer:]:
            for p in layer.parameters():
                p.requires_grad = True
