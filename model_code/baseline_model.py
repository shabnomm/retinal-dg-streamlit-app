from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


FREEZE_LAYERS = ("layer1", "layer2")


class ERMResNet50(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.5) -> None:
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for name, param in backbone.named_parameters():
            if name.split(".")[0] in FREEZE_LAYERS:
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits


def build_model() -> nn.Module:
    return ERMResNet50(num_classes=4, dropout=0.5)