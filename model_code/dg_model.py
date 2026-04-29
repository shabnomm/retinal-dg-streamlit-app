from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


FREEZE_LAYERS = ("layer1", "layer2")


# ---------------------------------------------------------------------
# Shared GRL for DANN
# ---------------------------------------------------------------------
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_


# ---------------------------------------------------------------------
# Mixup DG model
# ---------------------------------------------------------------------
class MixupResNet50(nn.Module):
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
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------
# CORAL DG model
# ---------------------------------------------------------------------
class CORALResNet50(nn.Module):
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

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------
# DANN DG model
# ---------------------------------------------------------------------
class DANNResNet50(nn.Module):
    def __init__(self, num_classes: int = 4, num_domains: int = 2, dropout: float = 0.5) -> None:
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

        self.grl = GradientReversalLayer(lambda_=0.0)

        self.domain_discriminator = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_domains),
        )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        class_logits = self.head(feats)
        domain_logits = self.domain_discriminator(self.grl(feats))
        return class_logits, domain_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def build_model_for_dataset(dataset_context: str) -> nn.Module:
    """
    Dataset-specific DG best model mapping:
    - ODIR -> Mixup
    - RFMiD v1 -> CORAL
    - RFMiD v2 -> DANN
    """
    if dataset_context == "ODIR":
        return MixupResNet50(num_classes=4, dropout=0.5)

    if dataset_context == "RFMiD v1":
        return CORALResNet50(num_classes=4, dropout=0.5)

    if dataset_context == "RFMiD v2":
        return DANNResNet50(num_classes=4, num_domains=2, dropout=0.5)

    raise ValueError(f"Unsupported DG dataset context: {dataset_context}")


def build_model() -> nn.Module:
    """
    Default fallback.
    Not ideal for dataset-specific loading, but kept for compatibility.
    """
    return MixupResNet50(num_classes=4, dropout=0.5)