from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


FREEZE_LAYERS = ("layer1", "layer2")


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


class DGADGResNet50(nn.Module):
    def __init__(self, num_classes: int = 4, num_domains: int = 2, dropout: float = 0.5) -> None:
        super().__init__()

        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for name, param in backbone.named_parameters():
            if name.split(".")[0] in FREEZE_LAYERS:
                param.requires_grad = False

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.cls_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.grl = GradientReversalLayer(lambda_=0.0)

        self.domain_disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_domains),
        )

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        cls_logits = self.cls_head(feats)
        dom_logits = self.domain_disc(self.grl(feats))
        return cls_logits, dom_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls_head(self.backbone(x))


def build_model() -> nn.Module:
    return DGADGResNet50(num_classes=4, num_domains=2, dropout=0.5)