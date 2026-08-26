from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    embedding_dim: int = 576


class FruitClassifier(nn.Module):
    """MobileNetV3-Small classifier that also exposes dataset embeddings."""

    def __init__(self, num_classes: int, pretrained: bool = False) -> None:
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        embedding_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_last_blocks(self, blocks: int = 4) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        feature_blocks: Sequence[nn.Module] = list(self.backbone.features.children())
        for block in feature_blocks[-blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
