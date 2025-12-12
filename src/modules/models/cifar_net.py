from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(
    in_channels: int, out_channels: int, dropout: float
) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(dropout),
    ]


class CIFARNet(nn.Module):
    """Compact CNN for CIFAR-10 classification."""

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        feature_plan = [
            (input_channels, 64, 0.1),
            (64, 128, 0.2),
            (128, 256, 0.3),
            (256, 512, 0.4),
        ]

        feature_layers: list[nn.Module] = []
        for in_ch, out_ch, dropout in feature_plan:
            feature_layers.extend(_conv_block(in_ch, out_ch, dropout))
        self.features = nn.Sequential(*feature_layers)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
