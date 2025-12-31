from typing import Any

import torch
from torch import nn


class EEGNet(nn.Module):
    """Modern CNN architecture for EEG signal classification.

    Supports configurable number of channels and signal length.
    Based on EEGNet architecture with depthwise and separable convolutions.
    """

    def __init__(
        self,
        num_channels: int = 127,
        signal_length: int = 2500,
        num_classes: int = 2,
        F1: int = 8,
        F2: int = 16,
        D: int = 2,
        kernel_length: int = 64,
        pool_size: int = 4,
        dropout_rate: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """EEGNet initialization.

        :param num_channels: Number of EEG channels.
        :param signal_length: Length of EEG signal (time points).
        :param num_classes: Number of output classes.
        :param F1: Number of temporal filters.
        :param F2: Number of pointwise filters.
        :param D: Depth multiplier for depthwise convolution.
        :param kernel_length: Length of temporal convolution kernel.
        :param pool_size: Size of pooling window.
        :param dropout_rate: Dropout rate.
        :param kwargs: Additional arguments (ignored).
        """
        super().__init__()

        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise Convolution
        self.conv2 = nn.Conv2d(
            F1,
            F1 * D,
            (num_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool_size))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Block 3: Separable Convolution
        self.conv3 = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened size after convolutions
        # After pool1: signal_length // pool_size
        # After pool2: (signal_length // pool_size) // 8
        # Use a dummy forward pass to calculate actual size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_channels, signal_length)
            # Block 1
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.elu1(x)
            x = self.pool1(x)
            x = self.dropout1(x)
            # Block 3
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.bn3(x)
            x = self.elu2(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            flattened_size = x.view(1, -1).size(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_classes),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers.

        :param x: Input tensor.
        :return: Feature tensor before classification.
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch, 1, channels, length).
        :return: Output logits of shape (batch, num_classes).
        """
        x = self._forward_features(x)
        x = self.classifier(x)
        return x


class EEGResNet(nn.Module):
    """ResNet-inspired architecture for EEG classification.

    Modern architecture with residual connections, supports configurable
    channels and signal length.
    """

    def __init__(
        self,
        num_channels: int = 127,
        signal_length: int = 2500,
        num_classes: int = 2,
        base_filters: int = 64,
        num_blocks: int = 4,
        dropout_rate: float = 0.3,
        **kwargs: Any,
    ) -> None:
        """EEGResNet initialization.

        :param num_channels: Number of EEG channels.
        :param signal_length: Length of EEG signal (time points).
        :param num_classes: Number of output classes.
        :param base_filters: Base number of filters.
        :param num_blocks: Number of residual blocks.
        :param dropout_rate: Dropout rate.
        :param kwargs: Additional arguments (ignored).
        """
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, base_filters, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        in_filters = base_filters
        for i in range(num_blocks):
            out_filters = base_filters * (2 ** (i // 2))
            self.blocks.append(
                self._make_residual_block(
                    in_filters, out_filters, kernel_size=(1, 3)
                )
            )
            in_filters = out_filters

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_filters, in_filters // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_filters // 2, num_classes),
        )

    def _make_residual_block(
        self, in_filters: int, out_filters: int, kernel_size: tuple[int, int]
    ) -> nn.Module:
        """Create residual block.

        :param in_filters: Input filters.
        :param out_filters: Output filters.
        :param kernel_size: Convolution kernel size.
        :return: Residual block module.
        """
        return ResidualBlock(in_filters, out_filters, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch, 1, channels, length).
        :return: Output logits of shape (batch, num_classes).
        """
        x = self.initial_conv(x)

        for block in self.blocks:
            x = block(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block for EEGResNet."""

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        kernel_size: tuple[int, int],
    ) -> None:
        """ResidualBlock initialization.

        :param in_filters: Input filters.
        :param out_filters: Output filters.
        :param kernel_size: Convolution kernel size.
        """
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(
            in_filters,
            out_filters,
            kernel_size,
            padding=(0, kernel_size[1] // 2),
        )
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_filters,
            out_filters,
            kernel_size,
            padding=(0, kernel_size[1] // 2),
        )
        self.bn2 = nn.BatchNorm2d(out_filters)

        # Shortcut path
        if in_filters != out_filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1)),
                nn.BatchNorm2d(out_filters),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor.
        :return: Output tensor.
        """
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)

        return out

