from typing import Any

import torch
from torch import nn


class ShallowConvNet(nn.Module):
    """Simplified shallow CNN architecture for EEG signal classification.

    Based on the working pipeline from dataset_preparation.ipynb.
    Simpler architecture with Conv1d, BatchNorm, AvgPool, and Dropout.
    Supports configurable number of channels and signal length.
    """

    def __init__(
        self,
        num_channels: int = 127,
        signal_length: int = 2500,
        num_classes: int = 2,
        out_channels: int = 32,
        kernel_size: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout_rate: float = 0.15,
        **kwargs: Any,
    ) -> None:
        """ShallowConvNet initialization.

        :param num_channels: Number of EEG channels.
        :param signal_length: Length of EEG signal (time points).
        :param num_classes: Number of output classes.
        :param out_channels: Number of output channels after first convolution.
        :param kernel_size: Size of temporal convolution kernel.
        :param pool_size: Size of pooling window.
        :param pool_stride: Stride of pooling.
        :param dropout_rate: Dropout rate.
        :param kwargs: Additional arguments (ignored).
        """
        super().__init__()

        # Input format: (batch, 1, channels, length)
        # We'll squeeze the first dimension to get (batch, channels, length) for Conv1d
        self.conv_time = nn.Conv1d(
            num_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.AvgPool1d(pool_size, stride=pool_stride, padding=pool_size // 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate output size after pooling
        pool_len = (signal_length + 2 * (pool_size // 2) - pool_size) // pool_stride + 1
        self.fc = nn.Linear(out_channels * pool_len, num_classes)

        # Initialize classifier head with smaller gain for better starting values
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch, 1, channels, length).
        :return: Output logits of shape (batch, num_classes).
        """
        # Convert from (batch, 1, channels, length) to (batch, channels, length)
        if x.ndim == 4:
            x = x.squeeze(1)  # Remove the channel dimension added by dataset

        # Temporal convolution
        x = self.conv_time(x)
        x = self.bn(x)
        # ReLU activation (no powers or logarithm as in original working pipeline)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classification
        x = self.fc(x)
        return x

