from typing import Any

import torch
from torch import nn


class SmallEEGNet(nn.Module):
    """A smaller EEG convnet: only 2 conv layers, fewer features.
    
    Based on the working pipeline from raw_classification_v1.ipynb.
    Assumes input shape: (batch_size, 30, 2500)
    """

    def __init__(
        self,
        num_channels: int = 30,
        signal_length: int = 2500,
        num_classes: int = 2,
        **kwargs: Any,
    ) -> None:
        """SmallEEGNet initialization.

        :param num_channels: Number of EEG channels.
        :param signal_length: Length of EEG signal (time points).
        :param num_classes: Number of output classes.
        :param kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.slicing = nn.Identity()
        
        # 1st conv block: Conv1d (5), BatchNorm, SiLU
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.act1 = nn.SiLU()
        
        # 2nd conv block: Conv1d (7), BatchNorm, SiLU
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.SiLU()
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.fc = nn.Linear(64, num_classes)
        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: Input tensor of shape (batch, channels, length).
        :return: Output logits of shape (batch, num_classes).
        """
        # Convert from (batch, 1, channels, length) to (batch, channels, length) if needed
        if x.ndim == 4:
            x = x.squeeze(1)
        
        x = self.slicing(x)            # (batch, channels, length)
        x = self.conv1(x)              # (batch, 32, length)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.conv2(x)              # (batch, 64, length)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.global_pool(x)        # (batch, 64, 1)
        x = x.squeeze(-1)              # (batch, 64)
        x = self.fc(x)                 # (batch, num_classes)
        return x

