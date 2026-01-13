"""Base loss class for custom loss implementations."""

import torch


class BaseLoss(torch.nn.Module):
    """Base class for custom loss functions.
    
    All custom loss functions should inherit from this class.
    This provides a consistent interface for loss implementations.
    """

    def __init__(self):
        """Initialize the base loss."""
        super().__init__()
