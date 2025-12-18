# src/modules/losses/components/dice_loss.py
import torch
import torch.nn.functional as F

from src.modules.losses.components.base_loss import BaseLoss


class DiceLoss(BaseLoss):
    """Dice loss implementation for semantic segmentation tasks.

    The Dice loss is defined as:
    Dice Loss = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)

    where X is the prediction and Y is the ground truth.
    """

    def __init__(self, smooth: float = 1e-5, reduction: str = 'mean'):
        """Initialize the Dice loss.

        :param smooth: Smoothing factor to avoid division by zero
        :param reduction: Specifies the reduction to apply to the output ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the Dice loss between predictions and targets.

        :param pred: Predicted tensor of shape (N, C, ...) where N is batch size,
            C is number of classes, and ... are spatial dimensions
        :param target: Target tensor of shape (N, ...) where N is batch size
            and ... are spatial dimensions
        :return: Computed Dice loss
        """
        # Ensure predictions are probabilities (apply softmax if needed)
        if pred.dim() > 2:
            # For multi-class case, apply softmax to get probabilities
            pred = F.softmax(pred, dim=1)

        # Flatten predictions and targets
        pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
        target = target.contiguous().view(target.size(0), -1)

        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
        target_one_hot = target_one_hot.permute(0, 2, 1).contiguous()

        # Compute intersection and union
        intersection = torch.sum(pred * target_one_hot, dim=2)
        union = torch.sum(pred, dim=2) + torch.sum(target_one_hot, dim=2)

        # Compute Dice coefficient
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Compute Dice loss
        dice_loss = 1. - dice_coeff

        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss
