import torch
from torch.nn import functional as fn


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces the given tensor using a specific criterion.

    Args:
        tensor (torch.Tensor): input tensor
        reduction (str): string with fixed values [elementwise_mean, none, sum]
    Raises:
        ValueError: when the reduction is not supported
    Returns:
        torch.Tensor: reduced tensor, or the tensor itself
    """
    if reduction in ("elementwise_mean", "mean"):
        return torch.mean(tensor)
    elif reduction == "sum":
        return torch.sum(tensor)
    elif reduction is None or reduction == "none":
        return tensor
    raise ValueError("Reduction parameter unknown.")


class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Focal Loss is designed to address the one-stage object detection scenario
    where there is an extreme imbalance between foreground and background classes.
    It down-weights easy examples and focuses training on hard negatives.
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float): Weighting factor for the rare class. Default: 1.0
        gamma (float): Focusing parameter. Higher gamma focuses more on hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output.
            Options: 'none', 'mean', 'sum'. Default: 'mean'
        ignore_index (int): Specifies a target value that is ignored and does
            not contribute to the input gradient. Default: 255
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the focal loss.
        
        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) where N is batch size
                and C is number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,) with class indices.
        
        Returns:
            torch.Tensor: The computed focal loss value.
        """
        ce_loss = fn.cross_entropy(
            inputs, targets, reduction="none", ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        return reduce(focal_loss, reduction=self.reduction)


