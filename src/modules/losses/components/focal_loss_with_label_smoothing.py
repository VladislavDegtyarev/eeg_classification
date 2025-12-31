import torch
from torch.nn import functional as fn


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces the given tensor using a specific criterion.

    :param tensor: input tensor
    :param reduction: string with fixed values [elementwise_mean, none, sum]
    :raises ValueError: when the reduction is not supported
    :return: reduced tensor, or the tensor itself
    """
    if reduction in ('elementwise_mean', 'mean'):
        return torch.mean(tensor)
    elif reduction == 'sum':
        return torch.sum(tensor)
    elif reduction is None or reduction == 'none':
        return tensor
    raise ValueError('Reduction parameter unknown.')


class FocalLossWithLabelSmoothing(torch.nn.Module):
    """Focal Loss with Label Smoothing for addressing class imbalance.

    Combines Focal Loss with Label Smoothing regularization.
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Label Smoothing helps prevent overconfidence.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    with label smoothing applied to targets.

    :param gamma: Focusing parameter. Higher gamma focuses more on hard examples. Default: 2.0
    :param label_smoothing: Label smoothing factor. Default: 0.1
    :param reduction: Specifies the reduction to apply to the output.
        Options: 'none', 'mean', 'sum'. Default: 'mean'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the focal loss with label smoothing.

        :param inputs: Predicted logits of shape (N, C) where N is batch size
            and C is number of classes.
        :param targets: Ground truth labels of shape (N,) with class indices.
        :return: The computed focal loss value.
        """
        num_classes = inputs.size(1)

        # Label smoothing: create smoothed one-hot targets
        with torch.no_grad():
            one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
            smoothed_targets = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        log_probs = fn.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = -torch.sum(smoothed_targets * log_probs, dim=1)

        p_t = torch.sum(smoothed_targets * probs, dim=1)
        focal_factor = (1 - p_t).pow(self.gamma)

        loss = focal_factor * ce_loss

        return reduce(loss, reduction=self.reduction)

