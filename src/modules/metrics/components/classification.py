from pathlib import Path
from typing import Any
from weakref import ref

import numpy as np
import pandas as pd
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import FLOAT32_EPSILON, rank_zero_only
from torchmetrics import (
    AveragePrecision as TorchAveragePrecision,
    ConfusionMatrix as TorchConfusionMatrix,
    Metric,
)

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BalancedAccuracy(Metric):
    """Balanced Accuracy metric for multiclass classification.
    
    Balanced Accuracy is the average of recall obtained on each class.
    For multiclass: balanced_accuracy = (1/n_classes) * sum(recall_i)
    where recall_i = TP_i / (TP_i + FN_i) for class i.
    """

    def __init__(
        self,
        task: str = 'multiclass',
        num_classes: int = 2,
        dist_sync_on_step: bool = False,
        top_k: int | None = None,  # Ignored parameter for compatibility
        **kwargs: Any,  # Accept any additional kwargs for compatibility
    ) -> None:
        """Initialize BalancedAccuracy metric.

        :param task: Task type ('multiclass' or 'binary').
        :param num_classes: Number of classes.
        :param dist_sync_on_step: Whether to sync across devices on each step.
        :param top_k: Ignored parameter (for compatibility with other metrics).
        :param kwargs: Additional keyword arguments (ignored for compatibility).
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.task = task
        self.num_classes = num_classes
        
        # Use internal confusion matrix to compute per-class recall
        self._confusion_matrix = TorchConfusionMatrix(
            task=task, num_classes=num_classes
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        :param preds: Predicted class probabilities or indices.
        :param targets: True class indices.
        """
        # Convert to class indices if needed
        if preds.ndim > 1:
            preds_class = preds.argmax(dim=1)
        else:
            preds_class = preds
        
        self._confusion_matrix.update(preds_class, targets)

    def compute(self) -> torch.Tensor:
        """Compute balanced accuracy.

        :return: Balanced accuracy as a scalar tensor.
        """
        cm = self._confusion_matrix.compute()
        
        # In torchmetrics ConfusionMatrix, cm[i, j] = samples with true label i predicted as j
        # (same orientation as sklearn: rows=true, cols=predicted)
        # So for class i:
        # - TP_i = cm[i, i] (true label i predicted as i)
        # - FN_i = sum of all cm[i, j] where j != i (true label i predicted as other classes)
        # - Total samples with true label i = sum(cm[i, :]) (sum of row i)
        # - Recall_i = TP_i / (TP_i + FN_i) = cm[i, i] / sum(cm[i, :])
        
        # Ensure we work with float dtype
        cm_float = cm.float()
        recalls = []
        
        # Only compute recall for classes that appear in the targets
        # This matches sklearn's balanced_accuracy_score behavior
        for i in range(self.num_classes):
            total_class_i = cm_float[i, :].sum()  # Total samples with true label i (sum of row i)
            
            if total_class_i > 0:
                tp = cm_float[i, i]  # True positives for class i (true i, predicted i)
                recall = tp / total_class_i
                recalls.append(recall)
        
        # Balanced accuracy is the mean of per-class recalls (only for classes present)
        # If no classes are present, return 0
        if len(recalls) == 0:
            return torch.tensor(0.0, device=cm.device, dtype=torch.float32)
        
        balanced_acc = torch.stack(recalls).mean()
        
        return balanced_acc

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self._confusion_matrix.reset()


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.add_state(
            'total', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        diff = preds.argmax(dim=1).eq(targets)
        self.correct += diff.sum()
        self.total += targets.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (self.total + FLOAT32_EPSILON)


class NDCG(Metric):
    def __init__(self, dist_sync_on_step: bool = False, k: int = 10) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('ndcg', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state(
            'count', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.k = k

    def dcg_score(
        self, y_true: torch.Tensor, y_score: torch.Tensor
    ) -> torch.Tensor:
        # if sequence smaller than k
        sequence_length = y_score.shape[1]
        if sequence_length < self.k:
            k = sequence_length
        else:
            k = self.k
        _, order = torch.topk(input=y_score, k=k, largest=True)
        y_true = torch.take(y_true, order)
        gains = torch.pow(2, y_true) - 1
        discounts = torch.log2(
            torch.arange(y_true.shape[1]).type_as(y_score) + 2.0
        )
        return torch.sum(gains / discounts)

    def ndcg_score(
        self, y_true: torch.Tensor, y_score: torch.Tensor
    ) -> torch.Tensor:
        best = self.dcg_score(y_true, y_true)
        actual = self.dcg_score(y_true, y_score)
        return actual / best

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert preds.shape == targets.shape
        self.ndcg += self.ndcg_score(targets, preds)
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.ndcg / self.count


class MRR(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, k: int | None = None
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('mrr', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state(
            'count', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.k = k

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert preds.shape == targets.shape
        if self.k is None:
            order = torch.argsort(input=preds, descending=True)
        else:
            sequence_length = preds.shape[1]
            if sequence_length < self.k:
                k = sequence_length
            else:
                k = self.k
            _, order = torch.topk(input=preds, k=k, largest=True)

        y_true = torch.take(targets, order)
        rr_score = y_true / (torch.arange(y_true.shape[1]).type_as(preds) + 1)

        self.mrr += torch.sum(rr_score) / torch.sum(y_true)
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.mrr / self.count


class SentiMRR(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, k: int | None = None
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'senti_mrr', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.add_state(
            'count', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.k = k

    def update(
        self, y_pred: torch.Tensor, s_c: torch.Tensor, s_mean: torch.Tensor
    ) -> None:
        assert y_pred.shape == s_c.shape
        if self.k is None:
            order = torch.argsort(input=y_pred, descending=True)
        else:
            sequence_length = y_pred.shape[0]
            if sequence_length < self.k:
                k = sequence_length
            else:
                k = self.k
            _, order = torch.topk(input=y_pred, k=k, largest=True)

        s_c = torch.take(s_c, order)
        senti_rr_score = s_c / (torch.arange(s_c.shape[0]).type_as(s_c) + 1.0)
        senti_rr_score = s_mean * torch.sum(senti_rr_score)
        senti_rr_score = torch.nn.functional.relu(senti_rr_score)

        self.senti_mrr += senti_rr_score
        self.count += 1.0

    def compute(self) -> torch.Tensor:
        return self.senti_mrr / self.count


class PrecisionAtRecall(Metric):
    def __init__(
        self, dist_sync_on_step: bool = False, recall_point: float = 0.95
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            'correct', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.add_state(
            'wrong', default=torch.tensor(0.0), dist_reduce_fx='sum'
        )
        self.recall_point = recall_point

    def update(self, distances: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels[torch.argsort(distances)]
        # Sliding threshold: get first index where recall >= recall_point.
        # This is the index where the number of elements with label==below the threshold reaches a fraction of
        # 'recall_point' of the total number of elements with label==1.
        # (np.argmax returns the first occurrence of a '1' in a bool array).
        threshold_index = torch.where(
            torch.cumsum(labels, dim=0)
            >= self.recall_point * torch.sum(labels)
        )
        threshold_index = threshold_index[0][0]
        self.correct += torch.sum(labels[threshold_index:] == 0)
        self.wrong += torch.sum(labels[:threshold_index] == 0)

    def compute(self) -> torch.Tensor:
        return self.correct.float() / (
            self.correct + self.wrong + FLOAT32_EPSILON
        )


class MeanAveragePrecision(Metric):
    """Mean Average Precision (MAP) metric for classification.
    
    Mean Average Precision is the mean of Average Precision (AP) across all classes.
    For multiclass classification, it computes AP for each class independently
    and then takes the mean. This is also known as macro-averaged AP.
    
    Average Precision is the area under the precision-recall curve.
    """

    def __init__(
        self,
        task: str = 'multiclass',
        num_classes: int = 2,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize MeanAveragePrecision metric.

        :param task: Task type ('multiclass' or 'binary').
        :param num_classes: Number of classes.
        :param dist_sync_on_step: Whether to sync across devices on each step.
        :param kwargs: Additional keyword arguments passed to torchmetrics.AveragePrecision.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.task = task
        self.num_classes = num_classes
        
        # Use torchmetrics AveragePrecision with macro averaging
        # This computes AP for each class and averages them
        self._average_precision = TorchAveragePrecision(
            task=task,
            num_classes=num_classes,
            average='macro',
            **kwargs,
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        :param preds: Predicted class probabilities of shape (N, C) where N is batch size
            and C is number of classes. For binary classification with task='binary',
            can also be (N,) with probabilities of positive class.
        :param targets: True class indices of shape (N,).
        """
        # For binary classification, torchmetrics AveragePrecision expects (N,) for preds
        # Extract positive class probabilities if needed
        if self.task == 'binary' and preds.ndim > 1 and preds.size(1) == 2:
            # Extract probabilities of positive class (class 1)
            preds = preds[:, 1]
        
        self._average_precision.update(preds, targets)

    def compute(self) -> torch.Tensor:
        """Compute mean average precision.

        :return: Mean Average Precision as a scalar tensor.
        """
        return self._average_precision.compute()

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self._average_precision.reset()


# Alias for convenience
MAP = MeanAveragePrecision


class NegativeLogLikelihood(Metric):
    """Negative Log Likelihood metric with breakdown by class and prediction correctness.
    
    Computes NLL = -log(p(y_true | x)) where p is the predicted probability of the true class.
    Logs:
    - NLL_total: Average NLL over all samples
    - NLL_positive: Average NLL for positive class samples (class 1 for binary, or all classes > 0 for multiclass)
    - NLL_negative: Average NLL for negative class samples (class 0)
    - NLL_correct: Average NLL for correctly predicted samples
    - NLL_incorrect: Average NLL for incorrectly predicted samples
    """

    def __init__(
        self,
        task: str = 'multiclass',
        num_classes: int = 2,
        split: str = 'train',
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize NegativeLogLikelihood metric.

        :param task: Task type ('multiclass' or 'binary').
        :param num_classes: Number of classes.
        :param split: Split name (train/valid/test).
        :param dist_sync_on_step: Whether to sync across devices on each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.task = task
        self.num_classes = num_classes
        self.split = split
        
        # Store predictions and targets as regular lists
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        
        # Use weak reference to LightningModule to avoid circular dependency
        self._lightning_module_ref: ref[Any] | None = None

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.preds.clear()
        self.targets.clear()

    def setup_lightning_module(self, lightning_module: Any) -> None:
        """Set weak reference to LightningModule for accessing trainer and logger.

        :param lightning_module: LightningModule instance.
        """
        self._lightning_module_ref = ref(lightning_module)
    
    @property
    def lightning_module(self) -> Any | None:
        """Get LightningModule instance from weak reference."""
        if self._lightning_module_ref is None:
            return None
        return self._lightning_module_ref()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions and targets.

        :param preds: Predicted class probabilities of shape (N, C) or (N,) for binary.
        :param targets: True class indices of shape (N,).
        """
        # Ensure predictions are probabilities in correct format
        # For binary classification with task='binary', preds might be (N,)
        # For multiclass or binary with task='multiclass', preds should be (N, C)
        if preds.ndim == 1:
            if self.task == 'binary' and self.num_classes == 2:
                # Binary case: convert (N,) to (N, 2)
                preds = torch.stack([1 - preds, preds], dim=1)
            else:
                # Should not happen, but handle gracefully
                raise ValueError(f"Unexpected preds shape {preds.shape} for task={self.task}, num_classes={self.num_classes}")
        elif preds.ndim == 2 and preds.size(1) == 1:
            # Binary case with (N, 1): convert to (N, 2)
            if self.task == 'binary' and self.num_classes == 2:
                preds = torch.cat([1 - preds, preds], dim=1)
        
        # Ensure preds sum to 1 (normalize if needed, though they should already be probabilities)
        # This is a safety check
        preds_sum = preds.sum(dim=1, keepdim=True)
        if not torch.allclose(preds_sum, torch.ones_like(preds_sum), atol=1e-3):
            # Normalize to ensure they sum to 1
            preds = preds / (preds_sum + 1e-8)
        
        # Move to CPU and store
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        
        self.preds.append(preds)
        self.targets.append(targets)

    @rank_zero_only
    def compute(self) -> torch.Tensor:
        """Compute NLL and perform logging.

        :return: Average NLL as a scalar tensor.
        """
        if not self.preds or not self.targets:
            return torch.tensor(0.0)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)
        
        # Ensure on CPU
        all_preds = all_preds.cpu()
        all_targets = all_targets.cpu().long()
        
        # Compute NLL for each sample: -log(p(y_true | x))
        # Get probability of true class for each sample
        probs_true_class = all_preds.gather(1, all_targets.unsqueeze(1)).squeeze(1)
        # Add small epsilon to avoid log(0)
        probs_true_class = torch.clamp(probs_true_class, min=1e-8)
        nll_per_sample = -torch.log(probs_true_class)
        
        # Compute overall NLL
        nll_total = nll_per_sample.mean()
        
        # For binary classification: positive = class 1, negative = class 0
        # For multiclass: positive = classes > 0, negative = class 0
        if self.num_classes == 2:
            # Binary case
            positive_mask = all_targets == 1
            negative_mask = all_targets == 0
        else:
            # Multiclass case: positive = any class > 0
            positive_mask = all_targets > 0
            negative_mask = all_targets == 0
        
        # Compute NLL for positive and negative samples
        nll_positive = nll_per_sample[positive_mask].mean() if positive_mask.any() else torch.tensor(0.0)
        nll_negative = nll_per_sample[negative_mask].mean() if negative_mask.any() else torch.tensor(0.0)
        
        # Compute predictions (class indices)
        pred_classes = all_preds.argmax(dim=1)
        correct_mask = pred_classes == all_targets
        incorrect_mask = pred_classes != all_targets
        
        # Compute NLL for correct and incorrect predictions
        nll_correct = nll_per_sample[correct_mask].mean() if correct_mask.any() else torch.tensor(0.0)
        nll_incorrect = nll_per_sample[incorrect_mask].mean() if incorrect_mask.any() else torch.tensor(0.0)
        
        # Log all values
        if self.lightning_module is not None:
            logging_params = {
                'on_step': False,
                'on_epoch': True,
                'sync_dist': False,
                'prog_bar': False,
            }
            
            self.lightning_module.log(
                f'NLL/{self.split}',
                nll_total.item(),
                **logging_params,
            )
            self.lightning_module.log(
                f'NLL_positive/{self.split}',
                nll_positive.item() if isinstance(nll_positive, torch.Tensor) else nll_positive,
                **logging_params,
            )
            self.lightning_module.log(
                f'NLL_negative/{self.split}',
                nll_negative.item() if isinstance(nll_negative, torch.Tensor) else nll_negative,
                **logging_params,
            )
            self.lightning_module.log(
                f'NLL_correct/{self.split}',
                nll_correct.item() if isinstance(nll_correct, torch.Tensor) else nll_correct,
                **logging_params,
            )
            self.lightning_module.log(
                f'NLL_incorrect/{self.split}',
                nll_incorrect.item() if isinstance(nll_incorrect, torch.Tensor) else nll_incorrect,
                **logging_params,
            )
        
        # Log to console
        log.info(f"\n{'='*60}")
        log.info(f"Negative Log Likelihood - {self.split.upper()}")
        log.info(f"{'='*60}")
        log.info(f"NLL (total): {nll_total.item():.4f}")
        log.info(f"NLL (positive): {nll_positive.item() if isinstance(nll_positive, torch.Tensor) else nll_positive:.4f}")
        log.info(f"NLL (negative): {nll_negative.item() if isinstance(nll_negative, torch.Tensor) else nll_negative:.4f}")
        log.info(f"NLL (correct): {nll_correct.item() if isinstance(nll_correct, torch.Tensor) else nll_correct:.4f}")
        log.info(f"NLL (incorrect): {nll_incorrect.item() if isinstance(nll_incorrect, torch.Tensor) else nll_incorrect:.4f}")
        log.info(f"{'='*60}\n")
        
        # Clear stored predictions and targets
        self.preds.clear()
        self.targets.clear()
        
        # Return average NLL (main metric value)
        return nll_total


# Alias for convenience
NLL = NegativeLogLikelihood


class ConfusionMatrixMetric(Metric):
    """Confusion Matrix metric with automatic logging to console, file, and WandB.

    This metric accumulates predictions and targets during an epoch,
    then computes and logs the confusion matrix in compute().
    """

    def __init__(
        self,
        task: str = 'multiclass',
        num_classes: int = 2,
        class_names: list[str] | None = None,
        split: str = 'train',
        dist_sync_on_step: bool = False,
    ) -> None:
        """Initialize ConfusionMatrixMetric.

        :param task: Task type ('multiclass' or 'binary').
        :param num_classes: Number of classes.
        :param class_names: List of class names for logging.
        :param split: Split name (train/valid/test).
        :param dist_sync_on_step: Whether to sync across devices on each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        # Store configuration
        self.task = task
        self.num_classes = num_classes
        self.split = split
        
        # Convert class_names from ListConfig to regular list if needed
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        if isinstance(class_names, (DictConfig, list)):
            class_names = OmegaConf.to_container(class_names, resolve=True) if isinstance(class_names, DictConfig) else list(class_names)
        self.class_names = class_names
        
        # Store predictions and targets as regular lists
        # (not using add_state as lists are not directly supported)
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        
        # Internal confusion matrix metric (always on CPU)
        # Store as regular attribute with underscore prefix to exclude from children()
        # This prevents recursion when PyTorch applies operations to the module
        self._confusion_matrix = TorchConfusionMatrix(
            task=task, num_classes=num_classes
        ).to('cpu')
        
        # Explicitly register as buffer/parameter to exclude from children traversal
        # But we'll manage it manually to avoid device movement issues
        
        # Use weak reference to LightningModule to avoid circular dependency
        # This prevents recursion errors when PyTorch applies operations to the module
        self._lightning_module_ref: ref[Any] | None = None

    def reset(self) -> None:
        """Reset metric state."""
        super().reset()
        self.preds.clear()
        self.targets.clear()

    def setup_lightning_module(self, lightning_module: Any) -> None:
        """Set weak reference to LightningModule for accessing trainer and logger.

        :param lightning_module: LightningModule instance.
        """
        self._lightning_module_ref = ref(lightning_module)
    
    @property
    def lightning_module(self) -> Any | None:
        """Get LightningModule instance from weak reference."""
        if self._lightning_module_ref is None:
            return None
        return self._lightning_module_ref()

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric with new predictions and targets.

        :param preds: Predicted class indices or probabilities.
        :param targets: True class indices.
        """
        # Convert to class indices if needed
        if preds.ndim > 1:
            preds_class = preds.argmax(dim=1)
        else:
            preds_class = preds
        
        # Move to CPU and store
        preds_class = preds_class.detach().cpu()
        targets = targets.detach().cpu()
        
        self.preds.append(preds_class)
        self.targets.append(targets)

    @rank_zero_only
    def compute(self) -> torch.Tensor:
        """Compute confusion matrix and perform logging.

        :return: Dummy tensor (metric value not used, logging is the main purpose).
        """
        if not self.preds or not self.targets:
            return torch.tensor(0.0)
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.preds)
        all_targets = torch.cat(self.targets)
        
        # Ensure on CPU
        all_preds = all_preds.cpu()
        all_targets = all_targets.cpu()
        
        # Reset and compute confusion matrix
        # Move _confusion_matrix to CPU explicitly before use
        confusion_matrix_cpu = self._confusion_matrix.to('cpu')
        confusion_matrix_cpu.reset()
        confusion_matrix_cpu.update(all_preds, all_targets)
        cm = confusion_matrix_cpu.compute().cpu().numpy()
        confusion_matrix_cpu.reset()
        
        # Perform logging
        self._log_to_console(cm)
        self._save_to_file(cm)
        self._log_to_wandb(cm, all_preds, all_targets)
        
        # Clear stored predictions and targets
        self.preds.clear()
        self.targets.clear()
        
        # Return dummy value (not used)
        return torch.tensor(0.0)

    @rank_zero_only
    def _log_to_console(self, cm: np.ndarray) -> None:
        """Log confusion matrix to console as a formatted table.

        :param cm: Confusion matrix as numpy array.
        """
        log.info(f"\n{'='*60}")
        log.info(f"Confusion Matrix - {self.split.upper()}")
        log.info(f"{'='*60}")

        # Create header row
        header = ["Pred\\True"] + [name[:10] for name in self.class_names]
        header_str = " | ".join(f"{h:>12}" for h in header)
        log.info(header_str)
        log.info("-" * len(header_str))

        # Create data rows
        for i, class_name in enumerate(self.class_names):
            row = [class_name[:10]] + [f"{cm[i, j]:>12}" for j in range(len(self.class_names))]
            row_str = " | ".join(f"{r:>12}" for r in row)
            log.info(row_str)

        log.info(f"{'='*60}\n")

    @rank_zero_only
    def _save_to_file(self, cm: np.ndarray) -> None:
        """Save confusion matrix to CSV file in split-specific folder.

        :param cm: Confusion matrix as numpy array.
        """
        try:
            # Get output directory - prefer Hydra's output directory, fallback to trainer
            output_dir = None
            
            # Try to get from Hydra first
            try:
                hydra_instance = GlobalHydra.instance()
                if hydra_instance.is_initialized():
                    hydra_cfg = hydra_instance.hydra
                    if hasattr(hydra_cfg, 'runtime') and hasattr(hydra_cfg.runtime, 'output_dir'):
                        output_dir = Path(hydra_cfg.runtime.output_dir)
            except Exception:
                pass
            
            # Fallback to trainer's default_root_dir
            if output_dir is None and self.lightning_module is not None:
                if hasattr(self.lightning_module, 'trainer') and self.lightning_module.trainer is not None:
                    if hasattr(self.lightning_module.trainer, 'default_root_dir') and self.lightning_module.trainer.default_root_dir:
                        output_dir = Path(self.lightning_module.trainer.default_root_dir)
            
            if output_dir is None:
                log.warning("Could not determine output directory, skipping confusion matrix file saving")
                return
            
            # Create split-specific directory
            confusion_matrix_dir = output_dir / 'confusion_matrices' / self.split
            confusion_matrix_dir.mkdir(parents=True, exist_ok=True)

            # Get current epoch
            epoch = 0
            if self.lightning_module is not None:
                if hasattr(self.lightning_module, 'trainer') and self.lightning_module.trainer is not None:
                    if hasattr(self.lightning_module.trainer, 'current_epoch'):
                        epoch = self.lightning_module.trainer.current_epoch

            # Save as CSV with format epochXXX.csv
            df_cm = pd.DataFrame(
                cm,
                index=self.class_names,
                columns=self.class_names,
            )
            csv_path = confusion_matrix_dir / f'epoch{epoch:03d}.csv'
            df_cm.to_csv(csv_path)
            log.info(f"Saved confusion matrix CSV to: {csv_path}")

        except Exception as e:
            log.warning(f"Failed to save confusion matrix to file: {e}")

    @rank_zero_only
    def _log_to_wandb(
        self, cm: np.ndarray, preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Log confusion matrix to wandb using wandb.plot.confusion_matrix().

        :param cm: Confusion matrix as numpy array (for reference, not used for plotting).
        :param preds: Predicted class indices tensor.
        :param targets: True class indices tensor.
        """
        # Check if wandb logger is available
        if self.lightning_module is None:
            return
        
        if not hasattr(self.lightning_module, 'logger') or self.lightning_module.logger is None:
            return

        wandb_logger = None
        if isinstance(self.lightning_module.logger, WandbLogger):
            wandb_logger = self.lightning_module.logger
        elif isinstance(self.lightning_module.logger, list):
            for logger in self.lightning_module.logger:
                if isinstance(logger, WandbLogger):
                    wandb_logger = logger
                    break

        if wandb_logger is None:
            return

        try:
            import wandb

            # Convert tensors to numpy arrays and then to lists
            y_true = targets.cpu().numpy().tolist()
            y_pred = preds.cpu().numpy().tolist()
            
            # Get current step/epoch for proper logging
            step = None
            if self.lightning_module is not None:
                if hasattr(self.lightning_module, 'trainer') and self.lightning_module.trainer is not None:
                    if hasattr(self.lightning_module.trainer, 'current_epoch'):
                        step = self.lightning_module.trainer.current_epoch
                    elif hasattr(self.lightning_module.trainer, 'global_step'):
                        step = self.lightning_module.trainer.global_step
            
            # Use WandB's native confusion matrix plotting
            conf_matrix = wandb.plot.confusion_matrix(
                y_true=y_true,
                preds=y_pred,
                class_names=self.class_names,
                title=f'Confusion Matrix - {self.split.upper()}',
            )
            
            # Log the confusion matrix
            wandb_logger.experiment.log(
                {
                    f'confusion_matrix/{self.split}': conf_matrix,
                },
                step=step,
                commit=True,
            )
            
            log.info(f"Logged confusion matrix to WandB for {self.split} split (epoch {step})")
        except ImportError as e:
            log.warning(f"Required packages not available (wandb): {e}, skipping confusion matrix logging to wandb")
        except Exception as e:
            log.warning(f"Failed to log confusion matrix to wandb: {e}")
