from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics
from src.modules.metrics.components.classification import ConfusionMatrixMetric
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SingleLitModule(BaseLitModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        class_names: list[str] | None = None,
        num_classes: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        :param network: Network config.
        :param optimizer: Optimizer config.
        :param scheduler: Scheduler config.
        :param logging: Logging config.
        :param class_names: List of class names for confusion matrix logging.
        :param num_classes: Number of classes. If None, inferred from network config.
        :param args: Additional arguments for pytorch_lightning.LightningModule.
        :param kwargs: Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        self.output_activation = hydra.utils.instantiate(
            network.output_activation, _partial_=True
        )

        # Get num_classes from network config if not provided
        if num_classes is None:
            num_classes = network.metrics.main.get('num_classes', 2)
        
        # Get class names or create default
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]
        self.class_names = class_names
        self.num_classes = num_classes

        main_metric, valid_metric_best, add_metrics = load_metrics(
            network.metrics
        )
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(postfix='/train')
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(postfix='/valid')
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix='/test')

        # Setup ConfusionMatrixMetric instances after cloning
        # Need to set split based on which add_metrics collection they're in
        self._setup_confusion_matrix_metrics()

        self.save_hyperparameters(logger=False)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        logits = self.forward(batch['image'])
        loss = self.loss(logits, batch['label'])
        preds = self.output_activation(logits)
        return loss, preds, batch['label']

    def _setup_confusion_matrix_metrics(self) -> None:
        """Setup ConfusionMatrixMetric instances from add_metrics.
        
        Finds ConfusionMatrixMetric in train/valid/test add_metrics,
        sets correct split based on which collection they're in,
        and sets up reference to LightningModule for accessing trainer and logger.
        """
        # Setup confusion matrix metrics for train
        for metric_name, metric in self.train_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.split = 'train'
                metric.setup_lightning_module(self)
        
        # Setup confusion matrix metrics for valid
        for metric_name, metric in self.valid_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.split = 'valid'
                metric.setup_lightning_module(self)
        
        # Setup confusion matrix metrics for test
        for metric_name, metric in self.test_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.split = 'test'
                metric.setup_lightning_module(self)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/train',
            loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f'{self.train_metric.__class__.__name__}/train',
            self.train_metric,
            **self.logging_params,
        )

        # Update metrics, excluding ConfusionMatrixMetric (handled separately)
        metrics_to_update = {k: v for k, v in self.train_add_metrics.items() 
                            if not isinstance(v, ConfusionMatrixMetric)}
        if metrics_to_update:
            from torchmetrics import MetricCollection
            temp_collection = MetricCollection(metrics_to_update)
            temp_collection(preds, targets)
            self.log_dict(temp_collection, **self.logging_params)
        
        # Update confusion matrix metrics manually (they accumulate during epoch)
        for metric_name, metric in self.train_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.update(preds, targets)

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        # Compute and log confusion matrix metrics manually
        # (they accumulate predictions during epoch, log in compute())
        for metric_name, metric in self.train_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.compute()

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/valid',
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f'{self.valid_metric.__class__.__name__}/valid',
            self.valid_metric,
            **self.logging_params,
        )

        # Update metrics, excluding ConfusionMatrixMetric (handled separately)
        metrics_to_update = {k: v for k, v in self.valid_add_metrics.items() 
                            if not isinstance(v, ConfusionMatrixMetric)}
        if metrics_to_update:
            from torchmetrics import MetricCollection
            temp_collection = MetricCollection(metrics_to_update)
            temp_collection(preds, targets)
            self.log_dict(temp_collection, **self.logging_params)
        
        # Update confusion matrix metrics manually (they accumulate during epoch)
        for metric_name, metric in self.valid_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.update(preds, targets)
        
        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        # In epoch_end hooks, on_step must be False
        logging_params = {**self.logging_params, 'on_step': False}
        self.log(
            f'{self.valid_metric.__class__.__name__}/valid_best',
            self.valid_metric_best.compute(),
            **logging_params,
        )
        
        # Compute and log confusion matrix metrics manually
        # (they accumulate predictions during epoch, log in compute())
        for metric_name, metric in self.valid_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.compute()

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/test', loss, **self.logging_params
        )

        self.test_metric(preds, targets)
        self.log(
            f'{self.test_metric.__class__.__name__}/test',
            self.test_metric,
            **self.logging_params,
        )

        # Update metrics, excluding ConfusionMatrixMetric (handled separately)
        metrics_to_update = {k: v for k, v in self.test_add_metrics.items() 
                            if not isinstance(v, ConfusionMatrixMetric)}
        if metrics_to_update:
            from torchmetrics import MetricCollection
            temp_collection = MetricCollection(metrics_to_update)
            temp_collection(preds, targets)
            self.log_dict(temp_collection, **self.logging_params)
        
        # Update confusion matrix metrics manually (they accumulate during epoch)
        for metric_name, metric in self.test_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.update(preds, targets)
        
        return {'loss': loss}

    def on_test_epoch_end(self) -> None:
        # Compute and log confusion matrix metrics manually
        # (they accumulate predictions during epoch, log in compute())
        for metric_name, metric in self.test_add_metrics.items():
            if isinstance(metric, ConfusionMatrixMetric):
                metric.compute()

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        logits = self.forward(batch['image'])
        preds = self.output_activation(logits)
        outputs = {'logits': logits, 'preds': preds}
        if 'label' in batch:
            outputs.update({'targets': batch['label']})
        if 'name' in batch:
            outputs.update({'names': batch['name']})
        return outputs


class MNISTLitModule(SingleLitModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        x, y = batch
        logits = self.forward(x['image'])
        loss = self.loss(logits, y)
        preds = self.output_activation(logits)
        return loss, preds, y

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        x, y = batch
        logits = self.forward(x['image'])
        preds = self.output_activation(logits)
        return {'logits': logits, 'preds': preds, 'targets': y}


class SingleVicRegLitModule(BaseLitModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        proj_hidden_dim: int,
        proj_output_dim: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders for
        Self-Supervised task using VicReg approach.

        :param network: Network config.
        :param optimizer: Optimizer config.
        :param scheduler: Scheduler config.
        :param logging: Logging config.
        :param proj_hidden_dim: Projector hidden dimensions.
        :param proj_output_dim: Projector output dimensions.
        :param args: Additional arguments for pytorch_lightning.LightningModule.
        :param kwargs: Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.model.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            # nn.Linear(proj_hidden_dim, proj_hidden_dim),
            # nn.BatchNorm1d(proj_hidden_dim),
            # nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        self.save_hyperparameters(logger=False)

    def forward(self, x: Any) -> Any:
        x = self.model.forward(x)
        return self.projector(x)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        z1 = self.forward(batch['z1'])
        z2 = self.forward(batch['z2'])
        loss = self.loss(z1, z2)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/train',
            loss,
            **self.logging_params,
        )
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/valid',
            loss,
            **self.logging_params,
        )
        return {'loss': loss}

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss = self.model_step(batch, batch_idx)
        self.log(
            f'{self.loss.__class__.__name__}/test', loss, **self.logging_params
        )
        return {'loss': loss}

    def on_test_epoch_end(self) -> None:
        pass


class SingleReIdLitModule(SingleLitModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        embeddings = self.forward(batch['image'])
        return embeddings, batch['label']

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        loss, logits = self.loss(embeddings, batch['label'])
        preds = self.output_activation(logits)
        self.log(
            f'{self.loss.__class__.__name__}/train',
            loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f'{self.train_metric.__class__.__name__}/train',
            self.train_metric,
            **self.logging_params,
        )
        return {'loss': loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        with torch.no_grad():
            loss, logits = self.loss(embeddings, batch['label'])
        preds = self.output_activation(logits)
        self.log(
            f'{self.loss.__class__.__name__}/valid',
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f'{self.valid_metric.__class__.__name__}/valid',
            self.valid_metric,
            **self.logging_params,
        )
        return {'loss': loss}

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        embeddings, targets = self.model_step(batch, batch_idx)
        with torch.no_grad():
            loss, logits = self.loss(embeddings, batch['label'])
        preds = self.output_activation(logits)
        self.log(
            f'{self.loss.__class__.__name__}/test', loss, **self.logging_params
        )

        self.test_metric(preds, targets)
        self.log(
            f'{self.test_metric.__class__.__name__}/test',
            self.test_metric,
            **self.logging_params,
        )
        return {'loss': loss}

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        outputs = {'embeddings': self.forward(batch['image'])}
        if 'name' in batch:
            outputs.update({'names': batch['name']})
        return outputs
