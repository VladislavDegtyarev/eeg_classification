from typing import Any, List

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics


class CIFAR10LitModule(BaseLitModule):
    """Example of LightningModule for CIFAR10 classification.

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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)
        self.output_activation = hydra.utils.instantiate(
            network.output_activation, _partial_=True
        )

        main_metric, valid_metric_best, add_metrics = load_metrics(
            network.metrics
        )
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(postfix="/train")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(postfix="/valid")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix="/test")

        self.save_hyperparameters(logger=False)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        logits = self.forward(batch["image"])
        loss = self.loss(logits, batch["label"])
        preds = self.output_activation(logits)
        return loss, preds, batch["label"]

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            self.train_metric,
            **self.logging_params,
        )

        self.train_add_metrics(preds, targets)
        self.log_dict(self.train_add_metrics, **self.logging_params)

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        # This method is called at the end of each training epoch
        # Note: outputs are no longer passed as parameter in v2.0+
        # If you need outputs, save them as instance attributes in training_step
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,
        )

        self.valid_add_metrics(preds, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_params)
        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/test", loss, **self.logging_params
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        return {"loss": loss}

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        logits = self.forward(batch["image"])
        preds = self.output_activation(logits)
        outputs = {"logits": logits, "preds": preds}
        if "label" in batch:
            outputs["targets"] = batch["label"]
        return outputs
