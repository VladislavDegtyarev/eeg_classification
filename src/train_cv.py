"""Training script with cross-validation support."""

import hashlib
from pathlib import Path
from typing import Any

import hydra
import pyrootutils
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import CSVLogger, Logger, WandbLogger

from src import utils

# --------------------------------------------------------------------------- #
# Setup root directory
# --------------------------------------------------------------------------- #

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    'version_base': '1.3',
    'config_path': str(root / 'configs'),
    'config_name': 'train.yaml',
}
log = utils.get_pylogger(__name__)


class UpdateWandbNameCallback(Callback):
    """Callback to update wandb run name at the start of training."""
    
    def __init__(self, fold_name: str):
        super().__init__()
        self.fold_name = fold_name
    
    def on_train_start(self, trainer, pl_module):
        """Update wandb run name when training starts."""
        if wandb.run is not None:
            wandb.run.name = self.fold_name
            log.info(f'Updated wandb run name to: {self.fold_name}')


def update_logger_for_fold(
    logger: Logger, fold_idx: int, base_output_dir: str, base_name: str | None = None
) -> Logger:
    """Update logger configuration for a specific fold.

    :param logger: Logger instance.
    :param fold_idx: Current fold index.
    :param base_output_dir: Base output directory.
    :param base_name: Base name from config (optional, will try to get from logger if not provided).
    :return: Updated logger instance.
    """
    fold_output_dir = Path(base_output_dir) / f'fold_{fold_idx}'

    if isinstance(logger, WandbLogger):
        # Finish any existing wandb run before creating a new one
        if wandb.run is not None:
            wandb.finish()
        
        # Get original name - prioritize base_name from config (has resolved date), then logger attributes
        original_name = None
        
        # First try base_name from config (should have resolved date from Hydra)
        if base_name:
            original_name = base_name
            log.debug(f'Using base_name from config: {original_name}')
        
        # If not available, try to get from logger's internal attributes
        if not original_name:
            # Try _name first (internal attribute that might have resolved value)
            if hasattr(logger, '_name') and logger._name:
                original_name = logger._name
                log.debug(f'Using logger._name: {original_name}')
            # Try name property
            elif hasattr(logger, 'name') and logger.name:
                original_name = logger.name
                log.debug(f'Using logger.name: {original_name}')
            # Try _wandb_init dict if logger was initialized
            elif hasattr(logger, '_wandb_init') and isinstance(logger._wandb_init, dict):
                original_name = logger._wandb_init.get('name')
                if original_name:
                    log.debug(f'Using logger._wandb_init name: {original_name}')
        
        # Fallback to default
        if not original_name:
            original_name = 'run'
            log.warning(f'Could not determine original name for fold {fold_idx}, using default "run"')
        
        fold_name = f'{original_name}_fold_{fold_idx}'
        unique_id = hashlib.md5(f'{original_name}_fold_{fold_idx}'.encode()).hexdigest()[:8]
        
        log.info(f'Creating WandbLogger for fold {fold_idx}: name="{fold_name}", id="fold_{fold_idx}_{unique_id}"')
        
        logger = WandbLogger(
            name=fold_name,
            save_dir=str(fold_output_dir),
            project=logger.project if hasattr(logger, 'project') else 'eeg-classification',
            offline=logger.offline if hasattr(logger, 'offline') else False,
            id=f'fold_{fold_idx}_{unique_id}',
            tags=logger.tags if hasattr(logger, 'tags') else [],
            group=logger.group if hasattr(logger, 'group') else '',
            job_type=logger.job_type if hasattr(logger, 'job_type') else '',
            log_model=logger.log_model if hasattr(logger, 'log_model') else False,
            prefix=logger.prefix if hasattr(logger, 'prefix') else ''
        )
    elif isinstance(logger, CSVLogger):
        logger = CSVLogger(
            save_dir=str(fold_output_dir),
            name=logger.name if hasattr(logger, 'name') else 'csv',
            prefix=logger.prefix if hasattr(logger, 'prefix') else ''
        )

    return logger


def train_fold(
    cfg: DictConfig, fold_idx: int, n_folds: int, base_output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Any] | None, dict[str, Any] | None]:
    """Train a single fold.

    :param cfg: Configuration composed by Hydra.
    :param fold_idx: Current fold index.
    :param n_folds: Total number of folds.
    :param base_output_dir: Base output directory (not modified during folds).
    :return: Tuple of (metrics_dict, object_dict, predictions, partition_info).
    """
    log.info(f'=' * 80)
    log.info(f'Starting fold {fold_idx + 1}/{n_folds}')
    log.info(f'=' * 80)

    # Set seed for this fold
    fold_seed = cfg.get('seed', 42)
    if cfg.get('cv', {}).get('random_state'):
        fold_seed = cfg.cv.random_state + fold_idx
    log.info(f'Seed for fold {fold_idx}: {fold_seed}')
    seed_everything(fold_seed, workers=True)

    # Create fold-specific output directory (flat structure)
    fold_output_dir = base_output_dir / f'fold_{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    # Init lightning datamodule
    log.info(f'Instantiating datamodule <{cfg.datamodule._target_}>')
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    # Setup fold-specific datasets
    if not hasattr(datamodule, 'setup_fold'):
        raise ValueError('Datamodule must support setup_fold() method for cross-validation')
    
    log.info(f'Setting up fold {fold_idx}...')
    datamodule.setup_fold(fold_idx)
    
    if not datamodule.is_fold_setup:
        raise RuntimeError(
            f'Fold {fold_idx} setup failed. '
            f'train_set={datamodule.train_set is not None}, '
            f'valid_set={datamodule.valid_set is not None}'
        )
    
    datamodule.setup(stage='fit')
    
    # Get partition info for this fold
    partition_info = None
    if hasattr(datamodule, 'get_fold_partition_info'):
        partition_info = datamodule.get_fold_partition_info(fold_idx)

    # Init lightning model
    log.info(f'Instantiating lightning model <{cfg.module._target_}>')
    model: LightningModule = hydra.utils.instantiate(cfg.module, _recursive_=False)

    # Init callbacks
    log.info('Instantiating callbacks...')
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get('callbacks'))

    # Update callback paths for fold
    for callback in callbacks:
        if hasattr(callback, 'dirpath') and callback.dirpath:
            callback.dirpath = str(fold_output_dir / 'checkpoints')

    # Init loggers
    log.info('Instantiating loggers...')
    logger_list: list[Logger] = utils.instantiate_loggers(cfg.get('logger'))

    # Update loggers for fold
    base_name = cfg.get('name', None)
    for logger in logger_list:
        update_logger_for_fold(logger, fold_idx, str(base_output_dir), base_name=base_name)

    # Init lightning ddp plugins
    log.info('Instantiating plugins...')
    plugins: list[Any] | None = utils.instantiate_plugins(cfg)

    # Init lightning trainer
    log.info(f'Instantiating trainer <{cfg.trainer._target_}>')
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger_list, plugins=plugins
    )

    # Send parameters from cfg to all lightning loggers
    object_dict = {
        'cfg': cfg,
        'datamodule': datamodule,
        'model': model,
        'callbacks': callbacks,
        'logger': logger_list,
        'trainer': trainer,
    }

    if logger_list:
        log.info('Logging hyperparameters!')
        utils.log_hyperparameters(object_dict)

    # Log metadata
    log.info('Logging metadata!')
    utils.log_metadata(cfg)

    # Train the model
    if cfg.get('train'):
        log.info('Starting training!')
        # Add callback to update wandb name when training starts
        base_name = cfg.get('name', None) or 'run'
        fold_name = f'{base_name}_fold_{fold_idx}'
        name_callback = UpdateWandbNameCallback(fold_name=fold_name)
        callbacks.append(name_callback)
        # Update trainer callbacks
        trainer.callbacks = callbacks
        
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get('ckpt_path'),
        )

    train_metrics = trainer.callback_metrics.copy()

    # Test the model
    if cfg.get('test'):
        log.info('Starting testing!')
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == '':
            log.warning(
                'Best ckpt not found! Using current weights for testing...'
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f'Best ckpt path: {ckpt_path}')

    test_metrics = trainer.callback_metrics.copy()

    # Generate predictions for validation set
    fold_predictions = None
    log.info('Generating predictions on validation set...')
    try:
        val_dataloader = datamodule.val_dataloader()
        fold_predictions = trainer.predict(
            model=model,
            dataloaders=val_dataloader,
            ckpt_path=ckpt_path if ckpt_path else None,
        )
        if fold_predictions:
            predictions_dir = fold_output_dir / 'predictions'
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions_params = cfg.extras.get('predictions_saving_params', {})
            utils.save_predictions(
                predictions=fold_predictions,
                dirname=str(predictions_dir),
                **predictions_params,
            )
            log.info(f'Saved predictions for fold {fold_idx} to: {predictions_dir}')
    except Exception as e:
        log.warning(f'Failed to generate predictions for fold {fold_idx}: {e}')
        fold_predictions = None

    # Save state dicts for best and last checkpoints
    if cfg.get('save_state_dict'):
        log.info('Starting saving state dicts!')
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=str(fold_output_dir),
            **cfg.extras.state_dict_saving_params,
        )

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # Get checkpoint path for summary
    checkpoint_path = None
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        checkpoint_path = trainer.checkpoint_callback.best_model_path

    # Close wandb run for this fold to ensure separate runs in wandb
    for logger in logger_list:
        if isinstance(logger, WandbLogger):
            if wandb.run is not None:
                log.info(f'Finishing wandb run for fold {fold_idx}')
                wandb.finish()

    return (
        metric_dict,
        {**object_dict, 'checkpoint_path': checkpoint_path},
        fold_predictions,
        partition_info,
    )


@utils.task_wrapper
def train_cv(cfg: DictConfig) -> tuple[dict, dict]:
    """Train model with cross-validation across multiple folds.

    :param cfg: Configuration composed by Hydra.
    :return: Dict with aggregated metrics and dict with all objects.
    """
    utils.log_gpu_memory_metadata()

    # Verify CV is enabled
    cv_cfg = cfg.get('cv')
    if not cv_cfg or not cv_cfg.get('enabled', False):
        raise ValueError(
            'Cross-validation is not enabled. Set cv.enabled=true in config.'
        )

    # Init datamodule to get number of folds
    log.info('Initializing datamodule to determine number of folds...')
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    if hasattr(datamodule, 'get_n_folds'):
        n_folds = datamodule.get_n_folds()
    else:
        raise ValueError('Datamodule must support get_n_folds() method')

    log.info(f'Starting cross-validation with {n_folds} folds')

    # Store original output directory
    original_output_dir = cfg.paths.output_dir
    base_output_dir = Path(original_output_dir)

    # Train each fold
    all_fold_metrics = []
    all_fold_checkpoints = []
    all_fold_objects = []
    all_fold_predictions = []
    all_fold_partitions = []

    for fold_idx in range(n_folds):
        try:
            fold_metrics, fold_objects, fold_predictions, fold_partition = train_fold(
                cfg, fold_idx, n_folds, base_output_dir
            )
            all_fold_metrics.append(fold_metrics)
            all_fold_checkpoints.append(
                fold_objects.get('checkpoint_path', '')
            )
            all_fold_objects.append(fold_objects)
            if fold_predictions:
                all_fold_predictions.append(fold_predictions)
            if fold_partition:
                all_fold_partitions.append(fold_partition)
            log.info(f'Fold {fold_idx + 1}/{n_folds} completed successfully')
        except Exception as e:
            log.error(f'Fold {fold_idx + 1}/{n_folds} failed with error: {e}')
            raise

    # Aggregate metrics across folds
    log.info('Aggregating metrics across folds...')
    from src.utils.cv_utils import (
        aggregate_cv_metrics,
        aggregate_predictions,
        save_cv_summary,
    )

    aggregated_metrics = aggregate_cv_metrics(all_fold_metrics)

    # Aggregate predictions if available
    if all_fold_predictions:
        log.info('Aggregating predictions across folds...')
        aggregated_predictions = aggregate_predictions(all_fold_predictions)
        predictions_dir = base_output_dir / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_params = cfg.extras.get('predictions_saving_params', {})
        utils.save_predictions(
            predictions=aggregated_predictions,
            dirname=str(predictions_dir),
            **predictions_params,
        )
        log.info(f'Saved aggregated predictions to: {predictions_dir}')

    # Save CV summary with partition info and predictions
    save_cv_summary(
        aggregated_metrics=aggregated_metrics,
        per_fold_metrics=all_fold_metrics,
        output_dir=str(base_output_dir),
        fold_checkpoint_paths=all_fold_checkpoints,
        fold_partitions=all_fold_partitions if all_fold_partitions else None,
        all_fold_predictions=all_fold_predictions if all_fold_predictions else None,
    )

    log.info('=' * 80)
    log.info('Cross-validation completed!')
    log.info('=' * 80)
    log.info('Aggregated metrics (mean ± std):')
    for metric_name, stats in aggregated_metrics.items():
        log.info(
            f'  {metric_name}: {stats["mean"]:.6f} ± {stats["std"]:.6f} '
            f'(min={stats["min"]:.6f}, max={stats["max"]:.6f}, n_folds={stats["n_folds"]})'
        )

    # Return aggregated metrics and summary
    return aggregated_metrics, {
        'n_folds': n_folds,
        'aggregated_metrics': aggregated_metrics,
        'per_fold_metrics': all_fold_metrics,
        'fold_objects': all_fold_objects,
    }


@utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for CV training.

    :param cfg: Configuration composed by Hydra.
    :return: Optimized metric value if specified, None otherwise.
    """
    # Train with cross-validation
    metric_dict, _ = train_cv(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get('optimized_metric')
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()

