import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pytorch_lightning import LightningModule, Trainer

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


def process_state_dict(
    state_dict: OrderedDict | dict,
    symbols: int = 0,
    exceptions: str | list[str] | None = None,
) -> OrderedDict:
    """Filter and map model state dict keys.

    :param state_dict: State dict.
    :param symbols: Determines how many symbols should be cut in the
        beginning of state dict keys. Default to 0.
    :param exceptions: Determines exceptions,
        i.e. substrings, which keys should not contain.
    :return: Filtered state dict.
    """

    new_state_dict = OrderedDict()
    if exceptions:
        if isinstance(exceptions, str):
            exceptions = [exceptions]
    for key, value in state_dict.items():
        is_exception = False
        if exceptions:
            for exception in exceptions:
                if key.startswith(exception):
                    is_exception = True
        if not is_exception:
            new_state_dict[key[symbols:]] = value

    return new_state_dict


def save_state_dicts(
    trainer: Trainer,
    model: LightningModule,
    dirname: str,
    symbols: int = 6,
    exceptions: str | list[str] | None = None,
) -> None:
    """Save model state dicts for last and best checkpoints.

    :param trainer: Lightning trainer.
    :param model: Lightning model.
    :param dirname: Saving directory.
    :param symbols: Determines how many symbols should be cut in the
        beginning of state dict keys. Default to 6 for cutting
        Lightning name prefix.
    :param exceptions: Determines exceptions,
        i.e. substrings, which keys should not contain.  Default to [loss].
    """

    # save state dict for last checkpoint
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f'{dirname}/last_ckpt.pth'
    torch.save(mapped_state_dict, path)
    log.info(f'Last ckpt state dict saved to: {path}')

    # save state dict for best checkpoint
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path == '':
        log.warning('Best ckpt not found! Skipping...')
        return

    best_ckpt_score = trainer.checkpoint_callback.best_model_score
    if best_ckpt_score is not None:
        prefix = str(best_ckpt_score.detach().cpu().item())
        prefix = prefix.replace('.', '_')
    else:
        log.warning('Best ckpt score not found! Use prefix <unknown>!')
        prefix = 'unknown'
    model = model.__class__.load_from_checkpoint(best_ckpt_path)
    mapped_state_dict = process_state_dict(
        model.state_dict(), symbols=symbols, exceptions=exceptions
    )
    path = f'{dirname}/best_ckpt_{prefix}.pth'
    torch.save(mapped_state_dict, path)
    log.info(f'Best ckpt state dict saved to: {path}')


def save_predictions_from_dataloader(
    predictions: list[Any], path: Path
) -> None:
    """Save predictions returned by `Trainer.predict` method for single
    dataloader.

    :param predictions: Predictions returned by `Trainer.predict` method.
    :param path: Path to predictions.
    """

    if path.suffix == '.csv':
        # Collect all rows first to determine fieldnames
        all_rows = []
        for batch in predictions:
            # Separate scalar keys (like 'fold') from array keys
            scalar_keys = []
            array_keys = []
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, (int, float, str, bool)):
                    scalar_keys.append(key)
                else:
                    array_keys.append(key)
            
            if not array_keys:
                # If no array keys, skip this batch
                continue
                
            batch_size = len(batch[array_keys[0]])
            for i in range(batch_size):
                # Process array keys (convert to list)
                row = {}
                for key in array_keys:
                    value = batch[key][i].tolist() if hasattr(batch[key][i], 'tolist') else batch[key][i]
                    # Rename 'names' to 'path'
                    row_key = 'path' if key == 'names' else key
                    row[row_key] = value
                # Add scalar keys as-is (they're the same for all items in batch)
                for key in scalar_keys:
                    row[key] = batch[key]
                all_rows.append(row)
        
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            with open(path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)

    elif path.suffix == '.parquet':
        # Collect all rows first
        all_rows = []
        for batch in predictions:
            # Separate scalar keys (like 'fold') from array keys
            scalar_keys = []
            array_keys = []
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, (int, float, str, bool)):
                    scalar_keys.append(key)
                else:
                    array_keys.append(key)
            
            if not array_keys:
                # If no array keys, skip this batch
                continue
                
            batch_size = len(batch[array_keys[0]])
            for i in range(batch_size):
                # Process array keys (convert to list)
                row = {}
                for key in array_keys:
                    value = batch[key][i].tolist() if hasattr(batch[key][i], 'tolist') else batch[key][i]
                    # Rename 'names' to 'path'
                    row_key = 'path' if key == 'names' else key
                    row[row_key] = value
                # Add scalar keys as-is (they're the same for all items in batch)
                for key in scalar_keys:
                    row[key] = batch[key]
                all_rows.append(row)
        
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_parquet(path, index=False)

    elif path.suffix == '.json':
        processed_predictions = {}
        for batch in predictions:
            # Separate scalar keys (like 'fold') from array keys
            scalar_keys = []
            array_keys = []
            for key in batch.keys():
                # Check if it's a scalar (int, float, str) or array-like
                value = batch[key]
                if isinstance(value, (int, float, str, bool)):
                    scalar_keys.append(key)
                else:
                    array_keys.append(key)
            
            if not array_keys:
                # If no array keys, skip this batch or handle differently
                continue
                
            batch_size = len(batch[array_keys[0]])
            for i in range(batch_size):
                # Process array keys (convert to list)
                item = {}
                for key in array_keys:
                    value = batch[key][i].tolist() if hasattr(batch[key][i], 'tolist') else batch[key][i]
                    # Rename 'names' to 'path'
                    item_key = 'path' if key == 'names' else key
                    item[item_key] = value
                # Add scalar keys as-is (they're the same for all items in batch)
                for key in scalar_keys:
                    item[key] = batch[key]
                
                # Use path as key if available, otherwise use index
                if 'names' in batch.keys():
                    processed_predictions[batch['names'][i]] = item
                else:
                    processed_predictions[len(processed_predictions)] = item
        with open(path, 'w') as json_file:
            json.dump(processed_predictions, json_file, ensure_ascii=False)

    else:
        raise NotImplementedError(f'{path.suffix} is not implemented!')


def save_predictions(
    predictions: list[Any], dirname: str, output_format: str = 'json'
) -> None:
    """Save predictions returned by `Trainer.predict` method.

    Due to `LightningDataModule.predict_dataloader` return type is
    Union[DataLoader, List[DataLoader]], so `Trainer.predict` method can return
    a list of dictionaries, one for each provided batch containing their
    respective predictions, or a list of lists, one for each provided dataloader
    containing their respective predictions, where each list contains dictionaries.

    :param predictions: Predictions returned by `Trainer.predict` method.
    :param dirname: Dirname for predictions.
    :param output_format: Output file format. It could be `json`, `csv`, or `parquet`.
        Default to `json`.
    """

    if not predictions:
        log.warning('Predictions is empty! Saving was cancelled ...')
        return

    if output_format not in ('json', 'csv', 'parquet'):
        raise NotImplementedError(
            f'{output_format} is not implemented! Use `json`, `csv`, or `parquet`.'
            'Or change `src.utils.saving.save_predictions` func logic.'
        )

    path = Path(dirname) / 'predictions'
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions[0], dict):
        target_path = path / f'predictions.{output_format}'
        save_predictions_from_dataloader(predictions, target_path)
        log.info(f'Saved predictions to: {str(target_path)}')
        return

    elif isinstance(predictions[0], list):
        for idx, predictions_idx in enumerate(predictions):
            if not predictions_idx:
                log.warning(
                    f'Predictions for DataLoader #{idx} is empty! Skipping...'
                )
                continue
            target_path = path / f'predictions_{idx}.{output_format}'
            save_predictions_from_dataloader(predictions_idx, target_path)
            log.info(
                f'Saved predictions for DataLoader #{idx} to: '
                f'{str(target_path)}'
            )
        return

    raise Exception(
        'Passed predictions format is not supported by default!\n'
        'Make sure that it is formed correctly! It requires as List[Dict[str, Any]] type'
        'in case of predict_dataloader returns DataLoader or List[List[Dict[str, Any]]]'
        'type in case of predict_dataloader returns List[DataLoader]!\n'
        'Or change `src.utils.saving.save_predictions` function logic.'
    )
