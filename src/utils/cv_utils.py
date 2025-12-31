"""Utilities for cross-validation support."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def create_cv_splitter(cv_cfg: DictConfig) -> Any:
    """Create sklearn cross-validation splitter from config.

    :param cv_cfg: Cross-validation configuration.
    :return: Sklearn CV splitter instance.
    """
    strategy = cv_cfg.get('strategy', 'KFold')
    n_splits = cv_cfg.get('n_splits', 5)
    shuffle = cv_cfg.get('shuffle', True)
    random_state = cv_cfg.get('random_state', 42)

    if strategy == 'KFold':
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    elif strategy == 'StratifiedKFold':
        return StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    elif strategy == 'GroupKFold':
        return GroupKFold(n_splits=n_splits)
    elif strategy == 'StratifiedGroupKFold':
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
    elif strategy == 'LeaveOneGroupOut':
        return LeaveOneGroupOut()
    else:
        raise ValueError(
            f'Unknown CV strategy: {strategy}. '
            'Supported: KFold, StratifiedKFold, GroupKFold, '
            'StratifiedGroupKFold, LeaveOneGroupOut'
        )


def load_combined_dataset(
    data_path: str, path_column: str, target_column: str, group_column: str | None = None
) -> pd.DataFrame:
    """Load a single combined dataset parquet file for CV splitting.

    :param data_path: Path to parquet file containing combined train+val data.
    :param path_column: Column name containing file paths.
    :param target_column: Column name containing target labels.
    :param group_column: Column name containing group IDs (e.g., subject_id).
    :return: DataFrame with all necessary columns for CV strategies.
    """
    annotation_path = Path(data_path)
    if not annotation_path.exists():
        raise RuntimeError(f"Dataset file '{annotation_path}' does not exist.")

    log.info(f'Loading combined dataset from: {annotation_path}')
    df = pd.read_parquet(annotation_path)

    # Verify required columns exist
    required_columns = [path_column, target_column]
    if group_column:
        required_columns.append(group_column)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f'Missing required columns in dataset: {missing_columns}. '
            f'Available columns: {list(df.columns)}'
        )

    log.info(f'Loaded {len(df)} samples from dataset')
    if group_column:
        n_groups = df[group_column].nunique()
        log.info(f'Number of unique groups ({group_column}): {n_groups}')

    return df


def aggregate_cv_metrics(
    all_fold_metrics: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Compute mean and standard deviation of metrics across folds.

    :param all_fold_metrics: List of metric dictionaries, one per fold.
    :return: Dictionary with aggregated metrics (mean and std).
    """
    if not all_fold_metrics:
        return {}

    # Collect all metric names
    all_metric_names = set()
    for fold_metrics in all_fold_metrics:
        all_metric_names.update(fold_metrics.keys())

    aggregated = {}
    for metric_name in all_metric_names:
        values = []
        for fold_metrics in all_fold_metrics:
            if metric_name in fold_metrics:
                # Convert tensor to float if needed
                value = fold_metrics[metric_name]
                if hasattr(value, 'item'):
                    values.append(float(value.item()))
                else:
                    values.append(float(value))

        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n_folds': len(values),
            }

    return aggregated


def aggregate_predictions(
    all_fold_predictions: list[list[Any]]
) -> list[Any]:
    """Aggregate predictions from all folds.

    :param all_fold_predictions: List of predictions per fold.
    :return: Aggregated predictions with fold identifiers.
    """
    aggregated = []
    for fold_idx, fold_predictions in enumerate(all_fold_predictions):
        if not fold_predictions:
            continue
        
        # Add fold identifier to each prediction batch
        for batch in fold_predictions:
            if isinstance(batch, dict):
                batch_with_fold = batch.copy()
                batch_with_fold['fold'] = fold_idx
                aggregated.append(batch_with_fold)
            else:
                # If batch is not a dict, wrap it
                aggregated.append({'fold': fold_idx, 'predictions': batch})
    
    return aggregated


def save_cv_summary(
    aggregated_metrics: dict[str, dict[str, float]],
    per_fold_metrics: list[dict[str, Any]],
    output_dir: str,
    fold_checkpoint_paths: list[str] | None = None,
    fold_partitions: list[dict[str, Any]] | None = None,
    all_fold_predictions: list[list[Any]] | None = None,
) -> None:
    """Save cross-validation summary to JSON and CSV files.

    :param aggregated_metrics: Aggregated metrics (mean ± std).
    :param per_fold_metrics: List of metrics per fold.
    :param output_dir: Output directory for saving summary.
    :param fold_checkpoint_paths: Optional list of checkpoint paths per fold.
    :param fold_partitions: Optional list of partition info per fold.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper function to convert numpy arrays and pandas types to JSON-serializable formats
    def convert_for_json(obj):
        """Recursively convert numpy arrays and types to JSON-serializable formats."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        return obj

    # Save JSON summary
    summary = {
        'aggregated_metrics': aggregated_metrics,
        'per_fold_metrics': [
            {
                k: float(v.item()) if hasattr(v, 'item') else float(v)
                for k, v in fold_metrics.items()
            }
            for fold_metrics in per_fold_metrics
        ],
    }

    if fold_checkpoint_paths:
        summary['fold_checkpoint_paths'] = fold_checkpoint_paths
    
    if fold_partitions:
        # Convert for JSON serialization (should already be converted, but ensure it)
        summary['fold_partitions'] = convert_for_json(fold_partitions)
    
    # Add predictions info (paths, not the actual predictions to keep JSON small)
    if all_fold_predictions:
        summary['predictions_info'] = {
            'n_folds_with_predictions': len([p for p in all_fold_predictions if p]),
            'predictions_dir': str(output_path / 'predictions'),
            'per_fold_predictions': [
                f'fold_{i}/predictions' if preds else None
                for i, preds in enumerate(all_fold_predictions)
            ],
        }

    json_path = output_path / 'cv_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f'Saved CV summary to: {json_path}')

    # Save CSV summary (aggregated metrics)
    if aggregated_metrics:
        csv_path = output_path / 'cv_summary.csv'
        rows = []
        for metric_name, stats in aggregated_metrics.items():
            rows.append(
                {
                    'metric': metric_name,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'n_folds': stats['n_folds'],
                }
            )
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv(csv_path, index=False)
        log.info(f'Saved CV summary CSV to: {csv_path}')

    # Save partition info if available
    if fold_partitions:
        log.info('Saving partition information...')
        
        # Convert for JSON serialization (already converted in get_fold_partition_info, but ensure it)
        partitions_json = convert_for_json(fold_partitions)
        
        # Save as JSON
        partitions_json_path = output_path / 'cv_partitions.json'
        with open(partitions_json_path, 'w') as f:
            json.dump(partitions_json, f, indent=2, ensure_ascii=False)
        log.info(f'Saved partition info JSON to: {partitions_json_path}')
        
        # Save as Parquet (more efficient for large datasets)
        all_partitions = []
        for partition in fold_partitions:
            # Combine train and val metadata
            train_df = pd.DataFrame(partition['train_metadata'])
            val_df = pd.DataFrame(partition['val_metadata'])
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            all_partitions.append(combined_df)
        
        if all_partitions:
            partitions_df = pd.concat(all_partitions, ignore_index=True)
            partitions_parquet_path = output_path / 'cv_partitions.parquet'
            partitions_df.to_parquet(partitions_parquet_path, index=False)
            log.info(f'Saved partition info Parquet to: {partitions_parquet_path}')

