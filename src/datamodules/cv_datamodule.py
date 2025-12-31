"""Cross-validation DataModule for fold-based data splitting."""

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.indexed_dataset import IndexedDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule
from src.utils.cv_utils import create_cv_splitter, load_combined_dataset
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class CrossValidationDataModule(SingleDataModule):
    """DataModule for cross-validation with fold-based splitting.

    Extends SingleDataModule to support cross-validation by:
    - Accepting a single combined dataset (train+val)
    - Splitting it into folds using sklearn CV splitters
    - Providing fold-specific train/validation datasets
    """

    def __init__(
        self,
        datasets: DictConfig,
        loaders: DictConfig,
        transforms: DictConfig,
        cv: DictConfig,
    ) -> None:
        """Initialize CV DataModule.

        :param datasets: Datasets config. Should have 'train' pointing to combined dataset.
        :param loaders: Loaders config.
        :param transforms: Transforms config.
        :param cv: Cross-validation configuration.
        """
        super().__init__(datasets=datasets, loaders=loaders, transforms=transforms)
        self.cv_cfg = cv

        # CV-specific attributes
        self.cv_splitter = None
        self.cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
        self.current_fold_idx: int | None = None
        self._fold_setup_complete: bool = False  # Persistent flag that can't be reset
        self._setup_lock: bool = False  # Lock to prevent Lightning from resetting state
        self.full_dataset: Dataset | None = None
        self.df_full: pd.DataFrame | None = None

        # Extract CV parameters
        self.group_column = cv.get('group_column')
        self.path_column = datasets.train.get('path_column')
        self.target_column = datasets.train.get('target_column')

        if not self.path_column or not self.target_column:
            raise ValueError(
                'path_column and target_column must be provided in datasets.train config'
            )
    
    @property
    def is_fold_setup(self) -> bool:
        """Check if fold is properly set up.
        
        :return: True if fold is set up, False otherwise.
        """
        return (
            self._fold_setup_complete 
            and self.current_fold_idx is not None
            and self.train_set is not None
            and self.valid_set is not None
        )

    def _load_full_dataset(self) -> tuple[Dataset, pd.DataFrame]:
        """Load the full combined dataset and its metadata.

        :return: Tuple of (full_dataset, dataframe_with_metadata).
        """
        if self.full_dataset is not None and self.df_full is not None:
            return self.full_dataset, self.df_full

        log.info('Loading full combined dataset for CV...')

        # Load dataset metadata from parquet
        train_cfg = self.cfg_datasets.train
        data_path = train_cfg.get('data_path')

        if not data_path:
            raise ValueError('data_path must be provided in datasets.train config')

        # Load parquet file to get metadata
        self.df_full = load_combined_dataset(
            data_path=data_path,
            path_column=self.path_column,
            target_column=self.target_column,
            group_column=self.group_column,
        )

        # Create full dataset with all transforms (will be subset later)
        transforms = TransformsWrapper(self.transforms.get('train'))
        self.full_dataset = hydra.utils.instantiate(
            train_cfg, transforms=transforms, _recursive_=False
        )

        log.info(f'Loaded full dataset with {len(self.full_dataset)} samples')

        return self.full_dataset, self.df_full

    def _generate_cv_splits(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate CV splits from the full dataset.

        :return: List of (train_indices, val_indices) tuples.
        """
        if self.cv_splits:
            return self.cv_splits

        log.info('Generating CV splits...')

        # Load full dataset and metadata
        _, df = self._load_full_dataset()

        # Create CV splitter
        self.cv_splitter = create_cv_splitter(self.cv_cfg)
        strategy = self.cv_cfg.get('strategy', 'KFold')

        # Get features and targets for splitting
        X = np.arange(len(df))  # Indices as features
        y = df[self.target_column].values  # Targets for stratification

        # Generate splits based on strategy
        if strategy in ('GroupKFold', 'StratifiedGroupKFold', 'LeaveOneGroupOut'):
            if not self.group_column:
                raise ValueError(
                    f'{strategy} requires group_column to be specified in CV config'
                )
            groups = df[self.group_column].values
            splits = list(self.cv_splitter.split(X, y, groups=groups))
        elif strategy in ('StratifiedKFold',):
            splits = list(self.cv_splitter.split(X, y))
        else:  # KFold
            splits = list(self.cv_splitter.split(X))

        self.cv_splits = splits
        n_folds = len(splits)
        log.info(f'Generated {n_folds} CV folds using {strategy}')

        # Log fold sizes
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            log.info(
                f'Fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}'
            )

        return self.cv_splits

    def setup_fold(self, fold_idx: int) -> None:
        """Set up datasets for a specific fold.

        :param fold_idx: Index of the fold to set up.
        """
        log.info(f'Setting up fold {fold_idx}...')

        # First, ensure global statistics are computed (needed for transforms)
        if self.global_channel_means is None or self.global_channel_stds is None:
            log.info('Computing global statistics before setting up fold...')
            train_cfg = self.cfg_datasets.get('train')
            train_transforms_cfg = self.transforms.get('train')

            if 'pad_or_crop' in train_transforms_cfg:
                pad_or_crop_cfg = train_transforms_cfg.get('pad_or_crop')
                self.global_channel_means, self.global_channel_stds = (
                    self._compute_global_statistics(train_cfg, pad_or_crop_cfg)
                )
                self._update_transforms_with_statistics()
            else:
                log.warning('pad_or_crop transform not found - statistics may not be computed')

        # Generate splits if not already done
        splits = self._generate_cv_splits()

        if fold_idx >= len(splits):
            raise ValueError(
                f'Fold index {fold_idx} out of range. '
                f'Total number of folds: {len(splits)}'
            )

        # Set current fold index and flag FIRST before creating datasets
        # This marks that setup_fold() has been called and prevents reset
        self.current_fold_idx = fold_idx
        self._fold_setup_complete = True  # Persistent flag
        self._setup_lock = True  # Lock to prevent Lightning from resetting state
        train_indices, val_indices = splits[fold_idx]

        # Create fold-specific datasets using IndexedDataset
        train_transforms = TransformsWrapper(self.transforms.get('train'))
        val_transforms = TransformsWrapper(self.transforms.get('valid'))

        # Create base datasets with appropriate transforms
        # Both use the same data source (combined dataset) but different transforms
        train_cfg = self.cfg_datasets.train.copy()
        base_train_dataset = hydra.utils.instantiate(
            train_cfg, transforms=train_transforms, _recursive_=False
        )

        val_cfg = self.cfg_datasets.train.copy()  # Use same data source
        base_val_dataset = hydra.utils.instantiate(
            val_cfg, transforms=val_transforms, _recursive_=False
        )

        # Create indexed subsets
        # Note: train_indices and val_indices are numpy arrays from CV splitter
        # They correspond to row indices in the parquet file, which map to
        # dataset indices since ClassificationDataset loads keys in order
        log.info(
            f'Creating IndexedDataset: base_train len={len(base_train_dataset)}, '
            f'train_indices len={len(train_indices)}, '
            f'base_val len={len(base_val_dataset)}, val_indices len={len(val_indices)}'
        )
        
        # Create indexed subsets - this MUST succeed
        try:
            self.train_set = IndexedDataset(base_train_dataset, train_indices.tolist())
            self.valid_set = IndexedDataset(base_val_dataset, val_indices.tolist())
        except Exception as e:
            raise RuntimeError(
                f'Failed to create IndexedDataset for fold {fold_idx}: {e}'
            ) from e

        # CRITICAL: Verify datasets were created successfully
        if self.train_set is None:
            raise RuntimeError(
                f'CRITICAL: train_set is None after IndexedDataset creation for fold {fold_idx}'
            )
        if self.valid_set is None:
            raise RuntimeError(
                f'CRITICAL: valid_set is None after IndexedDataset creation for fold {fold_idx}'
            )
        
        # Verify datasets have length
        if len(self.train_set) == 0:
            raise RuntimeError(f'train_set is empty for fold {fold_idx}')
        if len(self.valid_set) == 0:
            raise RuntimeError(f'valid_set is empty for fold {fold_idx}')

        # Test set remains separate if provided
        if self.cfg_datasets.get('test'):
            self.test_set = self._get_dataset_('test')

        log.info(
            f'Fold {fold_idx} setup complete: '
            f'train={len(self.train_set)}, val={len(self.valid_set)}, '
            f'_fold_setup_complete={self._fold_setup_complete}, '
            f'current_fold_idx={self.current_fold_idx}, '
            f'_setup_lock={self._setup_lock}'
        )
        
        # Final verification that everything is set correctly
        if not self.is_fold_setup:
            raise RuntimeError(
                f'CRITICAL: Fold {fold_idx} setup completed but is_fold_setup property returns False! '
                f'This indicates a bug in setup_fold() or is_fold_setup property.'
            )

    def get_fold_partition_info(self, fold_idx: int) -> dict[str, Any]:
        """Get partition information for a specific fold.

        :param fold_idx: Index of the fold.
        :return: Dictionary with partition info including indices and metadata.
        """
        if not self.cv_splits:
            self._generate_cv_splits()

        if fold_idx >= len(self.cv_splits):
            raise ValueError(
                f'Fold index {fold_idx} out of range. '
                f'Total number of folds: {len(self.cv_splits)}'
            )

        train_indices, val_indices = self.cv_splits[fold_idx]
        
        # Load full dataframe if not already loaded
        if self.df_full is None:
            self._load_full_dataset()

        # Create metadata DataFrames for train and validation sets
        train_metadata = self.df_full.iloc[train_indices].copy()
        train_metadata['fold'] = fold_idx
        train_metadata['split'] = 'train'
        train_metadata['original_index'] = train_indices

        val_metadata = self.df_full.iloc[val_indices].copy()
        val_metadata['fold'] = fold_idx
        val_metadata['split'] = 'val'
        val_metadata['original_index'] = val_indices

        # Convert DataFrames to dicts, ensuring numpy arrays are converted to lists
        train_metadata_dict = train_metadata.to_dict('records')
        val_metadata_dict = val_metadata.to_dict('records')
        
        # Convert any numpy arrays and pandas types to JSON-serializable formats
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            elif isinstance(obj, pd.Timedelta):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            return obj

        train_metadata_dict = convert_numpy_to_list(train_metadata_dict)
        val_metadata_dict = convert_numpy_to_list(val_metadata_dict)

        return {
            'fold_idx': fold_idx,
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'train_metadata': train_metadata_dict,
            'val_metadata': val_metadata_dict,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
        }

    def setup(self, stage: str | None = None) -> None:
        """Setup method called by Lightning.

        For CV mode, this method should NOT set up datasets - that's done by setup_fold().
        Lightning calls setup() automatically during fit(), so we preserve any datasets
        that were set up by setup_fold().

        :param stage: Lightning stage (not used in CV mode).
        """
        # CRITICAL CHECK #1: If setup lock is set, NEVER touch datasets
        # This prevents Lightning from resetting state after setup_fold() completes
        if self._setup_lock:
            log.debug(
                f'CV setup() called by Lightning with stage={stage}. '
                f'Setup lock is active - preserving fold state. '
                f'Fold setup: _fold_setup_complete={self._fold_setup_complete}, '
                f'current_fold_idx={self.current_fold_idx}, '
                f'train_set={self.train_set is not None}, '
                f'valid_set={self.valid_set is not None}.'
            )
            # Verify datasets are set - if not, this is a critical error
            if self.train_set is None or self.valid_set is None:
                raise RuntimeError(
                    f'CRITICAL ERROR: Setup lock is active but datasets are None! '
                    f'Fold {self.current_fold_idx}, train_set={self.train_set is not None}, '
                    f'valid_set={self.valid_set is not None}. '
                    f'This indicates setup_fold() failed or datasets were reset.'
                )
            # Return immediately - do NOT call parent setup() or do anything else
            # This prevents Lightning from resetting our fold-specific datasets
            return
        
        # CRITICAL CHECK #1.5: If datasets exist, preserve them even if lock isn't set
        # This handles the case where setup_fold() was called but lock wasn't set
        # (shouldn't happen, but defensive programming)
        if self.train_set is not None and self.valid_set is not None:
            log.debug(
                f'CV setup() called with stage={stage}. '
                f'Datasets already exist - preserving them. '
                f'Fold setup: _fold_setup_complete={self._fold_setup_complete}, '
                f'current_fold_idx={self.current_fold_idx}, '
                f'train_set={self.train_set is not None}, '
                f'valid_set={self.valid_set is not None}.'
            )
            # Set lock and flags to ensure state persists
            if self.current_fold_idx is not None:
                self._setup_lock = True
                self._fold_setup_complete = True
            # Return immediately - do NOT call parent setup() or do anything else
            return
        
        # CRITICAL CHECK #2: If fold is already set up, NEVER touch datasets
        # This is the most important check - Lightning will call this during trainer.fit()
        # Use both current_fold_idx and _fold_setup_complete flag for redundancy
        if self._fold_setup_complete or self.current_fold_idx is not None:
            log.debug(
                f'CV setup() called by Lightning with stage={stage}. '
                f'Fold setup detected (_fold_setup_complete={self._fold_setup_complete}, '
                f'current_fold_idx={self.current_fold_idx}). '
                f'Preserving datasets (train_set={self.train_set is not None}, '
                f'valid_set={self.valid_set is not None}).'
            )
            if self.train_set is None or self.valid_set is None:
                raise RuntimeError(
                    f'Fold {self.current_fold_idx} was set up but datasets are None. '
                    f'train_set={self.train_set is not None}, valid_set={self.valid_set is not None}'
                )
            self._setup_lock = True
            return
        
        if self.train_set is not None or self.valid_set is not None:
            if self.train_set is not None and self.valid_set is not None:
                self._fold_setup_complete = True
                if self.current_fold_idx is None:
                    self.current_fold_idx = 0
            return
        
        log.debug(f'CV setup() called with stage={stage} (no fold set up yet). Computing global statistics only.')
        
        # Compute global statistics if not already computed
        # This is safe to do before setup_fold() and will be reused
        if self.global_channel_means is None or self.global_channel_stds is None:
            train_cfg = self.cfg_datasets.get('train')
            train_transforms_cfg = self.transforms.get('train')

            if 'pad_or_crop' in train_transforms_cfg:
                pad_or_crop_cfg = train_transforms_cfg.get('pad_or_crop')
                self.global_channel_means, self.global_channel_stds = (
                    self._compute_global_statistics(train_cfg, pad_or_crop_cfg)
                )
                self._update_transforms_with_statistics()
        
        # IMPORTANT: Do NOT call parent setup() - it would try to set up datasets
        # which we don't want in CV mode. Datasets will be set up by setup_fold()
        # Do NOT set up datasets here - wait for setup_fold() to be called

    def get_n_folds(self) -> int:
        """Get the number of CV folds.

        :return: Number of folds.
        """
        splits = self._generate_cv_splits()
        return len(splits)

    def train_dataloader(
        self,
    ) -> DataLoader | list[DataLoader] | dict[str, DataLoader]:
        """Return train dataloader for current fold.

        :return: Train dataloader.
        """
        if not self.is_fold_setup:
            raise RuntimeError(
                f'Fold not set up! Call setup_fold(fold_idx) before accessing train_dataloader(). '
                f'current_fold_idx={self.current_fold_idx}, '
                f'train_set={self.train_set is not None}, '
                f'valid_set={self.valid_set is not None}'
            )
        return DataLoader(self.train_set, **self.cfg_loaders.get('train'))

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Return validation dataloader for current fold.

        :return: Validation dataloader.
        """
        if not self.is_fold_setup:
            raise RuntimeError(
                f'Fold not set up! Call setup_fold(fold_idx) before accessing val_dataloader(). '
                f'current_fold_idx={self.current_fold_idx}, '
                f'train_set={self.train_set is not None}, '
                f'valid_set={self.valid_set is not None}'
            )
        return DataLoader(self.valid_set, **self.cfg_loaders.get('valid'))

