from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datamodules.components.eeg_transforms import EEGNormalize, EEGPadOrCrop
from src.datamodules.components.transforms import TransformsWrapper
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class SingleDataModule(LightningDataModule):
    """Example of LightningDataModule for single dataset.

    A DataModule implements 5 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def predict_dataloader(self):
            # return predict dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        """DataModule with standalone train, val and test dataloaders.

        :param datasets: Datasets config.
        :param loaders: Loaders config.
        :param transforms: Transforms config.
        """

        super().__init__()
        self.cfg_datasets = datasets
        self.cfg_loaders = loaders
        self.transforms = transforms
        self.train_set: Dataset | None = None
        self.valid_set: Dataset | None = None
        self.test_set: Dataset | None = None
        self.predict_set: dict[str, Dataset] = OrderedDict()
        self.global_channel_means: np.ndarray | None = None
        self.global_channel_stds: np.ndarray | None = None

    def _get_dataset_(
        self, split_name: str, dataset_name: str | None = None
    ) -> Dataset:
        transforms = TransformsWrapper(self.transforms.get(split_name))
        cfg = self.cfg_datasets.get(split_name)
        if dataset_name:
            cfg = cfg.get(dataset_name)
        dataset: Dataset = hydra.utils.instantiate(cfg, transforms=transforms)
        return dataset

    def _compute_global_statistics(self, dataset_cfg: DictConfig, pad_or_crop_cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
        """Compute global channel-wise mean and std from dataset by loading files directly.
        
        :param dataset_cfg: Dataset configuration.
        :param pad_or_crop_cfg: PadOrCrop transform configuration.
        :return: Tuple of (channel_means, channel_stds) arrays of shape (num_channels,).
        """
        log.info("Вычисление глобальных статистик нормализации по каналам...")
        
        # Load annotation file
        data_path = dataset_cfg.get('data_path')
        path_column = dataset_cfg.get('path_column', 'crop_path')
        read_mode = dataset_cfg.get('read_mode', 'npy')
        
        if not data_path:
            raise ValueError("data_path must be provided in dataset config")
        
        annotation_path = Path(data_path)
        if not annotation_path.exists():
            raise RuntimeError(f"Annotation file '{annotation_path}' does not exist.")
        
        # Load parquet file
        df = pd.read_parquet(annotation_path)
        if path_column not in df.columns:
            raise ValueError(f"Column '{path_column}' not found in parquet file.")
        
        # Instantiate pad_or_crop transform
        pad_or_crop = hydra.utils.instantiate(pad_or_crop_cfg, _convert_='object')
        
        all_data = []
        file_paths = df[path_column].tolist()
        
        # Load data directly from files
        log.info(f"Загрузка {len(file_paths)} файлов для вычисления статистик...")
        for file_path in tqdm(file_paths, desc="Загрузка данных"):
            file_path = Path(file_path)
            if not file_path.exists():
                log.warning(f"Файл не найден: {file_path}, пропускаем")
                continue
            
            # Load raw data
            if read_mode == 'npy':
                image = np.load(file_path).astype(np.float32)
            else:
                raise ValueError(f"Unsupported read_mode for statistics: {read_mode}")
            
            # Apply only pad_or_crop transform (no normalization)
            if image.ndim == 2:
                # (channels, length)
                result = pad_or_crop(image=image)
                image = result['image']
            else:
                log.warning(f"Неожиданная размерность данных: {image.shape}, пропускаем")
                continue
            
            # Ensure correct shape: (channels, length)
            if image.ndim == 2:
                all_data.append(image.astype(np.float32))
            else:
                log.warning(f"Неожиданная форма данных после pad_or_crop: {image.shape}, пропускаем")
        
        if not all_data:
            raise ValueError("Не удалось загрузить данные для вычисления статистик")
        
        # Stack all data: (num_samples, num_channels, signal_length)
        all_data_array = np.stack(all_data, axis=0)
        log.info(f"Загружено {all_data_array.shape[0]} образцов, форма: {all_data_array.shape}")
        
        # Reshape to (num_channels, total_samples * signal_length)
        # This matches the notebook approach: transpose(1, 0, 2).reshape(30, -1)
        num_channels = all_data_array.shape[1]
        flat = all_data_array.transpose(1, 0, 2).reshape(num_channels, -1)
        
        # Compute mean and std per channel
        channel_means = flat.mean(axis=1)  # (num_channels,)
        channel_stds = flat.std(axis=1)   # (num_channels,)
        
        # Avoid division by zero
        channel_stds = np.where(channel_stds == 0, 1.0, channel_stds)
        
        log.info(f"Вычислены глобальные статистики:")
        log.info(f"  Channel means shape: {channel_means.shape}")
        log.info(f"  Channel stds shape: {channel_stds.shape}")
        log.info(f"  Means range: [{channel_means.min():.6e}, {channel_means.max():.6e}]")
        log.info(f"  Stds range: [{channel_stds.min():.6e}, {channel_stds.max():.6e}]")
        
        return channel_means, channel_stds

    def _update_transforms_with_statistics(self) -> None:
        """Update transforms config with computed global statistics."""
        if self.global_channel_means is None or self.global_channel_stds is None:
            return
        
        log.info("Обновление transforms с глобальными статистиками...")
        
        # Update transforms config for all splits
        updated_splits = []
        for split_name in ['train', 'valid', 'test', 'predict']:
            if split_name not in self.transforms:
                continue
            
            split_transforms = self.transforms.get(split_name)
            if not split_transforms or 'order' not in split_transforms:
                continue
            
            # Find normalize transform and update it
            if 'normalize' in split_transforms:
                normalize_cfg = split_transforms.get('normalize')
                if normalize_cfg:
                    target = normalize_cfg.get('_target_', '')
                    if 'EEGNormalize' in target:
                        # Update mean and std in config
                        # Convert numpy arrays to lists for YAML serialization
                        normalize_cfg['mean'] = self.global_channel_means.tolist()
                        normalize_cfg['std'] = self.global_channel_stds.tolist()
                        updated_splits.append(split_name)
        
        if updated_splits:
            log.info(f"Обновлен normalize transform для splits: {', '.join(updated_splits)}")
        else:
            log.warning("Не найдено normalize transforms для обновления")

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            # First, compute global statistics from train dataset files directly
            train_cfg = self.cfg_datasets.get('train')
            train_transforms_cfg = self.transforms.get('train')
            
            # Get pad_or_crop config
            if 'pad_or_crop' not in train_transforms_cfg:
                raise ValueError("pad_or_crop transform must be present in train transforms")
            pad_or_crop_cfg = train_transforms_cfg.get('pad_or_crop')
            
            # Compute global statistics by loading files directly
            self.global_channel_means, self.global_channel_stds = self._compute_global_statistics(
                train_cfg, pad_or_crop_cfg
            )
            
            # Update transforms with computed statistics
            self._update_transforms_with_statistics()
            
            # Now load all datasets with updated transforms
            self.train_set = self._get_dataset_('train')
            self.valid_set = self._get_dataset_('valid')
            self.test_set = self._get_dataset_('test')
        # load predict datasets only if it exists in config
        if (stage == 'predict') and self.cfg_datasets.get('predict'):
            for dataset_name in self.cfg_datasets.get('predict').keys():
                self.predict_set[dataset_name] = self._get_dataset_(
                    'predict', dataset_name=dataset_name
                )

    def train_dataloader(
        self,
    ) -> DataLoader | list[DataLoader] | dict[str, DataLoader]:
        return DataLoader(self.train_set, **self.cfg_loaders.get('train'))

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(self.valid_set, **self.cfg_loaders.get('valid'))

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(self.test_set, **self.cfg_loaders.get('test'))

    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get('predict'))
            )
        return loaders

    def teardown(self, stage: str | None = None):
        """Clean up after fit or test."""
        pass


class MultipleDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        """DataModule with multiple train, val and test dataloaders.

        :param datasets: Datasets config.
        :param loaders: Loaders config.
        :param transforms: Transforms config.
        """

        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )
        self.train_set: dict[str, Dataset] | None = None
        self.valid_set: dict[str, Dataset] | None = None
        self.test_set: dict[str, Dataset] | None = None
        self.predict_set: dict[str, Dataset] = OrderedDict()

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.train_set`, `self.valid_set`,
        `self.test_set`, `self.predict_set`.

        This method is called by lightning with both `trainer.fit()` and
        `trainer.test()`, so be careful not to execute things like random split
        twice!
        """
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            self.train_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get('train').keys():
                self.train_set[dataset_name] = self._get_dataset_(
                    'train', dataset_name=dataset_name
                )
            self.valid_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get('valid').keys():
                self.valid_set[dataset_name] = self._get_dataset_(
                    'valid', dataset_name=dataset_name
                )
            self.test_set = OrderedDict()
            for dataset_name in self.cfg_datasets.get('test').keys():
                self.test_set[dataset_name] = self._get_dataset_(
                    'test', dataset_name=dataset_name
                )
        # load predict datasets only if it exists in config
        if (stage == 'predict') and self.cfg_datasets.get('predict'):
            for dataset_name in self.cfg_datasets.get('predict').keys():
                self.predict_set[dataset_name] = self._get_dataset_(
                    'predict', dataset_name=dataset_name
                )

    def train_dataloader(
        self,
    ) -> DataLoader | list[DataLoader] | dict[str, DataLoader]:
        loaders = dict()
        for dataset_name, dataset in self.train_set.items():
            loaders[dataset_name] = DataLoader(
                dataset, **self.cfg_loaders.get('train')
            )
        return loaders

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = []
        for _, dataset in self.valid_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get('valid'))
            )
        return loaders

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = []
        for _, dataset in self.test_set.items():
            loaders.append(DataLoader(dataset, **self.cfg_loaders.get('test')))
        return loaders

    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = []
        for _, dataset in self.predict_set.items():
            loaders.append(
                DataLoader(dataset, **self.cfg_loaders.get('predict'))
            )
        return loaders
