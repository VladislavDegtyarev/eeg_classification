from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10

from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


def _to_chw_tensor(image: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert image (HWC numpy array or tensor) to CHW torch tensor."""
    if isinstance(image, torch.Tensor):
        tensor = image
    else:
        tensor = torch.from_numpy(image)
    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    return tensor.contiguous()


class _CIFAR10Transform:
    """Callable that adapts Albumentations pipeline to torchvision expectations."""

    def __init__(self, transforms_cfg: DictConfig) -> None:
        self.transforms = TransformsWrapper(transforms_cfg)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        result = self.transforms(image=image)
        transformed = result["image"] if isinstance(result, dict) else result
        return _to_chw_tensor(transformed).float()


class _CIFAR10Dataset(Dataset):
    """Wrap torchvision CIFAR-10 dataset to return dict samples."""

    def __init__(self, dataset: CIFAR10) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        sample = {
            "image": _to_chw_tensor(image).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return sample


class CIFAR10DataModule(SingleDataModule):
    """LightningDataModule tailored for CIFAR-10 classification."""

    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )
        self.cfg_datasets = datasets

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        """Download CIFAR-10 data if required."""
        data_dir = self.cfg_datasets.get("data_dir")
        CIFAR10(data_dir, train=True, download=True)
        CIFAR10(data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Populate train/valid/test datasets exactly once."""
        if self.train_set or self.valid_set or self.test_set:
            return

        data_dir = self.cfg_datasets.get("data_dir")
        train_transform = _CIFAR10Transform(self.transforms.train)
        eval_transform = _CIFAR10Transform(self.transforms.valid_test_predict)

        base_train = CIFAR10(
            data_dir,
            train=True,
            download=False,
            transform=train_transform,
        )
        base_eval = CIFAR10(
            data_dir,
            train=False,
            download=False,
            transform=eval_transform,
        )

        train_dataset: Dataset = _CIFAR10Dataset(base_train)
        valid_dataset: Dataset = _CIFAR10Dataset(
            CIFAR10(
                data_dir,
                train=False,
                download=False,
                transform=eval_transform,
            )
        )
        test_dataset: Dataset = _CIFAR10Dataset(base_eval)

        split = self.cfg_datasets.get("train_val_test_split")
        if split:
            generator = torch.Generator().manual_seed(
                self.cfg_datasets.get("seed", 0)
            )
            train_dataset, valid_dataset = random_split(
                train_dataset, split, generator=generator
            )

        self.train_set = train_dataset
        self.valid_set = valid_dataset
        self.test_set = test_dataset
