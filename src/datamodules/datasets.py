import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.datamodules.components.dataset import BaseDataset
from src.datamodules.components.h5_file import H5PyFile
from src.datamodules.components.parse import parse_image_paths


class ClassificationDataset(BaseDataset):
    def __init__(
        self,
        data_path: str | None = None,
        json_path: str | None = None,
        txt_path: str | None = None,
        parquet_path: str | None = None,
        transforms: Callable | None = None,
        read_mode: str = 'pillow',
        to_gray: bool = False,
        include_names: bool = False,
        shuffle_on_load: bool = True,
        label_type: str = 'torch.LongTensor',
        path_column: str | None = None,
        target_column: str | None = None,
        source_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """ClassificationDataset.

        :param data_path: Path to annotation file (json, txt, or parquet) or HDF5 file.
            File type is determined by extension. Preferred over json_path/txt_path/parquet_path.
        :param json_path: (Deprecated) Path to annotation json. Use data_path instead.
        :param txt_path: (Deprecated) Path to annotation txt. Use data_path instead.
        :param parquet_path: (Deprecated) Path to parquet file. Use data_path instead.
        :param transforms: Transforms.
        :param read_mode: Image read mode, `pillow`, `cv2`, or `npy`. Default to `pillow`.
        :param to_gray: Images to gray mode. Default to False.
        :param include_names: If True, then `__getitem__` method would return image
            name/path value with key `name`. Default to False.
        :param shuffle_on_load: Deterministically shuffle the dataset on load
            to avoid the case when Dataset slice contains only one class due to
            annotations dict keys order. Default to True.
        :param label_type: Label torch.tensor type. Default to torch.FloatTensor.
        :param path_column: Column name in parquet file containing file paths.
            Required when data_path points to parquet file.
        :param target_column: Column name in parquet file containing labels.
            Required when data_path points to parquet file.
        :param source_dir: Directory containing source files (images, .npy files, etc.).
            If None and data_path is not HDF5, files are expected to be at paths
            specified in annotations.
        :param kwargs: Additional keyword arguments for H5PyFile class.
        """

        super().__init__(transforms, read_mode, to_gray)
        
        # Determine annotation file path (backward compatibility)
        if data_path:
            annotation_path = Path(data_path)
        elif parquet_path:
            annotation_path = Path(parquet_path)
        elif json_path:
            annotation_path = Path(json_path)
        elif txt_path:
            annotation_path = Path(txt_path)
        else:
            raise ValueError(
                "One of data_path, json_path, txt_path, or parquet_path must be provided."
            )
        
        if not annotation_path.exists():
            raise RuntimeError(f"'{annotation_path}' does not exist.")
        
        suffix = annotation_path.suffix.lower()
        
        # Handle annotation files
        if suffix == '.parquet':
            if not path_column or not target_column:
                raise ValueError(
                    "path_column and target_column must be provided when using parquet file"
                )
            df = pd.read_parquet(annotation_path)
            if path_column not in df.columns:
                raise ValueError(f"Column '{path_column}' not found in parquet file.")
            if target_column not in df.columns:
                raise ValueError(f"Column '{target_column}' not found in parquet file.")
            self.annotation = {
                str(row[path_column]): row[target_column]
                for _, row in df.iterrows()
            }
        elif suffix == '.json':
            with open(annotation_path) as json_file:
                self.annotation = json.load(json_file)
        elif suffix == '.txt':
            self.annotation = {}
            with open(annotation_path) as txt_file:
                for line in txt_file:
                    _, label, path = line[:-1].split('\t')
                    self.annotation[path] = label
        else:
            raise ValueError(
                f"Unsupported annotation file type: {suffix}. "
                "Supported types: .json, .txt, .parquet"
            )
        
        # Handle source directory or HDF5 file
        self.data_file = None
        if source_dir:
            source_path = Path(source_dir)
            if source_path.is_file() and source_path.suffix.lower() == '.h5':
                # HDF5 file for data storage
                self.data_file = H5PyFile(str(source_path), **kwargs)
                self.data_path = Path('')
            else:
                # Regular directory with files
                self.data_path = source_path
        else:
            # If no source_dir provided, assume files are at paths from annotations
            self.data_path = Path('')

        self.keys = list(self.annotation)
        if shuffle_on_load:
            random.Random(shuffle_on_load).shuffle(self.keys)

        self.include_names = include_names
        self.label_type = label_type

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key if self.data_path else Path(key)
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image = self._process_image_(image)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        if self.include_names:
            return {'image': image.float(), 'label': label, 'name': key}
        return {'image': image.float(), 'label': label}

    def get_weights(self) -> list[float]:
        label_list = [self.annotation[key] for key in self.keys]
        weights = 1.0 / np.bincount(label_list)
        return weights.tolist()


class ClassificationVicRegDataset(ClassificationDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        data_file = self.data_file
        if data_file is None:
            source = self.data_path / key
        else:
            source = data_file[key]
        image = self._read_image_(source)
        image1, image2 = np.copy(image), np.copy(image)
        # albumentations returns random augmentation on each __call__
        z1 = self._process_image_(image1)
        z2 = self._process_image_(image2)
        label = torch.tensor(self.annotation[key]).type(self.label_type)
        return {"z1": z1.float(), "z2": z2.float(), "label": label}


class NoLabelsDataset(BaseDataset):
    def __init__(
        self,
        file_paths: Optional[List[str]] = None,
        dir_paths: Optional[List[str]] = None,
        txt_paths: Optional[List[str]] = None,
        json_paths: Optional[List[str]] = None,
        dirname: Optional[str] = None,
        transforms: Optional[Callable] = None,
        read_mode: str = "pillow",
        to_gray: bool = False,
        include_names: bool = False,
    ) -> None:
        """NoLabelsDataset.

        Args:
            file_paths (:obj:`List[str]`, optional): List of files.
            dir_paths (:obj:`List[str]`, optional): List of directories.
            txt_paths (:obj:`List[str]`, optional): List of TXT files.
            json_paths (:obj:`List[str]`, optional): List of JSON files.
            dirname (:obj:`str`, optional): Images source dir.
            transforms (Callable): Transforms.
            read_mode (str): Image read mode, `pillow` or `cv2`. Default to `pillow`.
            to_gray (bool): Images to gray mode. Default to False.
            include_names (bool): If True, then `__getitem__` method would return image
                name/path value with key `name`. Default to False.
        """

        super().__init__(transforms, read_mode, to_gray)
        if file_paths or dir_paths or txt_paths:
            self.keys = parse_image_paths(
                file_paths=file_paths, dir_paths=dir_paths, txt_paths=txt_paths
            )
        elif json_paths:
            self.keys = []
            for json_path in json_paths:
                with open(json_path) as json_file:
                    data = json.load(json_file)
                for path in data.keys():
                    self.keys.append(path)
        else:
            raise ValueError("Requires data_paths or json_paths.")
        self.dirname = Path(dirname if dirname else "")
        self.include_names = include_names

    def __getitem__(self, index: int) -> Dict[str, Any]:
        key = self.keys[index]
        path = self.dirname / Path(key)
        image = self._read_image_(path)
        image = self._process_image_(image)
        if self.include_names:
            return {"image": image, "name": key}
        return {"image": image}

    def __len__(self) -> int:
        return len(self.keys)
