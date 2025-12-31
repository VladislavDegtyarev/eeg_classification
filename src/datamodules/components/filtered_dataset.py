"""Filtered dataset wrapper that selects specific rows from a parquet-based dataset."""

from typing import Any, Callable

import pandas as pd
from torch.utils.data import Dataset


class FilteredDataset(Dataset):
    """Wrapper dataset that filters a base dataset by row indices from parquet file.

    This is more efficient than IndexedDataset for parquet-based datasets because
    it filters the annotation dict directly rather than creating a new dataset instance.
    """

    def __init__(
        self, base_dataset: Dataset, parquet_path: str, row_indices: list[int], path_column: str
    ) -> None:
        """Create a filtered dataset from base dataset using row indices.

        :param base_dataset: Base dataset instance (will be modified).
        :param parquet_path: Path to parquet file to filter.
        :param row_indices: List of row indices to select from parquet.
        :param path_column: Column name containing file paths (used as keys).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.parquet_path = parquet_path
        self.row_indices = row_indices

        # Load parquet and filter to get the keys we want
        df = pd.read_parquet(parquet_path)
        if path_column not in df.columns:
            raise ValueError(f'Column {path_column} not found in parquet file')

        # Get the paths for the selected rows
        filtered_df = df.iloc[row_indices]
        self.filtered_keys = filtered_df[path_column].tolist()

        # Create a mapping from filtered index to original dataset key
        # We need to match the keys in base_dataset.annotation
        if hasattr(base_dataset, 'annotation'):
            # Filter the annotation dict to only include our keys
            self.filtered_annotation = {
                str(key): base_dataset.annotation[str(key)]
                for key in self.filtered_keys
                if str(key) in base_dataset.annotation
            }
            # Update the base dataset's keys to only include filtered ones
            original_keys = base_dataset.keys.copy()
            base_dataset.keys = list(self.filtered_annotation.keys())
        else:
            raise ValueError('Base dataset must have annotation attribute')

    def __len__(self) -> int:
        """Return the length of the filtered dataset."""
        return len(self.filtered_keys)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item from filtered dataset.

        :param idx: Index in the filtered dataset.
        :return: Item from base dataset.
        """
        if idx >= len(self.filtered_keys):
            raise IndexError(
                f'Index {idx} out of range for filtered dataset of size {len(self.filtered_keys)}'
            )

        # Find the key and get its index in the base dataset
        key = str(self.filtered_keys[idx])
        if key in self.base_dataset.keys:
            key_idx = self.base_dataset.keys.index(key)
            return self.base_dataset[key_idx]
        else:
            raise KeyError(f'Key {key} not found in base dataset')

