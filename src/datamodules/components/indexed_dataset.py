"""Indexed dataset wrapper for efficient subset selection."""

from typing import Any, Callable

from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    """Wrapper dataset that selects a subset of indices from another dataset.

    This allows efficient fold-based splitting without reloading data.
    """

    def __init__(self, base_dataset: Dataset, indices: list[int]) -> None:
        """Create a subset dataset from base dataset using indices.

        :param base_dataset: Base dataset to subset.
        :param indices: List of indices to select from base dataset.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        """Return the length of the subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get item from subset by mapping to base dataset index.

        :param idx: Index in the subset.
        :return: Item from base dataset.
        """
        if idx >= len(self.indices):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self.indices)}')
        base_idx = self.indices[idx]
        return self.base_dataset[base_idx]

