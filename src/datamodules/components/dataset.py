import io
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        transforms: Callable | None = None,
        read_mode: str = 'pillow',
        to_gray: bool = False,
    ) -> None:
        """BaseDataset.

        :param transforms: Transforms.
        :param read_mode: Image read mode, `pillow`, `cv2`, or `npy`. Default to `pillow`.
        :param to_gray: Images to gray mode. Default to False.
        """

        self.read_mode = read_mode
        self.to_gray = to_gray
        self.transforms = transforms

    def _read_image_(self, image: Any) -> np.ndarray:
        """Read image from source.

        :param image: Image source. Could be str, Path, bytes, or numpy array.
        :return: Loaded image as numpy array.
        """

        if self.read_mode == 'npy':
            # Read .npy file (for EEG or other numpy array data)
            if isinstance(image, (str, Path)):
                image = np.load(image)
            elif isinstance(image, np.ndarray):
                image = image.copy()
            else:
                raise ValueError(
                    f"npy read_mode expects str, Path or np.ndarray, got {type(image)}"
                )
            # Ensure 2D or 3D array (for compatibility with image processing)
            if image.ndim == 1:
                image = image.reshape(1, -1)
            return image
        elif self.read_mode == 'pillow':
            if not isinstance(image, (str, Path)):
                image = io.BytesIO(image)
            image = np.asarray(Image.open(image).convert('RGB'))
        elif self.read_mode == 'cv2':
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError('use pillow, cv2, or npy')
        if self.to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image

    def _process_image_(self, image: np.ndarray) -> torch.Tensor:
        """Process image, including transforms, etc.

        :param image: Image in np.ndarray format.
        :return: Image prepared for dataloader.
        """

        if self.transforms:
            image = self.transforms(image=image)['image']
        
        # Convert to tensor if needed
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        
        # Handle different array dimensions
        if image.ndim == 2:
            # 2D array (e.g., EEG: channels x length) -> add channel dimension
            image = image.unsqueeze(0)  # (1, channels, length)
        elif image.ndim == 3:
            # Check if it's (H, W, C) format (RGB image) or (C, H, W) already
            # Assume (H, W, C) if last dimension is 3 or less (common for RGB)
            if image.shape[2] <= 3:
                # 3D array (e.g., RGB image: H x W x C) -> permute to (C, H, W)
                image = image.permute(2, 0, 1)
            # Otherwise assume it's already in (C, H, W) format
        # If already 4D or other shape, use as is
        
        return image

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()
