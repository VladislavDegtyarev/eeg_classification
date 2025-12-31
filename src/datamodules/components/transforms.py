from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        :param transforms_cfg: Transforms config.
        """

        augmentations = []
        if not transforms_cfg.get('order'):
            raise RuntimeError(
                'TransformsWrapper requires param <order>, i.e.'
                'order of augmentations as List[augmentation name]'
            )
        for augmentation_name in transforms_cfg.get('order'):
            augmentation = hydra.utils.instantiate(
                transforms_cfg.get(augmentation_name), _convert_='object'
            )
            augmentations.append(augmentation)
        
        # Check if all transforms are albumentations transforms
        # If yes, use albumentations.Compose, otherwise use custom composition
        try:
            self.augmentations = albumentations.Compose(augmentations)
            self.use_albumentations = True
        except (TypeError, AttributeError):
            # If not all transforms are albumentations, use custom composition
            self.augmentations = augmentations
            self.use_albumentations = False

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        :param image: Input image.
        :param kwargs: Additional arguments.
        :return: Transformation results.
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)
        
        if self.use_albumentations:
            return self.augmentations(image=image, **kwargs)
        else:
            # Custom composition for non-albumentations transforms
            result = {"image": image, **kwargs}
            for transform in self.augmentations:
                result = transform(**result)
            return result
