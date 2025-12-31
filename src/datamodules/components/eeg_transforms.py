from typing import Any

import numpy as np
import torch


class EEGPadOrCrop:
    """Pad or crop EEG signal to fixed size."""

    def __init__(
        self,
        num_channels: int = 127,
        signal_length: int = 2500,
        p: float = 1.0,
    ) -> None:
        """EEGPadOrCrop initialization.

        :param num_channels: Target number of channels.
        :param signal_length: Target signal length.
        :param p: Probability of applying transform.
        """
        self.num_channels = num_channels
        self.signal_length = signal_length
        self.p = p

    def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Apply padding or cropping.

        :param image: EEG signal array of shape (channels, length).
        :param kwargs: Additional arguments.
        :return: Dictionary with padded/cropped 'image'.
        """
        if np.random.random() > self.p:
            return {"image": image, **kwargs}

        eeg_signal = image.copy()
        current_channels, current_length = eeg_signal.shape

        # Handle channels dimension
        if current_channels < self.num_channels:
            # Pad with zeros
            pad_channels = self.num_channels - current_channels
            eeg_signal = np.pad(
                eeg_signal, ((0, pad_channels), (0, 0)), mode="constant"
            )
        elif current_channels > self.num_channels:
            # Crop
            eeg_signal = eeg_signal[: self.num_channels, :]

        # Handle length dimension
        if current_length < self.signal_length:
            # Pad with zeros
            pad_length = self.signal_length - current_length
            eeg_signal = np.pad(
                eeg_signal, ((0, 0), (0, pad_length)), mode="constant"
            )
        elif current_length > self.signal_length:
            # Crop
            eeg_signal = eeg_signal[:, : self.signal_length]

        return {"image": eeg_signal, **kwargs}


class EEGNormalize:
    """Normalize EEG signal by channel-wise z-score normalization."""

    def __init__(
        self,
        mean: float | np.ndarray | None = None,
        std: float | np.ndarray | None = None,
        per_channel: bool = True,
        p: float = 1.0,
    ) -> None:
        """EEGNormalize initialization.

        :param mean: Mean value(s) for normalization. 
            If None, computed from data (per sample if per_channel=True).
            If array of shape (num_channels,), used for global per-channel normalization.
        :param std: Standard deviation value(s) for normalization.
            If None, computed from data (per sample if per_channel=True).
            If array of shape (num_channels,), used for global per-channel normalization.
        :param per_channel: If True, normalize each channel independently.
        :param p: Probability of applying transform.
        """
        self.mean = mean
        self.std = std
        self.per_channel = per_channel
        self.p = p

    def __call__(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        """Apply normalization.

        :param image: EEG signal array of shape (channels, length).
        :param kwargs: Additional arguments.
        :return: Dictionary with normalized 'image'.
        """
        if np.random.random() > self.p:
            return {"image": image, **kwargs}

        eeg_signal = image.copy()

        if self.per_channel:
            # Normalize each channel independently
            if self.mean is None or self.std is None:
                # Compute mean and std per channel for this sample
                mean = eeg_signal.mean(axis=1, keepdims=True)
                std = eeg_signal.std(axis=1, keepdims=True)
                # Avoid division by zero
                std = np.where(std == 0, 1.0, std)
            else:
                # Use provided mean and std (can be global per-channel values)
                # Convert to numpy array if it's a list
                if isinstance(self.mean, (list, tuple)):
                    mean_arr = np.array(self.mean)
                elif isinstance(self.mean, np.ndarray):
                    mean_arr = self.mean
                else:
                    mean_arr = np.array([self.mean])
                
                # Ensure correct shape for broadcasting: (channels,) -> (channels, 1)
                if mean_arr.ndim == 1:
                    mean = mean_arr[:, np.newaxis]
                elif mean_arr.ndim == 2 and mean_arr.shape[1] == 1:
                    mean = mean_arr
                else:
                    mean = mean_arr
                
                # Convert to numpy array if it's a list
                if isinstance(self.std, (list, tuple)):
                    std_arr = np.array(self.std)
                elif isinstance(self.std, np.ndarray):
                    std_arr = self.std
                else:
                    std_arr = np.array([self.std])
                
                # Ensure correct shape for broadcasting: (channels,) -> (channels, 1)
                if std_arr.ndim == 1:
                    std = std_arr[:, np.newaxis]
                elif std_arr.ndim == 2 and std_arr.shape[1] == 1:
                    std = std_arr
                else:
                    std = std_arr
                
                # Avoid division by zero
                std = np.where(std == 0, 1.0, std)
        else:
            # Normalize globally
            if self.mean is None or self.std is None:
                mean = eeg_signal.mean()
                std = eeg_signal.std()
                if std == 0:
                    std = 1.0
            else:
                mean = self.mean
                std = self.std

        eeg_signal = (eeg_signal - mean) / std

        return {"image": eeg_signal, **kwargs}

