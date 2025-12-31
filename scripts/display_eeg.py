#!/usr/bin/env python3
"""
Simple script to display EEG data using MNE.

Usage:
    python display_eeg.py <file_path> [options]

Examples:
    # Display from .set file (EEGLAB format)
    python display_eeg.py /path/to/file.set
il
    # Display from .npy file
    python display_eeg.py /path/to/file.npy --sfreq 500

    # Display specific time range
    python display_eeg.py file.set --start 10 --duration 5
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np


def load_eeg_data(file_path: str, sfreq: float = None) -> mne.io.BaseRaw:
    """Load EEG data from file and return MNE Raw object."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load .set files (EEGLAB format)
    if file_path.suffix == ".set":
        print(f"Loading EEGLAB file: {file_path}")
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        print(f"Loaded: {raw.info['nchan']} channels, "
              f"sampling rate: {raw.info['sfreq']} Hz, "
              f"duration: {raw.times[-1]:.2f} seconds")
        return raw

    # Load .npy files (NumPy arrays)
    elif file_path.suffix == ".npy":
        if sfreq is None:
            raise ValueError(
                "For .npy files, you must specify --sfreq (sampling frequency in Hz)"
            )

        print(f"Loading NumPy array: {file_path}")
        data = np.load(file_path)

        # Handle different array shapes
        if data.ndim == 2:
            # Shape: (n_channels, n_samples)
            n_channels, n_samples = data.shape
        elif data.ndim == 3:
            # Shape: (batch, n_channels, n_samples) - take first item
            print(f"Warning: 3D array detected, using first item")
            data = data[0]
            n_channels, n_samples = data.shape
        else:
            raise ValueError(
                f"Unsupported array shape: {data.shape}. "
                "Expected (n_channels, n_samples) or (batch, n_channels, n_samples)"
            )

        # Create channel names
        ch_names = [f"EEG {i+1:03d}" for i in range(n_channels)]

        # Create MNE info object
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types="eeg",
        )

        # Create Raw object
        raw = mne.io.RawArray(data, info, verbose=False)
        print(f"Loaded: {n_channels} channels, "
              f"sampling rate: {sfreq} Hz, "
              f"duration: {raw.times[-1]:.2f} seconds")
        return raw

    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            "Supported formats: .set (EEGLAB), .npy (NumPy)"
        )


def display_eeg(
    raw: mne.io.BaseRaw,
    start: float = None,
    duration: float = None,
    channels: list = None,
    scalings: dict = None,
):
    """Display EEG data using MNE's interactive plotter."""
    # Set default scalings if not provided
    if scalings is None:
        scalings = {"eeg": 50e-6}  # 50 microvolts

    # Select channels if specified
    if channels:
        raw = raw.pick_channels(channels)

    # Select time range if specified
    if start is not None or duration is not None:
        if start is None:
            start = 0.0
        if duration is None:
            duration = raw.times[-1] - start
        raw = raw.crop(tmin=start, tmax=start + duration)

    print("\nDisplaying EEG data...")
    print("Close the plot window to exit.\n")

    # Display the data
    raw.plot(
        scalings=scalings,
        duration=10.0,  # Show 10 seconds at a time
        n_channels=min(20, len(raw.ch_names)),  # Show up to 20 channels
        show=True,
        block=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Display EEG data using MNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to EEG file (.set or .npy)",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Sampling frequency in Hz (required for .npy files)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Start time in seconds (default: beginning)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to display in seconds (default: all)",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help="Channel names to display (default: all)",
    )
    parser.add_argument(
        "--scaling",
        type=float,
        default=50e-6,
        help="Scaling factor for EEG channels in volts (default: 50e-6 = 50 µV)",
    )

    args = parser.parse_args()

    try:
        # Load data
        raw = load_eeg_data(args.file_path, sfreq=args.sfreq)

        # Display
        display_eeg(
            raw,
            start=args.start,
            duration=args.duration,
            channels=args.channels,
            scalings={"eeg": args.scaling},
        )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

