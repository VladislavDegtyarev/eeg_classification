# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG Classification framework built on PyTorch Lightning and Hydra. Designed for deep learning research on EEG signals (e.g., eye-state classification) with full reproducibility and configurability.

## Common Commands

```bash
# Install dependencies
poetry install

# Training
python src/train.py experiment=eeg_classification.yaml
python src/train.py experiment=eeg_cv.yaml  # cross-validation

# Override configs from CLI
python src/train.py experiment=eeg_classification.yaml module.optimizer.lr=0.001

# Hyperparameter search
python src/train.py -m hparams_search=mnist_optuna

# Evaluation
python src/eval.py ckpt_path=<path_to_checkpoint>

# Tests
pytest                           # all tests
pytest tests/test_train.py       # single file
pytest -k "not slow"             # skip slow tests

# Visualization
./scripts/tensorboard.sh
```

## Architecture

**Training Pipeline**: `src/train.py` → Hydra config composition → DataModule + LightningModule + Trainer → Training loop → Checkpoint/prediction saving

**Key Components**:
- `src/train.py` / `src/eval.py`: Entry points
- `src/modules/single_module.py`: Main LightningModule handling train/val/test steps
- `src/datamodules/datamodules.py`: DataModule implementations (SingleDataModule)
- `src/modules/models/`: Network architectures (EEGNet, SmallEEGNet, ShallowConvNet)
- `src/modules/losses/components/`: Loss functions (FocalLoss with label smoothing)
- `src/modules/metrics/components/`: Custom metrics (BalancedAccuracy, MeanAveragePrecision)

**Configuration System** (Hydra):
```
configs/
├── train.yaml              # Main config with defaults
├── experiment/             # Experiment-specific overrides
├── module/                 # LightningModule configs
│   └── network/            # Model architecture configs
├── datamodule/             # Data loading configs
├── trainer/                # PyTorch Lightning trainer configs
├── callbacks/              # Training callbacks
└── logger/                 # W&B, TensorBoard, CSV loggers
```

Experiments override defaults via composition:
```yaml
defaults:
  - override /datamodule: eeg.yaml
  - override /module: eeg_single.yaml
  - override /trainer: eeg_default.yaml
```

## Data Flow

1. Parquet files with columns (crop_path, target) define dataset splits
2. DataModule reads parquet → creates Dataset with transforms
3. Supports multiple read modes: npy, h5, raw
4. EEG-specific transforms in `src/datamodules/components/transforms/`

## Config Patterns

Objects are instantiated via `_target_`:
```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 0.001
```

Parameter interpolation: `num_classes: ${num_classes}`

Custom resolvers: `${replace:"__metric__/valid"}` for dynamic metric names

**Important**: Hydra replaces entire sections on override, not merge. Experiment configs must include all needed parameters.

## Code Quality

- Python 3.13+ required
- Strict mypy type checking
- Black formatting (88 chars)
- Pre-commit hooks: black, isort, mypy, flake8, bandit, nbstripout
