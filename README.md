# EEG Classification Template

This repository provides a robust and reproducible framework for EEG (or other ML) classification projects, leveraging the best practices and tooling from the modern Python ML ecosystem.

## Overview

An efficient workflow and results reproducibility are critical for machine learning projects:

- Quickly experiment with new models and compare various approaches.
- Build confidence and transparency into your training pipelines.
- Save developer and compute time.

This repository is based on [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and [Hydra](https://github.com/facebookresearch/hydra), providing a highly modular and extensible foundation for deep learning prototyping across CPUs, multi-GPUs, and TPUs. The template is inspired by and extends the excellent [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template), with further polish, features, and improved reproducibility.

### Features

- Easily configurable experiments (Classification, Segmentation, Metric Learning, etc.).
- Clear project structure and scalable code organization.
- Best-practice integrations: logging, versioning, checkpoints, Docker, CI, and more.
- Extendable for custom tasks and research pipelines.

## Quick Start

```shell
# Clone the repo
git clone https://github.com/gorodnitskiy/yet-another-lightning-hydra-template
cd yet-another-lightning-hydra-template

# Install Python requirements
pip install -r requirements.txt
```

Or use Docker for reproducible environments. See details in [Docker](#docker).

## Table of Contents

- [Main technologies](#main-technologies)
- [Project structure](#project-structure)
- [Workflow](#workflow---how-it-works)
- [Hydra configs](#hydra-configs)
- [Logs](#logs)
- [Data](#data)
- [Notebooks](#notebooks)
- [Hyperparameters search](#hyperparameters-search)
- [Docker](#docker)
- [Tests](#tests)
- [Continuous integration](#continuous-integration)

---

## Main technologies

- **[PyTorch Lightning](https://github.com/Lightning-AI/lightning)**: Modern PyTorch framework for reproducible deep learning research with maximum flexibility and scalability.
- **[Hydra](https://github.com/facebookresearch/hydra)**: Flexible, hierarchical, and composable configuration management.

## Project Structure

Typical structure is:

```
├── configs/        # Hydra config files (callbacks, datamodules, experiments, etc.)
├── data/           # Project datasets
├── logs/           # Hydra, Lightning, and logger outputs
├── notebooks/      # Jupyter notebooks
├── scripts/        # Shell or utility scripts
├── src/            # Source code
│   ├── callbacks/
│   ├── datamodules/
│   ├── modules/
│   ├── utils/
│   ├── eval.py
│   └── train.py
├── tests/          # Pytest-based unit tests
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Workflow - How It Works

Key practices for reproducibility in ML projects:

- Pin Python package versions (requirements.txt).
- Use version control for both code and data (e.g., DVC, Git).
- Track experiments (e.g., Weights & Biases, TensorBoard, MLflow, Neptune, DVC, CSV logs).
- Run in reproducible Docker environments.

### Typical Workflow

1. Implement a custom Lightning `DataModule` and `LightningModule`. See [`src/datamodules/datamodules.py`](src/datamodules/datamodules.py) and [`src/modules/single_module.py`](src/modules/single_module.py).
2. Create your experiment config files under `configs/`.
3. Launch training runs (with optional config overrides):

    ```shell
    python src/train.py experiment=your_experiment.yaml
    ```

    Hyperparameter search via Optuna/other sweepers:

    ```shell
    python src/train.py -m hparams_search=mnist_optuna
    ```

    Or override configs from the CLI:

    ```shell
    python src/train.py -m logger=csv module.optimizer.weight_decay=0.,0.00001,0.0001
    ```

4. Evaluate results and run predictions with different checkpoints or on new data.

Included: Example configs and code for MNIST; adapt to your EEG data/task as needed.

---

## LightningDataModule

- Build standard PyTorch Datasets with `__getitem__` and `__len__`.
- Optionally leverage or adapt provided Dataset classes in [`src/datamodules/datasets.py`](src/datamodules/datasets.py).
- Use [Lightning DataModule API](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#lightningdatamodule-api) for data loading and preparation:
  - `prepare_data`, `setup`, `train_dataloader`, `val_dataloader`, `test_dataloader`, `predict_dataloader`
- See example DataModule classes, config patterns, and notes on handling prediction datasets and output formats.

---

## LightningModule

Implement your training logic with LightningModule:

- Key methods: `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, `configure_optimizers`
- Flexible metrics/losses setup: specify in config with `_target_`, supports any torchmetrics, custom metrics/losses (`src/modules/metrics/components/`, `src/modules/losses/components/`)
- Example model configs, metrics, and losses for standard and more advanced tasks (self-supervised, metric learning, etc.)
- Modular, composable, and easily extendable.

---

## Training & Evaluation

- Modular [training script](src/train.py): instantiates DataModule, Module, Callbacks, Loggers, Trainer, handles logging/metadata/ckpts.
- [Evaluation script](src/eval.py): supports validation/inference across datasets, batch predictions, flexible outputs.
- Modern [Prediction API](src/utils/saving_utils.py) with hooks for custom output formats, multiple datasets, and memory-efficient saving.

---

## Callbacks & Extensions

- Out-of-the-box support for Lightning callbacks: checkpoints, early stopping, progress bars, summaries, etc. (see `configs/callbacks/`)
- Optional: Custom `LightProgressBar` for enhanced progress visualization.
- Integrated DDP/Plugin support for scalable multi-GPU/multi-node training.
- [GradCAM](https://github.com/jacobgil/pytorch-grad-cam) integration for model explainability and debugging.

---

## Hydra Configs

Combines [Hydra](https://github.com/facebookresearch/hydra) and [OmegaConf](https://omegaconf.readthedocs.io/) for flexible, hierarchical configuration:

- Run experiments via `@hydra.main` or Hydra's Compose API.
- Create/instantiate objects from configs using `_target_` and Hydra utils.
- Override config values from the command line as needed.
- Support for structured configs, sweepers (Optuna, Nevergrad, Ax), and custom plugins.
- Built-in custom resolvers for dynamic config expressions, e.g., `${replace:"__metric__/valid"}`.

### Config Override Best Practices

When overriding nested config sections in experiment configs, be aware that Hydra **replaces** entire sections rather than merging them. To avoid conflicts:

1. **Default configs** should avoid including parameters that are specific to certain metric/class types (e.g., `top_k` for Accuracy) if the config might be overridden with different types.

2. **Experiment configs** should explicitly include all necessary parameters when overriding sections. For example:
   ```yaml
   metrics:
     main:
       _target_: "torchmetrics.AUROC"
       task: "multiclass"
       num_classes: ${num_classes}
       average: "macro"
       # Don't include top_k here - AUROC doesn't use it
   ```

3. If using Accuracy, explicitly include `top_k: 1`:
   ```yaml
   metrics:
     main:
       _target_: "torchmetrics.Accuracy"
       task: "multiclass"
       num_classes: ${num_classes}
       top_k: 1  # Required for Accuracy
   ```

---

## Logs

- Logs and metadata (pip freeze, git status, hardware info, source/config copies) are saved per run.
- Structured logging in timestamped directories (`logs/task_name/runs/<datetime>/`).
- Easily modifiable via [Hydra config](configs/hydra/default.yaml) and [path configs](configs/paths/default.yaml).

---

## Data

Option to store datasets in [HDF5 format](https://docs.h5py.org/en/stable/) for efficient storage/access:

- Convenient utilities for creating/reading HDF5 datasets (`src/datamodules/components/h5_file.py`).
- Example code for dataset access in Datasets and throughout training pipeline.

---

## Notebooks

- Encouraged best practices for Jupyter notebook organization and naming.
- Suggest sections: Summary, Config, Libs, Analysis.
- Use versioned, descriptive filenames, e.g., `1.0-initials-data-exploration.ipynb`.

---

## Hyperparameters Search

- Plug-and-play hyperparameter optimization via Hydra's Optuna/Nevergrad/Ax sweeper plugins.
- Organize search configs under `configs/hparams_search/`.
- Launch with:

    ```shell
    python src/train.py -m hparams_search=mnist_optuna
    ```

- Results will appear in the logs/multirun directory.

---

## Docker

- Fully reproducible Docker builds. See provided `Dockerfile` and `.dockerignore`.
- Extrememely useful when deploying locally, on clusters, or in the cloud.
- NVidia GPU support, Miniconda setup, and instructions for fine-grained resource control (CPUs, GPUs).
- Example usage for building and running with volume and hardware mapping provided.

---

## Tests

- Extensive [pytest](https://docs.pytest.org/) based unittests in `tests/`.
- Covers: config instantiation, DataModules, models, losses, metrics, functional pipeline runs, DDP simulation, evaluation, sweeper integration, progress bars, and utility functions.
- Default: Fast runs on MNIST; easy to adapt for your custom data.

Example:

```shell
pytest                 # all tests
pytest tests/test_train.py
pytest tests/test_train.py::test_train_ddp_sim
pytest -k "not slow"   # skip slow tests
```

---

## Continuous Integration

- GitHub Actions workflows included for:
    - Full test suite on Linux, macOS, Windows.
    - Pre-commit checks for code quality (on main branch and PRs).
- Easily extensible or portable to GitLab CI (see provided `.gitlab-ci.yml` and docs).
- Enable GitHub Actions in your repo settings for CI.

---

For more information, check the in-code documentation and configs. Contributions and improvements welcome!
