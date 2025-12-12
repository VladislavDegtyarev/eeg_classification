# EEG Classification Project Structure

## Project Overview
This is a machine learning project for EEG signal classification using PyTorch and Lightning.

## Directory Structure

```
eeg_classification/
├── .dockerignore
├── .git/
├── .gitattributes
├── .github/
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── Makefile
├── README.md
├── README_CIFAR10.md
├── Test.ipynb
├── configs/
├── data/
├── notebooks/
├── outputs/
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── scripts/
├── setup.py
├── src/
│   ├── __init__.py
│   ├── callbacks/
│   ├── datamodules/
│   ├── eval.py
│   ├── modules/
│   ├── train.py
│   └── utils/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── helpers/
├── train_cifar10.py
└── ...
```

## Key Directories and Files

### `src/` - Source Code
- `__init__.py` - Package initialization
- `callbacks/` - Custom training callbacks
- `datamodules/` - Data loading and preprocessing modules
- `eval.py` - Evaluation scripts
- `modules/` - Model components and architectures
- `train.py` - Main training script
- `utils/` - Utility functions

### `configs/` - Configuration Files
- YAML configuration files for different training setups

### `data/` - Data Storage
- Dataset files and data processing scripts

### `notebooks/` - Jupyter Notebooks
- Experimental notebooks and exploratory analysis

### `scripts/` - Execution Scripts
- Shell scripts for training, evaluation, and monitoring

### `tests/` - Testing Framework
- Unit tests and integration tests
- `conftest.py` - Pytest configuration
- `helpers/` - Test helper functions

### `outputs/` - Training Outputs
- Model checkpoints and training logs
- Hydra configuration outputs

## Development Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install project in development mode:
   ```bash
   pip install -e .
   ```

3. Run training:
   ```bash
   python train_cifar10.py
   ```

## Technologies Used
- Python 3.x
- PyTorch
- PyTorch Lightning
- Hydra (configuration management)
- Jupyter Notebooks
- Docker
- Makefile (automation)
```
