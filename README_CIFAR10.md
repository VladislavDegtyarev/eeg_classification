# CIFAR-10 Classification with PyTorch Lightning

## 🚀 Quick Start

### Simple Training (Recommended)
```bash
python train_cifar10.py
```

### Manual Training
```bash
poetry run python src/train.py --config-name=cifar10_train
```

## 📊 Optimized Configuration

The configuration has been optimized for best performance on CIFAR-10:

### Model Architecture
- **Enhanced CIFARNet**: Deeper CNN with 512 feature maps
- **Global Average Pooling**: Better generalization
- **Progressive Dropout**: 0.1 → 0.4 across layers
- **Batch Normalization**: After each conv layer

### Training Settings
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 256 (optimized for GPU)
- **Learning Rate**: 0.01 with cosine annealing
- **Optimizer**: Adam with weight decay 5e-4
- **Mixed Precision**: 16-bit for speed

### Data Augmentation
- **Random Crop**: 32x32 with padding
- **Horizontal Flip**: 50% probability
- **Color Jitter**: Brightness, contrast, saturation, hue
- **Cutout**: Random erasing for regularization

### Expected Results
- **Validation Accuracy**: 85-90%+
- **Training Time**: ~10-15 minutes on RTX 5090
- **GPU Memory**: ~2-3 GB

## 📁 Output Structure

```
logs/train/runs/YYYY-MM-DD_HH-MM-SS/
├── checkpoints/
│   ├── epoch001-val_acc0.8500.ckpt
│   ├── epoch002-val_acc0.8700.ckpt
│   └── last.ckpt
├── csv/
│   └── version_0/
│       ├── hparams.yaml
│       └── metrics.csv
└── metadata/
    └── ...
```

## 🔧 Configuration Files

- **Main Config**: `configs/cifar10_train.yaml`
- **Model Config**: `configs/module/network/cifar10.yaml`
- **Data Config**: `configs/datamodule/cifar10.yaml`

## 📈 Monitoring

Training progress is logged to CSV files. Check:
- `logs/train/runs/*/csv/version_0/metrics.csv` for metrics
- `logs/train/runs/*/checkpoints/` for model checkpoints

## 🎯 Key Improvements

1. **Better Architecture**: Deeper network with more parameters
2. **Advanced Augmentation**: Color jitter + cutout for robustness
3. **Optimized Training**: Higher LR, cosine annealing, gradient clipping
4. **Better Regularization**: Progressive dropout, weight decay
5. **Simplified Usage**: One-command training

## 🚨 Troubleshooting

If training fails:
1. Check GPU availability: `nvidia-smi`
2. Verify data download: `ls data/cifar-10-batches-py/`
3. Check dependencies: `poetry install`
4. Run tests: `poetry run pytest tests/`

## 📊 Performance Tips

- Use larger batch size if you have more GPU memory
- Increase epochs for better convergence
- Adjust learning rate if training is unstable
- Monitor validation loss for overfitting
