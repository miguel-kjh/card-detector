# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Playing cards image classifier — 53 classes (52 standard cards + Joker).
Dataset: [Cards Image Dataset Classification (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)

## Environment

Conda environment: `claude_testing` (Python 3.12, PyTorch 2.10 + CUDA 12.8)

Key packages: `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `Pillow`, `tqdm`.

```bash
# Install new deps
conda run -n claude_testing pip install <package>
```

## Running

```bash
# Defaults: custom_cnn, 10 epochs, lr=1e-3, batch_size=32
conda run -n claude_testing python train.py

# PyTorch models
conda run -n claude_testing python train.py --model resnet18_mlp --epochs 20 --lr 5e-4
conda run -n claude_testing python train.py --model resnet18_finetune --epochs 15 --batch_size 64

# Sklearn (feature extraction + classifier, no epochs)
conda run -n claude_testing python train.py --model resnet18_sklearn --classifier svm
conda run -n claude_testing python train.py --model resnet18_sklearn --classifier random_forest

# With W&B tracking
conda run -n claude_testing python train.py --model resnet18_mlp --wandb
```

**All CLI flags:** `--model`, `--classifier`, `--epochs`, `--batch_size`, `--lr`, `--num_workers`, `--patience`, `--min_delta`, `--wandb`, `--data_dir`

## Architecture

### Entry point flow

`train.py` → `TrainConfig.from_args()` → `get_loaders(cfg)` → `Trainer` or `SklearnTrainer` → `fit()` → `evaluate()`

`cfg.is_sklearn()` routes between the two trainer types.

### Model system

All PyTorch models inherit `BaseModel(ABC, nn.Module)` from `src/models/base_model.py` and are registered in `src/models/__init__.py`:

| Registry key | Class | Strategy |
|---|---|---|
| `custom_cnn` | `CustomCNN` | 3×Conv+BN+ReLU+MaxPool, Linear head |
| `resnet18_mlp` | `ResNet18MLP` | Frozen backbone + MLP (512→256→num_classes) |
| `resnet18_finetune` | `ResNet18Finetune` | All layers trainable, Linear(512→num_classes) |
| `resnet18_sklearn` | `ResNet18Features` | Frozen, Identity fc → 512-dim vectors for sklearn |

To add a new model: create file in `src/models/`, inherit `BaseModel`, register in `__init__.py`, add key to `PYTORCH_MODELS` or `SKLEARN_MODELS` in `src/config.py`.

### Trainer split

- **`Trainer`** (`src/trainer.py`): PyTorch gradient training, CrossEntropyLoss, Adam, early stopping, per-epoch tqdm, optional W&B logging
- **`SklearnTrainer`** (`src/sklearn_trainer.py`): extracts 512-dim ResNet features once (no gradient), fits SVC or RandomForestClassifier, serializes with joblib

Both share the same interface: `fit(train_loader, val_loader)` + `evaluate(test_loader) -> dict`.

### Config & run isolation

`TrainConfig` (dataclass, `src/config.py`) auto-generates a timestamped run folder:
- `runs/<run_name>/best_model.pt` — best PyTorch checkpoint (or `sklearn_model.pkl`)
- `runs/<run_name>/results.csv` — per-epoch train/val metrics
- `runs/<run_name>/test_results.json` — final test metrics

Run name format: `{model}_{classifier}_{date}` for sklearn; `{model}_lr{lr}_bs{bs}_ep{ep}_{date}` for PyTorch.

### Data loading

`CardsDataset` reads `data/cards.csv` (columns: `filepaths`, `labels`, `data set`). Images at `data/<split>/<class>/`. Corrupted rows filtered via `os.path.exists`. Train uses RandomHorizontalFlip + RandomRotation(10); val/test use ImageNet normalization only.
