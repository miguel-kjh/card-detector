# Card Detector — Playing Cards Image Classifier

A 53-class image classifier for standard playing cards (52 cards + Joker), built with PyTorch and scikit-learn. Multiple architectures and training strategies were evaluated, from a custom CNN trained from scratch to ResNet-18 fine-tuning combined with SVM classifiers.

**Dataset:** [Cards Image Dataset Classification (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)
| Split | Samples |
|---|---|
| Train | 7,624 |
| Validation | 265 |
| Test | 265 |

---

## Project Structure

```
├── train.py                  # Entry point
├── src/
│   ├── config.py             # TrainConfig dataclass, CLI parsing
│   ├── data_loader.py        # CardsDataset, get_loaders()
│   ├── trainer.py            # PyTorch training loop (early stopping, W&B)
│   ├── sklearn_trainer.py    # Feature extraction + sklearn classifier
│   └── models/
│       ├── base_model.py     # BaseModel (ABC + nn.Module)
│       ├── custom_cnn.py     # CustomCNN
│       └── resnet18.py       # ResNet18MLP, ResNet18Finetune, ResNet18Features
├── ensemble.py               # Stacking ensemble experiments
├── confusion_matrix.py       # Confusion matrix generation
└── data/
    └── cards.csv             # Dataset metadata (filepaths, labels, split)
```

---

## Architectures

### CustomCNN
A 3-block CNN trained from scratch with no pretraining.

```
Input (3×224×224)
→ Conv(3→32) + BN + ReLU + MaxPool  → 32×112×112
→ Conv(32→64) + BN + ReLU + MaxPool → 64×56×56
→ Conv(64→128) + BN + ReLU + MaxPool → 128×28×28
→ Flatten → Linear(100352 → 53)
```

### ResNet18MLP (frozen backbone)
ImageNet-pretrained ResNet-18 with all backbone weights frozen. Only the classification head is trained.

```
ResNet-18 backbone (frozen) → 512-dim features
→ Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→53)
```

### ResNet18Finetune (full fine-tuning)
All layers of the ImageNet-pretrained ResNet-18 are made trainable. Only the final fully-connected layer is replaced.

```
ResNet-18 backbone (trainable) → Linear(512→53)
```

### ResNet18 + sklearn classifiers
The frozen ResNet-18 backbone is used as a feature extractor, producing 512-dimensional vectors. A classical classifier (SVM or Random Forest) is then trained on top.

Two variants were tested:
- **ImageNet backbone** — frozen weights, features not domain-adapted
- **Fine-tuned backbone** — weights adapted to playing cards first, then used for feature extraction

---

## Setup

**Requirements:** Conda with Python 3.12, PyTorch 2.10, CUDA 12.8.

```bash
conda create -n claude_testing python=3.12
conda activate claude_testing
pip install torch torchvision pandas numpy scikit-learn joblib Pillow tqdm
```

Place the dataset under `data/` following the structure from Kaggle (`data/train/`, `data/valid/`, `data/test/`, `data/cards.csv`).

---

## Usage

```bash
# Custom CNN (default)
conda run -n claude_testing python train.py

# ResNet18 frozen backbone + MLP head
conda run -n claude_testing python train.py --model resnet18_mlp --epochs 20 --lr 5e-4

# ResNet18 full fine-tuning (best single model)
conda run -n claude_testing python train.py --model resnet18_finetune --lr 1e-4 --batch_size 64 --epochs 30

# Sklearn classifiers (ImageNet features)
conda run -n claude_testing python train.py --model resnet18_sklearn --classifier svm
conda run -n claude_testing python train.py --model resnet18_sklearn --classifier random_forest

# With Weights & Biases tracking
conda run -n claude_testing python train.py --model resnet18_finetune --wandb
```

**All CLI flags:** `--model`, `--classifier`, `--epochs`, `--batch_size`, `--lr`, `--num_workers`, `--patience`, `--min_delta`, `--wandb`, `--data_dir`

Outputs are saved to `runs/<run_name>/`:
- `best_model.pt` — best checkpoint (PyTorch) or `sklearn_model.pkl`
- `results.csv` — per-epoch train/val metrics
- `test_results.json` — final test metrics

---

## Results

All experiments used batch size 64, Adam optimizer, and early stopping (patience=7, min_delta=1e-4).

### Strategy comparison

| Model | Strategy | LR | Epochs | Test Acc | Test F1 |
|---|---|---|---|---|---|
| `custom_cnn` | Trained from scratch | 1e-3 | 30 | 70.57% | 70.12% |
| `resnet18_mlp` | Frozen backbone + MLP | 1e-3 | 30 | 52.83% | 51.36% |
| `resnet18_sklearn` + SVM | ImageNet features + SVM | — | — | 46.04% | 45.74% |
| `resnet18_sklearn` + RF | ImageNet features + RF | — | — | 30.94% | 28.08% |
| `resnet18_finetune` | Full fine-tuning | 1e-4 | 19* | **97.36%** | **97.33%** |

*Early stopping triggered at epoch 19.

### Learning rate sweep (ResNet18Finetune)

| LR | Early stop | Test Acc | Test F1 |
|---|---|---|---|
| 1e-2 | No | 90.94% | 90.77% |
| 1e-3 | Epoch 13 | 95.85% | 95.74% |
| 5e-4 | Epoch 24 | 96.98% | 96.95% |
| **1e-4** | Epoch 19 | **97.36%** | **97.33%** |

Lower learning rates preserve pretrained knowledge and yield better results. `lr=1e-4` is optimal.

### Fine-tuned backbone + SVM

Extracting features from the fine-tuned (domain-adapted) backbone and training a separate SVM on top:

| Backbone | Classifier | Test Acc | Test F1 |
|---|---|---|---|
| ResNet-18 (ImageNet, frozen) | SVM | 46.04% | 45.74% |
| **ResNet-18 (fine-tuned on cards)** | **SVM** | **97.74%** | **97.74%** |

The domain-adapted backbone produces nearly linearly separable features for this task.

### Stacking ensemble

Base models: ResNet18Finetune + SVM + RandomForest, all using fine-tuned backbone features.

| Configuration | Test Acc | Test F1 |
|---|---|---|
| **SVM (fine-tuned features)** | **97.74%** | **97.74%** |
| NN finetune lr=1e-4 | 97.36% | 97.33% |
| RandomForest (fine-tuned features) | 97.36% | 97.33% |
| Stack [NN+SVM+RF] → LogReg | 97.36% | 97.33% |
| Weighted avg [NN×2 + SVM + RF] | 97.36% | 97.33% |
| Average [NN + SVM + RF] | 96.98% | 96.95% |
| Stack [NN+SVM+RF] → SVM (rbf) | 95.85% | 95.91% |

The ensemble does not outperform the best individual model. All base models share the same backbone, limiting diversity.

---

## Final Ranking

| # | Configuration | Test Acc | Test F1 |
|---|---|---|---|
| 1 | `resnet18_finetuned_sklearn` + SVM | **97.74%** | **97.74%** |
| 2 | `resnet18_finetune` lr=1e-4 | 97.36% | 97.33% |
| 2 | `resnet18_finetuned_sklearn` + RF | 97.36% | 97.33% |
| 2 | Stacking [NN+SVM+RF] → LogReg | 97.36% | 97.33% |
| 5 | `resnet18_finetune` lr=5e-4 | 96.98% | 96.95% |
| 6 | `resnet18_finetune` lr=1e-3 | 95.85% | 95.74% |
| 7 | `resnet18_finetune` lr=1e-2 | 90.94% | 90.77% |
| 8 | `custom_cnn` | 70.57% | 70.12% |
| 9 | `resnet18_mlp` | 52.83% | 51.36% |
| 10 | `resnet18_sklearn` + SVM (ImageNet) | 46.04% | 45.74% |
| 11 | `resnet18_sklearn` + RF (ImageNet) | 30.94% | 28.08% |

---

## Key Findings

- **Full fine-tuning dramatically outperforms a frozen backbone.** `resnet18_mlp` (frozen) reaches only 52.83%, while `resnet18_finetune` reaches 97.36%. Playing cards are visually very different from ImageNet, so the pretrained weights need to adapt.
- **Learning rate is critical for fine-tuning.** High LRs destroy pretrained knowledge; `lr=1e-4` allows gradual, effective adaptation.
- **Backbone quality determines sklearn classifier quality.** SVM on ImageNet features achieves 46%. The same SVM on fine-tuned features achieves 97.74% — the domain-adapted backbone learned a near-linearly separable representation.
- **The custom CNN is competitive given zero pretraining** (70.57%), but data limitations prevent it from matching transfer learning approaches.
- **Ensembles do not help when base models are strongly correlated.** All models share the same backbone, so stacking adds no meaningful diversity.

---

## Recommended Configurations

**Best accuracy:** Fine-tuned backbone + SVM (97.74%)
```bash
# Step 1 — Fine-tune the backbone
conda run -n claude_testing python train.py \
  --model resnet18_finetune --lr 1e-4 --batch_size 64 --epochs 30 --patience 7

# Step 2 — Train SVM on fine-tuned features
conda run -n claude_testing python train.py \
  --model resnet18_finetuned_sklearn --classifier svm \
  --backbone_checkpoint runs/<run_name>/best_model.pt
```

**Simplest deployment:** End-to-end fine-tuned ResNet-18 (97.36%)
```bash
conda run -n claude_testing python train.py \
  --model resnet18_finetune --lr 1e-4 --batch_size 64 --epochs 30
```
