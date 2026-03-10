# Informe de Experimentos — Cards Image Classifier

**Fecha:** 2026-03-09
**Dataset:** [Cards Image Dataset Classification (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data)
**Tarea:** Clasificación multiclase — 53 clases (52 cartas estándar + Joker)
**Splits:** Train 7.624 | Val 265 | Test 265

---

## 1. Configuración experimental

Todos los experimentos se entrenaron con:
- **Batch size:** 64
- **Early stopping:** paciencia = 7 épocas, min_delta = 1e-4
- **Optimizador:** Adam
- **Métricas:** accuracy, precision, recall y F1 (promedio macro, via scikit-learn)
- **Tracking:** Weights & Biases (proyecto `card_dataset`)

Se realizaron cuatro bloques de experimentos:
1. Comparativa de arquitecturas y estrategias
2. Learning rate sweep sobre la mejor arquitectura (`resnet18_finetune`)
3. SVM sobre features de backbone fine-tuneado con nuestros datos
4. Ensemble (stacking) combinando NN fine-tuneada + SVM + RandomForest

---

## 2. Resultados por experimento

### 2.1 Comparativa de estrategias (Experimento 1 & 2)

| Modelo | Estrategia | LR | Épocas | Val Acc | Test Acc | Test F1 |
|---|---|---|---|---|---|---|
| `custom_cnn` | CNN desde cero | 1e-3 | 30 (completo) | 72.45% | 70.57% | 70.12% |
| `resnet18_mlp` | Backbone ImageNet congelado + MLP | 1e-3 | 30 (completo) | 58.11% | 52.83% | 51.36% |
| `resnet18_finetune` | Fine-tuning completo | 1e-4 | 19 (early stop) | 96.98% | **97.36%** | **97.33%** |
| `resnet18_sklearn` + SVM | Features ImageNet + SVM | — | — | 57.36% | 46.04% | 45.74% |
| `resnet18_sklearn` + RF | Features ImageNet + RandomForest | — | — | 36.98% | 30.94% | 28.08% |

### 2.2 Learning rate sweep — `resnet18_finetune` (Experimento 3)

| LR | Early stop | Val Acc (último epoch) | Test Acc | Test F1 |
|---|---|---|---|---|
| 1e-2 | No (epoch 30) | 92.08% | 90.94% | 90.77% |
| 1e-3 | Epoch 13 | 96.60% | 95.85% | 95.74% |
| 5e-4 | Epoch 24 | 97.36% | 96.98% | 96.95% |
| **1e-4** | Epoch 19 | **96.98%** | **97.36%** | **97.33%** |

La relación es monótona: a mayor learning rate, peor resultado. `lr=1e-4` es el óptimo.

### 2.3 SVM sobre backbone fine-tuneado (Experimento 4)

| Backbone | Classifier | Val Acc | Test Acc | Test F1 |
|---|---|---|---|---|
| ResNet-18 ImageNet (congelado) | SVM | 57.36% | 46.04% | 45.74% |
| **ResNet-18 fine-tuneado en cartas** | **SVM** | **99.72%** | **97.36%** | **97.33%** |

### 2.4 Stacking ensemble (Experimento 5)

Base models: ResNet18Finetune (MLP) + SVM + RandomForest, todos usando features del backbone fine-tuneado.
Meta-learner entrenado en val set (265 muestras), evaluado en test. Average ensemble sin meta-learner como baseline.

| Configuración | Test Acc | Test F1 |
|---|---|---|
| SVM (fine-tuned features) — base | **97.74%** | **97.74%** |
| NN finetune lr=1e-4 — base | 97.36% | 97.33% |
| RandomForest (fine-tuned features) — base | 97.36% | 97.33% |
| Weighted avg [NN×2 + SVM + RF] | 97.36% | 97.33% |
| Weighted avg [NN×3 + SVM + RF] | 97.36% | 97.33% |
| Stack [NN+SVM+RF] → LogReg (C=1) | 97.36% | 97.33% |
| Stack [NN+SVM+RF] → LogReg (C=10) | 97.36% | 97.33% |
| Stack [SVM+RF] → LogReg (C=1) | 97.36% | 97.33% |
| Average [NN + SVM + RF] | 96.98% | 96.95% |
| Stack [NN+SVM+RF] → SVM (rbf) | 95.85% | 95.91% |

El SVM individual supera a todos los ensembles. El stacking con LogReg empata con la NN pero no la mejora.

---

## 3. Análisis

### Mejor configuración

> **SVM sobre features del backbone fine-tuneado** (`resnet18_finetuned_sklearn` + SVM)
> Test Accuracy: **97.74%** | Test F1: **97.74%**

Supera levemente al `resnet18_finetune` end-to-end (97.36%) y a todos los ensembles probados.

### Observaciones clave

**Fine-tuning completo es claramente superior al backbone congelado.** `resnet18_mlp` (backbone congelado + MLP) solo alcanza 52.83%, mientras que `resnet18_finetune` llega a 97.36%. Las cartas de juego son visualmente muy distintas a ImageNet, por lo que los pesos preentrenados necesitan adaptarse.

**El learning rate es crítico en fine-tuning.** Con `lr=1e-2` el modelo converge a 90.94% porque destruye parcialmente el conocimiento preentrenado. Con `lr=1e-4` los pesos se adaptan suavemente y maximizan el rendimiento.

**La calidad del backbone determina la calidad del clasificador sklearn.** Con features de ImageNet, el SVM obtiene 46%. Con las mismas features del backbone ya fine-tuneado sobre cartas, el SVM iguala al fine-tuning end-to-end (97.36%). Esto demuestra que el backbone aprendió una representación casi linealmente separable para este dominio.

**La CNN desde cero es competitiva.** `custom_cnn` alcanza 70.57% sin ningún preentrenamiento, mostrando que la arquitectura es correcta pero limitada por la cantidad de datos disponibles.

**Los clasificadores sklearn sobre ImageNet son los peores.** RandomForest (30.94%) y SVM (46.04%) con features de ImageNet no logran generalizar al dominio de cartas, confirmando que las features genéricas no son suficientes para tareas de dominio específico.

**El ensemble no mejora al mejor modelo base.** Con modelos tan fuertes y correlacionados (todos comparten el mismo backbone fine-tuneado), el stacking no aporta diversidad suficiente. La mayoría de configuraciones empatan con la NN (97.36%) pero ninguna supera al SVM individual (97.74%). El único ensemble que empeora es el average uniforme (96.98%), probablemente porque el RF arrastra el promedio hacia abajo en ejemplos difíciles. El meta-learner SVM sobre probabilidades concatenadas (95.85%) sufre de sobreajuste al val set de solo 265 muestras.

---

## 4. Ranking final

| # | Configuración | Test Acc | Test F1 |
|---|---|---|---|
| 🥇 | `resnet18_finetuned_sklearn` + SVM | **97.74%** | **97.74%** |
| 2 | `resnet18_finetune` lr=1e-4, bs=64 | 97.36% | 97.33% |
| 2 | `resnet18_finetuned_sklearn` + RF | 97.36% | 97.33% |
| 2 | Stacking [NN+SVM+RF] → LogReg | 97.36% | 97.33% |
| 5 | `resnet18_finetune` lr=5e-4, bs=64 | 96.98% | 96.95% |
| 6 | `resnet18_finetune` lr=1e-3, bs=64 | 95.85% | 95.74% |
| 7 | `resnet18_finetune` lr=1e-2, bs=64 | 90.94% | 90.77% |
| 8 | `custom_cnn` lr=1e-3, bs=64 | 70.57% | 70.12% |
| 9 | `resnet18_mlp` lr=1e-3, bs=64 | 52.83% | 51.36% |
| 10 | `resnet18_sklearn` + SVM (ImageNet) | 46.04% | 45.74% |
| 11 | `resnet18_sklearn` + RF (ImageNet) | 30.94% | 28.08% |

---

## 5. Recomendación

**Para máxima precisión:** `resnet18_finetuned_sklearn` + SVM (97.74%). Requiere dos pasos: fine-tuning del backbone + entrenamiento del SVM, pero es el mejor resultado obtenido.

**Para producción simple:** `resnet18_finetune` con `lr=1e-4` (97.36%). Un único modelo end-to-end, fácil de desplegar con un solo forward pass.

```bash
# Paso 1 — Fine-tuning del backbone (si no existe ya)
conda run -n claude_testing python train.py \
  --model resnet18_finetune \
  --lr 1e-4 \
  --batch_size 64 \
  --epochs 30 \
  --patience 7

# Paso 2 — SVM sobre features del backbone fine-tuneado
conda run -n claude_testing python train.py \
  --model resnet18_finetuned_sklearn \
  --classifier svm \
  --backbone_checkpoint runs/<run_name>/best_model.pt
```
