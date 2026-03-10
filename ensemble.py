"""
Stacking ensemble: ResNet18Finetune (MLP) + SVM + RandomForest
sobre features del backbone fine-tuneado con nuestros datos.

Pipeline:
  1. Extrae features (512-dim) con el backbone fine-tuneado.
  2. Entrena SVM y RF sobre features de train.
  3. Obtiene probabilidades de clase de los 3 modelos base en val y test.
  4. Entrena meta-learners en val (sin data leakage) y evalúa en test.
  5. Compara vs. average ensemble y modelos individuales.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from tqdm import tqdm

from src.config import TrainConfig
from src.data_loader import get_loaders
from src.models.resnet18 import ResNet18FinetunedFeatures, ResNet18Finetune

CHECKPOINT = "runs/resnet18_finetune_lr0.0001_bs64_ep30_20260309_165454/best_model.pt"
OUT_DIR    = "runs/ensemble_stacking"
NUM_CLASSES = 53


def _get_cfg():
    cfg = TrainConfig()
    cfg.batch_size  = 64
    cfg.num_workers = 4
    # Evitar side-effects de _resolve_paths; solo necesitamos data_dir
    return cfg


def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False, dynamic_ncols=True):
            feats.append(model(imgs.to(device)).cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def get_nn_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False, dynamic_ncols=True):
            logits = model(imgs.to(device))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def metrics(y_true, y_pred) -> dict:
    return {
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec":  recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def print_row(name, m):
    print(f"  {name:<40} Acc={m['acc']:.4f}  F1={m['f1']:.4f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = _get_cfg()
    train_loader, val_loader, test_loader, splits = get_loaders(cfg)
    print(f"Train: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}\n")

    # ------------------------------------------------------------------ #
    # 1. Extracción de features (backbone sin cabeza)                     #
    # ------------------------------------------------------------------ #
    print("Extrayendo features con backbone fine-tuneado...")
    feat_model = ResNet18FinetunedFeatures(CHECKPOINT, NUM_CLASSES).to(device)
    X_train, y_train = extract_features(feat_model, train_loader, device)
    X_val,   y_val   = extract_features(feat_model, val_loader,   device)
    X_test,  y_test  = extract_features(feat_model, test_loader,  device)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}\n")

    # ------------------------------------------------------------------ #
    # 2. Probabilidades del modelo neuronal (ResNet18Finetune completo)   #
    # ------------------------------------------------------------------ #
    print("Obteniendo probabilidades del modelo neuronal...")
    nn_model = ResNet18Finetune(NUM_CLASSES).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    nn_model.load_state_dict(state)

    nn_p_train, _ = get_nn_probs(nn_model, train_loader, device)
    nn_p_val,   _ = get_nn_probs(nn_model, val_loader,   device)
    nn_p_test,  _ = get_nn_probs(nn_model, test_loader,  device)
    del nn_model  # liberar VRAM

    # ------------------------------------------------------------------ #
    # 3. Entrenar SVM y RF sobre features de train                        #
    # ------------------------------------------------------------------ #
    print("Entrenando SVM (base)...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train, y_train)
    svm_p_val  = svm.predict_proba(X_val)
    svm_p_test = svm.predict_proba(X_test)

    print("Entrenando RandomForest (base)...")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    rf_p_val  = rf.predict_proba(X_val)
    rf_p_test = rf.predict_proba(X_test)

    # ------------------------------------------------------------------ #
    # 4. Resultados individuales                                          #
    # ------------------------------------------------------------------ #
    results = {}
    print("\n── Modelos base (test) ───────────────────────────────────────────")
    for name, p_test in [("NN finetune (lr=1e-4)", nn_p_test),
                         ("SVM (fine-tuned features)", svm_p_test),
                         ("RandomForest (fine-tuned features)", rf_p_test)]:
        m = metrics(y_test, p_test.argmax(axis=1))
        results[name] = m
        print_row(name, m)

    # ------------------------------------------------------------------ #
    # 5. Average ensemble                                                 #
    # ------------------------------------------------------------------ #
    print("\n── Average ensemble ──────────────────────────────────────────────")
    for name, weights in [
        ("Average [NN + SVM + RF]",          [1, 1, 1]),
        ("Weighted avg [NN×2 + SVM + RF]",   [2, 1, 1]),
        ("Weighted avg [NN×3 + SVM + RF]",   [3, 1, 1]),
    ]:
        w = np.array(weights) / sum(weights)
        avg_p = w[0]*nn_p_test + w[1]*svm_p_test + w[2]*rf_p_test
        m = metrics(y_test, avg_p.argmax(axis=1))
        results[name] = m
        print_row(name, m)

    # ------------------------------------------------------------------ #
    # 6. Stacking (meta-learner entrenado en val, inferencia en test)     #
    # ------------------------------------------------------------------ #
    print("\n── Stacking (val → meta-learner → test) ──────────────────────────")

    meta_val_all  = np.concatenate([nn_p_val,  svm_p_val,  rf_p_val],  axis=1)
    meta_test_all = np.concatenate([nn_p_test, svm_p_test, rf_p_test], axis=1)

    meta_val_no_nn  = np.concatenate([svm_p_val,  rf_p_val],  axis=1)
    meta_test_no_nn = np.concatenate([svm_p_test, rf_p_test], axis=1)

    stacking_configs = [
        # (nombre, meta-features val, meta-features test, meta-learner)
        ("Stack [NN+SVM+RF] → LogReg (C=1)",
         meta_val_all,    meta_test_all,
         LogisticRegression(max_iter=2000, C=1, random_state=42)),

        ("Stack [NN+SVM+RF] → LogReg (C=10)",
         meta_val_all,    meta_test_all,
         LogisticRegression(max_iter=2000, C=10, random_state=42)),

        ("Stack [NN+SVM+RF] → SVM (rbf)",
         meta_val_all,    meta_test_all,
         SVC(kernel="rbf", C=10, gamma="scale", probability=False)),

        ("Stack [SVM+RF] → LogReg (C=1)",
         meta_val_no_nn,  meta_test_no_nn,
         LogisticRegression(max_iter=2000, C=1, random_state=42)),
    ]

    for name, meta_val, meta_test, meta_clf in stacking_configs:
        meta_clf.fit(meta_val, y_val)
        preds = meta_clf.predict(meta_test)
        m = metrics(y_test, preds)
        results[name] = m
        print_row(name, m)

    # ------------------------------------------------------------------ #
    # 7. Guardar resultados                                               #
    # ------------------------------------------------------------------ #
    payload = {k: {mk: round(mv, 6) for mk, mv in v.items()} for k, v in results.items()}
    out_path = os.path.join(OUT_DIR, "ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResultados guardados en: {out_path}")

    # Ranking final
    print("\n── Ranking (Test Accuracy) ───────────────────────────────────────")
    for i, (name, m) in enumerate(sorted(results.items(), key=lambda x: -x[1]["acc"]), 1):
        print(f"  {i:>2}. {name:<40} Acc={m['acc']:.4f}  F1={m['f1']:.4f}")


if __name__ == "__main__":
    main()
