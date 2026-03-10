#!/usr/bin/env python3
"""
Genera la matriz de confusión del mejor modelo (por test accuracy).

Uso:
    conda run -n claude_testing python confusion_matrix.py
    conda run -n claude_testing python confusion_matrix.py --run resnet18_finetune_lr0.0001_bs64_ep30_20260309_165454
    conda run -n claude_testing python confusion_matrix.py --data_dir data --runs_dir runs
"""
import argparse
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PYTORCH_MODELS, SKLEARN_MODELS
from src.data_loader import CardsDataset, get_transforms
from src.models import get_model, get_feature_extractor


def find_best_run(runs_dir: str) -> tuple[str, float]:
    """Retorna el run con mayor test accuracy. En empate prefiere modelos PyTorch."""
    best_acc, best_run = -1.0, None
    for run_name in os.listdir(runs_dir):
        result_path = os.path.join(runs_dir, run_name, "test_results.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            data = json.load(f)
        acc = data.get("acc", -1)
        is_pytorch = infer_model_name(run_name) in PYTORCH_MODELS if any(
            run_name.startswith(m) for m in PYTORCH_MODELS | SKLEARN_MODELS) else False
        current_is_pytorch = (best_run is not None and infer_model_name(best_run) in PYTORCH_MODELS) if best_run else False
        if acc > best_acc or (acc == best_acc and is_pytorch and not current_is_pytorch):
            best_acc = acc
            best_run = run_name
    return best_run, best_acc


def infer_model_name(run_name: str) -> str:
    all_models = sorted(PYTORCH_MODELS | SKLEARN_MODELS, key=len, reverse=True)
    for model in all_models:
        if run_name.startswith(model):
            return model
    raise ValueError(f"No se pudo inferir el model name de: {run_name}")


def build_test_loader(data_dir: str) -> tuple[DataLoader, list[str]]:
    csv_path = os.path.join(data_dir, "cards.csv")
    df = pd.read_csv(csv_path)
    df = df[df["filepaths"].apply(lambda p: os.path.exists(os.path.join(data_dir, p)))]
    test_df = df[df["data set"] == "test"]
    _, val_tf = get_transforms()
    loader = DataLoader(
        CardsDataset(test_df, data_dir, val_tf),
        batch_size=64, shuffle=False, num_workers=4,
    )
    class_map = df[["class index", "labels"]].drop_duplicates().sort_values("class index")
    class_names = class_map["labels"].tolist()
    return loader, class_names


def predict_pytorch(run_dir: str, model_name: str, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, num_classes=53).to(device)
    checkpoint = os.path.join(run_dir, "best_model.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Inferencia", dynamic_ncols=True):
            preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def predict_sklearn(
    run_dir: str, model_name: str, loader: DataLoader, backbone_checkpoint: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "resnet18_finetuned_sklearn" and not backbone_checkpoint:
        raise ValueError(
            "resnet18_finetuned_sklearn requiere --backbone_checkpoint "
            "<ruta/a/best_model.pt del run resnet18_finetune>"
        )
    extractor = get_feature_extractor(model_name, backbone_checkpoint=backbone_checkpoint).to(device)
    extractor.eval()

    clf = joblib.load(os.path.join(run_dir, "sklearn_model.pkl"))

    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extrayendo features", dynamic_ncols=True):
            feats = extractor(imgs.to(device)).cpu().numpy()
            all_feats.append(feats)
            all_labels.extend(labels.numpy())
    X = np.vstack(all_feats)
    preds = clf.predict(X)
    return np.array(all_labels), preds


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: list[str], output_path: str):
    cm = confusion_matrix(labels, preds)
    acc = np.diag(cm).sum() / cm.sum()

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ticks = range(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(f"Matriz de Confusión  (acc = {acc:.4f})", fontsize=14, pad=15)

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=5,
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Genera la matriz de confusión del mejor modelo")
    parser.add_argument("--run",                 type=str, default="", help="Nombre del run (por defecto: el mejor)")
    parser.add_argument("--runs_dir",            type=str, default="runs")
    parser.add_argument("--data_dir",            type=str, default="data")
    parser.add_argument("--backbone_checkpoint", type=str, default="",
                        help="Checkpoint backbone para resnet18_finetuned_sklearn")
    args = parser.parse_args()

    if args.run:
        run_name = args.run
        result_path = os.path.join(args.runs_dir, run_name, "test_results.json")
        with open(result_path) as f:
            acc = json.load(f).get("acc", "?")
    else:
        run_name, acc = find_best_run(args.runs_dir)
        if run_name is None:
            print("No se encontraron runs con test_results.json")
            sys.exit(1)

    run_dir = os.path.join(args.runs_dir, run_name)
    print(f"Run:    {run_name}")
    print(f"Acc:    {acc:.6f}")

    model_name = infer_model_name(run_name)
    print(f"Modelo: {model_name}")

    loader, class_names = build_test_loader(args.data_dir)

    if model_name in PYTORCH_MODELS:
        labels, preds = predict_pytorch(run_dir, model_name, loader)
    else:
        labels, preds = predict_sklearn(run_dir, model_name, loader, args.backbone_checkpoint)

    output_path = os.path.join(run_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, class_names, output_path)


if __name__ == "__main__":
    main()
