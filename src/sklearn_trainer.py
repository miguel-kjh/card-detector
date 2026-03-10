import csv
import json
import os
import joblib
import numpy as np
import torch
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import TrainConfig

_CLASSIFIERS = {
    "svm":           lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}


class SklearnTrainer:
    """Entrena un clasificador sklearn sobre features extraídas con ResNet-18 congelado."""

    def __init__(self, feature_extractor, cfg: TrainConfig):
        self.extractor = feature_extractor.to(cfg.device)
        self.cfg = cfg
        self.clf = self._build_classifier()
        self._results = []

    def _build_classifier(self):
        if self.cfg.classifier not in _CLASSIFIERS:
            available = list(_CLASSIFIERS.keys())
            raise ValueError(f"Clasificador '{self.cfg.classifier}' no válido. Disponibles: {available}")
        return _CLASSIFIERS[self.cfg.classifier]()

    def _extract_features(self, loader, desc: str = "") -> tuple[np.ndarray, np.ndarray]:
        self.extractor.eval()
        all_feats, all_labels = [], []
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for imgs, labels in pbar:
                feats = self.extractor(imgs.to(self.cfg.device))
                all_feats.append(feats.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.concatenate(all_feats), np.concatenate(all_labels)

    def _compute_metrics(self, y_true, y_pred) -> dict:
        return {
            "acc":  accuracy_score(y_true, y_pred),
            "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "rec":  recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1":   f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def _save_csv(self):
        if not self._results:
            return
        keys = self._results[0].keys()
        with open(self.cfg.results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._results)

    def fit(self, train_loader, val_loader) -> None:
        print(f"Extrayendo features del conjunto de entrenamiento...")
        X_train, y_train = self._extract_features(train_loader, desc="Features [Train]")

        print(f"Entrenando {self.cfg.classifier} con {len(X_train)} muestras de {X_train.shape[1]} features...")
        self.clf.fit(X_train, y_train)

        # Guardar el clasificador sklearn
        clf_path = os.path.join(self.cfg.run_dir, "sklearn_model.pkl")
        joblib.dump(self.clf, clf_path)

        # Evaluar en train y val
        tr_metrics = self._compute_metrics(y_train, self.clf.predict(X_train))

        print("Extrayendo features del conjunto de validación...")
        X_val, y_val = self._extract_features(val_loader, desc="Features [Val]")
        vl_metrics = self._compute_metrics(y_val, self.clf.predict(X_val))

        print(f"\n{'':5}  {'── Train ──────────────────':28}  {'── Val ──────────────────':28}")
        print(f"{'':5}  {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  |  {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(
            f"{'final':>5}  {tr_metrics['acc']:>7.4f} {tr_metrics['prec']:>7.4f} "
            f"{tr_metrics['rec']:>7.4f} {tr_metrics['f1']:>7.4f}  |  "
            f"{vl_metrics['acc']:>7.4f} {vl_metrics['prec']:>7.4f} "
            f"{vl_metrics['rec']:>7.4f} {vl_metrics['f1']:>7.4f}"
        )

        row = {}
        for k, v in tr_metrics.items():
            row[f"train_{k}"] = round(v, 6)
        for k, v in vl_metrics.items():
            row[f"val_{k}"] = round(v, 6)
        self._results.append(row)
        self._save_csv()

        print(f"\nModelo guardado en:        {clf_path}")
        print(f"Historial guardado en:     {self.cfg.results_csv}")

    def _save_test_json(self, metrics: dict):
        path = os.path.join(self.cfg.run_dir, "test_results.json")
        payload = {k: round(v, 6) for k, v in metrics.items()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Test results guardados en: {path}")

    def evaluate(self, test_loader) -> dict:
        print("Extrayendo features del conjunto de test...")
        X_test, y_test = self._extract_features(test_loader, desc="Features [Test]")
        metrics = self._compute_metrics(y_test, self.clf.predict(X_test))
        self._save_test_json(metrics)
        return metrics
