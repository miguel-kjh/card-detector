import csv
import json
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import TrainConfig


class Trainer:
    def __init__(self, model, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self._wandb = None
        self._history = []   # lista de dicts por epoch para el CSV

        if cfg.use_wandb:
            self._wandb = self._init_wandb()

    def _init_wandb(self):
        import wandb
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.run_name,
            config={
                "model":      self.cfg.model,
                "epochs":     self.cfg.epochs,
                "batch_size": self.cfg.batch_size,
                "lr":         self.cfg.lr,
                "device":     self.cfg.device,
                "num_classes":self.cfg.num_classes,
            },
        )
        return wandb

    def _compute_metrics(self, all_labels, all_preds) -> dict:
        return {
            "acc":  accuracy_score(all_labels, all_preds),
            "prec": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "rec":  recall_score(all_labels, all_preds, average="macro", zero_division=0),
            "f1":   f1_score(all_labels, all_preds, average="macro", zero_division=0),
        }

    def run_epoch(self, loader, training: bool, desc: str = "") -> dict:
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        with torch.set_grad_enabled(training):
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.cfg.device), labels.to(self.cfg.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = self._compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(all_labels)
        return metrics

    def _save_csv(self):
        if not self._history:
            return
        keys = self._history[0].keys()
        with open(self.cfg.results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._history)

    def fit(self, train_loader, val_loader) -> None:
        cfg = self.cfg
        header = (f"{'Epoch':>5} {'Loss':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
                  f"  |  {'Loss':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print(f"{'':5}  {'── Train ──────────────────────────':36}  {'── Val ──────────────────────────':36}")
        print(header)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, cfg.epochs + 1):
            tr = self.run_epoch(train_loader, training=True,  desc=f"Epoch {epoch}/{cfg.epochs} [Train]")
            vl = self.run_epoch(val_loader,   training=False, desc=f"Epoch {epoch}/{cfg.epochs} [Val]  ")

            improved = vl["acc"] > best_val_acc + cfg.min_delta
            es_tag = "" if improved else f"  (no mejora {patience_counter + 1}/{cfg.patience})"
            print(
                f"{epoch:>5} {tr['loss']:>8.4f} {tr['acc']:>7.4f} {tr['prec']:>7.4f} {tr['rec']:>7.4f} {tr['f1']:>7.4f}"
                f"  |  {vl['loss']:>8.4f} {vl['acc']:>7.4f} {vl['prec']:>7.4f} {vl['rec']:>7.4f} {vl['f1']:>7.4f}"
                + es_tag
            )

            row = {"epoch": epoch}
            for k, v in tr.items():
                row[f"train_{k}"] = round(v, 6)
            for k, v in vl.items():
                row[f"val_{k}"] = round(v, 6)
            self._history.append(row)

            if self._wandb:
                self._wandb.log({
                    "train/loss": tr["loss"], "train/acc": tr["acc"],
                    "train/prec": tr["prec"], "train/rec": tr["rec"], "train/f1": tr["f1"],
                    "val/loss":   vl["loss"], "val/acc":   vl["acc"],
                    "val/prec":   vl["prec"], "val/rec":   vl["rec"], "val/f1":   vl["f1"],
                    "epoch": epoch,
                })

            if improved:
                best_val_acc = vl["acc"]
                patience_counter = 0
                torch.save(self.model.state_dict(), cfg.checkpoint)
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    print(f"\nEarly stopping en epoch {epoch} (paciencia={cfg.patience})")
                    break

        self._save_csv()
        print(f"\nCheckpoint guardado en: {cfg.checkpoint}")
        print(f"Historial guardado en:  {cfg.results_csv}")

    def _save_test_json(self, metrics: dict):
        path = os.path.join(self.cfg.run_dir, "test_results.json")
        payload = {k: round(v, 6) for k, v in metrics.items()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Test results guardados en: {path}")

    def evaluate(self, test_loader) -> dict:
        self.model.load_state_dict(torch.load(self.cfg.checkpoint))
        metrics = self.run_epoch(test_loader, training=False, desc="Test")

        self._save_test_json(metrics)

        if self._wandb:
            self._wandb.log({f"test/{k}": v for k, v in metrics.items()})
            self._wandb.finish()

        return metrics
