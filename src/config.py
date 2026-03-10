import argparse
import os
import torch
from dataclasses import dataclass, field
from datetime import datetime

PYTORCH_MODELS = {"custom_cnn", "resnet18_mlp", "resnet18_finetune"}
SKLEARN_MODELS  = {"resnet18_sklearn", "resnet18_finetuned_sklearn"}
ALL_MODELS      = sorted(PYTORCH_MODELS | SKLEARN_MODELS)


def _run_name(cfg: "TrainConfig") -> str:
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.model in SKLEARN_MODELS:
        return f"{cfg.model}_{cfg.classifier}_{date}"
    return f"{cfg.model}_lr{cfg.lr}_bs{cfg.batch_size}_ep{cfg.epochs}_{date}"


@dataclass
class TrainConfig:
    # Data
    data_dir: str = "data"
    num_workers: int = 4
    num_classes: int = 53

    # Model
    model: str = "custom_cnn"
    classifier: str = "svm"    # solo para resnet18_sklearn: svm | random_forest
    backbone_checkpoint: str = ""  # checkpoint para resnet18_finetuned_sklearn

    # Hyperparams (ignorados en resnet18_sklearn)
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3

    # Early stopping (ignorado en resnet18_sklearn)
    patience: int = 5
    min_delta: float = 1e-4

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "card_dataset"

    # Runtime (se rellenan automáticamente)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    run_name: str = ""
    run_dir: str = ""
    checkpoint: str = ""
    results_csv: str = ""

    def is_sklearn(self) -> bool:
        return self.model in SKLEARN_MODELS

    def _resolve_paths(self):
        if not self.run_name:
            self.run_name = _run_name(self)
        self.run_dir     = os.path.join("runs", self.run_name)
        self.checkpoint  = os.path.join(self.run_dir, "best_model.pt")
        self.results_csv = os.path.join(self.run_dir, "results.csv")
        os.makedirs(self.run_dir, exist_ok=True)

    @classmethod
    def from_args(cls) -> "TrainConfig":
        parser = argparse.ArgumentParser(description="Cards classifier training")
        parser.add_argument("--data_dir",    type=str,   default="data")
        parser.add_argument("--model",       type=str,   default="custom_cnn",
                            choices=ALL_MODELS, help="Arquitectura a usar")
        parser.add_argument("--classifier",  type=str,   default="svm",
                            choices=["svm", "random_forest"],
                            help="Clasificador sklearn (solo para resnet18_sklearn)")
        parser.add_argument("--backbone_checkpoint", type=str, default="",
                            help="Ruta al checkpoint de ResNet18Finetune (para resnet18_finetuned_sklearn)")
        parser.add_argument("--epochs",      type=int,   default=10)
        parser.add_argument("--batch_size",  type=int,   default=32)
        parser.add_argument("--lr",          type=float, default=1e-3)
        parser.add_argument("--num_workers", type=int,   default=4)
        parser.add_argument("--patience",    type=int,   default=5,
                            help="Epochs sin mejora para early stopping")
        parser.add_argument("--min_delta",   type=float, default=1e-4,
                            help="Mejora mínima que reinicia el contador de paciencia")
        parser.add_argument("--wandb",       action="store_true", default=False,
                            help="Activar logging con Weights & Biases")
        args = parser.parse_args()

        cfg = cls()
        cfg.data_dir    = args.data_dir
        cfg.model       = args.model
        cfg.classifier  = args.classifier
        cfg.epochs      = args.epochs
        cfg.batch_size  = args.batch_size
        cfg.lr          = args.lr
        cfg.num_workers = args.num_workers
        cfg.patience    = args.patience
        cfg.min_delta   = args.min_delta
        cfg.use_wandb              = args.wandb
        cfg.backbone_checkpoint    = args.backbone_checkpoint
        cfg._resolve_paths()
        return cfg

    def __str__(self) -> str:
        lines = ["TrainConfig:"]
        for k, v in self.__dict__.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
