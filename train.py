from src.config import TrainConfig
from src.data_loader import get_loaders
from src.models import get_model, get_feature_extractor
from src.trainer import Trainer
from src.sklearn_trainer import SklearnTrainer


def main():
    cfg = TrainConfig.from_args()
    print(cfg)

    train_loader, val_loader, test_loader, splits = get_loaders(cfg)
    print(f"\nDevice: {cfg.device}  |  Train: {splits['train']}  Val: {splits['val']}  Test: {splits['test']}")
    print(f"Model:  {cfg.model}\n")

    if cfg.is_sklearn():
        extractor = get_feature_extractor(cfg.model, cfg.backbone_checkpoint)
        trainer   = SklearnTrainer(extractor, cfg)
    else:
        model   = get_model(cfg.model, num_classes=cfg.num_classes)
        trainer = Trainer(model, cfg)

    trainer.fit(train_loader, val_loader)

    metrics = trainer.evaluate(test_loader)
    print(f"\n{'Metric':<12} {'Value':>8}")
    print("-" * 22)
    for name, key in [("Accuracy", "acc"), ("Precision", "prec"), ("Recall", "rec"), ("F1-score", "f1")]:
        print(f"{name:<12} {metrics[key]:>8.4f}")


if __name__ == "__main__":
    main()
