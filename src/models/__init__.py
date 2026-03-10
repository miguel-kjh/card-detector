from src.models.base_model import BaseModel
from src.models.custom_cnn import CustomCNN
from src.models.resnet18 import ResNet18MLP, ResNet18Features, ResNet18Finetune, ResNet18FinetunedFeatures

_PYTORCH_REGISTRY = {
    "custom_cnn":                CustomCNN,
    "resnet18_mlp":              ResNet18MLP,
    "resnet18_finetune":         ResNet18Finetune,
}

_SKLEARN_REGISTRY = {
    "resnet18_sklearn":          ResNet18Features,
    "resnet18_finetuned_sklearn": ResNet18FinetunedFeatures,
}


def get_model(name: str, num_classes: int) -> BaseModel:
    """Factory para modelos PyTorch (Trainer)."""
    if name not in _PYTORCH_REGISTRY:
        available = list(_PYTORCH_REGISTRY.keys())
        raise ValueError(f"Modelo PyTorch '{name}' no encontrado. Disponibles: {available}")
    return _PYTORCH_REGISTRY[name](num_classes=num_classes)


def get_feature_extractor(name: str, backbone_checkpoint: str = ""):
    """Factory para extractores de features (SklearnTrainer)."""
    if name not in _SKLEARN_REGISTRY:
        available = list(_SKLEARN_REGISTRY.keys())
        raise ValueError(f"Extractor '{name}' no encontrado. Disponibles: {available}")
    if name == "resnet18_finetuned_sklearn":
        if not backbone_checkpoint:
            raise ValueError("'resnet18_finetuned_sklearn' requiere --backbone_checkpoint")
        return ResNet18FinetunedFeatures(checkpoint_path=backbone_checkpoint)
    return _SKLEARN_REGISTRY[name]()
