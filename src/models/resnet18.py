import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.models.base_model import BaseModel


class ResNet18MLP(BaseModel):
    """Estrategia 1 — Backbone congelado + cabeza MLP entrenable.

    Todos los pesos de ResNet-18 se congelan. Sólo se entrena el MLP:
        Linear(512 → 256) → ReLU → Dropout(0.3) → Linear(256 → num_classes)
    """

    def __init__(self, num_classes: int = 53):
        super().__init__(num_classes)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Congelar todo el backbone
        for param in backbone.parameters():
            param.requires_grad = False

        # Reemplazar la capa fc por el MLP
        in_features = backbone.fc.in_features          # 512
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


class ResNet18Features(nn.Module):
    """Estrategia 2 — Extractor de features para clasificadores sklearn.

    Devuelve vectores de 512 dimensiones (salida del avg-pool de ResNet-18).
    No tiene cabeza de clasificación propia.
    """

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Eliminar la capa fc; el forward devuelve el vector de 512 features
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        self.model = backbone

    def forward(self, x):
        return self.model(x)   # (B, 512)


class ResNet18FinetunedFeatures(nn.Module):
    """Extractor de features usando un backbone ResNet-18 ya fine-tuneado con nuestros datos.

    Carga un checkpoint de ResNet18Finetune, elimina la cabeza clasificadora
    y devuelve vectores de 512 dimensiones para clasificadores sklearn.
    """

    def __init__(self, checkpoint_path: str, num_classes: int = 53):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        state = torch.load(checkpoint_path, map_location="cpu")
        # El checkpoint de ResNet18Finetune guarda pesos con prefijo "model."
        state = {k.removeprefix("model."): v for k, v in state.items()}
        backbone.load_state_dict(state)

        # Eliminar cabeza clasificadora
        backbone.fc = nn.Identity()
        for param in backbone.parameters():
            param.requires_grad = False
        self.model = backbone

    def forward(self, x):
        return self.model(x)   # (B, 512)


class ResNet18Finetune(BaseModel):
    """Estrategia 3 — Fine-tuning completo.

    Todas las capas son entrenables. Sólo se reemplaza la última capa fc
    por una capa lineal: Linear(512 → num_classes).
    """

    def __init__(self, num_classes: int = 53):
        super().__init__(num_classes)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)
