import torch.nn as nn
from src.models.base_model import BaseModel


class CustomCNN(BaseModel):
    """CNN propia con 3 bloques convolucionales y una capa lineal de clasificación.

    Arquitectura:
        Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → 112x112
        Block 2: Conv(32→64) + BN + ReLU + MaxPool  →  56x56
        Block 3: Conv(64→128)+ BN + ReLU + MaxPool  →  28x28
        Classifier: Linear(128*28*28 → num_classes)
    """

    def __init__(self, num_classes: int = 53):
        super().__init__(num_classes)
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(128 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
