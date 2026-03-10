from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Interfaz común para todas las arquitecturas del proyecto.

    Cualquier modelo nuevo debe heredar de esta clase e implementar
    `forward`. El constructor debe aceptar `num_classes` como argumento.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x: Tensor de entrada con forma (B, 3, H, W)
        Returns:
            logits: Tensor con forma (B, num_classes)
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
