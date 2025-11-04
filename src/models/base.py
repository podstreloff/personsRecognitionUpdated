from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np

class Detector(ABC):
    """Abstract person detector interface returning xyxy boxes, scores, labels."""
    @abstractmethod
    def __init__(self, device: str = "auto", conf: float = 0.3):
        ...

    @abstractmethod
    def warmup(self, img: np.ndarray) -> None:
        ...

    @abstractmethod
    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (boxes_xyxy[N,4], scores[N], labels[N]) in numpy float32/float32/int32."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...
