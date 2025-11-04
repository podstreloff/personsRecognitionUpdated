from typing import Tuple
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from .base import Detector
from PIL import Image

class DETRDetector(Detector):
    """На самом деле Faster R-CNN (ResNet50-FPN) как альтернативная архитектура к YOLO."""
    def __init__(self, device: str = "auto", conf: float = 0.3):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.preprocess = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        self.conf = conf

    def warmup(self, img: np.ndarray) -> None:
        _ = self.infer(img)

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        processed = self.preprocess(img).to(self.device)
        with torch.no_grad():
            out = self.model([processed])[0]
        boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"].detach().cpu().numpy().astype(np.int32)
        keep = (labels == 1) & (scores >= self.conf)  # 'person' = 1
        return boxes[keep], scores[keep], labels[keep]

    def name(self) -> str:
        return "Faster R-CNN R50-FPN"
    def infer(self, img: np.ndarray):
        # OpenCV -> numpy BGR  =>  PIL RGB
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB без cv2
        else:
            pil_img = img  # если уже PIL

        processed = self.preprocess(pil_img).to(self.device)  # Tensor[C,H,W]
        with torch.no_grad():
            out = self.model([processed])[0]

        boxes  = out["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = out["scores"].detach().cpu().numpy().astype(np.float32)
        labels = out["labels"].detach().cpu().numpy().astype(np.int32)

        keep = (labels == 1) & (scores >= self.conf)  # 'person' = 1
        return boxes[keep], scores[keep], labels[keep]