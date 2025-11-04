from typing import Tuple
import numpy as np
import torch
from ultralytics import YOLO
from .base import Detector

class YOLOv8Detector(Detector):
    def __init__(self, device: str = "auto", conf: float = 0.3, half: bool = False):
        self.model = YOLO("yolov8n.pt")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)
        self.conf = conf
        self.half = half and self.device.startswith("cuda")

    def warmup(self, img: np.ndarray) -> None:
        _ = self.infer(img)

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self.model.predict(img, verbose=False, conf=self.conf, device=self.device, half=self.half)
        res = results[0]
        if res.boxes is None:
            return (np.zeros((0,4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))
        boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
        labels = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
        person_idx = labels == 0  # COCO 'person' в YOLOv8 = 0
        return boxes[person_idx], scores[person_idx], labels[person_idx]

    def name(self) -> str:
        return "YOLOv8n"
