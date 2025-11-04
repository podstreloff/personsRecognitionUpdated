from typing import Tuple
import cv2
import numpy as np

def draw_dets(frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray, title: str = "", color=(0,255,0)) -> np.ndarray:
    img = frame.copy()
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        label = f"person {s:.2f}"
        (tw,th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return img
