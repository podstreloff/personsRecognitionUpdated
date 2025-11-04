from typing import Tuple, Generator, Optional
import cv2

def open_video(path: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, w, h, fps

def writer(path: str, w: int, h: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def resize_keep_aspect(frame, long_side: Optional[int]):
    if not long_side:
        return frame
    h, w = frame.shape[:2]
    if w >= h:
        new_w = long_side
        new_h = int(h * long_side / w)
    else:
        new_h = long_side
        new_w = int(w * long_side / h)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
