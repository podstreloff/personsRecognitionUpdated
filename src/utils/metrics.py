from typing import Dict, List, Tuple
import numpy as np
import time, csv, json, os

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    xa1, ya1, xa2, ya2 = np.split(boxes_a, 4, axis=1)
    xb1, yb1, xb2, yb2 = np.split(boxes_b, 4, axis=1)
    inter_x1 = np.maximum(xa1, xb1.T)
    inter_y1 = np.maximum(ya1, yb1.T)
    inter_x2 = np.minimum(xa2, xb2.T)
    inter_y2 = np.minimum(ya2, yb2.T)
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b.T - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)

class MetricAggregator:
    def __init__(self):
        self.frame_times: List[float] = []
        self.num_dets: List[int] = []
        self.avg_max_iou: List[float] = []
        self.prev_boxes = None

    def update(self, boxes: np.ndarray, start_t: float, end_t: float):
        self.frame_times.append(end_t - start_t)
        self.num_dets.append(int(len(boxes)))
        if self.prev_boxes is None or len(self.prev_boxes)==0 or len(boxes)==0:
            self.avg_max_iou.append(0.0)
        else:
            M = iou_matrix(boxes, self.prev_boxes)
            self.avg_max_iou.append(float(M.max(axis=1).mean()))
        self.prev_boxes = boxes

    def summary(self) -> Dict[str, float]:
        lat = np.array(self.frame_times, dtype=np.float64)
        dets = np.array(self.num_dets, dtype=np.float64)
        ious = np.array(self.avg_max_iou, dtype=np.float64)
        fps = 1.0/lat.mean() if lat.size else 0.0
        return {
            "frames": int(len(lat)),
            "avg_fps": float(fps),
            "latency_p50_ms": float(np.percentile(lat*1000, 50)) if lat.size else 0.0,
            "latency_p95_ms": float(np.percentile(lat*1000, 95)) if lat.size else 0.0,
            "detections_per_frame_mean": float(dets.mean()) if dets.size else 0.0,
            "detections_per_frame_std": float(dets.std(ddof=0)) if dets.size else 0.0,
            "temporal_consistency_mean_iou": float(ious.mean()) if ious.size else 0.0,
        }

    def save(self, path_json: str, path_csv: str):
        os.makedirs(os.path.dirname(path_json) or ".", exist_ok=True)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "latency_ms", "num_dets", "avg_max_iou"])
            for i, (t, n, iou) in enumerate(zip(self.frame_times, self.num_dets, self.avg_max_iou)):
                w.writerow([i, t*1000.0, n, iou])
