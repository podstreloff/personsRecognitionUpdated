# src/utils/denoise.py
from collections import deque
from typing import Deque
import cv2
import numpy as np

class Denoiser:
    def __init__(self,
                 method: str = "none",
                 ksize: int = 5,
                 sigma: float = 0.0,
                 bilateral_sigma_color: float = 50.0,
                 bilateral_sigma_space: float = 7.0,
                 nlm_h: float = 7.0,
                 nlm_h_color: float = 7.0,
                 nlm_template: int = 7,
                 nlm_search: int = 21,
                 multi_window: int = 5):
        """
        method:
          - 'none'             : no denoising
          - 'gauss'            : Gaussian blur (ksize odd)
          - 'median'           : Median blur (ksize odd)
          - 'bilateral'        : Bilateral filter (edge-preserving)
          - 'fastnlm'          : Fast Non-Local Means (colored, single frame)
          - 'fastnlm_multi'    : Fast NLM Colored Multi-frame (temporal window)
        """
        self.method = method.lower()
        self.ksize = max(1, int(ksize) | 1)  # ensure odd
        self.sigma = float(sigma)
        self.bilateral_sigma_color = float(bilateral_sigma_color)
        self.bilateral_sigma_space = float(bilateral_sigma_space)
        self.nlm_h = float(nlm_h)
        self.nlm_h_color = float(nlm_h_color)
        self.nlm_template = int(nlm_template)
        self.nlm_search = int(nlm_search)
        self.multi_window = max(1, int(multi_window))
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.multi_window)

    def _gauss(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (self.ksize, self.ksize), self.sigma or 0)

    def _median(self, frame: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(frame, self.ksize)

    def _bilateral(self, frame: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(frame, self.ksize, self.bilateral_sigma_color, self.bilateral_sigma_space)

    def _fastnlm(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame8 = (np.clip(frame, 0, 255)).astype(np.uint8)
        else:
            frame8 = frame
        return cv2.fastNlMeansDenoisingColored(frame8, None,
                                               h=self.nlm_h, hColor=self.nlm_h_color,
                                               templateWindowSize=self.nlm_template,
                                               searchWindowSize=self.nlm_search)

    def _fastnlm_multi(self, frame: np.ndarray) -> np.ndarray:
        self.buffer.append(frame)
        if len(self.buffer) < self.buffer.maxlen:
            return frame
        frames = list(self.buffer)
        frames8 = [f if f.dtype == np.uint8 else (np.clip(f, 0, 255)).astype(np.uint8) for f in frames]
        center = len(frames8) // 2
        out = cv2.fastNlMeansDenoisingColoredMulti(frames8, center, len(frames8),
                                                   None, h=self.nlm_h, hColor=self.nlm_h_color,
                                                   templateWindowSize=self.nlm_template,
                                                   searchWindowSize=self.nlm_search)
        return out

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return frame
        if self.method == "gauss":
            return self._gauss(frame)
        if self.method == "median":
            return self._median(frame)
        if self.method == "bilateral":
            return self._bilateral(frame)
        if self.method == "fastnlm":
            return self._fastnlm(frame)
        if self.method == "fastnlm_multi":
            return self._fastnlm_multi(frame)
        return frame
