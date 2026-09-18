"""
preprocess.py
-------------
Preprocessor: low-level image preprocessing (Module 1 of the CSE3010
syllabus) applied to every frame before detection — grayscale conversion,
CLAHE-based histogram equalization (for lighting robustness in dim cabin
light / glare), and light denoising.
"""

from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Preprocessor.to_grayscale received an empty frame.")
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def equalize_histogram(self, gray_frame: np.ndarray) -> np.ndarray:
        """CLAHE instead of plain equalizeHist to avoid over-amplifying noise
        in near-uniform regions (a known weakness of global equalization)."""
        return self._clahe.apply(gray_frame)

    def denoise(self, gray_frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray_frame, (3, 3), 0)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline used by MainController each frame."""
        gray = self.to_grayscale(frame)
        gray = self.equalize_histogram(gray)
        gray = self.denoise(gray)
        return gray
