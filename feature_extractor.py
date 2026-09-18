"""
feature_extractor.py
---------------------
FeatureExtractor: computes the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio
(MAR) — the core hand-crafted features from Module 3 (Feature Extraction) of
the CSE3010 syllabus, following Soukupova & Cech (2016).

Two input modes are supported:
  1. Landmark mode  — 6 (eye) / 8 (mouth) (x, y) points, e.g. from dlib's
     68-point model or MediaPipe FaceMesh. This is the geometrically exact
     formulation and is what the unit tests validate against hand-computed
     coordinates.
  2. Bounding-box mode — a (w, h) Haar-cascade ROI, used as a lightweight
     fallback when no landmark backend is installed. This trades precision
     for zero extra dependencies (keeps the project CPU-only / install-free).
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


Point = tuple[float, float]


def _euclidean(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


class FeatureExtractor:
    def __init__(self, history_len: int = 90):
        # Rolling history enables smoothing and the end-of-session trend plot
        self.ear_history: deque[float] = deque(maxlen=history_len)
        self.mar_history: deque[float] = deque(maxlen=history_len)

    # ---- Landmark-based (precise) ----------------------------------------

    @staticmethod
    def compute_EAR(eye_points: Sequence[Point]) -> float:
        """Standard 6-point EAR: p1..p6 going around the eye contour.
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        if len(eye_points) != 6:
            raise ValueError("compute_EAR expects exactly 6 (x, y) eye landmarks.")
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = _euclidean(p2, p6)
        vertical_2 = _euclidean(p3, p5)
        horizontal = _euclidean(p1, p4)
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    @staticmethod
    def compute_MAR(mouth_points: Sequence[Point]) -> float:
        """8-point MAR analogous to EAR: three vertical pairs over one
        horizontal (mouth-corner-to-corner) span.
        """
        if len(mouth_points) != 8:
            raise ValueError("compute_MAR expects exactly 8 (x, y) mouth landmarks.")
        p1, p2, p3, p4, p5, p6, p7, p8 = mouth_points
        vertical_1 = _euclidean(p2, p8)
        vertical_2 = _euclidean(p3, p7)
        vertical_3 = _euclidean(p4, p6)
        horizontal = _euclidean(p1, p5)
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)

    # ---- Bounding-box fallback (Haar-cascade mode) ------------------------

    @staticmethod
    def estimate_ear_from_bbox(eye_w: float, eye_h: float) -> float:
        """Approximates EAR as the open-eye height/width ratio of the Haar
        eye ROI. Less precise than landmark EAR but needs no extra model."""
        if eye_w == 0:
            return 0.0
        return float(eye_h) / float(eye_w)

    @staticmethod
    def estimate_mar_from_bbox(mouth_w: float, mouth_h: float) -> float:
        if mouth_w == 0:
            return 0.0
        return float(mouth_h) / float(mouth_w)

    # ---- Bookkeeping --------------------------------------------------

    def push(self, ear: float, mar: float) -> None:
        self.ear_history.append(ear)
        self.mar_history.append(mar)

    def smoothed_ear(self, window: int = 5) -> float:
        if not self.ear_history:
            return 0.0
        vals = list(self.ear_history)[-window:]
        return float(np.mean(vals))

    def smoothed_mar(self, window: int = 5) -> float:
        if not self.mar_history:
            return 0.0
        vals = list(self.mar_history)[-window:]
        return float(np.mean(vals))
