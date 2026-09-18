"""
face_detector.py
-----------------
FaceDetector: CPU-only face detection plus eye/mouth ROI extraction, using
OpenCV's built-in Haar cascades (haarcascade_frontalface_default,
haarcascade_eye, haarcascade_smile-as-mouth-proxy). Haar cascades were chosen
over deep-learning detectors to satisfy the no-GPU / low-resource
non-functional requirement (see report Section 7.2 for the full rationale).

If a more accurate landmark backend (dlib 68-point or MediaPipe FaceMesh) is
installed, FeatureExtractor can consume its landmarks instead — this class
only has to guarantee eye_roi / mouth_roi bounding boxes, so swapping the
backend does not require changes anywhere else in the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DetectionResult:
    face_box: Optional[tuple[int, int, int, int]]  # x, y, w, h
    eye_boxes: list[tuple[int, int, int, int]]
    mouth_box: Optional[tuple[int, int, int, int]]


class FaceDetector:
    def __init__(
        self,
        cascade_face: str = "haarcascade_frontalface_default.xml",
        cascade_eye: str = "haarcascade_eye.xml",
        cascade_mouth: str = "haarcascade_smile.xml",
    ):
        base = cv2.data.haarcascades
        self.face_cascade = self._load_cascade(os.path.join(base, cascade_face))
        self.eye_cascade = self._load_cascade(os.path.join(base, cascade_eye))
        self.mouth_cascade = self._load_cascade(os.path.join(base, cascade_mouth))

    @staticmethod
    def _load_cascade(path: str) -> cv2.CascadeClassifier:
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            raise FileNotFoundError(f"Could not load Haar cascade at '{path}'.")
        return cascade

    def detect_face(self, gray_frame: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Returns the largest detected face box (x, y, w, h), or None."""
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        # Largest face = most likely the driver, filters out passengers/posters
        return max(faces, key=lambda b: b[2] * b[3])

    def get_eye_roi(
        self, gray_frame: np.ndarray, face_box: tuple[int, int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        x, y, w, h = face_box
        upper_face = gray_frame[y:y + int(h * 0.6), x:x + w]
        eyes = self.eye_cascade.detectMultiScale(
            upper_face, scaleFactor=1.1, minNeighbors=6, minSize=(int(w * 0.12), int(h * 0.08))
        )
        # Convert back to full-frame coordinates
        return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]

    def get_mouth_roi(
        self, gray_frame: np.ndarray, face_box: tuple[int, int, int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        x, y, w, h = face_box
        lower_face = gray_frame[y + int(h * 0.55):y + h, x:x + w]
        mouths = self.mouth_cascade.detectMultiScale(
            lower_face, scaleFactor=1.7, minNeighbors=18, minSize=(int(w * 0.25), int(h * 0.12))
        )
        if len(mouths) == 0:
            return None
        mx, my, mw, mh = max(mouths, key=lambda b: b[2] * b[3])
        return (x + mx, y + int(h * 0.55) + my, mw, mh)

    def detect(self, gray_frame: np.ndarray) -> DetectionResult:
        face_box = self.detect_face(gray_frame)
        if face_box is None:
            return DetectionResult(face_box=None, eye_boxes=[], mouth_box=None)
        eyes = self.get_eye_roi(gray_frame, face_box)
        mouth = self.get_mouth_roi(gray_frame, face_box)
        return DetectionResult(face_box=face_box, eye_boxes=eyes, mouth_box=mouth)
