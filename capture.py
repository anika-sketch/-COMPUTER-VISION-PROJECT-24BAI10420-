"""
capture.py
----------
FrameCapture: thin, testable wrapper around cv2.VideoCapture (webcam I/O).
Isolating camera access in its own module means the rest of the pipeline can
be unit-tested with synthetic frames, and the camera-not-found failure mode
is handled in exactly one place (error-handling non-functional requirement).
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np


class CameraNotFoundError(RuntimeError):
    """Raised when the configured camera index cannot be opened."""


class FrameCapture:
    """Reads frames from a webcam (or a video file, for offline testing)."""

    def __init__(self, source: int | str = 0, target_fps: int = 15):
        self.source = source
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_read_time = 0.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise CameraNotFoundError(
                f"Could not open camera/source '{self.source}'. "
                f"Check that a webcam is connected and not in use by another app."
            )
        if self.target_fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

    def read_frame(self) -> Optional[np.ndarray]:
        """Return the next BGR frame, or None if the stream has ended."""
        if self._cap is None:
            raise RuntimeError("FrameCapture.open() must be called before read_frame().")
        ok, frame = self._cap.read()
        self._last_read_time = time.time()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
