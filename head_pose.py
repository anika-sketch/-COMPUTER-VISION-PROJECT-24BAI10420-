"""
head_pose.py
------------
HeadPoseEstimator: distraction detection via sparse Lucas-Kanade (KLT)
optical flow — Module 4 (Motion / Pattern Analysis) of the CSE3010 syllabus.

Rather than running a second, expensive head-pose detector, this module
tracks a handful of points already localized on the face (corners of the
face bounding box + eye centers) across consecutive frames and estimates a
tilt/yaw angle from their displacement. Sustained displacement beyond
`head_tilt_threshold_deg` indicates the driver has turned away from the road.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


class HeadPoseEstimator:
    def __init__(self, max_corners: int = 15):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self._baseline_centroid: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Call when the face is (re)acquired, to re-baseline tracking."""
        self.prev_gray = None
        self.prev_points = None
        self._baseline_centroid = None

    def _init_points(self, gray_frame: np.ndarray, face_box: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = face_box
        roi = gray_frame[y:y + h, x:x + w]
        corners = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        if corners is None:
            # Fall back to the four face-box corners if no strong features found
            corners = np.array(
                [[[0, 0]], [[w - 1, 0]], [[0, h - 1]], [[w - 1, h - 1]]], dtype=np.float32
            )
        corners[:, 0, 0] += x
        corners[:, 0, 1] += y
        return corners

    def track_optical_flow(
        self, gray_frame: np.ndarray, face_box: tuple[int, int, int, int]
    ) -> np.ndarray:
        """Runs one step of KLT tracking; returns the currently tracked points."""
        if self.prev_gray is None or self.prev_points is None or len(self.prev_points) < 3:
            self.prev_points = self._init_points(gray_frame, face_box)
            self.prev_gray = gray_frame
            self._baseline_centroid = self.prev_points.reshape(-1, 2).mean(axis=0)
            return self.prev_points

        next_points, status, _err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.prev_points, None, **self.lk_params
        )
        if next_points is None or status is None:
            self.reset()
            return self._init_points(gray_frame, face_box)

        good_new = next_points[status.flatten() == 1]
        if len(good_new) < 3:
            # Lost too many points (occlusion, fast motion) — re-acquire
            self.reset()
            return self._init_points(gray_frame, face_box)

        self.prev_points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray_frame
        return self.prev_points

    def estimate_tilt_angle(self) -> float:
        """Approximates head yaw/tilt (degrees) as the horizontal displacement
        of the tracked-point centroid relative to its baseline position,
        scaled into a pseudo-angle. Positive = turned right, negative = left.
        """
        if self.prev_points is None or self._baseline_centroid is None:
            return 0.0
        current_centroid = self.prev_points.reshape(-1, 2).mean(axis=0)
        dx = current_centroid[0] - self._baseline_centroid[0]
        dy = current_centroid[1] - self._baseline_centroid[1]
        # Empirical scale factor mapping pixel displacement -> degrees,
        # calibrated against the self-recorded validation clips (Sec. 7.1).
        angle = math.degrees(math.atan2(dx, 120.0))
        return angle
