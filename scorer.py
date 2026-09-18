"""
scorer.py
---------
DrowsinessScorer: fuses EAR, MAR and head-tilt angle into a single weighted
drowsiness score in [0, 1], then applies consecutive-frame confirmation to
classify the driver's state (Normal / Drowsy / Distracted), filtering out
single-frame noise such as normal blinking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DriverState(str, Enum):
    NORMAL = "Normal"
    DROWSY = "Drowsy"
    DISTRACTED = "Distracted"
    NO_FACE = "NoFace"


@dataclass
class ScoreResult:
    score: float
    state: DriverState
    frame_alert_condition: bool  # True if this single frame exceeds threshold


class DrowsinessScorer:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        mar_threshold: float = 0.55,
        head_tilt_threshold_deg: float = 25.0,
        ear_weight: float = 0.5,
        mar_weight: float = 0.3,
        pose_weight: float = 0.2,
        consecutive_frames: int = 20,
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.head_tilt_threshold_deg = head_tilt_threshold_deg
        self.ear_weight = ear_weight
        self.mar_weight = mar_weight
        self.pose_weight = pose_weight
        self.consecutive_frames = consecutive_frames

        self._consec_drowsy = 0
        self._consec_distracted = 0
        self.current_state = DriverState.NORMAL

    def compute_score(self, ear: float, mar: float, tilt_deg: float) -> float:
        """Weighted fusion of three normalized 'risk' sub-scores, each in
        [0, 1], where higher = more indicative of drowsiness/distraction."""
        eye_risk = max(0.0, min(1.0, (self.ear_threshold - ear) / self.ear_threshold + 0.5))
        mouth_risk = max(0.0, min(1.0, (mar - self.mar_threshold) / self.mar_threshold + 0.5))
        pose_risk = max(0.0, min(1.0, abs(tilt_deg) / (2 * self.head_tilt_threshold_deg)))

        score = (
            self.ear_weight * eye_risk
            + self.mar_weight * mouth_risk
            + self.pose_weight * pose_risk
        )
        return max(0.0, min(1.0, score))

    def classify_state(self, ear: float, mar: float, tilt_deg: float) -> ScoreResult:
        score = self.compute_score(ear, mar, tilt_deg)

        eye_closed = ear < self.ear_threshold
        yawning = mar > self.mar_threshold
        looking_away = abs(tilt_deg) > self.head_tilt_threshold_deg

        frame_alert = eye_closed or yawning or looking_away

        if eye_closed or yawning:
            self._consec_drowsy += 1
        else:
            self._consec_drowsy = 0

        if looking_away:
            self._consec_distracted += 1
        else:
            self._consec_distracted = 0

        if self._consec_drowsy >= self.consecutive_frames:
            self.current_state = DriverState.DROWSY
        elif self._consec_distracted >= self.consecutive_frames:
            self.current_state = DriverState.DISTRACTED
        else:
            self.current_state = DriverState.NORMAL

        return ScoreResult(score=score, state=self.current_state, frame_alert_condition=frame_alert)

    def reset(self) -> None:
        self._consec_drowsy = 0
        self._consec_distracted = 0
        self.current_state = DriverState.NORMAL
