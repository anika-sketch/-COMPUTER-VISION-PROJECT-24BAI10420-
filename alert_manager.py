"""
alert_manager.py
-----------------
AlertManager: raises the real-time visual (on-screen indicator + banner) and
audio alert when the DrowsinessScorer reports a sustained Drowsy/Distracted
state. Audio playback is best-effort and non-blocking-safe: if no audio
backend/output device is available (e.g. headless CI), it degrades to a
console warning rather than crashing the monitoring loop (robustness NFR).
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

from scorer import DriverState

STATE_COLORS = {
    DriverState.NORMAL: (0, 200, 0),        # green, BGR
    DriverState.DROWSY: (0, 0, 255),        # red
    DriverState.DISTRACTED: (0, 165, 255),  # orange
    DriverState.NO_FACE: (0, 255, 255),     # yellow
}


class AlertManager:
    def __init__(self, sound_path: str = "assets/alert.wav"):
        self.sound_path = sound_path
        self.alert_active = False
        self._audio_backend_ok = self._probe_audio_backend()

    @staticmethod
    def _probe_audio_backend() -> bool:
        try:
            import simpleaudio  # noqa: F401
            return True
        except ImportError:
            return False

    def trigger_visual_alert(self, frame: np.ndarray, state: DriverState, score: float) -> np.ndarray:
        """Draws the state indicator + banner onto the frame and returns it."""
        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.circle(frame, (30, 30), 15, color, -1)
        cv2.putText(
            frame, f"{state.value} ({score:.2f})", (55, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
        if state in (DriverState.DROWSY, DriverState.DISTRACTED):
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 40), (w, h), color, -1)
            cv2.putText(
                frame, f"ALERT: {state.value.upper()} - STAY FOCUSED",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return frame

    def trigger_audio_alert(self) -> None:
        """Fire-and-forget audio alert on a background thread so it never
        blocks the per-frame processing loop (latency NFR)."""
        if self.alert_active:
            return  # avoid overlapping/spamming alerts
        self.alert_active = True
        threading.Thread(target=self._play_sound, daemon=True).start()

    def _play_sound(self) -> None:
        try:
            if self._audio_backend_ok and os.path.exists(self.sound_path):
                import simpleaudio
                wave_obj = simpleaudio.WaveObject.from_wave_file(self.sound_path)
                play_obj = wave_obj.play()
                play_obj.wait_done()
            else:
                # Graceful degradation: no audio device/library -> console beep
                print("\a[AlertManager] DROWSINESS ALERT (audio backend unavailable)")
        except Exception as exc:  # never let alerting crash the main loop
            print(f"[AlertManager] Audio alert failed: {exc}")
        finally:
            self.alert_active = False

    def reset(self) -> None:
        self.alert_active = False
