"""
config.py
---------
ConfigManager: loads/saves EAR/MAR thresholds and alert sensitivity from a
JSON-backed configuration file. Centralising configuration here satisfies the
"configurable sensitivity" functional requirement and keeps thresholds out of
the detection/scoring logic (maintainability, separation of concerns).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


@dataclass
class AppConfig:
    """Typed, validated configuration values for the whole pipeline."""

    ear_threshold: float = 0.21
    mar_threshold: float = 0.55
    head_tilt_threshold_deg: float = 25.0
    consecutive_frames: int = 20
    ear_weight: float = 0.5
    mar_weight: float = 0.3
    pose_weight: float = 0.2
    target_fps: int = 15
    camera_index: int = 0
    alert_sound: str = "assets/alert.wav"
    log_dir: str = "reports"
    cascade_face: str = "haarcascade_frontalface_default.xml"
    cascade_eye: str = "haarcascade_eye.xml"
    cascade_mouth: str = "haarcascade_smile.xml"

    def validate(self) -> None:
        if not (0.0 < self.ear_threshold < 1.0):
            raise ValueError("ear_threshold must be between 0 and 1")
        if not (0.0 < self.mar_threshold < 2.0):
            raise ValueError("mar_threshold must be between 0 and 2")
        if self.consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")
        weight_sum = round(self.ear_weight + self.mar_weight + self.pose_weight, 3)
        if weight_sum != 1.0:
            raise ValueError(
                f"ear_weight + mar_weight + pose_weight must sum to 1.0 (got {weight_sum})"
            )


class ConfigManager:
    """Reads and writes AppConfig to/from a JSON file on disk."""

    def __init__(self, path: str = DEFAULT_CONFIG_PATH):
        self.path = path
        self._config = AppConfig()

    def load_config(self) -> AppConfig:
        """Load config.json if present, else fall back to defaults."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._config = AppConfig(**{**asdict(self._config), **data})
            except (json.JSONDecodeError, TypeError) as exc:
                # Error handling: corrupt config should not crash the app;
                # fall back to safe defaults and warn the caller.
                print(f"[ConfigManager] Warning: could not parse {self.path} "
                      f"({exc}); using default configuration.")
                self._config = AppConfig()
        else:
            print(f"[ConfigManager] No config file at {self.path}; using defaults.")
        self._config.validate()
        return self._config

    def save_config(self, config: AppConfig | None = None) -> None:
        config = config or self._config
        config.validate()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(config), f, indent=2)
        self._config = config

    @property
    def config(self) -> AppConfig:
        return self._config
