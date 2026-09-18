"""
session_logger.py
------------------
SessionLogger: append-only per-frame logging to CSV during a session, plus
end-of-session summary export (CSV + a fatigue-trend PNG plot). Logging is
kept append-only and I/O-light so it never becomes the bottleneck in the
real-time loop (performance NFR).
"""

from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd

from scorer import DriverState


@dataclass
class FrameEvent:
    timestamp: str
    frame_index: int
    ear: float
    mar: float
    head_tilt_angle: float
    drowsiness_score: float
    state: str


class SessionLogger:
    def __init__(self, log_dir: str = "reports"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.log_path = os.path.join(self.log_dir, f"{self.session_id}.csv")
        self.records: list[FrameEvent] = []
        self.start_time = datetime.now()
        self._frame_index = 0
        self._alert_events: list[int] = []  # frame indices where an alert fired

        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame_index", "ear", "mar", "head_tilt_angle",
                              "drowsiness_score", "state"])

    def log_event(self, state: DriverState, score: float, ear: float = 0.0,
                  mar: float = 0.0, head_tilt_angle: float = 0.0) -> FrameEvent:
        event = FrameEvent(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            frame_index=self._frame_index,
            ear=round(ear, 4),
            mar=round(mar, 4),
            head_tilt_angle=round(head_tilt_angle, 2),
            drowsiness_score=round(score, 4),
            state=state.value,
        )
        self.records.append(event)
        if state in (DriverState.DROWSY, DriverState.DISTRACTED):
            self._alert_events.append(self._frame_index)

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(event.__dict__.values()))

        self._frame_index += 1
        return event

    def export_report(self) -> dict:
        """Writes a fatigue-trend PNG and returns summary statistics used for
        both the console printout and the project-report quantitative table.
        """
        if not self.records:
            return {"frames": 0}

        df = pd.DataFrame([r.__dict__ for r in self.records])
        duration_s = max((datetime.now() - self.start_time).total_seconds(), 1e-6)
        avg_fps = len(df) / duration_s

        state_counts = df["state"].value_counts().to_dict()
        drowsy_or_distracted = df["state"].isin(["Drowsy", "Distracted"]).sum()
        false_alert_rate_per_10min = (
            len(self._alert_events) / (duration_s / 600.0) if duration_s > 0 else 0.0
        )

        summary = {
            "session_id": self.session_id,
            "frames": len(df),
            "duration_s": round(duration_s, 2),
            "avg_fps": round(avg_fps, 2),
            "avg_drowsiness_score": round(df["drowsiness_score"].mean(), 4),
            "state_counts": state_counts,
            "alert_events": len(self._alert_events),
            "alerts_per_10min": round(false_alert_rate_per_10min, 2),
        }

        plot_path = os.path.join(self.log_dir, f"{self.session_id}_fatigue_trend.png")
        self._plot_trend(df, plot_path)
        summary["plot_path"] = plot_path

        summary_path = os.path.join(self.log_dir, f"{self.session_id}_summary.csv")
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        summary["summary_path"] = summary_path
        return summary

    @staticmethod
    def _plot_trend(df: pd.DataFrame, out_path: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df["frame_index"], df["drowsiness_score"], color="#2b6cb0", linewidth=1.4,
                label="Drowsiness score")
        alert_mask = df["state"].isin(["Drowsy", "Distracted"])
        if alert_mask.any():
            ax.scatter(df.loc[alert_mask, "frame_index"], df.loc[alert_mask, "drowsiness_score"],
                       color="#e53e3e", s=14, zorder=3, label="Alert frames")
        ax.axhline(0.6, color="gray", linestyle="--", linewidth=0.8, label="Alert region (approx.)")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Drowsiness score")
        ax.set_title("End-of-Session Fatigue Trend")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
