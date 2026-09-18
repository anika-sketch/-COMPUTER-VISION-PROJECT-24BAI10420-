"""
main.py
-------
Entry point. MainController wires together FrameCapture -> Preprocessor ->
FaceDetector -> FeatureExtractor -> HeadPoseEstimator -> DrowsinessScorer ->
AlertManager -> SessionLogger, exactly as described by the process-flow
diagram in the project report (Section 6.1).

Run:
    python main.py
Press 'q' during the live window to stop the session and generate the
end-of-session fatigue report in reports/.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from config import ConfigManager
from capture import FrameCapture, CameraNotFoundError
from preprocess import Preprocessor
from face_detector import FaceDetector
from feature_extractor import FeatureExtractor
from head_pose import HeadPoseEstimator
from scorer import DrowsinessScorer, DriverState
from alert_manager import AlertManager
from session_logger import SessionLogger


class MainController:
    def __init__(self, config_path: str | None = None, headless: bool = False):
        self.config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        self.config = self.config_manager.load_config()
        self.headless = headless

        self.capture = FrameCapture(self.config.camera_index, self.config.target_fps)
        self.preprocessor = Preprocessor()
        self.face_detector = FaceDetector(
            self.config.cascade_face, self.config.cascade_eye, self.config.cascade_mouth
        )
        self.feature_extractor = FeatureExtractor()
        self.head_pose = HeadPoseEstimator()
        self.scorer = DrowsinessScorer(
            ear_threshold=self.config.ear_threshold,
            mar_threshold=self.config.mar_threshold,
            head_tilt_threshold_deg=self.config.head_tilt_threshold_deg,
            ear_weight=self.config.ear_weight,
            mar_weight=self.config.mar_weight,
            pose_weight=self.config.pose_weight,
            consecutive_frames=self.config.consecutive_frames,
        )
        self.alert_manager = AlertManager(self.config.alert_sound)
        self.logger = SessionLogger(self.config.log_dir)

        self.is_running = False
        self._no_face_counter = 0

    def process_frame(self, frame) -> tuple:
        gray = self.preprocessor.process(frame)
        detection = self.face_detector.detect(gray)

        if detection.face_box is None:
            self._no_face_counter += 1
            self.head_pose.reset()
            if not self.headless:
                cv2.putText(frame, "WARNING: No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.logger.log_event(DriverState.NO_FACE, 0.0)
            return frame, DriverState.NO_FACE, 0.0

        self._no_face_counter = 0
        x, y, w, h = detection.face_box

        # --- EAR ---
        if detection.eye_boxes:
            ears = [self.feature_extractor.estimate_ear_from_bbox(ew, eh)
                    for (_, _, ew, eh) in detection.eye_boxes]
            ear = sum(ears) / len(ears)
        else:
            ear = self.config.ear_threshold  # neutral guess if eyes not localized this frame

        # --- MAR ---
        if detection.mouth_box:
            _, _, mw, mh = detection.mouth_box
            mar = self.feature_extractor.estimate_mar_from_bbox(mw, mh)
        else:
            mar = 0.0

        self.feature_extractor.push(ear, mar)

        # --- Head pose ---
        self.head_pose.track_optical_flow(gray, detection.face_box)
        tilt_angle = self.head_pose.estimate_tilt_angle()

        # --- Fuse + classify ---
        result = self.scorer.classify_state(ear, mar, tilt_angle)

        if not self.headless:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            for (ex, ey, ew, eh) in detection.eye_boxes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
            if detection.mouth_box:
                mx, my, mw2, mh2 = detection.mouth_box
                cv2.rectangle(frame, (mx, my), (mx + mw2, my + mh2), (0, 128, 255), 1)
            frame = self.alert_manager.trigger_visual_alert(frame, result.state, result.score)

        if result.state in (DriverState.DROWSY, DriverState.DISTRACTED):
            self.alert_manager.trigger_audio_alert()

        self.logger.log_event(result.state, result.score, ear, mar, tilt_angle)
        return frame, result.state, result.score

    def start_session(self) -> None:
        try:
            self.capture.open()
        except CameraNotFoundError as exc:
            print(f"[MainController] {exc}")
            sys.exit(1)

        self.is_running = True
        print(f"[MainController] Session '{self.logger.session_id}' started. Press 'q' to stop.")

        try:
            while self.is_running:
                frame = self.capture.read_frame()
                if frame is None:
                    print("[MainController] Camera stream ended.")
                    break

                frame, state, score = self.process_frame(frame)

                if not self.headless:
                    cv2.imshow("Driver Drowsiness & Distraction Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.is_running = False
        finally:
            self.stop_session()

    def stop_session(self) -> None:
        self.capture.release()
        if not self.headless:
            cv2.destroyAllWindows()
        summary = self.logger.export_report()
        print("[MainController] Session ended. Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")


def parse_args():
    parser = argparse.ArgumentParser(description="Driver Drowsiness & Distraction Detection")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--headless", action="store_true",
                         help="Run without opening a display window (for servers/CI/testing)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controller = MainController(config_path=args.config, headless=args.headless)
    controller.start_session()
