from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_rules import detect_violations
from src.detection import parse_yolo_results, run_detection
from src.risk_scoring import calculate_risk_score
from src.visualization import draw_detections, draw_risk_banner
from src.worker_tracking import WorkerTracker


def open_video_source(source: str | int) -> cv2.VideoCapture:
    """Open webcam index or video file path."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    return capture


def get_video_writer(
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create a video writer for saving processed frames."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {path}")

    return writer


def calculate_fps(start_time: float, frame_count: int) -> float:
    """Calculate average FPS."""
    elapsed = time.time() - start_time

    if elapsed <= 0:
        return 0.0

    return frame_count / elapsed


def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS value on frame."""
    annotated = image.copy()

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (12, annotated.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def process_frame(
    frame: np.ndarray,
    model: Any,
    conf: float,
    risk_config: dict[str, Any],
    imgsz: int = 640,
    device: str | int | None = None,
    tracker: WorkerTracker | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Process one video frame.

    Steps:
    1. Run YOLO detection
    2. Parse detections
    3. Optionally assign worker IDs to Person detections
    4. Apply compliance rules
    5. Calculate risk score
    6. Draw boxes and risk banner
    """
    results = run_detection(
        model=model,
        source=frame,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=False,
        project="results",
        name="video_tmp",
    )

    detections = parse_yolo_results(results)

    if tracker is not None:
        detections = tracker.update(detections)

    compliance_result = detect_violations(detections)
    risk_result = calculate_risk_score(detections, risk_config)

    annotated = draw_detections(frame, detections)
    annotated = draw_risk_banner(
        annotated,
        risk_result,
        compliance_result["violations"],
    )

    frame_result = {
        "detections": detections,
        "is_compliant": compliance_result["is_compliant"],
        "violations": compliance_result["violations"],
        "risk": risk_result,
    }

    return annotated, frame_result


def main() -> None:
    """Small import/syntax smoke test."""
    print("video_processor.py loaded successfully.")


if __name__ == "__main__":
    main()
