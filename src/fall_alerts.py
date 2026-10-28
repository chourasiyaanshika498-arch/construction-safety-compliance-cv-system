from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_current_timestamp() -> str:
    """Return ISO-style timestamp for alert records."""
    return datetime.now().isoformat(timespec="seconds")


def get_file_timestamp() -> str:
    """Return filesystem-friendly timestamp for snapshot filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_fall_alert(
    detection: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """
    Create one critical fall emergency alert from a Fall-Detected detection.

    Output:
    {
        "alert_type": "fall_emergency",
        "severity": "critical",
        "message": "Fall detected. Immediate response required.",
        "timestamp": "...",
        "class_name": "Fall-Detected"
    }
    """
    if timestamp is None:
        timestamp = get_current_timestamp()

    alert = {
        "alert_type": "fall_emergency",
        "severity": "critical",
        "message": "Fall detected. Immediate response required.",
        "timestamp": timestamp,
        "class_name": detection.get("class_name", "Fall-Detected"),
        "confidence": detection.get("confidence"),
        "bbox_xyxy": detection.get("bbox_xyxy"),
    }

    if "source" in detection:
        alert["source"] = detection["source"]

    if "worker_id" in detection:
        alert["worker_id"] = detection["worker_id"]

    return alert


def detect_fall_alerts(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return critical alerts for all Fall-Detected detections."""
    alerts: list[dict[str, Any]] = []
    timestamp = get_current_timestamp()

    for detection in detections:
        if detection.get("class_name") != "Fall-Detected":
            continue

        alerts.append(create_fall_alert(detection, timestamp))

    return alerts


def save_alert_snapshot(
    frame: np.ndarray,
    output_dir: str | Path = "results/alerts",
) -> Path:
    """
    Save a frame snapshot for emergency review.

    This function does not draw anything by itself.
    It saves the frame exactly as received.
    """
    directory = Path(output_dir)

    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory

    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"fall_alert_{get_file_timestamp()}.jpg"

    success = cv2.imwrite(str(output_path), frame)

    if not success:
        raise IOError(f"Failed to save alert snapshot: {output_path}")

    return output_path


def main() -> None:
    """Manual smoke test for fall alerts."""
    sample_detections = [
        {
            "class_id": 0,
            "class_name": "Fall-Detected",
            "confidence": 0.94,
            "bbox_xyxy": [120, 180, 420, 520],
            "source": "manual_test.jpg",
        },
        {
            "class_id": 3,
            "class_name": "Hardhat",
            "confidence": 0.88,
            "bbox_xyxy": [10, 20, 80, 90],
        },
    ]

    alerts = detect_fall_alerts(sample_detections)
    print("Alerts:", alerts)

    if len(alerts) != 1:
        raise AssertionError("Expected exactly one fall alert.")

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snapshot_path = save_alert_snapshot(dummy_frame)
    print(f"Snapshot saved to: {snapshot_path}")
    print("Fall alerts smoke test passed.")


if __name__ == "__main__":
    main()
