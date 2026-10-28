from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


SAFE_CLASSES = {
    "Hardhat",
    "Safety Vest",
    "Gloves",
    "Goggles",
    "Mask",
    "Person",
    "Safety Cone",
    "Ladder",
}

VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
    "NO-Mask",
    "NO-Gloves",
    "NO-Goggles",
    "Fall-Detected",
}


def _risk_color(risk_level: str) -> tuple[int, int, int]:
    """Return BGR color based on risk level."""
    colors = {
        "safe": (40, 180, 40),
        "low": (0, 200, 255),
        "medium": (0, 165, 255),
        "high": (0, 80, 255),
        "critical": (0, 0, 255),
        "unknown": (160, 160, 160),
    }
    return colors.get(risk_level.lower(), colors["unknown"])


def _class_color(class_name: str) -> tuple[int, int, int]:
    """Return BGR color based on class type."""
    if class_name == "Fall-Detected":
        return (0, 0, 255)

    if class_name in VIOLATION_CLASSES:
        return (0, 80, 255)

    if class_name in SAFE_CLASSES:
        return (40, 180, 40)

    return (255, 180, 40)


def _clip_bbox(
    bbox: list[float] | tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Clip bounding box coordinates to image size."""
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    return x1, y1, x2, y2


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw bounding boxes and class labels on an image."""
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        class_name = str(detection.get("class_name", "unknown"))
        confidence = detection.get("confidence")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
        color = _class_color(class_name)

        label = class_name
        if confidence is not None:
            label = f"{class_name} {float(confidence):.2f}"

        if "worker_id" in detection:
            label = f"ID {detection['worker_id']} | {label}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        text_width, text_height = text_size

        label_y1 = max(0, y1 - text_height - 8)
        label_y2 = max(text_height + 8, y1)

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (min(x1 + text_width + 8, width - 1), label_y2),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(label_y2 - 5, text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_risk_banner(
    image: np.ndarray,
    risk_result: dict[str, Any],
    violations: list[dict[str, Any]],
) -> np.ndarray:
    """Draw risk score, risk level, and violation summary on image."""
    annotated = image.copy()
    height, width = annotated.shape[:2]

    risk_score = int(risk_result.get("risk_score", 0))
    risk_level = str(risk_result.get("risk_level", "unknown")).upper()
    color = _risk_color(risk_result.get("risk_level", "unknown"))

    banner_height = 78
    cv2.rectangle(annotated, (0, 0), (width, banner_height), color, -1)

    title = f"Risk Level: {risk_level} | Risk Score: {risk_score}"
    cv2.putText(
        annotated,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if violations:
        summary = ", ".join(
            violation.get("type", "violation") for violation in violations[:3]
        )
        if len(violations) > 3:
            summary += f", +{len(violations) - 3} more"

        text = f"Violations: {summary}"
    else:
        text = "Violations: none"

    cv2.putText(
        annotated,
        text,
        (12, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if violations:
        panel_width = min(420, width)
        panel_y1 = banner_height + 8
        line_height = 24
        panel_height = min(140, 12 + line_height * min(len(violations), 5))

        cv2.rectangle(
            annotated,
            (0, panel_y1),
            (panel_width, panel_y1 + panel_height),
            (30, 30, 30),
            -1,
        )

        for idx, violation in enumerate(violations[:5]):
            line = f"- {violation.get('severity', '').upper()}: {violation.get('message', '')}"
            y = panel_y1 + 24 + idx * line_height

            cv2.putText(
                annotated,
                line[:56],
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated


def save_annotated_image(image: np.ndarray, output_path: str | Path) -> None:
    """Save annotated image to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Failed to save image to: {path}")


def main() -> None:
    """Manual smoke test for visualization utilities."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        {
            "class_name": "NO-Hardhat",
            "confidence": 0.91,
            "bbox_xyxy": [100, 120, 260, 360],
        }
    ]

    risk_result = {
        "risk_score": 30,
        "risk_level": "medium",
    }

    violations = [
        {
            "type": "missing_hardhat",
            "severity": "high",
            "message": "Worker detected without hardhat.",
        }
    ]

    image = draw_detections(image, detections)
    image = draw_risk_banner(image, risk_result, violations)
    save_annotated_image(image, "results/visualization_smoke_test.jpg")

    print("Saved smoke test image to results/visualization_smoke_test.jpg")


if __name__ == "__main__":
    main()
