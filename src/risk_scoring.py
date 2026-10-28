from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml


def load_risk_config(path: str | Path = "configs/risk_rules.yaml") -> dict[str, Any]:
    """Load risk scoring configuration from YAML."""
    config = load_yaml(path)

    if "risk_weights" not in config:
        raise KeyError("Missing 'risk_weights' in risk config.")

    if "risk_levels" not in config:
        raise KeyError("Missing 'risk_levels' in risk config.")

    return config


def calculate_risk_score(
    detections: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Calculate risk score from detections.

    Important design:
    - Duplicate detections of the same violation class are counted once.
    - Fall-Detected is always critical.
    - PPE violations can make the scene low/medium/high, but repeated false positives
      should not automatically create a fake critical emergency.
    """
    risk_weights: dict[str, int] = config["risk_weights"]

    detected_violation_classes: list[str] = []
    violation_counts: dict[str, int] = {}

    for detection in detections:
        class_name = detection.get("class_name")

        if class_name not in risk_weights:
            continue

        violation_counts[class_name] = violation_counts.get(class_name, 0) + 1

        if class_name not in detected_violation_classes:
            detected_violation_classes.append(class_name)

    # Emergency case: fall detection should dominate the score.
    if "Fall-Detected" in detected_violation_classes:
        score = int(risk_weights["Fall-Detected"])
        return {
            "risk_score": score,
            "risk_level": get_risk_level(score, config),
            "contributing_classes": detected_violation_classes,
            "violation_counts": violation_counts,
            "scoring_note": "Fall-Detected triggers critical emergency risk.",
        }

    # Count each PPE violation class once to avoid fake critical scores
    # caused by duplicate boxes or false positives.
    score = sum(
        int(risk_weights[class_name])
        for class_name in detected_violation_classes
    )

    # PPE-only cases should not become critical just because many duplicates exist.
    score = min(score, 99)

    return {
        "risk_score": score,
        "risk_level": get_risk_level(score, config),
        "contributing_classes": detected_violation_classes,
        "violation_counts": violation_counts,
        "scoring_note": "Duplicate violation classes are counted once. Critical is reserved for emergency classes.",
    }


def get_risk_level(score: int, config: dict[str, Any]) -> str:
    """Map a numeric risk score to a configured risk level."""
    risk_levels: dict[str, dict[str, int]] = config["risk_levels"]

    for level_name, bounds in risk_levels.items():
        min_score = int(bounds["min"])
        max_score = int(bounds["max"])

        if min_score <= score <= max_score:
            return level_name

    return "unknown"


def main() -> None:
    config = load_risk_config()

    sample_detections = [
        {"class_name": "NO-Hardhat"},
        {"class_name": "NO-Hardhat"},
        {"class_name": "NO-Hardhat"},
        {"class_name": "NO-Safety Vest"},
        {"class_name": "Hardhat"},
    ]

    print(calculate_risk_score(sample_detections, config))


if __name__ == "__main__":
    main()
