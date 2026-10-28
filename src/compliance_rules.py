from __future__ import annotations

from typing import Any


VIOLATION_RULES: dict[str, dict[str, str]] = {
    "NO-Hardhat": {
        "type": "missing_hardhat",
        "severity": "high",
        "message": "Worker detected without hardhat.",
    },
    "NO-Safety Vest": {
        "type": "missing_safety_vest",
        "severity": "high",
        "message": "Worker detected without safety vest.",
    },
    "NO-Mask": {
        "type": "missing_mask",
        "severity": "medium",
        "message": "Worker detected without mask.",
    },
    "NO-Gloves": {
        "type": "missing_gloves",
        "severity": "medium",
        "message": "Worker detected without gloves.",
    },
    "NO-Goggles": {
        "type": "missing_goggles",
        "severity": "medium",
        "message": "Worker detected without goggles.",
    },
    "Fall-Detected": {
        "type": "fall_emergency",
        "severity": "critical",
        "message": "Fall detected. Immediate response required.",
    },
}


def detect_violations(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Convert object detections into safety compliance violations.

    Input detection example:
    {
        "class_id": 8,
        "class_name": "NO-Hardhat",
        "confidence": 0.91,
        "bbox_xyxy": [10, 20, 100, 200]
    }

    Output:
    {
        "is_compliant": false,
        "violations": [...]
    }
    """
    violations: list[dict[str, Any]] = []

    for detection in detections:
        class_name = detection.get("class_name")

        if class_name not in VIOLATION_RULES:
            continue

        rule = VIOLATION_RULES[class_name]

        violation = {
            "type": rule["type"],
            "severity": rule["severity"],
            "class_name": class_name,
            "message": rule["message"],
            "confidence": detection.get("confidence"),
            "bbox_xyxy": detection.get("bbox_xyxy"),
        }

        if "source" in detection:
            violation["source"] = detection["source"]

        violations.append(violation)

    return {
        "is_compliant": is_compliant(violations),
        "violations": violations,
    }


def is_compliant(violations: list[dict[str, Any]]) -> bool:
    """Return True only when no safety violations are detected."""
    return len(violations) == 0


def main() -> None:
    """Small manual test for this module."""
    sample_detections = [
        {
            "class_id": 8,
            "class_name": "NO-Hardhat",
            "confidence": 0.92,
            "bbox_xyxy": [10, 20, 120, 220],
        },
        {
            "class_id": 13,
            "class_name": "Safety Vest",
            "confidence": 0.88,
            "bbox_xyxy": [30, 40, 160, 250],
        },
    ]

    result = detect_violations(sample_detections)
    print(result)


if __name__ == "__main__":
    main()
