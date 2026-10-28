from src.compliance_rules import detect_violations, is_compliant


def test_no_hardhat_creates_violation():
    detections = [
        {
            "class_name": "NO-Hardhat",
            "confidence": 0.91,
            "bbox_xyxy": [10, 20, 100, 200],
        }
    ]

    result = detect_violations(detections)

    assert result["is_compliant"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["type"] == "missing_hardhat"
    assert result["violations"][0]["severity"] == "high"


def test_fall_detected_creates_critical_violation():
    detections = [
        {
            "class_name": "Fall-Detected",
            "confidence": 0.95,
            "bbox_xyxy": [10, 20, 100, 200],
        }
    ]

    result = detect_violations(detections)

    assert result["is_compliant"] is False
    assert len(result["violations"]) == 1
    assert result["violations"][0]["type"] == "fall_emergency"
    assert result["violations"][0]["severity"] == "critical"


def test_safe_detections_are_compliant():
    detections = [
        {
            "class_name": "Hardhat",
            "confidence": 0.90,
            "bbox_xyxy": [10, 20, 100, 200],
        },
        {
            "class_name": "Safety Vest",
            "confidence": 0.87,
            "bbox_xyxy": [30, 40, 120, 230],
        },
    ]

    result = detect_violations(detections)

    assert result["is_compliant"] is True
    assert result["violations"] == []
    assert is_compliant(result["violations"]) is True
