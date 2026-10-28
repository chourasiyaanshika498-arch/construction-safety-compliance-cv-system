from src.risk_scoring import calculate_risk_score, get_risk_level, load_risk_config


def test_risk_score_calculation_is_correct():
    config = load_risk_config()

    detections = [
        {"class_name": "NO-Hardhat"},
        {"class_name": "NO-Safety Vest"},
        {"class_name": "Hardhat"},
    ]

    result = calculate_risk_score(detections, config)

    assert result["risk_score"] == 55
    assert result["risk_level"] == "medium"
    assert result["contributing_classes"] == ["NO-Hardhat", "NO-Safety Vest"]


def test_critical_risk_when_fall_detected_exists():
    config = load_risk_config()

    detections = [
        {"class_name": "Fall-Detected"},
    ]

    result = calculate_risk_score(detections, config)

    assert result["risk_score"] == 100
    assert result["risk_level"] == "critical"


def test_safe_score_level():
    config = load_risk_config()

    assert get_risk_level(0, config) == "safe"


def test_low_score_level():
    config = load_risk_config()

    assert get_risk_level(10, config) == "low"
