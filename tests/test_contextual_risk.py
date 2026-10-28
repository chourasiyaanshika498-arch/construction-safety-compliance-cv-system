from src.contextual_risk import apply_contextual_risk, create_contextual_violation
from src.risk_scoring import load_risk_config


def test_create_contextual_violation():
    violation = create_contextual_violation("working_at_height", note="Roof edge")

    assert violation["type"] == "working_at_height"
    assert violation["severity"] == "high"
    assert violation["risk_weight"] == 80
    assert violation["note"] == "Roof edge"


def test_contextual_risk_changes_safe_to_high():
    config = load_risk_config()

    violations, risk = apply_contextual_risk(
        violations=[],
        risk_result={
            "risk_score": 0,
            "risk_level": "safe",
            "contributing_classes": [],
        },
        risk_config=config,
        context_flags=["working_at_height"],
        note="Worker appears near exposed edge.",
    )

    assert len(violations) == 1
    assert violations[0]["type"] == "working_at_height"
    assert risk["risk_score"] == 80
    assert risk["risk_level"] == "high"
    assert "working_at_height" in risk["contextual_factors"]
