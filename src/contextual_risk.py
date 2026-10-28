from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk_scoring import get_risk_level


CONTEXTUAL_RISK_RULES: dict[str, dict[str, Any]] = {
    "working_at_height": {
        "label": "Working at height / exposed edge",
        "risk_weight": 80,
        "severity": "high",
        "message": "Scene marked as working at height or exposed edge. Fall protection review required.",
    },
    "unprotected_edge": {
        "label": "Unprotected edge",
        "risk_weight": 80,
        "severity": "high",
        "message": "Scene marked as an unprotected edge. Guardrail, safety net, or fall arrest review required.",
    },
    "ladder_or_scaffold": {
        "label": "Ladder or scaffold risk",
        "risk_weight": 40,
        "severity": "medium",
        "message": "Scene marked as ladder/scaffold work. Extra PPE and stability review required.",
    },
}


def get_contextual_risk_options() -> dict[str, str]:
    return {key: value["label"] for key, value in CONTEXTUAL_RISK_RULES.items()}


def create_contextual_violation(risk_type: str, note: str | None = None) -> dict[str, Any]:
    if risk_type not in CONTEXTUAL_RISK_RULES:
        raise KeyError(f"Unknown contextual risk type: {risk_type}")

    rule = CONTEXTUAL_RISK_RULES[risk_type]

    violation = {
        "type": risk_type,
        "severity": rule["severity"],
        "class_name": "Contextual-Risk",
        "message": rule["message"],
        "risk_weight": rule["risk_weight"],
        "source": "manual_context_override",
    }

    if note:
        violation["note"] = note

    return violation


def apply_contextual_risk(
    violations: list[dict[str, Any]],
    risk_result: dict[str, Any],
    risk_config: dict[str, Any],
    context_flags: list[str] | None = None,
    note: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context_flags = context_flags or []

    updated_violations = [dict(violation) for violation in violations]
    updated_risk = dict(risk_result)

    current_score = int(updated_risk.get("risk_score", 0))
    contextual_factors: list[str] = []

    for risk_type in context_flags:
        if risk_type not in CONTEXTUAL_RISK_RULES:
            continue

        violation = create_contextual_violation(risk_type, note=note)
        updated_violations.append(violation)

        current_score += int(CONTEXTUAL_RISK_RULES[risk_type]["risk_weight"])
        contextual_factors.append(risk_type)

    updated_risk["risk_score"] = current_score
    updated_risk["risk_level"] = get_risk_level(current_score, risk_config)
    updated_risk["contextual_factors"] = contextual_factors

    contributing_classes = list(updated_risk.get("contributing_classes", []))
    contributing_classes.extend(contextual_factors)
    updated_risk["contributing_classes"] = contributing_classes

    return updated_violations, updated_risk


def main() -> None:
    from src.risk_scoring import load_risk_config

    config = load_risk_config()
    violations, risk = apply_contextual_risk(
        violations=[],
        risk_result={"risk_score": 0, "risk_level": "safe", "contributing_classes": []},
        risk_config=config,
        context_flags=["working_at_height"],
        note="Manual test",
    )

    print("Violations:", violations)
    print("Risk:", risk)


if __name__ == "__main__":
    main()
