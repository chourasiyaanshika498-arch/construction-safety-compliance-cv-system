from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_timestamp() -> str:
    """Return ISO-style timestamp for log records and reports."""
    return datetime.now().isoformat(timespec="seconds")


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def create_log_record(
    source: str,
    detections: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    risk_result: dict[str, Any],
    alerts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Create one structured safety log record.

    This can represent:
    - one image
    - one video
    - one video frame
    """
    return {
        "timestamp": get_timestamp(),
        "source": source,
        "detections": detections,
        "violations": violations,
        "risk": risk_result,
        "alerts": alerts or [],
        "is_compliant": len(violations) == 0,
    }


def save_log_record(record: dict[str, Any], output_path: str | Path) -> Path:
    """Save one log record as JSON."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    return path


def _load_json_file(path: Path) -> Any:
    """Load JSON content from a file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_records(data: Any, file_path: Path) -> list[dict[str, Any]]:
    """
    Normalize different log shapes into a list of records.

    Supported:
    - list of frame records
    - one dictionary log record
    - dictionary with records key
    """
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return [record for record in data["records"] if isinstance(record, dict)]

        if isinstance(data.get("detections"), list):
            return [data]

    print(f"Warning: unsupported log format skipped: {file_path}")
    return []


def load_log_records(log_dir: str | Path) -> list[dict[str, Any]]:
    """Load all JSON log records from a directory."""
    directory = resolve_path(log_dir)

    if not directory.exists():
        raise FileNotFoundError(f"Log directory does not exist: {directory}")

    json_files = sorted(directory.glob("*.json"))

    records: list[dict[str, Any]] = []

    for json_file in json_files:
        try:
            data = _load_json_file(json_file)
            records.extend(_normalize_records(data, json_file))
        except Exception as error:
            print(f"Warning: failed to read {json_file}: {error}")

    return records


def _get_risk_score(record: dict[str, Any]) -> float:
    """Safely extract risk score from a record."""
    risk = record.get("risk", {})

    if isinstance(risk, dict):
        return float(risk.get("risk_score", 0) or 0)

    return 0.0


def _get_violation_types(record: dict[str, Any]) -> list[str]:
    """Extract violation type names from a record."""
    violations = record.get("violations", [])

    if not isinstance(violations, list):
        return []

    types = []

    for violation in violations:
        if not isinstance(violation, dict):
            continue

        types.append(str(violation.get("type", "unknown_violation")))

    return types


def _get_alerts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract alerts from a record."""
    alerts = record.get("alerts", [])

    if isinstance(alerts, list):
        return [alert for alert in alerts if isinstance(alert, dict)]

    return []


def summarize_logs(log_dir: str | Path) -> dict[str, Any]:
    """Summarize all safety logs in a directory."""
    records = load_log_records(log_dir)

    total_processed_items = len(records)
    total_detections = 0
    total_violations = 0
    compliant_items = 0
    critical_alerts_count = 0
    risk_scores: list[float] = []
    violation_counter: Counter[str] = Counter()

    for record in records:
        detections = record.get("detections", [])
        violations = record.get("violations", [])

        if isinstance(detections, list):
            total_detections += len(detections)

        if isinstance(violations, list):
            total_violations += len(violations)

        violation_counter.update(_get_violation_types(record))

        alerts = _get_alerts(record)
        critical_alerts_count += sum(
            1
            for alert in alerts
            if alert.get("severity") == "critical"
            or alert.get("alert_type") == "fall_emergency"
        )

        risk_scores.append(_get_risk_score(record))

        if record.get("is_compliant") is True:
            compliant_items += 1
        elif isinstance(violations, list) and len(violations) == 0:
            compliant_items += 1

    if total_processed_items > 0:
        average_risk_score = sum(risk_scores) / total_processed_items
        compliance_percentage = (compliant_items / total_processed_items) * 100
    else:
        average_risk_score = 0.0
        compliance_percentage = 0.0

    most_common_violation = None
    if violation_counter:
        most_common_violation = violation_counter.most_common(1)[0][0]

    return {
        "generated_at": get_timestamp(),
        "log_dir": str(resolve_path(log_dir)),
        "total_processed_items": total_processed_items,
        "total_detections": total_detections,
        "total_violations": total_violations,
        "most_common_violation": most_common_violation,
        "critical_alerts_count": critical_alerts_count,
        "average_risk_score": round(average_risk_score, 2),
        "compliance_percentage": round(compliance_percentage, 2),
        "violation_breakdown": dict(violation_counter),
    }


def export_json(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Export summary as JSON."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return path


def export_csv(summary: dict[str, Any], output_path: str | Path) -> Path:
    """Export summary as one-row CSV."""
    path = resolve_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat_summary = dict(summary)
    flat_summary["violation_breakdown"] = json.dumps(
        summary.get("violation_breakdown", {}),
        ensure_ascii=False,
    )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(flat_summary.keys()))
        writer.writeheader()
        writer.writerow(flat_summary)

    return path


def main() -> None:
    """Manual smoke test for reporting utilities."""
    log_dir = resolve_path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    record = create_log_record(
        source="manual_test.jpg",
        detections=[
            {"class_name": "Person"},
            {"class_name": "NO-Hardhat"},
        ],
        violations=[
            {
                "type": "missing_hardhat",
                "severity": "high",
                "message": "Worker detected without hardhat.",
            }
        ],
        risk_result={
            "risk_score": 30,
            "risk_level": "medium",
        },
        alerts=[],
    )

    save_log_record(record, log_dir / "manual_reporting_test.json")

    summary = summarize_logs(log_dir)
    print(summary)


if __name__ == "__main__":
    main()
