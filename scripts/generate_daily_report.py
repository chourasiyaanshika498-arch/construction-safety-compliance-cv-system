from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting import export_csv, export_json, summarize_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily safety report from logs.")

    parser.add_argument(
        "--input",
        default="results/logs",
        help="Input directory containing JSON logs.",
    )
    parser.add_argument(
        "--output",
        default="reports/daily_report.json",
        help="Output report path.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Report output format.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Generate Daily Safety Report")
    print("=" * 60)
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print("=" * 60)

    summary = summarize_logs(args.input)

    if args.format == "json":
        output_path = export_json(summary, args.output)
    else:
        output_path = export_csv(summary, args.output)

    print("Report generated successfully.")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    print("Summary:")
    print(f"- Total processed items : {summary['total_processed_items']}")
    print(f"- Total detections      : {summary['total_detections']}")
    print(f"- Total violations      : {summary['total_violations']}")
    print(f"- Most common violation : {summary['most_common_violation']}")
    print(f"- Critical alerts       : {summary['critical_alerts_count']}")
    print(f"- Average risk score    : {summary['average_risk_score']}")
    print(f"- Compliance percentage : {summary['compliance_percentage']}%")


if __name__ == "__main__":
    main()
