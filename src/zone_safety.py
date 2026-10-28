from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.worker_tracking import bbox_center


def load_zones(path: str | Path) -> list[dict[str, Any]]:
    """Load safety zones from a JSON config file."""
    zone_path = Path(path)

    if not zone_path.is_absolute():
        zone_path = PROJECT_ROOT / zone_path

    if not zone_path.exists():
        raise FileNotFoundError(f"Zone config not found: {zone_path}")

    with zone_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    zones = data.get("zones")

    if not isinstance(zones, list):
        raise ValueError("Zone config must contain a 'zones' list.")

    return zones


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[list[float]] | list[tuple[float, float]],
) -> bool:
    """
    Check if a point is inside a polygon using the ray-casting algorithm.

    point: (x, y)
    polygon: [[x1, y1], [x2, y2], ...]
    """
    x, y = point
    inside = False

    if len(polygon) < 3:
        return False

    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (float(xj - xi) * float(y - yi) / float(yj - yi + 1e-12)) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def worker_inside_zone(
    worker_detection: dict[str, Any],
    zone: dict[str, Any],
) -> bool:
    """Return True if a worker detection center is inside a zone polygon."""
    if worker_detection.get("class_name") != "Person":
        return False

    bbox = worker_detection.get("bbox_xyxy")
    if not bbox:
        return False

    polygon = zone.get("polygon")
    if not polygon:
        return False

    center = bbox_center(bbox)
    return point_in_polygon(center, polygon)


def _worker_has_ppe_nearby(
    worker_detection: dict[str, Any],
    detections: list[dict[str, Any]],
    ppe_class: str,
    max_center_distance: float = 180.0,
) -> bool:
    """
    Approximate whether a worker has a PPE item nearby.

    This is a simple heuristic until person-to-PPE association is improved.
    It checks if a required PPE detection center is close to the worker center.
    """
    worker_bbox = worker_detection.get("bbox_xyxy")
    if not worker_bbox:
        return False

    worker_center = bbox_center(worker_bbox)

    for detection in detections:
        if detection.get("class_name") != ppe_class:
            continue

        ppe_bbox = detection.get("bbox_xyxy")
        if not ppe_bbox:
            continue

        ppe_center = bbox_center(ppe_bbox)

        dx = worker_center[0] - ppe_center[0]
        dy = worker_center[1] - ppe_center[1]
        center_distance = (dx * dx + dy * dy) ** 0.5

        if center_distance <= max_center_distance:
            return True

    return False


def evaluate_zone_compliance(
    worker_detection: dict[str, Any],
    detections: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate zone-based safety for one worker.

    Rule:
    If a worker is inside a high-risk zone and required PPE is missing,
    create one zone violation per missing PPE item.
    """
    violations: list[dict[str, Any]] = []

    if worker_detection.get("class_name") != "Person":
        return violations

    for zone in zones:
        if not worker_inside_zone(worker_detection, zone):
            continue

        required_ppe = zone.get("required_ppe", [])

        for ppe_class in required_ppe:
            has_ppe = _worker_has_ppe_nearby(
                worker_detection=worker_detection,
                detections=detections,
                ppe_class=ppe_class,
            )

            if has_ppe:
                continue

            violations.append(
                {
                    "type": "zone_missing_ppe",
                    "severity": "high" if zone.get("type") == "high_risk" else "medium",
                    "zone_name": zone.get("name", "Unnamed Zone"),
                    "zone_type": zone.get("type", "unknown"),
                    "worker_id": worker_detection.get("worker_id"),
                    "missing_ppe": ppe_class,
                    "message": (
                        f"Worker inside {zone.get('name', 'zone')} "
                        f"without required PPE: {ppe_class}."
                    ),
                    "bbox_xyxy": worker_detection.get("bbox_xyxy"),
                }
            )

    return violations


def main() -> None:
    zones = load_zones("configs/zones.example.json")

    worker = {
        "class_name": "Person",
        "worker_id": 1,
        "bbox_xyxy": [200, 200, 300, 400],
    }

    detections = [worker]

    violations = evaluate_zone_compliance(worker, detections, zones)

    print("Zones:", zones)
    print("Violations:", violations)


if __name__ == "__main__":
    main()
