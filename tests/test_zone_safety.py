from src.zone_safety import (
    evaluate_zone_compliance,
    load_zones,
    point_in_polygon,
    worker_inside_zone,
)


def test_point_inside_polygon_returns_true():
    polygon = [[100, 100], [600, 100], [600, 500], [100, 500]]
    assert point_in_polygon((250, 250), polygon) is True


def test_point_outside_polygon_returns_false():
    polygon = [[100, 100], [600, 100], [600, 500], [100, 500]]
    assert point_in_polygon((50, 50), polygon) is False


def test_worker_inside_zone_returns_true():
    zone = {
        "name": "Ladder Zone",
        "type": "high_risk",
        "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
        "required_ppe": ["Hardhat", "Safety Vest"],
    }

    worker = {
        "class_name": "Person",
        "worker_id": 1,
        "bbox_xyxy": [200, 200, 300, 400],
    }

    assert worker_inside_zone(worker, zone) is True


def test_worker_inside_zone_without_ppe_creates_violation():
    zones = [
        {
            "name": "Ladder Zone",
            "type": "high_risk",
            "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
            "required_ppe": ["Hardhat", "Safety Vest"],
        }
    ]

    worker = {
        "class_name": "Person",
        "worker_id": 1,
        "bbox_xyxy": [200, 200, 300, 400],
    }

    detections = [worker]

    violations = evaluate_zone_compliance(worker, detections, zones)

    assert len(violations) == 2
    assert violations[0]["type"] == "zone_missing_ppe"
    assert violations[0]["severity"] == "high"
    assert violations[0]["zone_name"] == "Ladder Zone"


def test_worker_inside_zone_with_required_ppe_has_no_violation():
    zones = [
        {
            "name": "Ladder Zone",
            "type": "high_risk",
            "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
            "required_ppe": ["Hardhat"],
        }
    ]

    worker = {
        "class_name": "Person",
        "worker_id": 1,
        "bbox_xyxy": [200, 200, 300, 400],
    }

    hardhat = {
        "class_name": "Hardhat",
        "bbox_xyxy": [220, 150, 280, 210],
    }

    detections = [worker, hardhat]

    violations = evaluate_zone_compliance(worker, detections, zones)

    assert violations == []


def test_load_zones_from_example_config():
    zones = load_zones("configs/zones.example.json")

    assert len(zones) == 1
    assert zones[0]["name"] == "Ladder Zone"
    assert zones[0]["type"] == "high_risk"
