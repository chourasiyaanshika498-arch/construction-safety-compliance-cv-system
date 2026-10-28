from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def bbox_center(bbox: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the center point of a bounding box in xyxy format."""
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass
class Track:
    """Internal track state for one worker."""
    worker_id: int
    center: tuple[float, float]
    missing_frames: int = 0


class WorkerTracker:
    """
    Lightweight center-based tracker for Person detections.

    This tracker is intentionally simple:
    - It tracks detections where class_name == "Person"
    - It matches current person centers to previous centers by nearest distance
    - It keeps tracks alive for a few missing frames
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_missing_frames: int = 10,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.tracks: dict[int, Track] = {}
        self.next_worker_id = 1

    def reset(self) -> None:
        """Clear all tracks and restart worker IDs."""
        self.tracks.clear()
        self.next_worker_id = 1

    def _create_track(self, center: tuple[float, float]) -> int:
        worker_id = self.next_worker_id
        self.next_worker_id += 1

        self.tracks[worker_id] = Track(
            worker_id=worker_id,
            center=center,
            missing_frames=0,
        )

        return worker_id

    def _match_track(
        self,
        center: tuple[float, float],
        used_track_ids: set[int],
    ) -> int | None:
        best_worker_id: int | None = None
        best_distance = float("inf")

        for worker_id, track in self.tracks.items():
            if worker_id in used_track_ids:
                continue

            current_distance = distance(center, track.center)

            if current_distance < best_distance:
                best_distance = current_distance
                best_worker_id = worker_id

        if best_worker_id is None:
            return None

        if best_distance > self.max_distance:
            return None

        return best_worker_id

    def _remove_old_tracks(self) -> None:
        old_track_ids = [
            worker_id
            for worker_id, track in self.tracks.items()
            if track.missing_frames > self.max_missing_frames
        ]

        for worker_id in old_track_ids:
            del self.tracks[worker_id]

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Assign stable worker IDs to Person detections.

        Non-person detections are returned unchanged.
        """
        updated_detections = [dict(detection) for detection in detections]

        person_indices = [
            index
            for index, detection in enumerate(updated_detections)
            if detection.get("class_name") == "Person" and detection.get("bbox_xyxy")
        ]

        used_track_ids: set[int] = set()

        for index in person_indices:
            detection = updated_detections[index]
            center = bbox_center(detection["bbox_xyxy"])

            matched_worker_id = self._match_track(center, used_track_ids)

            if matched_worker_id is None:
                matched_worker_id = self._create_track(center)
            else:
                self.tracks[matched_worker_id].center = center
                self.tracks[matched_worker_id].missing_frames = 0

            detection["worker_id"] = matched_worker_id
            used_track_ids.add(matched_worker_id)

        for worker_id, track in self.tracks.items():
            if worker_id not in used_track_ids:
                track.missing_frames += 1

        self._remove_old_tracks()

        return updated_detections


def main() -> None:
    """Manual smoke test for WorkerTracker."""
    tracker = WorkerTracker(max_distance=50)

    frame_1 = [
        {
            "class_name": "Person",
            "confidence": 0.90,
            "bbox_xyxy": [100, 100, 200, 300],
        }
    ]

    frame_2 = [
        {
            "class_name": "Person",
            "confidence": 0.88,
            "bbox_xyxy": [110, 105, 210, 305],
        }
    ]

    tracked_1 = tracker.update(frame_1)
    tracked_2 = tracker.update(frame_2)

    print("Frame 1:", tracked_1)
    print("Frame 2:", tracked_2)

    assert tracked_1[0]["worker_id"] == tracked_2[0]["worker_id"]
    print("Worker tracking smoke test passed.")


if __name__ == "__main__":
    main()
