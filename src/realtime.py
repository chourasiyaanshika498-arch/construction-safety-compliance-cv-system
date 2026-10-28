from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import load_yolo_model
from src.risk_scoring import load_risk_config
from src.utils import ensure_dir
from src.video_processor import (
    calculate_fps,
    draw_fps,
    get_video_writer,
    open_video_source,
    process_frame,
)
from src.worker_tracking import WorkerTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time PPE safety compliance on webcam or video."
    )

    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Device: 0, 1, cpu, etc.")
    parser.add_argument("--save-output", action="store_true", help="Save processed video.")
    parser.add_argument("--output", default="results/videos/output.mp4", help="Output video path.")
    parser.add_argument("--display", action="store_true", help="Display video window.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit for processed frames. Use 0 for no limit.",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save frame-level JSON log to results/logs.",
    )
    parser.add_argument(
        "--disable-tracking",
        action="store_true",
        help="Disable worker ID tracking.",
    )
    parser.add_argument(
        "--tracking-max-distance",
        type=float,
        default=80.0,
        help="Maximum center distance for matching the same worker.",
    )

    return parser.parse_args()


def save_realtime_log(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save realtime frame results as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Real-Time Safety Compliance")
    print("=" * 60)
    print(f"Weights     : {args.weights}")
    print(f"Source      : {args.source}")
    print(f"Confidence  : {args.conf}")
    print(f"Display     : {args.display}")
    print(f"Save output : {args.save_output}")
    print(f"Tracking    : {not args.disable_tracking}")
    print(f"Output      : {args.output}")
    print("=" * 60)

    if not args.display and not args.save_output and args.max_frames == 0:
        print(
            "Warning: neither --display nor --save-output is enabled. "
            "Use --max-frames for a quick headless test."
        )

    model = load_yolo_model(args.weights)
    risk_config = load_risk_config()

    tracker = None
    if not args.disable_tracking:
        tracker = WorkerTracker(max_distance=args.tracking_max_distance)

    capture = open_video_source(args.source)

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video width/height.")

    writer = None
    if args.save_output:
        writer = get_video_writer(args.output, source_fps, width, height)

    frame_count = 0
    start_time = time.time()
    records: list[dict[str, Any]] = []

    try:
        while True:
            success, frame = capture.read()

            if not success:
                print("No more frames or failed to read frame.")
                break

            frame_count += 1

            processed_frame, frame_result = process_frame(
                frame=frame,
                model=model,
                conf=args.conf,
                risk_config=risk_config,
                imgsz=args.imgsz,
                device=args.device,
                tracker=tracker,
            )

            fps = calculate_fps(start_time, frame_count)
            processed_frame = draw_fps(processed_frame, fps)

            if args.save_output and writer is not None:
                writer.write(processed_frame)

            if args.save_log:
                records.append(
                    {
                        "frame_index": frame_count,
                        "fps": round(fps, 2),
                        **frame_result,
                    }
                )

            if args.display:
                cv2.imshow("Construction Safety Compliance", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopped by user.")
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        if args.display:
            cv2.destroyAllWindows()

    final_fps = calculate_fps(start_time, frame_count)

    print("=" * 60)
    print("Realtime Summary")
    print("=" * 60)
    print(f"Frames processed : {frame_count}")
    print(f"Average FPS      : {final_fps:.2f}")

    if args.save_output:
        print(f"Output video     : {args.output}")

    if args.save_log:
        log_dir = ensure_dir("results/logs")
        log_path = log_dir / f"realtime_log_{int(time.time())}.json"
        save_realtime_log(records, log_path)
        print(f"Frame log        : {log_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
