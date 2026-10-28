from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import load_yolo_model, normalize_source, parse_yolo_results, run_detection
from src.utils import ensure_dir, get_timestamp, resolve_path, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 PPE prediction on image, folder, video, or webcam."
    )

    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument("--source", required=True, help="Image, folder, video path, or webcam index.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Device: 0, 1, cpu, etc.")
    parser.add_argument("--save-json", action="store_true", help="Save detections as JSON.")
    parser.add_argument("--project", default="results", help="Prediction output directory.")
    parser.add_argument("--name", default="predictions", help="Prediction run name.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_path = resolve_path(args.project)
    output_dir = ensure_dir(project_path / args.name)

    print("=" * 60)
    print("YOLOv8 Prediction")
    print("=" * 60)
    print(f"Weights : {args.weights}")
    print(f"Source  : {args.source}")
    print(f"Conf    : {args.conf}")
    print(f"Image   : {args.imgsz}")
    print(f"Device  : {args.device}")
    print(f"Output  : {output_dir}")
    print("=" * 60)

    model = load_yolo_model(args.weights)
    source = normalize_source(args.source)

    results = run_detection(
        model=model,
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(project_path),
        name=args.name,
    )

    detections = parse_yolo_results(results)

    print("=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    print(f"Total detections: {len(detections)}")

    if args.save_json:
        json_path = output_dir / f"detections_{get_timestamp()}.json"
        save_json(detections, json_path)
        print(f"Detections JSON saved to: {json_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
