from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import get_safe_device
from src.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO PPE model.")
    parser.add_argument("--weights", required=True, help="Path to trained weights.")
    parser.add_argument("--data", default="configs/ppe_yolov8.yaml", help="Dataset YAML config.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Evaluation split.")
    parser.add_argument("--imgsz", type=int, default=512, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, auto, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    safe_device = get_safe_device(args.device)

    print("=" * 70)
    print("YOLO PPE EVALUATION")
    print("=" * 70)
    print(f"Weights : {args.weights}")
    print(f"Data    : {data_path}")
    print(f"Split   : {args.split}")
    print(f"Image   : {args.imgsz}")
    print(f"Batch   : {args.batch}")
    print(f"Device  : {args.device} -> {safe_device}")
    print("=" * 70)

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=safe_device,
    )

    print("=" * 70)
    print("Evaluation Metrics")
    print("=" * 70)
    try:
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50   : {metrics.box.map50:.4f}")
        print(f"mAP75   : {metrics.box.map75:.4f}")
    except Exception:
        print(metrics)
    print("=" * 70)


if __name__ == "__main__":
    main()
