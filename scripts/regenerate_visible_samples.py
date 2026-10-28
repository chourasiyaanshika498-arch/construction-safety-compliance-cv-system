from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection import load_yolo_model, parse_yolo_results, run_detection
from src.visualization import draw_detections

DEFAULT_SOURCE = "/workspace/Personal Protective Equipment - Combined Model.v1i.yolov8/test/images"
DEFAULT_WEIGHTS = "runs/train/ppe_yolov8n_100e_reasonable/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate website samples using real model predictions.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default="assets/sample_predictions")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    source = Path(args.source)
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source images folder not found: {source}")

    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("sample_*.jpg"):
        old_file.unlink()

    json_path = output / "sample_predictions.json"
    if json_path.exists():
        json_path.unlink()

    image_paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(sorted(source.glob(ext)))

    model = load_yolo_model(weights)
    saved: list[dict] = []
    checked = 0

    for image_path in image_paths:
        if len(saved) >= args.num_samples:
            break
        checked += 1
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        results = run_detection(
            model=model,
            source=image,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            project=PROJECT_ROOT / "results",
            name="sample_tmp",
        )
        detections = parse_yolo_results(results)
        if not detections:
            continue
        annotated = draw_detections(image, detections)
        out_name = f"sample_{len(saved)+1:02d}_{image_path.name}"
        out_path = output / out_name
        cv2.imwrite(str(out_path), annotated)
        saved.append(
            {
                "image": str(out_path.relative_to(PROJECT_ROOT)),
                "source": str(image_path),
                "num_detections": len(detections),
                "detections": detections,
            }
        )
        print(f"Saved {out_path.name} with {len(detections)} detections")

    (output / ".gitkeep").touch()
    json_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 70)
    print(f"Checked images : {checked}")
    print(f"Saved samples  : {len(saved)}")
    print(f"Output folder  : {output}")
    print(f"JSON report    : {json_path}")
    print("=" * 70)

    if not saved:
        raise SystemExit("No detections found. Try lowering --conf to 0.15 or train the model for more epochs.")


if __name__ == "__main__":
    main()
