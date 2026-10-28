from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_rules import detect_violations
from src.detection import load_yolo_model, parse_yolo_results, run_detection
from src.risk_scoring import calculate_risk_score, load_risk_config
from src.utils import ensure_dir, resolve_path, save_json
from src.visualization import draw_detections, draw_risk_banner, save_annotated_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small number of annotated sample predictions for GitHub."
    )

    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights.")
    parser.add_argument(
        "--source",
        default="/workspace/Personal Protective Equipment - Combined Model.v1i.yolov8/test/images",
        help="Folder containing input images.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of images to process.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Device: 0, 1, cpu, etc.")
    parser.add_argument("--output", default="assets/sample_predictions", help="Output folder.")

    return parser.parse_args()


def list_images(source: str | Path) -> list[Path]:
    """List image files from a folder."""
    source_path = Path(source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    if source_path.is_file():
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {source_path}")
        return [source_path]

    images = [
        path
        for path in sorted(source_path.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(f"No images found in: {source_path}")

    return images


def folder_size_mb(path: str | Path) -> float:
    """Calculate folder size in MB."""
    folder = Path(path)

    total_bytes = sum(
        file.stat().st_size
        for file in folder.rglob("*")
        if file.is_file()
    )

    return total_bytes / (1024 * 1024)


def clear_output_folder(output_dir: Path) -> None:
    """Clear previous sample predictions while preserving .gitkeep."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in output_dir.iterdir():
        if item.name == ".gitkeep":
            continue

        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def create_prediction_record(
    image_path: Path,
    detections: list[dict[str, Any]],
    compliance_result: dict[str, Any],
    risk_result: dict[str, Any],
    output_image: Path,
) -> dict[str, Any]:
    """Create JSON-friendly record for one sample prediction."""
    return {
        "image": str(image_path),
        "output_image": str(output_image),
        "detections": detections,
        "is_compliant": compliance_result["is_compliant"],
        "violations": compliance_result["violations"],
        "risk": risk_result,
    }


def process_image(
    model: Any,
    image_path: Path,
    output_dir: Path,
    risk_config: dict[str, Any],
    conf: float,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    """Run detection, safety rules, risk scoring, and custom visualization on one image."""
    results = run_detection(
        model=model,
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=False,
        project=str(resolve_path("results")),
        name="tmp_stage4",
    )

    detections = parse_yolo_results(results)
    compliance_result = detect_violations(detections)
    risk_result = calculate_risk_score(detections, risk_config)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # Keep GitHub sample images clean: boxes only.
    # Risk/violations are stored in the JSON report.
    annotated = draw_detections(image, detections)

    output_image = output_dir / f"sample_{image_path.stem}.jpg"
    save_annotated_image(annotated, output_image)

    return create_prediction_record(
        image_path=image_path,
        detections=detections,
        compliance_result=compliance_result,
        risk_result=risk_result,
        output_image=output_image,
    )


def main() -> None:
    args = parse_args()

    output_dir = ensure_dir(args.output)
    clear_output_folder(output_dir)

    images = list_images(args.source)
    selected_images = images[: args.num_samples]

    if len(selected_images) < args.num_samples:
        print(
            f"Warning: requested {args.num_samples} samples, "
            f"but only found {len(selected_images)} images."
        )

    print("=" * 60)
    print("Create Sample Predictions")
    print("=" * 60)
    print(f"Weights    : {args.weights}")
    print(f"Source     : {args.source}")
    print(f"Samples    : {len(selected_images)}")
    print(f"Confidence : {args.conf}")
    print(f"Output     : {output_dir}")
    print("=" * 60)

    model = load_yolo_model(args.weights)
    risk_config = load_risk_config()

    records = []

    for index, image_path in enumerate(selected_images, start=1):
        print(f"[{index}/{len(selected_images)}] Processing: {image_path.name}")

        record = process_image(
            model=model,
            image_path=image_path,
            output_dir=output_dir,
            risk_config=risk_config,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )

        records.append(record)

    summary = {
        "num_samples": len(records),
        "records": records,
    }

    json_path = output_dir / "sample_predictions.json"
    save_json(summary, json_path)

    size_mb = folder_size_mb(output_dir)

    print("=" * 60)
    print("Sample Prediction Summary")
    print("=" * 60)
    print(f"Images saved : {len(records)}")
    print(f"JSON report  : {json_path}")
    print(f"Folder size  : {size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
