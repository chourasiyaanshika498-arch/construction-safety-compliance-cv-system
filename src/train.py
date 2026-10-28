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
    parser = argparse.ArgumentParser(description="Train a YOLO PPE detection model with GPU-safe defaults.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--data", default="configs/ppe_yolov8.yaml", help="Dataset YAML config.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size.")
    parser.add_argument("--batch", type=float, default=-1, help="Batch size. Use -1 for Ultralytics auto-batch.")
    parser.add_argument("--device", default="0", help="Device: 0, cpu, auto, etc.")
    parser.add_argument("--project", default="runs/train", help="Training output directory.")
    parser.add_argument("--name", default="ppe_yolov8n_100e_reasonable", help="Training run name.")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer setting.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR schedule.")
    parser.add_argument("--cache", default="disk", help="False, ram, or disk.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for the final N epochs.")
    parser.add_argument("--save-period", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for experiments. Use 1.0 for full training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument("--plots", action="store_true", help="Save plots.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision AMP.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA is not available.")
    return parser.parse_args()


def normalize_batch(batch: float) -> int | float:
    if batch == -1:
        return -1
    if float(batch).is_integer():
        return int(batch)
    return batch


def normalize_cache(cache: str) -> bool | str:
    value = str(cache).strip().lower()
    if value in {"false", "0", "none", "no"}:
        return False
    if value in {"true", "1", "yes"}:
        return True
    return cache


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)
    safe_device = get_safe_device(args.device)

    if args.require_gpu and safe_device == "cpu":
        raise SystemExit("GPU is required, but CUDA is not available in this container.")

    print("=" * 80)
    print("FORMAL YOLO PPE TRAINING")
    print("=" * 80)
    print(f"Model       : {args.model}")
    print(f"Data        : {data_path}")
    print(f"Epochs      : {args.epochs}")
    print(f"Image size  : {args.imgsz}")
    print(f"Batch       : {args.batch}")
    print(f"Device      : {args.device} -> {safe_device}")
    print(f"Workers     : {args.workers}")
    print(f"Cache       : {args.cache}")
    print(f"AMP         : {not args.no_amp}")
    print(f"Save period : {args.save_period}")
    print(f"Output      : {project_path / args.name}")
    print("=" * 80)

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=normalize_batch(args.batch),
        device=safe_device,
        project=str(project_path),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        cache=normalize_cache(args.cache),
        seed=args.seed,
        close_mosaic=args.close_mosaic,
        save_period=args.save_period,
        fraction=args.fraction,
        resume=args.resume,
        plots=args.plots,
        amp=not args.no_amp,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
