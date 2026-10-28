from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_safe_device(device: str | int = "auto") -> str:
    """Resolve a safe Ultralytics device without crashing the website."""
    requested = str(device).strip().lower()

    if requested in {"", "auto"}:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested in {"0", "cuda", "cuda:0"}:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    return requested


def load_yolo_model(weights_path: str | Path) -> YOLO:
    """Load a YOLO model from a local or pretrained weights path."""
    return YOLO(str(weights_path))


def run_detection(
    model: YOLO,
    source: Any,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | int = "auto",
    save: bool = False,
    project: str | Path = "results",
    name: str = "predictions",
) -> list[Any]:
    """Run YOLO prediction with CPU fallback for website stability."""
    safe_device = get_safe_device(device)

    try:
        return model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            device=safe_device,
            save=save,
            project=str(project),
            name=name,
            exist_ok=True,
            verbose=False,
        )
    except ValueError as error:
        message = str(error)
        if "Invalid CUDA" in message or "device=0" in message or "CUDA" in message:
            print("WARNING: CUDA prediction failed. Retrying on CPU for application stability.")
            return model.predict(
                source=source,
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                save=save,
                project=str(project),
                name=name,
                exist_ok=True,
                verbose=False,
            )
        raise


def parse_yolo_results(results: list[Any]) -> list[dict[str, Any]]:
    """Convert Ultralytics results into stable dictionaries used by the project."""
    detections: list[dict[str, Any]] = []

    for result in results:
        names = result.names
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append(
                {
                    "source": Path(str(result.path)).name if getattr(result, "path", None) else "image",
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(float(value), 2) for value in xyxy],
                }
            )

    return detections
