#!/usr/bin/env bash
set -Eeuo pipefail

# Formal pre-training repair script for construction-safety-compliance-cv-system.
# Run from the project root:
#   cd /workspace/construction-safety-compliance-cv-system
#   bash scripts/final_pre_training_repair.sh

PROJECT_ROOT="$(pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="backups/pre_training_repair_${STAMP}"
DATASET_DIR="/workspace/Personal Protective Equipment - Combined Model.v1i.yolov8"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_project_root() {
  [ -d "src" ] || fail "Missing src directory. Run this script from the project root."
  [ -d "scripts" ] || fail "Missing scripts directory. Run this script from the project root."
  [ -d "configs" ] || fail "Missing configs directory. Run this script from the project root."
  [ -d "app" ] || fail "Missing app directory. Run this script from the project root."
}

backup_file() {
  local file="$1"
  if [ -e "$file" ]; then
    mkdir -p "${BACKUP_DIR}/$(dirname "$file")"
    cp -a "$file" "${BACKUP_DIR}/${file}"
  fi
}

write_file() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  cat > "$path"
}

require_project_root
mkdir -p "$BACKUP_DIR" .streamlit scripts src configs docs reports results assets/sample_predictions backups

log "Creating a single backup folder: ${BACKUP_DIR}"
for f in \
  requirements.txt .gitignore .streamlit/config.toml README.md \
  src/detection.py src/train.py src/evaluate.py \
  scripts/gpu_preflight.py scripts/project_healthcheck.py \
  scripts/train_reasonable_gpu.sh scripts/run_website.sh \
  scripts/regenerate_visible_samples.py scripts/check_training_progress.sh \
  app/streamlit_app.py configs/ppe_yolov8.yaml configs/risk_rules.yaml
  do
    backup_file "$f"
  done

log "Moving scattered legacy backup files into one formal backup folder"
mkdir -p "${BACKUP_DIR}/legacy_files"
find app src -maxdepth 1 -type f \( -name '*before*.py' -o -name '*backup*.py' \) -print -exec mv {} "${BACKUP_DIR}/legacy_files/" \; 2>/dev/null || true
[ -f final_repair_and_training_setup.sh ] && mv final_repair_and_training_setup.sh "${BACKUP_DIR}/legacy_files/" || true

log "Writing formal requirements.txt"
write_file requirements.txt <<'EOF'
# Core application and model packages.
# Torch/Torchvision are intentionally not pinned here because the GPU Docker image already provides CUDA-enabled builds.
ultralytics
opencv-python
numpy
pandas
pyyaml
pillow
matplotlib
tqdm
pytest
streamlit
streamlit-webrtc
av
EOF

log "Writing formal .gitignore"
write_file .gitignore <<'EOF'
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.pytest_cache/
.env

# Large datasets and training outputs
runs/
results/**/*.mp4
results/**/*.avi
results/**/*.mov
results/**/*.mkv
results/**/*.json
reports/*.json
reports/*.csv
reports/*.md

# Model artifacts
*.pt
*.pth
*.onnx
*.engine
*.zip

# Local editor/system files
.DS_Store
.vscode/
EOF

log "Writing Streamlit configuration"
write_file .streamlit/config.toml <<'EOF'
[server]
headless = true
address = "0.0.0.0"
port = 8501
maxUploadSize = 1024
runOnSave = true

[browser]
gatherUsageStats = false
EOF

log "Writing dataset configuration"
write_file configs/ppe_yolov8.yaml <<'EOF'
path: /workspace/Personal Protective Equipment - Combined Model.v1i.yolov8

train: train/images
val: valid/images
test: test/images

nc: 14
names:
  - Fall-Detected
  - Gloves
  - Goggles
  - Hardhat
  - Ladder
  - Mask
  - NO-Gloves
  - NO-Goggles
  - NO-Hardhat
  - NO-Mask
  - NO-Safety Vest
  - Person
  - Safety Cone
  - Safety Vest
EOF

log "Writing risk rules"
write_file configs/risk_rules.yaml <<'EOF'
risk_weights:
  NO-Hardhat: 30
  NO-Safety Vest: 25
  NO-Mask: 15
  NO-Gloves: 10
  NO-Goggles: 10
  Fall-Detected: 100

risk_levels:
  safe:
    min: 0
    max: 0
  low:
    min: 1
    max: 25
  medium:
    min: 26
    max: 60
  high:
    min: 61
    max: 99
  critical:
    min: 100
    max: 999
EOF

log "Writing GPU preflight script"
write_file scripts/gpu_preflight.py <<'PY'
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_nvidia_smi() -> None:
    print("=" * 70)
    print("NVIDIA-SMI")
    print("=" * 70)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi was not found inside this container.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CUDA/GPU readiness before YOLO training.")
    parser.add_argument("--require-gpu", action="store_true", help="Exit with error if CUDA is not available.")
    parser.add_argument("--matmul", action="store_true", help="Run a small CUDA matrix multiplication test.")
    args = parser.parse_args()

    print("=" * 70)
    print("GPU PREFLIGHT CHECK")
    print("=" * 70)
    print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
    print("NVIDIA_DRIVER_CAPABILITIES:", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    run_nvidia_smi()

    try:
        import torch
    except Exception as error:
        print("Torch import failed:", repr(error))
        if args.require_gpu:
            raise SystemExit(1)
        return

    print("=" * 70)
    print("PYTORCH CUDA")
    print("=" * 70)
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        if args.matmul:
            x = torch.randn(2048, 2048, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print("GPU matmul OK:", tuple(y.shape))
        print("GPU status: READY")
        return

    print("GPU status: NOT READY")
    if args.require_gpu:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY

log "Writing safe detection module"
write_file src/detection.py <<'PY'
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
PY

log "Writing formal training module"
write_file src/train.py <<'PY'
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
PY

log "Writing safe evaluation module"
write_file src/evaluate.py <<'PY'
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
PY

log "Writing reasonable GPU training script"
write_file scripts/train_reasonable_gpu.sh <<'BASH2'
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="ppe_yolov8n_100e_reasonable"
LAST_CKPT="runs/train/${RUN_NAME}/weights/last.pt"
BEST_CKPT="runs/train/${RUN_NAME}/weights/best.pt"
DEVICE="${1:-0}"

python scripts/gpu_preflight.py --require-gpu --matmul
python scripts/check_dataset.py

if [ -f "$LAST_CKPT" ]; then
  echo "Found previous checkpoint: $LAST_CKPT"
  echo "Resuming interrupted training."
  python src/train.py \
    --model "$LAST_CKPT" \
    --data configs/ppe_yolov8.yaml \
    --epochs 100 \
    --imgsz 512 \
    --batch -1 \
    --device "$DEVICE" \
    --workers 8 \
    --patience 100 \
    --optimizer auto \
    --cos-lr \
    --close-mosaic 10 \
    --cache disk \
    --save-period 5 \
    --name "$RUN_NAME" \
    --resume \
    --require-gpu \
    --plots
else
  echo "Starting a new reasonable-time full training run."
  python src/train.py \
    --model yolov8n.pt \
    --data configs/ppe_yolov8.yaml \
    --epochs 100 \
    --imgsz 512 \
    --batch -1 \
    --device "$DEVICE" \
    --workers 8 \
    --patience 100 \
    --optimizer auto \
    --cos-lr \
    --close-mosaic 10 \
    --cache disk \
    --save-period 5 \
    --name "$RUN_NAME" \
    --require-gpu \
    --plots
fi

if [ -f "$BEST_CKPT" ]; then
  python src/evaluate.py --weights "$BEST_CKPT" --split val --imgsz 512 --batch 16 --device "$DEVICE" || true
  python src/evaluate.py --weights "$BEST_CKPT" --split test --imgsz 512 --batch 16 --device "$DEVICE" || true
  python scripts/regenerate_visible_samples.py --weights "$BEST_CKPT" --num-samples 12 --conf 0.25 --imgsz 512 --device "$DEVICE" || true
fi
BASH2
chmod +x scripts/train_reasonable_gpu.sh

log "Writing training progress monitor"
write_file scripts/check_training_progress.sh <<'BASH2'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RUN_NAME="${1:-ppe_yolov8n_100e_reasonable}"
watch -n 5 "nvidia-smi; echo; ls -lah runs/train/${RUN_NAME}/weights 2>/dev/null || true; echo; tail -n 5 runs/train/${RUN_NAME}/results.csv 2>/dev/null || true"
BASH2
chmod +x scripts/check_training_progress.sh

log "Writing formal healthcheck script"
write_file scripts/project_healthcheck.py <<'PY'
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(name: str, ok: bool) -> None:
    print(f"{'OK' if ok else 'FAIL'} - {name}")


def main() -> None:
    print("=" * 70)
    print("PROJECT HEALTHCHECK")
    print("=" * 70)

    required_files = [
        "app/streamlit_app.py",
        "src/detection.py",
        "src/train.py",
        "src/evaluate.py",
        "src/risk_scoring.py",
        "src/compliance_rules.py",
        "configs/ppe_yolov8.yaml",
        "configs/risk_rules.yaml",
        "scripts/train_reasonable_gpu.sh",
        "scripts/gpu_preflight.py",
    ]

    failed = False
    for rel in required_files:
        exists = (ROOT / rel).exists()
        status(rel, exists)
        failed = failed or not exists

    print("\nPython package availability:")
    for package in ["torch", "ultralytics", "cv2", "streamlit", "pandas", "yaml"]:
        ok = importlib.util.find_spec(package) is not None
        status(package, ok)
        failed = failed or not ok

    print("\nSyntax check:")
    files_to_compile = [
        "app/streamlit_app.py",
        "src/detection.py",
        "src/train.py",
        "src/evaluate.py",
        "scripts/gpu_preflight.py",
        "scripts/regenerate_visible_samples.py",
    ]
    result = subprocess.run([sys.executable, "-m", "py_compile", *files_to_compile], cwd=ROOT)
    status("py_compile", result.returncode == 0)
    failed = failed or result.returncode != 0

    print("\nDataset check:")
    dataset = subprocess.run([sys.executable, "scripts/check_dataset.py"], cwd=ROOT)
    status("dataset", dataset.returncode == 0)
    failed = failed or dataset.returncode != 0

    print("\nTest suite:")
    pytest = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    status("pytest", pytest.returncode == 0)
    failed = failed or pytest.returncode != 0

    print("=" * 70)
    if failed:
        raise SystemExit(1)
    print("Project healthcheck passed.")


if __name__ == "__main__":
    main()
PY

log "Writing visible sample regeneration script"
write_file scripts/regenerate_visible_samples.py <<'PY'
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
PY

log "Writing website runner"
write_file scripts/run_website.sh <<'BASH2'
#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
streamlit cache clear || true
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
BASH2
chmod +x scripts/run_website.sh

log "Patching Streamlit defaults without rewriting the full app"
python - <<'PY'
from pathlib import Path
import re

path = Path("app/streamlit_app.py")
text = path.read_text(encoding="utf-8")

text = text.replace('APP_VERSION = "WEBSITE_FINAL_REPAIR_V2"', 'APP_VERSION = "FORMAL_PRE_TRAINING_READY_V3"')
text = text.replace('APP_VERSION = "WEBSITE_REPAIR_STABLE_V1"', 'APP_VERSION = "FORMAL_PRE_TRAINING_READY_V3"')

text = re.sub(
    r'DEFAULT_WEIGHTS_CANDIDATES = \[.*?\]',
    '''DEFAULT_WEIGHTS_CANDIDATES = [
    "runs/train/ppe_yolov8n_100e_reasonable/weights/best.pt",
    "runs/train/ppe_yolov8n_100e_auto_batch/weights/best.pt",
    "runs/train/ppe_yolov8n_100e_fast/weights/best.pt",
    "runs/train/ppe_yolov8s_100e/weights/best.pt",
    "runs/detect/runs/train/test_ppe_yolov8n/weights/best.pt",
    "yolov8n.pt",
]''',
    text,
    flags=re.S,
)

text = text.replace('imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 768], index=3)',
                    'imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 768], index=2)')
text = text.replace('device = st.sidebar.text_input("Device", "auto")', 'device = st.sidebar.text_input("Device", "auto")')
text = text.replace('use_container_width=True', "width='stretch'")

path.write_text(text, encoding="utf-8")
PY

log "Writing formal README"
write_file README.md <<'EOF'
# Construction Safety Compliance CV System

A computer vision system for construction-site safety monitoring using YOLO object detection, PPE compliance rules, risk scoring, worker tracking, zone safety checks, fall alerts, reports, and a Streamlit dashboard.

## Main Features

- PPE and construction-object detection using YOLO.
- Hardhat, safety vest, mask, gloves, goggles, ladder, person, cone, and fall-detection classes.
- Rule-based compliance checking.
- Risk scoring with critical emergency handling for fall events.
- Image, camera snapshot, video upload, live camera, samples, and reports in one dashboard.
- GPU preflight checks before training.
- Reasonable-time training script with auto batch, AMP, checkpoint saving, and resume support.

## Dataset

The expected dataset path is:

```text
/workspace/Personal Protective Equipment - Combined Model.v1i.yolov8
```

The YOLO config file is:

```text
configs/ppe_yolov8.yaml
```

## Pre-Training Check

```bash
python scripts/gpu_preflight.py --require-gpu --matmul
python scripts/project_healthcheck.py
```

## Training

```bash
bash scripts/train_reasonable_gpu.sh 0
```

The main trained model will be saved to:

```text
runs/train/ppe_yolov8n_100e_reasonable/weights/best.pt
```

If training is interrupted, run the same command again. The script will resume from `last.pt` when available.

## Website

```bash
bash scripts/run_website.sh
```

Open:

```text
http://localhost:8501
```

## Important Note

This system is a safety-assistance prototype. It supports visual inspection and reporting, but it does not replace official site supervision or certified safety procedures.
EOF

log "Fixing permissions"
chmod +x scripts/*.sh || true

log "Checking package imports; installing only if required"
python - <<'PY' || {
import importlib.util
required = ["torch", "ultralytics", "cv2", "streamlit", "pandas", "yaml"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
print("Missing packages:", missing)
raise SystemExit(1 if missing else 0)
PY
  echo "Some packages are missing. Attempting safe Python package repair."
  python -m pip install --ignore-installed blinker==1.9.0 || true
  python -m pip install ultralytics streamlit streamlit-webrtc av pytest --upgrade-strategy only-if-needed
}

log "Running syntax checks"
python -m py_compile app/streamlit_app.py src/*.py scripts/*.py

log "Running GPU preflight check"
python scripts/gpu_preflight.py --require-gpu --matmul

log "Running dataset check"
python scripts/check_dataset.py

log "Running tests if pytest is available"
python -m pytest tests -q || true

log "Final status"
echo "Backup folder: ${BACKUP_DIR}"
echo "Recommended training command: bash scripts/train_reasonable_gpu.sh 0"
echo "Progress monitor command: bash scripts/check_training_progress.sh"
echo "Website command: bash scripts/run_website.sh"
echo "Pre-training repair completed successfully."
