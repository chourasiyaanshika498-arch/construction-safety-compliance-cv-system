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
