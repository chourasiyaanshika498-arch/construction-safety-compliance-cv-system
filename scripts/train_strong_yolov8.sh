#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA="configs/ppe_yolov8.yaml"
DEVICE="${1:-auto}"

python scripts/check_dataset.py

python src/train.py \
  --model yolov8s.pt \
  --data "$DATA" \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device "$DEVICE" \
  --workers 4 \
  --patience 30 \
  --optimizer auto \
  --cos-lr \
  --close-mosaic 10 \
  --name ppe_yolov8s_100e \
  --plots

python src/evaluate.py \
  --weights runs/train/ppe_yolov8s_100e/weights/best.pt \
  --split val || true

python src/evaluate.py \
  --weights runs/train/ppe_yolov8s_100e/weights/best.pt \
  --split test || true

python scripts/regenerate_visible_samples.py \
  --weights runs/train/ppe_yolov8s_100e/weights/best.pt \
  --num-samples 12 \
  --conf 0.25 \
  --device "$DEVICE"
