#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RUN_NAME="${1:-ppe_yolov8n_100e_reasonable}"
watch -n 5 "nvidia-smi; echo; ls -lah runs/train/${RUN_NAME}/weights 2>/dev/null || true; echo; tail -n 5 runs/train/${RUN_NAME}/results.csv 2>/dev/null || true"
