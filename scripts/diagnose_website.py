from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
print(f"Project root: {ROOT}")

required_paths = [
    "app/streamlit_app.py",
    "src/detection.py",
    "src/train.py",
    "src/risk_scoring.py",
    "src/compliance_rules.py",
    "configs/ppe_yolov8.yaml",
    "configs/risk_rules.yaml",
]

print("\nRequired files:")
for rel in required_paths:
    path = ROOT / rel
    print(f"- {rel}: {'OK' if path.exists() else 'MISSING'}")

weights_candidates = [
    "runs/train/ppe_yolov8s_100e/weights/best.pt",
    "runs/detect/runs/train/test_ppe_yolov8n/weights/best.pt",
    "runs/train/test_ppe_yolov8n/weights/best.pt",
    "yolov8n.pt",
]

print("\nWeights candidates:")
for rel in weights_candidates:
    print(f"- {rel}: {'OK' if (ROOT / rel).exists() else 'missing'}")

packages = ["streamlit", "cv2", "numpy", "pandas", "ultralytics", "streamlit_webrtc", "av", "torch"]
print("\nPackages:")
for package in packages:
    print(f"- {package}: {'OK' if importlib.util.find_spec(package) else 'MISSING'}")

try:
    import torch
    print("\nTorch/CUDA:")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA count:", torch.cuda.device_count())
    print("Torch CUDA version:", torch.version.cuda)
except Exception as error:
    print("Torch check failed:", error)

print("\nPython:")
print(sys.version)
