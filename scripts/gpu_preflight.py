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
