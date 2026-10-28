import platform
import sys
from importlib.metadata import version, PackageNotFoundError


def get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "Not installed"


def main() -> None:
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")

    torch_version = get_package_version("torch")
    ultralytics_version = get_package_version("ultralytics")

    print(f"PyTorch version: {torch_version}")
    print(f"Ultralytics version: {ultralytics_version}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version used by PyTorch: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")

            conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
            x = torch.randn(4, 3, 640, 640, device="cuda")
            y = conv(x)
            torch.cuda.synchronize()

            print("cuDNN Conv2D test: OK")
            print(f"Conv2D output shape: {tuple(y.shape)}")
        else:
            print("GPU test skipped because CUDA is not available.")

    except Exception as error:
        print("GPU / PyTorch test failed.")
        print(f"Error: {error}")

    print("=" * 60)


if __name__ == "__main__":
    main()
