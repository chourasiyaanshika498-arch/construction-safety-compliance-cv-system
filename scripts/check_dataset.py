from pathlib import Path
import sys
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML file: {path}")

    return data


def count_files(folder: Path, extensions: set[str]) -> int:
    if not folder.exists():
        raise FileNotFoundError(f"Required folder not found: {folder}")

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in extensions
    )


def validate_config(config: dict) -> None:
    required_keys = ["path", "train", "val", "test", "nc", "names"]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key in dataset config: {key}")

    if config["nc"] != len(config["names"]):
        raise ValueError(
            f"nc mismatch: nc={config['nc']} but names={len(config['names'])}"
        )


def check_split(dataset_root: Path, split: str) -> dict:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    image_count = count_files(image_dir, IMAGE_EXTENSIONS)
    label_count = count_files(label_dir, {".txt"})

    return {
        "split": split,
        "images": image_count,
        "labels": label_count,
        "status": "OK" if image_count == label_count else "ERROR",
    }


def print_classes(names: list[str]) -> None:
    print("Class Names:")
    for idx, name in enumerate(names):
        print(f"  {idx}: {name}")
    print()


def main() -> int:
    try:
        config_path = project_root() / "configs" / "ppe_yolov8.yaml"
        config = load_yaml(config_path)
        validate_config(config)

        dataset_root = Path(config["path"]).resolve()

        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

        names = config["names"]

        split_results = [
            check_split(dataset_root, "train"),
            check_split(dataset_root, "valid"),
            check_split(dataset_root, "test"),
        ]

        print_classes(names)

        print("=" * 60)
        print("Dataset Check")
        print("=" * 60)
        print(f"Dataset path: {dataset_root}")
        print(f"Classes: {len(names)}")
        print("Split Summary:")

        for result in split_results:
            print(
                f"{result['split']:<5}: "
                f"{result['images']} images / "
                f"{result['labels']} labels / "
                f"{result['status']}"
            )

        all_ok = all(result["status"] == "OK" for result in split_results)

        print(f"Dataset status: {'OK' if all_ok else 'ERROR'}")
        print("=" * 60)

        return 0 if all_ok else 1

    except Exception as e:
        print("=" * 60)
        print("Dataset Check Failed")
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
