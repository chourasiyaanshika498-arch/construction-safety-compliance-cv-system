from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the absolute path of the project root."""
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root if it is not absolute."""
    path = Path(path)

    if path.is_absolute():
        return path

    return project_root() / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    file_path = resolve_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in: {file_path}")

    return data


def save_json(data: Any, path: str | Path) -> None:
    """Save data as a formatted JSON file."""
    file_path = resolve_path(path)
    ensure_dir(file_path.parent)

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_exists(path: str | Path) -> bool:
    """Return True if a file exists."""
    return resolve_path(path).is_file()


def get_timestamp() -> str:
    """Return a filesystem-friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
