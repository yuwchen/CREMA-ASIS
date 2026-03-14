"""File system and configuration I/O helpers."""

import os
from pathlib import Path

import yaml


def get_all_files(rootdir: str, suffix: str = "") -> list:
    """Recursively collect all files under *rootdir* that end with *suffix*.

    Args:
        rootdir: Root directory to walk.
        suffix: File extension filter (e.g. ``".wav"``).  Empty matches all.

    Returns:
        Sorted list of absolute file paths.
    """
    filelist = []
    for subdir, _dirs, files in os.walk(rootdir):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(suffix):
                filelist.append(filepath)
    return sorted(filelist)


def load_yaml(path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to ``.yaml`` file.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_prompt(path: str) -> str:
    """Load a text prompt file.

    Args:
        path: Path to ``.txt`` prompt file.

    Returns:
        Prompt string with leading/trailing whitespace stripped.
    """
    with open(path, "r") as f:
        return f.read().strip()


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent.parent
