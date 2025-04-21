"""
Utility functions for the llama_demo project.
"""

from pathlib import Path


def find_project_root(marker="pyproject.toml") -> Path:
    """Find the project root directory by searching upwards for a marker file."""
    current_dir = Path(__file__).resolve().parent
    while True:
        if (current_dir / marker).exists():
            return current_dir
        parent_dir = current_dir.parent
        if parent_dir == current_dir:
            # Reached filesystem root
            raise FileNotFoundError(
                f"Project root marker '{marker}' not found starting from {Path(__file__).resolve().parent}"
            )
        current_dir = parent_dir


def get_default_data_dir() -> Path:
    """Get the default data directory path relative to the project root."""
    project_root = find_project_root()
    return project_root / "data"
