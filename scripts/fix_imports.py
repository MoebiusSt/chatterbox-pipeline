#!/usr/bin/env python3
"""
Script to automatically fix import sorting in all Python files using isort.
"""

import subprocess
from pathlib import Path


def main():
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Run isort on src and scripts directories
    subprocess.run(
        [
            "isort",
            "--profile",
            "black",
            "--multi-line",
            "3",
            "--line-length",
            "88",
            "--src",
            "src",
            str(project_root / "src"),
            str(project_root / "scripts"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
