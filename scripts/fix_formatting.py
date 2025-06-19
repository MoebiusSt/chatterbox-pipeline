#!/usr/bin/env python3
"""
Script to automatically fix code formatting in all Python files using black.
"""

import subprocess
from pathlib import Path


def main():
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Run black on src and scripts directories
    subprocess.run(
        [
            "black",
            "--line-length",
            "88",
            "--target-version",
            "py312",
            str(project_root / "src"),
            str(project_root / "scripts"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
