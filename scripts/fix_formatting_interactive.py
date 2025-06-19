#!/usr/bin/env python3
"""
Interactive script to fix code formatting in Python files using black.
Shows diff and asks for confirmation before applying changes to each file.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def get_python_files(directory: Path) -> List[Path]:
    """Get all Python files in directory recursively."""
    return list(directory.glob("**/*.py"))


def show_diff(file_path: Path) -> Tuple[int, str]:
    """Show diff of what black would change."""
    result = subprocess.run(
        [
            "black",
            "--diff",
            "--line-length",
            "88",
            "--target-version",
            "py312",
            str(file_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


def apply_black(file_path: Path) -> bool:
    """Apply black formatting to file."""
    result = subprocess.run(
        ["black", "--line-length", "88", "--target-version", "py312", str(file_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Get all Python files
    src_files = get_python_files(project_root / "src")
    script_files = get_python_files(project_root / "scripts")
    all_files = src_files + script_files

    print(f"Found {len(all_files)} Python files to check.")

    # Process each file
    for file_path in all_files:
        print(f"\nChecking {file_path.relative_to(project_root)}...")

        # Show diff
        returncode, diff = show_diff(file_path)

        if returncode == 0:
            print("No changes needed.")
            continue

        if not diff:
            print("No changes needed.")
            continue

        print("\nChanges that would be made:")
        print("=" * 80)
        print(diff)
        print("=" * 80)

        # Ask for confirmation
        while True:
            response = input("\nApply these changes? [y/n/s/q] ").lower()
            if response in ["y", "n", "s", "q"]:
                break
            print("Please enter y (yes), n (no), s (skip remaining), or q (quit)")

        if response == "q":
            print("\nExiting...")
            sys.exit(0)
        elif response == "s":
            print("\nSkipping remaining files...")
            break
        elif response == "y":
            if apply_black(file_path):
                print("Changes applied successfully.")
            else:
                print("Error applying changes.")
        else:  # response == 'n'
            print("Skipping this file.")


if __name__ == "__main__":
    main()
