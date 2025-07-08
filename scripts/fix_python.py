#!/usr/bin/env python3
"""
Comprehensive Python code fixing script.
Runs myPy, Black, isort, and other code quality checks with interactive options.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class PythonCodeFixer:
    def __init__(self, project_root: Path, auto_fix: bool = False):
        self.project_root = project_root
        self.auto_fix = auto_fix
        self.src_dir = project_root / "src"
        self.scripts_dir = project_root / "scripts"

    def get_python_files(self) -> List[Path]:
        """Get all Python files in src and scripts directories."""
        src_files = list(self.src_dir.glob("**/*.py")) if self.src_dir.exists() else []
        script_files = (
            list(self.scripts_dir.glob("**/*.py")) if self.scripts_dir.exists() else []
        )
        return src_files + script_files

    def run_mypy(self) -> Tuple[bool, str]:
        """Run myPy type checking."""
        print("\n🔍 Running myPy type checking...")

        # Use mypy.ini configuration to avoid path conflicts
        config_file = self.project_root / "mypy.ini"

        result = subprocess.run(
            [
                "mypy",
                str(self.src_dir),
                f"--config-file={config_file}",
            ],
            capture_output=True,
            text=True,
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print("✅ MyPy: No type errors found")
        else:
            print("❌ MyPy: Type errors found")
            print("=" * 80)
            print(output)
            print("=" * 80)

        return success, output

    def run_syntax_check(self) -> Tuple[bool, List[str]]:
        """Check Python syntax for all files."""
        print("\n🔍 Checking Python syntax...")

        errors = []
        files = self.get_python_files()

        for file_path in files:
            result = subprocess.run(
                ["python", "-m", "py_compile", str(file_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                error_msg = f"{file_path.relative_to(self.project_root)}: {result.stderr.strip()}"
                errors.append(error_msg)

        if not errors:
            print(f"✅ Syntax: All {len(files)} files are syntactically correct")
        else:
            print("❌ Syntax errors found:")
            for error in errors:
                print(f"  {error}")

        return len(errors) == 0, errors

    def run_black_check(self) -> Tuple[bool, Dict[str, str]]:
        """Check Black formatting."""
        print("\n🔍 Checking Black formatting...")

        changes = {}
        files = self.get_python_files()

        for file_path in files:
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

            if result.returncode != 0 and result.stdout:
                changes[str(file_path.relative_to(self.project_root))] = result.stdout

        if not changes:
            print(f"✅ Black: All {len(files)} files are properly formatted")
        else:
            print(f"❌ Black: {len(changes)} files need formatting")

        return len(changes) == 0, changes

    def apply_black_formatting(self, files_to_fix: List[str] = None) -> bool:
        """Apply Black formatting."""
        print("\n🔧 Applying Black formatting...")

        if files_to_fix is None:
            target_dirs = [str(self.src_dir), str(self.scripts_dir)]
        else:
            target_dirs = [str(self.project_root / f) for f in files_to_fix]

        for target in target_dirs:
            if Path(target).exists():
                result = subprocess.run(
                    [
                        "black",
                        "--line-length",
                        "88",
                        "--target-version",
                        "py312",
                        target,
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"❌ Failed to format {target}: {result.stderr}")
                    return False

        print("✅ Black formatting applied")
        return True

    def run_isort_check(self) -> Tuple[bool, Dict[str, str]]:
        """Check import sorting."""
        print("\n🔍 Checking import sorting...")

        changes = {}
        files = self.get_python_files()

        for file_path in files:
            result = subprocess.run(
                [
                    "isort",
                    "--diff",
                    "--check-only",
                    "--profile",
                    "black",
                    "--line-length",
                    "88",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0 and result.stdout:
                changes[str(file_path.relative_to(self.project_root))] = result.stdout

        if not changes:
            print(f"✅ isort: All {len(files)} files have properly sorted imports")
        else:
            print(f"❌ isort: {len(changes)} files need import sorting")

        return len(changes) == 0, changes

    def apply_isort_formatting(self, files_to_fix: List[str] = None) -> bool:
        """Apply import sorting."""
        print("\n🔧 Applying import sorting...")

        if files_to_fix is None:
            target_dirs = [str(self.src_dir), str(self.scripts_dir)]
        else:
            target_dirs = [str(self.project_root / f) for f in files_to_fix]

        for target in target_dirs:
            if Path(target).exists():
                result = subprocess.run(
                    ["isort", "--profile", "black", "--line-length", "88", target],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    print(f"❌ Failed to sort imports in {target}: {result.stderr}")
                    return False

        print("✅ Import sorting applied")
        return True

    def run_flake8_check(self) -> Tuple[bool, str]:
        """Run flake8 linting (optional)."""
        print("\n🔍 Running flake8 linting...")

        # Check if flake8 is available
        try:
            subprocess.run(["flake8", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  flake8 not available, skipping")
            return True, ""

        result = subprocess.run(
            [
                "flake8",
                str(self.src_dir),
                str(self.scripts_dir),
                "--max-line-length=120",  # More lenient line length
                "--extend-ignore=E203,W503,E501",  # Black compatibility + ignore line length
            ],
            capture_output=True,
            text=True,
        )

        success = result.returncode == 0
        output = result.stdout + result.stderr

        if success:
            print("✅ flake8: No linting issues found")
        else:
            print("❌ flake8: Linting issues found")
            print("=" * 80)
            print(output)
            print("=" * 80)

        return success, output

    def interactive_fix(self):
        """Run interactive fixing process."""
        print(f"🐍 Python Code Fixer - Project: {self.project_root.name}")
        print("=" * 50)
        print(
            "⚠️  SAFE MODE: Only formatting fixes (Black) will be applied automatically"
        )
        print("⚠️  Import sorting and other changes require manual confirmation")
        print("=" * 50)

        # Step 1: Syntax check (must pass first)
        syntax_ok, syntax_errors = self.run_syntax_check()
        if not syntax_ok:
            print("\n❌ CRITICAL: Syntax errors must be fixed manually first!")
            print("The following files have syntax errors:")
            for error in syntax_errors:
                print(f"  {error}")
            return False

        # Step 2: MyPy check
        mypy_ok, mypy_output = self.run_mypy()

        # Step 3: Black formatting check (SAFE - only formatting)
        black_ok, black_changes = self.run_black_check()
        if not black_ok:
            if not self.auto_fix:
                print("\n📝 Black formatting issues found:")
                for file_path, diff in black_changes.items():
                    print(f"\n{file_path}:")
                    print(diff[:500] + "..." if len(diff) > 500 else diff)

                if self._ask_user("Apply Black formatting? (SAFE - only code style)"):
                    self.apply_black_formatting()
                    black_ok = True
            else:
                print("\n🔧 Auto-applying Black formatting (safe)...")
                self.apply_black_formatting()
                black_ok = True

        # Step 4: Import sorting check (POTENTIALLY RISKY)
        isort_ok, isort_changes = self.run_isort_check()
        if not isort_ok:
            print("\n📦 Import sorting issues found:")
            for file_path, diff in isort_changes.items():
                print(f"\n{file_path}:")
                print(diff[:300] + "..." if len(diff) > 300 else diff)

            print(
                "\n⚠️  WARNING: Import reordering can potentially break code in rare cases"
            )
            print("   (e.g., if imports have side effects or circular dependencies)")

            if not self.auto_fix:
                if self._ask_user("Apply import sorting? (CAUTION: Review changes!)"):
                    self.apply_isort_formatting()
                    isort_ok = True
            else:
                print("🚫 Skipping import sorting in auto-fix mode for safety")

        # Step 5: flake8 check (informational only - NO AUTO-FIX)
        flake8_ok, flake8_output = self.run_flake8_check()
        if not flake8_ok:
            print("\n📋 MANUAL REVIEW REQUIRED:")
            print("   - F401 (unused imports): Remove carefully, may break code")
            print("   - E402 (imports not at top): Often intentional, check context")
            print("   - F811 (redefinition): Needs code restructuring")
            print("   - F821 (undefined name): Fix undefined variables")
            print("   - Other issues: Review and fix manually")

        # Summary
        print("\n📊 SUMMARY")
        print("=" * 50)
        print(f"✅ Syntax:     {'PASS' if syntax_ok else 'FAIL'}")
        print(
            f"{'✅' if mypy_ok else '❌'} MyPy:       {'PASS' if mypy_ok else 'FAIL'}"
        )
        print(f"✅ Black:      {'PASS' if black_ok else 'FAIL'}")
        print(
            f"{'✅' if isort_ok else '⚠️ '} isort:      {'PASS' if isort_ok else 'NEEDS REVIEW'}"
        )
        print(
            f"{'✅' if flake8_ok else '⚠️ '} flake8:     {'PASS' if flake8_ok else 'NEEDS MANUAL FIX'}"
        )

        # Only Black is critical for safe auto-fixing
        safe_critical_ok = syntax_ok and black_ok
        print(f"\n🎯 Safe Auto-Fix: {'COMPLETE' if safe_critical_ok else 'NEEDS WORK'}")
        if not isort_ok or not flake8_ok:
            print("⚠️  Manual review recommended for import sorting and linting issues")

        return safe_critical_ok

    def _ask_user(self, question: str) -> bool:
        """Ask user a yes/no question."""
        while True:
            response = input(f"\n{question} [y/n]: ").lower().strip()
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print("Please enter 'y' or 'n'")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Python code fixing")
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically apply fixes without asking",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="Only check, don't apply any fixes"
    )
    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).resolve().parent.parent

    # Initialize fixer
    fixer = PythonCodeFixer(project_root, auto_fix=args.auto_fix)

    if args.check_only:
        # Run all checks without fixing
        syntax_ok, _ = fixer.run_syntax_check()
        mypy_ok, _ = fixer.run_mypy()
        black_ok, _ = fixer.run_black_check()
        isort_ok, _ = fixer.run_isort_check()
        flake8_ok, _ = fixer.run_flake8_check()

        all_ok = syntax_ok and mypy_ok and black_ok and isort_ok
        sys.exit(0 if all_ok else 1)
    else:
        # Run interactive fixing
        success = fixer.interactive_fix()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
