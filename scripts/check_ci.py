#!/usr/bin/env python3
"""
Check CI status for commits.
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
# or use PYTHONPATH=$(pwd)/src when running this script

import requests

from src.git_ci_config.github_config import check_ci_status


def extract_error_details(log_text: str) -> str:
    """Extract relevant error messages and code references from log text."""
    error_lines = []

    # Common error patterns
    patterns = [
        r"Error:.*",
        r"Exception:.*",
        r"Failed:.*",
        r'File ".*", line \d+',
        r"Traceback.*",
        r"AssertionError:.*",
        r"TypeError:.*",
        r"ValueError:.*",
        r"ImportError:.*",
        r"ModuleNotFoundError:.*",
        r"AttributeError:.*",
        r"KeyError:.*",
        r"IndexError:.*",
        r"RuntimeError:.*",
        r"OSError:.*",
        r"PermissionError:.*",
        r"FileNotFoundError:.*",
    ]

    # Test-specific patterns
    test_patterns = [
        r"=+ ERRORS =+",
        r"=+ FAILURES =+",
        r"=+ short test summary info =+",
        r"ERROR collecting.*",
        r"FAILED.*",
        r"ImportError while importing test module.*",
        r"Hint:.*",
        r"Traceback:.*",
        r"Interrupted:.*",
        r"\d+ error.*",
        r"\d+ failure.*",
    ]

    # Combine patterns
    pattern = "|".join(f"({p})" for p in patterns + test_patterns)

    # Find all matches
    lines = log_text.split("\n")
    in_error_section = False
    error_section_lines = []

    for line in lines:
        # Check for error section markers
        if re.search(r"=+ ERRORS =+|=+ FAILURES =+", line):
            in_error_section = True
            error_section_lines.append(line)
            continue

        if in_error_section:
            if re.search(r"=+ short test summary info =+", line):
                in_error_section = False
            else:
                error_section_lines.append(line)
                continue

        # Check for individual error patterns
        if re.search(pattern, line):
            error_lines.append(line.strip())

    # Combine error section and individual errors
    if error_section_lines:
        return "\n".join(error_section_lines)
    return "\n".join(error_lines) if error_lines else "No specific error details found"


def get_job_logs(run_id: int, token: str) -> dict:
    """Get detailed job logs for a workflow run."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get jobs
    jobs_url = f"https://api.github.com/repos/MoebiusSt/tts_pipeline_enhanced/actions/runs/{run_id}/jobs"
    response = requests.get(jobs_url, headers=headers)
    response.raise_for_status()
    jobs = response.json()["jobs"]

    # Get logs for each job
    job_logs = {}
    for job in jobs:
        job_id = job["id"]
        logs_url = f"https://api.github.com/repos/MoebiusSt/tts_pipeline_enhanced/actions/jobs/{job_id}/logs"
        log_response = requests.get(logs_url, headers=headers)
        if log_response.status_code == 200:
            # Only store error details for failed jobs
            if job["conclusion"] == "failure":
                job_logs[job["name"]] = {
                    "status": job["status"],
                    "conclusion": job["conclusion"],
                    "error_details": extract_error_details(log_response.text),
                }
            else:
                job_logs[job["name"]] = {
                    "status": job["status"],
                    "conclusion": job["conclusion"],
                }

    return job_logs


def format_duration(seconds: int) -> str:
    """Format duration in seconds to human readable string."""
    return str(timedelta(seconds=seconds))


def get_commit_sha(commit_ref: str = None) -> str:
    """Get commit SHA from reference or latest commit."""
    import subprocess

    if commit_ref:
        # Get specific commit
        result = subprocess.run(
            ["git", "rev-parse", commit_ref], capture_output=True, text=True
        )
    else:
        # Get latest commit
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )

    if result.returncode != 0:
        print(f"Error: Could not get commit SHA for {commit_ref or 'HEAD'}")
        sys.exit(1)

    return result.stdout.strip()


def check_workflow_status(
    commit_sha: str, token: str, log_file: Path = None, is_first_write: bool = True
):
    """Check workflow status for a commit."""
    try:
        status = check_ci_status(commit_sha)
        if not status:
            output = f"No CI workflow found for commit {commit_sha}"
            if log_file:
                mode = "w" if is_first_write else "a"
                with open(log_file, mode) as f:
                    f.write(f"{output}\n")
            else:
                print(output)
            return

        # Print or write status
        output = f"\nCI Status for commit {commit_sha}:\n"
        output += f"Workflow: {status['name']}\n"
        output += f"Status: {status['status']}\n"
        output += f"Conclusion: {status['conclusion']}\n"

        if log_file:
            mode = "w" if is_first_write else "a"
            with open(log_file, mode) as f:
                f.write(output)
        else:
            print(output)

        # If workflow is complete, get logs
        if status["status"] == "completed":
            # Extract run ID from URL
            run_id = status["html_url"].split("/")[-1]

            # Get detailed job logs
            if log_file:
                with open(log_file, "a") as f:
                    f.write("\nFetching job logs...\n")
            else:
                print("\nFetching job logs...")

            job_logs = get_job_logs(run_id, token)

            # Print or write job results
            output = "\nJob Results:\n"
            for job_name, job_info in job_logs.items():
                output += f"\n{job_name}:\n"
                output += f"Status: {job_info['status']}\n"
                output += f"Conclusion: {job_info['conclusion']}\n"
                if job_info["conclusion"] == "failure" and "error_details" in job_info:
                    output += "\nError Details:\n"
                    output += job_info["error_details"]

            if log_file:
                with open(log_file, "a") as f:
                    f.write(output)
            else:
                print(output)

    except Exception as e:
        error_msg = f"Error checking status: {e}"
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"{error_msg}\n")
        else:
            print(error_msg)


def get_github_token() -> str:
    """Get GitHub token from environment or config file."""
    # Try environment variable first
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token.strip()

    # Try config files
    config_files = [
        Path.home() / ".github_token",  # User's home directory
        Path.cwd() / "github_token.txt",  # Current directory
        Path(__file__).parent / "github_token.txt",  # Script directory
        Path(__file__).parent.parent
        / "src"
        / "config"
        / "github_token.txt",  # Config directory
    ]

    for config_file in config_files:
        if config_file.exists():
            try:
                token = config_file.read_text().strip()
                if token:
                    return token
            except Exception:
                continue

    return None


def main():
    """Check CI status for commits."""
    parser = argparse.ArgumentParser(description="Check CI status for commits")
    parser.add_argument("--commit", "-c", help="Specific commit reference to check")
    parser.add_argument(
        "--log",
        "-l",
        nargs="?",
        const="check_ci_latest_commit.log",
        help="Write output to log file (default: logs/check_ci_latest_commit.log)",
    )
    parser.add_argument(
        "--watch",
        "-w",
        action="store_true",
        help="Watch workflow status until completion",
    )
    args = parser.parse_args()

    # Get token
    token = get_github_token()
    if not token:
        print(
            "Error: GitHub token not found. Please set GITHUB_TOKEN environment variable or create one of:"
        )
        print("  - ~/.github_token")
        print("  - github_token.txt in current directory")
        print("  - github_token.txt in script directory")
        print("  - src/git_ci_config/github_token.txt")
        sys.exit(1)

    # Get commit SHA
    commit_sha = get_commit_sha(args.commit)

    # Setup log file if requested
    log_file = None
    if args.log:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Handle case where user provides path with logs/ prefix
        log_filename = args.log
        if log_filename.startswith("logs/"):
            log_filename = log_filename[5:]  # Remove "logs/" prefix
        elif log_filename.startswith("logs\\"):
            log_filename = log_filename[5:]  # Remove "logs\" prefix (Windows)

        log_file = log_dir / log_filename

    # Check status
    if args.watch:
        # Watch until completion
        start_time = time.time()
        max_wait_time = 600  # 10 minutes
        check_interval = 10  # Check every 10 seconds

        is_first_write = True
        while time.time() - start_time < max_wait_time:
            check_workflow_status(commit_sha, token, log_file, is_first_write)
            is_first_write = (
                False  # After first write, append mode for subsequent writes
            )

            # Check if workflow is complete
            status = check_ci_status(commit_sha)
            if status and status["status"] == "completed":
                break

            time.sleep(check_interval)
    else:
        # Single check
        check_workflow_status(commit_sha, token, log_file, is_first_write=True)


if __name__ == "__main__":
    main()
