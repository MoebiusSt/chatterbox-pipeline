"""
GitHub API configuration for CI status checks.
"""

import os
from pathlib import Path
from typing import Optional


def get_github_token() -> Optional[str]:
    """
    Get GitHub token from environment variable or config file.
    Returns None if no token is found.
    """
    # First try environment variable
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token

    # Then try config file
    config_path = Path(__file__).parent / "github_token.txt"
    if config_path.exists():
        return config_path.read_text().strip()

    return None


def check_ci_status(commit_sha: str) -> Optional[dict]:
    """
    Check CI status for a specific commit.

    Args:
        commit_sha: The commit SHA to check

    Returns:
        dict: CI status information
    """
    import requests

    token = get_github_token()
    if not token:
        raise ValueError(
            "No GitHub token found. Please set GITHUB_TOKEN environment variable or create github_token.txt"
        )

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get workflow runs
    url = "https://api.github.com/repos/MoebiusSt/chatterbox-pipeline/actions/runs"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    # Find run for specific commit
    runs = response.json()["workflow_runs"]
    for run in runs:
        if run["head_sha"] == commit_sha:
            return {
                "status": run["status"],
                "conclusion": run["conclusion"],
                "name": run["name"],
                "created_at": run["created_at"],
                "html_url": run["html_url"],
            }

    return None
