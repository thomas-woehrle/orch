"""Refuse to submit from a dirty or unpushed checkout. See docs/orchestrator.md B2."""

from __future__ import annotations

import subprocess
from pathlib import Path

from orch.errors import PreconditionError


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def git_state(repo: Path) -> tuple[str, str]:
    """Return ``(sha, origin url)`` or raise if the tree is dirty or unpushed."""
    status = _git(repo, "status", "--porcelain")
    if status.returncode != 0:
        raise PreconditionError(status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        raise PreconditionError(
            "working tree is not clean; commit or stash before orch run"
        )
    containing = _git(repo, "branch", "-r", "--contains", "HEAD")
    if containing.returncode != 0:
        raise PreconditionError(containing.stderr.strip() or "git branch failed")
    pushed = any(
        line.strip().startswith("origin/") for line in containing.stdout.splitlines()
    )
    if not pushed:
        raise PreconditionError(
            "HEAD is not contained in any origin ref; push before orch run"
        )
    sha = _git(repo, "rev-parse", "HEAD")
    remote = _git(repo, "remote", "get-url", "origin")
    if sha.returncode != 0 or remote.returncode != 0:
        raise PreconditionError("could not read HEAD or origin")
    return sha.stdout.strip(), remote.stdout.strip()
