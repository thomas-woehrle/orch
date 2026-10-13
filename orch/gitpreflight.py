"""Refuse to submit from a dirty or unpushed checkout. See docs/orchestrator.md B2 and B6."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from orch.errors import PreconditionError
from orch.orchignore import check_protected, load, matches


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


@dataclass(frozen=True)
class GitState:
    sha: str
    remote: str
    orchignore_sha256: str | None = None
    orchignore_lines: list[str] | None = None


def _dirty_paths(porcelain: str) -> list[str]:
    """Paths from ``git status --porcelain -z``. Renames contribute the new path."""
    if not porcelain:
        return []
    parts = porcelain.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(parts):
        entry = parts[index]
        if not entry:
            index += 1
            continue
        status = entry[:2]
        path = entry[3:]
        if "R" in status or "C" in status:
            index += 1
            if index < len(parts) and parts[index]:
                path = parts[index]
        if path:
            paths.append(path)
        index += 1
    return paths


def git_state(repo: Path) -> GitState:
    """Return the recorded SHA, or raise if the tree is dirty or unpushed.

    A path may be dirty only when ``.orchignore`` excludes it from the cluster
    worktree. ``.orchignore`` itself is never allowed to be dirty.
    """
    lines, digest = load(repo)
    check_protected(repo, lines)
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "-z")
    if status.returncode != 0:
        raise PreconditionError(status.stderr.strip() or "git status failed")
    dirty = _dirty_paths(status.stdout)
    if ".orchignore" in dirty:
        raise PreconditionError(".orchignore is not clean; commit it before orch run")
    allowed = matches(repo, dirty) if lines is not None else set()
    blocked = [path for path in dirty if path not in allowed]
    if blocked:
        raise PreconditionError(
            "working tree is not clean; commit or stash before orch run:\n"
            + "\n".join(blocked)
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
    return GitState(
        sha=sha.stdout.strip(),
        remote=remote.stdout.strip(),
        orchignore_sha256=digest,
        orchignore_lines=lines,
    )
