"""``.orchignore``: paths the cluster worktree does not contain. See docs/orchestrator.md B6."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from orch.errors import UsageError

PROTECTED = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "orch.yaml",
    "orch_adapters.py",
    ".orchignore",
)


def load(repo_root: Path) -> tuple[list[str] | None, str | None]:
    """Return ``(patterns, sha256)``. Both are ``None`` when the file is absent."""
    path = repo_root / ".orchignore"
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    lines: list[str] = []
    for line in raw.decode().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines, hashlib.sha256(raw).hexdigest()


def matches(repo_root: Path, paths: list[str]) -> set[str]:
    """Paths in ``paths`` that ``.orchignore`` ignores.

    ``git check-ignore`` reports every source. Only matches whose pattern file
    is ``.orchignore`` count, so a path ignored by ``.gitignore`` alone is not
    exempt from the clean-tree check.
    """
    if not paths:
        return set()
    ignore = repo_root / ".orchignore"
    if not ignore.is_file():
        return set()
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "-c",
            f"core.excludesFile={ignore}",
            "check-ignore",
            "-v",
            "--no-index",
            "--stdin",
        ],
        input="\n".join(paths) + "\n",
        check=False,
        capture_output=True,
        text=True,
    )
    # Exit 0 when something matches, 1 when nothing does.
    if result.returncode not in {0, 1}:
        raise UsageError(result.stderr.strip() or "git check-ignore failed")
    found: set[str] = set()
    for line in result.stdout.splitlines():
        # <source>:<linenum>:<pattern>\t<pathname>
        meta, sep, pathname = line.partition("\t")
        if not sep:
            continue
        source = meta.split(":", 1)[0]
        if source == str(ignore) or Path(source).name == ".orchignore":
            found.add(pathname)
    return found


def check_protected(repo_root: Path, lines: list[str] | None) -> None:
    """Refuse a file that would drop something staging or the package needs."""
    if not lines:
        return
    hit = sorted(matches(repo_root, list(PROTECTED)))
    if hit:
        raise UsageError(
            ".orchignore must not match " + ", ".join(hit) + " (these paths stay on the cluster)"
        )
