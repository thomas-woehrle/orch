"""``.orchignore`` relaxes the clean-tree rule only for paths the cluster drops."""

import stat
import subprocess
from pathlib import Path

import pytest

from orch.conn import Local
from orch.errors import PreconditionError, UsageError
from orch.gitpreflight import git_state
from orch.staging import staging_command


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "orch@example.com")
    _git(path, "config", "user.name", "orch")
    return path


def _pushed(tmp: Path) -> Path:
    origin = tmp / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True)
    repo = _repo(tmp / "repo")
    _git(repo, "remote", "add", "origin", str(origin))
    return repo


def _commit_and_push(repo: Path, message: str = "init") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    _git(repo, "push", "-u", "origin", "HEAD")
    return _git(repo, "rev-parse", "HEAD")


def test_ignored_dirt_is_allowed_and_protected_paths_are_not(tmp_path: Path):
    repo = _pushed(tmp_path)
    (repo / "train.py").write_text("print(1)\n")
    (repo / "docs").mkdir()
    (repo / "docs" / "a.md").write_text("doc\n")
    (repo / ".orchignore").write_text("docs/\n")
    _commit_and_push(repo)

    (repo / "docs" / "a.md").write_text("edited\n")
    (repo / "docs" / "new.md").write_text("untracked\n")
    state = git_state(repo)
    assert state.orchignore_lines == ["docs/"]
    assert state.orchignore_sha256

    (repo / "train.py").write_text("print(2)\n")
    with pytest.raises(PreconditionError, match="train.py"):
        git_state(repo)
    _git(repo, "checkout", "--", "train.py")

    (repo / ".orchignore").write_text("docs/\ntests/\n")
    with pytest.raises(PreconditionError, match=".orchignore"):
        git_state(repo)
    _git(repo, "checkout", "--", ".orchignore")

    (repo / "stray.txt").write_text("nope\n")
    with pytest.raises(PreconditionError, match="stray.txt"):
        git_state(repo)


def test_orchignore_cannot_drop_the_package(tmp_path: Path):
    repo = _pushed(tmp_path)
    (repo / "orch.yaml").write_text("version: 1\n")
    (repo / ".orchignore").write_text("*.yaml\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "ignore")
    _git(repo, "push", "origin", "HEAD")
    with pytest.raises(UsageError, match="orch.yaml"):
        git_state(repo)


def test_sparse_checkout_drops_ignored_paths(tmp_path: Path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True)
    source = _repo(tmp_path / "source")
    _git(source, "remote", "add", "origin", str(origin))
    (source / "train.py").write_text("print(1)\n")
    (source / "tests").mkdir()
    (source / "tests" / "foo.py").write_text("assert True\n")
    (source / "uv.lock").write_text("lock\n")
    (source / ".orchignore").write_text("tests/\n")
    sha = _commit_and_push(source)
    cluster = tmp_path / "cluster"
    subprocess.run(["git", "clone", str(origin), str(cluster)], check=True)

    stub = tmp_path / "bin"
    stub.mkdir()
    uv = stub / "uv"
    uv.write_text('#!/bin/sh\nmkdir -p "$UV_PROJECT_ENVIRONMENT"\nexit 0\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)

    command = staging_command(str(cluster), sha, ["tests/"])
    assert "'/*'" in command and "'!tests/'" in command
    Local().run(f"export PATH={stub}:$PATH\n{command}")
    worktree = cluster / ".worktrees" / sha
    assert (worktree / "train.py").is_file()
    assert not (worktree / "tests" / "foo.py").exists()
