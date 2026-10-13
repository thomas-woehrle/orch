"""B2: a clean, pushed checkout is required."""

import subprocess
from pathlib import Path

import pytest

from orch.errors import PreconditionError
from orch.gitpreflight import git_state


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_clean_pushed_tree_and_refusals(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "orch@example.com")
    _git(repo, "config", "user.name", "orch")
    (repo / "readme").write_text("ok\n")
    _git(repo, "add", "readme")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", "https://example.com/repo.git")
    _git(repo, "update-ref", "refs/remotes/origin/master", "HEAD")
    state = git_state(repo)
    assert len(state.sha) == 40
    assert state.remote.endswith("repo.git")
    assert state.orchignore_sha256 is None
    (repo / "readme").write_text("dirty\n")
    with pytest.raises(PreconditionError, match="not clean"):
        git_state(repo)
    _git(repo, "checkout", "--", "readme")
    _git(repo, "update-ref", "-d", "refs/remotes/origin/master")
    with pytest.raises(PreconditionError, match="origin"):
        git_state(repo)
