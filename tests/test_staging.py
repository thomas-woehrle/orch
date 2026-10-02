"""Staging refusal and the worktree script."""

from orch.errors import PreconditionError
from orch.staging import refuse_if_active, staging_command
import pytest


def test_staging_script_substitutes_sha():
    command = staging_command("/repo", "abc123")
    assert "flock .orch-staging.lock" in command
    assert "git worktree add --detach .worktrees/abc123 abc123" in command
    assert "uv sync --frozen" in command
    assert "__SHA__" not in command


def test_resync_refusal():
    refuse_if_active(local_hash="a", remote_hash="a", active_names=["job"], resync=False)
    refuse_if_active(local_hash="a", remote_hash="b", active_names=[], resync=False)
    refuse_if_active(local_hash="a", remote_hash="b", active_names=["job"], resync=True)
    with pytest.raises(PreconditionError, match="--resync"):
        refuse_if_active(local_hash="a", remote_hash="b", active_names=["sdft (1)"], resync=False)
