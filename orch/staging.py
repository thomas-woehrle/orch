"""Per-SHA worktrees and the shared venv. See docs/orchestrator.md §6.3."""

from __future__ import annotations

import hashlib
from pathlib import Path

from orch.cache import Cache
from orch.config import Cluster
from orch.errors import PreconditionError
from orch.render import shquote

# The body is a quoted heredoc so pattern quotes survive and the inner shell
# expands $lock_hash itself. See docs/orchestrator.md §6.3.
_SCRIPT = """
cd __REPO_DIR__
flock .orch-staging.lock bash <<'ORCH_STAGE'
git fetch --quiet origin
git cat-file -e __SHA__^{commit}
__WORKTREE__
cd .worktrees/__SHA__
export UV_PROJECT_ENVIRONMENT=__REPO_DIR__/.venv
lock_hash=$(sha256sum uv.lock | cut -c1-16)
if [ "$(cat __REPO_DIR__/.venv/.orch-lock-hash 2>/dev/null)" != "$lock_hash" ]; then
  uv sync --frozen
  echo "$lock_hash" > __REPO_DIR__/.venv/.orch-lock-hash
fi
ORCH_STAGE
"""


def lock_hash(repo_root: Path) -> str:
    return hashlib.sha256((repo_root / "uv.lock").read_bytes()).hexdigest()[:16]


def refuse_if_active(
    *,
    local_hash: str,
    remote_hash: str,
    active_names: list[str],
    resync: bool,
) -> None:
    if local_hash == remote_hash or not active_names:
        return
    if resync:
        return
    raise PreconditionError(
        "uv.lock changed while jobs are active ("
        + ", ".join(active_names)
        + "). Re-run with --resync to update the shared venv."
    )


def active_job_names(cache: Cache, cluster_name: str, qstat: dict[str, str] | None) -> list[str]:
    names: list[str] = []
    for entry in cache.entries.values():
        if entry.cluster != cluster_name:
            continue
        state = "unknown" if qstat is None else qstat.get(entry.scheduler_id, "gone")
        if state in {"queued", "running", "unknown"}:
            names.append(f"{entry.name} ({entry.scheduler_id})")
    return names


def staging_command(
    repo_dir: str, sha: str, orchignore_lines: list[str] | None = None
) -> str:
    if orchignore_lines:
        patterns = " ".join(
            shquote(item) for item in ["/*", *(f"!{line}" for line in orchignore_lines)]
        )
        worktree = (
            f"if [ ! -d .worktrees/{sha} ]; then\n"
            f"  git worktree add --detach --no-checkout .worktrees/{sha} {sha}\n"
            f"  git -C .worktrees/{sha} sparse-checkout set --no-cone {patterns}\n"
            f"  git -C .worktrees/{sha} checkout --quiet {sha}\n"
            f"fi\n"
        )
    else:
        worktree = (
            f"[ -d .worktrees/{sha} ] || git worktree add --detach .worktrees/{sha} {sha}\n"
        )
    return (
        _SCRIPT.replace("__REPO_DIR__", repo_dir)
        .replace("__SHA__", sha)
        .replace("__WORKTREE__\n", worktree)
    )


def stage(
    conn,
    cluster: Cluster,
    repo_root: Path,
    sha: str,
    *,
    cache: Cache,
    qstat: dict[str, str] | None,
    resync: bool,
    orchignore_lines: list[str] | None = None,
) -> None:
    local = lock_hash(repo_root)
    remote = conn.run(
        f"cat {cluster.repo_dir}/.venv/.orch-lock-hash",
        check=False,
    )
    remote_hash = remote.stdout.strip() if remote.returncode == 0 else ""
    refuse_if_active(
        local_hash=local,
        remote_hash=remote_hash,
        active_names=active_job_names(cache, cluster.name, qstat),
        resync=resync,
    )
    conn.run(staging_command(cluster.repo_dir, sha, orchignore_lines), stream=True)
