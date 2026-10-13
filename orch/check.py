"""``orch cluster check``. See docs/orchestrator.md §8."""

from __future__ import annotations

import time
from pathlib import Path

from orch.backends import backend_for
from orch.config import Cluster, LocalConfig
from orch.conn import open_conn
from orch.errors import PreconditionError
from orch.gitpreflight import _git
from orch.render import shquote


def _norm_remote(url: str) -> str:
    text = url.strip().rstrip("/")
    if text.endswith(".git"):
        text = text[:-4]
    if text.startswith("git@"):
        host, path = text[4:].split(":", 1)
        text = f"https://{host}/{path}"
    return text.lower()


def check_cluster(
    cluster: Cluster,
    local: LocalConfig,
    repo_root: Path,
    *,
    gpu: bool = False,
) -> list[str]:
    """Test invariants A1–A5 and A7. ``--gpu`` also runs A6 on a compute node."""
    notes: list[str] = []
    backend = backend_for(cluster)
    if cluster.backend == "local" and Path(cluster.repo_dir).resolve() == repo_root.resolve():
        raise PreconditionError(
            "A3 local repo_dir must be a separate clone, not the checkout orch is invoked from"
        )
    with open_conn(cluster) as conn:
        where = f"ssh {cluster.ssh}" if cluster.ssh else "local"
        notes.append(f"A1 {where}: ok")
        tool_check = " && ".join(f"command -v {shquote(tool)}" for tool in ("git", "uv", *backend.tools))
        conn.run(tool_check)
        notes.append("A2 login PATH has git, uv, " + ", ".join(backend.tools))
        remote = conn.run(f"git -C {shquote(cluster.repo_dir)} remote get-url origin")
        local_remote = _git(repo_root, "remote", "get-url", "origin")
        if local_remote.returncode != 0:
            raise PreconditionError("local checkout has no origin")
        if _norm_remote(remote.stdout) != _norm_remote(local_remote.stdout):
            raise PreconditionError(
                f"A3 origin mismatch: cluster {remote.stdout.strip()} vs local {local_remote.stdout.strip()}"
            )
        conn.run(f"git -C {shquote(cluster.repo_dir)} fetch --quiet origin")
        notes.append("A3 clone origin matches and fetch works")
        conn.run(f"test -d {shquote(cluster.work_dir)} && test -w {shquote(cluster.work_dir)}")
        notes.append("A4 work_dir exists and is writable")
        conn.run("curl -fsS -o /dev/null --max-time 20 https://pypi.org/simple/pip/")
        conn.run("curl -fsS -o /dev/null --max-time 20 https://github.com")
        notes.append("A5 login node reached PyPI and GitHub")
        for key, value in cluster.env.items():
            if value.startswith("/"):
                conn.run(f"mkdir -p {shquote(value)} && test -w {shquote(value)}")
            notes.append(f"A7 env {key}")
        if gpu:
            notes.append(_gpu_roundtrip(conn, cluster, local, backend, repo_root))
    return notes


def _gpu_roundtrip(conn, cluster: Cluster, local: LocalConfig, backend, repo_root: Path) -> str:
    resources = {"gpus": 1, "time": "00:15:00", "mem": "16G"}
    stamp = str(int(time.time()))
    # `check-a<stamp>` parses like a run id, so the local backend can find the
    # state file at work_dir/<run id>/state while the session is orch_check_a<stamp>.
    run_id = f"check-a{stamp}"
    run_dir = f"{cluster.work_dir}/{run_id}"
    venv_python = f"{cluster.repo_dir}/.venv/bin/python"
    head = backend.render_head(
        cluster,
        name=f"orch_check_a{stamp}",
        run_dir=run_dir,
        resources=resources,
        hold_on=[],
        device=None,
        src_dir=str(repo_root) if cluster.backend == "local" else "",
        venv_python=venv_python,
    )
    script = f"""{head}
set -euo pipefail
{cluster.setup.strip()}
export PATH={shquote(str(Path(venv_python).parent))}:$PATH
{venv_python} - <<'PY'
import torch, vllm, wandb
assert torch.cuda.is_available(), "no CUDA device"
run = wandb.init(entity={local.wandb.entity!r}, project={local.wandb.project!r}, job_type="cluster-check", name="orch-cluster-check", tags=["orch-cluster-check"])
wandb.log({{"orch_check/ok": 1}})
print("ORCH_CHECK_OK", run.url)
run.finish()
PY
"""
    conn.run(f"mkdir -p {shquote(run_dir)}")
    conn.put_text(f"{run_dir}/job.sh", script)
    scheduler_id = backend.submit(conn, run_dir, [])
    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        states = backend.status_map(conn)
        if scheduler_id not in states:
            break
        time.sleep(15)
    else:
        raise PreconditionError(f"A6 gpu check job {scheduler_id} did not finish within 30 minutes")
    out = conn.run(f"cat {shquote(run_dir + '/scheduler.out')}", check=False)
    err = conn.run(f"cat {shquote(run_dir + '/scheduler.err')}", check=False)
    if "ORCH_CHECK_OK" not in out.stdout:
        raise PreconditionError(
            f"A6 gpu check failed.\nstdout:\n{out.stdout}\nstderr:\n{err.stderr}"
        )
    return f"A6 gpu check {scheduler_id}: ok"
