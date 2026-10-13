"""Slurm backend. See docs/orchestrator.md §6.4."""

from __future__ import annotations

from typing import Any

from orch.config import Cluster
from orch.errors import RemoteError, UsageError
from orch.render import _jinja, render_header, render_job_body, validate_resources

HEAD = """#!/bin/bash -l
#SBATCH -J {{ name }}
#SBATCH -o {{ run_dir }}/scheduler.out
#SBATCH -e {{ run_dir }}/scheduler.err
{{ header }}
"""


class SlurmBackend:
    name = "slurm"
    tools = ("sbatch", "squeue", "scancel")
    job_id_env = "SLURM_JOB_ID"

    def validate_resources(self, resources: dict[str, Any], cluster: Cluster) -> None:
        if not cluster.header:
            raise UsageError(f"cluster {cluster.name} has no header")
        validate_resources(resources, cluster.header)

    def render_head(
        self,
        cluster: Cluster,
        *,
        name: str,
        run_dir: str,
        resources: dict[str, Any],
        hold_on: list[str],
        device: int | None,
        src_dir: str = "",
        venv_python: str = "",
    ) -> str:
        del hold_on, device, src_dir, venv_python  # holds are an sbatch flag
        header = render_header(cluster.header or "", resources)
        return _jinja().from_string(HEAD).render(
            header=header, name=name, run_dir=run_dir
        ).rstrip("\n")

    def render_job_script(
        self,
        job: Any,
        cluster: Cluster,
        env: dict[str, str],
        src_dir: str,
        venv_python: str,
        hold_on: list[str],
    ) -> str:
        del hold_on
        run_dir = f"{cluster.work_dir}/{job.run_id}"
        head = self.render_head(
            cluster,
            name=f"orch_{job.fingerprint}_a{job.attempt}",
            run_dir=run_dir,
            resources=job.resources,
            hold_on=[],
            device=None,
        )
        body = render_job_body(
            setup=cluster.setup,
            env=env,
            run_dir=run_dir,
            src_dir=src_dir,
            venv_python=venv_python,
        )
        return head + "\n" + body

    def submit(self, conn: object, run_dir: str, hold_on: list[str]) -> str:
        dependency = ""
        if hold_on:
            dependency = (
                " --dependency=afterok:"
                + ":".join(hold_on)
                + " --kill-on-invalid-dep=yes"
            )
        result = conn.run(f"sbatch --parsable{dependency} {run_dir}/job.sh")  # type: ignore[attr-defined]
        line = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        scheduler_id = line.split(";", 1)[0]
        if not scheduler_id:
            raise RemoteError(f"sbatch returned no job id for {run_dir}")
        return scheduler_id

    def status_map(self, conn: object) -> dict[str, str]:
        result = conn.run("squeue -h -u $USER -o '%i|%t|%j'", check=False)  # type: ignore[attr-defined]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RemoteError(f"squeue failed: {detail or result.returncode}")
        return parse_squeue(result.stdout)

    def cancel(self, conn: object, scheduler_id: str) -> None:
        conn.run(f"scancel {scheduler_id}")  # type: ignore[attr-defined]


def parse_squeue(text: str) -> dict[str, str]:
    """Map Slurm job ids to queued or running. Other states are left out (gone)."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id, raw = parts[0].strip(), parts[1].strip()
        if not job_id:
            continue
        state = classify_slurm_state(raw)
        if state is not None:
            found[job_id] = state
    return found


def classify_slurm_state(state: str) -> str | None:
    token = state.strip()
    if token in {"PD", "S"}:
        return "queued"
    if token in {"R", "CG", "CF"}:
        return "running"
    return None
