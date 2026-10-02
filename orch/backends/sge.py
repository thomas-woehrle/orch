"""Grid Engine backend. See docs/orchestrator.md §6.4."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from orch.config import Cluster
from orch.errors import RemoteError
from orch.plan import Job
from orch.render import render_header, render_job_script


class SGEBackend:
    name = "sge"

    def render_job_script(
        self,
        job: Job,
        cluster: Cluster,
        env: dict[str, str],
        src_dir: str,
        venv_python: str,
    ) -> str:
        run_dir = f"{cluster.work_dir}/{job.run_id}"
        header = render_header(cluster.header, job.resources)
        return render_job_script(
            header=header,
            setup=cluster.setup,
            env=env,
            fingerprint=job.fingerprint,
            attempt=job.attempt,
            run_dir=run_dir,
            src_dir=src_dir,
            venv_python=venv_python,
        )

    def submit(self, conn: object, run_dir: str, hold_on: list[str]) -> str:
        holds = ""
        if hold_on:
            holds = " -hold_jid " + ",".join(hold_on)
        result = conn.run(f"qsub -terse{holds} {run_dir}/job.sh")  # type: ignore[attr-defined]
        scheduler_id = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
        if not scheduler_id:
            raise RemoteError(f"qsub returned no job id for {run_dir}")
        return scheduler_id

    def status_map(self, conn: object) -> dict[str, str]:
        result = conn.run("qstat -xml", check=False)  # type: ignore[attr-defined]
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RemoteError(f"qstat failed: {detail or result.returncode}")
        if not result.stdout.strip():
            return {}
        return parse_qstat_xml(result.stdout)

    def cancel(self, conn: object, scheduler_id: str) -> None:
        conn.run(f"qdel {scheduler_id}")  # type: ignore[attr-defined]


def parse_qstat_xml(text: str) -> dict[str, str]:
    """Map SGE job numbers to queued, running, or error."""
    root = ET.fromstring(text)
    found: dict[str, str] = {}
    for job in root.iter("job_list"):
        number = (job.findtext("JB_job_number") or "").strip()
        if not number:
            continue
        raw = (job.findtext("state") or job.attrib.get("state") or "").strip()
        found[number] = classify_sge_state(raw)
    return found


def classify_sge_state(state: str) -> str:
    token = state.split()[0] if state else ""
    if not token:
        return "unknown"
    if "E" in token or token.lower() == "error":
        return "error"
    lowered = token.lower()
    if lowered in {"r", "t", "rr", "running"} or lowered.startswith("r"):
        return "running"
    if "q" in lowered or lowered == "pending":
        return "queued"
    return "unknown"
