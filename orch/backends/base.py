"""Scheduler backends."""

from typing import Protocol

from orch.config import Cluster
from orch.plan import Job


class Backend(Protocol):
    name: str

    def render_job_script(self, job: Job, cluster: Cluster, env: dict[str, str], src_dir: str, venv_python: str) -> str: ...

    def submit(self, conn: object, run_dir: str, hold_on: list[str]) -> str: ...

    def status_map(self, conn: object) -> dict[str, str]: ...

    def cancel(self, conn: object, scheduler_id: str) -> None: ...
