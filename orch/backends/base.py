"""Scheduler backends. See docs/orchestrator.md §6.4."""

from typing import Any, Protocol

from orch.config import Cluster


class Backend(Protocol):
    name: str
    tools: tuple[str, ...]
    job_id_env: str

    def validate_resources(self, resources: dict[str, Any], cluster: Cluster) -> None: ...

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
    ) -> str: ...

    def render_job_script(
        self,
        job: Any,
        cluster: Cluster,
        env: dict[str, str],
        src_dir: str,
        venv_python: str,
        hold_on: list[str],
    ) -> str: ...

    def submit(self, conn: object, run_dir: str, hold_on: list[str]) -> str: ...

    def status_map(self, conn: object) -> dict[str, str]: ...

    def cancel(self, conn: object, scheduler_id: str) -> None: ...
