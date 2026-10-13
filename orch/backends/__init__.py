"""Scheduler backends. See docs/orchestrator.md §6.4."""

from __future__ import annotations

from orch.backends.base import Backend
from orch.config import Cluster
from orch.errors import UsageError


def backend_for(cluster: Cluster) -> Backend:
    if cluster.backend == "sge":
        from orch.backends.sge import SGEBackend

        return SGEBackend()
    if cluster.backend == "slurm":
        from orch.backends.slurm import SlurmBackend

        return SlurmBackend()
    if cluster.backend == "local":
        from orch.backends.local import LocalBackend

        return LocalBackend(cluster.work_dir)
    raise UsageError(f"unknown backend {cluster.backend!r}")
