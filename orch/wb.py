"""W&B lookups used at resolve time and during reconcile."""

from __future__ import annotations

from pathlib import Path

from orch.config import find_repo_root, load_local
from orch.errors import UsageError


class WandbLookup:
    """Read-only W&B queries. Constructing it imports wandb."""

    def __init__(self, entity: str, project: str) -> None:
        self.entity = entity
        self.project = project
        self._api = None

    @classmethod
    def from_repo(cls, repo_root: Path | None = None) -> WandbLookup:
        root = repo_root or find_repo_root()
        local = load_local(root)
        return cls(local.wandb.entity, local.wandb.project)

    def api(self):
        if self._api is None:
            import wandb

            self._api = wandb.Api()
        return self._api

    def _run_path(self, run_id: str) -> str:
        return f"{self.entity}/{self.project}/{run_id}"

    def run(self, run_id: str):
        try:
            return self.api().run(self._run_path(run_id))
        except Exception as exc:
            text = str(exc).lower()
            if "could not find" in text or "not found" in text or "404" in text:
                return None
            raise

    def run_state(self, run_id: str) -> str | None:
        run = self.run(run_id)
        if run is None:
            return None
        return run.state

    def run_name(self, run_id: str) -> str | None:
        run = self.run(run_id)
        if run is None:
            return None
        return run.name

    def has_model_artifact(self, run_id: str) -> bool:
        run = self.run(run_id)
        if run is None:
            return False
        return any(item.type == "model" for item in run.logged_artifacts())

    def artifact_exists(self, name: str) -> bool:
        try:
            self.api().artifact(name)
        except Exception as exc:
            text = str(exc).lower()
            if "could not find" in text or "not found" in text or "404" in text:
                return False
            raise
        return True

    def probe_attempts(self, fingerprint: str) -> dict[int, str]:
        found: dict[int, str] = {}
        attempt = 1
        while attempt <= 1000:
            state = self.run_state(f"{fingerprint}-a{attempt}")
            if state is None:
                break
            found[attempt] = state
            attempt += 1
        return found

    def highest_finished_attempt(self, fingerprint: str) -> int | None:
        best: int | None = None
        for attempt, state in self.probe_attempts(fingerprint).items():
            if state == "finished":
                best = attempt
        return best

    def resolve_artifact_version(
        self, entity: str, project: str, name: str, alias: str
    ) -> str:
        try:
            artifact = self.api().artifact(f"{entity}/{project}/{name}:{alias}")
        except Exception as exc:
            raise UsageError(
                f"artifact alias {entity}/{project}/{name}:{alias} does not resolve ({exc})"
            ) from exc
        version = getattr(artifact, "version", None)
        if not isinstance(version, str) or not version:
            raise UsageError(
                f"artifact alias {alias!r} did not resolve to a concrete version"
            )
        return version
