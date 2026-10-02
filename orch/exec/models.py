"""Download the model an eval job named. See docs/orchestrator.md §5.4 and §6.6."""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any


def _download_artifact(run, artifact, work_dir: Path) -> Path:
    digest = artifact.digest
    root = work_dir / "cache" / "models" / digest
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f"{digest}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run.use_artifact(artifact)
        if (root / ".complete").is_file():
            return root
        root.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(root))
        (root / ".complete").write_text("ok\n")
        return root


def resolve_model(job: dict[str, Any], run, work_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return a local model directory and the ``config.orch.model`` block."""
    ref = job["model_ref"]
    if not isinstance(ref, str):
        raise RuntimeError("eval job has no model_ref")
    if ref.startswith("hf:"):
        from huggingface_hub import snapshot_download

        body = ref[3:]
        repo, revision = body.split("@", 1) if "@" in body else (body, None)
        path = Path(snapshot_download(repo, revision=revision))
        return path, {
            "kind": "hf",
            "id": repo,
            "resolved_revision": path.name,
        }
    if ref.startswith("run:"):
        import wandb

        run_id = ref[4:]
        api = wandb.Api()
        entity = job["wandb"]["entity"]
        project = job["wandb"]["project"]
        producer = api.run(f"{entity}/{project}/{run_id}")
        artifacts = [item for item in producer.logged_artifacts() if item.type == "model"]
        if not artifacts:
            raise RuntimeError(f"{run_id} has no model artifact")
        path = _download_artifact(run, artifacts[0], work_dir)
        return path, {
            "kind": "run",
            "id": run_id,
            "artifact_name": artifacts[0].name,
            "artifact_version": artifacts[0].version,
            "producer_run_id": run_id,
        }
    if ref.startswith("artifact:"):
        name = ref[len("artifact:") :]
        artifact = run.use_artifact(name)
        path = _download_artifact(run, artifact, work_dir)
        return path, {
            "kind": "artifact",
            "id": name,
            "artifact_name": artifact.name,
            "artifact_version": artifact.version,
        }
    raise RuntimeError(f"unsupported model reference {ref}")
