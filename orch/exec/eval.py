"""Evaluation wrapper. See docs/orchestrator.md §6.6."""

from __future__ import annotations

import importlib
import os
import shlex
import sys
from pathlib import Path

from orch.exec.common import (
    child_env,
    finite_metrics,
    init_run,
    read_job,
    save_files,
    tee_subprocess,
    update_config,
    write_failure,
)
from orch.exec.models import resolve_model


def _parser(dotted: str):
    module_name, func_name = dotted.split(":")
    return getattr(importlib.import_module(module_name), func_name)


def upstream_block(producer) -> dict | None:
    """A snapshot of the run that produced the model, for the W&B table."""
    if producer is None:
        return None
    config = getattr(producer, "config", None) or {}
    orch = config.get("orch") or {}
    spec_path = orch.get("spec_path") or ""
    return {
        "run_id": getattr(producer, "id", None) or orch.get("run_id"),
        "run_name": getattr(producer, "name", None),
        "fingerprint": orch.get("fingerprint"),
        "spec": Path(spec_path).stem if spec_path else None,
        "entry": orch.get("entry"),
        "params": config.get("params") or {},
    }


def _artifact_producer(job: dict):
    ref = job.get("model_ref") or ""
    if not isinstance(ref, str) or not ref.startswith("artifact:"):
        return None
    import wandb

    try:
        artifact = wandb.Api().artifact(ref[len("artifact:") :])
        return artifact.logged_by()
    except Exception:
        return None


def _upstream_ready(job: dict) -> tuple[bool, str, object | None]:
    upstream = job.get("upstream_run_id")
    if not upstream:
        return True, "", _artifact_producer(job)
    import wandb

    api = wandb.Api()
    entity = job["wandb"]["entity"]
    project = job["wandb"]["project"]
    try:
        producer = api.run(f"{entity}/{project}/{upstream}")
    except Exception as exc:
        return False, f"missing ({exc})", None
    artifacts = [item for item in producer.logged_artifacts() if item.type == "model"]
    if producer.state != "finished" or not artifacts:
        return False, producer.state, None
    return True, producer.state, producer


def execute(run_dir: Path) -> int:
    job = read_job(run_dir)
    ready, state, producer = _upstream_ready(job)
    if not ready:
        print(
            f"upstream {job.get('upstream_run_id')} not finished (state={state}); "
            "exiting without creating a run"
        )
        return 0
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    run, run_dir = init_run(job, run_dir)
    code = 1
    try:
        work_dir = Path(job["cluster"]["work_dir"])
        model_path, model_meta = resolve_model(job, run, work_dir)
        os.environ["ORCH_MODEL_PATH"] = str(model_path)
        command = job["command"].replace("{model_path}", shlex.quote(str(model_path)))
        update_config(
            run,
            job,
            state="running",
            extra={"model": model_meta},
            top_extra={"upstream": upstream_block(producer)},
        )
        if model_meta.get("resolved_revision"):
            run.config.update(
                {"orch": {**dict(run.config.get("orch", {})), "model": model_meta}},
                allow_val_change=True,
            )
        code = tee_subprocess(
            command,
            cwd=job["cluster"]["src_dir"],
            env={**child_env(job), "ORCH_MODEL_PATH": str(model_path)},
            run_dir=run_dir,
        )
        stem = Path(job["spec_path"]).stem
        if code == 0:
            if not job.get("parser"):
                raise RuntimeError(f"entry {job['entry']} has no parser")
            src = job["cluster"]["src_dir"]
            if src not in sys.path:
                sys.path.insert(0, src)
            metrics = _parser(job["parser"])(run_dir / "output")
            finite_metrics(metrics)
            run.log({f"{stem}/{key}": float(value) for key, value in metrics.items()})
            update_config(run, job, state="succeeded", extra={"model": model_meta})
        else:
            run.tags = list(run.tags) + ["failed"]
            run.summary["orch/exit_code"] = code
            run.summary["orch/failure_reason"] = "nonzero_exit"
            update_config(run, job, state="failed", extra={"model": model_meta})
        save_files(run, run_dir, extra=[run_dir / "output"])
        run.finish(exit_code=code)
    except Exception as exc:
        write_failure(run_dir, exc)
        run.tags = list(getattr(run, "tags", []) or []) + ["failed"]
        run.summary["orch/exit_code"] = code
        run.summary["orch/failure_reason"] = f"parser:{exc}"
        update_config(run, job, state="failed")
        save_files(run, run_dir, extra=[run_dir / "output"])
        run.finish(exit_code=code or 1)
        if code == 0:
            return 1
        return code
    return code
