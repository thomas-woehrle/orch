"""Shared wrapper pieces. See docs/orchestrator.md §6.5."""

from __future__ import annotations

import json
import math
import os
import shlex
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.render import utc_now


def read_job(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "job.json").read_text())


def write_job(run_dir: Path, job: dict[str, Any]) -> None:
    (run_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")


def tee_subprocess(command: str, *, cwd: str, env: dict[str, str], run_dir: Path) -> int:
    args = shlex.split(command)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    def pump(stream, path: Path, mirror) -> None:
        with path.open("w") as handle:
            assert stream is not None
            for line in stream:
                handle.write(line)
                handle.flush()
                mirror.write(line)
                mirror.flush()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, stdout_path, sys.stdout)),
        threading.Thread(target=pump, args=(proc.stderr, stderr_path, sys.stderr)),
    ]
    for thread in threads:
        thread.start()
    code = proc.wait()
    for thread in threads:
        thread.join()
    return code


def child_env(job: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in job.get("env", {}).items()})
    venv_python = job.get("cluster", {}).get("venv_python")
    if venv_python:
        bin_dir = str(Path(venv_python).parent)
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _collision(exc: Exception) -> bool:
    text = str(exc).lower()
    return "deleted" in text or "already been used" in text or "previously created" in text


def init_run(job: dict[str, Any], run_dir: Path):
    """Attach to the pre-minted id. A deleted id bumps the attempt and renames the run dir."""
    import wandb

    while True:
        try:
            run = wandb.init(
                id=job["run_id"],
                entity=job["wandb"]["entity"],
                project=job["wandb"]["project"],
                name=job["run_name"],
                job_type=job["kind"],
                tags=list(job["wandb"].get("tags") or []),
                group=job["wandb"].get("group"),
                resume="allow",
                dir=str(run_dir / "wandb"),
                reinit="create_new",
            )
            return run, run_dir
        except Exception as exc:
            if not _collision(exc):
                raise
            old_dir = run_dir
            job["attempt"] = int(job["attempt"]) + 1
            job["run_id"] = f"{job['fingerprint']}-a{job['attempt']}"
            base = job["run_name"].rsplit(" (a", 1)[0]
            job["run_name"] = f"{base} (a{job['attempt']})"
            job["wandb"]["tags"] = [
                f"attempt:{job['attempt']}" if str(tag).startswith("attempt:") else tag
                for tag in job["wandb"].get("tags") or []
            ]
            new_dir = old_dir.parent / job["run_id"]
            old_dir.rename(new_dir)
            run_dir = new_dir
            for key, value in list(job.get("env", {}).items()):
                if isinstance(value, str) and str(old_dir) in value:
                    job["env"][key] = value.replace(str(old_dir), str(new_dir))
            if isinstance(job.get("command"), str):
                job["command"] = job["command"].replace(str(old_dir), str(new_dir))
            if "ORCH_RUN_DIR" in job.get("env", {}):
                job["env"]["ORCH_RUN_DIR"] = str(run_dir)
            write_job(run_dir, job)
            os.environ["WANDB_RUN_ID"] = job["run_id"]
            os.environ["WANDB_NAME"] = job["run_name"]


def orch_config(job: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    block = {
        "schema_version": 1,
        "orch_version": job.get("orch_version"),
        "fingerprint": job["fingerprint"],
        "fingerprint_version": job["fingerprint_version"],
        "fingerprint_input": job.get("fingerprint_input"),
        "attempt": job["attempt"],
        "run_id": job["run_id"],
        "kind": job["kind"],
        "entry": job["entry"],
        "spec_path": job["spec_path"],
        "description": job.get("spec", {}).get("description", ""),
        "runtime_keys": job.get("spec", {}).get("runtime_keys", []),
        "resources": job.get("spec", {}).get("resources", {}),
        "evals": job.get("spec", {}).get("evals", []),
        "overrides": job.get("overrides", []),
        "model_ref": job.get("model_ref"),
        "upstream_run_id": job.get("upstream_run_id"),
        "cluster": {
            key: job.get("cluster", {}).get(key)
            for key in ("name", "backend", "repo_dir", "work_dir", "src_dir")
        },
        "scheduler_id": os.environ.get("JOB_ID"),
        "git": job.get("git"),
        "submitted_from": job.get("submitted_from"),
        "submitted_at": job.get("submitted_at"),
        "started_at": utc_now(),
        "host": socket.gethostname(),
    }
    if extra:
        block.update(extra)
    return block


def flat_keys(job: dict[str, Any], *, state: str | None = None) -> dict[str, Any]:
    return {
        "orch_fingerprint": job["fingerprint"],
        "orch_attempt": job["attempt"],
        "orch_kind": job["kind"],
        "orch_entry": job["entry"],
        "orch_spec": Path(job["spec_path"]).stem,
        "orch_model_ref": job.get("model_ref"),
        "orch_upstream_run_id": job.get("upstream_run_id"),
        "orch_state": state,
    }


def update_config(run, job: dict[str, Any], *, state: str | None, extra: dict[str, Any] | None = None) -> None:
    # fingerprint input lives in fingerprint.json, which is the canonical record.
    # The job document does not duplicate it, so read the file when present.
    run_dir = Path(job["env"]["ORCH_RUN_DIR"])
    fp_path = run_dir / "fingerprint.json"
    if fp_path.is_file() and "fingerprint_input" not in job:
        job = dict(job)
        job["fingerprint_input"] = json.loads(fp_path.read_text())["input"]
    run.config.update(
        {
            "params": job.get("spec", {}).get("params", {}),
            "orch": orch_config(job, extra=extra),
            **flat_keys(job, state=state),
        },
        allow_val_change=True,
    )


def save_files(run, run_dir: Path, extra: list[Path] | None = None) -> None:
    names = [
        "job.sh",
        "job.json",
        "params.yaml",
        "fingerprint.json",
        "stdout.log",
        "stderr.log",
        "sge.out",
        "sge.err",
    ]
    paths = [run_dir / name for name in names]
    for path in extra or []:
        if path.is_dir():
            paths.extend(item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            paths.append(path)
    for path in paths:
        if path.is_file():
            run.save(str(path), base_path=str(run_dir), policy="now")


def finite_metrics(values: dict[str, float]) -> None:
    if not values:
        raise ValueError("parser returned no metrics")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"metric {key} is not a finite float: {value!r}")


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
