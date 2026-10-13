"""Training wrapper. See docs/orchestrator.md §6.5."""

from __future__ import annotations

from pathlib import Path

from orch.exec.common import (
    child_env,
    init_run,
    now,
    read_job,
    release_wandb,
    save_files,
    tee_subprocess,
    update_config,
)


def model_ready(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    patterns = ("*.safetensors", "*.bin", "*.pt", "*.gguf")
    return any(any(model_dir.glob(pattern)) for pattern in patterns)


def execute(run_dir: Path) -> int:
    import wandb

    job = read_job(run_dir)
    for name in ("model", "checkpoints", "wandb"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    # Prologue: the run exists, with its orch config, before the script starts.
    # finish() so the trainer is the only writer while it runs.
    run, run_dir = init_run(job, run_dir)
    update_config(run, job, state="running")
    run.finish()
    # The trainer resumes this run from its own process. The service that
    # served the prologue still owns the run id until it is shut down.
    release_wandb(wandb)
    code = tee_subprocess(
        job["command"],
        cwd=job["cluster"]["src_dir"],
        env=child_env(job),
        run_dir=run_dir,
    )
    run, run_dir = init_run(job, run_dir)
    model_dir = run_dir / "model"
    try:
        ready = code == 0 and model_ready(model_dir)
        update_config(
            run,
            job,
            state="succeeded" if ready else "failed",
            extra={"ended_at": now()},
        )
        if ready:
            artifact = wandb.Artifact(
                name=f"model-{job['fingerprint']}",
                type="model",
                metadata={
                    "fingerprint": job["fingerprint"],
                    "attempt": job["attempt"],
                    "run_id": job["run_id"],
                    "spec": job["spec_path"],
                    "entry": job["entry"],
                    "model_name": job.get("spec", {}).get("params", {}).get("model_name"),
                    "params": {
                        key: value
                        for key, value in job.get("spec", {}).get("params", {}).items()
                        if key not in set(job.get("spec", {}).get("runtime_keys") or [])
                    },
                    "git_sha": job.get("git", {}).get("sha"),
                    "cluster": job.get("cluster", {}).get("name"),
                },
            )
            artifact.add_dir(str(model_dir))
            logged = run.log_artifact(artifact, aliases=[f"a{job['attempt']}", "latest"])
            logged.wait()
        else:
            run.tags = list(run.tags) + ["failed"]
            run.summary["orch/exit_code"] = code
            if code == 0:
                run.summary["orch/failure_reason"] = "model_dir_missing"
        save_files(run, run_dir)
        run.finish(exit_code=code)
    except Exception:
        run.finish(exit_code=code or 1)
        raise
    return code
