"""``orch exec`` on the compute node."""

from __future__ import annotations

from pathlib import Path


def execute(run_dir: Path) -> int:
    import json

    job = json.loads((run_dir / "job.json").read_text())
    if job["kind"] == "train":
        from orch.exec.train import execute as run

        return run(run_dir)
    from orch.exec.eval import execute as run

    return run(run_dir)
