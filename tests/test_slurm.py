"""Slurm job scripts and squeue parsing. No cluster required."""

from types import SimpleNamespace

from orch.backends.slurm import SlurmBackend, parse_squeue
from orch.config import Cluster
from orch.conn import Result
from orch.render import build_env

HEADER = """#SBATCH -p gpu
#SBATCH --gres=gpu:{{ gpus }}
#SBATCH --time={{ time }}
#SBATCH --mem={{ mem }}
"""


def _cluster() -> Cluster:
    return Cluster(
        name="dmog",
        backend="slurm",
        ssh="dmog",
        repo_dir="/repo",
        work_dir="/work",
        header=HEADER,
        setup='source "${flight_ROOT:-/opt/flight}"/etc/setup.sh\n',
    )


def test_squeue_states():
    text = "\n".join(
        [
            "123|R|orch_abc_a1",
            "124|PD|orch_def_a1",
            "125|CG|finishing",
            "126|F|failed",
            "127|S|suspended",
            "128|CF|configuring",
        ]
    )
    assert parse_squeue(text) == {
        "123": "running",
        "124": "queued",
        "125": "running",
        "127": "queued",
        "128": "running",
    }


def test_sbatch_parses_cluster_suffix_and_afterok():
    class Conn:
        def __init__(self) -> None:
            self.command = ""

        def run(self, command: str, check: bool = True, stream: bool = False) -> Result:
            del check, stream
            self.command = command
            return Result(0, "12345;dmog\n", "")

    conn = Conn()
    assert SlurmBackend().submit(conn, "/work/fp-a1", ["10", "11"]) == "12345"
    assert "sbatch --parsable" in conn.command
    assert "--dependency=afterok:10:11" in conn.command
    assert "--kill-on-invalid-dep=yes" in conn.command


def test_golden_slurm_job_sh():
    env = build_env(
        cluster_env={"UV_CACHE_DIR": "/cache/uv"},
        entry_env={},
        entity="ent",
        project="proj",
        job_kind="train",
        run_id="3f9a2c1d7e4b-a1",
        run_name="sdft",
        tags=["kind:train"],
        group=None,
        run_dir="/work/3f9a2c1d7e4b-a1",
    )
    job = SimpleNamespace(
        run_id="3f9a2c1d7e4b-a1",
        fingerprint="3f9a2c1d7e4b",
        attempt=1,
        resources={"gpus": 1, "time": "12:00:00", "mem": "24G"},
    )
    script = SlurmBackend().render_job_script(
        job, _cluster(), env, "/repo/.worktrees/abc", "/repo/.venv/bin/python", []
    )
    assert script.startswith("#!/bin/bash -l\n")
    assert "#SBATCH -J orch_3f9a2c1d7e4b_a1\n" in script
    assert "#SBATCH -o /work/3f9a2c1d7e4b-a1/scheduler.out\n" in script
    assert "#SBATCH --gres=gpu:1\n" in script
    assert "#SBATCH --time=12:00:00\n" in script
    assert 'source "${flight_ROOT:-/opt/flight}"/etc/setup.sh\n' in script
    assert script.rstrip().endswith(
        "exec /repo/.venv/bin/python -m orch exec /work/3f9a2c1d7e4b-a1"
    )
