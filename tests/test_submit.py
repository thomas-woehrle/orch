"""Submission order, holds, and the run-directory ledger."""

import json
from pathlib import Path

from orch.backends.sge import SGEBackend
from orch.cache import Cache
from orch.config import Cluster, load_orch
from orch.driver import submit_jobs
from orch.plan import build_jobs
from orch.ssh import Result

HEADER = """#$ -q gpu
#$ -l gpu={{ gpus }}
{% if gpu_type %}#$ -l {{ gpu_type }}=true{% endif %}
#$ -l h_rt={{ h_rt }}
#$ -l h_rss={{ mem }}
"""


class MemSSH:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        self._next = 100

    def run(self, command: str, check: bool = True, stream: bool = False) -> Result:
        del check, stream
        self.commands.append(command)
        if command.startswith("qsub"):
            self._next += 1
            return Result(0, f"{self._next}\n", "")
        return Result(0, "", "")

    def put_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def exists(self, path: str) -> bool:
        prefix = path.rstrip("/") + "/"
        return path in self.files or any(key.startswith(prefix) for key in self.files)


def _cluster() -> Cluster:
    return Cluster(
        name="eddie",
        backend="sge",
        ssh="eddie",
        repo_dir="/repo",
        work_dir="/work",
        header=HEADER,
        setup="module load cuda\n",
        env={"UV_CACHE_DIR": "/cache/uv"},
    )


def test_submit_trains_before_evals_and_holds():
    root = Path(__file__).resolve().parents[1]
    cluster = _cluster()
    jobs = build_jobs(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        cluster=cluster,
        cache=Cache(None),
        probe=lambda _fp: {},
        run_state=lambda _run: None,
    )
    conn = MemSSH()
    submit_jobs(
        jobs,
        conn=conn,
        backend=SGEBackend(),
        cluster=cluster,
        entries=load_orch(root).entries,
        cache=Cache(None),
        qstat={},
        entity="ent",
        project="proj",
        sha="abc123",
        remote="https://github.com/thomas-woehrle/sdft-exp.git",
        probe=lambda _fp: {},
    )
    qsubs = [command for command in conn.commands if command.startswith("qsub")]
    assert len(qsubs) == 3
    assert "hold_jid" not in qsubs[0]
    train_id = qsubs[0].split()[-1].split("/")[-2]
    assert train_id.endswith("-a1")
    assert all(f"-hold_jid {next(id_ for id_ in [str(101)])}" in command or "-hold_jid 101" in command for command in qsubs[1:])
    script = conn.files[f"/work/{train_id}/job.sh"]
    assert "python -m orch exec" in script
    assert "module load cuda" in script
    params = conn.files[f"/work/{train_id}/params.yaml"]
    assert "learning_rate" in params
    document = json.loads(conn.files[f"/work/{train_id}/job.json"])
    assert document["git"]["sha"] == "abc123"
    assert document["command"].startswith("python train.py --config ")
    eval_paths = [path for path in conn.files if path.endswith("job.json") and train_id not in path]
    eval_job = json.loads(conn.files[eval_paths[0]])
    assert "{model_path}" in eval_job["command"]
    assert eval_job["upstream_run_id"].endswith("-a1")


def test_ledger_bumps_attempt_when_run_dir_exists():
    root = Path(__file__).resolve().parents[1]
    cluster = _cluster()
    jobs = build_jobs(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        cluster=cluster,
        cache=Cache(None),
        probe=lambda _fp: {},
        run_state=lambda _run: None,
        no_evals=True,
    )
    conn = MemSSH()
    conn.files[f"/work/{jobs[0].run_id}/job.json"] = "{}"
    submit_jobs(
        jobs,
        conn=conn,
        backend=SGEBackend(),
        cluster=cluster,
        entries=load_orch(root).entries,
        cache=Cache(None),
        qstat={},
        entity="ent",
        project="proj",
        sha="abc",
        remote="https://example.com/repo.git",
        probe=lambda _fp: {},
    )
    assert jobs[0].attempt == 2
    assert jobs[0].run_id.endswith("-a2")
    assert "/work/" + jobs[0].run_id + "/job.sh" in conn.files
