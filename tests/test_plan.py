"""Experiment expansion and dedup."""

from pathlib import Path

import pytest

from orch.cache import Cache
from orch.config import Cluster, load_orch
from orch.errors import UsageError
from orch.plan import build_jobs
from orch.render import job_document

HEADER = """#$ -q gpu
#$ -l gpu={{ gpus }}
{% if gpu_type %}#$ -l {{ gpu_type }}=true{% endif %}
#$ -l h_rt={{ time }}
#$ -l h_rss={{ mem }}
"""


def cluster() -> Cluster:
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


def _build(root: Path, target: str, **kwargs):
    return build_jobs(
        repo_root=root,
        target=root / target,
        entries=load_orch(root).entries,
        cluster=cluster(),
        cache=Cache(None),
        probe=lambda _fp: {},
        run_state=lambda _run: None,
        **kwargs,
    )


def test_dedup_unions_evals(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        """\
version: 1
runs:
  - config: configs/train/sft_tooluse.yaml
    evals: [configs/eval/tooluse.yaml]
  - config: configs/train/sft_tooluse.yaml
    evals: [configs/eval/prior_tasks.yaml]
"""
    )
    jobs = _build(root, str(experiment))
    trains = [job for job in jobs if job.kind == "train"]
    assert len(trains) == 1
    children = [job for job in jobs if job.upstream_run_id == trains[0].run_id]
    assert {job.stem for job in children} == {"tooluse", "prior_tasks"}


def _document(job):
    return job_document(
        job=job,
        cluster_name="eddie",
        backend="sge",
        repo_dir="/repo",
        work_dir="/work",
        src_dir="/src",
        venv_python="/venv/bin/python",
        sha="abc",
        remote="origin",
        env={},
        entity="e",
        project="p",
        submitted_from="host",
        submitted_at="2026-01-01T00:00:00Z",
    )


def test_tooluse_train_still_schedules_child_evals():
    root = Path(__file__).resolve().parents[1]
    jobs = _build(root, "configs/train/sft_tooluse.yaml")
    train = next(job for job in jobs if job.kind == "train")
    assert train.artifact is True
    assert _document(train)["artifact"] is True
    children = [job for job in jobs if job.kind == "eval"]
    assert {job.stem for job in children} == {"tooluse", "prior_tasks"}


def test_sequential_train_has_no_child_evals():
    root = Path(__file__).resolve().parents[1]
    jobs = _build(root, "configs/train/sequential_sdft.yaml")
    assert [job.kind for job in jobs] == ["train"]
    assert jobs[0].artifact is False
    assert _document(jobs[0])["artifact"] is False
    cleared = _build(root, "configs/train/sequential_sdft.yaml", no_evals=True)
    assert [job.kind for job in cleared] == ["train"]


def test_sequential_experiment_is_two_trains_without_evals():
    root = Path(__file__).resolve().parents[1]
    jobs = _build(root, "experiments/sequential_tooluse_science.yaml")
    assert [job.kind for job in jobs] == ["train", "train"]
    assert {job.params["method"] for job in jobs} == {"sdft", "sft"}
    assert all(job.artifact is False for job in jobs)
    assert all(job.params["model_name"] == "Qwen/Qwen2.5-3B-Instruct" for job in jobs)
    sdft = next(job for job in jobs if job.params["method"] == "sdft")
    assert [phase["name"] for phase in sdft.params["phases"]] == ["tooluse", "science"]
    assert sdft.params["phases"][1]["train_data"] == "data/science_data/train_data"
    assert sdft.params["eval_steps"] == 50


def test_artifact_false_rejects_attached_evals(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(UsageError, match="cannot schedule evals"):
        _build(
            root,
            "configs/train/sequential_sdft.yaml",
            evals_override=["configs/eval/tooluse.yaml"],
        )
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        """\
version: 1
runs:
  - config: configs/train/sequential_sft.yaml
    evals: [configs/eval/science.yaml]
"""
    )
    with pytest.raises(UsageError, match="cannot schedule evals"):
        _build(root, str(experiment))


def test_no_evals_and_finished_upstream_skips_resubmit():
    root = Path(__file__).resolve().parents[1]
    jobs = _build(root, "configs/train/sft_tooluse.yaml", no_evals=True)
    assert len(jobs) == 1
    assert jobs[0].kind == "train"

    fingerprint = jobs[0].fingerprint

    def probe(fp: str) -> dict[int, str]:
        if fp == fingerprint:
            return {1: "finished"}
        return {}

    again = build_jobs(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        cluster=cluster(),
        cache=Cache(None),
        probe=probe,
        run_state=lambda run: "finished" if run.endswith("-a1") else None,
        no_evals=False,
    )
    train = next(job for job in again if job.kind == "train")
    assert train.decision == "skip" and train.reason == "finished"
    children = [job for job in again if job.upstream_run_id == train.run_id]
    assert children and all(job.decision == "submit" for job in children)
    # The held evals target the finished attempt, not a new one.
    assert all(job.upstream_run_id == f"{fingerprint}-a1" for job in children)
