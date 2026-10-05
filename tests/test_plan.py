"""Experiment expansion and dedup."""

from pathlib import Path

from orch.cache import Cache
from orch.config import Cluster, load_orch
from orch.plan import build_jobs

HEADER = """#$ -q gpu
#$ -l gpu={{ gpus }}
{% if gpu_type %}#$ -l {{ gpu_type }}=true{% endif %}
#$ -l h_rt={{ h_rt }}
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


def test_tooluse_main_expands_ablation_grid():
    root = Path(__file__).resolve().parents[1]
    jobs = _build(root, "experiments/tooluse_main.yaml")
    trains = [job for job in jobs if job.kind == "train"]
    evals = [job for job in jobs if job.kind == "eval"]
    # 2 methods x 2 learning rates x 2 batch sizes on Qwen2.5-3B-Instruct,
    # plus the same SDFT grid with an alternate teacher prompt. Each training
    # run has two held evals, and there are two base-model evals.
    assert len(trains) == 12
    assert len(evals) == 26
    assert len(jobs) == 38
    assert all(job.decision == "submit" and job.attempt == 1 for job in jobs)
    grid = {
        (
            job.params["method"],
            job.params["model_name"],
            job.params["learning_rate"],
            job.params["gradient_accumulation_steps"],
        )
        for job in trains
    }
    assert grid == {
        (method, "Qwen/Qwen2.5-3B-Instruct", lr, batch)
        for method in ("sdft", "sft")
        for lr in (1.0e-5, 5.0e-5)
        for batch in (16, 32)
    }
    alternate_prompt = (
        "{prompt}\n"
        "This is an example for a response to the question:\n"
        "{privileged_context}\n"
        "\n"
        "Now answer with a response of your own. The first words of your response are filled out by another, less capable agent"
    )
    sdft = [job for job in trains if job.params["method"] == "sdft"]
    alternate = [job for job in sdft if job.params["teacher_prompt_template"] == alternate_prompt]
    paper = [job for job in sdft if job.params["teacher_prompt_template"] != alternate_prompt]
    cells = {
        (1.0e-5, 16),
        (1.0e-5, 32),
        (5.0e-5, 16),
        (5.0e-5, 32),
    }
    assert {
        (job.params["learning_rate"], job.params["gradient_accumulation_steps"]) for job in alternate
    } == cells
    assert {
        (job.params["learning_rate"], job.params["gradient_accumulation_steps"]) for job in paper
    } == cells
    assert all("teacher_prompt_template" not in job.params for job in trains if job.params["method"] == "sft")
    for train in trains:
        children = [job for job in evals if job.upstream_run_id == train.run_id]
        assert len(children) == 2
        assert {job.stem for job in children} == {"tooluse", "prior_tasks"}
    base = [job for job in evals if job.upstream_run_id is None]
    assert {job.model_ref for job in base} == {"hf:Qwen/Qwen2.5-3B-Instruct"}
    assert len(base) == 2


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
