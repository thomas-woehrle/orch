"""Scheduler state is per cluster. See docs/orchestrator.md §6.2 and §8."""

from pathlib import Path

from orch.cache import Cache, CacheEntry
from orch.config import load_orch
from orch.driver import LazyStates
from orch.ops import assemble_status_row
from orch.plan import build_jobs, existing_attempt_ids
from tests.test_plan import _build, cluster


def test_attempt_queued_elsewhere_is_pending_and_blocks_evals():
    root = Path(__file__).resolve().parents[1]
    first = _build(root, "configs/train/sft_tooluse.yaml", no_evals=True)
    fingerprint = first[0].fingerprint
    cache = Cache(None)
    cache.upsert(
        f"{fingerprint}-a1",
        CacheEntry("other", "9", fingerprint, 1, "train", "name", "2026-01-01T00:00:00Z"),
    )
    calls: list[str] = []

    def states(name: str):
        calls.append(name)
        if name == "other":
            return {"9": "queued"}
        return {}

    jobs = build_jobs(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        cluster=cluster(),
        cache=cache,
        probe=lambda _fp: {},
        run_state=lambda _run: None,
        scheduler_states=states,
    )
    train = next(job for job in jobs if job.kind == "train")
    assert (train.decision, train.reason) == ("skip", "pending")
    assert "eddie" not in calls
    children = [job for job in jobs if job.kind == "eval"]
    assert children
    assert all(job.decision == "skip" and job.reason == "upstream-not-available" for job in children)


def test_pending_on_the_target_cluster_still_plans_evals():
    root = Path(__file__).resolve().parents[1]
    first = _build(root, "configs/train/sft_tooluse.yaml", no_evals=True)
    fingerprint = first[0].fingerprint
    cache = Cache(None)
    cache.upsert(
        f"{fingerprint}-a1",
        CacheEntry("eddie", "9", fingerprint, 1, "train", "name", "2026-01-01T00:00:00Z"),
    )

    def states(name: str):
        return {"9": "queued"} if name == "eddie" else {}

    jobs = build_jobs(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        cluster=cluster(),
        cache=cache,
        probe=lambda _fp: {},
        run_state=lambda _run: None,
        scheduler_states=states,
    )
    train = next(job for job in jobs if job.kind == "train")
    assert train.reason == "pending"
    children = [job for job in jobs if job.upstream_run_id == train.run_id]
    assert children and all(job.decision == "submit" for job in children)


def test_dry_run_states_do_not_query():
    def query(name: str):
        raise AssertionError(name)

    states = LazyStates(query, dry_run=True)
    assert states("eddie") is None
    assert states("other") is None


def test_existing_attempts_skip_the_attempt_that_would_be_submitted_next():
    root = Path(__file__).resolve().parents[1]
    fingerprint = _build(root, "configs/train/sft_tooluse.yaml", no_evals=True)[0].fingerprint

    def probe(fp: str) -> dict[int, str]:
        if fp == fingerprint:
            return {1: "failed"}
        return {}

    ids = existing_attempt_ids(
        repo_root=root,
        target=root / "configs/train/sft_tooluse.yaml",
        entries=load_orch(root).entries,
        probe=probe,
        cache=Cache(None),
    )
    assert ids == [f"{fingerprint}-a1"]


def test_status_row_uses_wandb_when_the_cache_has_no_entry():
    class Run:
        name = "sft_tooluse model=Qwen2.5-1.5B-Instruct"
        state = "running"
        config = {
            "orch": {
                "kind": "train",
                "fingerprint": "1d6f67bdec81",
                "attempt": 1,
                "cluster": {"name": "eddie"},
                "scheduler_id": "42",
            }
        }

    row = assemble_status_row(
        "1d6f67bdec81-a1",
        cached=None,
        run=Run(),
        scheduler_state="running",
        entity="ent",
        project="proj",
    )
    assert row["run_name"].startswith("sft_tooluse")
    assert row["kind"] == "train"
    assert row["fingerprint"] == "1d6f67bdec81"
    assert row["attempt"] == 1
    assert row["cluster"] == "eddie"
    assert row["scheduler_id"] == "42"
    assert row["wandb_state"] == "running"
