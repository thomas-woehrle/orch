"""Expand specs and experiments into jobs. See docs/orchestrator.md §4.5 and §6.1."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from orch.cache import Cache
from orch.config import Cluster, Entry
from orch.errors import UsageError
from orch.identity import canonical_json, fingerprint_hex, fingerprint_input
from orch.modelref import ModelLookup
from orch.naming import format_run_name
from orch.reconcile import Decision, attempt_views, decide, upstream_allows
from orch.render import render_command, validate_resources
from orch.resolve import Override, ResolvedSpec, resolve_spec
from orch.yamlio import load_path

Probe = Callable[[str], dict[int, str]]
RunState = Callable[[str], str | None]


@dataclass
class Job:
    run_id: str
    run_name: str
    fingerprint: str
    attempt: int
    kind: str
    entry: str
    spec_path: str
    stem: str
    spec: dict[str, Any]
    overrides: list[dict[str, Any]]
    params: dict[str, Any]
    resources: dict[str, Any]
    model_ref: str | None
    upstream_run_id: str | None
    model_handle: str | None
    name_segments: list[tuple[str, Any]]
    override_segments: list[tuple[str, Any]]
    fingerprint_input: dict[str, Any]
    canonical: str
    entry_cmd: str
    command: str
    parser: str | None
    entry_env: dict[str, str]
    tags: list[str]
    group: str | None
    decision: str
    reason: str
    scheduler_id: str | None = None
    hold_on: list[str] = field(default_factory=list)
    wandb_state: str | None = None
    scheduler_state: str | None = None

    def row(self, *, entity: str, project: str) -> dict[str, Any]:
        url = f"https://wandb.ai/{entity}/{project}/runs/{self.run_id}"
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "fingerprint": self.fingerprint,
            "attempt": self.attempt,
            "kind": self.kind,
            "decision": self.decision,
            "reason": self.reason,
            "scheduler_id": self.scheduler_id,
            "wandb_state": self.wandb_state,
            "scheduler_state": self.scheduler_state,
            "url": url,
            "resources": self.resources if self.decision == "submit" else None,
            "dependency": self.upstream_run_id,
            "overrides": list(self.overrides),
        }


@dataclass
class _Item:
    spec_path: Path
    extra_set: dict[str, Any] | None
    model: str | None
    evals: list[str] | None
    tags: list[str]


def load_target(path: Path) -> tuple[str, Any]:
    """Return ``('run', None)`` or ``('experiment', data)``."""
    if not path.is_file():
        raise UsageError(f"spec not found: {path}")
    data = load_path(path)
    if isinstance(data, dict) and "runs" in data and "kind" not in data:
        return "experiment", data
    return "run", data


def _string_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise UsageError(f"{what} must be a list of strings")
    return list(value)


def _items_from_experiment(path: Path, data: dict[str, Any], repo_root: Path) -> tuple[list[_Item], list[str]]:
    unknown = set(data) - {"version", "description", "runs", "tags"}
    if unknown:
        raise UsageError(f"unknown experiment key: {', '.join(sorted(unknown))}")
    if data.get("version") not in (None, 1):
        raise UsageError("experiment version must be 1")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs:
        raise UsageError(f"{path} has no runs")
    shared = _string_list(data.get("tags"), "experiment tags")
    items: list[_Item] = []
    for index, raw in enumerate(runs):
        if not isinstance(raw, dict):
            raise UsageError(f"runs[{index}] must be a mapping")
        extra = set(raw) - {"config", "set", "model", "evals", "tags"}
        if extra:
            raise UsageError(f"runs[{index}] has unknown keys: {', '.join(sorted(extra))}")
        config = raw.get("config")
        if not isinstance(config, str) or not config:
            raise UsageError(f"runs[{index}] is missing config")
        set_map = raw.get("set")
        if set_map is not None and not isinstance(set_map, dict):
            raise UsageError(f"runs[{index}].set must be a mapping")
        model = raw.get("model")
        if model is not None and not isinstance(model, str):
            raise UsageError(f"runs[{index}].model must be a string")
        evals = raw.get("evals")
        if evals is not None:
            evals = _string_list(evals, f"runs[{index}].evals")
        tags = shared + _string_list(raw.get("tags"), f"runs[{index}].tags")
        items.append(
            _Item(
                spec_path=repo_root / config,
                extra_set=set_map,
                model=model,
                evals=evals,
                tags=tags,
            )
        )
    return items, shared


def _auto_tags(resolved: ResolvedSpec, cluster_name: str, attempt: int, user_tags: list[str]) -> list[str]:
    tags = [
        f"kind:{resolved.kind}",
        f"spec:{resolved.stem}",
        f"entry:{resolved.entry}",
        f"cluster:{cluster_name}",
        f"attempt:{attempt}",
    ]
    gpu_type = resolved.resources.get("gpu_type")
    if gpu_type:
        tags.append(f"gpu_type:{gpu_type}")
    for tag in user_tags:
        if tag not in tags:
            tags.append(tag)
    return tags


def _job_from_resolved(
    resolved: ResolvedSpec,
    *,
    decision: Decision,
    cluster: Cluster,
    entries: dict[str, Entry],
    user_tags: list[str],
    group: str | None,
    wb_attempts: dict[int, str],
    scheduler_state: str | None,
    wandb_state: str | None,
) -> Job:
    attempt = decision.attempt
    run_id = f"{resolved.fingerprint}-a{attempt}"
    run_dir = f"{cluster.work_dir}/{run_id}"
    entry = entries[resolved.entry]
    handle = resolved.model.handle if resolved.model is not None else None
    name = format_run_name(
        resolved.stem,
        resolved.name_segments,
        resolved.override_segments,
        attempt=attempt,
        model_handle=handle,
    )
    tags = _auto_tags(resolved, cluster.name, attempt, user_tags)
    return Job(
        run_id=run_id,
        run_name=name,
        fingerprint=resolved.fingerprint,
        attempt=attempt,
        kind=resolved.kind,
        entry=resolved.entry,
        spec_path=resolved.spec_path,
        stem=resolved.stem,
        spec=resolved.spec_dict(),
        overrides=list(resolved.overrides),
        params=resolved.params,
        resources=resolved.resources,
        model_ref=None if resolved.model is None else resolved.model.canonical,
        upstream_run_id=None if resolved.model is None else resolved.model.upstream_run_id,
        model_handle=handle,
        name_segments=list(resolved.name_segments),
        override_segments=list(resolved.override_segments),
        fingerprint_input=resolved.fingerprint_input,
        canonical=resolved.canonical,
        entry_cmd=entry.cmd,
        command=render_command(entry.cmd, run_dir=run_dir, params=resolved.params),
        parser=entry.parser,
        entry_env=dict(entry.env),
        tags=tags,
        group=group,
        decision=decision.action,
        reason=decision.reason,
        wandb_state=wandb_state,
        scheduler_state=scheduler_state,
    )


def build_jobs(
    *,
    repo_root: Path,
    target: Path,
    entries: dict[str, Entry],
    cluster: Cluster,
    cli_overrides: list[Override] | None = None,
    model_ref: str | None = None,
    evals_override: list[str] | None = None,
    no_evals: bool = False,
    tags: list[str] | None = None,
    group: str | None = None,
    force: bool = False,
    lookup: ModelLookup | None = None,
    cache: Cache | None = None,
    probe: Probe,
    run_state: RunState,
    qstat: dict[str, str] | None = None,
) -> list[Job]:
    """Resolve, expand, dedup, and reconcile. Does not touch a cluster filesystem."""
    if no_evals and evals_override is not None:
        raise UsageError("pass only one of --evals and --no-evals")
    cache = cache or Cache(None)
    cli_overrides = list(cli_overrides or [])
    user_tags = list(tags or [])
    probe_cache: dict[str, dict[int, str]] = {}
    state_cache: dict[str, str | None] = {}

    def cached_probe(fingerprint: str) -> dict[int, str]:
        if fingerprint not in probe_cache:
            probe_cache[fingerprint] = probe(fingerprint)
        return probe_cache[fingerprint]

    def cached_state(run_id: str) -> str | None:
        if run_id not in state_cache:
            state_cache[run_id] = run_state(run_id)
        return state_cache[run_id]

    kind, data = load_target(target)
    if kind == "experiment":
        if model_ref:
            raise UsageError("an experiment spec carries its own model references; do not pass --model")
        items, _shared = _items_from_experiment(target, data, repo_root)
        for item in items:
            for tag in user_tags:
                if tag not in item.tags:
                    item.tags.append(tag)
    else:
        items = [
            _Item(
                spec_path=target,
                extra_set=None,
                model=model_ref,
                evals=evals_override,
                tags=user_tags,
            )
        ]

    # First pass: resolve trains (and remember standalone eval items).
    trains: dict[str, tuple[ResolvedSpec, list[str], list[str]]] = {}
    train_order: list[str] = []
    standalone: list[_Item] = []
    for item in items:
        resolved = resolve_spec(
            item.spec_path,
            repo_root=repo_root,
            entries=entries,
            cli_overrides=cli_overrides,
            extra_set=item.extra_set,
            model_ref=item.model,
            lookup=lookup,
            require_model=item.model is not None or kind == "run",
        )
        # A run-spec invocation of an eval without --model already failed above
        # when require_model is true. Experiment evals pass model.
        if resolved.kind == "train" and item.model:
            raise UsageError("train spec does not take a model reference")
        if resolved.kind == "eval" and not item.model:
            raise UsageError(f"eval spec {resolved.spec_path} requires a model reference")
        validate_resources(resolved.resources, cluster.header)
        if resolved.kind == "train":
            eval_paths = list(resolved.evals)
            if item.evals is not None:
                eval_paths = list(item.evals)
            if evals_override is not None:
                eval_paths = list(evals_override)
            if no_evals:
                eval_paths = []
            bucket = trains.get(resolved.fingerprint)
            if bucket is None:
                trains[resolved.fingerprint] = (resolved, eval_paths, list(item.tags))
                train_order.append(resolved.fingerprint)
            else:
                existing, paths, existing_tags = bucket
                for rel in eval_paths:
                    if rel not in paths:
                        paths.append(rel)
                for tag in item.tags:
                    if tag not in existing_tags:
                        existing_tags.append(tag)
                trains[resolved.fingerprint] = (existing, paths, existing_tags)
        else:
            standalone.append(item)

    def reconcile(resolved: ResolvedSpec, *, item_force: bool) -> tuple[Decision, dict[int, str], str | None, str | None]:
        wb_attempts = cached_probe(resolved.fingerprint)
        views = attempt_views(wb_attempts, cache.for_fingerprint(resolved.fingerprint), qstat)
        decision = decide(views, force=item_force)
        current = next((view for view in views if view.attempt == decision.attempt), None)
        # For a brand new submit there is no view at that attempt.
        if decision.action == "submit":
            previous = next((view for view in views if view.attempt == decision.attempt - 1), None)
            sched = None
            wandb_state = None
            if previous is not None and decision.attempt - 1 in wb_attempts:
                wandb_state = wb_attempts.get(decision.attempt - 1)
            return decision, wb_attempts, wandb_state, sched
        sched = None if current is None else current.scheduler_state
        return decision, wb_attempts, wb_attempts.get(decision.attempt), sched

    train_jobs: dict[str, Job] = {}
    for fingerprint in train_order:
        resolved, _paths, item_tags = trains[fingerprint]
        decision, wb_attempts, wandb_state, sched = reconcile(resolved, item_force=force)
        train_jobs[fingerprint] = _job_from_resolved(
            resolved,
            decision=decision,
            cluster=cluster,
            entries=entries,
            user_tags=item_tags,
            group=group,
            wb_attempts=wb_attempts,
            scheduler_state=sched,
            wandb_state=wandb_state if decision.action == "skip" else None,
        )

    eval_resolved: list[tuple[ResolvedSpec, list[str], Decision]] = []

    def add_eval(resolved: ResolvedSpec, item_tags: list[str]) -> None:
        # Dedup by fingerprint, union tags onto the first copy.
        for index, (existing, tags_so_far, _decision) in enumerate(eval_resolved):
            if existing.fingerprint == resolved.fingerprint:
                for tag in item_tags:
                    if tag not in tags_so_far:
                        tags_so_far.append(tag)
                eval_resolved[index] = (existing, tags_so_far, _decision)
                return
        decision, _wb, _state, _sched = reconcile(resolved, item_force=force)
        if resolved.model is not None and resolved.model.upstream_run_id:
            upstream = _upstream_decision(
                resolved.model.upstream_run_id,
                train_jobs,
                cached_state,
                cache,
                qstat,
            )
            if decision.action == "submit" and not upstream_allows(upstream):
                decision = Decision("skip", decision.attempt, "upstream-not-available")
        eval_resolved.append((resolved, item_tags, decision))

    for fingerprint in train_order:
        resolved, paths, item_tags = trains[fingerprint]
        parent = train_jobs[fingerprint]
        for rel in paths:
            child = resolve_spec(
                repo_root / rel,
                repo_root=repo_root,
                entries=entries,
                model_ref=f"run:{parent.run_id}",
                lookup=lookup,
            )
            if child.model is not None:
                child.model = dataclasses.replace(child.model, handle=parent.run_name)
                child_input = fingerprint_input(
                    child.kind,
                    child.entry,
                    child.params,
                    child.runtime_keys,
                    child.model.canonical,
                )
                # Handle is not part of the fingerprint; canonical model ref is.
                # Recompute only if we changed the canonical form (we did not).
                assert child_input["model_ref"] == child.fingerprint_input["model_ref"]
            validate_resources(child.resources, cluster.header)
            add_eval(child, list(item_tags))

    for item in standalone:
        resolved = resolve_spec(
            item.spec_path,
            repo_root=repo_root,
            entries=entries,
            cli_overrides=cli_overrides,
            extra_set=item.extra_set,
            model_ref=item.model,
            lookup=lookup,
        )
        validate_resources(resolved.resources, cluster.header)
        add_eval(resolved, list(item.tags))

    jobs: list[Job] = []
    emitted_trains: set[str] = set()
    # Preserve experiment order: each train, then the evals that cite it, and
    # standalone evals where they were declared. Deduped evals stay at the
    # first position.
    eval_jobs = []
    for resolved, item_tags, decision in eval_resolved:
        wb_attempts = cached_probe(resolved.fingerprint)
        views = attempt_views(wb_attempts, cache.for_fingerprint(resolved.fingerprint), qstat)
        current = next((view for view in views if view.attempt == decision.attempt), None)
        sched = None if current is None else current.scheduler_state
        wandb_state = wb_attempts.get(decision.attempt) if decision.action == "skip" else None
        if decision.reason == "upstream-not-available":
            wandb_state = wb_attempts.get(decision.attempt)
        eval_jobs.append(
            _job_from_resolved(
                resolved,
                decision=decision,
                cluster=cluster,
                entries=entries,
                user_tags=item_tags,
                group=group,
                wb_attempts=wb_attempts,
                scheduler_state=sched,
                wandb_state=wandb_state,
            )
        )

    evals_by_upstream: dict[str | None, list[Job]] = {}
    for job in eval_jobs:
        evals_by_upstream.setdefault(job.upstream_run_id, []).append(job)
    placed: set[str] = set()
    for fingerprint in train_order:
        parent = train_jobs[fingerprint]
        jobs.append(parent)
        emitted_trains.add(fingerprint)
        for child in evals_by_upstream.get(parent.run_id, []):
            if child.fingerprint in placed:
                continue
            jobs.append(child)
            placed.add(child.fingerprint)
    for job in eval_jobs:
        if job.fingerprint not in placed:
            jobs.append(job)
            placed.add(job.fingerprint)
    return jobs


def _upstream_decision(
    run_id: str,
    train_jobs: dict[str, Job],
    run_state: RunState,
    cache: Cache,
    qstat: dict[str, str] | None,
) -> Decision:
    for job in train_jobs.values():
        if job.run_id == run_id:
            return Decision(job.decision, job.attempt, job.reason)
    cached = cache.by_run_id(run_id)
    state = run_state(run_id)
    if state == "finished":
        return Decision("skip", _attempt_of(run_id), "finished")
    if state == "running" or (cached is not None and _cached_in_flight(cached, qstat)):
        return Decision("skip", _attempt_of(run_id), "running" if state == "running" else "pending")
    if state in {"failed", "crashed", "killed"}:
        return Decision("skip", _attempt_of(run_id), state)
    return Decision("skip", _attempt_of(run_id), "unavailable")


def _attempt_of(run_id: str) -> int:
    suffix = run_id.rsplit("-a", 1)[-1]
    return int(suffix) if suffix.isdigit() else 1


def _cached_in_flight(entry: Any, qstat: dict[str, str] | None) -> bool:
    if qstat is None:
        return True
    return qstat.get(entry.scheduler_id, "gone") in {"queued", "running", "unknown"}


def retarget_eval(job: Job, *, run_id: str, handle: str) -> None:
    """Point an eval at a train run id chosen after the run-directory ledger bump."""
    if job.model_ref is None or not job.model_ref.startswith("run:"):
        return
    payload = fingerprint_input(
        job.kind,
        job.entry,
        job.spec["params"],
        job.spec["runtime_keys"],
        f"run:{run_id}",
    )
    canonical = canonical_json(payload)
    fingerprint = fingerprint_hex(canonical)
    job.fingerprint = fingerprint
    job.fingerprint_input = payload
    job.canonical = canonical
    job.model_ref = f"run:{run_id}"
    job.upstream_run_id = run_id
    job.model_handle = handle
    job.run_name = format_run_name(
        job.stem,
        job.name_segments,
        job.override_segments,
        attempt=job.attempt,
        model_handle=handle,
    )


def refresh_identity(job: Job) -> None:
    """Recompute run id, name, command paths, and the attempt tag after an attempt bump."""
    job.run_id = f"{job.fingerprint}-a{job.attempt}"
    job.run_name = format_run_name(
        job.stem,
        job.name_segments,
        job.override_segments,
        attempt=job.attempt,
        model_handle=job.model_handle,
    )
    # command paths embed the run id; rebuild from the entry command is the
    # caller's job because the template is not stored. Replace the old run dir
    # suffix by rewriting the trailing run id segment.
    job.tags = [
        f"attempt:{job.attempt}" if tag.startswith("attempt:") else tag for tag in job.tags
    ]
