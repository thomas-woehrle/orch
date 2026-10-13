"""status, logs, tag, and cancel. See docs/orchestrator.md §8."""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

from orch.backends import backend_for
from orch.cache import Cache
from orch.config import find_repo_root, load_local, load_orch, select_cluster
from orch.conn import open_conn
from orch.errors import UsageError
from orch.driver import LazyStates
from orch.plan import existing_attempt_ids, load_target

_RUN_ID = re.compile(r"^[0-9a-f]{12}-a\d+$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{12}$")
from orch.wb import WandbLookup


def _cache_and_lookup(repo: Path) -> tuple[Cache, WandbLookup]:
    local = load_local(repo)
    return Cache.load(local.wandb.entity, local.wandb.project), WandbLookup(
        local.wandb.entity, local.wandb.project
    )


def _runs_for_target(repo: Path, target: str, lookup: WandbLookup, cache: Cache) -> list[str]:
    """Expand a run id, fingerprint, spec, or experiment into existing run ids."""
    if _RUN_ID.fullmatch(target):
        return [target]
    path = Path(target)
    if not path.is_absolute():
        path = repo / target
    if path.is_file():
        kind, _data = load_target(path)
        if kind == "experiment" or path.suffix in {".yaml", ".yml"}:
            return existing_attempt_ids(
                repo_root=repo,
                target=path,
                entries=load_orch(repo).entries,
                probe=lookup.probe_attempts,
                cache=cache,
            )
    if _FINGERPRINT.fullmatch(target):
        numbers = sorted(
            set(lookup.probe_attempts(target))
            | {entry.attempt for entry in cache.for_fingerprint(target)}
        )
        return [f"{target}-a{number}" for number in numbers]
    return [target]


def assemble_status_row(
    run_id: str,
    *,
    cached,
    run,
    scheduler_state: str | None,
    entity: str,
    project: str,
) -> dict:
    """Prefer the run id and the W&B run over the local cache.

    The fingerprint and attempt are the run id. Name, kind, cluster, and
    scheduler id come from the run once the wrapper has started, and from the
    cache only for jobs that have not.
    """
    match = re.fullmatch(r"([0-9a-f]{12})-a(\d+)", run_id)
    orch = {}
    if run is not None:
        orch = (getattr(run, "config", None) or {}).get("orch") or {}
    cluster_block = orch.get("cluster") or {}
    fingerprint = match.group(1) if match else orch.get("fingerprint")
    attempt = int(match.group(2)) if match else orch.get("attempt")
    if fingerprint is None and cached is not None:
        fingerprint = cached.fingerprint
    if attempt is None and cached is not None:
        attempt = cached.attempt
    run_name = getattr(run, "name", None) if run is not None else None
    if not run_name and cached is not None:
        run_name = cached.name
    return {
        "run_id": run_id,
        "run_name": run_name,
        "fingerprint": fingerprint,
        "attempt": attempt,
        "kind": orch.get("kind") or (None if cached is None else cached.kind),
        "cluster": cluster_block.get("name") or (None if cached is None else cached.cluster),
        "decision": None,
        "reason": None,
        "scheduler_id": orch.get("scheduler_id") or (None if cached is None else cached.scheduler_id),
        "wandb_state": None if run is None else getattr(run, "state", None),
        "scheduler_state": scheduler_state,
        "url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
        "dependency": None,
    }


def _note_cache_override(cache: Cache) -> None:
    if os.environ.get("WANDB_ENTITY") or os.environ.get("WANDB_PROJECT"):
        print(
            f"cache: {cache.path} (WANDB_ENTITY/WANDB_PROJECT override orch.local.yaml)",
            file=sys.stderr,
        )


def _active_run_ids(cache: Cache, states: LazyStates, *, cluster_name: str | None) -> list[str]:
    clusters = sorted({entry.cluster for entry in cache.entries.values()})
    if cluster_name is not None:
        clusters = [name for name in clusters if name == cluster_name]
    found: list[str] = []
    for name in clusters:
        mapping = states(name) or {}
        for run_id, entry in cache.entries.items():
            if entry.cluster != name:
                continue
            if mapping.get(entry.scheduler_id, "gone") in {"queued", "running"}:
                found.append(run_id)
    return found


def status_rows(
    repo: Path,
    targets: list[str],
    *,
    no_remote: bool,
    all_entries: bool = False,
    cluster_name: str | None = None,
) -> list[dict]:
    local = load_local(repo)
    cache, lookup = _cache_and_lookup(repo)
    _note_cache_override(cache)

    def query(name: str) -> dict[str, str]:
        cluster = select_cluster(local, name)
        try:
            with open_conn(cluster) as conn:
                return backend_for(cluster).status_map(conn)
        except Exception as exc:
            print(f"warning: scheduler query failed for {name} ({exc})", file=sys.stderr, flush=True)
            return {}

    states = LazyStates(query, dry_run=no_remote)
    if targets:
        run_ids: list[str] = []
        for target in targets:
            run_ids.extend(_runs_for_target(repo, target, lookup, cache))
    elif all_entries:
        run_ids = [
            run_id
            for run_id, entry in cache.entries.items()
            if cluster_name is None or entry.cluster == cluster_name
        ]
    else:
        run_ids = [] if no_remote else _active_run_ids(cache, states, cluster_name=cluster_name)

    rows = []
    for run_id in run_ids:
        cached = cache.by_run_id(run_id)
        run = lookup.run(run_id)
        draft = assemble_status_row(
            run_id,
            cached=cached,
            run=run,
            scheduler_state=None,
            entity=local.wandb.entity,
            project=local.wandb.project,
        )
        sched = None
        if draft["cluster"] and draft["scheduler_id"]:
            if no_remote:
                sched = "unknown"
            else:
                mapping = states(draft["cluster"])
                sched = "unknown" if mapping is None else mapping.get(draft["scheduler_id"], "gone")
        draft["scheduler_state"] = sched
        rows.append(draft)
    if cluster_name is not None:
        rows = [row for row in rows if row.get("cluster") == cluster_name]
    return rows


_LOG_NAMES = {
    "stdout": ["stdout.log"],
    "stderr": ["stderr.log"],
    "scheduler.out": ["scheduler.out", "sge.out"],
    "scheduler.err": ["scheduler.err", "sge.err"],
    "sge.out": ["scheduler.out", "sge.out"],
    "sge.err": ["scheduler.err", "sge.err"],
}


def show_logs(
    repo: Path,
    run_id: str,
    *,
    which: str = "stdout",
    follow: bool = False,
) -> None:
    names = _LOG_NAMES.get(which)
    if names is None:
        raise UsageError(
            "--which must be stdout, stderr, scheduler.out, or scheduler.err"
        )
    local = load_local(repo)
    cache = Cache.load(local.wandb.entity, local.wandb.project)
    lookup = WandbLookup(local.wandb.entity, local.wandb.project)
    cached = cache.by_run_id(run_id)
    state = lookup.run_state(run_id)
    remote = state in {None, "running"} and cached is not None
    if remote:
        cluster = select_cluster(local, cached.cluster)
        follow_flag = "-f " if follow else ""
        with open_conn(cluster) as conn:
            path = next(
                (
                    f"{cluster.work_dir}/{run_id}/{name}"
                    for name in names
                    if conn.exists(f"{cluster.work_dir}/{run_id}/{name}")
                ),
                f"{cluster.work_dir}/{run_id}/{names[0]}",
            )
            conn.run(f"tail -n +1 {follow_flag}{shlex.quote(path)}", stream=True, check=False)
        return
    run = lookup.run(run_id)
    if run is None:
        raise UsageError(f"no W&B run {run_id}")
    match = None
    found_name = names[0]
    for item in run.files():
        for name in names:
            if item.name.endswith(name):
                match = item
                found_name = name
                break
        if match is not None:
            break
    if match is None:
        raise UsageError(f"{run_id} has no file {found_name}")
    name = found_name
    import tempfile

    dest = Path(tempfile.mkdtemp())
    match.download(root=str(dest), replace=True)
    found = next(dest.rglob(name))
    print(found.read_text(), end="")


def edit_tags(repo: Path, targets: list[str], *, add: list[str], remove: list[str]) -> list[str]:
    if not add and not remove:
        raise UsageError("pass --add or --remove")
    cache, lookup = _cache_and_lookup(repo)
    run_ids: list[str] = []
    for target in targets:
        run_ids.extend(_runs_for_target(repo, target, lookup, cache))
    changed = []
    for run_id in run_ids:
        run = lookup.run(run_id)
        if run is None:
            continue
        tags = [tag for tag in list(run.tags) if tag not in remove]
        for tag in add:
            if tag not in tags:
                tags.append(tag)
        run.tags = tags
        run.update()
        changed.append(run_id)
    return changed


def scheduler_miss(run_id: str, cache: Cache) -> str:
    """Why this argument cannot be passed to qdel."""
    attempts = cache.for_fingerprint(run_id)
    if attempts:
        shown = ", ".join(
            f"{run_id}-a{entry.attempt} (scheduler {entry.scheduler_id})" for entry in attempts
        )
        return f"fingerprint, not a run id. Pass {shown}"
    return "no scheduler job in the local cache"


def cancel_summaries(
    run_ids: list[str],
    cache: Cache,
    *,
    deleted: dict[str, str],
    tagged: set[str],
    wandb_missing: set[str],
) -> list[str]:
    """One line per argument. Raise when no scheduler job was deleted and no run was tagged."""
    lines: list[str] = []
    acted = False
    for run_id in run_ids:
        parts: list[str] = []
        if run_id in deleted:
            parts.append(f"qdel {deleted[run_id]}")
            acted = True
        else:
            parts.append(scheduler_miss(run_id, cache))
        if run_id in tagged:
            parts.append("tagged cancelled")
            acted = True
        elif run_id in wandb_missing:
            parts.append("no W&B run")
        lines.append(f"{run_id}: " + "; ".join(parts))
    if not acted:
        raise UsageError("\n".join(lines))
    return lines


def cancel_runs(repo: Path, run_ids: list[str]) -> list[str]:
    local = load_local(repo)
    cache = Cache.load(local.wandb.entity, local.wandb.project)
    lookup = WandbLookup(local.wandb.entity, local.wandb.project)
    deleted: dict[str, str] = {}
    by_cluster: dict[str, list[tuple[str, str]]] = {}
    for run_id in run_ids:
        cached = cache.by_run_id(run_id)
        if cached is None:
            continue
        by_cluster.setdefault(cached.cluster, []).append((run_id, cached.scheduler_id))
    for cluster_name, jobs in by_cluster.items():
        cluster = select_cluster(local, cluster_name)
        backend = backend_for(cluster)
        with open_conn(cluster) as conn:
            for run_id, scheduler_id in jobs:
                backend.cancel(conn, scheduler_id)
                deleted[run_id] = scheduler_id
    tagged: set[str] = set()
    wandb_missing: set[str] = set()
    for run_id in run_ids:
        run = lookup.run(run_id)
        if run is None:
            wandb_missing.add(run_id)
            continue
        tags = list(run.tags)
        if "cancelled" not in tags:
            tags.append("cancelled")
        run.tags = tags
        run.update()
        tagged.add(run_id)
    return cancel_summaries(
        run_ids,
        cache,
        deleted=deleted,
        tagged=tagged,
        wandb_missing=wandb_missing,
    )


def repo() -> Path:
    return find_repo_root()
