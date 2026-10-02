"""status, logs, tag, and cancel. See docs/orchestrator.md §8."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from orch.cache import Cache
from orch.config import find_repo_root, load_local, load_orch, select_cluster
from orch.errors import UsageError
from orch.plan import build_jobs, load_target
from orch.wb import WandbLookup


def _cache_and_lookup(repo: Path) -> tuple[Cache, WandbLookup]:
    local = load_local(repo)
    return Cache.load(local.wandb.entity, local.wandb.project), WandbLookup(
        local.wandb.entity, local.wandb.project
    )


def _runs_for_target(repo: Path, target: str, lookup: WandbLookup, cache: Cache) -> list[str]:
    """Expand a run id, fingerprint, spec, or experiment into run ids."""
    if "-a" in target and not target.endswith(".yaml") and not target.endswith(".yml"):
        return [target]
    path = Path(target)
    if not path.is_absolute():
        path = repo / target
    if path.is_file():
        kind, _data = load_target(path)
        if kind == "experiment" or path.suffix in {".yaml", ".yml"}:
            # Fingerprints only; attempt listing comes from W&B plus the cache.
            from orch.config import select_cluster

            local = load_local(repo)
            cluster = select_cluster(local, None)
            jobs = build_jobs(
                repo_root=repo,
                target=path,
                entries=load_orch(repo).entries,
                cluster=cluster,
                cache=cache,
                probe=lookup.probe_attempts,
                run_state=lookup.run_state,
                model_ref=None,
            )
            return [job.run_id for job in jobs]
    if len(target) == 12 and all(ch in "0123456789abcdef" for ch in target):
        attempts = lookup.probe_attempts(target)
        cached = [entry.attempt for entry in cache.for_fingerprint(target)]
        numbers = sorted(set(attempts) | set(cached))
        return [f"{target}-a{number}" for number in numbers] or [f"{target}-a1"]
    return [target]


def status_rows(repo: Path, targets: list[str], *, no_remote: bool) -> list[dict]:
    local = load_local(repo)
    cache, lookup = _cache_and_lookup(repo)
    qstat: dict[str, str] = {}
    if not no_remote and cache.entries:
        from orch.backends.sge import SGEBackend
        from orch.ssh import SSH

        cluster = select_cluster(local, None)
        try:
            with SSH(cluster.ssh, cluster.password_env) as conn:
                qstat = SGEBackend().status_map(conn)
        except Exception as exc:
            qstat = {}
            print(f"warning: qstat failed ({exc})", file=sys.stderr, flush=True)
    if targets:
        run_ids = []
        for target in targets:
            run_ids.extend(_runs_for_target(repo, target, lookup, cache))
    else:
        run_ids = list(cache.entries)
    rows = []
    for run_id in run_ids:
        cached = cache.by_run_id(run_id)
        state = lookup.run_state(run_id)
        sched = None
        if cached is not None:
            sched = "unknown" if no_remote else qstat.get(cached.scheduler_id, "gone")
        rows.append(
            {
                "run_id": run_id,
                "run_name": None if cached is None else cached.name,
                "fingerprint": None if cached is None else cached.fingerprint,
                "attempt": None if cached is None else cached.attempt,
                "kind": None if cached is None else cached.kind,
                "decision": None,
                "reason": None,
                "scheduler_id": None if cached is None else cached.scheduler_id,
                "wandb_state": state,
                "scheduler_state": sched,
                "url": f"https://wandb.ai/{local.wandb.entity}/{local.wandb.project}/runs/{run_id}",
                "dependency": None,
            }
        )
    return rows


def show_logs(
    repo: Path,
    run_id: str,
    *,
    which: str = "stdout",
    follow: bool = False,
) -> None:
    if which not in {"stdout", "stderr", "sge.out", "sge.err"}:
        raise UsageError("--which must be stdout, stderr, sge.out, or sge.err")
    local = load_local(repo)
    cache = Cache.load(local.wandb.entity, local.wandb.project)
    lookup = WandbLookup(local.wandb.entity, local.wandb.project)
    cached = cache.by_run_id(run_id)
    state = lookup.run_state(run_id)
    remote = state in {None, "running"} and cached is not None
    if remote:
        cluster = select_cluster(local, cached.cluster)
        filename = {
            "stdout": "stdout.log",
            "stderr": "stderr.log",
            "sge.out": "sge.out",
            "sge.err": "sge.err",
        }[which]
        path = f"{cluster.work_dir}/{run_id}/{filename}"
        from orch.ssh import SSH

        follow_flag = "-f " if follow else ""
        with SSH(cluster.ssh, cluster.password_env) as conn:
            conn.run(f"tail -n +1 {follow_flag}{shlex.quote(path)}", stream=True, check=False)
        return
    run = lookup.run(run_id)
    if run is None:
        raise UsageError(f"no W&B run {run_id}")
    name = {
        "stdout": "stdout.log",
        "stderr": "stderr.log",
        "sge.out": "sge.out",
        "sge.err": "sge.err",
    }[which]
    match = None
    for item in run.files():
        if item.name.endswith(name):
            match = item
            break
    if match is None:
        raise UsageError(f"{run_id} has no file {name}")
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


def cancel_runs(repo: Path, run_ids: list[str]) -> None:
    local = load_local(repo)
    cache = Cache.load(local.wandb.entity, local.wandb.project)
    lookup = WandbLookup(local.wandb.entity, local.wandb.project)
    from orch.backends.sge import SGEBackend
    from orch.ssh import SSH

    by_cluster: dict[str, list[str]] = {}
    for run_id in run_ids:
        cached = cache.by_run_id(run_id)
        if cached is None:
            continue
        by_cluster.setdefault(cached.cluster, []).append(cached.scheduler_id)
    backend = SGEBackend()
    for cluster_name, ids in by_cluster.items():
        cluster = select_cluster(local, cluster_name)
        with SSH(cluster.ssh, cluster.password_env) as conn:
            for scheduler_id in ids:
                backend.cancel(conn, scheduler_id)
    for run_id in run_ids:
        run = lookup.run(run_id)
        if run is None:
            continue
        tags = list(run.tags)
        if "cancelled" not in tags:
            tags.append("cancelled")
        run.tags = tags
        run.update()


def repo() -> Path:
    return find_repo_root()
