"""Resolve, reconcile, stage, and submit. See docs/orchestrator.md §6.1."""

from __future__ import annotations

import json
from pathlib import Path

from orch.backends import backend_for
from orch.cache import Cache, CacheEntry
from orch.conn import open_conn
from orch.config import Cluster, Entry, LocalConfig, load_local, load_orch, select_cluster
from orch.errors import RemoteError, UsageError
from orch.gitpreflight import git_state
from orch.plan import Job, build_jobs, refresh_identity, retarget_eval
from orch.render import (
    build_env,
    fingerprint_json,
    hostname,
    job_document,
    render_command,
    utc_now,
)
from orch.resolve import cli_overrides_from_argv, format_override
from orch.staging import stage
from orch.wb import WandbLookup
from orch.yamlio import dump_text


def report_rows(jobs: list[Job], *, entity: str, project: str) -> list[dict]:
    return [job.row(entity=entity, project=project) for job in jobs]


def format_report(rows: list[dict], *, as_json: bool) -> str:
    if as_json:
        return json.dumps(rows, indent=2, ensure_ascii=False)
    if not rows:
        return "(no jobs)"
    headers = [
        "run_id",
        "run_name",
        "kind",
        "cluster",
        "decision",
        "reason",
        "fingerprint",
        "scheduler_id",
        "scheduler_state",
        "wandb_state",
        "dependency",
    ]
    widths = {key: len(key) for key in headers}
    rendered = []
    for row in rows:
        cells = {key: "" if row.get(key) is None else str(row.get(key)) for key in headers}
        rendered.append(cells)
        for key, value in cells.items():
            widths[key] = max(widths[key], len(value))
    lines = ["  ".join(key.ljust(widths[key]) for key in headers)]
    for cells in rendered:
        lines.append("  ".join(cells[key].ljust(widths[key]) for key in headers))
    listing = [
        f"  {row['run_id']}  {format_override(item)}"
        for row in rows
        for item in row.get("overrides") or []
    ]
    if listing:
        lines.append("")
        lines.append("overrides:")
        lines.extend(listing)
    return "\n".join(lines)


def _rerender(job: Job, cluster: Cluster) -> None:
    run_dir = f"{cluster.work_dir}/{job.run_id}"
    job.command = render_command(job.entry_cmd, run_dir=run_dir, params=job.params)


def _apply_ledger(conn, job: Job, cluster: Cluster) -> str | None:
    """Increment the attempt while the run directory already exists. Returns the old run id."""
    old = job.run_id
    changed = False
    while conn.exists(f"{cluster.work_dir}/{job.fingerprint}-a{job.attempt}"):
        job.attempt += 1
        changed = True
    if not changed:
        return None
    refresh_identity(job)
    _rerender(job, cluster)
    return old


def _holds(
    job: Job, jobs: list[Job], cache: Cache, qstat: dict[str, str], *, cluster_name: str
) -> list[str]:
    """Holds are per cluster: a scheduler id from another cluster cannot be named."""
    if not job.upstream_run_id:
        return []
    for other in jobs:
        if other.run_id == job.upstream_run_id and other.scheduler_id:
            if other.decision == "submit" and other.cluster == cluster_name:
                return [other.scheduler_id]
    cached = cache.by_run_id(job.upstream_run_id)
    if cached is None or cached.cluster != cluster_name:
        return []
    state = qstat.get(cached.scheduler_id, "gone")
    if state in {"queued", "running", "unknown"}:
        return [cached.scheduler_id]
    return []


class LazyStates:
    """Fetch each cluster's scheduler map on first use. A dry run fetches nothing."""

    def __init__(self, query, *, dry_run: bool) -> None:
        self._query = query
        self._dry_run = dry_run
        self._maps: dict[str, dict[str, str] | None] = {}

    def __call__(self, cluster_name: str) -> dict[str, str] | None:
        if cluster_name not in self._maps:
            self._maps[cluster_name] = None if self._dry_run else self._query(cluster_name)
        return self._maps[cluster_name]


def submit_jobs(
    jobs: list[Job],
    *,
    conn,
    backend,
    cluster: Cluster,
    entries: dict[str, Entry],
    cache: Cache,
    qstat: dict[str, str],
    entity: str,
    project: str,
    sha: str,
    remote: str,
    probe,
    orchignore_sha256: str | None = None,
) -> list[Job]:
    """Write per-attempt files and qsub. Trains first, then evals."""
    del entries, probe  # attempts were assigned in build_jobs; ledger adjusts them
    src_dir = f"{cluster.repo_dir}/.worktrees/{sha}"
    venv_python = f"{cluster.repo_dir}/.venv/bin/python"
    submitted_from = hostname()
    submitted_at = utc_now()

    for job in jobs:
        if job.decision != "submit" or job.kind != "train":
            continue
        old = _apply_ledger(conn, job, cluster)
        if old is None:
            continue
        for child in jobs:
            if child.upstream_run_id == old:
                retarget_eval(child, run_id=job.run_id, handle=job.run_name)
                child.attempt = 1
                child.decision = "submit"
                child.reason = "new"
                refresh_identity(child)
                _rerender(child, cluster)
                _apply_ledger(conn, child, cluster)

    ordered = [job for job in jobs if job.decision == "submit" and job.kind == "train"]
    ordered += [job for job in jobs if job.decision == "submit" and job.kind == "eval"]
    submitted: list[Job] = []
    try:
        for job in ordered:
            if job.kind == "eval":
                _apply_ledger(conn, job, cluster)
            run_dir = f"{cluster.work_dir}/{job.run_id}"
            holds = _holds(job, jobs, cache, qstat, cluster_name=cluster.name)
            job.hold_on = holds
            env = build_env(
                cluster_env=cluster.env,
                entry_env=job.entry_env,
                spec_env=job.spec_env,
                entity=entity,
                project=project,
                job_kind=job.kind,
                run_id=job.run_id,
                run_name=job.run_name,
                tags=job.tags,
                group=job.group,
                run_dir=run_dir,
            )
            script = backend.render_job_script(
                job, cluster, env, src_dir, venv_python, holds
            )
            document = job_document(
                job=job,
                cluster_name=cluster.name,
                backend=cluster.backend,
                repo_dir=cluster.repo_dir,
                work_dir=cluster.work_dir,
                src_dir=src_dir,
                venv_python=venv_python,
                sha=sha,
                remote=remote,
                env=env,
                entity=entity,
                project=project,
                submitted_from=submitted_from,
                submitted_at=submitted_at,
                job_id_env=backend.job_id_env,
                device=job.resources.get("device"),
                orchignore_sha256=orchignore_sha256,
                hold_on=holds,
            )
            conn.put_text(f"{run_dir}/job.sh", script)
            conn.put_text(f"{run_dir}/job.json", json.dumps(document, indent=2) + "\n")
            conn.put_text(f"{run_dir}/params.yaml", dump_text(job.params))
            conn.put_text(
                f"{run_dir}/fingerprint.json",
                fingerprint_json(job.fingerprint_input, job.canonical, job.fingerprint),
            )
            job.scheduler_id = backend.submit(conn, run_dir, holds)
            cache.upsert(
                job.run_id,
                CacheEntry(
                    cluster=cluster.name,
                    scheduler_id=job.scheduler_id,
                    fingerprint=job.fingerprint,
                    attempt=job.attempt,
                    kind=job.kind,
                    name=job.run_name,
                    submitted_at=submitted_at,
                ),
            )
            cache.save()
            submitted.append(job)
    except Exception as exc:
        done = ", ".join(job.run_id for job in submitted) or "nothing"
        raise RemoteError(f"submission failed after submitting {done}: {exc}") from exc
    return submitted


def execute_run(
    target: Path,
    *,
    repo_root: Path,
    cluster_name: str | None = None,
    model_ref: str | None = None,
    evals: list[str] | None = None,
    no_evals: bool = False,
    tags: list[str] | None = None,
    group: str | None = None,
    force: bool = False,
    resync: bool = False,
    dry_run: bool = False,
    argv_overrides: bool = True,
) -> list[dict]:
    local = load_local(repo_root)
    orch = load_orch(repo_root)
    cluster = select_cluster(local, cluster_name)
    overrides = cli_overrides_from_argv() if argv_overrides else []
    lookup = WandbLookup(local.wandb.entity, local.wandb.project)
    cache = Cache.load(local.wandb.entity, local.wandb.project)
    conn = None
    backend = backend_for(cluster)
    try:
        if not dry_run:
            conn = open_conn(cluster)
            try:
                conn.connect()
            except RemoteError:
                raise
            except Exception as exc:
                raise RemoteError(f"could not reach {cluster.name}: {exc}") from exc

        def query(name: str) -> dict[str, str]:
            assert conn is not None
            if name == cluster.name:
                return backend.status_map(conn)
            other = select_cluster(local, name)
            with open_conn(other) as extra:
                return backend_for(other).status_map(extra)

        states = LazyStates(query, dry_run=dry_run)
        jobs = build_jobs(
            repo_root=repo_root,
            target=target if target.is_absolute() else repo_root / target,
            entries=orch.entries,
            cluster=cluster,
            cli_overrides=overrides,
            model_ref=model_ref,
            evals_override=evals,
            no_evals=no_evals,
            tags=tags,
            group=group,
            force=force,
            lookup=lookup,
            cache=cache,
            probe=lookup.probe_attempts,
            run_state=lookup.run_state,
            scheduler_states=states,
        )
        if dry_run:
            return report_rows(jobs, entity=local.wandb.entity, project=local.wandb.project)
        assert conn is not None
        qstat = states(cluster.name) or {}
        git = git_state(repo_root)
        sha, remote = git.sha, git.remote
        if any(job.decision == "submit" for job in jobs):
            stage(
                conn,
                cluster,
                repo_root,
                sha,
                cache=cache,
                qstat=qstat,
                resync=resync,
                orchignore_lines=git.orchignore_lines,
            )
            submit_jobs(
                jobs,
                conn=conn,
                backend=backend,
                cluster=cluster,
                entries=orch.entries,
                cache=cache,
                qstat=qstat,
                entity=local.wandb.entity,
                project=local.wandb.project,
                sha=sha,
                remote=remote,
                probe=lookup.probe_attempts,
                orchignore_sha256=git.orchignore_sha256,
            )
    finally:
        if conn is not None:
            conn.close()
    return report_rows(jobs, entity=local.wandb.entity, project=local.wandb.project)


def require_run_spec(path: Path) -> None:
    from orch.plan import load_target

    kind, _data = load_target(path)
    if kind == "experiment":
        raise UsageError(f"{path} is an experiment spec")


def local_or_die(repo_root: Path) -> LocalConfig:
    return load_local(repo_root)
