"""``orch`` command line. See docs/orchestrator.md §8."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import click
import typer

from orch import __version__
from orch.config import find_repo_root, load_orch
from orch.errors import OrchError, UsageError
from orch.resolve import cli_overrides_from_argv, format_override, resolve_spec
from orch.yamlio import dump_text

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Submit training and evaluation jobs and record them in Weights & Biases.",
)
cluster_app = typer.Typer(help="Check a cluster entry from orch.local.yaml.", no_args_is_help=True)
app.add_typer(cluster_app, name="cluster")


def _fail(exc: OrchError) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=exc.exit_code)


def _artifact_needs_lookup(model: str) -> bool:
    version = model.rsplit(":", 1)[-1]
    return re.fullmatch(r"v\d+", version) is None


def _lookup_for(repo_root: Path, model: str | None):
    """W&B is required for fingerprint: refs and artifact aliases.

    ``run:`` uses it only to fill the upstream run name, and stays offline
    when local config or the API is unavailable. ``hf:`` never touches W&B.
    """
    if model is None:
        return None
    needs_api = model.startswith("fingerprint:") or (
        model.startswith("artifact:") and _artifact_needs_lookup(model)
    )
    if needs_api:
        from orch.wb import WandbLookup

        return WandbLookup.from_repo(repo_root)
    if model.startswith("run:"):
        from orch.wb import WandbLookup

        try:
            return WandbLookup.from_repo(repo_root)
        except OrchError:
            return None
    return None


@app.command()
def inspect(
    spec: Annotated[Path, typer.Argument(help="Run spec path, relative to the repository root.")],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model reference (hf:, run:, fingerprint:, artifact:). Required for eval specs."),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override <path>=<value>. Value is parsed as YAML 1.2."),
    ] = None,
    param: Annotated[
        list[str] | None,
        typer.Option("-p", help="Param override k=v. Sets an identity-bearing or runtime key."),
    ] = None,
    resource: Annotated[
        list[str] | None,
        typer.Option("-r", help="Resource override k=v."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON instead of text.")] = False,
) -> None:
    """Print the resolved spec, fingerprint input, fingerprint, run name, and overrides.

    Each params override is marked ``[identity]`` or ``[runtime]``. Does not
    submit anything and does not touch a cluster. ``fingerprint:`` references
    and ``artifact:`` aliases are the exception: canonicalising them looks the
    finished attempt or concrete version up in W&B.
    """
    del set_, param, resource  # parsed from argv so mixed flag order is kept
    try:
        repo = find_repo_root()
        orch = load_orch(repo)
        resolved = resolve_spec(
            spec if spec.is_absolute() else repo / spec,
            repo_root=repo,
            entries=orch.entries,
            cli_overrides=cli_overrides_from_argv(),
            model_ref=model,
            lookup=_lookup_for(repo, model),
        )
    except OrchError as exc:
        _fail(exc)
    payload = {
        "spec_path": resolved.spec_path,
        "fingerprint": resolved.fingerprint,
        "run_id": f"{resolved.fingerprint}-a?",
        "run_name": resolved.run_name(None),
        "fingerprint_input": resolved.fingerprint_input,
        "resolved_spec": resolved.spec_dict(),
        "overrides": resolved.overrides,
    }
    if json_output:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    typer.echo(f"spec: {resolved.spec_path}")
    typer.echo(f"fingerprint: {resolved.fingerprint}")
    typer.echo(f"run_id: {resolved.fingerprint}-a?")
    typer.echo(f"run_name: {resolved.run_name(None)}")
    typer.echo("")
    typer.echo("fingerprint_input:")
    typer.echo(dump_text(resolved.fingerprint_input).rstrip())
    typer.echo("")
    typer.echo("resolved_spec:")
    typer.echo(dump_text(resolved.spec_dict()).rstrip())
    typer.echo("")
    if resolved.overrides:
        typer.echo("overrides:")
        for item in resolved.overrides:
            typer.echo(f"  {format_override(item)}")
    else:
        typer.echo("overrides: []")


@app.command()
def run(
    target: Annotated[Path, typer.Argument(help="Run spec or experiment spec, relative to the repository root.")],
    cluster: Annotated[str | None, typer.Option("--cluster", help="Cluster name. Defaults to default_cluster.")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model reference. Required for an eval spec, forbidden for a train spec.")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override <path>=<value>, parsed as YAML 1.2.")] = None,
    param: Annotated[list[str] | None, typer.Option("-p", help="Param override k=v. Sets an identity-bearing or runtime key.")] = None,
    resource: Annotated[list[str] | None, typer.Option("-r", help="Resource override k=v.")] = None,
    evals: Annotated[str | None, typer.Option("--evals", help="Comma-separated eval specs replacing the train spec's list.")] = None,
    no_evals: Annotated[bool, typer.Option("--no-evals", help="Do not submit the train spec's evals.")] = False,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Tag applied to every job. Repeatable.")] = None,
    group: Annotated[str | None, typer.Option("--group", help="W&B group. Unset unless passed.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Submit a new attempt even when the latest one finished.")] = False,
    resync: Annotated[bool, typer.Option("--resync", help="Allow uv sync while jobs are active on this cluster.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Resolve and reconcile, then print decisions without submitting.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Resolve, reconcile, stage code, and submit training and evaluation jobs."""
    del set_, param, resource
    from orch.driver import execute_run, format_report

    eval_paths = [item.strip() for item in evals.split(",") if item.strip()] if evals else None
    try:
        rows = execute_run(
            target,
            repo_root=find_repo_root(),
            cluster_name=cluster,
            model_ref=model,
            evals=eval_paths,
            no_evals=no_evals,
            tags=tag,
            group=group,
            force=force,
            resync=resync,
            dry_run=dry_run,
        )
    except OrchError as exc:
        _fail(exc)
    typer.echo(format_report(rows, as_json=json_output))


@app.command(name="eval")
def eval_cmd(
    target: Annotated[Path, typer.Argument(help="Eval spec path.")],
    model: Annotated[str, typer.Option("--model", help="Model reference (hf:, run:, fingerprint:, artifact:).")],
    cluster: Annotated[str | None, typer.Option("--cluster", help="Cluster name.")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override <path>=<value>.")] = None,
    param: Annotated[list[str] | None, typer.Option("-p", help="Param override k=v. Sets an identity-bearing or runtime key.")] = None,
    resource: Annotated[list[str] | None, typer.Option("-r", help="Resource override k=v.")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Tag applied to the job. Repeatable.")] = None,
    group: Annotated[str | None, typer.Option("--group", help="W&B group.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Submit a new attempt even when the latest one finished.")] = False,
    resync: Annotated[bool, typer.Option("--resync", help="Allow uv sync while jobs are active.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the decision without submitting.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Submit one evaluation. Equivalent to ``orch run <eval-spec> --model REF``."""
    del set_, param, resource
    from orch.driver import execute_run, format_report

    try:
        rows = execute_run(
            target,
            repo_root=find_repo_root(),
            cluster_name=cluster,
            model_ref=model,
            tags=tag,
            group=group,
            force=force,
            resync=resync,
            dry_run=dry_run,
        )
    except OrchError as exc:
        _fail(exc)
    typer.echo(format_report(rows, as_json=json_output))


@app.command()
def status(
    targets: Annotated[list[str] | None, typer.Argument(help="Spec, experiment, fingerprint, or run id. Omit for the local cache.")] = None,
    no_remote: Annotated[bool, typer.Option("--no-remote", help="Do not query qstat.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show W&B and scheduler state for cached or requested runs."""
    from orch.driver import format_report
    from orch.ops import repo, status_rows

    try:
        rows = status_rows(repo(), list(targets or []), no_remote=no_remote)
    except OrchError as exc:
        _fail(exc)
    typer.echo(format_report(rows, as_json=json_output))


@app.command()
def logs(
    run_id: Annotated[str, typer.Argument(help="W&B run id, fingerprint-aN.")],
    follow: Annotated[bool, typer.Option("--follow", help="Follow a remote log. Only for a job that has not finished.")] = False,
    which: Annotated[str, typer.Option("--which", help="stdout, stderr, sge.out, or sge.err.")] = "stdout",
) -> None:
    """Print a finished run's file from W&B, or tail it on the cluster while it is in flight."""
    from orch.ops import repo, show_logs

    try:
        show_logs(repo(), run_id, which=which, follow=follow)
    except OrchError as exc:
        _fail(exc)


@app.command()
def tag(
    targets: Annotated[list[str], typer.Argument(help="Run id, fingerprint, spec, or experiment.")],
    add: Annotated[list[str] | None, typer.Option("--add", help="Tag to add. Repeatable.")] = None,
    remove: Annotated[list[str] | None, typer.Option("--remove", help="Tag to remove. Repeatable.")] = None,
) -> None:
    """Add or remove W&B tags."""
    from orch.ops import edit_tags, repo

    try:
        changed = edit_tags(repo(), targets, add=list(add or []), remove=list(remove or []))
    except OrchError as exc:
        _fail(exc)
    for run_id in changed:
        typer.echo(run_id)


@app.command()
def cancel(
    run_ids: Annotated[list[str], typer.Argument(help="Run ids to qdel. A W&B run is tagged cancelled when it exists.")],
) -> None:
    """Delete scheduler jobs and tag the W&B runs cancelled."""
    from orch.ops import cancel_runs, repo

    try:
        cancel_runs(repo(), run_ids)
    except OrchError as exc:
        _fail(exc)
    for run_id in run_ids:
        typer.echo(run_id)


@cluster_app.command("check")
def cluster_check(
    name: Annotated[str, typer.Argument(help="Cluster name in orch.local.yaml.")],
    gpu: Annotated[bool, typer.Option("--gpu", help="Submit a short GPU job that imports torch and vllm and round-trips W&B.")] = False,
) -> None:
    """Test cluster invariants A1–A5 and A7. ``--gpu`` also tests A6."""
    from orch.check import check_cluster
    from orch.config import load_local, select_cluster

    try:
        root = find_repo_root()
        local = load_local(root)
        notes = check_cluster(select_cluster(local, name), local, root, gpu=gpu)
    except OrchError as exc:
        _fail(exc)
    for note in notes:
        typer.echo(note)


@app.command(name="exec")
def exec_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Attempt directory containing job.json. Internal.")],
) -> None:
    """Run the training or evaluation wrapper on a compute node. Not for interactive use."""
    from orch.exec import execute

    try:
        code = execute(run_dir)
    except OrchError as exc:
        _fail(exc)
    raise typer.Exit(code=code)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", help="Print orch version and exit."),
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=1)


def main() -> None:
    try:
        # standalone_mode=False makes click return an Exit code instead of
        # terminating the process, so validation errors keep exit code 1.
        code = app(standalone_mode=False)
    except OrchError as exc:
        _fail(exc)
    except click.exceptions.Exit as exc:
        code = 0 if exc.exit_code is None else exc.exit_code
    except click.exceptions.UsageError as exc:
        exc.show()
        code = 1
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        code = 1
    if code:
        raise SystemExit(code)
