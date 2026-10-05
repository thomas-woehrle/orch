"""Render params, fingerprint files, env, and job.sh. See docs/orchestrator.md §6.4."""

from __future__ import annotations

import json
import socket
from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment, StrictUndefined, Undefined, nodes

from orch import __version__
from orch.errors import UsageError
from orch.identity import FINGERPRINT_VERSION, fingerprint_document

JOB_SH = """#!/bin/bash
{{ header }}
#$ -N orch_{{ fingerprint }}_a{{ attempt }}
#$ -o {{ run_dir }}/sge.out
#$ -e {{ run_dir }}/sge.err
set -euo pipefail
{{ setup }}
{% for key, value in env.items() %}export {{ key }}={{ value | shquote }}
{% endfor %}
cd {{ src_dir }}
exec {{ venv_python }} -m orch exec {{ run_dir }}
"""


def shquote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def render_flags(params: dict[str, Any]) -> str:
    """Render params as CLI flags. True is a bare flag; false and null are omitted."""
    parts: list[str] = []
    for key, value in params.items():
        if value is True:
            parts.append(f"--{key}")
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            joined = ",".join(str(item) for item in value)
            parts.append(f"--{key} {shquote(joined)}")
        else:
            parts.append(f"--{key} {shquote(str(value))}")
    return " ".join(parts)


def render_command(template: str, *, run_dir: str, params: dict[str, Any]) -> str:
    """Fill entry placeholders. ``{model_path}`` is left for the eval wrapper."""
    flags = render_flags(params)
    command = template
    command = command.replace("{params}", f"{run_dir}/params.yaml")
    command = command.replace("{flags}", flags)
    command = command.replace("{run_dir}", run_dir)
    command = command.replace("{model_dir}", f"{run_dir}/model")
    command = command.replace("{output_dir}", f"{run_dir}/output")
    return " ".join(part for part in command.split(" ") if part != "")


def _jinja(*, strict: bool = True) -> Environment:
    # Header templates use `{% if gpu_type %}` for optional resources. Strict
    # undefined raises on that test; the default undefined is falsy instead.
    # Unguarded variables are rejected by validate_resources before rendering.
    kind = StrictUndefined if strict else Undefined
    env = Environment(undefined=kind, keep_trailing_newline=True, autoescape=False)
    env.filters["shquote"] = shquote
    return env


def template_names(template: str) -> tuple[set[str], set[str]]:
    """Return ``(unguarded names, every name)`` used by a header template."""
    parsed = Environment().parse(template)
    unguarded: set[str] = set()
    every: set[str] = set()

    def walk(node: nodes.Node, guards: set[str]) -> None:
        if isinstance(node, nodes.If):
            # find_all does not include the node itself, and a bare `{% if x %}`
            # test is a Name.
            tested = {name.name for name in node.test.find_all(nodes.Name)}
            if isinstance(node.test, nodes.Name):
                tested.add(node.test.name)
            every.update(tested)
            for child in node.body:
                walk(child, guards | tested)
            for child in node.else_:
                walk(child, guards)
            return
        if isinstance(node, nodes.Name) and node.ctx == "load":
            every.add(node.name)
            if node.name not in guards:
                unguarded.add(node.name)
        for child in node.iter_child_nodes():
            walk(child, guards)

    walk(parsed, set())
    return unguarded, every


def validate_resources(resources: dict[str, Any], header: str) -> None:
    unguarded, every = template_names(header)
    missing = sorted(unguarded - set(resources))
    if missing:
        raise UsageError(
            "header template variable without a value and without a guard: "
            + ", ".join(missing)
        )
    unused = sorted(set(resources) - every)
    if unused:
        raise UsageError(
            "resource key not referenced by the cluster header template: "
            + ", ".join(unused)
        )


def render_header(header: str, resources: dict[str, Any]) -> str:
    validate_resources(resources, header)
    rendered = _jinja(strict=False).from_string(header).render(**resources)
    return rendered.strip("\n")


def render_job_script(
    *,
    header: str,
    setup: str,
    env: dict[str, str],
    fingerprint: str,
    attempt: int,
    run_dir: str,
    src_dir: str,
    venv_python: str,
) -> str:
    return _jinja().from_string(JOB_SH).render(
        header=header,
        setup=setup.strip("\n"),
        env=env,
        fingerprint=fingerprint,
        attempt=attempt,
        run_dir=run_dir,
        src_dir=src_dir,
        venv_python=venv_python,
    )


def build_env(
    *,
    cluster_env: dict[str, str],
    entry_env: dict[str, str],
    entity: str,
    project: str,
    job_kind: str,
    run_id: str,
    run_name: str,
    tags: list[str],
    group: str | None,
    run_dir: str,
) -> dict[str, str]:
    env: dict[str, str] = {}
    env.update(cluster_env)
    env.update(entry_env)
    env["WANDB_RUN_ID"] = run_id
    env["WANDB_PROJECT"] = project
    env["WANDB_ENTITY"] = entity
    env["WANDB_NAME"] = run_name
    env["WANDB_JOB_TYPE"] = job_kind
    env["WANDB_TAGS"] = ",".join(tags)
    if group:
        env["WANDB_RUN_GROUP"] = group
    env["WANDB_RESUME"] = "allow"
    env["WANDB_DIR"] = f"{run_dir}/wandb"
    env["ORCH_RUN_DIR"] = run_dir
    env["ORCH_PARAMS"] = f"{run_dir}/params.yaml"
    if job_kind == "train":
        env["ORCH_MODEL_DIR"] = f"{run_dir}/model"
    else:
        env["ORCH_OUTPUT_DIR"] = f"{run_dir}/output"
    return env


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def hostname() -> str:
    return socket.gethostname()


def fingerprint_json(payload: dict[str, Any], canonical: str, fingerprint: str) -> str:
    return json.dumps(
        fingerprint_document(payload, canonical, fingerprint), indent=2, ensure_ascii=False
    ) + "\n"


def job_document(
    *,
    job: Any,
    cluster_name: str,
    backend: str,
    repo_dir: str,
    work_dir: str,
    src_dir: str,
    venv_python: str,
    sha: str,
    remote: str,
    env: dict[str, str],
    entity: str,
    project: str,
    submitted_from: str,
    submitted_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "orch_version": __version__,
        "fingerprint": job.fingerprint,
        "fingerprint_version": FINGERPRINT_VERSION,
        "attempt": job.attempt,
        "run_id": job.run_id,
        "run_name": job.run_name,
        "kind": job.kind,
        "entry": job.entry,
        "spec_path": job.spec_path,
        "spec": job.spec,
        "overrides": list(job.overrides),
        "model_ref": job.model_ref,
        "upstream_run_id": job.upstream_run_id,
        "wandb": {
            "entity": entity,
            "project": project,
            "job_type": job.kind,
            "tags": list(job.tags),
            "group": job.group,
        },
        "cluster": {
            "name": cluster_name,
            "backend": backend,
            "repo_dir": repo_dir,
            "work_dir": work_dir,
            "src_dir": src_dir,
            "venv_python": venv_python,
        },
        "git": {"sha": sha, "remote": remote},
        "command": job.command,
        "parser": job.parser,
        "env": env,
        "submitted_from": submitted_from,
        "submitted_at": submitted_at,
    }
