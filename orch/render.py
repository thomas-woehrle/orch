"""Render params, fingerprint files, env, and job.sh. See docs/orchestrator.md §6.4."""

from __future__ import annotations

import json
import re
import socket
from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment, StrictUndefined, Undefined, nodes

from orch import __version__
from orch.errors import UsageError
from orch.identity import FINGERPRINT_VERSION, fingerprint_document

# Shared by every backend. The shebang, scheduler directives, and (for local)
# the hold/device preamble are rendered by the backend. See docs/orchestrator.md §6.4.
JOB_BODY = """set -euo pipefail
{{ setup }}
{% for key, value in env.items() %}export {{ key }}={{ value | shquote }}
{% endfor %}
cd {{ src_dir }}
{% if exec_wrapper %}exec {% endif %}{{ venv_python }} -m orch exec {{ run_dir }}
"""

CANONICAL_RESOURCE_KEYS = {"gpus", "gpu_type", "time", "mem", "device"}


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
    """Canonical resource keys, then the header template. ``device`` is local-only."""
    unknown = sorted(set(resources) - CANONICAL_RESOURCE_KEYS)
    if unknown:
        raise UsageError(
            "resource key outside the canonical set "
            "(gpus, gpu_type, time, mem, device): " + ", ".join(unknown)
        )
    if "device" in resources:
        raise UsageError("resource key 'device' is only valid on a local cluster")
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


def render_job_body(
    *,
    setup: str,
    env: dict[str, str],
    run_dir: str,
    src_dir: str,
    venv_python: str,
    exec_wrapper: bool = True,
) -> str:
    """The scheduler-neutral half of ``job.sh``.

    ``exec_wrapper`` is false for the local backend: its preamble installs an
    EXIT trap that must run after the wrapper returns.
    """
    return _jinja().from_string(JOB_BODY).render(
        setup=setup.strip("\n"),
        env=env,
        run_dir=run_dir,
        src_dir=src_dir,
        venv_python=venv_python,
        exec_wrapper=exec_wrapper,
    )


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill_placeholders(value: str, *, run_dir: str, kind: str) -> str:
    """Fill ``{run_dir}``, ``{params}``, ``{output_dir}`` and ``{model_dir}``."""
    allowed = {
        "run_dir": run_dir,
        "params": f"{run_dir}/params.yaml",
        "output_dir": f"{run_dir}/output",
        "model_dir": f"{run_dir}/model",
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in allowed:
            raise UsageError(f"unknown env placeholder {{{name}}}")
        if name == "output_dir" and kind != "eval":
            raise UsageError("{output_dir} is only valid on an eval job")
        if name == "model_dir" and kind != "train":
            raise UsageError("{model_dir} is only valid on a train job")
        return allowed[name]

    return _PLACEHOLDER.sub(replace, value)


def build_env(
    *,
    cluster_env: dict[str, str],
    entry_env: dict[str, str],
    spec_env: dict[str, str] | None = None,
    entity: str,
    project: str,
    job_kind: str,
    run_id: str,
    run_name: str,
    tags: list[str],
    group: str | None,
    run_dir: str,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(cluster_env)
    merged.update(entry_env)
    merged.update(spec_env or {})
    env: dict[str, str] = {
        key: fill_placeholders(str(value), run_dir=run_dir, kind=job_kind)
        for key, value in merged.items()
    }
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
    job_id_env: str = "JOB_ID",
    device: int | None = None,
    orchignore_sha256: str | None = None,
    hold_on: list[str] | None = None,
) -> dict[str, Any]:
    spec = dict(job.spec)
    spec["env"] = dict(getattr(job, "spec_env", None) or spec.get("env") or {})
    return {
        "schema_version": 2,
        "orch_version": __version__,
        "fingerprint": job.fingerprint,
        "fingerprint_version": FINGERPRINT_VERSION,
        "attempt": job.attempt,
        "run_id": job.run_id,
        "run_name": job.run_name,
        "kind": job.kind,
        "entry": job.entry,
        "spec_path": job.spec_path,
        "spec": spec,
        "overrides": list(job.overrides),
        "model_ref": job.model_ref,
        "upstream_run_id": job.upstream_run_id,
        "artifact": getattr(job, "artifact", True),
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
            "job_id_env": job_id_env,
            "device": device,
        },
        "git": {"sha": sha, "remote": remote, "orchignore_sha256": orchignore_sha256},
        "command": job.command,
        "parser": job.parser,
        "env": env,
        "hold_on": list(hold_on or []),
        "submitted_from": submitted_from,
        "submitted_at": submitted_at,
    }
