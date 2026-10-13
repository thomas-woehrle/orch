"""Load and validate ``orch.yaml`` and ``orch.local.yaml``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from orch.errors import UsageError
from orch.yamlio import load_path


@dataclass
class Entry:
    name: str
    kind: str
    cmd: str
    parser: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class WandbTarget:
    entity: str
    project: str


@dataclass
class Cluster:
    name: str
    backend: str
    repo_dir: str
    work_dir: str
    ssh: str | None = None
    header: str | None = None
    setup: str = ""
    env: dict[str, str] = field(default_factory=dict)
    password_env: str | None = None
    devices: list[int] | None = None


@dataclass
class OrchConfig:
    version: int
    entries: dict[str, Entry]


@dataclass
class LocalConfig:
    version: int
    wandb: WandbTarget
    default_cluster: str
    clusters: dict[str, Cluster]


def find_repo_root(start: Path | None = None) -> Path:
    cwd = (start or Path.cwd()).resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "orch.yaml").is_file():
            return candidate
    raise UsageError("invoke orch from the repository root (orch.yaml not found)")


def _require_mapping(value: object, what: str) -> dict:
    if not isinstance(value, dict):
        raise UsageError(f"{what} must be a mapping")
    return value


def _string_map(value: object, what: str) -> dict[str, str]:
    raw = _require_mapping(value, what)
    out: dict[str, str] = {}
    for key, item in raw.items():
        if not isinstance(key, str):
            raise UsageError(f"{what} keys must be strings")
        if item is None:
            raise UsageError(f"{what}.{key} must not be null")
        out[key] = str(item)
    return out


def load_orch(repo_root: Path) -> OrchConfig:
    path = repo_root / "orch.yaml"
    data = _require_mapping(load_path(path), "orch.yaml")
    version = data.get("version")
    if version != 1:
        raise UsageError(f"orch.yaml version must be 1, got {version!r}")
    entries_raw = _require_mapping(data.get("entries"), "orch.yaml entries")
    entries: dict[str, Entry] = {}
    for name, body in entries_raw.items():
        body = _require_mapping(body, f"entry {name}")
        kind = body.get("kind")
        if kind not in ("train", "eval"):
            raise UsageError(f"entry {name}: kind must be 'train' or 'eval'")
        cmd = body.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise UsageError(f"entry {name}: cmd must be a string")
        parser = body.get("parser")
        if kind == "eval" and not isinstance(parser, str):
            raise UsageError(f"entry {name}: eval entries require a parser")
        if parser is not None and not isinstance(parser, str):
            raise UsageError(f"entry {name}: parser must be a string")
        env = _string_map(body.get("env", {}), f"entry {name} env")
        entries[str(name)] = Entry(
            name=str(name), kind=kind, cmd=cmd, parser=parser, env=env
        )
    return OrchConfig(version=1, entries=entries)


def load_local(repo_root: Path) -> LocalConfig:
    path = repo_root / "orch.local.yaml"
    if not path.is_file():
        raise UsageError(
            "orch.local.yaml not found (see orch.local.example.yaml); "
            "it holds the W&B target and cluster entries"
        )
    data = _require_mapping(load_path(path), "orch.local.yaml")
    version = data.get("version")
    if version != 1:
        raise UsageError(f"orch.local.yaml version must be 1, got {version!r}")
    wandb_raw = _require_mapping(data.get("wandb"), "wandb")
    entity = os.environ.get("WANDB_ENTITY") or wandb_raw.get("entity")
    project = os.environ.get("WANDB_PROJECT") or wandb_raw.get("project")
    if not isinstance(entity, str) or not entity:
        raise UsageError("wandb.entity must be set (or WANDB_ENTITY)")
    if not isinstance(project, str) or not project:
        raise UsageError("wandb.project must be set (or WANDB_PROJECT)")
    default_cluster = data.get("default_cluster")
    if not isinstance(default_cluster, str) or not default_cluster:
        raise UsageError("default_cluster must be a cluster name")
    clusters_raw = _require_mapping(data.get("clusters"), "clusters")
    clusters: dict[str, Cluster] = {}
    for name, body in clusters_raw.items():
        body = _require_mapping(body, f"cluster {name}")
        clusters[str(name)] = _load_cluster(str(name), body)
    if default_cluster not in clusters:
        raise UsageError(f"default_cluster {default_cluster!r} is not defined")
    return LocalConfig(
        version=1,
        wandb=WandbTarget(entity=entity, project=project),
        default_cluster=default_cluster,
        clusters=clusters,
    )


_BACKENDS = ("sge", "slurm", "local")


def _require_text(body: dict, name: str, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UsageError(f"cluster {name}: {key} must be a string")
    return value


def _load_cluster(name: str, body: dict) -> Cluster:
    backend = body.get("backend")
    if backend not in _BACKENDS:
        raise UsageError(
            f"cluster {name}: backend must be one of {', '.join(_BACKENDS)}"
        )
    repo_dir = _require_text(body, name, "repo_dir")
    work_dir = _require_text(body, name, "work_dir")
    setup = body.get("setup", "")
    if not isinstance(setup, str):
        raise UsageError(f"cluster {name}: setup must be a string")
    env = _string_map(body.get("env", {}), f"cluster {name} env")
    if backend == "local":
        return _load_local_cluster(name, body, repo_dir, work_dir, setup, env)
    return _load_remote_cluster(name, body, backend, repo_dir, work_dir, setup, env)


def _load_local_cluster(
    name: str,
    body: dict,
    repo_dir: str,
    work_dir: str,
    setup: str,
    env: dict[str, str],
) -> Cluster:
    if body.get("ssh"):
        raise UsageError(f"cluster {name}: a local cluster does not take ssh")
    if body.get("password_env") is not None:
        raise UsageError(f"cluster {name}: a local cluster does not take password_env")
    if body.get("header") is not None:
        raise UsageError(f"cluster {name}: a local cluster does not take a header")
    devices = body.get("devices")
    if (
        not isinstance(devices, list)
        or not devices
        or not all(type(item) is int for item in devices)
    ):
        raise UsageError(f"cluster {name}: devices must be a non-empty list of integers")
    return Cluster(
        name=name,
        backend="local",
        repo_dir=repo_dir,
        work_dir=work_dir,
        setup=setup,
        env=env,
        devices=list(devices),
    )


def _load_remote_cluster(
    name: str,
    body: dict,
    backend: str,
    repo_dir: str,
    work_dir: str,
    setup: str,
    env: dict[str, str],
) -> Cluster:
    if "devices" in body:
        raise UsageError(f"cluster {name}: devices is only valid for a local cluster")
    password_env = body.get("password_env")
    if password_env is not None and not isinstance(password_env, str):
        raise UsageError(f"cluster {name}: password_env must be a string")
    return Cluster(
        name=name,
        backend=backend,
        ssh=_require_text(body, name, "ssh"),
        repo_dir=repo_dir,
        work_dir=work_dir,
        header=_require_text(body, name, "header"),
        setup=setup,
        env=env,
        password_env=password_env,
    )


def select_cluster(local: LocalConfig, name: str | None) -> Cluster:
    chosen = name or local.default_cluster
    if chosen not in local.clusters:
        raise UsageError(f"unknown cluster {chosen!r}")
    return local.clusters[chosen]
