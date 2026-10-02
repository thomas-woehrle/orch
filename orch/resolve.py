"""Spec and params resolution. See docs/orchestrator.md §5.1–§5.3 and §5.7."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orch.config import Entry
from orch.errors import UsageError
from orch.identity import canonical_json, fingerprint_hex, fingerprint_input
from orch.modelref import ModelLookup, ModelResolution, resolve_model_ref
from orch.naming import format_run_name
from orch.yamlio import load_path, parse_value

SPEC_KEYS = {
    "kind",
    "entry",
    "extends",
    "params",
    "set",
    "runtime_keys",
    "resources",
    "evals",
    "name_keys",
    "description",
}
OVERRIDE_ROOTS = {"params", "resources", "evals", "runtime_keys", "name_keys"}
FORBIDDEN_ROOTS = {"kind", "entry", "extends"}


@dataclass
class Override:
    path: str
    value: Any
    raw: str


@dataclass
class ResolvedSpec:
    spec_path: str
    stem: str
    kind: str
    entry: str
    params: dict[str, Any]
    runtime_keys: list[str]
    resources: dict[str, Any]
    evals: list[str]
    name_keys: dict[str, str]
    description: str
    overrides: list[dict[str, Any]]
    name_segments: list[tuple[str, Any]]
    override_segments: list[tuple[str, Any]]
    model: ModelResolution | None
    fingerprint_input: dict[str, Any]
    canonical: str
    fingerprint: str

    def run_name(self, attempt: int | None = None) -> str:
        handle = self.model.handle if self.model is not None else None
        return format_run_name(
            self.stem,
            self.name_segments,
            self.override_segments,
            attempt=attempt,
            model_handle=handle,
        )

    def spec_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "entry": self.entry,
            "params": self.params,
            "runtime_keys": list(self.runtime_keys),
            "resources": self.resources,
            "evals": list(self.evals),
            "name_keys": dict(self.name_keys),
            "description": self.description,
        }


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recurse into mappings; lists, scalars, and null replace. Base key order is kept."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def repo_relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _split_assignment(token: str) -> tuple[str, str]:
    if "=" not in token:
        raise UsageError(f"override {token!r} must be path=value")
    path, raw_value = token.split("=", 1)
    if not path:
        raise UsageError(f"override {token!r} is missing a path")
    return path, raw_value


def parse_assignment(token: str, *, prefix: str = "") -> Override:
    path, raw_value = _split_assignment(token)
    if prefix:
        path = f"{prefix}{path}"
    try:
        value = parse_value(raw_value)
    except Exception as exc:
        raise UsageError(f"cannot parse override value {raw_value!r}: {exc}") from exc
    shown = token if not prefix else f"{path}={raw_value}"
    return Override(path=path, value=value, raw=shown)


def cli_overrides_from_argv(argv: list[str] | None = None) -> list[Override]:
    """Pull ``--set`` / ``-p`` / ``-r`` out of argv, preserving order.

    ``-p`` sets a param, identity-bearing or runtime. ``-r`` sets a resource.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    prefixes = {"--set": "", "-p": "params.", "-r": "resources."}
    overrides: list[Override] = []
    index = 0
    while index < len(args):
        token = args[index]
        matched = None
        value_inline = None
        for flag in prefixes:
            if token == flag:
                matched = flag
                break
            if token.startswith(flag + "="):
                matched = flag
                value_inline = token[len(flag) + 1 :]
                break
        if matched is None:
            index += 1
            continue
        if value_inline is None:
            if index + 1 >= len(args):
                raise UsageError(f"{matched} requires a value")
            value_inline = args[index + 1]
            index += 2
        else:
            index += 1
        overrides.append(parse_assignment(value_inline, prefix=prefixes[matched]))
    return overrides


def _check_keys(data: dict[str, Any], where: str) -> None:
    unknown = sorted(set(data) - SPEC_KEYS)
    if unknown:
        raise UsageError(f"unknown spec key in {where}: {', '.join(unknown)}")


def _load_spec_merged(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(item.name for item in (*stack, path))
        raise UsageError(f"extends cycle: {cycle}")
    if not path.is_file():
        raise UsageError(f"extends target does not exist: {path}")
    loaded = load_path(path)
    if not isinstance(loaded, dict):
        raise UsageError(f"{path} is not a mapping")
    if "runs" in loaded and "kind" not in loaded:
        raise UsageError(f"{path.name} is an experiment spec, not a run spec")
    _check_keys(loaded, path.name)
    parent_rel = loaded.get("extends")
    if parent_rel is None:
        return loaded
    if not isinstance(parent_rel, str) or not parent_rel:
        raise UsageError(f"{path.name}: extends must be a path")
    parent = _load_spec_merged(path.parent / parent_rel, (*stack, path))
    child = {key: value for key, value in loaded.items() if key != "extends"}
    return deep_merge(parent, child)


def _load_params_file(path: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], list[str]]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(item.name for item in (*stack, path))
        raise UsageError(f"params extends cycle: {cycle}")
    if not path.is_file():
        raise UsageError(f"params file does not exist: {path}")
    loaded = load_path(path)
    if not isinstance(loaded, dict):
        raise UsageError(f"params file {path} is not a mapping")
    runtime = _string_list(loaded.pop("runtime_keys", []), f"{path.name} runtime_keys")
    parent_rel = loaded.pop("extends", None)
    if parent_rel is None:
        return loaded, runtime
    if not isinstance(parent_rel, str) or not parent_rel:
        raise UsageError(f"{path.name}: extends must be a path")
    parent_params, parent_runtime = _load_params_file(
        path.parent / parent_rel, (*stack, path)
    )
    merged = deep_merge(parent_params, loaded)
    union: list[str] = []
    for key in [*parent_runtime, *runtime]:
        if key not in union:
            union.append(key)
    return merged, union


def _string_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise UsageError(f"{what} must be a list of non-empty strings")
    for item in value:
        if "." in item:
            raise UsageError(f"{what} entry {item!r} must be a top-level key")
    return list(value)


def _set_overrides(set_map: Any, where: str) -> list[Override]:
    if set_map is None:
        return []
    if not isinstance(set_map, dict):
        raise UsageError(f"{where} set must be a mapping of path to value")
    overrides: list[Override] = []
    for path, value in set_map.items():
        if not isinstance(path, str) or not path:
            raise UsageError(f"{where} set keys must be dotted paths")
        overrides.append(Override(path=path, value=copy.deepcopy(value), raw=_raw(path, value)))
    return overrides


def _raw(path: str, value: Any) -> str:
    return f"{path}={_format_raw_value(value)}"


def _format_raw_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_raw_value(item) for item in value) + "]"
    if isinstance(value, dict):
        body = ", ".join(f"{key}: {_format_raw_value(item)}" for key, item in value.items())
        return "{" + body + "}"
    return str(value)


def annotated_overrides(items: list[Override], runtime_keys: list[str]) -> list[dict[str, Any]]:
    """One record per override. Params records carry ``identity`` or ``runtime``."""
    classified = set(runtime_keys)
    records: list[dict[str, Any]] = []
    for item in items:
        record: dict[str, Any] = {"path": item.path, "value": copy.deepcopy(item.value)}
        top = _top_param(item.path)
        if top is not None:
            record["classification"] = "runtime" if top in classified else "identity"
        records.append(record)
    return records


def format_override(record: dict[str, Any]) -> str:
    text = _raw(record["path"], record["value"])
    classification = record.get("classification")
    if classification:
        return f"{text} [{classification}]"
    return text


def _apply_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = root
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            cursor[part] = {}
            current = cursor[part]
        cursor = current
    cursor[parts[-1]] = copy.deepcopy(value)


def _lookup(root: dict[str, Any], path: str) -> Any:
    cursor: Any = root
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def _check_override_path(path: str) -> str:
    root = path.split(".", 1)[0]
    if root in FORBIDDEN_ROOTS:
        raise UsageError(f"override path {path!r} targets {root}; not allowed")
    if root not in OVERRIDE_ROOTS:
        raise UsageError(
            f"override path {path!r} must target one of {', '.join(sorted(OVERRIDE_ROOTS))}"
        )
    return root


def _top_param(path: str) -> str | None:
    parts = path.split(".")
    if parts[0] != "params" or len(parts) < 2 or not parts[1]:
        return None
    return parts[1]


def resolve_spec(
    spec_path: Path,
    *,
    repo_root: Path,
    entries: dict[str, Entry],
    cli_overrides: list[Override] | None = None,
    extra_set: dict[str, Any] | None = None,
    model_ref: str | None = None,
    lookup: ModelLookup | None = None,
    require_model: bool = True,
) -> ResolvedSpec:
    spec_path = spec_path.resolve()
    merged = _load_spec_merged(spec_path, ())
    file_set = _set_overrides(merged.pop("set", None), spec_path.name)
    merged.pop("extends", None)

    params_declared: list[str] = []
    params_value = merged.get("params")
    if isinstance(params_value, str):
        params, params_declared = _load_params_file(repo_root / params_value, ())
        merged["params"] = params
    elif isinstance(params_value, dict):
        inline = copy.deepcopy(params_value)
        if "extends" in inline:
            raise UsageError("inline params cannot use extends; point params at a file")
        params_declared = _string_list(
            inline.pop("runtime_keys", []), "inline params runtime_keys"
        )
        merged["params"] = inline
    elif params_value is not None:
        raise UsageError("params must be a mapping or a path")

    spec_level = _string_list(merged.get("runtime_keys", []), "spec runtime_keys")
    overrides = [
        *file_set,
        *_set_overrides(extra_set, "experiment"),
        *(cli_overrides or []),
    ]
    runtime_overrides = [
        item for item in overrides if item.path == "runtime_keys" or item.path.startswith("runtime_keys.")
    ]
    rest = [item for item in overrides if item not in runtime_overrides]
    for item in runtime_overrides:
        _check_override_path(item.path)
        if item.path != "runtime_keys":
            raise UsageError(
                "runtime_keys overrides must replace the whole list "
                "(--set runtime_keys=[...])"
            )
        spec_level = _string_list(item.value, "runtime_keys override")
        merged["runtime_keys"] = list(spec_level)

    def union() -> list[str]:
        return sorted(set(params_declared) | set(spec_level))

    runtime_keys = union()
    applied: list[Override] = []
    for item in rest:
        _check_override_path(item.path)
        _apply_path(merged, item.path, item.value)
        applied.append(item)

    runtime_keys = union()
    merged["runtime_keys"] = runtime_keys
    _validate_resolved(
        merged,
        spec_path=spec_path,
        repo_root=repo_root,
        entries=entries,
        model_ref=model_ref,
        require_model=require_model,
    )

    model: ModelResolution | None = None
    if merged["kind"] == "eval" and model_ref:
        model = resolve_model_ref(model_ref, lookup)

    stem = spec_path.stem
    name_keys = dict(merged.get("name_keys") or {})
    view = {
        "params": merged["params"],
        "resources": merged["resources"],
        "evals": merged.get("evals") or [],
        "runtime_keys": runtime_keys,
        "name_keys": name_keys,
    }
    name_segments = [
        (alias, _lookup(view, path)) for alias, path in name_keys.items()
    ]
    covered = set(name_keys.values())
    override_segments_map: dict[str, Any] = {}
    for item in applied:
        top = _top_param(item.path)
        if top is None or top in runtime_keys:
            continue
        if item.path in covered:
            continue
        relative = item.path[len("params.") :]
        override_segments_map.pop(relative, None)
        override_segments_map[relative] = _lookup(merged, item.path)
    override_segments = list(override_segments_map.items())

    payload = fingerprint_input(
        merged["kind"],
        merged["entry"],
        merged["params"],
        runtime_keys,
        None if model is None else model.canonical,
    )
    canonical = canonical_json(payload)
    return ResolvedSpec(
        spec_path=repo_relative(spec_path, repo_root),
        stem=stem,
        kind=merged["kind"],
        entry=merged["entry"],
        params=merged["params"],
        runtime_keys=runtime_keys,
        resources=merged["resources"],
        evals=list(merged.get("evals") or []),
        name_keys=name_keys,
        description=merged.get("description") or "",
        overrides=annotated_overrides([*runtime_overrides, *applied], runtime_keys),
        name_segments=name_segments,
        override_segments=override_segments,
        model=model,
        fingerprint_input=payload,
        canonical=canonical,
        fingerprint=fingerprint_hex(canonical),
    )


def _validate_resolved(
    merged: dict[str, Any],
    *,
    spec_path: Path,
    repo_root: Path,
    entries: dict[str, Entry],
    model_ref: str | None,
    require_model: bool,
) -> None:
    kind = merged.get("kind")
    entry_name = merged.get("entry")
    if kind not in ("train", "eval"):
        raise UsageError(f"{spec_path.name}: kind must be 'train' or 'eval'")
    if not isinstance(entry_name, str) or entry_name not in entries:
        raise UsageError(f"unknown entry: {entry_name!r}")
    entry = entries[entry_name]
    if entry.kind != kind:
        raise UsageError(
            f"kind {kind!r} does not match entry {entry_name!r} kind {entry.kind!r}"
        )
    if not isinstance(merged.get("params"), dict):
        raise UsageError("params must be a mapping")
    resources = merged.get("resources")
    if not isinstance(resources, dict):
        raise UsageError(f"{spec_path.name}: resources must be a mapping")
    description = merged.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise UsageError("description must be a string")
    merged["description"] = description

    evals = merged.get("evals") or []
    if kind == "eval" and evals:
        raise UsageError("eval specs cannot list evals")
    if not isinstance(evals, list) or not all(isinstance(item, str) for item in evals):
        raise UsageError("evals must be a list of spec paths")
    if kind == "train":
        for rel in evals:
            child = repo_root / rel
            peeked = _load_spec_merged(child, ())
            if peeked.get("kind") != "eval":
                raise UsageError(f"eval spec {rel} does not have kind: eval")
            if not isinstance(peeked.get("entry"), str):
                raise UsageError(f"eval spec {rel} is missing entry")

    name_keys = merged.get("name_keys") or {}
    if not isinstance(name_keys, dict):
        raise UsageError("name_keys must be a mapping")
    view = {
        "params": merged["params"],
        "resources": resources,
        "evals": evals,
        "runtime_keys": merged.get("runtime_keys") or [],
        "name_keys": name_keys,
        "description": description,
        "kind": kind,
        "entry": entry_name,
    }
    for alias, path in name_keys.items():
        if not isinstance(alias, str) or not isinstance(path, str):
            raise UsageError("name_keys must map an alias to a dotted path")
        try:
            value = _lookup(view, path)
        except KeyError:
            raise UsageError(f"name_keys path {path!r} does not resolve") from None
        if isinstance(value, (dict, list)):
            raise UsageError(f"name_keys path {path!r} is not a scalar")

    if kind == "train" and model_ref:
        raise UsageError("train spec does not take a model reference")
    if require_model and kind == "eval" and not model_ref:
        raise UsageError("eval spec requires a model reference")
