"""Model reference parsing and canonicalisation. See docs/orchestrator.md §5.4."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from orch.errors import UsageError

_FINGERPRINT = re.compile(r"^[0-9a-f]{12}$")
_VERSION = re.compile(r"^v\d+$")


class ModelLookup(Protocol):
    def highest_finished_attempt(self, fingerprint: str) -> int | None: ...

    def resolve_artifact_version(self, entity: str, project: str, name: str, alias: str) -> str: ...

    def run_name(self, run_id: str) -> str | None: ...


@dataclass(frozen=True)
class ModelResolution:
    raw: str
    canonical: str
    upstream_run_id: str | None
    handle: str
    form: str  # hf | run | artifact


def _hf_handle(repo: str) -> str:
    segment = repo.rsplit("/", 1)[-1]
    return f"hf:{segment}"


def _artifact_parts(body: str) -> tuple[str, str, str, str]:
    if ":" not in body:
        raise UsageError(
            f"artifact reference {body!r} must be entity/project/name:version"
        )
    path, version = body.rsplit(":", 1)
    parts = path.split("/")
    if len(parts) != 3 or not all(parts) or not version:
        raise UsageError(
            f"artifact reference must be entity/project/name:version, got {body!r}"
        )
    return parts[0], parts[1], parts[2], version


def require_eval_model_artifact(model: str, lookup: Any) -> None:
    """Refuse ``orch eval`` when the named model artifact is not already logged.

    ``hf:`` is a hub repo, so there is nothing to look up. ``run:`` and
    ``fingerprint:`` must already have a model artifact. A train that is still
    queued or running is refused here; child evals submitted with a train by
    ``orch run`` do not call this.
    """
    if model.startswith("hf:"):
        return
    if lookup is None:
        raise UsageError("checking the model artifact needs a W&B lookup")
    resolved = resolve_model_ref(model, lookup)
    if resolved.form == "artifact":
        name = resolved.canonical[len("artifact:") :]
        if not lookup.artifact_exists(name):
            raise UsageError(f"{resolved.canonical} does not exist")
        return
    if resolved.upstream_run_id is None or not lookup.has_model_artifact(resolved.upstream_run_id):
        raise UsageError(f"{model} has no model artifact")


def resolve_model_ref(raw: str, lookup: ModelLookup | None = None) -> ModelResolution:
    if not isinstance(raw, str) or not raw:
        raise UsageError("model reference must be a non-empty string")

    if raw.startswith("hf:"):
        body = raw[3:]
        if not body:
            raise UsageError("hf: reference is missing a repo")
        repo, _rev = body.split("@", 1) if "@" in body else (body, None)
        if not repo:
            raise UsageError(f"malformed model reference {raw!r}")
        return ModelResolution(
            raw=raw,
            canonical=raw,
            upstream_run_id=None,
            handle=_hf_handle(repo),
            form="hf",
        )

    if raw.startswith("run:"):
        run_id = raw[4:]
        if not run_id:
            raise UsageError("run: reference is missing a run id")
        handle = run_id
        if lookup is not None:
            name = lookup.run_name(run_id)
            if name:
                handle = name
        return ModelResolution(
            raw=raw,
            canonical=raw,
            upstream_run_id=run_id,
            handle=handle,
            form="run",
        )

    if raw.startswith("fingerprint:"):
        fp = raw[len("fingerprint:") :]
        if not _FINGERPRINT.fullmatch(fp):
            raise UsageError(
                f"fingerprint: reference must be 12 hex characters, got {fp!r}"
            )
        if lookup is None:
            raise UsageError(
                "fingerprint: references need a W&B lookup to resolve the "
                "finished attempt; this is the inspect exception to the "
                "offline rule"
            )
        attempt = lookup.highest_finished_attempt(fp)
        if attempt is None:
            raise UsageError(f"fingerprint: reference {fp} has no finished attempt")
        run_id = f"{fp}-a{attempt}"
        handle = run_id
        name = lookup.run_name(run_id)
        if name:
            handle = name
        return ModelResolution(
            raw=raw,
            canonical=f"run:{run_id}",
            upstream_run_id=run_id,
            handle=handle,
            form="run",
        )

    if raw.startswith("artifact:"):
        entity, project, name, version = _artifact_parts(raw[len("artifact:") :])
        if _VERSION.fullmatch(version):
            concrete = version
        else:
            if lookup is None:
                raise UsageError(
                    "artifact: aliases need a W&B lookup to resolve to a "
                    "concrete version; this is the inspect exception to the "
                    "offline rule"
                )
            concrete = lookup.resolve_artifact_version(entity, project, name, version)
            if not _VERSION.fullmatch(concrete):
                raise UsageError(
                    f"artifact alias {version!r} resolved to {concrete!r}, "
                    "which is not a concrete version"
                )
        canonical = f"artifact:{entity}/{project}/{name}:{concrete}"
        return ModelResolution(
            raw=raw,
            canonical=canonical,
            upstream_run_id=None,
            handle=f"artifact:{name}:{concrete}",
            form="artifact",
        )

    raise UsageError(
        f"model reference {raw!r} must start with hf:, run:, fingerprint:, or artifact:"
    )
