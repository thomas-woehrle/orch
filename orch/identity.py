"""Content-derived identity. See docs/orchestrator.md §5.5."""

from __future__ import annotations

import hashlib
import json
from typing import Any

FINGERPRINT_VERSION = 1


def identity_params(params: dict[str, Any], runtime_keys: list[str]) -> dict[str, Any]:
    excluded = set(runtime_keys)
    return {key: value for key, value in params.items() if key not in excluded}


def fingerprint_input(
    kind: str,
    entry: str,
    params: dict[str, Any],
    runtime_keys: list[str],
    model_ref: str | None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "entry": entry,
        "params": identity_params(params, runtime_keys),
        "model_ref": model_ref,
    }


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def fingerprint_hex(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def fingerprint_document(
    payload: dict[str, Any], canonical: str, fingerprint: str
) -> dict[str, Any]:
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "input": payload,
        "canonical": canonical,
        "fingerprint": fingerprint,
    }
