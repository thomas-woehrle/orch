"""Human run names. See docs/orchestrator.md §5.6."""

from __future__ import annotations

import re
from typing import Any


def shorten(value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    else:
        text = str(value)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = re.sub(r"\s+", "_", text)
    if len(text) > 32:
        text = text[:31] + "…"
    return text


def format_run_name(
    stem: str,
    name_segments: list[tuple[str, Any]],
    override_segments: list[tuple[str, Any]],
    *,
    attempt: int | None = None,
    model_handle: str | None = None,
) -> str:
    head = stem if not model_handle else f"{stem}@{model_handle}"
    parts = [head]
    for alias, value in name_segments:
        parts.append(f"{alias}={shorten(value)}")
    for key, value in override_segments:
        parts.append(f"+{key}={shorten(value)}")
    if attempt is not None and attempt > 1:
        parts.append(f"(a{attempt})")
    return " ".join(parts)
