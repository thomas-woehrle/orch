"""The single YAML 1.2 loader and dumper (docs/orchestrator.md C6)."""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def _yaml() -> YAML:
    """Parser pinned to YAML 1.2. The dumper does not emit a version directive."""
    yaml = YAML(typ="safe", pure=True)
    yaml.version = (1, 2)
    yaml.default_flow_style = False
    yaml.allow_unicode = True
    yaml.sort_base_mapping_type_on_output = False
    return yaml


def _plain(value: Any) -> Any:
    """Convert loader types to plain JSON-compatible Python, preserving key order."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)
    raise TypeError(f"unsupported YAML value type: {type(value).__name__}")


def load_path(path: Path) -> Any:
    return load_text(Path(path).read_text())


def load_text(text: str) -> Any:
    if text.strip() == "":
        return None
    loaded = _yaml().load(text)
    if loaded is None:
        return None
    return _plain(loaded)


def parse_value(text: str) -> Any:
    """Parse one CLI ``--set`` value as a YAML 1.2 scalar or flow collection."""
    if text == "":
        return ""
    return load_text(text)


def dump_text(data: Any) -> str:
    buf = StringIO()
    yaml = _yaml()
    # Parsing stays on 1.2; don't write `%YAML 1.2` into files the scripts read.
    yaml.version = None
    yaml.explicit_start = False
    yaml.dump(data, buf)
    text = buf.getvalue()
    if not text.endswith("\n"):
        text += "\n"
    return text
