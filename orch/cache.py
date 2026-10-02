"""Disposable local cache of scheduler ids. See docs/orchestrator.md §6.8."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    cluster: str
    scheduler_id: str
    fingerprint: str
    attempt: int
    kind: str
    name: str
    submitted_at: str


class Cache:
    def __init__(self, path: Path | None, entries: dict[str, CacheEntry] | None = None) -> None:
        self.path = path
        self.entries: dict[str, CacheEntry] = entries or {}

    @classmethod
    def load(cls, entity: str, project: str) -> Cache:
        safe = f"{entity}__{project}".replace("/", "_")
        path = Path.home() / ".local" / "state" / "orch" / f"{safe}.json"
        if not path.is_file():
            return cls(path, {})
        raw = json.loads(path.read_text())
        entries = {run_id: CacheEntry(**body) for run_id, body in raw.items()}
        return cls(path, entries)

    def for_fingerprint(self, fingerprint: str) -> list[CacheEntry]:
        found = [
            entry for entry in self.entries.values() if entry.fingerprint == fingerprint
        ]
        return sorted(found, key=lambda entry: entry.attempt)

    def by_run_id(self, run_id: str) -> CacheEntry | None:
        return self.entries.get(run_id)

    def upsert(self, run_id: str, entry: CacheEntry) -> None:
        self.entries[run_id] = entry

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {run_id: asdict(entry) for run_id, entry in self.entries.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
