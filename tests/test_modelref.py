"""Model reference canonical form. See docs/orchestrator.md §5.4."""

from __future__ import annotations

import pytest

from orch.errors import UsageError
from orch.modelref import require_eval_model_artifact, resolve_model_ref


class FakeLookup:
    def __init__(self) -> None:
        self.finished = {"abc123abc123": 2}
        self.names = {"abc123abc123-a2": "sdft_tooluse model=Qwen2.5-1.5B-Instruct"}
        self.aliases = {("e", "p", "model-abc", "latest"): "v4"}

    def highest_finished_attempt(self, fingerprint: str) -> int | None:
        return self.finished.get(fingerprint)

    def resolve_artifact_version(self, entity: str, project: str, name: str, alias: str) -> str:
        return self.aliases[(entity, project, name, alias)]

    def run_name(self, run_id: str) -> str | None:
        return self.names.get(run_id)


def test_hf_and_run_are_already_canonical():
    hf = resolve_model_ref("hf:Qwen/Qwen2.5-1.5B-Instruct")
    assert hf.canonical == "hf:Qwen/Qwen2.5-1.5B-Instruct"
    assert hf.upstream_run_id is None
    assert hf.handle == "hf:Qwen2.5-1.5B-Instruct"

    pinned = resolve_model_ref("hf:Qwen/Qwen2.5-1.5B-Instruct@abc")
    assert pinned.canonical == "hf:Qwen/Qwen2.5-1.5B-Instruct@abc"
    assert pinned.handle == "hf:Qwen2.5-1.5B-Instruct"

    run = resolve_model_ref("run:abc123abc123-a1", FakeLookup())
    assert run.canonical == "run:abc123abc123-a1"
    assert run.upstream_run_id == "abc123abc123-a1"


def test_fingerprint_and_artifact_need_lookup():
    with pytest.raises(UsageError, match="W&B lookup"):
        resolve_model_ref("fingerprint:abc123abc123")
    with pytest.raises(UsageError, match="no finished attempt"):
        resolve_model_ref("fingerprint:aaaaaaaaaaaa", FakeLookup())

    resolved = resolve_model_ref("fingerprint:abc123abc123", FakeLookup())
    assert resolved.canonical == "run:abc123abc123-a2"
    assert resolved.upstream_run_id == "abc123abc123-a2"
    assert resolved.handle == "sdft_tooluse model=Qwen2.5-1.5B-Instruct"

    with pytest.raises(UsageError, match="W&B lookup"):
        resolve_model_ref("artifact:e/p/model-abc:latest")
    concrete = resolve_model_ref("artifact:e/p/model-abc:v3")
    assert concrete.canonical == "artifact:e/p/model-abc:v3"
    assert concrete.handle == "artifact:model-abc:v3"
    aliased = resolve_model_ref("artifact:e/p/model-abc:latest", FakeLookup())
    assert aliased.canonical == "artifact:e/p/model-abc:v4"
    assert aliased.upstream_run_id is None


class ArtifactLookup(FakeLookup):
    def __init__(self) -> None:
        super().__init__()
        self.models = {"abc123abc123-a2"}
        self.artifacts = {"e/p/model-abc:v4"}

    def has_model_artifact(self, run_id: str) -> bool:
        return run_id in self.models

    def artifact_exists(self, name: str) -> bool:
        return name in self.artifacts


def test_eval_requires_a_logged_model_artifact():
    require_eval_model_artifact("hf:Qwen/Qwen2.5-1.5B-Instruct", None)
    lookup = ArtifactLookup()
    require_eval_model_artifact("run:abc123abc123-a2", lookup)
    require_eval_model_artifact("fingerprint:abc123abc123", lookup)
    require_eval_model_artifact("artifact:e/p/model-abc:latest", lookup)
    with pytest.raises(UsageError, match="no model artifact"):
        require_eval_model_artifact("run:abc123abc123-a1", lookup)
    with pytest.raises(UsageError, match="does not exist"):
        require_eval_model_artifact("artifact:e/p/model-abc:v3", lookup)


def test_bad_forms():
    with pytest.raises(UsageError, match="must start with"):
        resolve_model_ref("Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(UsageError, match="12 hex"):
        resolve_model_ref("fingerprint:nope", FakeLookup())
