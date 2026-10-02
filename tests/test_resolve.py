"""Resolution, extends, overrides, and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from orch.config import load_orch
from orch.errors import UsageError
from orch.resolve import (
    Override,
    cli_overrides_from_argv,
    deep_merge,
    parse_assignment,
    resolve_spec,
)

ORCH_YAML = """\
version: 1
entries:
  train:
    kind: train
    cmd: "python train.py --config {params}"
  tooluse_eval:
    kind: eval
    cmd: "python eval_tooluse.py --model_path {model_path} --output_dir {output_dir} {flags}"
    parser: "orch_adapters:parse_eval_results"
"""


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def repo(tmp_path: Path) -> Path:
    write(tmp_path, "orch.yaml", ORCH_YAML)
    write(
        tmp_path,
        "hparams/base.yaml",
        """\
runtime_keys: [steps_per_generation]
method: sdft
model_name: Qwen/Qwen2.5-1.5B-Instruct
learning_rate: 5.0e-5
gradient_accumulation_steps: 32
steps_per_generation: 32
""",
    )
    write(tmp_path, "hparams/child.yaml", "extends: base.yaml\nseed: 1\n")
    write(
        tmp_path,
        "configs/eval/tooluse.yaml",
        """\
kind: eval
entry: tooluse_eval
runtime_keys: [gpu_memory_utilization]
params: {max_new_tokens: 1024, temperature: 0.0, gpu_memory_utilization: 0.8}
resources: {gpus: 1, h_rt: "01:00:00", mem: 24G}
""",
    )
    write(
        tmp_path,
        "configs/train/job.yaml",
        """\
kind: train
entry: train
description: demo
params: hparams/child.yaml
resources: {gpus: 1, h_rt: "12:00:00", mem: 24G}
evals: [configs/eval/tooluse.yaml]
name_keys: {model: params.model_name, lr: params.learning_rate}
""",
    )
    return tmp_path


def resolve(root: Path, rel: str = "configs/train/job.yaml", **kwargs):
    return resolve_spec(
        root / rel,
        repo_root=root,
        entries=load_orch(root).entries,
        **kwargs,
    )


def test_deep_merge_order_lists_and_null():
    merged = deep_merge(
        {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2]},
        {"nested": {"y": None, "z": 3}, "items": [9], "b": 4},
    )
    assert list(merged) == ["a", "nested", "items", "b"]
    assert merged["nested"]["y"] is None
    assert merged["nested"]["z"] == 3
    assert merged["items"] == [9]


def test_extends_chain_strips_reserved_keys_and_unions_runtime(tmp_path: Path):
    root = repo(tmp_path)
    resolved = resolve(root)
    assert "extends" not in resolved.params
    assert "runtime_keys" not in resolved.params
    assert resolved.params["method"] == "sdft"
    assert resolved.params["seed"] == 1
    assert resolved.params["learning_rate"] == pytest.approx(5e-5)
    assert isinstance(resolved.params["learning_rate"], float)
    assert resolved.runtime_keys == ["steps_per_generation"]
    assert resolved.resources["mem"] == "24G"


def test_file_relative_extends_and_repo_relative_params(tmp_path: Path):
    root = repo(tmp_path)
    write(
        tmp_path,
        "configs/train/derived.yaml",
        """\
extends: job.yaml
set:
  params.model_name: Qwen/Qwen2.5-7B-Instruct
  resources.h_rt: "24:00:00"
""",
    )
    resolved = resolve(root, "configs/train/derived.yaml")
    assert resolved.params["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert resolved.resources["h_rt"] == "24:00:00"
    assert resolved.spec_path == "configs/train/derived.yaml"
    model_override = next(item for item in resolved.overrides if item["path"] == "params.model_name")
    assert model_override["classification"] == "identity"
    assert model_override["value"] == "Qwen/Qwen2.5-7B-Instruct"


def test_override_grammar_and_classification(tmp_path: Path):
    root = repo(tmp_path)
    parsed = parse_assignment("learning_rate=5e-5", prefix="params.")
    assert parsed.value == pytest.approx(5e-5)
    assert isinstance(parsed.value, float)
    assert parse_assignment("flag=true", prefix="params.").value is True
    listed = parse_assignment("tasks=[a, b]", prefix="params.")
    assert listed.value == ["a", "b"]

    base = resolve(root)
    changed = resolve(
        root,
        cli_overrides=[parse_assignment("learning_rate=1.0e-5", prefix="params.")],
    )
    assert changed.params["learning_rate"] == pytest.approx(1e-5)
    assert changed.fingerprint != base.fingerprint
    assert changed.overrides == [
        {
            "path": "params.learning_rate",
            "value": changed.params["learning_rate"],
            "classification": "identity",
        }
    ]

    same = resolve(
        root,
        cli_overrides=[parse_assignment("steps_per_generation=16", prefix="params.")],
    )
    assert same.fingerprint == base.fingerprint
    assert same.run_name() == base.run_name()
    assert same.overrides == [
        {"path": "params.steps_per_generation", "value": 16, "classification": "runtime"}
    ]


def test_runtime_keys_override_applies_first_and_keeps_params_declared(tmp_path: Path):
    root = repo(tmp_path)
    demoted = resolve(
        root,
        cli_overrides=[
            parse_assignment("learning_rate=1.0e-5", prefix="params."),
            parse_assignment("runtime_keys=[learning_rate]"),
        ],
    )
    assert "learning_rate" in demoted.runtime_keys
    assert "steps_per_generation" in demoted.runtime_keys
    assert "learning_rate" not in demoted.fingerprint_input["params"]
    learning_rate = next(item for item in demoted.overrides if item["path"] == "params.learning_rate")
    assert learning_rate["classification"] == "runtime"


def test_cli_override_order():
    overrides = cli_overrides_from_argv(
        ["--set", "params.a=1", "-p", "b=true", "-r", "gpus=2"]
    )
    assert [item.path for item in overrides] == [
        "params.a",
        "params.b",
        "resources.gpus",
    ]
    assert overrides[1].value is True
    assert overrides[2].value == 2


def test_validation_errors(tmp_path: Path):
    root = repo(tmp_path)
    write(tmp_path, "configs/train/bad.yaml", "kind: train\nentry: train\nnope: 1\n")
    with pytest.raises(UsageError, match="unknown spec key"):
        resolve(root, "configs/train/bad.yaml")

    write(
        tmp_path,
        "configs/train/mismatch.yaml",
        """\
kind: eval
entry: train
params: {}
resources: {gpus: 1}
""",
    )
    with pytest.raises(UsageError, match="does not match entry"):
        resolve(root, "configs/train/mismatch.yaml", model_ref="hf:org/model")

    write(tmp_path, "configs/train/loop.yaml", "extends: loop.yaml\nkind: train\n")
    with pytest.raises(UsageError, match="extends cycle"):
        resolve(root, "configs/train/loop.yaml")

    write(
        tmp_path,
        "configs/train/badname.yaml",
        """\
kind: train
entry: train
params: {a: 1}
resources: {gpus: 1}
name_keys: {a: params.missing}
""",
    )
    with pytest.raises(UsageError, match="does not resolve"):
        resolve(root, "configs/train/badname.yaml")

    with pytest.raises(UsageError, match="targets kind"):
        resolve(
            root,
            cli_overrides=[Override("kind", "eval", "kind=eval")],
        )
    with pytest.raises(UsageError, match="requires a model reference"):
        resolve(root, "configs/eval/tooluse.yaml")
    with pytest.raises(UsageError, match="does not take a model reference"):
        resolve(root, model_ref="hf:org/model")


def test_two_files_same_resolution_share_fingerprint(tmp_path: Path):
    root = repo(tmp_path)
    write(
        tmp_path,
        "configs/train/copy.yaml",
        (root / "configs/train/job.yaml").read_text(),
    )
    assert resolve(root).fingerprint == resolve(root, "configs/train/copy.yaml").fingerprint


def test_resources_and_description_do_not_change_fingerprint(tmp_path: Path):
    root = repo(tmp_path)
    base = resolve(root)
    moved = resolve(
        root,
        cli_overrides=[parse_assignment("h_rt=1:00:00", prefix="resources.")],
    )
    assert moved.fingerprint == base.fingerprint
    assert moved.run_name() == base.run_name()
