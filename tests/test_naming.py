"""Run names. See docs/orchestrator.md §5.6."""

from __future__ import annotations

from pathlib import Path

from orch.config import load_orch
from orch.naming import format_run_name, shorten
from orch.resolve import parse_assignment, resolve_spec


def test_shorten_rules():
    assert shorten("Qwen/Qwen2.5-1.5B-Instruct") == "Qwen2.5-1.5B-Instruct"
    assert shorten(5e-5) == "5e-05"
    assert shorten(32) == "32"
    assert shorten("a value with spaces") == "a_value_with_spaces"
    long = "x" * 40
    assert shorten(long) == ("x" * 31) + "…"
    assert len(shorten(long)) == 32


def test_train_name_segments_and_attempt_suffix():
    name = format_run_name(
        "sdft_tooluse",
        [("model", "Qwen/Qwen2.5-1.5B-Instruct"), ("lr", 5e-5), ("bs", 32)],
        [],
    )
    assert name == "sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32"
    assert format_run_name("sdft_tooluse", [], [], attempt=2).endswith("(a2)")
    assert "(a1)" not in format_run_name("sdft_tooluse", [], [], attempt=1)


def test_eval_handle_and_uncovered_override(tmp_path: Path):
    name = format_run_name(
        "tooluse",
        [],
        [],
        model_handle="sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32",
    )
    assert name == "tooluse@sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32"

    root = tmp_path
    (root / "orch.yaml").write_text(
        """\
version: 1
entries:
  train:
    kind: train
    cmd: "python train.py --config {params}"
"""
    )
    (root / "configs/train").mkdir(parents=True)
    (root / "configs/train/job.yaml").write_text(
        """\
kind: train
entry: train
params: {model_name: Qwen/Qwen2.5-1.5B-Instruct, learning_rate: 5.0e-5, seed: 1}
resources: {gpus: 1}
name_keys: {model: params.model_name}
"""
    )
    resolved = resolve_spec(
        root / "configs/train/job.yaml",
        repo_root=root,
        entries=load_orch(root).entries,
        cli_overrides=[
            parse_assignment("seed=2", prefix="params."),
            parse_assignment("learning_rate=1.0e-5", prefix="params."),
        ],
    )
    assert resolved.run_name() == "job model=Qwen2.5-1.5B-Instruct +seed=2 +learning_rate=1e-05"


def test_hf_handle():
    name = format_run_name("tooluse", [], [], model_handle="hf:Qwen2.5-1.5B-Instruct")
    assert name == "tooluse@hf:Qwen2.5-1.5B-Instruct"
