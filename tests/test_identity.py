"""Fingerprint canonicalisation. See docs/orchestrator.md §5.5."""

from __future__ import annotations

import json
from pathlib import Path

from orch.config import load_orch
from orch.identity import canonical_json, fingerprint_hex, fingerprint_input
from orch.resolve import resolve_spec

# Locked against configs/train/sdft_tooluse.yaml. A change to the identity
# scheme or to that spec's identity-bearing params fails this test.
SDFT_TOOLUSE_FINGERPRINT = "944267f6c6fe"
SDFT_TOOLUSE_CANONICAL = (
    '{"entry":"train","kind":"train","model_ref":null,"params":{"bf16":true,'
    '"disable_dropout":false,"distillation_alpha":0.0,"distillation_is_clip":null,'
    '"distillation_mode":"full_logits","dtype":"bfloat16","generate_from_teacher":false,'
    '"gradient_accumulation_steps":32,"learning_rate":5e-05,"lr_scheduler_type":"cosine",'
    '"max_completion_length":2048,"max_grad_norm":1.0,"max_prompt_length":1024,'
    '"method":"sdft","model_name":"Qwen/Qwen2.5-1.5B-Instruct","num_generations":1,'
    '"num_loss_tokens_to_skip":3,"num_train_epochs":2,"per_device_train_batch_size":1,'
    '"seed":42,"shuffle_dataset":true,"teacher_model_kind":"ema",'
    '"teacher_prompt_template":"\\n{prompt}\\n\\nThis is an example for a response to the question:\\n{privileged_context}\\n\\nNow answer with a response of your own, including the thinking process.\\n",'
    '"teacher_sync_steps":1,"teacher_update_rate":0.01,"temperature":1.0,'
    '"train_data":"data/tooluse_data/train_data","use_vllm":false,"val_size":0.05,'
    '"warmup_steps":0.1,"weight_decay":0.0}}'
)


def _resolve(root: Path, rel: str, **kwargs):
    return resolve_spec(
        root / rel,
        repo_root=root,
        entries=load_orch(root).entries,
        **kwargs,
    )


def test_key_order_does_not_matter_but_list_order_and_number_types_do():
    left = fingerprint_input("train", "train", {"b": 1, "a": [1, 2]}, [], None)
    right = fingerprint_input("train", "train", {"a": [1, 2], "b": 1}, [], None)
    assert canonical_json(left) == canonical_json(right)

    as_int = fingerprint_hex(canonical_json(fingerprint_input("train", "e", {"n": 2}, [], None)))
    as_float = fingerprint_hex(
        canonical_json(fingerprint_input("train", "e", {"n": 2.0}, [], None))
    )
    assert as_int != as_float
    assert json.loads(canonical_json(fingerprint_input("train", "e", {"n": 2}, [], None)))[
        "params"
    ]["n"] == 2

    absent = fingerprint_hex(canonical_json(fingerprint_input("train", "e", {}, [], None)))
    explicit_null = fingerprint_hex(
        canonical_json(fingerprint_input("train", "e", {"n": None}, [], None))
    )
    assert absent != explicit_null


def test_unicode_round_trip_and_runtime_exclusion_and_model_ref():
    payload = fingerprint_input(
        "eval",
        "tooluse_eval",
        {"prompt": "café", "steps_per_generation": 16, "lr": 1},
        ["steps_per_generation"],
        "hf:Qwen/Qwen2.5-1.5B-Instruct",
    )
    canonical = canonical_json(payload)
    assert "café" in canonical
    assert "steps_per_generation" not in canonical
    assert "hf:Qwen/Qwen2.5-1.5B-Instruct" in canonical
    again = fingerprint_input(
        "eval",
        "tooluse_eval",
        {"lr": 1, "prompt": "café", "steps_per_generation": 32},
        ["steps_per_generation"],
        "hf:Qwen/Qwen2.5-1.5B-Instruct",
    )
    assert fingerprint_hex(canonical_json(again)) == fingerprint_hex(canonical)


def test_committed_sdft_tooluse_fingerprint_is_stable():
    root = Path(__file__).resolve().parents[1]
    resolved = _resolve(root, "configs/train/sdft_tooluse.yaml")
    assert resolved.fingerprint_input["kind"] == "train"
    assert resolved.fingerprint_input["entry"] == "train"
    assert resolved.fingerprint_input["model_ref"] is None
    assert "steps_per_generation" not in resolved.fingerprint_input["params"]
    assert "learning_rate" in resolved.fingerprint_input["params"]
    assert resolved.fingerprint_input["params"]["distillation_is_clip"] is None
    # Re-resolve and compare to the canonical string captured below.
    assert resolved.canonical == SDFT_TOOLUSE_CANONICAL
    assert resolved.fingerprint == fingerprint_hex(SDFT_TOOLUSE_CANONICAL)
    assert resolved.fingerprint == SDFT_TOOLUSE_FINGERPRINT
