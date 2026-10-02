"""Wrapper helpers that do not need a cluster."""

from pathlib import Path

import pytest

from orch.exec.common import finite_metrics
from orch.exec.train import model_ready


def test_model_dir_contract(tmp_path: Path):
    assert not model_ready(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert not model_ready(tmp_path)
    (tmp_path / "model.safetensors").write_text("weights")
    assert model_ready(tmp_path)


def test_metrics_must_be_finite():
    finite_metrics({"accuracy": 0.5})
    with pytest.raises(ValueError):
        finite_metrics({})
    with pytest.raises(ValueError):
        finite_metrics({"accuracy": float("nan")})
