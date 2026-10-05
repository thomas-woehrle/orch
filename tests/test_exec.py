"""Wrapper helpers that do not need a cluster."""

from pathlib import Path

import pytest

from orch.exec.common import finite_metrics, write_failure
from orch.exec.train import model_ready


def test_model_dir_contract(tmp_path: Path):
    assert not model_ready(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    assert not model_ready(tmp_path)
    (tmp_path / "model.safetensors").write_text("weights")
    assert model_ready(tmp_path)


def test_wrapper_failure_is_appended_to_stderr_log(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    try:
        raise RuntimeError("entry tooluse_eval has no parser")
    except RuntimeError as exc:
        write_failure(tmp_path, exc)
    logged = (tmp_path / "stderr.log").read_text()
    assert "RuntimeError: entry tooluse_eval has no parser" in logged
    assert "RuntimeError: entry tooluse_eval has no parser" in capsys.readouterr().err


def test_metrics_must_be_finite():
    finite_metrics({"accuracy": 0.5})
    with pytest.raises(ValueError):
        finite_metrics({})
    with pytest.raises(ValueError):
        finite_metrics({"accuracy": float("nan")})
