"""Cluster entries in orch.local.yaml."""

from pathlib import Path

import pytest

from orch.config import _load_cluster, load_orch
from orch.conn import Local
from orch.errors import UsageError


def test_remote_cluster_requires_ssh_and_header():
    cluster = _load_cluster(
        "eddie",
        {
            "backend": "sge",
            "ssh": "eddie",
            "repo_dir": "/repo",
            "work_dir": "/work",
            "header": "#$ -l gpu={{ gpus }}",
            "setup": "module load cuda",
        },
    )
    assert cluster.backend == "sge"
    assert cluster.devices is None
    with pytest.raises(UsageError, match="devices"):
        _load_cluster(
            "eddie",
            {
                "backend": "slurm",
                "ssh": "dmog",
                "repo_dir": "/repo",
                "work_dir": "/work",
                "header": "#SBATCH --time={{ time }}",
                "devices": [0],
            },
        )


def test_local_cluster_takes_devices_and_refuses_ssh():
    cluster = _load_cluster(
        "local",
        {
            "backend": "local",
            "repo_dir": "/repo",
            "work_dir": "/work",
            "devices": [0, 1, 2],
            "setup": "",
        },
    )
    assert cluster.ssh is None
    assert cluster.header is None
    assert cluster.devices == [0, 1, 2]
    with pytest.raises(UsageError, match="ssh"):
        _load_cluster(
            "local",
            {
                "backend": "local",
                "ssh": "localhost",
                "repo_dir": "/repo",
                "work_dir": "/work",
                "devices": [0],
            },
        )
    with pytest.raises(UsageError, match="devices"):
        _load_cluster(
            "local",
            {"backend": "local", "repo_dir": "/repo", "work_dir": "/work", "devices": []},
        )


def test_train_artifact_defaults_true_and_sequential_opts_out():
    root = Path(__file__).resolve().parent / "fixtures" / "project"
    entries = load_orch(root).entries
    assert entries["train"].artifact is True
    assert entries["sequential_train"].artifact is False
    assert entries["tooluse_eval"].artifact is True


def test_eval_entry_rejects_artifact_flag(tmp_path: Path):
    (tmp_path / "orch.yaml").write_text(
        """\
version: 1
entries:
  tooluse_eval:
    kind: eval
    cmd: "python eval_tooluse.py"
    parser: "orch_adapters:parse_eval_results"
    artifact: false
"""
    )
    with pytest.raises(UsageError, match="artifact is only valid on train entries"):
        load_orch(tmp_path)


def test_local_conn_round_trip(tmp_path):
    conn = Local()
    conn.connect()
    dest = tmp_path / "nested" / "job.json"
    conn.put_text(str(dest), "ok\n")
    assert conn.exists(str(dest))
    result = conn.run(f"cat {dest}")
    assert result.stdout == "ok\n"
    missing = conn.run("false", check=False)
    assert missing.returncode != 0
    conn.close()
