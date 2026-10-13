"""Cluster entries in orch.local.yaml."""

import pytest

from orch.config import _load_cluster
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
