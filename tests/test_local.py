"""Local backend: claims, job.sh, and tmux state. No GPU required."""

from types import SimpleNamespace

import pytest

from orch.backends.local import (
    LocalBackend,
    claim,
    run_id_for_session,
    session_name_for,
)
from orch.config import Cluster
from orch.conn import Result
from orch.errors import UsageError
from orch.render import build_env

DEVICES = [0, 1, 2, 3]


def _cluster() -> Cluster:
    return Cluster(
        name="local",
        backend="local",
        repo_dir="/repo",
        work_dir="/work",
        devices=DEVICES,
        setup="",
    )


def test_session_names_round_trip():
    assert session_name_for("3f9a2c1d7e4b-a1") == "orch_3f9a2c1d7e4b_a1"
    assert run_id_for_session("orch_3f9a2c1d7e4b_a12") == "3f9a2c1d7e4b-a12"
    assert run_id_for_session("noise") is None


def test_claim_takes_the_first_free_device(tmp_path):
    assert claim(tmp_path, [0, 1], "orch_a_a1", set(), set()) == 0
    assert (tmp_path / ".devices" / "0").read_text().strip() == "orch_a_a1"
    # 0 is held by a live session; 1 is free.
    assert claim(tmp_path, [0, 1], "orch_b_a1", {"orch_a_a1"}, set()) == 1


def test_claim_skips_a_busy_device_and_reuses_a_dead_session(tmp_path):
    (tmp_path / ".devices").mkdir()
    (tmp_path / ".devices" / "0").write_text("orch_dead_a1\n")
    # 0's owner is gone, but a process is on it. 1 is claimed by a live session.
    (tmp_path / ".devices" / "1").write_text("orch_live_a1\n")
    assert claim(tmp_path, [0, 1], "orch_new_a1", {"orch_live_a1"}, {0}) is None
    # The operator's process left. The dead claim can be taken.
    assert claim(tmp_path, [0, 1], "orch_new_a1", {"orch_live_a1"}, set()) == 0
    assert (tmp_path / ".devices" / "0").read_text().strip() == "orch_new_a1"


def test_local_resources():
    backend = LocalBackend("/work")
    cluster = _cluster()
    backend.validate_resources({"gpus": 1, "time": "01:00:00", "mem": "24G"}, cluster)
    backend.validate_resources({"gpus": 1, "device": 2}, cluster)
    with pytest.raises(UsageError, match="gpus"):
        backend.validate_resources({"gpus": 2}, cluster)
    with pytest.raises(UsageError, match="device"):
        backend.validate_resources({"gpus": 1, "device": 9}, cluster)


def test_job_sh_holds_and_pinned_device():
    env = build_env(
        cluster_env={},
        entry_env={},
        entity="ent",
        project="proj",
        job_kind="eval",
        run_id="3f9a2c1d7e4b-a1",
        run_name="tooluse",
        tags=["kind:eval"],
        group=None,
        run_dir="/work/3f9a2c1d7e4b-a1",
    )
    job = SimpleNamespace(
        run_id="3f9a2c1d7e4b-a1",
        fingerprint="3f9a2c1d7e4b",
        attempt=1,
        resources={"gpus": 1, "time": "01:00:00", "mem": "24G"},
    )
    script = LocalBackend("/work").render_job_script(
        job,
        _cluster(),
        env,
        "/repo/.worktrees/abc",
        "/repo/.venv/bin/python",
        ["orch_train_a1"],
    )
    assert script.startswith("#!/bin/bash\n")
    assert "while tmux has-session -t orch_train_a1" in script
    assert "orch.backends.local claim /work 0,1,2,3 $ORCH_JOB_ID" in script
    assert "export ORCH_JOB_ID=orch_3f9a2c1d7e4b_a1" in script
    assert "export CUDA_VISIBLE_DEVICES=$ORCH_DEVICE" in script
    assert script.rstrip().endswith(
        "/repo/.venv/bin/python -m orch exec /work/3f9a2c1d7e4b-a1"
    )
    assert "\nexec /repo/.venv/bin/python" not in script

    job.resources = {"gpus": 1, "device": 3}
    pinned = LocalBackend("/work").render_job_script(
        job, _cluster(), env, "/repo/.worktrees/abc", "/repo/.venv/bin/python", []
    )
    assert "ORCH_DEVICE=3\n" in pinned
    assert "claim" not in pinned
    assert "while tmux has-session" not in pinned


class _Conn:
    def __init__(self, sessions: str, files: dict[str, str]) -> None:
        self.sessions = sessions
        self.files = files
        self.commands: list[str] = []

    def run(self, command: str, check: bool = True, stream: bool = False) -> Result:
        del stream
        self.commands.append(command)
        if "list-sessions" in command:
            if not self.sessions:
                return Result(1, "", "no server")
            return Result(0, self.sessions, "")
        if command.startswith("cat "):
            path = command.split(" ", 1)[1].strip("'")
            if path not in self.files:
                return Result(1, "", "missing")
            return Result(0, self.files[path], "")
        if command.startswith("tmux new-session") or command.startswith("mkdir"):
            return Result(0, "", "")
        return Result(0, "", "")


def test_status_map_reads_state_files():
    conn = _Conn(
        "orch_3f9a2c1d7e4b_a1\norch_aaaaaaaaaaaa_a2\nnoise\n",
        {
            "/work/3f9a2c1d7e4b-a1/state": "running\n",
            # a2 has no state file yet
        },
    )
    states = LocalBackend("/work").status_map(conn)
    assert states == {
        "orch_3f9a2c1d7e4b_a1": "running",
        "orch_aaaaaaaaaaaa_a2": "queued",
    }


def test_submit_session_name_matches_the_script():
    conn = _Conn("", {})
    session = LocalBackend("/work").submit(conn, "/work/3f9a2c1d7e4b-a1", [])
    assert session == "orch_3f9a2c1d7e4b_a1"
    assert any("tmux new-session -d -s 'orch_3f9a2c1d7e4b_a1'" in command for command in conn.commands)
