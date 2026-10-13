"""Local backend: one tmux session per attempt. See docs/orchestrator.md §6.4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from orch.config import Cluster
from orch.errors import RemoteError, UsageError
from orch.render import _jinja, render_job_body, shquote
from orch.render import CANONICAL_RESOURCE_KEYS

HEAD = """#!/bin/bash
exec >>{{ run_dir }}/scheduler.out 2>>{{ run_dir }}/scheduler.err
export ORCH_JOB_ID={{ name }}
echo held > {{ run_dir }}/state
{% for dep in hold_on %}
while tmux has-session -t {{ dep }} 2>/dev/null; do sleep 20; done
{% endfor %}
{% if src_dir %}
cd {{ src_dir }}
{% endif %}
{% if device is not none %}
ORCH_DEVICE={{ device }}
flock {{ work_dir }}/.devices/lock sh -c 'echo $ORCH_JOB_ID > {{ work_dir }}/.devices/{{ device }}'
{% else %}
until ORCH_DEVICE=$(flock {{ work_dir }}/.devices/lock {{ venv_python }} -m orch.backends.local claim {{ work_dir }} {{ devices }} $ORCH_JOB_ID); do
  sleep 20
done
{% endif %}
export CUDA_VISIBLE_DEVICES=$ORCH_DEVICE
trap 'code=$?; rm -f {{ work_dir }}/.devices/$ORCH_DEVICE; echo exit:$code > {{ run_dir }}/state' EXIT
echo running > {{ run_dir }}/state
"""


def session_name_for(run_id: str) -> str:
    """``<fingerprint>-a<n>`` becomes the tmux session ``orch_<fingerprint>_a<n>``."""
    return "orch_" + run_id.replace("-", "_", 1)


def run_id_for_session(session: str) -> str | None:
    if not session.startswith("orch_"):
        return None
    body = session[len("orch_") :]
    fingerprint, separator, attempt = body.rpartition("_a")
    if not separator or not fingerprint or not attempt.isdigit():
        return None
    return f"{fingerprint}-a{attempt}"


class LocalBackend:
    name = "local"
    tools = ("tmux", "nvidia-smi", "flock")
    job_id_env = "ORCH_JOB_ID"

    def __init__(self, work_dir: str | None = None) -> None:
        self.work_dir = work_dir

    def validate_resources(self, resources: dict[str, Any], cluster: Cluster) -> None:
        unknown = sorted(set(resources) - CANONICAL_RESOURCE_KEYS)
        if unknown:
            raise UsageError(
                "resource key outside the canonical set "
                "(gpus, gpu_type, time, mem, device): " + ", ".join(unknown)
            )
        if resources.get("gpus") != 1:
            raise UsageError("a local cluster requires gpus: 1")
        device = resources.get("device")
        if device is not None and device not in (cluster.devices or []):
            raise UsageError(
                f"device {device} is not in cluster {cluster.name}'s devices"
            )

    def render_head(
        self,
        cluster: Cluster,
        *,
        name: str,
        run_dir: str,
        resources: dict[str, Any],
        hold_on: list[str],
        device: int | None,
        src_dir: str = "",
        venv_python: str = "",
    ) -> str:
        del resources  # time and mem are accepted and ignored; device is passed in
        devices = ",".join(str(item) for item in (cluster.devices or []))
        return _jinja().from_string(HEAD).render(
            name=name,
            run_dir=run_dir,
            hold_on=hold_on,
            device=device,
            src_dir=src_dir,
            venv_python=venv_python,
            work_dir=cluster.work_dir,
            devices=devices,
        ).rstrip("\n")

    def render_job_script(
        self,
        job: Any,
        cluster: Cluster,
        env: dict[str, str],
        src_dir: str,
        venv_python: str,
        hold_on: list[str],
    ) -> str:
        run_dir = f"{cluster.work_dir}/{job.run_id}"
        head = self.render_head(
            cluster,
            name=session_name_for(job.run_id),
            run_dir=run_dir,
            resources=job.resources,
            hold_on=hold_on,
            device=job.resources.get("device"),
            src_dir=src_dir,
            venv_python=venv_python,
        )
        body = render_job_body(
            setup=cluster.setup,
            env=env,
            run_dir=run_dir,
            src_dir=src_dir,
            venv_python=venv_python,
            exec_wrapper=False,
        )
        return head + "\n" + body

    def submit(self, conn: object, run_dir: str, hold_on: list[str]) -> str:
        del hold_on  # the script waits; tmux has no hold flag
        if not self.work_dir:
            raise UsageError("local backend needs work_dir")
        session = session_name_for(Path(run_dir).name)
        conn.run(f"mkdir -p {shquote(self.work_dir + '/.devices')}")  # type: ignore[attr-defined]
        conn.run(
            "tmux new-session -d -s "
            + shquote(session)
            + " bash "
            + shquote(run_dir + "/job.sh")
        )  # type: ignore[attr-defined]
        return session

    def status_map(self, conn: object) -> dict[str, str]:
        if not self.work_dir:
            raise UsageError("local backend needs work_dir")
        listed = conn.run("tmux list-sessions -F '#S'", check=False)  # type: ignore[attr-defined]
        if listed.returncode != 0:
            return {}
        found: dict[str, str] = {}
        for line in listed.stdout.splitlines():
            session = line.strip()
            run_id = run_id_for_session(session)
            if run_id is None:
                continue
            state = conn.run(  # type: ignore[attr-defined]
                f"cat {shquote(self.work_dir + '/' + run_id + '/state')}",
                check=False,
            )
            text = state.stdout.strip() if state.returncode == 0 else ""
            if text == "running" or text.startswith("exit:"):
                found[session] = "running"
            else:
                # held, missing, or still starting
                found[session] = "queued"
        return found

    def cancel(self, conn: object, scheduler_id: str) -> None:
        result = conn.run(f"tmux kill-session -t {shquote(scheduler_id)}", check=False)  # type: ignore[attr-defined]
        if result.returncode != 0:
            raise RemoteError(
                f"tmux kill-session failed ({result.returncode}): {result.stderr.strip()}"
            )


def claim(
    work_dir: Path,
    devices: list[int],
    session: str,
    live_sessions: set[str],
    busy_devices: set[int],
) -> int | None:
    """Claim the first free device. ``None`` when every device is taken.

    A claim file whose session is gone is stale and may be taken. A device with
    a compute process on it is left alone, so the operator's own jobs are
    respected. The caller holds the devices lock.
    """
    root = work_dir / ".devices"
    root.mkdir(parents=True, exist_ok=True)
    for index in devices:
        if index in busy_devices:
            continue
        path = root / str(index)
        if path.is_file():
            owner = path.read_text().strip()
            if owner and owner != session and owner in live_sessions:
                continue
        path.write_text(session + "\n")
        return index
    return None


def _live_sessions() -> set[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _busy_devices() -> set[int]:
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if gpus.returncode != 0:
        raise RuntimeError(gpus.stderr.strip() or "nvidia-smi --query-gpu failed")
    uuid_to_index: dict[str, int] = {}
    for line in gpus.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if apps.returncode != 0:
        raise RuntimeError(apps.stderr.strip() or "nvidia-smi --query-compute-apps failed")
    busy: set[int] = set()
    for line in apps.stdout.splitlines():
        uuid = line.strip()
        if uuid in uuid_to_index:
            busy.add(uuid_to_index[uuid])
    return busy


def _claim_main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: python -m orch.backends.local claim WORK_DIR DEVICES SESSION")
    work_dir = Path(argv[0])
    devices = [int(item) for item in argv[1].split(",") if item != ""]
    session = argv[2]
    chosen = claim(work_dir, devices, session, _live_sessions(), _busy_devices())
    if chosen is None:
        return 1
    print(chosen)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "claim":
        raise SystemExit("usage: python -m orch.backends.local claim WORK_DIR DEVICES SESSION")
    return _claim_main(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
