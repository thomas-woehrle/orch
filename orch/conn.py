"""How orch reaches a cluster: SSH or a local subprocess. See docs/orchestrator.md §6.4."""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from orch.errors import RemoteError


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str


class Conn(Protocol):
    def connect(self) -> None: ...

    def close(self) -> None: ...

    def run(self, command: str, *, check: bool = True, stream: bool = False) -> Result: ...

    def put_text(self, path: str, content: str) -> None: ...

    def exists(self, path: str) -> bool: ...


class Local:
    """Run login-side commands on this machine. ``ssh`` is unset on the cluster."""

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> Local:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def run(self, command: str, *, check: bool = True, stream: bool = False) -> Result:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if stream:
            out, err = _stream(proc)
        else:
            out, err = proc.communicate()
        code = proc.returncode if proc.returncode is not None else 1
        if check and code != 0:
            raise RemoteError(f"command failed ({code}): {command}\n{err}".rstrip())
        return Result(code, out, err)

    def put_text(self, path: str, content: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)

    def exists(self, path: str) -> bool:
        return Path(path).exists()


def _stream(proc: subprocess.Popen[str]) -> tuple[str, str]:
    out_parts: list[str] = []
    err_parts: list[str] = []

    def pump(stream, parts: list[str], mirror) -> None:
        assert stream is not None
        for line in stream:
            parts.append(line)
            mirror.write(line)
            mirror.flush()

    threads = [
        threading.Thread(target=pump, args=(proc.stdout, out_parts, sys.stdout)),
        threading.Thread(target=pump, args=(proc.stderr, err_parts, sys.stderr)),
    ]
    for thread in threads:
        thread.start()
    proc.wait()
    for thread in threads:
        thread.join()
    return "".join(out_parts), "".join(err_parts)


def open_conn(cluster) -> Conn:
    """SSH when the cluster names an alias; a local subprocess otherwise."""
    if not cluster.ssh:
        return Local()
    from orch.ssh import SSH

    return SSH(cluster.ssh, cluster.password_env)


__all__ = ["Conn", "Local", "Result", "open_conn"]
