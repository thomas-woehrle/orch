"""paramiko sessions that honour ``~/.ssh/config``. See docs/orchestrator.md A1."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from orch.conn import Result
from orch.errors import RemoteError, UsageError
from orch.render import shquote


class SSH:
    def __init__(self, alias: str, password_env: str | None = None) -> None:
        self.alias = alias
        self.password_env = password_env
        self.client = None
        self._sftp = None

    def __enter__(self) -> SSH:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def connect(self) -> None:
        import paramiko

        config_path = Path.home() / ".ssh" / "config"
        hostname = self.alias
        port = 22
        username = None
        key_filename = None
        proxy_command = None
        if config_path.is_file():
            ssh_config = paramiko.SSHConfig.from_path(str(config_path))
            host = ssh_config.lookup(self.alias)
            hostname = host.get("hostname", self.alias)
            port = int(host.get("port", 22))
            username = host.get("user")
            identity = host.get("identityfile")
            if isinstance(identity, list) and identity:
                key_filename = identity[0]
            elif isinstance(identity, str):
                key_filename = identity
            proxy_command = host.get("proxycommand")

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        common = dict(
            hostname=hostname,
            port=port,
            username=username,
            timeout=30,
            key_filename=key_filename,
        )
        if proxy_command:
            common["sock"] = paramiko.ProxyCommand(proxy_command)
        try:
            client.connect(**common, allow_agent=True, look_for_keys=True)
        except Exception as key_error:
            password = os.environ.get(self.password_env) if self.password_env else None
            if not password:
                if not sys.stdin.isatty():
                    raise UsageError(
                        f"SSH key auth failed for {self.alias} and no password is available "
                        f"({self.password_env or 'password_env is unset'}): {key_error}"
                    ) from key_error
                import getpass

                password = getpass.getpass(f"Password for {username or ''}@{hostname}: ")
            try:
                client.connect(
                    **common,
                    password=password,
                    allow_agent=False,
                    look_for_keys=False,
                )
            except Exception as exc:
                raise UsageError(f"could not SSH to {self.alias}: {exc}") from exc
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        self.client = client

    def close(self) -> None:
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self.client is not None:
            self.client.close()
            self.client = None

    def run(self, command: str, *, check: bool = True, stream: bool = False) -> Result:
        if self.client is None:
            raise RemoteError("SSH connection is not open")
        wrapped = "bash -lc " + shquote(command)
        _stdin, stdout, stderr = self.client.exec_command(wrapped)
        channel = stdout.channel
        if stream:
            out_parts: list[str] = []
            err_parts: list[str] = []
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    data = channel.recv(4096).decode(errors="replace")
                    sys.stdout.write(data)
                    sys.stdout.flush()
                    out_parts.append(data)
                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(4096).decode(errors="replace")
                    sys.stderr.write(data)
                    sys.stderr.flush()
                    err_parts.append(data)
                if not channel.recv_ready() and not channel.recv_stderr_ready():
                    time.sleep(0.1)
            while channel.recv_ready():
                data = channel.recv(4096).decode(errors="replace")
                sys.stdout.write(data)
                out_parts.append(data)
            while channel.recv_stderr_ready():
                data = channel.recv_stderr(4096).decode(errors="replace")
                sys.stderr.write(data)
                err_parts.append(data)
            code = channel.recv_exit_status()
            out, err = "".join(out_parts), "".join(err_parts)
        else:
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            code = channel.recv_exit_status()
        if check and code != 0:
            raise RemoteError(f"remote command failed ({code}): {command}\n{err}".rstrip())
        return Result(code, out, err)

    def sftp(self):
        if self.client is None:
            raise RemoteError("SSH connection is not open")
        if self._sftp is None:
            self._sftp = self.client.open_sftp()
        return self._sftp

    def exists(self, path: str) -> bool:
        try:
            self.sftp().stat(path)
            return True
        except OSError:
            return False

    def put_text(self, path: str, content: str) -> None:
        self.run("mkdir -p " + shquote(str(Path(path).parent)))
        with self.sftp().file(path, "w") as handle:
            handle.write(content)
