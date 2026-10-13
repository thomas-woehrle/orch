"""SSH connection retry. The password attempt must not reuse a ProxyCommand pipe."""

import paramiko

from orch.ssh import SSH


class _Config:
    @staticmethod
    def from_path(path):
        return _Config()

    def lookup(self, alias):
        return {
            "hostname": "dmog.example",
            "port": "22",
            "user": "tew4000",
            "proxycommand": "ssh -W dmog.example:22 jump",
        }


class _Sock:
    def __init__(self, command):
        self.command = command


class _Transport:
    def set_keepalive(self, interval):
        self.interval = interval


class _Client:
    socks = []
    connects = []

    def load_system_host_keys(self):
        return None

    def set_missing_host_key_policy(self, policy):
        return None

    def connect(self, **kwargs):
        sock = kwargs.get("sock")
        _Client.socks.append(sock)
        _Client.connects.append(kwargs)
        if kwargs.get("password") is None:
            raise paramiko.AuthenticationException("Authentication failed.")

    def close(self):
        return None

    def get_transport(self):
        return _Transport()


def test_password_retry_opens_a_fresh_proxy(monkeypatch, tmp_path):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host dmog\n")
    monkeypatch.setattr("orch.ssh.Path.home", lambda: tmp_path)
    monkeypatch.setattr(paramiko, "SSHConfig", _Config)
    monkeypatch.setattr(paramiko, "ProxyCommand", _Sock)
    monkeypatch.setattr(paramiko, "SSHClient", _Client)
    monkeypatch.setenv("ORCH_DMOG_PASSWORD", "second-hop")
    _Client.socks = []
    _Client.connects = []

    SSH("dmog", password_env="ORCH_DMOG_PASSWORD").connect()

    assert len(_Client.socks) == 2
    assert _Client.socks[0] is not _Client.socks[1]
    assert _Client.connects[1]["password"] == "second-hop"
    assert _Client.connects[1]["look_for_keys"] is False
