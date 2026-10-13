"""Train and eval wrappers with a fake wandb. No cluster and no network."""

import json
import sys
from pathlib import Path

from orch.exec.common import read_job
from orch.exec.eval import execute as execute_eval
from orch.exec.eval import upstream_block
from orch.exec.train import execute as execute_train


def _write_job(run_dir: Path, **overrides) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "src").mkdir()
    document = {
        "schema_version": 1,
        "fingerprint": "aaaaaaaaaaaa",
        "fingerprint_version": 1,
        "attempt": 1,
        "run_id": "aaaaaaaaaaaa-a1",
        "run_name": "name",
        "kind": "train",
        "entry": "train",
        "spec_path": "configs/train/sft_tooluse.yaml",
        "spec": {
            "params": {"learning_rate": 1e-5},
            "runtime_keys": [],
            "resources": {},
            "description": "",
            "evals": [],
            "env": {},
        },
        "overrides": [],
        "model_ref": None,
        "upstream_run_id": None,
        "wandb": {"entity": "ent", "project": "proj", "tags": ["attempt:1"], "group": None},
        "cluster": {
            "name": "local",
            "backend": "local",
            "src_dir": str(run_dir / "src"),
            "work_dir": str(run_dir.parent),
            "repo_dir": "/repo",
        },
        "git": {"sha": "abc", "remote": "https://example.com/repo.git"},
        "command": "python train.py",
        "parser": "orch_adapters:parse_eval_results",
        "env": {"ORCH_RUN_DIR": str(run_dir)},
    }
    document.update(overrides)
    (run_dir / "job.json").write_text(json.dumps(document))


class _Config(dict):
    def update(self, payload, allow_val_change=False):
        super().update(payload)


def _install_wandb(monkeypatch, events, *, fail_first=False):
    state = {"n": 0, "run": None}

    class Artifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add_dir(self, path):
            events.append("add_dir")

    class Logged:
        def wait(self):
            events.append("artifact")

    class Run:
        def __init__(self):
            self.config = _Config()
            self.tags = []
            self.summary = {}
            self.id = "aaaaaaaaaaaa-a1"
            self.name = "name"

        def finish(self, exit_code=None):
            events.append(("finish", self.config.get("orch_state"), exit_code))

        def log_artifact(self, artifact, aliases=None):
            return Logged()

        def save(self, *args, **kwargs):
            return None

        def log(self, data):
            events.append(("log", data))

    class Api:
        def run(self, path):
            class Producer:
                id = "bbbbbbbbbbbb-a1"
                name = "sdft_tooluse lr=1e-05"
                state = "finished"
                config = {
                    "params": {"learning_rate": 1e-5, "method": "sdft"},
                    "orch": {
                        "fingerprint": "bbbbbbbbbbbb",
                        "entry": "train",
                        "spec_path": "configs/train/sdft_tooluse.yaml",
                        "run_id": "bbbbbbbbbbbb-a1",
                    },
                }

                def logged_artifacts(self):
                    class Model:
                        type = "model"

                    return [Model()]

            return Producer()

        def artifact(self, name):
            class ArtifactRef:
                def logged_by(self):
                    return Api().run("x")

            return ArtifactRef()

    class Wandb:
        def init(self, **kwargs):
            state["n"] += 1
            if fail_first and state["n"] == 1:
                raise Exception("run id was previously created and deleted")
            events.append(("init", kwargs["id"]))
            run = Run()
            state["run"] = run
            original = run.config.update

            def update(payload, allow_val_change=False):
                events.append(("config", payload.get("orch_state"), payload.get("upstream")))
                original(payload, allow_val_change=allow_val_change)

            run.config.update = update
            return run

        def Api(self):
            return Api()

        def teardown(self):
            events.append("teardown")

    wandb = Wandb()
    wandb.Artifact = Artifact
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return state


def test_schema_v1_job_defaults_the_scheduler_variable(tmp_path: Path):
    run_dir = tmp_path / "aaaaaaaaaaaa-a1"
    _write_job(run_dir)
    job = read_job(run_dir)
    assert job["cluster"]["job_id_env"] == "JOB_ID"


def test_train_prologue_precedes_the_script(tmp_path: Path, monkeypatch):
    events: list = []
    _install_wandb(monkeypatch, events)
    run_dir = tmp_path / "aaaaaaaaaaaa-a1"
    _write_job(run_dir)

    def script(command, *, cwd, env, run_dir):
        events.append("script")
        model = run_dir / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_text("w")
        return 0

    monkeypatch.setattr("orch.exec.train.tee_subprocess", script)
    assert execute_train(run_dir) == 0
    assert events[:5] == [
        ("init", "aaaaaaaaaaaa-a1"),
        ("config", "running", None),
        ("finish", "running", None),
        "teardown",
        "script",
    ]
    assert events[5][0] == "init"
    assert ("config", "succeeded", None) in events
    assert "artifact" in events
    assert events[-1] == ("finish", "succeeded", 0)


def test_artifact_false_succeeds_without_a_model(tmp_path: Path, monkeypatch):
    events: list = []
    state = _install_wandb(monkeypatch, events)
    run_dir = tmp_path / "aaaaaaaaaaaa-a1"
    _write_job(run_dir, artifact=False)

    monkeypatch.setattr("orch.exec.train.tee_subprocess", lambda *args, **kwargs: 0)
    assert execute_train(run_dir) == 0
    assert "artifact" not in events
    assert ("config", "succeeded", None) in events
    assert state["run"].config["orch"]["artifact"] is False
    assert events[-1] == ("finish", "succeeded", 0)


def test_missing_artifact_field_still_requires_a_model(tmp_path: Path, monkeypatch):
    events: list = []
    state = _install_wandb(monkeypatch, events)
    run_dir = tmp_path / "aaaaaaaaaaaa-a1"
    _write_job(run_dir)

    monkeypatch.setattr("orch.exec.train.tee_subprocess", lambda *args, **kwargs: 0)
    assert execute_train(run_dir) == 0
    assert "artifact" not in events
    assert ("config", "failed", None) in events
    assert state["run"].summary["orch/failure_reason"] == "model_dir_missing"
    assert state["run"].config["orch"]["artifact"] is True


def test_deleted_run_id_bumps_before_the_script(tmp_path: Path, monkeypatch):
    events: list = []
    seen: dict = {}
    _install_wandb(monkeypatch, events, fail_first=True)
    run_dir = tmp_path / "aaaaaaaaaaaa-a1"
    _write_job(run_dir)

    def script(command, *, cwd, env, run_dir):
        seen["run_id"] = json.loads((run_dir / "job.json").read_text())["run_id"]
        seen["dir"] = run_dir.name
        model = run_dir / "model"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text("{}")
        (model / "model.safetensors").write_text("w")
        return 0

    monkeypatch.setattr("orch.exec.train.tee_subprocess", script)
    assert execute_train(run_dir) == 0
    assert seen == {"run_id": "aaaaaaaaaaaa-a2", "dir": "aaaaaaaaaaaa-a2"}
    assert events[0] == ("init", "aaaaaaaaaaaa-a2")


def test_eval_copies_upstream_and_hf_is_null(tmp_path: Path, monkeypatch):
    events: list = []
    _install_wandb(monkeypatch, events)
    run_dir = tmp_path / "cccccccccccc-a1"
    _write_job(
        run_dir,
        kind="eval",
        entry="tooluse_eval",
        spec_path="configs/eval/tooluse.yaml",
        run_id="cccccccccccc-a1",
        fingerprint="cccccccccccc",
        model_ref="run:bbbbbbbbbbbb-a1",
        upstream_run_id="bbbbbbbbbbbb-a1",
        command="python eval.py {model_path}",
    )
    monkeypatch.setattr(
        "orch.exec.eval.resolve_model",
        lambda job, run, work: (tmp_path / "model", {"kind": "run", "id": "bbbbbbbbbbbb-a1"}),
    )
    monkeypatch.setattr("orch.exec.eval.tee_subprocess", lambda *args, **kwargs: 0)
    monkeypatch.setattr("orch.exec.eval._parser", lambda dotted: (lambda path: {"accuracy": 1.0}))
    assert execute_eval(run_dir) == 0
    upstream = next(item[2] for item in events if item[0] == "config" and item[2] is not None)
    assert upstream["params"]["learning_rate"] == 1e-5
    assert upstream["spec"] == "sdft_tooluse"
    assert upstream["fingerprint"] == "bbbbbbbbbbbb"
    assert upstream_block(None) is None
    from orch.exec.eval import _upstream_ready

    ready, _state, producer = _upstream_ready(
        {"model_ref": "hf:Qwen/Qwen2.5-1.5B-Instruct", "upstream_run_id": None}
    )
    assert ready and producer is None
