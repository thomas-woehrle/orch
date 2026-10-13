"""job.sh rendering and flag/env rules."""

from types import SimpleNamespace

from orch.backends.sge import SGEBackend, classify_sge_state, parse_qstat_xml
from orch.config import Cluster
from orch.render import (
    build_env,
    render_flags,
    render_header,
    validate_resources,
)
from orch.errors import UsageError
import pytest

HEADER = """#$ -q gpu
#$ -l gpu={{ gpus }}
{% if gpu_type %}#$ -l {{ gpu_type }}=true{% endif %}
#$ -l h_rt={{ time }}
#$ -l h_rss={{ mem }}
"""

SETUP = """. /etc/profile.d/modules.sh
module load cuda
"""


def test_flags_quoting_and_omission():
    rendered = render_flags(
        {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "verbose": True,
            "skip": False,
            "clip": None,
            "tasks": ["hellaswag", "mmlu"],
        }
    )
    assert rendered == "--max_new_tokens '1024' --temperature '0.0' --verbose --tasks 'hellaswag,mmlu'"


def test_header_with_and_without_gpu_type():
    plain = render_header(HEADER, {"gpus": 1, "time": "12:00:00", "mem": "24G"})
    assert plain == "\n".join(
        [
            "#$ -q gpu",
            "#$ -l gpu=1",
            "",
            "#$ -l h_rt=12:00:00",
            "#$ -l h_rss=24G",
        ]
    )
    typed = render_header(
        HEADER, {"gpus": 1, "gpu_type": "h200", "time": "24:00:00", "mem": "24G"}
    )
    assert "#$ -l h200=true" in typed
    with pytest.raises(UsageError, match="canonical"):
        validate_resources({"gpus": 1, "time": "1", "mem": "1", "extra": 1}, HEADER)
    with pytest.raises(UsageError, match="without a guard"):
        validate_resources({"gpus": 1}, HEADER)
    with pytest.raises(UsageError, match="device"):
        validate_resources({"gpus": 1, "time": "1", "mem": "1", "device": 0}, HEADER)


def test_golden_job_sh():
    env = build_env(
        cluster_env={"UV_CACHE_DIR": "/cache/uv", "HF_HOME": "/cache/hf"},
        entry_env={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        entity="thomas-woehrle-team",
        project="orch-exp",
        job_kind="eval",
        run_id="3f9a2c1d7e4b-a1",
        run_name="tooluse@sdft",
        tags=["kind:eval", "attempt:1"],
        group=None,
        run_dir="/work/3f9a2c1d7e4b-a1",
    )
    cluster = Cluster(
        name="eddie",
        backend="sge",
        ssh="eddie",
        repo_dir="/repo",
        work_dir="/work",
        header=HEADER,
        setup=SETUP,
    )
    job = SimpleNamespace(
        run_id="3f9a2c1d7e4b-a1",
        fingerprint="3f9a2c1d7e4b",
        attempt=1,
        resources={"gpus": 1, "time": "01:00:00", "mem": "24G"},
    )
    script = SGEBackend().render_job_script(
        job, cluster, env, "/repo/.worktrees/abc", "/repo/.venv/bin/python", []
    )
    assert script == GOLDEN
    job.resources = {"gpus": 1, "gpu_type": "h200", "time": "01:00:00", "mem": "24G"}
    typed = SGEBackend().render_job_script(
        job, cluster, env, "/repo/.worktrees/abc", "/repo/.venv/bin/python", []
    )
    assert "#$ -l h200=true\n#$ -l h_rt=01:00:00" in typed


def test_env_precedence_puts_orch_last():
    env = build_env(
        cluster_env={
            "WANDB_PROJECT": "from-cluster",
            "HF_HOME": "/hf",
            "HF_METRICS_CACHE": "/hf/metrics",
        },
        entry_env={"WANDB_PROJECT": "from-entry", "HF_METRICS_CACHE": "{run_dir}/metrics"},
        entity="ent",
        project="proj",
        job_kind="train",
        run_id="fp-a1",
        run_name="name",
        tags=["kind:train"],
        group="g",
        run_dir="/work/fp-a1",
    )
    assert env["HF_HOME"] == "/hf"
    # The entry's placeholder wins over the cluster, and is filled per attempt.
    assert env["HF_METRICS_CACHE"] == "/work/fp-a1/metrics"
    assert env["WANDB_PROJECT"] == "proj"
    assert env["WANDB_RUN_GROUP"] == "g"
    assert env["ORCH_MODEL_DIR"] == "/work/fp-a1/model"
    assert "ORCH_OUTPUT_DIR" not in env
    assert list(env).index("HF_HOME") < list(env).index("WANDB_RUN_ID")


def test_env_placeholders_reject_unknown_and_the_wrong_kind():
    from orch.render import fill_placeholders

    assert fill_placeholders("a {run_dir} {params}", run_dir="/work/fp-a1", kind="train") == (
        "a /work/fp-a1 /work/fp-a1/params.yaml"
    )
    assert fill_placeholders("{model_dir}", run_dir="/work/fp-a1", kind="train") == "/work/fp-a1/model"
    with pytest.raises(UsageError, match="unknown env placeholder"):
        fill_placeholders("{flags}", run_dir="/work/fp-a1", kind="train")
    with pytest.raises(UsageError, match="model_dir"):
        fill_placeholders("{model_dir}", run_dir="/work/fp-a1", kind="eval")
    with pytest.raises(UsageError, match="output_dir"):
        fill_placeholders("{output_dir}", run_dir="/work/fp-a1", kind="train")


def test_qstat_parser():
    xml = """<?xml version="1.0"?>
<job_info>
  <queue_info>
    <job_list state="running">
      <JB_job_number>10</JB_job_number>
      <state>r</state>
    </job_list>
  </queue_info>
  <job_info>
    <job_list state="pending">
      <JB_job_number>11</JB_job_number>
      <state>qw</state>
    </job_list>
    <job_list state="error">
      <JB_job_number>12</JB_job_number>
      <state>Eqw</state>
    </job_list>
  </job_info>
</job_info>
"""
    assert parse_qstat_xml(xml) == {"10": "running", "11": "queued", "12": "error"}
    assert classify_sge_state("hqw") == "queued"


GOLDEN = """#!/bin/bash
#$ -q gpu
#$ -l gpu=1

#$ -l h_rt=01:00:00
#$ -l h_rss=24G
#$ -N orch_3f9a2c1d7e4b_a1
#$ -o /work/3f9a2c1d7e4b-a1/scheduler.out
#$ -e /work/3f9a2c1d7e4b-a1/scheduler.err
set -euo pipefail
. /etc/profile.d/modules.sh
module load cuda
export UV_CACHE_DIR='/cache/uv'
export HF_HOME='/cache/hf'
export VLLM_USE_FLASHINFER_SAMPLER='0'
export WANDB_RUN_ID='3f9a2c1d7e4b-a1'
export WANDB_PROJECT='orch-exp'
export WANDB_ENTITY='thomas-woehrle-team'
export WANDB_NAME='tooluse@sdft'
export WANDB_JOB_TYPE='eval'
export WANDB_TAGS='kind:eval,attempt:1'
export WANDB_RESUME='allow'
export WANDB_DIR='/work/3f9a2c1d7e4b-a1/wandb'
export ORCH_RUN_DIR='/work/3f9a2c1d7e4b-a1'
export ORCH_PARAMS='/work/3f9a2c1d7e4b-a1/params.yaml'
export ORCH_OUTPUT_DIR='/work/3f9a2c1d7e4b-a1/output'

cd /repo/.worktrees/abc
exec /repo/.venv/bin/python -m orch exec /work/3f9a2c1d7e4b-a1
"""
