# `orch` — experiment orchestrator, v0.2 design

Status: v0 design agreed 2026-10-02 and implemented; v0.2 design agreed
2026-10-12, implementation not started. Where v0.2 changes v0 behaviour the
text below describes v0.2 and the decision log (§14, entries 20 onwards) says
why.
Audience: whoever implements v0.2 (human or agent) and, afterwards, anyone who
needs to know how runs, identities, and W&B records relate.

Sections 2–8 are normative: they define vocabulary, invariants, file formats,
semantics, flows, the W&B schema, and the CLI. The implementation must match
them, and they are intended to stand as the reference documentation afterwards.
Sections 9–17 are informative: layout, codebase changes, a worked example, the
implementation sequence, testing, the decision log, risks, deferred items, and
documentation deliverables.

What v0.2 adds, in one paragraph: three scheduler backends (`sge` for eddie,
`slurm` for dmog, `local` for the 8-GPU workstation) behind one cluster
abstraction, with cluster-neutral resource keys and lazily fetched per-cluster
scheduler state; a `.orchignore` file that relaxes the clean-tree rule for
paths the cluster never needs; per-job `env` with run-directory placeholders;
a train wrapper that writes the orch config block before the script starts;
`orch status` that lists active jobs across clusters and derives its fields
from the run id and W&B before the cache; and a `config.upstream` block on
eval runs so they can be grouped by the properties of the training run that
produced their model. The fingerprint scheme is unchanged
(`fingerprint_version` stays 1): every identity computed by v0 is the same
under v0.2.

---

## 0. Purpose

Producing one results figure for this project currently takes: two evaluation
runs for the base model, one training run per method, two evaluation runs per
resulting checkpoint, manual download of checkpoints, manual upload of
evaluation results to W&B, and manual bookkeeping of which evaluation belongs
to which training run. Changing a single hyperparameter repeats all of it.

`orch` makes the whole chain a single command. You maintain the training and
evaluation scripts plus small configuration files; `orch` resolves
configuration, assigns each job a content-derived identity, stages code on a
cluster, submits training and dependent evaluation jobs to the scheduler, and
wraps every job so that all metrics, artifacts, configuration, logs, and
lineage land in Weights & Biases. W&B is the only durable state. Anything with
a W&B API key — a person, a notebook, an agent — can reconstruct what was run
and with what result, without access to any cluster filesystem.

---

## 1. Scope

### 1.1 In scope (v0, kept)

- Run specs (one job each) and experiment specs (lists of run specs).
- Training jobs that produce a model artifact; evaluation jobs that consume a
  model reference (`hf:`, `run:`, `fingerprint:`, `artifact:`).
- Content-derived identity (fingerprint), attempt counters, idempotent
  reconcile-style submission.
- Code staging by git SHA with a shared `uv` environment per cluster.
- A per-job wrapper that owns the W&B run for both training and evaluation.
- CLI: `run` (with `--dry-run`, `--force`, `--no-evals`, `--resync`), `eval`,
  `status`, `logs`, `tag`, `cancel`, `inspect`, `cluster check [--gpu]`, and
  the internal `exec`.
- The codebase changes needed to satisfy the contracts: a unified `train.py`,
  small adjustments to the evaluation scripts, result parsers, and example
  configuration.

### 1.2 Added in v0.2

- Three scheduler backends behind one `Backend` protocol and one `Conn`
  protocol: `sge` (eddie, over SSH), `slurm` (dmog, over SSH), `local` (the
  orchestrator machine itself, no SSH). A cluster entry in `orch.local.yaml`
  names its backend; everything downstream of resolution is backend-neutral.
- Cluster-neutral resource keys (`gpus`, `gpu_type`, `time`, `mem`, `device`)
  mapped by each cluster's header template.
- Reconcile that understands attempts spread over several clusters, fetching
  scheduler state lazily per cluster.
- `.orchignore`: paths that are allowed to be dirty locally because the
  cluster worktree excludes them.
- `env` on entries, specs, and experiment runs, with `{run_dir}`,
  `{output_dir}`, `{model_dir}` placeholders.
- A two-phase train wrapper so `config.orch` and the flat `orch_*` keys exist
  from the moment the job starts, with `orch_state: running` while it runs.
- `orch status` listing active jobs on every cluster by default, with fields
  derived from the run id and W&B before the local cache.
- `config.upstream` on eval runs: a copy of the producing training run's
  params and identity, for grouping in the W&B table.
- Scheduler-neutral names: `scheduler.out`/`scheduler.err`, a per-backend job
  id variable, `orch_cluster` as a flat key.

### 1.3 Out of scope

Listed so they are not accidentally built: `plan` as a separate command;
`report` and any figure generation; hyperparameter sweeps and
best-of-sweep promotion; a watch/daemon loop; `invalidate`; strict
`identity_keys`; automatic cluster selection; dataset artifacts; automatic
pinning of Hugging Face revisions; worktree garbage collection; an MCP or
HTTP API; a `dft` training method; cancelling downstream jobs when an attempt
fails; multi-GPU local jobs; local jobs from unpushed commits; running a
local-backend cluster from more than one orchestrator machine. See §16 for the
shape each of these is expected to take.

---

## 2. Vocabulary

| Term                    | Meaning                                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **spec** / **run spec** | A YAML file describing one job: its kind, entry, params, resources, downstream evals, naming. `orch run <spec>` is the primitive.                            |
| **experiment spec**     | A YAML file listing several run specs with overrides. Expands to a set of jobs.                                                                              |
| **params file**         | A YAML mapping of hyperparameters, owned by the script that consumes it. Opaque to `orch` except for two reserved keys.                                      |
| **entry**               | A named binding in `orch.yaml` from a spec kind to a command template, parser, and environment. Entries are the only place `orch` knows which scripts exist. |
| **kind**                | `train` or `eval`.                                                                                                                                           |
| **fingerprint**         | 12-hex-character content hash identifying an experiment. Shared by all attempts of the same experiment.                                                      |
| **attempt**             | Integer ≥ 1. One submission of a fingerprint.                                                                                                                |
| **run ID**              | W&B run ID, `{fingerprint}-a{attempt}`. Unique per attempt.                                                                                                  |
| **scheduler ID**        | The backend's handle for one attempt: a `qsub`/`sbatch` job number, or a tmux session name for `local`. Lives in the local cache and on the W&B run; never used as identity. |
| **job**                 | The fully resolved, submittable description of one attempt (`job.json`).                                                                                     |
| **cluster**             | A named entry in `orch.local.yaml`: a backend, an optional SSH target, paths, scheduler header, setup, env. The orchestrator machine itself is a cluster whose backend is `local`. |
| **backend**             | The scheduler driver a cluster uses: `sge`, `slurm`, or `local`. It renders the job script's directive block, submits, reports state, and cancels. One backend serves any number of clusters. |
| **connection**          | How orch reaches a cluster's login side: `SSH` (paramiko) or `Local` (subprocess). Both expose `run`, `put_text`, `exists`.                                  |
| **device**              | `local` only: one GPU index on the orchestrator machine, claimed by a job for its lifetime and exported as `CUDA_VISIBLE_DEVICES`.                            |
| **wrapper**             | `orch exec`, which runs on the compute node around the actual script and owns the W&B run.                                                                   |
| **model reference**     | A string naming a model: `hf:<repo>[@rev]`, `run:<run_id>`, `fingerprint:<fp>`, `artifact:<entity>/<project>/<name>:<version>`.                              |
| **upstream**            | For an eval job, the training run whose model it evaluates. Its identity-bearing params are copied to the eval run's `config.upstream`.                      |
| **downstream**          | For a training job, the eval jobs holding on it. Orch does not track downstream jobs in v0.2 (§16).                                                         |
| **hold**                | A scheduler dependency: the job does not start until the named upstream job has exited, for any reason. `-hold_jid` (SGE), `--dependency` (Slurm), a wait loop in `job.sh` (local). |
| **reconcile**           | Deciding, per job, whether to submit, skip, or resubmit, by comparing the desired set of jobs with W&B and scheduler state.                                  |
| **burned attempt**      | An attempt number that was minted (run directory created, scheduler id cached) but never produced a W&B run. Reconcile skips over it; numbering has a gap.   |

---

## 3. Invariants and contracts

Everything in this section is a hard requirement. The orchestrator relies on
each one; `orch cluster check` tests group A and the testable parts of B.

### A. Cluster (per entry in `orch.local.yaml`)

- **A1.** The cluster's login side is reachable from the orchestrator machine.
  With `ssh` set: over SSH using key-based or password authentication; the
  alias is resolved through `~/.ssh/config`; a password, if used, comes from
  the environment variable named by `password_env` or an interactive prompt,
  never from a file. Without `ssh` (backend `local`): the login side is the
  orchestrator machine itself and commands run in a local subprocess.
- **A2.** `git`, `uv`, and the backend's tools are on `PATH` in a _login_
  shell: `qsub`, `qstat`, `qdel` for `sge`; `sbatch`, `squeue`, `scancel` for
  `slurm`; `tmux`, `nvidia-smi`, `flock` for `local`. All login-side commands
  run as `bash -lc '<cmd>'`, on every backend.
- **A3.** `repo_dir` is a git clone whose `origin` is the same remote the
  orchestrator machine pushes to, and the login side can fetch from it. The
  orchestrator owns this clone: it creates per-SHA worktrees under
  `repo_dir/.worktrees/` and manages `repo_dir/.venv`. Nothing else should
  modify it. For a `local` cluster, `repo_dir` is therefore a separate clone,
  never the checkout `orch` is invoked from; `cluster check` refuses the two
  being the same path.
- **A4.** `work_dir` exists, is writable, and is on a filesystem visible to
  both login and compute sides. Nothing in it is assumed to persist;
  everything of value is pushed to W&B before a job exits.
- **A5.** The login side can reach PyPI and GitHub. Compute nodes can reach
  `api.wandb.ai` and `huggingface.co`. W&B credentials are available to compute
  nodes via `~/.netrc` (from `wandb login` on the login node, home being
  shared) or a `WANDB_API_KEY` in the cluster `env`. Hugging Face credentials
  likewise if gated models are used.
- **A6.** The cluster's `header` template, rendered with a spec's `resources`,
  produces a job the scheduler accepts. The `setup` block yields a shell in
  which the venv's `torch` sees a GPU. The orchestrator runs both verbatim. A
  `local` cluster has no `header`; its `devices` list and the resource rules in
  §4.3 take its place.
- **A7.** `env` in the cluster block is exported in every job. Quota-sensitive
  paths (`UV_CACHE_DIR`, `HF_HOME`, `WANDB_DIR` defaults) belong here. Values
  may use the placeholders of §4.1.
- **A8.** The backend exports a job id into the job environment: `JOB_ID`
  (SGE), `SLURM_JOB_ID` (Slurm), `ORCH_JOB_ID` (set by the local launcher).
  `job.json` names the variable (`cluster.job_id_env`); the wrapper records
  its value as `scheduler_id`.
- **A9.** `local` only: the tmux server on the orchestrator machine outlives
  the shell that submitted a job. Jobs are tmux sessions; `tmux kill-server`
  kills every local job.

### B. Repository

- **B1.** The repository is a `uv` project with a committed `uv.lock`.
  `uv sync --frozen` yields an environment in which every entry's command and
  `python -m orch` run.
- **B2.** `orch run` and `orch eval` refuse to submit unless the working tree
  is clean apart from paths matched by `.orchignore` (B6), and `HEAD` is
  contained in `origin`'s corresponding branch (or any `origin` ref). The SHA
  is recorded on every run. There is no other override. `--dry-run` does not
  check B2: it does not stage code or record a SHA, and the fingerprint does
  not include one.
- **B3.** Entry commands are relative to the repository root and run with the
  working directory set to the job's source tree (a worktree at the recorded
  SHA, minus `.orchignore` paths).
- **B4.** Data the scripts need is tracked in the repository or referenced by
  paths in params. `orch` does not manage data.
- **B5.** `orch.yaml` (entries) is committed. `orch.local.yaml` (W&B target,
  clusters) is gitignored; `orch.local.example.yaml` is committed.
- **B6.** `.orchignore` at the repository root, when present, is committed and
  clean, and lists paths the cluster worktree does not need, in gitignore
  syntax. Those paths are excluded from every cluster worktree by sparse
  checkout and are therefore allowed to be modified or untracked at submit
  time. It must not match any of: `pyproject.toml`, `uv.lock`, `README.md`
  (named by `pyproject.toml`), `orch.yaml`, `orch_adapters.py`, `.orchignore`,
  anything under `orch/`; `orch run` refuses such a file. Its SHA-256 is
  recorded in `job.json` and `config.orch.git`.

### C. Configuration and identity

- **C1.** A params file is a YAML 1.2 mapping. `extends` and `runtime_keys`
  are the only reserved keys; both are removed before the script sees the file.
- **C2.** A spec's top-level keys are exactly: `kind`, `entry`, `extends`,
  `params`, `set`, `runtime_keys`, `resources`, `evals`, `name_keys`,
  `description`, `env`. Unknown keys are an error.
- **C3.** The script receives a fully resolved params file: `extends` chains
  applied, `set` and CLI overrides applied, reserved keys removed. It never
  reads the original.
- **C4.** A key is identity-bearing unless listed in `runtime_keys`. The
  fingerprint is `hash(kind, entry, params − runtime_keys, model_ref)`. Spec
  path, the `runtime_keys` list itself, `resources`, `evals`, `name_keys`,
  `description`, `env`, cluster, git SHA, and W&B target are excluded.
  Nothing in v0.2 changes the fingerprint of any v0 identity.
- **C5.** The fingerprint input is written to `fingerprint.json` per attempt
  and logged to W&B; `fingerprint_version` is logged alongside it.
- **C6.** All YAML, including CLI `--set` values, is parsed with one YAML 1.2
  loader so that `5e-5` is a float everywhere.

### D. Training script

- **D1.** Non-interactive. Configured by `--config <params.yaml>` (or by flags
  if its entry renders `{flags}`).
- **D2.** Does not set W&B run identity: no `id`, `project`, `entity`, or
  `name` in any `wandb.init`, and `TrainingArguments.run_name` left unset.
  `report_to="wandb"` is expected; the run it attaches to is the one the
  wrapper pre-assigned through the environment.
- **D3.** Writes the final model and tokenizer in `save_pretrained` layout to
  `$ORCH_MODEL_DIR`. Intermediate checkpoints go under `$ORCH_RUN_DIR` and are
  not uploaded.
- **D4.** Exit code 0 if and only if training succeeded and D3 holds.

### E. Evaluation script

- **E1.** Non-interactive. Takes a local model directory and an output
  directory, via `$ORCH_MODEL_PATH` / `$ORCH_OUTPUT_DIR` or the `{model_path}`
  / `{output_dir}` placeholders in its entry command. Params via `--config` or
  `{flags}`.
- **E2.** Writes results under the output directory. The entry's `parser`
  turns that into a flat `{metric_name: float}` mapping. The script does not
  import `wandb`.
- **E3.** Exit code 0 if and only if evaluation succeeded.

### F. W&B

- **F1.** Every attempt whose wrapper starts ends up as a W&B run with the
  pre-minted run ID, including attempts whose script failed before initialising
  W&B. Two exceptions: an eval whose upstream did not succeed exits without
  creating a run (§6.6), and a job that dies before the wrapper starts (a
  `job.sh` setup failure) is visible only through the scheduler and
  `scheduler.err` (§6.7). In both cases the next reconcile treats the attempt
  as burned.
- **F2.** A successful training run has exactly one logged artifact of type
  `model`, named `model-{fingerprint}`, with aliases `a{attempt}` and `latest`.
- **F3.** Every run carries the configuration described in §7, including the
  flat `orch_*` keys used for filtering, from the moment the wrapper starts,
  not only after the script exits. While the script runs, `orch_state` is
  `running`.
- **F4.** An eval run whose model came from a training run carries
  `config.upstream`, a copy of that run's `params` and identity (§7.1), so the
  W&B table can group evals by properties of the run that produced the model.

---

## 4. File formats

All paths inside spec and params files follow two rules: `extends` is
relative to the file that contains it; `params` and `evals` are relative to
the repository root. `orch` is always invoked from the repository root.

### 4.1 `orch.yaml` (committed; codebase-specific layer)

```yaml
version: 1

entries:
  train:
    kind: train
    cmd: "python train.py --config {params}"
  tooluse_eval:
    kind: eval
    cmd: "python eval_tooluse.py --model_path {model_path} --output_dir {output_dir} {flags}"
    parser: "orch_adapters:parse_eval_results"
    env: { VLLM_USE_FLASHINFER_SAMPLER: "0" }
  science_eval:
    kind: eval
    cmd: "python eval_science.py --model_path {model_path} --output_dir {output_dir} {flags}"
    parser: "orch_adapters:parse_eval_results" # both scripts write the same eval_results.json
    env: { VLLM_USE_FLASHINFER_SAMPLER: "0" }
  lm_eval:
    kind: eval
    cmd: "lm_eval --model hf --model_args pretrained={model_path} --device cuda:0 --output_path {output_dir} --confirm_run_unsafe_code {flags}"
    parser: "orch_adapters:parse_lmeval"
    env:
      HF_ALLOW_CODE_EVAL: "1"
      # HumanEval's code_eval metric locks an arrow file under this cache;
      # concurrent jobs sharing HF_HOME delete it out from under each other.
      HF_METRICS_CACHE: "{run_dir}/metrics"
```

Placeholders available in `cmd` and in every `env` value (cluster, entry,
spec, experiment):

| Placeholder    | Value                                            | In `env`? |
| -------------- | ------------------------------------------------ | --------- |
| `{params}`     | Absolute path of the resolved `params.yaml`      | yes       |
| `{flags}`      | Params rendered as CLI flags (§5.3)              | no        |
| `{run_dir}`    | Absolute path of the attempt's run directory     | yes       |
| `{model_dir}`  | Train only: `$ORCH_MODEL_DIR`                    | yes       |
| `{model_path}` | Eval only: local directory of the resolved model | no        |
| `{output_dir}` | Eval only: `$ORCH_OUTPUT_DIR`                    | yes       |

`parser` is `module:function`, importable from the source tree, with signature
`(output_dir: Path) -> dict[str, float]`. Env precedence is in §6.4. Isolation
between concurrent jobs is the entry's responsibility, expressed through
`env`: orch gives every attempt a private run directory and the placeholders
to point script-specific caches at it; it does not know which caches a given
script shares.

### 4.2 `orch.local.yaml` (gitignored; operator-specific)

```yaml
version: 1

wandb:
  entity: thomas-woehrle-team
  project: orch-exp

default_cluster: eddie

clusters:
  eddie:
    backend: sge
    ssh: eddie # alias in ~/.ssh/config
    password_env: ORCH_EDDIE_PASSWORD # optional; prompt if unset and keys fail
    repo_dir: /exports/eddie/scratch/s2865446/projects/sdft-exp
    work_dir: /exports/eddie/scratch/s2865446/orch
    env:
      UV_CACHE_DIR: /exports/eddie/scratch/s2865446/.cache/uv
      HF_HOME: /exports/eddie/scratch/s2865446/.cache/hf
    header: |
      #$ -q gpu
      #$ -l gpu={{ gpus }}
      {% if gpu_type %}#$ -l {{ gpu_type }}=true{% endif %}
      #$ -l h_rt={{ time }}
      #$ -l h_rss={{ mem }}
    setup: |
      . /etc/profile.d/modules.sh
      module load cuda

  dmog:
    backend: slurm
    ssh: dmog
    repo_dir: /mnt/scratch/<user>/sdft-exp
    work_dir: /mnt/scratch/<user>/orch
    env:
      UV_CACHE_DIR: /mnt/scratch/<user>/.cache/uv
      HF_HOME: /mnt/scratch/<user>/.cache/hf
    # All dmog GPU nodes are A40s; gpu_type is deliberately not referenced, so
    # a spec that pins one fails validation here instead of being ignored.
    header: |
      #SBATCH -p gpu
      #SBATCH --gres=gpu:{{ gpus }}
      #SBATCH --time={{ time }}
      #SBATCH --mem={{ mem }}
    setup: |
      source "${flight_ROOT:-/opt/flight}"/etc/setup.sh

  local:
    backend: local
    # No ssh: commands run on this machine. repo_dir is a separate clone (A3).
    repo_dir: /localdisk/home/s2865446/orch/sdft-exp
    work_dir: /localdisk/home/s2865446/orch/work
    # GPU indices jobs may claim. One job claims one device.
    devices: [0, 1, 2, 3, 4, 5, 6, 7]
    env:
      HF_HOME: /localdisk/home/s2865446/.cache/hf
    setup: ""
```

Common keys: `backend` (`sge` | `slurm` | `local`), `repo_dir`, `work_dir`,
`env`, `setup`. `ssh` and `password_env` are required for `sge` and `slurm`
and forbidden for `local`. `header` is a Jinja2 template rendered with the
spec's `resources`; required for `sge` and `slurm`, forbidden for `local`.
`devices` is a non-empty list of integers, required for `local` and forbidden
otherwise. `WANDB_ENTITY` and `WANDB_PROJECT` in the orchestrator's
environment override the `wandb` block (and select the local cache file; see
§6.8). The API key is never in this file.

### 4.3 Run spec

```yaml
# configs/train/sdft_tooluse.yaml
kind: train
entry: train
description: SDFT on the tool-use task, paper defaults.
params: hparams/sdft_tooluse.yaml
resources: { gpus: 1, time: "12:00:00", mem: 24G }
evals:
  - configs/eval/tooluse.yaml
  - configs/eval/prior_tasks.yaml
name_keys:
  {
    model: params.model_name,
    lr: params.learning_rate,
    bs: params.gradient_accumulation_steps,
  }
```

```yaml
# configs/train/sdft_tooluse_7b.yaml  — a derived spec, no new params file
extends: sdft_tooluse.yaml
set:
  params.model_name: Qwen/Qwen2.5-7B-Instruct
  resources.gpu_type: h200
  resources.time: "24:00:00"
```

```yaml
# configs/eval/tooluse.yaml
kind: eval
entry: tooluse_eval
runtime_keys: [gpu_memory_utilization]
params: { max_new_tokens: 1024, temperature: 0.0, gpu_memory_utilization: 0.8 }
resources: { gpus: 1, time: "01:00:00", mem: 24G }
```

```yaml
# configs/eval/prior_tasks.yaml
kind: eval
entry: lm_eval
runtime_keys: [batch_size]
params:
  tasks: hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval
  batch_size: auto
  apply_chat_template: false
resources: { gpus: 1, time: "06:00:00", mem: 32G }
```

Field reference:

| Key            | Required               | Meaning                                                                       |
| -------------- | ---------------------- | ----------------------------------------------------------------------------- |
| `kind`         | yes (unless inherited) | `train` or `eval`; must match the entry's `kind`.                             |
| `entry`        | yes (unless inherited) | Name in `orch.yaml`.                                                          |
| `extends`      | no                     | Parent spec, relative to this file.                                           |
| `params`       | yes (unless inherited) | Path to a params file (relative to repo root) or an inline mapping.           |
| `set`          | no                     | Overrides applied after `extends` resolution, same grammar as CLI `--set`.    |
| `runtime_keys` | no                     | Additional non-identity keys, unioned with those declared in the params file. |
| `resources`    | yes (unless inherited) | Mapping of the canonical resource keys below.                                 |
| `evals`        | no; train only         | Eval specs to submit against this job's model. Default on.                    |
| `name_keys`    | no                     | `{alias: absolute.path}` rendered into the run name (§5.6).                   |
| `description`  | no                     | Free text, logged to `config.orch.description`.                               |
| `env`          | no                     | `{NAME: value}` exported into the job, over cluster and entry `env` (§6.4). Placeholders of §4.1 allowed. Not identity-bearing. |

Resource keys are cluster-neutral; each cluster's header template maps them
to its scheduler's syntax. Specs may set only these:

| Key        | Type           | Meaning                                                                                                      |
| ---------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| `gpus`     | int            | GPUs for the job. `local` requires `1`.                                                                      |
| `gpu_type` | string         | A GPU class the cluster header knows (`a100`, `h200` on eddie). `local` ignores it.                          |
| `time`     | `HH:MM:SS`     | Wall-clock limit. SGE `h_rt`, Slurm `--time`. `local` ignores it.                                            |
| `mem`      | e.g. `24G`     | Memory request. SGE `h_rss`, Slurm `--mem`. `local` ignores it.                                              |
| `device`   | int            | `local` only: pin the job to this GPU index instead of claiming a free one. Refused by `sge` and `slurm`.    |

Validation for `sge` and `slurm` is unchanged from v0: every key the spec sets
must be referenced by the cluster's header template, and every unguarded
template variable must have a value. For `local` there is no template; the
rule is that `gpus == 1`, `device`, if set, is in the cluster's `devices`, and
no other key than the five above is present. Because `resources` is not part
of the fingerprint, renaming `h_rt` to `time` changes no identity.

### 4.4 Params file

```yaml
# hparams/sdft_base.yaml
runtime_keys:
  - steps_per_generation
  - vllm_mode
  - vllm_gpu_memory_utilization
  - vllm_enable_sleep_mode
  - use_liger_kernel
  - per_device_eval_batch_size
  - logging_steps

method: sdft
model_name: Qwen/Qwen2.5-1.5B-Instruct
dtype: bfloat16
train_data: data/tooluse_data/train_data
val_size: 0.05
seed: 42
bf16: true
learning_rate: 5.0e-5
warmup_steps: 0.1
lr_scheduler_type: cosine
weight_decay: 0.0
max_grad_norm: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
num_train_epochs: 2
max_prompt_length: 1024
max_completion_length: 1024
num_generations: 1
temperature: 1.0
generate_from_teacher: false
teacher_model_kind: ema
teacher_sync_steps: 1
teacher_update_rate: 0.01
distillation_mode: full_logits
distillation_alpha: 0.0
distillation_is_clip: null
num_loss_tokens_to_skip: 3
disable_dropout: false
shuffle_dataset: true
use_vllm: false
teacher_prompt_template: |

  {prompt}

  This is an example for a response to the question:
  {privileged_context}

  Now answer with a response of your own, including the thinking process.

# runtime (declared above; values are defaults for a 24 GB GPU)
steps_per_generation: 32
vllm_mode: colocate
vllm_gpu_memory_utilization: 0.3
vllm_enable_sleep_mode: true
use_liger_kernel: true
per_device_eval_batch_size: 1
logging_steps: 1
```

```yaml
# hparams/sdft_tooluse.yaml
extends: sdft_base.yaml
# nothing to override for the paper configuration; exists so the spec has a stable name
```

```yaml
# hparams/sft_tooluse.yaml
method: sft
model_name: Qwen/Qwen2.5-1.5B-Instruct
dtype: bfloat16
train_data: data/tooluse_data/train_data
val_size: 0.05
seed: 42
bf16: true
learning_rate: 5.0e-5
warmup_steps: 0.1
lr_scheduler_type: cosine
weight_decay: 0.0
max_grad_norm: 1.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 32
num_train_epochs: 2
max_length: 2048
runtime_keys: [use_liger_kernel, per_device_eval_batch_size, logging_steps]
use_liger_kernel: true
per_device_eval_batch_size: 1
logging_steps: 1
```

Rules:

- `runtime_keys` names top-level keys only. If a nested value is technical,
  split it into its own top-level key.
- A key listed in `runtime_keys` need not be present; it may be added later by
  `-p key=value`.
- `null` is a value, not a deletion. There is no way to remove an inherited
  key; set the code default explicitly instead (§16, `unset`).
- Keys the script does not recognise must make the script fail (D1/E1 imply a
  strict loader). This is how typos in params files surface.

### 4.5 Experiment spec

```yaml
# experiments/tooluse_main.yaml
version: 1
description: Data for the "new task vs prior tasks" figure.
runs:
  - {
      config: configs/eval/tooluse.yaml,
      model: "hf:Qwen/Qwen2.5-1.5B-Instruct",
    }
  - {
      config: configs/eval/prior_tasks.yaml,
      model: "hf:Qwen/Qwen2.5-1.5B-Instruct",
    }
  - { config: configs/train/sdft_tooluse.yaml }
  - { config: configs/train/sft_tooluse.yaml }
  - {
      config: configs/train/sft_tooluse.yaml,
      set: { params.learning_rate: 1.0e-5 },
    }
tags: ["exp:tooluse-main"]
```

Each `runs[]` entry: `config` (required), `set` (optional, same grammar as
CLI), `model` (required for eval configs, forbidden for train configs),
`evals` (optional override of the train spec's list), `tags` (optional,
appended), `env` (optional, merged over the spec's `env`). Spec-level `tags`
apply to every job. Two entries that resolve to the same fingerprint collapse
to one job whose `evals` are the union; their `env` mappings must be equal or
the experiment is rejected (an `env` difference is not an identity difference,
so there is nothing to disambiguate by).

### 4.6 Per-attempt files (`work_dir/<run_id>/`)

| File                               | Written by   | Purpose                                                                                   |
| ---------------------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| `job.json`                         | orchestrator | The job: resolved spec, identity, W&B target, command, env, cluster, git. Schema in §6.1. |
| `params.yaml`                      | orchestrator | What the script receives (C3).                                                            |
| `fingerprint.json`                 | orchestrator | Exact canonical fingerprint input (C5).                                                   |
| `job.sh`                           | orchestrator | Scheduler script (§6.4).                                                                  |
| `scheduler.out`, `scheduler.err`   | scheduler    | Job-level stdout/stderr. (v0 named these `sge.out`/`sge.err`; `orch logs` accepts both.) |
| `state`                            | local launcher | `local` only: `held`, `running`, or `exit:<code>` (§6.4).                               |
| `stdout.log`, `stderr.log`         | wrapper      | The script's stdout/stderr, tee'd.                                                        |
| `model/`                           | train script | `$ORCH_MODEL_DIR`.                                                                        |
| `checkpoints/`                     | train script | Intermediate checkpoints; not uploaded.                                                   |
| `output/`                          | eval script  | `$ORCH_OUTPUT_DIR`.                                                                       |
| `wandb/`                           | wandb        | `WANDB_DIR`; keeps W&B's local files off the home quota.                                  |

Shared across attempts: `work_dir/cache/models/` for downloaded model
artifacts, keyed by artifact digest, with a `.complete` marker and a lock file
for concurrent downloads. Hugging Face downloads use `HF_HOME`. On a `local`
cluster, `work_dir/.devices/` holds one claim file per GPU index, named by the
index and containing the claiming tmux session name, plus `.devices/lock` for
`flock` (§6.4).

---

## 5. Resolution, identity, naming

### 5.1 Resolution pipeline

Input: a spec path, an optional model reference, a list of override strings,
`orch.yaml`, `orch.local.yaml`.

1. **Load spec.** YAML 1.2. Validate keys against C2.
2. **Resolve spec** `extends`**.** Recursively resolve the parent first, then
   deep-merge the child over it (§5.2). Remove `extends`. Cycles are an error.
3. **Resolve params.** If `params` is a string, load the file relative to the
   repository root; follow its `extends` chain (relative to each file);
   deep-merge; collect `runtime_keys` as the union over the chain; remove both
   reserved keys. Store the mapping as `spec.params` and the union as the
   params-declared runtime keys. If `params` is a mapping, use it directly.
4. **Union runtime keys.** `spec.runtime_keys` ∪ params-declared keys. Store
   as `spec.runtime_keys` (sorted list).
5. **Apply overrides** in order: spec `set` entries (in file order), then
   experiment `set` entries, then CLI `--set`/`-p`/`-r` in command-line
   order. Later wins. See §5.3 for grammar.
6. **Validate.** `entry` exists and its `kind` matches `spec.kind`; `params`
   is a mapping; every `name_keys` path resolves to a scalar; for a train spec
   each `evals` path exists and resolves to a spec of `kind: eval`; for an
   eval spec a model reference was supplied; for a train spec none was;
   `env` is a mapping of string to scalar; `resources` passes the target
   cluster's rules (§4.3).
7. **Resolve the model reference** (eval only) to its canonical string (§5.4).
8. **Compute the fingerprint** (§5.5) and the run name (§5.6).
9. **Reconcile** (§6.2) to determine the attempt number and the run ID.

`orch inspect <spec>` runs steps 1–8 and prints the resolved spec, the
fingerprint input, the fingerprint, and the name, without touching W&B or any
cluster (it uses attempt `?` in the displayed run ID).

### 5.2 Merge rules

`deep_merge(base, override)`: for each key in `override`, if both values are
mappings, recurse; otherwise the override value replaces the base value. Lists
replace. `null` replaces. The result preserves base key order with new keys
appended; order is irrelevant for identity.

### 5.3 Overrides

Grammar: `<path>=<value>`, where `<path>` is a dotted path rooted at the spec
(`params.learning_rate`, `resources.gpu_type`, `evals`), and `<value>` is
parsed as a YAML 1.2 scalar or flow collection (`5e-5` → float, `true` →
bool, `[a, b]` → list, `Qwen/Qwen2.5-7B-Instruct` → string). Intermediate
mappings are created as needed. Paths may only target `params`, `resources`,
`evals`, `runtime_keys`, `name_keys`, `env`; targeting `kind`, `entry`, or
`extends` is an error.

Shorthands:

| Flag           | Expands to            |
| -------------- | --------------------- |
| `-p k=v`       | `--set params.k=v`    |
| `-r k=v`       | `--set resources.k=v` |
| `--set path=v` | as written            |

There is no distinction at the CLI between identity-bearing and runtime
params; `-p` sets either. Whether an override affects identity follows solely
from `runtime_keys`. Instead of refusing, `run` and `inspect` annotate every
params override in their output with `[identity]` or `[runtime]` (and
`--dry-run` shows the resulting decision), so a mismatch between intent and
classification is visible before anything is submitted.

Overrides targeting `runtime_keys` are applied first, regardless of their
position, and the union from step 4 is recomputed before the remaining
overrides are applied and annotated. `--set runtime_keys=[...]` replaces the
_spec-level_ list only; keys declared in the params file always remain in the
union.

`{flags}` rendering (for entries that take CLI flags rather than `--config`):
for each `(k, v)` in the resolved params, in key order: `true` → `--k`;
`false` or `null` → omitted; list → `--k a,b,c`; otherwise `--k <v>` with
shell quoting.

### 5.4 Model references

| Form                                                    | Canonical form used in the fingerprint                                                    | Resolution on the compute node                                                              |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `hf:<repo>`                                             | `hf:<repo>`                                                                               | `snapshot_download(repo)`; resolved commit SHA recorded as metadata only                    |
| `hf:<repo>@<rev>`                                       | `hf:<repo>@<rev>` as written                                                              | `snapshot_download(repo, revision=rev)`                                                     |
| `run:<run_id>`                                          | `run:<run_id>`                                                                            | The `model` artifact logged by that run (`run.logged_artifacts()`), downloaded to the cache |
| `fingerprint:<fp>`                                      | `run:<fp>-a<n>` where `n` is the highest _finished_ attempt at submit time; error if none | as `run:`                                                                                   |
| `artifact:<entity>/<project>/<name>:<version-or-alias>` | `artifact:…:v<N>` with the alias resolved to a concrete version at submit time            | `use_artifact(...).download(root=cache)`                                                    |

An eval job planned together with its upstream train job in the same `orch run` uses `run:<upstream run_id>` where the upstream run ID is the one being
submitted (or the current attempt if the train job is skipped as finished or
in flight). `run:` and `fingerprint:` references, however they were supplied,
set the job's `upstream_run_id`, which enables the wrapper's upstream check
(§6.6 step 1); `hf:` and `artifact:` references leave it `null`.

The handle used in eval run names: the upstream run's name for `run:`/
`fingerprint:`; `hf:<last path segment>` for `hf:`; `artifact:<name>:<vN>` for
`artifact:`.

### 5.5 Fingerprint

```
input = {
  "kind":      spec.kind,
  "entry":     spec.entry,
  "params":    {k: v for k, v in spec.params.items() if k not in spec.runtime_keys},
  "model_ref": canonical model reference string, or null for train
}
canonical = json.dumps(input, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
fingerprint = sha256(canonical.encode("utf-8")).hexdigest()[:12]
fingerprint_version = 1
```

Properties the implementation must preserve:

- Key order never matters; list order does; an absent key and an explicit
  `null` differ; `2` and `2.0` differ (YAML int vs float).
- Changing anything in `runtime_keys`, `resources`, `evals`, `name_keys`,
  `description`, the spec path, or the `extends` chain structure does not
  change the fingerprint. Two spec files that resolve to the same input are the
  same experiment.
- Renaming an entry in `orch.yaml` changes every fingerprint that used it.
  Entry names are stable identifiers.
- `fingerprint.json` contains `{"fingerprint_version": 1, "input": <input>, "canonical": "<canonical string>", "fingerprint": "<hex>"}`.

### 5.6 Run names

```
<stem> [<alias>=<value> ...] [+<key>=<value> ...] [(a<n>)]
```

- `stem`: the spec file name without extension, e.g. `sdft_tooluse`. For an
  experiment entry, the stem of the referenced spec.
- `name_keys` segments, in declaration order, rendered as `alias=value`
  regardless of whether the value came from a file or an override.
- Remaining **params** overrides (from `set` or CLI) not already covered by a
  `name_keys` path, as `+key=value`, in application order. Overrides to other
  sections never appear in the name.
- `(a<n>)` appended when `n > 1`.
- Values are shortened: strings containing `/` keep only the last segment;
  values longer than 32 characters are truncated with `…`; whitespace becomes
  `_`.
- Eval runs: `<eval stem>@<model handle>`, then the same `name_keys` and
  override segments, then `(a<n>)`. Example: `tooluse@sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32`.

Names are for humans. Uniqueness is the run ID's job.

### 5.7 Validation errors (non-exhaustive, all fatal before any remote action)

Unknown spec key; `kind`/entry mismatch; `extends` cycle or missing file;
params not a mapping; `name_keys` path unresolved or non-scalar; override
path targeting `kind`, `entry`, or `extends`; eval spec without a model reference; train spec with a
model reference; resource key not referenced by the target cluster's header
template; header template variable without a value and without a guard;
resource key outside the canonical set; `device` on a non-local cluster or
not in the cluster's `devices`; `gpus != 1` on a local cluster; `env` value
that is not a scalar or uses an unknown placeholder; two deduplicated
experiment entries with different `env`; `.orchignore` matching a protected
path (B6); `fingerprint:` reference with no finished attempt; `artifact:`
alias that does not resolve.

---

## 6. Runtime flows

### 6.1 `orch run`

```
orch run <spec | experiment> [--cluster C] [--model REF] [-p/-r/--set ...]
         [--evals a.yaml,b.yaml | --no-evals] [--tag T]... [--group G]
         [--force] [--resync] [--dry-run] [--json]
```

1. **Preconditions.** Load `orch.yaml` and `orch.local.yaml`; select the
   cluster (`--cluster` or `default_cluster`); open its connection (SSH or
   local). A real submit checks B2 before staging: every path in
   `git status --porcelain --untracked-files=all` matches `.orchignore`
   (B6), `.orchignore` itself is clean, and `git branch -r --contains HEAD` is
   non-empty. Record `sha`, `remote`, and the `.orchignore` hash. `--dry-run`
   skips this check and opens no connection.
2. **Expand.** For a run spec: one job, plus one eval job per entry in `evals`
   (unless `--no-evals`; `--evals` replaces the list). For an experiment spec:
   each entry as above; collapse identical fingerprints and union their
   `evals`. Resolve each per §5.1 steps 1–8, validating `resources` against
   the target cluster.
3. **Reconcile** (§6.2) each train job, then each eval job (which needs its
   upstream's run ID). Scheduler state is fetched lazily, per cluster, the
   first time reconcile meets a cache entry from that cluster. Produce, per
   job: `submit` (with attempt number) or `skip` (with reason: finished,
   running, pending, upstream-not-available).
4. **Dry run stops here** and prints the table: run ID, name, kind, decision,
   reason, fingerprint, and for each `submit` the resource line and the
   dependency. Exit 0.
5. **Stage code** (§6.3). Skipped if nothing is to be submitted.
6. **Render and write** per job to `work_dir/<run_id>/`: `job.json`,
   `params.yaml`, `fingerprint.json`, `job.sh`. Holds are computed before
   rendering because the `local` backend writes them into `job.sh`.
7. **Submit** train jobs first, then eval jobs with a hold on the upstream
   when the upstream was submitted in this invocation or is in the cache as
   queued/running on the same cluster. Record `scheduler_id` and the cluster
   in the local cache.
8. **Report.** Print the table with scheduler IDs and W&B URLs
   (`https://wandb.ai/<entity>/<project>/runs/<run_id>`). `--json` emits the
   same as a list of objects.

A hold is only possible within one cluster. An eval whose upstream is in
flight on another cluster is skipped with `upstream-not-available`, like any
eval whose upstream is neither finished nor being submitted here; the next
`orch run` after the upstream finishes submits it.

`job.json` schema (version 2; v0 wrote version 1, which differs by the keys
marked below):

```json
{
  "schema_version": 2,
  "orch_version": "0.2.0",
  "fingerprint": "3f9a2c1d7e4b",
  "fingerprint_version": 1,
  "attempt": 1,
  "run_id": "3f9a2c1d7e4b-a1",
  "run_name": "sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32",
  "kind": "train",
  "entry": "train",
  "spec_path": "configs/train/sdft_tooluse.yaml",
  "spec": {
    "kind": "...",
    "entry": "...",
    "params": {},
    "runtime_keys": [],
    "resources": {},
    "evals": [],
    "name_keys": {},
    "description": "",
    "env": {}
  },
  "overrides": [{"path": "params.learning_rate", "value": 1e-5, "classification": "identity"}],
  "model_ref": null,
  "upstream_run_id": null,
  "wandb": {
    "entity": "...",
    "project": "orch-exp",
    "job_type": "train",
    "tags": [
      "kind:train",
      "spec:sdft_tooluse",
      "entry:train",
      "cluster:eddie",
      "attempt:1"
    ],
    "group": null
  },
  "cluster": {
    "name": "eddie",
    "backend": "sge",
    "repo_dir": "...",
    "work_dir": "...",
    "src_dir": ".../.worktrees/<sha>",
    "venv_python": ".../.venv/bin/python",
    "job_id_env": "JOB_ID",
    "device": null
  },
  "git": {
    "sha": "...",
    "remote": "https://github.com/thomas-woehrle/sdft-exp.git",
    "orchignore_sha256": null
  },
  "command": "python train.py --config /exports/.../3f9a2c1d7e4b-a1/params.yaml",
  "parser": null,
  "env": {
    "...": "merged cluster, entry, spec, experiment env with placeholders filled, then ORCH_* and WANDB_* variables"
  },
  "hold_on": [],
  "_note": "for eval jobs, command still contains the literal {model_path} placeholder; the wrapper substitutes it after resolving the model (§6.6 step 3)",
  "submitted_from": "w8724.see.ed.ac.uk",
  "submitted_at": "2026-10-02T13:00:00Z"
}
```

New in version 2: `spec.env`, `cluster.job_id_env` (A8), `cluster.device`
(the pinned index for `local`, else `null`; the claimed index is not known at
submit time), `git.orchignore_sha256` (`null` when no `.orchignore` exists),
`hold_on` (scheduler ids this job waited on). The wrapper reads `schema_version`
and accepts 1 and 2; a version-1 document has `job_id_env` defaulting to
`JOB_ID`.

### 6.2 Reconcile

For a job with fingerprint `F`, determine the current attempt by probing W&B
for run IDs `F-a1`, `F-a2`, … until the first that does not exist (direct
`api.run()` lookups; deterministic, no filter semantics). Combine with the
local cache (§6.8) and, when not `--dry-run`, with the scheduler state of the
cluster each cached attempt was submitted to.

Scheduler state is a per-cluster map `scheduler_id → queued | running |
error`, with absence meaning `gone`. It is fetched lazily: the first time
reconcile meets a cache entry whose `cluster` has not been queried, orch opens
that cluster's connection, calls the backend's `status_map`, and memoises the
result for the rest of the invocation. The target cluster is queried on every
real run regardless (staging needs it). A cluster no planned fingerprint has
cached attempts on is never contacted. `--dry-run` fetches nothing and treats
every cached attempt as `unknown`, which the table reads as still pending.

| W&B state of highest attempt  | Scheduler state           | Decision                                      |
| ----------------------------- | ------------------------- | --------------------------------------------- |
| no attempts exist             | not in cache              | submit `a1`                                   |
| no attempts exist             | cache: queued/running     | skip (pending)                                |
| no attempts exist             | cache: gone               | submit `a2` (`a1` is considered burned)       |
| `running`                     | queued/running or unknown | skip (running)                                |
| `running`                     | gone                      | treat as crashed → submit next                |
| `finished`                    | any                       | skip (finished)                               |
| `failed`, `crashed`, `killed` | any                       | submit next                                   |
| any open                      | error (SGE `Eqw`)         | skip (error); not auto-resubmitted            |

`--force` submits the next attempt whenever this table would skip an existing
one (`pending`, `running`, `finished`, `error`, or an unrecognized W&B
state). The reason is `force`. The previous attempt is left as it is. Rows
that already submit the next attempt keep their own reason. With no attempts,
`--force` still submits `a1`.

Identity is cluster-independent: a fingerprint that finished on eddie is
`finished` when planned on `local`, and a fingerprint pending on dmog is
`pending` when planned anywhere else. The next attempt of a fingerprint may be
submitted to a different cluster than the previous one.

Eval jobs additionally require an upstream: if the upstream is `finished`, or
is being submitted or is in flight on the target cluster in this invocation,
the eval proceeds (with a hold where applicable); otherwise it is skipped with
reason `upstream-not-available`.

Attempt numbers only increase and are never reused. Before submitting attempt
`n` (not in `--dry-run`), the orchestrator also checks whether
`work_dir/F-a{n}/` already exists on the target cluster; if it does, that
attempt was consumed by an earlier submission the cache does not know about
(lost cache, another orchestrator machine), and `n` is incremented. The run
directory thus doubles as a ledger that survives cache loss, per cluster. If
`wandb.init` on the compute node rejects a run ID as previously used (deleted
run), the wrapper increments the attempt, renames its run directory
accordingly, and proceeds; the orchestrator's next reconcile discovers the
actual attempt by probing. With the two-phase train wrapper (§6.5) this bump
happens before the script starts, so the script never sees the pre-bump id.

### 6.3 Code staging

One connection to the target cluster per `orch run` (SSH, or a local
subprocess for backend `local`). All commands via `bash -lc`. The script is
the same on every backend.

```
cd {repo_dir}
flock .orch-staging.lock bash -c '
  git fetch --quiet origin
  git cat-file -e {sha}^{commit}                       # must exist on origin
  if [ ! -d .worktrees/{sha} ]; then
    git worktree add --detach --no-checkout .worktrees/{sha} {sha}
    if [ -n "{orchignore_patterns}" ]; then
      git -C .worktrees/{sha} sparse-checkout set --no-cone --stdin <<EOF
/*
{negated .orchignore lines}
EOF
    fi
    git -C .worktrees/{sha} checkout --quiet {sha}
  fi
  cd .worktrees/{sha}
  export UV_PROJECT_ENVIRONMENT={repo_dir}/.venv
  lock_hash=$(sha256sum uv.lock | cut -c1-16)
  if [ "$(cat {repo_dir}/.venv/.orch-lock-hash 2>/dev/null)" != "$lock_hash" ]; then
    uv sync --frozen
    echo "$lock_hash" > {repo_dir}/.venv/.orch-lock-hash
  fi
'
```

- Per-SHA worktrees give every job a source tree that cannot change under it
  while it runs. Jobs `cd` into their worktree; `python -m orch` imports `orch`
  from there (the working directory precedes site-packages on `sys.path`).
- `.orchignore` (B6) is applied as a non-cone sparse checkout: `/*` includes
  everything, then each `.orchignore` line is negated (`!pattern`). Non-cone
  sparse-checkout patterns use gitignore syntax, the same syntax
  `git check-ignore` applies at preflight, so the set of paths allowed to be
  dirty locally and the set of paths absent from the worktree are the same by
  construction. The patterns come from the `.orchignore` at the staged SHA,
  which is the committed, clean file the preflight read. A worktree that
  already exists is not re-sparsified; changing `.orchignore` is a new commit
  and therefore a new worktree.
- One shared venv at `repo_dir/.venv`, selected via `UV_PROJECT_ENVIRONMENT`,
  because a per-worktree venv would duplicate torch and vllm. On a `local`
  cluster this is a second venv next to the developer's, on purpose (A3).
- `uv sync --frozen` runs only when `uv.lock` changed. If it must run while
  the cache shows active jobs on this cluster, `orch run` refuses unless
  `--resync` is passed, and prints which jobs are active. This is the one
  remaining shared-mutable-state hazard and it is made loud rather than
  hidden. "Active" is judged from this cluster's scheduler state only.
- Staging output is streamed to the terminal; `uv sync` can take minutes on a
  cold cache. Paramiko keepalives are enabled.

### 6.4 Submission

`job.sh` has a backend-owned head and a shared body. The body is the same
on every backend:

```bash
set -euo pipefail
{{ setup }}
{% for k, v in env.items() %}export {{ k }}={{ v | shquote }}
{% endfor %}
cd {{ src_dir }}
exec {{ venv_python }} -m orch exec {{ run_dir }}
```

The backend supplies the shebang, the scheduler directive block (job name
`orch_{{ fingerprint }}_a{{ attempt }}`, `scheduler.out`/`scheduler.err` in
the run directory, then the cluster's rendered `header`), and, for `local`, a
preamble that implements holds and device claims. Every backend implements:

```python
class Backend(Protocol):
    name: str                      # sge | slurm | local
    tools: tuple[str, ...]         # checked by cluster check (A2)
    job_id_env: str                # A8
    def validate_resources(self, resources: dict, cluster: Cluster) -> None: ...
    def render_job_script(self, job: Job, cluster: Cluster, env: dict[str, str],
                          src_dir: str, venv_python: str, hold_on: list[str]) -> str: ...
    def submit(self, conn: Conn, run_dir: str, hold_on: list[str]) -> str: ...  # scheduler id
    def status_map(self, conn: Conn) -> dict[str, str]: ...  # id -> queued|running|error; absent = gone
    def cancel(self, conn: Conn, scheduler_id: str) -> None: ...
```

and `Conn` is:

```python
class Conn(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def run(self, command: str, *, check: bool = True, stream: bool = False) -> Result: ...  # bash -lc
    def put_text(self, path: str, content: str) -> None: ...
    def exists(self, path: str) -> bool: ...
```

`SSH` (paramiko, v0) and `Local` (subprocess and `pathlib`) implement it. The
driver, staging, submission, status, logs, and cancel code sees only these two
protocols.

Holds release when the referenced job _exits for any reason_ on every
backend (Slurm's `afterok` narrows this to a successful exit, see below). The
correctness of dependent evals therefore rests on the wrapper's upstream check
(§6.6), not on the scheduler.

**SGE (eddie).** Shebang `#!/bin/bash`. Directives `#$ -N`, `#$ -o`, `#$ -e`.
Submission `qsub -terse [-hold_jid <id>[,<id>]] {run_dir}/job.sh`; the single
line of output is the scheduler ID. `qstat -xml` for status: `qw`/`hqw` →
queued, `r`/`t` → running, any `E` → error. `qdel <id>` to cancel. Job id
variable `JOB_ID`.

**Slurm (dmog).** Shebang `#!/bin/bash -l`. Directives `#SBATCH -J`,
`#SBATCH -o`, `#SBATCH -e`. Submission
`sbatch --parsable [--dependency=afterok:<id>[:<id>] --kill-on-invalid-dep=yes] {run_dir}/job.sh`;
the output is `<id>` or `<id>;<cluster>`, the part before `;` is the scheduler
ID. `afterok` plus `--kill-on-invalid-dep=yes` makes Slurm cancel a held eval
whose train job exited non-zero, which removes most of the queue wait that
`afterany` would incur; the wrapper's upstream check stays authoritative
because a train job can exit 0 and still be `failed` (no model directory).
`squeue -h -u $USER -o '%i|%t|%j'` for status: `PD` → queued, `R`, `CG`, `CF`
→ running, `S` → queued, anything else or absent → gone. `scancel <id>` to
cancel. Job id variable `SLURM_JOB_ID`.

**Local (orchestrator machine).** No scheduler exists, so `job.sh` carries the
scheduler's duties itself and tmux provides detachment. Shebang `#!/bin/bash`.
Submission `tmux new-session -d -s orch_{fp}_a{n} bash {run_dir}/job.sh`; the
session name is the scheduler ID. The preamble, before the shared body:

```bash
exec >>{{ run_dir }}/scheduler.out 2>>{{ run_dir }}/scheduler.err
export ORCH_JOB_ID=orch_{{ fingerprint }}_a{{ attempt }}
echo held > {{ run_dir }}/state
{% for dep in hold_on %}
while tmux has-session -t {{ dep }} 2>/dev/null; do sleep 20; done
{% endfor %}
cd {{ src_dir }}
{% if device is not none %}
ORCH_DEVICE={{ device }}
flock {{ work_dir }}/.devices/lock sh -c 'echo $ORCH_JOB_ID > {{ work_dir }}/.devices/{{ device }}'
{% else %}
until ORCH_DEVICE=$(flock {{ work_dir }}/.devices/lock {{ venv_python }} -m orch.backends.local claim {{ work_dir }} {{ devices | join(",") }} $ORCH_JOB_ID); do
  sleep 20
done
{% endif %}
export CUDA_VISIBLE_DEVICES=$ORCH_DEVICE
trap 'code=$?; rm -f {{ work_dir }}/.devices/$ORCH_DEVICE; echo exit:$code > {{ run_dir }}/state' EXIT
echo running > {{ run_dir }}/state
```

The body's final `exec` is replaced by a plain call on this backend so the
`EXIT` trap runs. `claim` is a small pure function invoked through the venv's
Python: a device is free when its claim file is absent or names a tmux
session that no longer exists, and `nvidia-smi --query-compute-apps` reports
no process on it (so GPUs busy with the operator's own work are respected;
compute-apps reports GPU UUIDs, so indices are mapped through
`nvidia-smi --query-gpu=index,uuid`); it writes the claim and prints the
index, or exits non-zero when nothing is free. A pinned `device` is claimed without the free test. A job with no free
device waits in its tmux session, which is the local equivalent of sitting in
a queue; there is no submit-time admission check (decision 23). The 20-second
polls cost a tmux socket call and one `nvidia-smi` invocation. `status_map`
lists tmux sessions named `orch_*` and reads each `state` file: `held` →
queued, `running` → running, no live session → gone. `cancel` is
`tmux kill-session -t <id>`, which also drops the claim on the next
`claim` call because the session is gone. Job id variable `ORCH_JOB_ID`.
`gpus` must be 1; `time` and `mem` are accepted and ignored.

Environment exported into every job (`env` in `job.json`), in increasing
precedence: cluster `env`, entry `env`, spec `env`, experiment-run `env`, CLI
`--set env.K=V`, then the orch variables below. Placeholders (§4.1) are filled
in all user-supplied values after the merge.

| Variable                        | Value                                                      |
| ------------------------------- | ---------------------------------------------------------- |
| `WANDB_RUN_ID`                  | run ID                                                     |
| `WANDB_PROJECT`, `WANDB_ENTITY` | from `orch.local.yaml`                                     |
| `WANDB_NAME`                    | run name                                                   |
| `WANDB_JOB_TYPE`                | `train` or `eval`                                          |
| `WANDB_TAGS`                    | comma-separated tags                                       |
| `WANDB_RUN_GROUP`               | only if `--group` was given                                |
| `WANDB_RESUME`                  | `allow`                                                    |
| `WANDB_DIR`                     | `{run_dir}/wandb`                                          |
| `ORCH_RUN_DIR`                  | `{run_dir}`                                                |
| `ORCH_PARAMS`                   | `{run_dir}/params.yaml`                                    |
| `ORCH_MODEL_DIR`                | train: `{run_dir}/model`                                   |
| `ORCH_MODEL_PATH`               | eval: set by the wrapper after resolution, not in `job.sh` |
| `ORCH_OUTPUT_DIR`               | eval: `{run_dir}/output`                                   |
| `CUDA_VISIBLE_DEVICES`          | local: set by the preamble after the claim, not in `job.json` |

The v0 `HF_METRICS_CACHE` line in the orch layer is gone; the `lm_eval` entry
sets it (§4.1).

### 6.5 `orch exec` — train

Runs on the compute node with `cwd = src_dir`. Two phases, so that the run
carries its orch configuration while the script runs (F3) while still never
having two writers on the run at once (decision 7):

1. Read `job.json`. Create `model/`, `checkpoints/`, `wandb/`.
2. **Prologue.** `wandb.init(id=run_id, entity, project, name=run_name,
   job_type="train", tags, resume="allow", dir=run_dir/wandb)`. If W&B
   rejects the id as previously used (deleted run), bump the attempt, rename
   the run directory, rewrite `job.json` and the `WANDB_RUN_ID`/`WANDB_NAME`
   the script will inherit, and retry; this now happens before the script
   exists, so the script never sees a stale id. Then
   `run.config.update({"params": spec.params, "orch": <§7 block>, **flat
   orch_* keys}, allow_val_change=True)` with `orch_state="running"`, and
   `run.finish()`. The prologue takes seconds; the run reads `finished`
   until the trainer's own `wandb.init` resumes it, which is accepted
   (decision 28).
3. Launch `command` as a subprocess with the environment above, tee'ing
   stdout/stderr to `stdout.log`/`stderr.log` and to the wrapper's own
   stdout/stderr (so `scheduler.out`/`scheduler.err` also contain them). The
   trainer attaches to the run through `WANDB_RUN_ID` and
   `WANDB_RESUME=allow`. Wait.
4. **Epilogue.** `wandb.init(id=run_id, …, resume="allow")` again; this
   attaches to the run whether or not the script ever initialised W&B.
5. If exit code is 0 and `model/config.json` exists and the directory
   contains at least one weights file: build artifact `model-{fingerprint}`
   (type `model`, metadata per §7), `add_dir(model/)`, `log_artifact(aliases=  [f"a{attempt}", "latest"])`, wait for upload. Set `orch_state=succeeded`.
   Otherwise: add tag `failed`, set `orch_state=failed`,
   `summary["orch/exit_code"]`, and if the exit code was 0 but D3 failed,
   record `orch/failure_reason="model_dir_missing"`. Either way re-apply the
   config update so `orch.scheduler_id`, `orch.host`, and `orch.ended_at`
   are final.
6. `wandb.save` of `job.sh`, `job.json`, `params.yaml`, `fingerprint.json`,
   `stdout.log`, `stderr.log`, `scheduler.out`, `scheduler.err` with
   `policy="now"` and `base_path=run_dir`.
7. `run.finish(exit_code=<subprocess exit code>)`. Exit with the same code.

### 6.6 `orch exec` — eval

1. Read `job.json`. If `upstream_run_id` is set: fetch that run via the API.
   If its state is not `finished`, print
   `upstream <id> not finished (state=…); exiting without creating a run`.
   If it is `finished` but has no `model` artifact, print
   `upstream <id> finished but has no model artifact; exiting without creating a run`.
   Either way exit 0. No W&B run is created (F1 exception); the next `orch run` will plan
   a new eval against whichever attempt succeeds. Keep the fetched producer
   run object for step 2.
2. `wandb.init(id=run_id, …, job_type="eval")`, then
   `config.update({"params": …, "orch": … with orch_state="running",
   "upstream": <block>, **flat})`. The `upstream` block (F4, §7.1) is copied
   from the producer: for `run:` (and therefore `fingerprint:`) references,
   from the run fetched in step 1; for `artifact:` references, from
   `artifact.logged_by()` when that run exists; for `hf:` references it is
   `null`. It is a snapshot at eval time.
3. Resolve the model reference to a local directory:

- `run:` → `api.run(id).logged_artifacts()` filtered to type `model`;
  `run.use_artifact(artifact)`; download into
  `work_dir/cache/models/<digest>/` under a lock, honouring a `.complete`
  marker.
- `artifact:` → `run.use_artifact(name:vN)`; same cache.
- `hf:` → `snapshot_download(repo, revision)`; record the resolved commit in
  `config.orch.model.resolved_revision`.
  Export `ORCH_MODEL_PATH`; substitute `{model_path}` in `command`.

4. Launch the subprocess as in §6.5 step 3.
5. On exit 0: import the entry's parser, call it with `output/`, validate a
   non-empty mapping of finite floats, and `wandb.log({f"{stem}/{k}": v})`
   where `stem` is the eval spec stem. Set `orch_state=succeeded`. On
   non-zero exit or parser failure: tag `failed`, `orch_state=failed`,
   `summary["orch/exit_code"]`, `orch/failure_reason`.
6. `wandb.save` of the files in §6.5 step 6 plus everything under `output/`
   (`base_path=run_dir`).
7. `run.finish(exit_code)`. Exit with the same code.

### 6.7 Failure matrix

| Situation                                                | Visible as                                                    | Next `orch run` does                                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Script crashes (non-zero exit)                           | W&B run `failed`, tag `failed`, logs attached                 | submits next attempt                                                                         |
| Script exits 0 but no model dir                          | W&B run `failed`, `failure_reason=model_dir_missing`          | submits next attempt                                                                         |
| Job killed by scheduler (`time`)                         | W&B run `crashed` after heartbeat timeout; `scheduler.err` says why | submits next attempt (after W&B flips; before that, "running + scheduler gone" rule applies) |
| `job.sh` fails before Python (module load, missing venv) | no W&B run; cache: scheduler gone                             | submits next attempt; `orch logs` shows `scheduler.err` from the cluster                     |
| Eval's upstream failed                                   | no W&B run for the eval (SGE, local: the eval starts and exits; Slurm: cancelled by `afterok`); upstream `failed` | plans evals against the upstream's next attempt                            |
| Deleted W&B run ID collides                              | train: prologue bumps attempt and renames run dir before the script starts; eval: as v0 | probing finds the real attempt                                               |
| Staging fails (fetch, sync)                              | `orch run` aborts before writing any job                      | nothing to clean up                                                                          |
| Local: no free device                                    | tmux session alive, `state` = `held`, `status` shows `queued` | nothing; the job starts when a device frees. `orch cancel` to give up                        |
| Local: tmux server killed                                | every local session gone; W&B runs `crashed` after timeout    | submits next attempts; the device claims are stale and ignored                               |
| Local: operator process on a device                      | `claim` skips that device                                     | nothing                                                                                      |
| Attempt in flight on a cluster that was not queried      | cannot happen for cached attempts (lazy fetch); with a lost cache the attempt is invisible and may be re-minted on the target cluster | the two attempts share a W&B id via `resume=allow`; see §15 |

### 6.8 Local cache

`~/.local/state/orch/<entity>__<project>.json`:
`{run_id: {cluster, scheduler_id, fingerprint, attempt, kind, name, submitted_at}}`. It is a cache of scheduler handles only. `cluster` routes the
scheduler query (§6.2) and `cancel`/`logs`. Losing it means `status` degrades
to W&B-only for jobs that have not yet started, and that a queued job may be
re-planned as a new attempt once the run-directory ledger check (§6.2)
notices the gap. It never causes data loss. Because the file name is derived
from the W&B entity and project, a shell that exports `WANDB_ENTITY` or
`WANDB_PROJECT` for another purpose reads a different cache (and targets a
different project); `orch status` prints the cache path it is using when the
environment overrides the file.

---

## 7. W&B schema

Project: `orch-exp` (from `orch.local.yaml`). `job_type`: `train` | `eval`.
`group`: unset unless `--group`.

### 7.1 Config

Nested block, for humans and for completeness:

```yaml
config:
  params: { <resolved params, including runtime keys> }
  orch:
    schema_version: 2
    orch_version: "0.2.0"
    fingerprint: 3f9a2c1d7e4b
    fingerprint_version: 1
    fingerprint_input: { kind, entry, params, model_ref } # verbatim §5.5 input
    attempt: 1
    run_id: 3f9a2c1d7e4b-a1
    kind: train
    entry: train
    spec_path: configs/train/sdft_tooluse.yaml
    description: "..."
    runtime_keys: [...]
    resources: { ... }
    evals: [...]
    env: { ... } # spec and experiment env, before merging
    overrides: [...]
    model_ref: null | "run:..." | "hf:..." | "artifact:..."
    model: {
        kind,
        id,
        resolved_revision | artifact_name,
        artifact_version,
        producer_run_id,
      } # eval only
    upstream_run_id: null | "..."
    cluster: { name, backend, repo_dir, work_dir, src_dir, device } # device: local only
    scheduler_id: "4821937" # from the backend's job id variable (A8)
    git: { sha, remote, orchignore_sha256 }
    submitted_from: "w8724.see.ed.ac.uk"
    submitted_at: "..."
    started_at: "..." # wrapper start
    ended_at: "..." # wrapper epilogue; absent while running
    host: "<compute node>"
  upstream: # eval only; null for hf: references
    run_id: 3f9a2c1d7e4b-a1
    run_name: "sdft_tooluse model=Qwen2.5-1.5B-Instruct lr=5e-05 bs=32"
    fingerprint: 3f9a2c1d7e4b
    spec: sdft_tooluse
    entry: train
    params: { <the producer's config.params, verbatim> }
```

Flat keys, for filtering (`api.runs(path, {"config.orch_fingerprint": ...})`):
`orch_fingerprint`, `orch_attempt`, `orch_kind`, `orch_entry`, `orch_spec`
(stem), `orch_cluster`, `orch_model_ref`, `orch_upstream_run_id`,
`orch_state` (`running`|`succeeded`|`failed`). All of them exist from the
wrapper's first config write, before the script starts (F3).

`upstream` is nested, never merged into the eval's own `params`: train and
eval params share key names (`temperature`, `seed`) and must not collide. In
the W&B table it appears as dotted columns (`upstream.params.learning_rate`),
which is what grouping and filtering in the UI operate on; the W&B API's
filters on nested config are unreliable, hence the flat `orch_*` keys for
programmatic access and the nested block for the UI. It is a snapshot taken
when the eval starts.

HF's `WandbCallback` adds `TrainingArguments` and the model config at the top
level of `config` on training runs; `params`, `orch`, and `upstream` are
nested to avoid collisions.

### 7.2 Tags

Automatic: `kind:<kind>`, `spec:<stem>`, `entry:<entry>`, `cluster:<name>`,
`attempt:<n>`, `gpu_type:<v>` when set, `failed` on failure. User: `--tag` on
`run`/`eval`, `tags` in experiment specs, `orch tag` after the fact.

### 7.3 Artifacts

Training runs log one artifact:

- name `model-{fingerprint}`, type `model`, aliases `a{attempt}`, `latest`
- metadata: `fingerprint`, `attempt`, `run_id`, `spec`, `entry`, `model_name`
  (from params), `params` (identity-bearing subset), `git_sha`, `cluster`
- contents: the `model/` directory

Eval runs call `use_artifact` on the model they consumed (for `run:` and
`artifact:` references), which gives W&B's lineage graph train → eval.

### 7.4 Metrics

Training runs: whatever the trainer logs (`train/*`, `eval/*`, including the
in-training `eval_tooluse_accuracy`).

Eval runs: `<eval stem>/<parser key>`, logged once, so they appear in
`summary`. Examples: `tooluse/accuracy`, `tooluse/num_correct`,
`prior_tasks/mmlu/acc`, `prior_tasks/hellaswag/acc_norm`. Parsers should
normalise lm-eval's `metric,filter` keys to `metric` when the filter is
`none`.

### 7.5 Files

Per §6.5 step 6 and §6.6 step 6. Model checkpoints are never saved as run
files; only the artifact. Runs recorded by v0 have `sge.out`/`sge.err` where
v0.2 runs have `scheduler.out`/`scheduler.err`; `orch logs` tries both.

### 7.6 Reading results without `orch`

Given a fingerprint `F`: runs `F-a1…F-an`, highest `finished` one is current.
Given an experiment: recompute fingerprints with `orch inspect` or filter by
`tags`. Join an eval to its training run through `orch_upstream_run_id`, or
read the copy in `config.upstream` directly. The method/task of a training
run is in `config.params.method` and `config.orch_spec`; for an eval run the
producing method is `config.upstream.params.method`.

---

## 8. CLI reference

All commands accept `--json`. Exit codes: `0` success; `1` usage or validation
error (nothing submitted); `2` precondition failure (dirty tree, unpushed
HEAD, cluster check failed, `--resync` required); `3` remote failure during
staging or submission (partial submission possible; the report says what was
submitted).

| Command | Purpose |
| --- | --- |
| `orch run <spec \| experiment> [--cluster C] [--model REF] [-p/-r/--set …] [--evals … \| --no-evals] [--tag T] [--group G] [--force] [--resync] [--dry-run]` | Resolve, reconcile, stage, submit to cluster `C`. `--model` is required for an eval spec and forbidden for a train spec. `--set env.K=V` sets a job environment variable. |
| `orch eval <eval-spec> --model REF [same flags]` | Sugar for `orch run <eval-spec> --model REF`. |
| `orch inspect <spec> [--model REF] [--cluster C] [-p/-r/--set …]` | Offline: print the resolved spec, fingerprint input, fingerprint, name, each params override annotated `[identity]`/`[runtime]`, and the resources validated against cluster `C` (default cluster when omitted). |
| `orch status [<spec \| experiment \| fingerprint \| run_id>…] [--all] [--cluster C] [--no-remote]` | See below. |
| `orch logs <run_id> [--follow] [--which stdout \| stderr \| scheduler.out \| scheduler.err]` | Finished run: download from W&B files. In flight: tail from the run directory on the cluster the cache names. `sge.out`/`sge.err` are accepted aliases. |
| `orch tag <run_id \| fingerprint \| spec \| experiment>… --add T [--add T]… [--remove T]…` | Edit tags via the API. |
| `orch cancel <run_id>…` | Cancel through the backend of the cluster the cache names for each id; tag the W&B run `cancelled` if it exists. |
| `orch cluster check <name> [--gpu]` | Test invariants A1–A5, A7, and A9 for the cluster's backend (seconds). `--gpu` submits a short job through the backend that imports torch and vllm, asserts a visible GPU, and performs a W&B round-trip. On `local` the check job claims a device like any job. |
| `orch exec <run_dir>` | Internal: the wrapper. |

`orch status` semantics:

- **No arguments:** every job that is `queued` or `running` on any cluster
  that appears in the local cache, one scheduler query per cluster. This is
  the "what is going on right now" view and the reason the command exists
  with more than one cluster. `--cluster C` restricts to one cluster.
  `--all` lists every cache entry instead, as v0 did.
- **A run id or fingerprint:** all attempts known to W&B or the cache.
- **A spec or experiment:** every *existing* attempt of each fingerprint the
  target resolves to (W&B probe ∪ cache), not the attempt the planner would
  submit next; `orch run --dry-run` is the command for that. In v0 `status`
  listed planned ids, which produced rows that had never been submitted and
  therefore had no name, fingerprint, or scheduler id.
- **Field sources,** in order of preference: `fingerprint` and `attempt` are
  parsed from the run id; `run_name` is the W&B run's name (`run.name`, which
  is what `WANDB_NAME` set; `config.orch` does not duplicate it), else the
  cache; `kind`, `cluster`, and `scheduler_id` come from `config.orch` when
  the run exists, else from the cache; `scheduler_state` from the cluster
  named by whichever of those two supplied the scheduler id; `wandb_state`
  from W&B. A row therefore has every column for any attempt that reached the
  wrapper, cache or no cache.
- `--no-remote` skips scheduler queries; `scheduler_state` reads `unknown`.

`--json` output for `run`/`status` is a list of objects with keys `run_id`,
`run_name`, `fingerprint`, `attempt`, `kind`, `cluster`, `decision`, `reason`,
`scheduler_id`, `wandb_state`, `scheduler_state`, `url`. A `run` row also
includes `overrides`: params entries carry `classification` `identity` or
`runtime`.

---

## 9. Package layout and interfaces

```
orch/
  __init__.py
  __main__.py          # python -m orch
  cli.py               # typer app; one function per command
  config.py            # load/validate orch.yaml and orch.local.yaml; dataclasses
  yamlio.py            # the single YAML 1.2 loader/dumper (ruamel.yaml)
  resolve.py           # spec/params loading, extends, set, runtime_keys, validation
  identity.py          # canonical JSON, fingerprint, fingerprint.json
  naming.py            # run names, value shortening
  modelref.py          # parse/canonicalise model references; submit-time resolution
  plan.py              # spec/experiment → list[Job]; eval expansion; dedup
  wb.py                # W&B API: probe attempts, run state, artifacts, tags
  reconcile.py         # decision table (§6.2)
  cache.py             # local scheduler-id cache
  conn.py              # Conn protocol, Local implementation, open_conn(cluster)
  ssh.py               # paramiko Conn: run(), put_text(), exists(); password handling
  orchignore.py        # read .orchignore, protected-path check, dirty-path filter (via git check-ignore)
  gitpreflight.py      # B2 with .orchignore
  staging.py           # §6.3, sparse checkout
  render.py            # shared job.sh body, job.json, params.yaml, fingerprint.json, env merge and placeholders
  backends/
    __init__.py        # backend_for(cluster)
    base.py            # Backend protocol
    sge.py             # qsub/qstat/qdel
    slurm.py           # sbatch/squeue/scancel
    local.py           # tmux sessions, holds, device claims; `python -m orch.backends.local claim`
  exec/
    __init__.py        # entry point for `orch exec`
    common.py          # subprocess tee, W&B init/finish, file upload, config block
    train.py           # §6.5, two phases
    eval.py            # §6.6, upstream block
    models.py          # model reference → local path, cache, locks
  check.py             # orch cluster check, per backend

orch_adapters.py       # repo root; codebase-specific parsers referenced from orch.yaml
.orchignore            # repo root; paths the cluster worktree does not need
tests/
```

Interfaces are given in §6.4 (`Backend`, `Conn`). In addition:

```python
Parser = Callable[[Path], dict[str, float]]

# backends/local.py, also usable as a pure function in tests
def claim(work_dir: Path, devices: list[int], session: str,
          live_sessions: set[str], busy_devices: set[int]) -> int | None: ...

# reconcile.py
SchedulerStates = Callable[[str], dict[str, str] | None]   # cluster name -> status map; None when not queried
```

No new dependencies: `subprocess`, `pathlib`, `shutil`, and `fcntl` cover the
local connection and backend; `git check-ignore` and `git sparse-checkout`
cover `.orchignore`. On the cluster, jobs invoke `python -m orch` from the
worktree and do not depend on the editable install.

---

## 10. Codebase changes

### 10.1 Training

- New `train.py --config <params.yaml>`. Reads the mapping; pops the
  orchestration-level keys `method`, `model_name`, `dtype`, `train_data`,
  `val_size`; passes the remainder to `SDFTConfig(**rest)` or
  `SFTConfig(**rest)` according to `method`, so unknown keys fail at
  construction. Sets, itself and not from params: `output_dir = $ORCH_RUN_DIR/checkpoints`, `report_to="wandb"`, `eval_strategy="epoch"`,
  `save_strategy="epoch"`, `load_best_model_at_end=True`,
  `metric_for_best_model="eval_tooluse_accuracy"`, `greater_is_better=True`,
  `model_init_kwargs={"device_map": {"": 0}, "dtype": dtype}`. Leaves
  `run_name` unset. After `trainer.train()`: `trainer.save_model($ORCH_MODEL_DIR)`
  and `tokenizer.save_pretrained($ORCH_MODEL_DIR)`.
- New `sdft_exp/tooluse.py` (or similar) holding `format_for_sft`,
  `format_for_sdft`, `tooluse_is_correct`, `ToolUseAccuracyCallback`,
  `paper_teacher_prompt_template` default, currently duplicated between
  `main.py` and `sft_tooluse.py`.
- Delete `main.py` and `sft_tooluse.py` once `train.py` reproduces both.
- Note on SDFT's `eval_strategy`: `main.py` currently passes `"epochs"`
  while the `IntervalStrategy` value is `"epoch"`; check which the installed
  TRL accepts and use that in `train.py`.
- Smoke runs use `-p max_steps=5`, which is identity-bearing, so they never
  share a fingerprint with a real run.

### 10.2 Evaluation

- `eval_tooluse.py`: add `--data_dir` (default unchanged) and
  `--gpu_memory_utilization` (default 0.8). No W&B.
- `eval_science.py`: same two flags.
- `lm_eval` used as-is through its entry.
- New `orch_adapters.py` with `parse_eval_results(output_dir)` (reads
  `eval_results.json` as written by both `eval_tooluse.py` and
  `eval_science.py`; returns `accuracy`, `num_correct`, `num_total`) and
  `parse_lmeval(output_dir)` (newest `results_*.json` under `output_dir`
  recursively; flatten `results` as `upload_lmeval_wandb.py` does today;
  normalise `,none` suffixes).
- Delete `run_tooluse_eval_upload.sh`, `run_lmeval_upload.sh`,
  `upload_lmeval_wandb.py`, `export_wandb_env.sh`, `sft_tooluse_sweep.sh`.

### 10.3 Configuration tree

```
orch.yaml
orch.local.example.yaml
configs/
  train/sdft_tooluse.yaml
  train/sft_tooluse.yaml
  eval/tooluse.yaml
  eval/science.yaml
  eval/prior_tasks.yaml
hparams/
  sdft_base.yaml
  sdft_tooluse.yaml
  sft_tooluse.yaml
experiments/
  tooluse_main.yaml
```

### 10.4 Housekeeping

`.gitignore`: add `orch.local.yaml`, `.worktrees/`, and `.orch-staging.lock`
(the latter two appear only in cluster checkouts but keep `git status` clean
there). `README.md`: replace the single lm-eval command with a pointer to
`orch/README.md`.

### 10.5 v0.2 changes outside `orch/`

- Every spec under `configs/`: `h_rt` → `time` in `resources`. Fingerprints
  are unaffected.
- `orch.yaml`: the `lm_eval` entry gains `HF_METRICS_CACHE: "{run_dir}/metrics"`
  in `env` (§4.1).
- New `.orchignore` at the repo root. Initial content: `docs/`, `tests/`,
  `figures/`, `evals/`, `models/`, `wandb/`, `output/`, `*.ipynb`. Not
  `orch/README.md`: B6 protects everything under `orch/`. Not `*.md`:
  `pyproject.toml` names `README.md` and B6 refuses to exclude it. Not
  `configs/`, `hparams/`, `experiments/`: harmless either way and useful to
  have on the cluster for `orch inspect` from a login node.
- `orch.local.example.yaml`: `dmog` and `local` entries as in §4.2, every
  field commented, with `<user>` placeholders for dmog paths.
- `tests/test_contracts.py`: `configs/train/sdft_qwen35.yaml` →
  `configs/train/sdft_tooluse_qwen35.yaml` (stale path from a rename).
- `orch/__init__.py`: `__version__ = "0.2.0"`.

---

## 11. Worked example: data for the "new task vs prior tasks" figure

Once, per cluster:

```
wandb login            # on eddie's login node; writes ~/.netrc on shared home
orch cluster check eddie --gpu
```

Then, from a clean, pushed checkout:

```
orch run experiments/tooluse_main.yaml --dry-run
orch run experiments/tooluse_main.yaml
```

The first `orch run` submits: two base-model evals (no dependencies), three
training jobs, and six evals each holding on its training job. W&B ends up
with eleven runs. Each training run has a `model-<fp>` artifact; each eval run
has `orch_upstream_run_id` (or `orch_model_ref=hf:…`), `use_artifact` lineage,
and `tooluse/accuracy` or `prior_tasks/*` in its summary. The figure's x-axis
is `tooluse/accuracy`, the y-axis an aggregate of `prior_tasks/*`, grouped by
the upstream's `config.params.method`.

Changing the learning rate: `orch run configs/train/sdft_tooluse.yaml -p learning_rate=1e-5` submits one training job and two evals; nothing else is
touched. Rerunning the experiment spec afterwards reports everything as
finished and submits nothing.

An OOM on the SDFT run: `orch run configs/train/sdft_tooluse.yaml -p steps_per_generation=16` resubmits as attempt 2 with the same fingerprint (the override is reported as `[runtime]`);
its evals are planned against `…-a2`.

"SFT+Re-invoke" is an eval spec `configs/eval/tooluse_reinvoke.yaml` (an entry
and parser the evaluation code must grow; not part of v0) added to the SFT
spec's `evals`.

Running the same experiment on the workstation instead:

```
orch cluster check local --gpu
orch run experiments/tooluse_main.yaml --cluster local --dry-run
orch run experiments/tooluse_main.yaml --cluster local
orch status
```

`cluster check local` confirms tmux, nvidia-smi, the separate clone, and the
device list, and runs the GPU round-trip through a claimed device. The run
submits one tmux session per job; the two base-model evals and the training
jobs start as soon as devices are free, evals hold on their training session
and take the device it releases. `orch status` shows each session as
`queued` (held or waiting for a device) or `running`. A fingerprint that
already finished on eddie is reported `finished` and not resubmitted;
`--force` applies as before. `tmux attach -t orch_<fp>_a<n>` shows a job's
scheduler-level output live; `orch logs <run_id> --follow` tails
`stdout.log` as on a cluster.

In the W&B table, grouping eval runs by `upstream.params.learning_rate` and
`upstream.params.method` gives the "new task vs prior tasks" figure's rows
directly; base-model evals fall into the null group.

---

## 12. Implementation sequence

Each milestone ends with a verification that does not depend on later
milestones. M1–M3 need no cluster. M2 can proceed in parallel with M1.

**M1 — Resolution, identity, naming (pure).** `yamlio`, `config`, `resolve`,
`identity`, `naming`, `modelref` (parsing only), `inspect`. Verification:
unit tests in §13.1 pass; `orch inspect configs/train/sdft_tooluse.yaml`
prints a resolved spec, fingerprint input, fingerprint, and name;
`orch inspect … -p learning_rate=1e-5` changes the fingerprint and the name
and reports the override as `[identity]`; `… -p steps_per_generation=16`
changes neither and reports `[runtime]`; `… -r gpu_type=h200` adds the line
to the rendered header and changes nothing else.

**M2 — Codebase contracts.** `train.py`, shared module, eval flag additions,
`orch_adapters.py`, configuration tree, `orch.yaml`. Verification: with
`ORCH_RUN_DIR`, `ORCH_MODEL_DIR`, `WANDB_MODE=offline` set by hand,
`python train.py --config <resolved sdft params with max_steps: 3>` and the
same for `sft` run on the 8-GPU box and leave a loadable `model/`; `python eval_tooluse.py --model_path model/ --output_dir out/` writes results the
parser reads; `parse_lmeval` reads the existing
`output/Qwen__Qwen2.5-1.5B-Instruct/results_*.json`.

**M3 — Planning, W&B lookup, rendering, dry run.** `plan`, `wb`, `reconcile`,
`cache`, `render`, `backends/base`, `backends/sge` (render only). Verification:
`orch run experiments/tooluse_main.yaml --dry-run` against the real `orch-exp`
project prints correct decisions for an empty project; unit tests for the
decision table with a fake W&B; rendered `job.sh` for the eddie header with
and without `gpu_type` matches expectations byte-for-byte in a golden test.

**M4 — SSH, staging, cluster check.** `ssh`, `staging`, `check`, SGE
`submit`/`status`/`cancel`. Verification: `orch cluster check eddie` passes
with key or password auth; `orch cluster check eddie --gpu` submits, runs, and
reports a W&B round-trip from a compute node; staging creates a worktree and
syncs the venv once, and a second invocation is a no-op.

**M5 — Wrapper, train path.** `exec/common`, `exec/train`. Verification:
`orch run configs/train/sft_tooluse.yaml -p max_steps=5 --no-evals` on eddie
produces a `finished` W&B run named per §5.6 with `config.params`,
`config.orch`, flat keys, uploaded files, and a `model-<fp>` artifact with
aliases `a1` and `latest`. Then `-p max_steps=5 -p not_a_real_key=1 --no-evals` makes `train.py` reject the unknown key at config construction and
produces a `failed` run with logs attached and no artifact; `orch run` of the
same command again reports that it would submit attempt 2.

**M6 — Wrapper, eval path, dependencies.** `exec/eval`, `exec/models`.
Verification: `orch eval configs/eval/tooluse.yaml --model hf:Qwen/Qwen2.5-1.5B-Instruct`
yields `tooluse/accuracy` in summary and the results files attached; `orch run configs/train/sft_tooluse.yaml -p max_steps=5` submits train plus two evals
with `-hold_jid`, and all three finish with `use_artifact` lineage visible;
killing the training job (`orch cancel`) causes both evals to exit without
creating runs, and a subsequent `orch run` plans them against attempt 2.

**M7 — Experiment specs and remaining commands.** `status`, `logs`, `tag`,
`cancel`, `--force`, `--resync`, experiment expansion with dedup.
Verification: `orch run experiments/tooluse_main.yaml` end to end on eddie
(real epochs; this is the long test); rerun reports all finished; `--force` on
one spec yields attempt 2 and fresh evals; `orch status` and `orch logs` work
for queued, running, and finished runs.

**M8 — Cleanup and documentation.** Delete superseded scripts; write the
deliverables in §17; add `orch.local.example.yaml`; update `.gitignore` and
`README.md`.

### v0.2 sequence

M1–M8 are done. The v0.2 milestones follow the same rule: each ends with a
verification that does not depend on later ones. M9–M11 and M13–M15 need no
cluster; M12 needs this machine's GPUs; the dmog and eddie checks at the end
of M12 need those clusters and are the only steps that cannot be verified
here. The detailed task breakdown lives in the implementation plan, not in
this document.

**M9 — Backend and connection seam.** `Conn` protocol with `Local`;
`backend_for(cluster)`; `Backend.validate_resources`; `job.sh` split into
backend head and shared body; `scheduler.out/err`; `cluster.job_id_env`;
canonical resource keys and the spec rename; cluster config validation per
backend. Verification: the golden `job.sh` test for eddie is byte-identical
apart from the two log file names; all existing tests pass; `orch inspect`
and `orch run --dry-run` behave as before on every committed spec.

**M10 — Multi-cluster reconcile and ops.** Lazy per-cluster scheduler
states; `cluster` in every row; `status` semantics of §8; `cancel` and `logs`
by cached cluster; `orch_cluster` flat key. Verification: unit tests with two
fake clusters show an attempt pending on cluster A is `pending` when planning
on B and that B's planner never contacts a cluster without relevant cache
entries; `status` rows for a run with no cache entry still carry name, kind,
fingerprint, scheduler id from a fake W&B config.

**M11 — Slurm backend.** Directive block, `sbatch --parsable`, `squeue`
parsing, `scancel`, `afterok`. Verification: unit tests on recorded `squeue`
output and a golden Slurm `job.sh`.

**M12 — Local backend.** tmux submission, preamble with holds and claims,
`claim`, `status_map`, `cancel`, `cluster check local [--gpu]`. Verification:
`claim` unit tests over live-session and busy-device fixtures; a golden local
`job.sh`; on this machine, `orch cluster check local --gpu` passes, then
`orch run configs/train/sft_tooluse.yaml -p max_steps=5 --cluster local`
produces a finished train run, two evals that held and then ran, correct
`config.orch.cluster`, and `orch status` showing them while in flight; then
`orch cluster check dmog --gpu` and `orch cluster check eddie --gpu`.

**M13 — Env and `.orchignore`.** Placeholders, `env` on specs and experiment
runs, `--set env.K=V`, `HF_METRICS_CACHE` moved into `orch.yaml`;
`orchignore.py`, preflight filter, sparse-checkout staging, protected set.
Verification: unit tests for merge precedence and placeholders; a preflight
test with a dirty ignored path passing and a dirty `train.py` failing; a
protected-path rejection test; a staging test that the sparse-checkout
pattern list is `/*` plus negated lines.

**M14 — Wrapper changes.** Two-phase train wrapper with `orch_state=running`
and the prologue-time collision bump; `config.upstream` in the eval wrapper;
`schema_version` 2 with version-1 fallback. Verification: wrapper tests with
a fake `wandb` module asserting the order prologue → script → epilogue, the
config keys present after the prologue, and the upstream block contents; one
`-p max_steps=5` smoke run on `local` confirming the trainer resumes the run
and the final state is `finished` with the artifact.

**M15 — Documentation and version.** README (§17), example config, this
document's decision log checked against the implementation, `0.2.0`.

---

## 13. Testing

### 13.1 Unit (no network, no cluster)

- `resolve`: extends chains (spec and params), deep-merge semantics, path
  rules (file-relative vs repo-relative), reserved-key stripping,
  `runtime_keys` union, override grammar and value parsing,
  identity/runtime annotation of overrides, validation errors from §5.7.
- `identity`: canonicalisation fixtures (key order, `2` vs `2.0`, null vs
  absent, unicode), runtime-key exclusion, model-ref inclusion, a golden
  fingerprint for the committed `sdft_tooluse` spec so accidental changes to
  the scheme are caught.
- `naming`: stem, `name_keys`, override segments, shortening, attempt suffix,
  eval handles.
- `render`: golden `job.sh` for the eddie header with and without `gpu_type`;
  `{flags}` rendering; env precedence; shell quoting.
- `reconcile`: every row of the §6.2 table against a fake W&B and fake
  scheduler state; eval upstream rules; two-cluster cases: attempt pending
  elsewhere, attempt burned elsewhere, lazy fetch touching only clusters with
  relevant cache entries, dry-run touching none.
- `plan`: experiment expansion, dedup with union of evals, `--no-evals`,
  `--evals`; `env` merge and the equal-`env` rule for deduplicated entries.
- `render`: golden `job.sh` for each backend (eddie header with and without
  `gpu_type`; dmog header; local with and without holds, with and without a
  pinned device); env precedence across the five layers; placeholder filling;
  rejection of unknown placeholders.
- `backends`: `qstat -xml`, `squeue`, and `tmux list-sessions` + `state`
  parsing fixtures; `claim` as a pure function (free device, all claimed,
  stale claim from a dead session, operator process on a device, pinned
  device); resource validation per backend.
- `conn`: `Local.run`, `put_text`, `exists` against a temporary directory;
  the staging script executed by `Local` against a temporary bare repo
  produces a sparse worktree without the ignored paths.
- `orchignore`: pattern matching delegated to `git check-ignore` in a
  temporary repo; the protected set; hash recorded in `job.json`.
- `gitpreflight`: dirty ignored path passes; dirty tracked non-ignored path
  fails; dirty `.orchignore` fails; untracked file under an ignored directory
  passes.
- `exec`: with a fake `wandb`, train prologue writes config with
  `orch_state=running` before the subprocess starts, epilogue finalises;
  collision bump in the prologue renames the run directory; eval writes
  `upstream` for a `run:` reference and `null` for `hf:`; both read
  `job_id_env`.
- `ops`: `status` field precedence (W&B config over cache), active-only
  default, `--all`, spec targets listing existing attempts only.

### 13.2 Integration (fake backend)

A `FakeBackend` and in-memory `Conn` that record what would have been written
and submitted. Drive `orch run` end to end against a recorded W&B fixture and
assert the submission order, holds, and file contents, once per backend.

### 13.3 On the clusters

The M4–M7 and M12/M14 verifications above, kept as a checklist in
`orch/README.md`. Use `-p max_steps=5` for anything that is not the final M7
run. The `local` checklist runs on this machine without any cluster.

---

## 14. Decision log

Each entry: decision, reason, what would make us revisit it.

1. **W&B is the only durable state.** The orchestrator must be restartable
   anywhere and figures must be producible from W&B alone. Revisit never; the
   local cache is explicitly disposable.
2. **Content-derived identity; git SHA excluded.** Identity should answer "is
   this the same experiment", and code fixes should not orphan results. The
   SHA is metadata; `--force` creates new attempts. Revisit if attempts
   routinely pile up because code changes do change results, in which case add
   `revision` to the base params file (§16).
3. **Deny-list (**`runtime_keys`**) rather than allow-list.** An unclassified new
   key must default to identity-bearing, because the failure mode of the
   opposite default is silent conflation of experiments. For this codebase the
   deny-list is also the shorter list. Revisit by adding the strict
   `identity_keys` form as a per-file opt-in; never by flipping the default.
4. `runtime_keys` **declared in the params file, not the spec.** The person
   writing the params file knows whether a knob changes results. Specs may add
   to the list for context-dependent cases.
5. **A key in both identity and runtime roles is impossible by
   construction.** One namespace, one classification, derived only from
   `runtime_keys`. An earlier draft had separate guarded CLI flags for
   identity and runtime params (`-p`/`-r`); they were dropped because they
   added no correctness and the same information is delivered by annotating
   overrides in `run`/`inspect` output. `-r` now means `resources`.
6. **Eval identity keyed on the upstream run ID (attempt), not fingerprint.**
   A rerun of training must invalidate its evals.
7. **Wrapper owns the W&B run; training attaches through the environment;
   eval scripts are W&B-free.** All W&B plumbing is written once. Training
   scripts must not set run identity (D2).
8. **Wrapper logs the model artifact;** `WANDB_LOG_MODEL` **is not used.** HF's
   callback names artifacts after `run_name`, which collides, and ties the
   contract to HF. The wrapper's artifact gets uniform metadata.
9. **Reconcile on explicit invocation; no auto-retry loop.** A deterministic
   failure would otherwise spin. Resubmission of failed attempts happens only
   when a human or agent runs `orch run`, and is always printed.
10. **Strict clean, pushed tree; no rsync escape hatch.** Reproducibility
    through SHA + logged params is the point. An `--allow-dirty` rsync mode was
    considered and removed deliberately during design; do not reintroduce it
    without revisiting this entry. `--dry-run` skips the check: it does not
    stage or record a SHA, and identity does not depend on one.
11. **paramiko over the** `ssh` **binary.** Password auth and connection reuse;
    `~/.ssh/config` is still honoured.
12. **Per-SHA worktrees with one shared venv.** A running job must not see its
    source change; duplicating torch/vllm per worktree is too expensive. The
    shared venv remains the one mutable hazard, made explicit via `--resync`.
13. **Operator config (`orch.local.yaml`) separate from codebase config
    (`orch.yaml`).** Paths, SSH aliases, and the W&B entity are properties of
    who runs it, not of the code; W&B records them per run anyway.
14. **No W&B** `group` **by default; tags instead.** Group is single-valued and
    assigned at creation; experiments are many-to-many and often known only
    afterwards. Fingerprints make an experiment spec a query, not a label.
15. **Scheduler holds plus an in-wrapper upstream check.** `-hold_jid`
    releases on any exit; correctness lives in the wrapper.
16. `hf:` **references are not auto-pinned to a commit.** Pinning requires
    network at resolve time and would surprise users who write `hf:<repo>`.
    Explicit `@rev` is honoured; the resolved commit is recorded as metadata.
17. **Codebase-specific first; generic seam kept narrow.** Only `orch.yaml`
    entries and `orch_adapters.py` know about this codebase. Extraction into a
    general tool is possible later but is not a v0 goal.
18. **Data stays in git.** 346 MB, already tracked; dataset artifacts are
    deferred.
19. `steps_per_generation` **is a runtime key for this codebase.** TRL
    regenerates every `steps_per_generation × num_iterations` micro-steps and
    slices the generation batch accordingly; for values dividing
    `gradient_accumulation_steps` no weight update occurs between slices, so
    only generation chunking changes. Values above `gradient_accumulation_steps`
    would change on-policyness and must not be used as runtime overrides.

v0.2 (2026-10-12):

20. **"Cluster" stays the operator-facing word; "backend" is the driver.** A
    cluster is a named entry in `orch.local.yaml`; its backend is `sge`,
    `slurm`, or `local`. The orchestrator machine is a cluster called `local`
    with backend `local`. The word is slightly odd for one workstation and
    was kept anyway because every command, flag, tag, and config key already
    says cluster and the meaning is clear. Revisit only if a third kind of
    target (a cloud API, say) makes "cluster" actively misleading.
21. **Canonical resource keys, mapped per cluster.** `gpus`, `gpu_type`,
    `time`, `mem`, `device`. v0's free-form resources leaked SGE's `h_rt` into
    every spec; a spec must be submittable to any cluster. The header
    template remains the mapping, and the v0 rule that every key must be
    referenced by the header is what makes an unsupported key loud. Revisit
    if a cluster needs a key that has no cluster-neutral meaning.
22. **The local backend is tmux plus two files, not a daemon.** Holds and
    device waits are loops in `job.sh`; state is a file per attempt and a
    claim file per device; `status_map` reads them. A daemon would have to be
    kept alive, upgraded, and debugged; `job.sh` already runs for the life of
    the job and dies with it. tmux was chosen over `nohup`/`setsid` because
    the operator already runs jobs in tmux and can attach. Revisit if local
    jobs ever need preemption or priorities.
23. **No submit-time admission check on `local`; jobs wait for a device.**
    The alternative was to refuse `orch run` when more jobs would start than
    devices are free. That check would exist for one backend only and would
    diverge from the cluster model, where a submitted job waits until it is
    scheduled. Waiting in the tmux session is that model. Exiting instead
    would burn an attempt number per full machine. Operator decision.
24. **`local` `repo_dir` is a separate clone and `orch run --cluster local`
    requires a pushed HEAD, like any cluster.** Staging into the developer's
    checkout would share its `.venv` and let an edit change code under a
    running job. Fetching from the local checkout instead of `origin` would
    let `git.sha` on W&B name a commit nobody can fetch. The cost is one more
    venv on `/localdisk`. Revisit if disk becomes the constraint.
25. **Scheduler state is fetched lazily per cluster and keyed by the cache's
    `cluster` field.** Querying every configured cluster on every `orch run`
    would open SSH connections nobody needs; querying only the target cluster
    would misread an attempt pending elsewhere as burned and resubmit it.
    Lazy fetch touches exactly the clusters that hold cached attempts of the
    fingerprints being planned. Attempts that are not in the cache and not in
    W&B are invisible on every design; §15 records the residual.
26. `.orchignore` **is a bounded revision of decision 10, coupled to sparse
    checkout.** Decision 10 refused any escape from the clean-tree rule to
    keep "the recorded SHA is what ran" exact. `.orchignore` keeps that exact
    for every path the cluster has: a path may be dirty only if the worktree
    excludes it, and both facts come from one file interpreted by git with one
    syntax. The protected set stops the file from removing what staging
    itself needs. There is still no rsync and no `--allow-dirty`.
27. **Isolation between concurrent jobs is expressed in entry `env`, not in
    the orch layer.** v0 briefly set `HF_METRICS_CACHE` in `render.py`; that
    bakes a HuggingFace `evaluate` quirk into a layer that is supposed to know
    nothing about the scripts (decision 17). Orch's contribution is the
    private run directory and the `{run_dir}`-family placeholders in env
    values; which caches a script shares is the entry's knowledge. `env` on
    specs and experiment runs exists for the same reason and is not
    identity-bearing, like `resources`.
28. **Two-phase train wrapper.** The wrapper initialises the run, writes the
    orch config block with `orch_state=running`, and finishes it before the
    script starts; the trainer resumes the run; the wrapper resumes it again
    for the epilogue. Each phase has exactly one writer, which keeps
    decision 7. The run reads `finished` for the seconds between prologue and
    the trainer's init; accepted over the alternative of holding the run open
    across two processes, which W&B does not support. Side effect: the
    deleted-id collision bump now happens before the script runs.
29. **`config.upstream` is a nested copy, not a join and not a merge.** W&B
    has no joins, so the eval run must carry what it wants to be grouped by.
    Train and eval params share key names, so the copy is nested. It is a
    snapshot at eval time; params do not change after the fact. The flat
    `orch_upstream_run_id` remains the programmatic join key.
30. **`status` derives from the run id and W&B before the cache.** v0's
    `status` took name, kind, fingerprint, and scheduler id from the cache
    only, and listed planner output for spec targets; both produced blank
    rows. The fingerprint is a substring of the run id; everything else is
    on the W&B run once the wrapper has started, which with decision 28 is
    before the script. The cache is now only for jobs that have not started.
    With no arguments `status` shows active jobs on every cluster, because
    that is the question a second cluster makes urgent.
31. **Downstream cancellation stays deferred.** Cancelling a failed train
    job's held evals from the wrapper needs a downstream ledger written after
    the children are submitted and a backend `cancel` that works from a
    compute node, which is a new cluster invariant. Not worth it for a queue
    wait. Slurm's `afterok` gives most of the benefit there for free
    (decision 32).
32. **Slurm holds use `afterok` with `--kill-on-invalid-dep=yes`.** SGE's
    `-hold_jid` releases on any exit and the wrapper decides; Slurm can do
    better at no cost by cancelling the held eval when the train job exits
    non-zero. The wrapper's check remains authoritative because exit 0 with
    no model directory is still a failure. The two backends therefore differ
    in how fast a doomed eval disappears, not in any outcome.
33. **Scheduler-neutral names.** `scheduler.out`/`scheduler.err`, a
    per-backend `job_id_env` in `job.json`, `orch_cluster` as a flat key. Old
    names are accepted on read so v0 runs stay inspectable.
34. **`orch status` reads the run name from `run.name`.** `config.orch` does
    not store it; `WANDB_NAME` is the run's name. Kind, cluster, and scheduler
    id still come from `config.orch`, which the wrapper writes before the
    script starts (decision 28).
35. **The local GPU check runs `claim` from the orchestrator checkout.** The
    clone's installed package is whatever was last pushed, which may not yet
    contain `orch.backends.local`. `cluster check local --gpu` therefore
    `cd`s to the checkout `orch` was launched from before `python -m
    orch.backends.local claim`. Real jobs `cd` to their worktree instead.
    The staging script is a quoted heredoc rather than `bash -c '...'`, so
    the sparse-checkout patterns can be shell-quoted. Same steps as §6.3.

---

## 15. Risks

- **Environment brittleness on the cluster** (modules vs wheel CUDA, uv on
  login nodes, quotas). Mitigation: `cluster check [--gpu]`, worktrees,
  lock-hash-gated sync, quota-sensitive paths in cluster `env`.
- **Shared venv mutation under running jobs.** Mitigation: `--resync`
  refusal; rare in practice because `uv.lock` changes rarely.
- **Scratch purge** on eddie. Mitigation by design: nothing in `work_dir` is
  needed after a job ends; the model cache is rebuilt on demand.
- **W&B storage.** ~3 GB per 1.5B model artifact; a sweep of nine runs is
  ~27 GB. Only final models are logged. Consider artifact TTLs on sweep
  fingerprints later.
- **W&B API behaviour**: nested-config filters are unreliable, hence the flat
  `orch_*` keys; deleted run IDs cannot be reused, hence the attempt bump;
  `resume="allow"` semantics across wandb versions should be pinned by test.
- **Heartbeat lag.** A killed job shows as `running` for several minutes;
  the "running + scheduler gone" rule covers it but depends on the cache.
- **Grid Engine** `Eqw` **state** (job in error). The backend reports `error`;
  `orch status` shows it; `orch cancel` clears it. Not auto-resubmitted.
- **Long queue times** make the feedback loop slow. Mitigation: `--dry-run`,
  `inspect`, `-p max_steps=5` smoke runs, `cluster check --gpu`.
- **Library drift**: HF `WandbCallback` behaviour around `run_name`/env, TRL's
  experimental SDFT API, vllm/torch pins. Pin via `uv.lock`; the M2 and M5
  verifications catch regressions.
- **paramiko and 2FA.** If eddie ever requires keyboard-interactive 2FA,
  `ssh.py` needs an interactive handler. Not expected now.
- **Misclassified runtime keys.** The deny-list default and the
  `[identity]`/`[runtime]` annotations reduce but cannot eliminate this;
  `config.params` records every value so suspect runs can be found
  afterwards.

v0.2:

- **tmux server death kills every local job.** There is no systemd on the
  workstation, so sessions survive logouts, but `tmux kill-server` or a
  machine reboot ends all of them at once; W&B flips them to `crashed` after
  the heartbeat timeout and the next `orch run` resubmits. A9 names the
  assumption; `cluster check local` cannot test it.
- **Device detection races.** `claim` treats a device as busy if an orch claim
  is live or `nvidia-smi` lists a process on it. A process launched by hand a
  moment before a claim may not yet appear in `nvidia-smi`, and a claimed
  job's own process appears only after it loads CUDA. The claim file is the
  primary guard between orch jobs; `nvidia-smi` is a courtesy toward the
  operator's manual work. Two jobs on one device is an OOM, not corruption.
- **Stale claims after a crash.** A claim whose session is dead is ignored by
  the next `claim`, so no cleanup is needed; a claim file whose session name
  was reused (same run id resubmitted after a lost cache) would be honoured
  until that session ends. Run ids are not reused by design, so this needs a
  lost cache plus a re-minted attempt.
- **Two orchestrator machines sharing one `local` cluster** is unsupported:
  the claims and the tmux server are per machine. Out of scope (§1.3).
- **Cross-cluster re-minting with a lost cache.** The run-directory ledger is
  per cluster. If the cache is lost while an attempt is queued on cluster A
  and the same fingerprint is planned on cluster B, B does not see A's
  directory and may mint the same attempt number. Both wrappers would attach
  to one W&B run with `resume=allow`. The cache is lost rarely and the window
  is "queued, not yet started"; `orch status --all` before planning on a new
  cluster after a cache loss avoids it. Not mitigated further.
- **Slurm dependency on a departed job.** Slurm forgets completed jobs after
  `MinJobAge` (minutes). Holds are only placed on jobs the cache shows as
  queued or running, so a hold on a departed id requires the state to change
  between `squeue` and `sbatch`; `--kill-on-invalid-dep=yes` then cancels the
  eval, which the next `orch run` re-plans.
- **Sparse checkout and tooling.** `uv sync --frozen` builds the project and
  reads `pyproject.toml` (which names `README.md`); the protected set covers
  what staging is known to need. A future tool that reads an excluded path on
  the cluster fails there with a missing file, which is the intended failure
  mode, and the fix is a shorter `.orchignore`.
- **Prologue flicker.** For the seconds between the train prologue's
  `finish()` and the trainer's `wandb.init`, the run reads `finished`. A
  reconcile that happens to run in that window would read `finished` for an
  attempt that is about to resume. The window is seconds and the attempt is
  in the cache as `running` on the scheduler; the table row "finished, any"
  would skip it, which is the harmless direction.
- **Three backends, one tested here.** Only `local` can be verified on this
  machine; `sge` keeps its golden tests from v0; `slurm` is verified by unit
  tests on recorded output until the first `cluster check dmog --gpu`.

---

## 16. Deferred (v0.3 and later), with intended shape

- `orch invalidate <run_id|fingerprint> --reason`: tag `invalid`; reconcile
  treats an invalid current attempt as failed.
- **Strict** `identity_keys`: optional per params file; when present, every
  key must appear in exactly one list.
- `revision` **convention**: a `revision: N` key in base params files to force
  new identities for everything derived from them.
- **Downstream cancellation.** `submit_jobs` writes `downstream.json`
  (`[{run_id, scheduler_id}]`) into the parent's run directory after the
  children are submitted, appending when a later `orch run` adds evals to a
  still-running train; `CacheEntry` gains `upstream_run_id`; the train
  wrapper calls `backend.cancel` for each downstream job when the attempt
  fails; `orch cancel <train>` cascades. Requires a new invariant: `cancel`
  works from a compute node (`qdel`, `scancel`, `tmux kill-session`).
- **Multi-GPU local jobs**: `gpus: n` claims `n` devices and exports them
  comma-separated.
- **Local jobs from unpushed commits**: fetch the SHA from the developer's
  checkout path instead of `origin`, recorded as such on the run. Decision 24
  says why not yet.
- `report`: metric selection config, joined tables; figures as a separate
  layer.
- **Sweeps**: a `grid:` block in experiment specs; best-by-validation
  promotion as an alias on the winning artifact.
- `--watch`: a reconcile loop with an attempt cap.
- **Worktree and model-cache garbage collection** (`orch cluster gc`),
  including stale `work_dir/.devices` claims and finished local run
  directories.
- **Dataset artifacts** for full lineage.
- `hf:` **revision auto-pinning** behind a flag.
- **An** `unset` **mechanism** for `extends`.
- **Dependency groups** so the orchestrator machine need not install torch
  and vllm.
- **MCP or HTTP surface** over the Python API.

---

## 17. Documentation deliverables (M8, extended in M15)

Written by the implementer, after the system exists, and checked against it:

1. `orch/README.md` answering, in this order: how to install and run the
   first `orch run --dry-run`; how to set up a cluster entry and its
   credentials and make `cluster check` pass, for each of eddie, dmog, and
   `local` (the separate clone, tmux, the device list); how to choose a
   cluster per run and what `orch status` shows across them; how to add a new
   training method (params file, spec, what `train.py` must do); how to add a
   new eval (script contract, parser, entry, spec, and when to give the entry
   an `env` for cache isolation); what `.orchignore` does and what it may not
   contain; what happens on failure and how to retry; where to find things in
   W&B (pointer to §7 with two or three concrete API snippets, including
   grouping evals by `upstream.params.*`); troubleshooting by symptom
   (`cluster check` failures, `Eqw`, "upstream not finished", dirty-tree
   refusal and `.orchignore`, `--resync`, a local job stuck in `queued`
   because no device is free).
2. `orch.local.example.yaml` with every field commented, three cluster
   entries.
3. **Example specs and params files** in `configs/`, `hparams/`,
   `experiments/`, each with a `description` and comments that explain
   non-obvious choices (notably every `runtime_keys` entry).
4. `--help` **text** for every command and flag, consistent with §8.
5. **This document**, updated where implementation forced a change, with the
   decision log extended accordingly.

Acceptance: a reader who has not seen this document or the design conversation
can, from the README alone, add a new eval suite and run it against an
existing fingerprint on eddie, and run the same suite on the workstation
with `--cluster local`.
