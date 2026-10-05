# `orch` — experiment orchestrator, v0 design

Status: design agreed 2026-10-02; implementation not started.
Audience: whoever implements v0 (human or agent) and, afterwards, anyone who
needs to know how runs, identities, and W&B records relate.

Sections 2–8 are normative: they define vocabulary, invariants, file formats,
semantics, flows, the W&B schema, and the CLI. The implementation must match
them, and they are intended to stand as the reference documentation afterwards.
Sections 9–17 are informative: layout, codebase changes, a worked example, the
implementation sequence, testing, the decision log, risks, deferred items, and
documentation deliverables.

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

### 1.1 In scope for v0

- One scheduler backend: Grid Engine, targeting the `eddie` cluster, driven
  over SSH from a single orchestrator machine.
- Run specs (one job each) and experiment specs (lists of run specs).
- Training jobs that produce a model artifact; evaluation jobs that consume a
  model reference (`hf:`, `run:`, `fingerprint:`, `artifact:`).
- Content-derived identity (fingerprint), attempt counters, idempotent
  reconcile-style submission.
- Code staging by git SHA with a shared `uv` environment.
- A per-job wrapper that owns the W&B run for both training and evaluation.
- CLI: `run` (with `--dry-run`, `--force`, `--no-evals`, `--resync`), `eval`,
  `status`, `logs`, `tag`, `cancel`, `inspect`, `cluster check [--gpu]`, and
  the internal `exec`.
- The codebase changes needed to satisfy the contracts: a unified `train.py`,
  small adjustments to the evaluation scripts, result parsers, and example
  configuration.

### 1.2 Out of scope for v0

Listed so they are not accidentally built: `plan` as a separate command;
`report` and any figure generation; hyperparameter sweeps and
best-of-sweep promotion; a watch/daemon loop; `invalidate`; strict
`identity_keys`; a second cluster or a local backend; automatic cluster
selection; dataset artifacts; automatic pinning of Hugging Face revisions;
worktree garbage collection; an MCP or HTTP API; a `dft` training method.
See §16 for the shape each of these is expected to take.

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
| **scheduler ID**        | What `qsub` returns for one attempt. Lives in the local cache and on the W&B run; never used as identity.                                                    |
| **job**                 | The fully resolved, submittable description of one attempt (`job.json`).                                                                                     |
| **cluster**             | A named entry in `orch.local.yaml`: SSH target, paths, scheduler header, setup, env.                                                                         |
| **wrapper**             | `orch exec`, which runs on the compute node around the actual script and owns the W&B run.                                                                   |
| **model reference**     | A string naming a model: `hf:<repo>[@rev]`, `run:<run_id>`, `fingerprint:<fp>`, `artifact:<entity>/<project>/<name>:<version>`.                              |
| **upstream**            | For an eval job, the training run whose model it evaluates.                                                                                                  |
| **reconcile**           | Deciding, per job, whether to submit, skip, or resubmit, by comparing the desired set of jobs with W&B and scheduler state.                                  |

---

## 3. Invariants and contracts

Everything in this section is a hard requirement. The orchestrator relies on
each one; `orch cluster check` tests group A and the testable parts of B.

### A. Cluster (per entry in `orch.local.yaml`)

- **A1.** The cluster is reachable over SSH from the orchestrator machine using
  either key-based or password authentication. The SSH alias in the cluster
  block is resolved through `~/.ssh/config`. A password, if used, comes from
  the environment variable named by `password_env` or an interactive prompt,
  never from a file.
- **A2.** `git`, `uv`, `qsub`, `qstat`, and `qdel` are on `PATH` in a _login_
  shell. All remote commands run as `bash -lc '<cmd>'`.
- **A3.** `repo_dir` is a git clone whose `origin` is the same remote the
  orchestrator machine pushes to, and the login node can fetch from it. The
  orchestrator owns this clone: it creates per-SHA worktrees under
  `repo_dir/.worktrees/` and manages `repo_dir/.venv`. Nothing else should
  modify it.
- **A4.** `work_dir` exists, is writable, and is on a filesystem visible to
  both login and compute nodes. Nothing in it is assumed to persist; everything
  of value is pushed to W&B before a job exits.
- **A5.** The login node can reach PyPI and GitHub. Compute nodes can reach
  `api.wandb.ai` and `huggingface.co`. W&B credentials are available to compute
  nodes via `~/.netrc` (from `wandb login` on the login node, home being
  shared) or a `WANDB_API_KEY` in the cluster `env`. Hugging Face credentials
  likewise if gated models are used.
- **A6.** The cluster's `header` template, rendered with a spec's `resources`,
  produces a job the scheduler accepts. The `setup` block yields a shell in
  which the venv's `torch` sees a GPU. The orchestrator runs both verbatim.
- **A7.** `env` in the cluster block is exported in every job. Quota-sensitive
  paths (`UV_CACHE_DIR`, `HF_HOME`, `WANDB_DIR` defaults) belong here.
- **A8.** Grid Engine exports `JOB_ID` into the job environment. The wrapper
  records it.

### B. Repository

- **B1.** The repository is a `uv` project with a committed `uv.lock`.
  `uv sync --frozen` yields an environment in which every entry's command and
  `python -m orch` run.
- **B2.** `orch run` and `orch eval` refuse to submit unless the working tree
  is clean and `HEAD` is contained in `origin`'s corresponding branch (or any
  `origin` ref). The SHA is recorded on every run. There is no override.
  `--dry-run` does not check B2: it does not stage code or record a SHA, and
  the fingerprint does not include one.
- **B3.** Entry commands are relative to the repository root and run with the
  working directory set to the job's source tree (a worktree at the recorded
  SHA).
- **B4.** Data the scripts need is tracked in the repository or referenced by
  paths in params. `orch` does not manage data in v0.
- **B5.** `orch.yaml` (entries) is committed. `orch.local.yaml` (W&B target,
  clusters) is gitignored; `orch.local.example.yaml` is committed.

### C. Configuration and identity

- **C1.** A params file is a YAML 1.2 mapping. `extends` and `runtime_keys`
  are the only reserved keys; both are removed before the script sees the file.
- **C2.** A spec's top-level keys are exactly: `kind`, `entry`, `extends`,
  `params`, `set`, `runtime_keys`, `resources`, `evals`, `name_keys`,
  `description`. Unknown keys are an error.
- **C3.** The script receives a fully resolved params file: `extends` chains
  applied, `set` and CLI overrides applied, reserved keys removed. It never
  reads the original.
- **C4.** A key is identity-bearing unless listed in `runtime_keys`. The
  fingerprint is `hash(kind, entry, params − runtime_keys, model_ref)`. Spec
  path, the `runtime_keys` list itself, `resources`, `evals`, `name_keys`,
  `description`, cluster, git SHA, and W&B target are excluded.
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
  `job.sh` setup failure) is visible only through the scheduler and `sge.err`
  (§6.7). In both cases the next reconcile treats the attempt as burned.
- **F2.** A successful training run has exactly one logged artifact of type
  `model`, named `model-{fingerprint}`, with aliases `a{attempt}` and `latest`.
- **F3.** Every run carries the configuration described in §7, including the
  flat `orch_*` keys used for filtering.

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
    env: { HF_ALLOW_CODE_EVAL: "1" }
```

Placeholders available in `cmd`:

| Placeholder    | Value                                            |
| -------------- | ------------------------------------------------ |
| `{params}`     | Absolute path of the resolved `params.yaml`      |
| `{flags}`      | Params rendered as CLI flags (§5.3)              |
| `{run_dir}`    | Absolute path of the attempt's run directory     |
| `{model_dir}`  | Train only: `$ORCH_MODEL_DIR`                    |
| `{model_path}` | Eval only: local directory of the resolved model |
| `{output_dir}` | Eval only: `$ORCH_OUTPUT_DIR`                    |

`parser` is `module:function`, importable from the source tree, with signature
`(output_dir: Path) -> dict[str, float]`. Entry `env` is merged over cluster
`env`.

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
      #$ -l h_rt={{ h_rt }}
      #$ -l h_rss={{ mem }}
    setup: |
      . /etc/profile.d/modules.sh
      module load cuda
```

`header` is a Jinja2 template rendered with the spec's `resources`.
`WANDB_ENTITY` and `WANDB_PROJECT` in the orchestrator's environment override
the `wandb` block. The API key is never in this file.

### 4.3 Run spec

```yaml
# configs/train/sdft_tooluse.yaml
kind: train
entry: train
description: SDFT on the tool-use task, paper defaults.
params: hparams/sdft_tooluse.yaml
resources: { gpus: 1, h_rt: "12:00:00", mem: 24G }
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
  resources.h_rt: "24:00:00"
```

```yaml
# configs/eval/tooluse.yaml
kind: eval
entry: tooluse_eval
runtime_keys: [gpu_memory_utilization]
params: { max_new_tokens: 1024, temperature: 0.0, gpu_memory_utilization: 0.8 }
resources: { gpus: 1, h_rt: "01:00:00", mem: 24G }
```

```yaml
# configs/eval/prior_tasks.yaml
kind: eval
entry: lm_eval
runtime_keys: [batch_size]
params:
  tasks: hellaswag,mmlu,truthfulqa,winogrande,humaneval,ifeval
  batch_size: auto
resources: { gpus: 1, h_rt: "06:00:00", mem: 32G }
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
| `resources`    | yes (unless inherited) | Free-form mapping interpreted by the cluster header template.                 |
| `evals`        | no; train only         | Eval specs to submit against this job's model. Default on.                    |
| `name_keys`    | no                     | `{alias: absolute.path}` rendered into the run name (§5.6).                   |
| `description`  | no                     | Free text, logged to `config.orch.description`.                               |

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
max_completion_length: 2048
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
- `null` is a value, not a deletion. There is no way to remove an inherited key
  in v0; set the code default explicitly instead.
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
appended). Spec-level `tags` apply to every job. Two entries that resolve to
the same fingerprint collapse to one job whose `evals` are the union.

### 4.6 Per-attempt files (`work_dir/<run_id>/`)

| File                       | Written by   | Purpose                                                                                   |
| -------------------------- | ------------ | ----------------------------------------------------------------------------------------- |
| `job.json`                 | orchestrator | The job: resolved spec, identity, W&B target, command, env, cluster, git. Schema in §6.1. |
| `params.yaml`              | orchestrator | What the script receives (C3).                                                            |
| `fingerprint.json`         | orchestrator | Exact canonical fingerprint input (C5).                                                   |
| `job.sh`                   | orchestrator | Scheduler script (§6.4).                                                                  |
| `sge.out`, `sge.err`       | scheduler    | Job-level stdout/stderr.                                                                  |
| `stdout.log`, `stderr.log` | wrapper      | The script's stdout/stderr, tee'd.                                                        |
| `model/`                   | train script | `$ORCH_MODEL_DIR`.                                                                        |
| `checkpoints/`             | train script | Intermediate checkpoints; not uploaded.                                                   |
| `output/`                  | eval script  | `$ORCH_OUTPUT_DIR`.                                                                       |
| `wandb/`                   | wandb        | `WANDB_DIR`; keeps W&B's local files off the home quota.                                  |

Shared across attempts: `work_dir/cache/models/` for downloaded model
artifacts, keyed by artifact digest, with a `.complete` marker and a lock file
for concurrent downloads. Hugging Face downloads use `HF_HOME`.

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
   eval spec a model reference was supplied; for a train spec none was.
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
`evals`, `runtime_keys`, `name_keys`; targeting `kind`, `entry`, or `extends`
is an error.

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
`fingerprint:` reference with no finished attempt; `artifact:` alias that does
not resolve.

---

## 6. Runtime flows

### 6.1 `orch run`

```
orch run <spec | experiment> [--cluster C] [--model REF] [-p/-r/--set ...]
         [--evals a.yaml,b.yaml | --no-evals] [--tag T]... [--group G]
         [--force] [--resync] [--dry-run] [--json]
```

1. **Preconditions.** Load `orch.yaml` and `orch.local.yaml`; select the
   cluster (`--cluster` or `default_cluster`). A real submit checks B2 before
   staging: `git status --porcelain` empty, and `git branch -r --contains HEAD`
   non-empty. Record `sha` and `remote`. `--dry-run` skips this check.
2. **Expand.** For a run spec: one job, plus one eval job per entry in `evals`
   (unless `--no-evals`; `--evals` replaces the list). For an experiment spec:
   each entry as above; collapse identical fingerprints and union their
   `evals`. Resolve each per §5.1 steps 1–8.
3. **Reconcile** (§6.2) each train job, then each eval job (which needs its
   upstream's run ID). Produce, per job: `submit` (with attempt number) or
   `skip` (with reason: finished, running, queued, upstream-not-finished-and-
   not-being-submitted).
4. **Dry run stops here** and prints the table: run ID, name, kind, decision,
   reason, fingerprint, and for each `submit` the resource line and the
   dependency. Exit 0.
5. **Stage code** (§6.3). Skipped if nothing is to be submitted.
6. **Render and write** per job to `work_dir/<run_id>/`: `job.json`,
   `params.yaml`, `fingerprint.json`, `job.sh`. Written over SFTP in one
   connection.
7. **Submit** train jobs first, then eval jobs with `-hold_jid <upstream
scheduler ID>`when the upstream was submitted in this invocation or is in the cache as queued/running. Record`scheduler_id` in the local cache.
8. **Report.** Print the table with scheduler IDs and W&B URLs
   (`https://wandb.ai/<entity>/<project>/runs/<run_id>`). `--json` emits the
   same as a list of objects.

`job.json` schema (version 1):

```json
{
  "schema_version": 1,
  "orch_version": "0.1.0",
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
    "description": ""
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
    "venv_python": ".../.venv/bin/python"
  },
  "git": {
    "sha": "...",
    "remote": "https://github.com/thomas-woehrle/sdft-exp.git"
  },
  "command": "python train.py --config /exports/.../3f9a2c1d7e4b-a1/params.yaml",
  "parser": null,
  "env": {
    "...": "merged cluster env, entry env, ORCH_* and WANDB_* variables"
  },
  "_note": "for eval jobs, command still contains the literal {model_path} placeholder; the wrapper substitutes it after resolving the model (§6.6 step 3)",
  "submitted_from": "w8724.see.ed.ac.uk",
  "submitted_at": "2026-10-02T13:00:00Z"
}
```

### 6.2 Reconcile

For a job with fingerprint `F`, determine the current attempt by probing W&B
for run IDs `F-a1`, `F-a2`, … until the first that does not exist (direct
`api.run()` lookups; deterministic, no filter semantics). Combine with the
local cache (§6.8) and, when not `--dry-run`, with `qstat` for cached
scheduler IDs.

| W&B state of highest attempt  | Scheduler state           | Decision                                      |
| ----------------------------- | ------------------------- | --------------------------------------------- |
| no attempts exist             | not in cache              | submit `a1`                                   |
| no attempts exist             | cache: queued/running     | skip (pending)                                |
| no attempts exist             | cache: gone               | submit `a2` (`a1` is considered burned)       |
| `running`                     | queued/running or unknown | skip (running)                                |
| `running`                     | gone                      | treat as crashed → submit next                |
| `finished`                    | any                       | skip (finished); with `--force` → submit next |
| `failed`, `crashed`, `killed` | any                       | submit next                                   |

Eval jobs additionally require an upstream: if the upstream is `finished`, or
is being submitted or is in flight in this invocation, the eval proceeds (with
a hold where applicable); otherwise it is skipped with reason
`upstream-not-available`.

Attempt numbers only increase and are never reused. Before submitting attempt
`n` (not in `--dry-run`), the orchestrator also checks whether
`work_dir/F-a{n}/` already exists on the cluster; if it does, that attempt was
consumed by an earlier submission the cache does not know about (lost cache,
another orchestrator machine), and `n` is incremented. The run directory thus
doubles as a ledger that survives cache loss. If `wandb.init` on the compute
node rejects a run ID as previously used (deleted run), the wrapper increments
the attempt, renames its run directory accordingly, and proceeds; the
orchestrator's next reconcile discovers the actual attempt by probing.

### 6.3 Code staging

One SSH connection per `orch run`. All commands via `bash -lc`.

```
cd {repo_dir}
flock .orch-staging.lock bash -c '
  git fetch --quiet origin
  git cat-file -e {sha}^{commit}                       # must exist on origin
  [ -d .worktrees/{sha} ] || git worktree add --detach .worktrees/{sha} {sha}
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
- One shared venv at `repo_dir/.venv`, selected via `UV_PROJECT_ENVIRONMENT`,
  because a per-worktree venv would duplicate torch and vllm.
- `uv sync --frozen` runs only when `uv.lock` changed. If it must run while
  the cache shows active jobs on this cluster, `orch run` refuses unless
  `--resync` is passed, and prints which jobs are active. This is the one
  remaining shared-mutable-state hazard and it is made loud rather than
  hidden.
- Staging output is streamed to the terminal; `uv sync` can take minutes on a
  cold cache. Paramiko keepalives are enabled.

### 6.4 Submission (Grid Engine)

`job.sh` is rendered from a built-in Jinja2 template:

```bash
#!/bin/bash
{{ header }}
#$ -N orch_{{ fingerprint }}_a{{ attempt }}
#$ -o {{ run_dir }}/sge.out
#$ -e {{ run_dir }}/sge.err
set -euo pipefail
{{ setup }}
{% for k, v in env.items() %}export {{ k }}={{ v | shquote }}
{% endfor %}
cd {{ src_dir }}
exec {{ venv_python }} -m orch exec {{ run_dir }}
```

Submission: `qsub -terse [-hold_jid <id>] {run_dir}/job.sh`; the single line
of output is the scheduler ID. `qstat -xml` is used for status. `-hold_jid`
releases when the referenced job _exits for any reason_; the correctness of
dependent evals therefore rests on the wrapper's upstream check (§6.6), not on
the scheduler.

Environment exported into every job (`env` in `job.json`), in increasing
precedence: cluster `env`, entry `env`, then:

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

### 6.5 `orch exec` — train

Runs on the compute node with `cwd = src_dir`.

1. Read `job.json`. Create `model/`, `checkpoints/`, `wandb/`.
2. Launch `command` as a subprocess with the environment above, tee'ing
   stdout/stderr to `stdout.log`/`stderr.log` and to the wrapper's own
   stdout/stderr (so `sge.out`/`sge.err` also contain them). Wait.
3. `wandb.init(id=run_id, entity, project, name=run_name, job_type="train",
tags, resume="allow")`.` allow` attaches to the run the trainer created, or
   creates it if the script died before initialising W&B.
4. `run.config.update({"params": spec.params, "orch": <§7 block>,
**flat orch_* keys}, allow_val_change=True)`.
5. If exit code is 0 and `model/config.json` exists and the directory
   contains at least one weights file: build artifact `model-{fingerprint}`
   (type `model`, metadata per §7), `add_dir(model/)`, `log_artifact(aliases=  [f"a{attempt}", "latest"])`, wait for upload. Set `orch_state=succeeded`.
   Otherwise: add tag `failed`, set `orch_state=failed`,
   `summary["orch/exit_code"]`, and if the exit code was 0 but D3 failed,
   record `orch/failure_reason="model_dir_missing"`.
6. `wandb.save` of `job.sh`, `job.json`, `params.yaml`, `fingerprint.json`,
   `stdout.log`, `stderr.log`, `sge.out`, `sge.err` with `policy="now"` and
   `base_path=run_dir`.
7. `run.finish(exit_code=<subprocess exit code>)`. Exit with the same code.

### 6.6 `orch exec` — eval

1. Read `job.json`. If `upstream_run_id` is set: fetch that run via the API.
   If its state is not `finished` or it has no `model` artifact, print
   `upstream <id> not finished (state=…); exiting without creating a run` and
   exit 0. No W&B run is created (F1 exception); the next `orch run` will plan
   a new eval against whichever attempt succeeds.
2. `wandb.init(id=run_id, …, job_type="eval", config={"params": …, "orch": …,
**flat})`.
3. Resolve the model reference to a local directory:

- `run:` → `api.run(id).logged_artifacts()` filtered to type `model`;
  `run.use_artifact(artifact)`; download into
  `work_dir/cache/models/<digest>/` under a lock, honouring a `.complete`
  marker.
- `artifact:` → `run.use_artifact(name:vN)`; same cache.
- `hf:` → `snapshot_download(repo, revision)`; record the resolved commit in
  `config.orch.model.resolved_revision`.
  Export `ORCH_MODEL_PATH`; substitute `{model_path}` in `command`.

4. Launch the subprocess as in §6.5 step 2.
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
| Job killed by scheduler (h_rt)                           | W&B run `crashed` after heartbeat timeout; `sge.err` says why | submits next attempt (after W&B flips; before that, "running + scheduler gone" rule applies) |
| `job.sh` fails before Python (module load, missing venv) | no W&B run; cache: scheduler gone                             | submits next attempt; `orch logs` shows `sge.err` from the cluster                           |
| Eval's upstream failed                                   | no W&B run for the eval; upstream `failed`                    | plans evals against the upstream's next attempt                                              |
| Deleted W&B run ID collides                              | wrapper bumps attempt and renames run dir                     | probing finds the real attempt                                                               |
| Staging fails (fetch, sync)                              | `orch run` aborts before writing any job                      | nothing to clean up                                                                          |

### 6.8 Local cache

`~/.local/state/orch/<entity>__<project>.json`:
`{run_id: {cluster, scheduler_id, fingerprint, attempt, kind, name, submitted_at}}`. It is a cache of scheduler handles only. Losing it means
`status` degrades to W&B-only for jobs that have not yet started, and that a
queued job may be re-planned as a new attempt once the run-directory ledger
check (§6.2) notices the gap. It never causes data loss.

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
    schema_version: 1
    orch_version: "0.1.0"
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
    cluster: { name, backend, repo_dir, work_dir, src_dir }
    scheduler_id: "4821937" # from $JOB_ID
    git: { sha, remote }
    submitted_from: "w8724.see.ed.ac.uk"
    submitted_at: "..."
    started_at: "..." # wrapper start
    host: "<compute node>"
```

Flat keys, for filtering (`api.runs(path, {"config.orch_fingerprint": ...})`):
`orch_fingerprint`, `orch_attempt`, `orch_kind`, `orch_entry`, `orch_spec`
(stem), `orch_model_ref`, `orch_upstream_run_id`, `orch_state`
(`succeeded`|`failed`).

HF's `WandbCallback` adds `TrainingArguments` and the model config at the top
level of `config` on training runs; `params` and `orch` are nested to avoid
collisions.

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
files; only the artifact.

### 7.6 Reading results without `orch`

Given a fingerprint `F`: runs `F-a1…F-an`, highest `finished` one is current.
Given an experiment: recompute fingerprints with `orch inspect` or filter by
`tags`. Join an eval to its training run through `orch_upstream_run_id`. The
method/task of a training run is in `config.params.method` and
`config.orch_spec`.

---

## 8. CLI reference

All commands accept `--json`. Exit codes: `0` success; `1` usage or validation
error (nothing submitted); `2` precondition failure (dirty tree, unpushed
HEAD, cluster check failed, `--resync` required); `3` remote failure during
staging or submission (partial submission possible; the report says what was
submitted).

| Command                                                | Purpose                                                                                                                                             |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `orch run <spec                                        | experiment> [--cluster C] [--model REF] [-p/-r/--set …] [--evals …                                                                               | --no-evals] [--tag T] [--group G] [--force] [--resync] [--dry-run]` | Resolve, reconcile, stage, submit. `--model` is required for an eval spec and forbidden for a train spec.                                             |
| `orch eval <eval-spec> --model REF [same flags]`       | Sugar for `orch run <eval-spec> --model REF`.                                                                                                       |
| `orch inspect <spec> [--model REF] [-p/-r/--set …]`    | Offline: print the resolved spec, fingerprint input, fingerprint, name, and each params override annotated `[identity]`/`[runtime]`.                |
| `orch status [<spec                                    | experiment                                                                                                                                          | run_id>…] [--no-remote]`                                            | Table of run ID, name, kind, attempt, W&B state, scheduler state, URL. With no arguments: everything in the local cache. `--no-remote` skips `qstat`. |
| `orch logs <run_id> [--follow] [--which stdout         | stderr                                                                                                                                              | sge.out                                                             | sge.err]`                                                                                                                                             | Finished run: download from W&B files. In flight: tail over SSH from the run directory. |
| `orch tag <run_id                                      | fingerprint                                                                                                                                         | spec                                                                | experiment>… --add T [--add T]… [--remove T]…`                                                                                                        | Edit tags via the API.                                                                  |
| `orch cancel <run_id>…`                                | `qdel` the scheduler ID; tag the W&B run `cancelled` if it exists.                                                                                  |
| `orch cluster check <name> [--gpu]`                    | Test invariants A1–A5, A7 (seconds). `--gpu` submits a short job that imports torch and vllm, asserts a visible GPU, and performs a W&B round-trip. |
| `orch exec <run_dir>`                                  | Internal: the wrapper.                                                                                                                              |

`--json` output for `run`/`status` is a list of objects with keys `run_id`,
`run_name`, `fingerprint`, `attempt`, `kind`, `decision`, `reason`,
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
  ssh.py               # paramiko connection: run(), put(), stream(); password handling
  staging.py           # §6.3
  render.py            # job.sh, job.json, params.yaml, fingerprint.json
  backends/
    base.py            # Backend protocol
    sge.py             # qsub/qstat/qdel
  exec/
    __init__.py        # entry point for `orch exec`
    common.py          # subprocess tee, W&B init/finish, file upload
    train.py           # §6.5
    eval.py            # §6.6
    models.py          # model reference → local path, cache, locks
  check.py             # orch cluster check

orch_adapters.py       # repo root; codebase-specific parsers referenced from orch.yaml
tests/
```

Interfaces:

```python
class Backend(Protocol):
    name: str
    def render_job_script(self, job: Job, cluster: Cluster) -> str: ...
    def submit(self, conn: SSH, run_dir: str, hold_on: list[str]) -> str: ...   # scheduler_id
    def status(self, conn: SSH, scheduler_id: str) -> Literal["queued", "running", "gone", "error"]: ...
    def cancel(self, conn: SSH, scheduler_id: str) -> None: ...

Parser = Callable[[Path], dict[str, float]]
```

Dependencies added to `pyproject.toml`: `ruamel.yaml`, `jinja2`, `paramiko`,
`typer`. `wandb` and `huggingface_hub` are already present. Add a build system
(`hatchling`, `packages = ["orch"]`) and `[project.scripts] orch = "orch.cli:main"` so `uv run orch …` works on the orchestrator machine; on the
cluster, jobs invoke `python -m orch` from the worktree and do not depend on
the editable install.

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
  scheduler state; eval upstream rules.
- `plan`: experiment expansion, dedup with union of evals, `--no-evals`,
  `--evals`.

### 13.2 Integration (fake backend)

A `FakeBackend` and in-memory SFTP that record what would have been written
and submitted. Drive `orch run` end to end against a recorded W&B fixture and
assert the submission order, holds, and file contents.

### 13.3 On eddie

The M4–M7 verifications above, kept as a checklist in `orch/README.md`. Use
`-p max_steps=5` for anything that is not the final M7 run.

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

---

## 16. Deferred (v0.1 and later), with intended shape

- `orch invalidate <run_id|fingerprint> --reason`: tag `invalid`; reconcile
  treats an invalid current attempt as failed.
- **Strict** `identity_keys`: optional per params file; when present, every
  key must appear in exactly one list.
- `revision` **convention**: a `revision: N` key in base params files to force
  new identities for everything derived from them.
- **Second backend (Slurm) and a** `local` **backend** for the 8-GPU box; with it,
  a canonical resource key set (`gpus`, `gpu_type`, `time`, `mem`) every
  header template must handle.
- `report`: metric selection config, joined tables; figures as a separate
  layer.
- **Sweeps**: a `grid:` block in experiment specs; best-by-validation
  promotion as an alias on the winning artifact.
- `--watch`: a reconcile loop with an attempt cap.
- **Worktree and model-cache garbage collection** (`orch cluster gc`).
- **Dataset artifacts** for full lineage.
- `hf:` **revision auto-pinning** behind a flag.
- **An** `unset` **mechanism** for `extends`.
- **Dependency groups** so the orchestrator machine need not install torch
  and vllm.
- **MCP or HTTP surface** over the Python API.

---

## 17. Documentation deliverables (M8)

Written by the implementer, after the system exists, and checked against it:

1. `orch/README.md` answering, in this order: how to install and run the
   first `orch run --dry-run`; how to set up a cluster entry and its
   credentials and make `cluster check` pass; how to add a new training method
   (params file, spec, what `train.py` must do); how to add a new eval (script
   contract, parser, entry, spec); what happens on failure and how to retry;
   where to find things in W&B (pointer to §7 with two or three concrete API
   snippets); troubleshooting by symptom (`cluster check` failures, `Eqw`,
   "upstream not finished", dirty-tree refusal, `--resync`).
2. `orch.local.example.yaml` with every field commented.
3. **Example specs and params files** in `configs/`, `hparams/`,
   `experiments/`, each with a `description` and comments that explain
   non-obvious choices (notably every `runtime_keys` entry).
4. `--help` **text** for every command and flag, consistent with §8.
5. **This document**, updated where implementation forced a change, with the
   decision log extended accordingly.

Acceptance: a reader who has not seen this document or the design conversation
can, from the README alone, add a new eval suite and run it against an
existing fingerprint on eddie.
