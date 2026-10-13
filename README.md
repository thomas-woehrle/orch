# orch

`orch` turns a YAML spec into a Weights & Biases run and, when you ask it to, a job on a cluster. A cluster is an entry in `orch.local.yaml`: eddie (Grid Engine), dmog (Slurm), or `local` (this machine). The fingerprint of the spec is the run id. W&B is the record of what ran; the cluster only executes it.

Invoke `orch` from the repository root: `uv run orch …` or `.venv/bin/orch …`.

## Install and a first dry run

```bash
uv sync
cp orch.local.example.yaml orch.local.yaml
```

Edit `orch.local.yaml`: W&B entity and project, the SSH alias, and the scratch paths. `WANDB_ENTITY` and `WANDB_PROJECT` in the environment override the file. The API key is never written there; `wandb login` on this machine supplies it.

Confirm a spec resolves:

```bash
uv run orch inspect configs/train/sdft_tooluse.yaml
```

That prints the resolved params, the fingerprint input, the fingerprint, and the run name. It does not submit anything. A `fingerprint:` model reference or an `artifact:` alias looks the canonical form up in W&B; `hf:` and `run:` do not.

A dry run resolves the experiment, asks W&B which attempts already exist, and prints what it would submit:

```bash
uv run orch run experiments/tooluse_main.yaml --dry-run
```

A dry run does not need a clean or pushed tree. A real `orch run` does: the working tree must be clean and `HEAD` must already be on `origin`, because that SHA is what gets staged and recorded. Paths listed in `.orchignore` (gitignore syntax) are the exception: they are left out of the cluster worktree, so they may be dirty or untracked. The file itself must be committed and clean, and it must not match `pyproject.toml`, `uv.lock`, `README.md`, `orch.yaml`, `orch_adapters.py`, `.orchignore`, or anything under `orch/`. `--json` prints the same rows as JSON. Exit codes: `0` ok, `1` a bad spec or flag, `2` the tree or the cluster is not ready, `3` a remote step failed after something may already have been submitted.

Choose a cluster with `--cluster`. The default is `default_cluster`.

```bash
uv run orch run experiments/tooluse_main.yaml --cluster local --dry-run
uv run orch status
uv run orch status --all
uv run orch status --cluster eddie
```

With no arguments, `orch status` lists jobs that are queued or running on every cluster in the local cache. `--all` lists the whole cache. `--no-remote` skips the scheduler query. A row's fingerprint is read from the run id. The name, kind, cluster, and scheduler id come from the W&B run once the wrapper has started, and from the cache only while the job is still queued.

## Set up eddie

1. Add an SSH alias named `eddie` (or whatever `ssh:` says) to `~/.ssh/config`, with the host, user, and key. If key auth is not available, export the variable named by `password_env` or let `orch` prompt.
2. On eddie's login node, run `wandb login` so `~/.netrc` exists on the shared home directory. Compute nodes read that file.
3. From this machine:

```bash
uv run orch cluster check eddie
uv run orch cluster check eddie --gpu
```

The first check confirms the scratch directories, `git`, `uv`, the origin URL, and the login environment. `--gpu` submits a short job that imports torch and vllm, checks CUDA, and writes a one-point W&B run.

The first real `orch run` creates `.worktrees/<sha>` and syncs one shared virtualenv from `uv.lock`. A later run with the same lockfile does not sync again. If the lockfile changed while jobs are still queued or running, `orch` refuses until you pass `--resync`.

## Set up dmog

Same shape as eddie, with `backend: slurm`. The example entry in `orch.local.example.yaml` has the GPU header (`--gres`, `--time`, `--mem`) and the flight setup line. Fill in the scratch paths, add an SSH alias `dmog`, clone the repo there, install `uv` on the login node, and run `wandb login` once. Then:

```bash
uv run orch cluster check dmog
uv run orch cluster check dmog --gpu
```

A held eval uses `--dependency=afterok`. If the training job exits non-zero, Slurm cancels the eval. The wrapper still checks that the training run finished and logged a model, because a training job can exit 0 without a model directory.

## Set up local

`local` runs jobs on this machine, in detached tmux sessions, one GPU each. `repo_dir` is a separate clone, not the checkout you edit. `devices` lists the GPU indexes jobs may claim.

```bash
git clone git@github.com:thomas-woehrle/sdft-exp.git /localdisk/home/s2865446/orch/sdft-exp
mkdir -p /localdisk/home/s2865446/orch/work
uv run orch cluster check local
uv run orch cluster check local --gpu
```

`orch run --cluster local` still requires a pushed commit: the clone fetches that SHA from origin. A job with no free GPU waits in its tmux session (`orch status` shows it as `queued`) until one is free, including one your own processes are using. `tmux attach -t orch_<fingerprint>_a<n>` shows the session. `orch cancel <run_id>` kills it. Evals wait until the training session exits, then claim a GPU. `time` and `mem` are accepted and not enforced. `gpus` must be 1. `-r device=3` pins a GPU.

Resources in every spec use the same keys: `gpus`, `time` (`HH:MM:SS`), `mem` (for example `24G`), and optional `gpu_type`. Each cluster's header maps them. `device` is local only.

## Add a training method

1. Add a params file under `hparams/`. `method` selects the trainer (`sdft` or `sft`). List technical knobs in `runtime_keys`; every other key is part of the fingerprint. Comment each runtime key.
2. Add `configs/train/<name>.yaml` with `kind: train`, `entry: train`, `params`, `resources`, and the `evals` that should follow the run. `name_keys` chooses which scalars appear in the W&B run name.
3. `train.py` already pops `method`, `model_name`, `dtype`, `train_data`, and `val_size`, then constructs `SDFTConfig` or `SFTConfig` with the remaining mapping. An unknown key fails the run. A new method needs a new branch there and a config class that rejects unknown keys. Do not set `run_name`; the wrapper's `WANDB_NAME` is the run name. The script must save the model and tokenizer to `$ORCH_MODEL_DIR`.
4. `uv run orch inspect configs/train/<name>.yaml`, then `uv run orch run configs/train/<name>.yaml --dry-run`.

`-p` sets any param. `max_steps` is identity-bearing, so `-p max_steps=5` gets its own fingerprint. `steps_per_generation` is a runtime key, so `-p steps_per_generation=16` keeps the fingerprint and is listed as `[runtime]`. `-r` sets a resource, for example `-r gpu_type=h200`.

## Add an eval

1. The script takes the model directory and an output directory, writes a JSON (or, for lm-eval, `results_*.json`) under that directory, and does not import `wandb`. `eval_tooluse.py` is the pattern: `--model_path`, `--output_dir`, plus flags that come from the spec's `params`.
2. Add a parser in `orch_adapters.py`. Signature: `(output_dir: Path) -> dict[str, float]`. Drop non-numeric fields. For lm-eval, a `,none` filter suffix is stripped and any other `metric,filter` pair is kept.
3. Register an entry in `orch.yaml`:

```yaml
my_eval:
  kind: eval
  cmd: "python eval_mine.py --model_path {model_path} --output_dir {output_dir} {flags}"
  parser: "orch_adapters:parse_mine"
```

`{model_path}` is filled in on the compute node. `{flags}` is the params mapping rendered as CLI flags.

If the script shares a cache with other jobs, point that cache at the attempt's directory from the entry `env`. Values may contain `{run_dir}`, `{params}`, `{output_dir}` (eval), and `{model_dir}` (train). The `lm_eval` entry sets `HF_METRICS_CACHE` this way, because HumanEval locks a file under the shared Hugging Face cache. A spec or an experiment run can set `env` too; it is not part of the fingerprint. `--set env.NAME=value` does the same from the command line.

4. Add `configs/eval/mine.yaml` with `kind: eval`, `entry: my_eval`, `params`, and `resources`. Put batch size, GPU memory fraction, and similar knobs in `runtime_keys` when they do not change the measured result.
5. Run it against a model that already finished. `fingerprint:` uses the highest finished attempt:

```bash
uv run orch eval configs/eval/mine.yaml --model fingerprint:<12 hex> --dry-run
uv run orch eval configs/eval/mine.yaml --model fingerprint:<12 hex>
```

`hf:Qwen/Qwen2.5-1.5B-Instruct`, `run:<run_id>`, and `artifact:model-<fp>:latest` are the other forms. Metrics land in the W&B summary as `<spec stem>/<key>`, for example `tooluse/accuracy`.

To attach the eval to a training spec, list it under that spec's `evals`. `orch run` submits the eval held on the training job (`-hold_jid` on eddie, `--dependency=afterok` on dmog, a wait loop on `local`). The hold releases when the training job exits. The eval's fingerprint includes the training run id, so a new training attempt gets new evals. An eval whose training run is in flight on a different cluster is skipped until that run has finished.

## When a run fails

`orch run` prints one row per job: submit, or skip with a reason (`finished`, `running`, `pending`, `failed` is not a skip — a failed attempt is submitted again as the next attempt). A failed training run does not get an artifact. `orch logs <run_id>` prints `stdout`, `stderr`, `scheduler.out`, or `scheduler.err` (`--which`; `sge.out` and `sge.err` still work). While the job has not finished, `--follow` tails the file on the cluster that ran it. The training wrapper writes `config.orch` and the flat `orch_*` keys before the script starts, with `orch_state: running`, so a run can be filtered while it trains.

Run the same command again to submit the next attempt. `--force` submits a new attempt even when the latest one finished. `orch cancel <run_id>` deletes the scheduler job on the cluster that owns it and tags the W&B run `cancelled`. `orch status` with no arguments shows what is queued or running. `orch status <spec>` lists attempts that already exist, not the attempt a new `orch run` would submit; use `--dry-run` for that.

An eval whose training run never reached `finished` exits without creating a W&B run. The next `orch run` plans that eval against the new attempt.

## Where the results are

Design notes for the W&B layout are in [docs/orchestrator.md](../docs/orchestrator.md) §7. The fields you filter and join on:

```python
import wandb

api = wandb.Api()
runs = api.runs("thomas-woehrle-team/orch-exp", {"config.orch_fingerprint": "<12 hex>"})
run = api.run("thomas-woehrle-team/orch-exp/<fp>-a1")
run.summary["tooluse/accuracy"]
run.config["params"]["method"]
run.config["orch_upstream_run_id"]  # eval → the training run id
run.config["upstream"]["params"]["learning_rate"]  # copy of the producing run

artifact = api.artifact("thomas-woehrle-team/orch-exp/model-<fp>:latest")
```

Flat keys for filters: `orch_fingerprint`, `orch_attempt`, `orch_kind`, `orch_spec`, `orch_cluster`, `orch_model_ref`, `orch_upstream_run_id`, `orch_state` (`running`, `succeeded`, or `failed`). Training logs one artifact, `model-<fingerprint>`, with aliases `a<n>` and `latest`.

`config.upstream` is on eval runs whose model came from a training run. In the W&B table it shows up as dotted columns, so you can group evals by `upstream.params.learning_rate` or `upstream.params.method`. Base-model evals (`hf:`) have `upstream` null. It is a copy taken when the eval starts, not a join: train and eval params share names like `temperature`, and merging them would collide.

## Troubleshooting

- **`cluster check` fails.** Read the line it prints. Typical causes: the SSH alias does not resolve, scratch paths are missing or not writable, `git` or `uv` is absent on the login node, or the remote `origin` URL is a different repository. `--gpu` fails separately when the queue, CUDA, or `wandb login` on eddie is the problem.
- **`Eqw`.** The job is in an error state before it starts (bad directive, queue, or memory request). `orch status` shows `error`. `orch cancel <run_id>` clears it. Fix the header or the resources, then run again. `Eqw` is not resubmitted automatically.
- **`upstream not finished`.** The eval started after its training job had already left the queue without a `finished` W&B run. No eval run is created. Check the training logs, then `orch run` the spec again so the eval is planned against the next attempt.
- **Dirty tree or unpushed HEAD (exit 2).** Commit and push before a real `orch run`. Paths in `.orchignore` may stay dirty; anything else, including `.orchignore` itself, may not. There is no flag that skips this check on submit. `--dry-run` does not apply it.
- **Local job stays `queued`.** It is waiting for its training session, or for a GPU. `nvidia-smi` shows what is busy. `orch cancel <run_id>` gives up the wait.
- **`--resync` required (exit 2).** `uv.lock` changed and the cluster still has queued or running jobs from this project. Wait for them, or pass `--resync` if you accept that the shared virtualenv will change underneath those jobs.

## Checklist on eddie

Use `-p max_steps=5` except for the last item.

- [ ] `orch cluster check eddie` passes with key or password auth.
- [ ] `orch cluster check eddie --gpu` reports a W&B round-trip from a compute node.
- [ ] A second `orch run` does not sync the virtualenv again.
- [ ] `orch run configs/train/sft_tooluse.yaml -p max_steps=5 --no-evals` finishes with the expected name, config, files, and a `model-<fp>` artifact aliased `a1` and `latest`.
- [ ] The same command plus `-p not_a_real_key=1` finishes as `failed`, with logs and no artifact. Running it again plans attempt 2.
- [ ] `orch eval configs/eval/tooluse.yaml --model hf:Qwen/Qwen2.5-1.5B-Instruct` records `tooluse/accuracy`.
- [ ] `orch run configs/train/sft_tooluse.yaml -p max_steps=5` submits the training job and two held evals, and the evals record lineage to the model artifact.
- [ ] `orch cancel` on that training job makes both evals exit without creating runs. The next `orch run` plans them against attempt 2.
- [ ] `orch run experiments/tooluse_main.yaml` runs to completion. A second invocation submits nothing. `--force` on one spec submits attempt 2 and new evals.
- [ ] `orch status` and `orch logs` work for a queued job, a running job, and a finished job.
- [ ] `orch cluster check local` and `orch cluster check local --gpu` pass. `orch run --cluster local` of a five-step train submits a tmux session, holds its evals, and records `config.orch` while the train is still running.
- [ ] `orch cluster check dmog --gpu` passes, and a five-step train there holds its evals with `afterok`.
