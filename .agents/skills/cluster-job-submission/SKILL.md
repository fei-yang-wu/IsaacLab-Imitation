---
name: cluster-job-submission
description: Submit, plan, monitor, and cancel Slurm jobs on ICE or Skynet through this repo's control plane (python -m imitation_experiments.pipeline.cluster). Covers campaign.yaml authoring, the plan/submit/status/logs/cancel verbs, walltime-segmented chained runs, cluster storage and data-I/O rules, and which paths, resources, and behavior differ per cluster profile. Use when the user asks to submit a cluster job, plan a run, check job status, tail cluster logs, cancel a Slurm job, write or edit a campaign.yaml, chain or resume a long run, or mentions ICE, PACE, Skynet, apptainer, sbatch, storage quota, or the retired cluster_interface.sh.
---

# Cluster job submission

All cluster submission in this repo goes through one CLI:

```bash
pixi run python -m imitation_experiments.pipeline.cluster <plan|submit|status|logs|cancel> ...
```

`docker/cluster/cluster_interface.sh` and every `submit_job_slurm_*.sh` /
`submit_job_pbs.sh` are retired (2026-08-15) — running any of them just
prints a deprecation error and exits. If a script, doc, or memory tells you
to invoke them directly, it is stale; redirect to this CLI instead.

## Non-Negotiables

- `plan` never submits anything. It resolves config, runs a login-node
  preflight over ssh, and writes a frozen, sha-sealed plan — nothing more.
- Do not run `submit` until the user has seen the plan's preflight output and
  explicitly confirms the exact `submit --confirm <PLAN_SHA>` command.
- A campaign with no `campaign.yaml` has **no working submission path**.
  Writing one is real design work (declares arms, resources, dataset paths) —
  do not silently invent one; confirm the arm list and resources with the
  user first, or point them at
  `experiments/campaigns/2026-08-14-latent-quant-ice-repeats/campaign.yaml`
  as the reference shape.
- Only the `ice` profile has been validated against a real submission
  (2026-08-15, jobs 5577564/5577565, full pretrain→afterok→lowlevel chain).
  The `skynet` profile is transcribed from the retired `.env.skynet_runtime`
  and untested — treat its resource defaults as placeholders, not a budget,
  and say so if the user is about to spend real Skynet time on it.
- Never print token file contents (`hf_token_file`, `wandb_api_key_file`).

## The four verbs

```bash
# 1. Plan — resolves + preflights + freezes. Read the preflight table.
pixi run python -m imitation_experiments.pipeline.cluster plan \
    --campaign <path/to/campaign.yaml> --arm <arm> --seed <seed> \
    [--profile ice|skynet] [--set vars.key=value ...] [--only-stage <name>] [--skip-preflight]

# 2. Submit — only after the user confirms the printed PLAN_SHA.
pixi run python -m imitation_experiments.pipeline.cluster submit \
    --plan <plan_dir> --confirm <PLAN_SHA>

# 3. Status — squeue + sacct reconciled in one call.
pixi run python -m imitation_experiments.pipeline.cluster status --submission <plan_dir>

# 4. Logs — tails the recorded Slurm log path, %j/%x already substituted.
pixi run python -m imitation_experiments.pipeline.cluster logs \
    --submission <plan_dir> --stage <name> [--follow]

# Cancel — requires --yes.
pixi run python -m imitation_experiments.pipeline.cluster cancel --submission <plan_dir> --yes
```

`--profile` defaults to the `profile:` field inside `campaign.yaml`; pass it
explicitly only to override.

`plan` writes to `logs/cluster_control/<campaign>/<plan_id>/` locally
(`plan.json`, one `batch_<stage>.sh` and `job_env.<stage>.resolved.sh` per
stage). `submit` mirrors a `submission-*.json` there and on the remote
control root. `status`/`logs`/`cancel` read the submission record — pass
`--submission <plan_dir>` explicitly, or omit it to auto-discover the newest
one (optionally filtered with `--campaign <name>`).

## Writing a campaign.yaml

Read `experiments/campaigns/2026-08-14-latent-quant-ice-repeats/campaign.yaml`
first — it is the reference shape (14 arms, shared blocks via YAML anchors,
`mul`/`ceil_div`/`floor_div`/`concat` OmegaConf resolvers registered in
`imitation_experiments.pipeline.cluster.config`). Structure:

```yaml
name: <campaign-name>
profile: ice                 # or skynet
wandb_project: ...
wandb_group: ...
vars: {...}                  # campaign-wide values, ${vars.arm}/${vars.seed} always available
preflight:
  require_container_paths: [/data/..., /storage/ice-shared/...]   # checked before any sbatch
  output_container_path: /data/.../${vars.arm}_seed${vars.seed}
shared_env: {...}            # frozen env vars added to every stage
arms:
  <arm-name>:
    vars: {...}              # arm-local overrides, merged over campaign vars
    stages:
      - name: pretrain
        executable: scripts/rlopt/train_hl_skill_diffsr.py
        args: [...]
        time_limit: "15:59:00"   # see cluster caps below
      - name: lowlevel
        executable: scripts/rlopt/train.py
        args: [...]
        depends_on: pretrain    # becomes an afterok dependency at submit time
```

Paths inside `args`/`preflight` must be container-visible absolute paths
(`/data/...` or an `extra_bind_paths` entry) — `preflight.py`'s
`container_to_remote` maps these to the real remote path per profile and
fails the plan if a path isn't visible under the job's binds. This is what
catches "manifest missing" and node-local-tmp failures before `sbatch`, not
after.

Every configurable resource value should be a named `vars` field with a
sane default, overridable via `--set vars.key=value` — never hand-edit the
YAML for a one-off run.

## What's cluster-dependent (source of truth: `pipeline/cluster/conf/profile_<name>.yaml`)

| | `ice` (validated) | `skynet` (EXPERIMENTAL) |
|---|---|---|
| ssh alias | `ice` | `skynet` |
| `data_dir` (binds to `/data`) | `/home/hice1/fwu91/scratch/Research/IsaacLab/data` | `/coc/flash12/fwu91/Research/IsaacLab/data` |
| `shared_sif_path` | `.../isaaclabsif/isaaclab-runtime-3.0.0b2-cu130.sif` under the path above | same filename under the Skynet path above |
| `extra_bind_paths` | `/storage/ice-shared/vip-vwt` (shared dataset allocation) | none |
| default `slurm.gres` | `gpu:h100:1` | `gpu:a40:1` |
| default `slurm.partition`/`qos` | `ice-gpu` / `coe-ice` | `wu-lab` / `short` |
| default `slurm.time_limit` | `15:59:00` | `00:30:00` — a smoke-test placeholder, **not a real training budget**; override per stage |
| **walltime hard cap** | **16h.** `sbatch` rejects anything over this with "Requested time limit is invalid" — found live 2026-08-15 (job chain 5577560), the 23:59:00 default in the old `run.sh` had the same bug | not yet characterized on the new path; do not assume Skynet's old 2-day precedent still holds |
| `min_free_gb` quota check | 50 GB min-free gate (ICE has a 300 GB **hard** home quota — TIMEOUT wipes node-local output, so checkpoints must land under `/data`) | 50 GB min-free gate, quota ceiling itself unverified for the new path |
| token files | `/home/hice1/fwu91/.hf_token`, `/home/hice1/fwu91/.wandb_api_key` | `/nethome/fwu91/.hf_token`, `/nethome/fwu91/.wandb_api_key` |

Everything else — the CLI verbs, `campaign.yaml` schema, preflight checks,
plan/submit semantics, `afterok` chaining — is cluster-agnostic; only the
resolved values under `env:`/`slurm:` in the profile YAML change per cluster.
When adding a new cluster, copy `profile_ice.yaml` as the template and expect
to validate it with a real smoke submission before trusting it (see below).

## Chained runs and walltime

**Never shrink a run's frame budget or `max_iterations` to fit a walltime.**
A shrunken per-segment budget is an agent-invented training parameter. The
submitted config states the real experimental budget; the scheduler, not the
config, decides where the segments break.

Every segment of a chained run carries the **full** frame target:

- `slurm.py` emits `#SBATCH --signal=TERM@300`. `train_impl.py` routes SIGTERM
  through the SIGINT handler, and `ppo.py train()` writes a final resume
  checkpoint at the current global step before it re-raises. The re-raise
  keeps the interrupted run's exit code nonzero on purpose.
- Checkpoints carry `cumulative_env_frames` (global, across segments).
  `init_metadata` seeds `frames_processed` from it, trims the remaining
  budget, and offsets the log and save cadence. Checkpoint filenames therefore
  carry global steps. Checkpoints from before 2026-08-16 lack the key and get
  no filename fallback, because their steps are segment-local.
- Chain segments with `dependency_kind: afterany` — a TIMEOUT predecessor must
  still release its successor. Keep `pretrain -> lowlevel1` on `afterok`.
- A segment that finds the budget already complete runs 0 iterations and exits
  cleanly.
- `--only-stage` accepts a comma-separated stage list, so a lowlevel chain can
  run against an encoder already on disk.
- An environment-side step counter (for example a curriculum ramp) stays
  **segment-local**. Only the trainer's frame budget is global. Pin a ramp
  length that completes inside segment 1.

W&B continuity: set `WANDB_RUN_ID: <campaign>-<arm>-s<seed>` on every chained
stage and use one segment-less `exp_name`, with the segment identity in the
tags. `logging_utils._build_metrics_logger` reads `WANDB_RUN_ID` and
`WANDB_RESUME` (default `allow`) and pins `exp_name`, so the resumed run keeps
its name and logs at the global frame step. Interactive runs with no
`WANDB_RUN_ID` are unaffected. Chains submitted before 2026-08-16 still emit
one run per segment starting at 0; stitch them by arm name.

## Storage and data I/O

- **Write checkpoints to persistent storage, never node-local disk.** On ICE a
  Slurm TIMEOUT is a hard SIGKILL: the job dies before the log sync-back runs
  and the epilog wipes `/tmp`. Three full-walltime jobs once produced zero
  retained checkpoints — about 48 GPU-hours lost. Send output under the
  `/data` bind (`agent.logger.log_dir=/data/...`, `--pretrain-output-dir
  /data/...`); the campaign's `preflight.output_container_path` declares it.
- **ICE has a 300 GB hard per-user quota** on `/storage/ice1`
  (`lfs quota -uh $USER /storage/ice1`). The filesystem itself has petabytes
  free, so a "disk full" error there is always the quota. It fails silently in
  two ways: `hf download` dies with `Disk quota exceeded (os error 122)` and
  simply stops (watch byte deltas, not process liveness), and a training job
  fails its checkpoint saves mid-run. `lfs quota` lags a deletion by a few
  seconds — re-query before you conclude a prune did nothing. The cheapest
  space is intermediate checkpoints: keep only the highest `model_step_*.pt`
  per finished run directory (one prune freed 83.9 GiB), after `squeue -u
  $USER` proves no run still owns them.
- **The workspace archive ships SOURCE only, and it is content addressed.**
  Every heavy dependency — PyTorch, Isaac Sim, Isaac Lab — already lives in the
  container image's `container-runtime` Pixi environment, so nothing built
  belongs in the archive. `submit` packs the repo (~21 MB, ~1 s) and publishes
  it to `<control_root>/workspaces/<sha256>.tar.gz`; each plan directory gets a
  `workspace.tar.gz` symlink into that store, so N arms built from one tree
  upload once and store one copy. A re-submit of an unchanged tree skips the
  upload entirely (`workspace already on the cluster, reused`).
  `build_workspace_archive` fails the submit above `ARCHIVE_MAX_BYTES`
  (400 MB) and prints the heaviest paths — that guard exists because
  `external/Embodied-Control/.pixi` (7.4 GB of built environments) rode along
  in every submission until 2026-08-17, at 3.2 GB compressed per plan.
- **Pruning the workspace store.** Store entries are shared, so never delete
  one while any plan that links to it has a queued or running segment. Delete
  entries no plan directory points at:
  `find <control_root>/workspaces -name '*.tar.gz' | while read f; do grep -qrl "$(basename "$f")" <control_root>/plans/*/workspace.tar.gz 2>/dev/null || echo "$f"; done`
  — check `squeue -u $USER` first. Legacy plan directories that still hold a
  real 3.2 GB `workspace.tar.gz` (pre-2026-08-17 submissions) are the bigger
  win: delete the file only when every job id in that plan's
  `submission-*.json` is absent from `squeue`, since each chained segment
  re-extracts the archive at start.
- **Never memory-map a reference dataset that lives on a network filesystem.**
  Mapping defers reads to per-step page faults, and a reference gather touches
  random rows every step, which is what a parallel filesystem is worst at.
  Measured on ICE with 129k reference arrays (49.4 GB), 16,384 envs, H200:
  ~48 fps mapped with three jobs cold-starting on one node, against ~70,971
  fps resident in host RAM. Pass
  `env.data.reference_arrays_resident=true` on any network-filesystem cluster.
  Local NVMe keeps the mapped default. The tell for this failure is a huge job
  log whose last real line is `Initialized LazyTensorStorage` many minutes
  earlier.
- A new job-environment knob only reaches the job when the control plane emits
  it into the frozen `job_env.<stage>.resolved.sh`. Confirm its effect in the
  job log before you trust it; two earlier plumbing "fixes" silently did
  nothing.
- Check `df -h /tmp` on the real node before you size anything node-local. The
  ICE H200 nodes had 200 GB of `/tmp`, not the multi-TB figure once recorded.

## Skynet profile notes

The `skynet` profile is EXPERIMENTAL — see the non-negotiables above. These
facts come from the retired bash workflow and still describe the cluster:

- Apptainer/Singularity exists on **compute nodes only**. Do not conclude it is
  missing because the login node lacks it. Only the wu-lab A40 nodes
  `dendrite` and `synapse`, plus a few L40S nodes such as `bishop`, have it —
  a job that lands anywhere else fails. Never widen the nodelist.
- QoS `short` caps at 2 days, `long` at 7 days; both override the partition's
  4 h MaxTime. The `wu-lab` partition is PriorityTier 2 and preempts overcap
  Tier-1 jobs with REQUEUE.
- Keep data, caches, SIFs, and logs under
  `/coc/flash12/fwu91/Research/IsaacLab`, not `/nethome/fwu91`.
- Prefix remote Slurm commands with
  `export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH;`.
- Compute-node `/tmp` is node-local, invisible from the login node, and
  collects orphaned `/tmp/isaaclab-<jobid>` directories when a job crashes.
  Synapse once reached 99% full with 89 orphans. Inspect and clean through
  `srun -w <node> -p wu-lab --qos=debug`.
- Never start two fresh submissions at the same moment: concurrent SIF
  extraction races corrupt both.
- Pixi (non-container) Isaac Sim jobs need `ACCEPT_EULA=Y`,
  `PRIVACY_CONSENT=Y`, `OMNI_KIT_ACCEPT_EULA=YES`. Set them as profile `env`
  keys in `profile_skynet.yaml`, never by hand at the call site.
- Docker exists on compute nodes but may deny `/var/run/docker.sock`.
- To test compute-node capabilities, submit a small diagnostic sbatch
  (partition `wu-lab`, qos `short`, `--gres=gpu:a40:1`, 5 minutes) that prints
  `hostname`, `PATH`, and `command -v` plus `--version` for `apptainer`,
  `singularity`, and `nvidia-smi`. Ask the user first — it is a submission.

## Before Submitting

1. `plan` the (arm, seed); read every `[PREFLIGHT]` line — do not proceed
   past a `FAIL`.
2. Present the exact `submit --confirm <PLAN_SHA>` command and wait for
   explicit user confirmation.
3. On `skynet` specifically, or for any profile/campaign combination not yet
   run for real: submit a cheap smoke first (small `--set` overrides on
   updates/frame budgets, an isolated `preflight.output_container_path`) and
   confirm it completes before spending real budget — this is exactly the
   ICE cutover procedure that found the 16h walltime cap.

## Monitoring

```bash
pixi run python -m imitation_experiments.pipeline.cluster status --submission <plan_dir>
pixi run python -m imitation_experiments.pipeline.cluster logs --submission <plan_dir> --stage <name> --follow
```

Raw fallback when the verbs don't expose what you need:

```bash
ssh <ice|skynet> 'squeue -u $USER -o "%i %T %M %j %R"'
ssh <ice|skynet> 'sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode -P'
```

(Skynet needs `export PATH=/opt/slurm/Ubuntu-20.04/24.11.0/bin:$PATH;`
prefixed to remote Slurm commands; ICE does not.)

When the process state inside a running allocation matters, use
`srun --overlap`:

```bash
ssh <ice|skynet> 'srun --jobid=<jobid> --overlap --ntasks=1 bash -lc "ps -u fwu91 -o pid,ppid,stat,etime,pcpu,pmem,args | head -80"'
```

## Cancellation

Only cancel when the user asks or explicitly approves:

```bash
pixi run python -m imitation_experiments.pipeline.cluster cancel --submission <plan_dir> --yes
```

## Known failure modes

- **"Requested time limit is invalid"** at `submit` — walltime over ICE's 16h
  cap; lower `time_limit` in the campaign's stage spec. `submit` auto-cancels
  any already-submitted stage in the chain when a later sbatch fails, so a
  partial chain never lingers — but re-plan/re-submit after fixing the value.
- **A `[PREFLIGHT] FAIL dataset:...`** — the path in `preflight.require_container_paths`
  or a stage arg isn't visible under the profile's `data_dir`/`extra_bind_paths`.
  Fix the path or add the allocation to the profile's `extra_bind_paths`, not
  by disabling the check.
- **`run_singularity.sh` errors "No frozen job environment found"** — the
  workspace wasn't produced by `plan`/`submit` (e.g. someone invoked it
  directly, or copied a workspace by hand). Resubmit through the CLI.
- **A campaign has no `campaign.yaml`** — it predates the control plane and
  was deliberately left unmigrated. There is no ad-hoc fallback; write the
  YAML (see above) before it can submit again.
