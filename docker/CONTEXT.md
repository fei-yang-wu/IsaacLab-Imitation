# CONTEXT.md — `docker/` (containers and cluster submission)

Bounded context: container build and cluster job submission. Read the
repository root [`CONTEXT.md`](../CONTEXT.md) first.

## Ubiquitous language

- **Control plane** — `python -m imitation_experiments.pipeline.cluster`
  (implementation:
  `source/imitation_experiments/imitation_experiments/pipeline/cluster/`),
  the `plan`/`submit`/`status`/`logs`/`cancel` CLI that packages the
  workspace and submits jobs. Replaced `cluster/cluster_interface.sh` and
  every `cluster/submit_job_slurm_*.sh`/`submit_job_pbs.sh` on 2026-08-15;
  those files are now deprecation shims that error with a pointer here
  instead of running. Retired because config used to forward through five
  hand-maintained env allow-lists that silently reverted unregistered
  variables.
- **Campaign spec** — one `campaign.yaml` per campaign, declaring arms,
  stages, resources, and preflight requirements; consumed by
  `imitation_experiments.pipeline.cluster.config`. See
  `experiments/campaigns/2026-08-14-latent-quant-ice-repeats/campaign.yaml`
  for a worked 14-arm example (real-ICE validated 2026-08-15, jobs
  5577564/5577565).
- **Cluster profile** — `cluster/conf/profile_<name>.yaml` under the
  control-plane package: ssh alias, data/SIF/log paths, Slurm defaults, and
  an open `env` dict frozen verbatim into every job (no allow-list). `ice`
  is validated against a real submission; `skynet` is transcribed from the
  retired `.env.skynet_runtime` but experimental.
- **Plan** — the frozen, sha-sealed output of `... plan`: resolved config,
  rendered batch script, and per-stage `job_env.<stage>.resolved.sh`.
  `submit --confirm <PLAN_SHA>` refuses to run a plan whose sha, or whose
  git state, has drifted since planning.
- **Overlay** — a local dependency checkout mounted over the pinned
  submodule. Legacy-path concept (`CLUSTER_RLOPT_LOCAL_PATH`); the control
  plane has no overlay support yet — a migrating campaign that needs one
  should flag it.
- **Workspace archive** — the verified tar of this repo a job extracts on
  compute-local storage (full-tree copies can block on Skynet NFS). Its
  hash lands in the plan's `submission-*.json`.
- **Skynet** — SLURM cluster for large training and paper-scale batch
  evaluation. Apptainer runs only on the dendrite/synapse partitions (plus
  bishop L40S). Jobs over two days need QoS `long`.
- **ICE / PACE** — Georgia Tech cluster (H200). 300 GB hard storage cap
  (the control-plane `plan` preflight checks free space against each
  profile's `min_free_gb`). Slurm `TIMEOUT` wipes node-local output: write
  checkpoints to `/data`, never node-local paths. Walltime is capped at 16h;
  a longer `time_limit` is rejected by `sbatch` at submit time.
- **SIF** — the Singularity/Apptainer image a job runs in
  (`run_singularity.sh`).

## Rules

- Default cluster budget: about 1B environment frames per task/run and a
  two-day walltime, unless the user sets another budget.
- Prefer Skynet for training; prefer the local workstation for inference,
  playback, and video, because a fresh Isaac container is expensive per
  job.
- Never read reference data by mmap from a network filesystem; load it
  resident into RAM.
- Do not submit until `... plan`'s preflight passes and the user has
  confirmed the exact `submit --confirm <PLAN_SHA>` command.

## Validation

```bash
pixi run test-experiments   # covers the control plane and the deprecation shims
bash -n docker/cluster/run_singularity.sh
```
