# CONTEXT.md — `docker/` (containers and cluster submission)

Bounded context: container build and cluster job submission. Read the
repository root [`CONTEXT.md`](../CONTEXT.md) first.

## Ubiquitous language

- **Cluster interface** — `cluster/cluster_interface.sh`, the front door
  that packages the workspace and submits jobs.
- **`.env.cluster`** — `cluster/.env.cluster`, the submission-time
  configuration (dataset manifest paths, overlay switches). For simple G1
  Dance102 runs, set `CLUSTER_G1_MANIFEST_PATH` there; commented out means
  the default 40 trajectories.
- **Overlay** — a local dependency checkout mounted over the pinned
  submodule. `CLUSTER_RLOPT_LOCAL_PATH` commented out means jobs use the
  submodule-pinned `RLOpt`. Leave overlays disabled unless a task
  explicitly needs an unpinned experiment.
- **Workspace archive** — the verified tar of this repo a job extracts on
  compute-local storage (full-tree copies can block on Skynet NFS). Its
  hash lands in `cluster_submission.json`.
- **Skynet** — SLURM cluster for large training and paper-scale batch
  evaluation. Apptainer runs only on the dendrite/synapse partitions (plus
  bishop L40S). Jobs over two days need QoS `long`.
- **ICE / PACE** — Georgia Tech cluster (H200). 300 GB hard storage cap.
  Slurm `TIMEOUT` wipes node-local output: write checkpoints to `/data`,
  never node-local paths.
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
- Do not submit until the relevant local check passes.

## Validation

```bash
bash -n docker/cluster/cluster_interface.sh
bash -n docker/cluster/submit_job_slurm_skynet.sh
```
