---
name: skynet-job-submission
description: Pointer to the merged cluster submission skill. Skynet submission, monitoring, cancellation, and its cluster-specific constraints now live in the cluster-job-submission skill. Use when the user mentions Skynet, sky1, wu-lab, dendrite or synapse, or Apptainer on Skynet compute nodes.
---

# Skynet job submission — merged

Skynet is a **profile** of the repo-owned cluster control plane, not a
separate workflow. Everything that used to be here is now in one place:

**Use the `cluster-job-submission` skill.**

It carries the four CLI verbs (`plan`, `submit`, `status`, `logs`, `cancel`),
the `campaign.yaml` schema, the per-profile path and resource table, walltime
segmentation and chaining, the storage and data-I/O rules, and a
"Skynet profile notes" section with the cluster-specific facts: Apptainer on
compute nodes only (`dendrite`, `synapse`, and a few L40S nodes such as
`bishop`), QoS `short` 2 days / `long` 7 days, the
`/coc/flash12/fwu91/Research/IsaacLab` storage root, the Slurm `PATH` prefix
for remote commands, node-local `/tmp` orphan cleanup, the SIF extraction
race, the Isaac Sim EULA variables, and the compute-node diagnostic job.

Two facts worth repeating here:

- `docker/cluster/cluster_interface.sh`, `submit_job_slurm_skynet.sh`, and
  `submit_job_slurm_skynet_pixi.sh` are deprecation shims. Running one prints
  an error and exits. Any document that tells you to invoke them is stale.
- The `skynet` profile has never been exercised against a real submission.
  Run a cheap smoke chain and confirm it completes before any real spend.
