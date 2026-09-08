# Containers and cluster submission

- Control plane: imitation_experiments.pipeline.cluster; plan, submit, status,
  logs, cancel. Retired cluster_interface.sh and submit_job_slurm_* wrappers
  are not submission entrypoints.
- Campaign spec: campaign.yaml declares arms, stages, resources, env, and gates.
- Cluster profile: pipeline/cluster/conf/profile_<name>.yaml supplies host,
  paths, defaults, and bind settings. Verify live limits before planning.
- Plan: resolved config, batch script, and stage env files sealed by PLAN_SHA.
- Workspace archive: submitted working-tree snapshot with recorded hash.
- SIF: Singularity/Apptainer image used by run_singularity.sh.

ICE and Skynet run Slurm. Persistent checkpoint storage must survive walltime
and compute-node cleanup. Keep dataset reads resident or staged locally rather
than memory-mapped from network storage. Job paths must be container-visible.

Frozen stage env files govern control-plane jobs; editing .env.cluster does
not change them. Use cluster-job-submission for operational details and
AGENTS.md for budget policy.
