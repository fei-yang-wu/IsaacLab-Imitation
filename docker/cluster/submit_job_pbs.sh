#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired along with cluster_interface.sh. PBS was
# never adopted by the control plane; it supports Slurm only.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/submit_job_pbs.sh has been retired.

This was an internal helper invoked by the now-retired
docker/cluster/cluster_interface.sh (CLUSTER_JOB_SCHEDULER=PBS). The control
plane (source/imitation_experiments/imitation_experiments/pipeline/cluster/)
supports Slurm only. Removed implementation:
`git log -- docker/cluster/submit_job_pbs.sh`.
EOF
exit 2
