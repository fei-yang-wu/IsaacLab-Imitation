#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired along with cluster_interface.sh, which used
# to invoke this as its generic Slurm submitter.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/submit_job_slurm.sh has been retired.

This was an internal helper invoked by the now-retired
docker/cluster/cluster_interface.sh. Use the control plane instead:

    pixi run python -m imitation_experiments.pipeline.cluster plan --campaign ... --arm ... --seed ...
    pixi run python -m imitation_experiments.pipeline.cluster submit --plan ... --confirm ...

See docker/cluster/cluster_interface.sh for the full pointer, or
source/imitation_experiments/imitation_experiments/pipeline/cluster/slurm.py
for the batch-script rendering that replaced this file. Removed
implementation: `git log -- docker/cluster/submit_job_slurm.sh`.
EOF
exit 2
