#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired along with cluster_interface.sh.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/submit_job_slurm_skynet_pixi.sh has been retired.

This was an internal Pixi-runtime helper invoked by the now-retired
docker/cluster/cluster_interface.sh. Use the control plane instead:

    pixi run python -m imitation_experiments.pipeline.cluster plan --campaign ... --arm ... --seed ...
    pixi run python -m imitation_experiments.pipeline.cluster submit --plan ... --confirm ...

Removed implementation:
`git log -- docker/cluster/submit_job_slurm_skynet_pixi.sh`.
EOF
exit 2
