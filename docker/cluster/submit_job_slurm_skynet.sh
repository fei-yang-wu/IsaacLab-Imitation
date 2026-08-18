#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired along with cluster_interface.sh, which used
# to invoke this as the Skynet Slurm submitter.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/submit_job_slurm_skynet.sh has been retired.

This was an internal helper invoked by the now-retired
docker/cluster/cluster_interface.sh. Use the control plane instead:

    pixi run python -m imitation_experiments.pipeline.cluster plan --campaign ... --arm ... --seed ...
    pixi run python -m imitation_experiments.pipeline.cluster submit --plan ... --confirm ...

The Skynet profile
(source/imitation_experiments/imitation_experiments/pipeline/cluster/conf/profile_skynet.yaml)
is EXPERIMENTAL and not yet validated against a real Skynet submission;
validate before real spend. Removed implementation:
`git log -- docker/cluster/submit_job_slurm_skynet.sh`.
EOF
exit 2
