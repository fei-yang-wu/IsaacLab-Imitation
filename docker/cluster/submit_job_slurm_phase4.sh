#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired along with cluster_interface.sh, which used
# to invoke this as the phase4-no-language array submitter.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/submit_job_slurm_phase4.sh has been retired.

This was an internal helper invoked by the now-retired
docker/cluster/cluster_interface.sh (CLUSTER_SLURM_SUBMIT_SCRIPT=phase4). Use
the control plane instead:

    pixi run python -m imitation_experiments.pipeline.cluster plan --campaign ... --arm ... --seed ...
    pixi run python -m imitation_experiments.pipeline.cluster submit --plan ... --confirm ...

experiments/paper/submit_phase4_no_language_skynet.sh, which selected this
submitter, needs its own migration to a campaign.yaml before it can run
again. Removed implementation:
`git log -- docker/cluster/submit_job_slurm_phase4.sh`.
EOF
exit 2
