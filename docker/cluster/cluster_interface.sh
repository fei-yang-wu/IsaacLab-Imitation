#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired in favor of the repo-owned control plane.
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] docker/cluster/cluster_interface.sh has been retired.

Cluster submission now goes through the repo-owned control plane:

    pixi run python -m imitation_experiments.pipeline.cluster plan \
        --campaign <path/to/campaign.yaml> --arm <arm> --seed <seed>
    pixi run python -m imitation_experiments.pipeline.cluster submit \
        --plan <plan_dir> --confirm <PLAN_SHA>
    pixi run python -m imitation_experiments.pipeline.cluster status
    pixi run python -m imitation_experiments.pipeline.cluster logs --submission <dir> --stage <name>

Worked example (declared campaign.yaml + thin submit.sh, real-ICE validated
2026-08-15): experiments/campaigns/2026-08-14-latent-quant-ice-repeats/.

Implementation: source/imitation_experiments/imitation_experiments/pipeline/cluster/.

This ssh/rsync-based submitter is retired because it forwarded config through
five hand-maintained env allow-lists that silently reverted unregistered
variables. It is kept only as a guard so old invocations fail loudly instead
of silently misconfiguring a job; the removed implementation is in git
history (`git log -- docker/cluster/cluster_interface.sh`).
EOF
exit 2
