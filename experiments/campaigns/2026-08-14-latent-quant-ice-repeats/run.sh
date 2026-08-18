#!/usr/bin/env bash
# DEPRECATED 2026-08-15 — retired after the control plane's real-ICE cutover
# (jobs 5577564/5577565, full pretrain -> afterok -> lowlevel chain, exit 0).
set -euo pipefail
cat >&2 <<'EOF'
[DEPRECATED] This launcher has been retired. Use the control plane instead:

    ./submit.sh <arm> <seed> [--set vars.frame_cap=... | --only-stage lowlevel]

campaign.yaml declares all 14 arms; see README.md for the plan/submit/status
flow. Removed implementation: `git log -- experiments/campaigns/2026-08-14-latent-quant-ice-repeats/run.sh`.
EOF
exit 2
