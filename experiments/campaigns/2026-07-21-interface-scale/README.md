# 2026-07-21 interface-scale and recovery campaign

This folder is a historical index for the tracker-scale, SONIC-contract, and
resumable ICE work performed on 2026-07-21 and diagnosed on 2026-07-22. It is
not the current submission surface.

Important outcomes:

- the held full-body and EE chunk trackers completed their 5B-frame runs;
- per-step latent renewal collapsed in the controlled isolation and was
  rejected for the held-h10 ablation;
- the h10 held-latent jobs reached roughly 4.5B frames but their node-local
  checkpoints were lost at Slurm timeout;
- persistent central log binding was subsequently fixed and verified.

Read the exact chronology and current caveats in
[`wiki/current-status.md`](../../../wiki/current-status.md) and
[`wiki/ablation-experiment-plan.md`](../../../wiki/ablation-experiment-plan.md).

The one-shot launchers for this campaign were pruned on 2026-07-23 after their caller closure was checked. They remain available in Git history, with every path and its classification recorded in [`experiments/PRUNED_SCRIPTS.md`](../../PRUNED_SCRIPTS.md). This includes the old `submit_lafan1_*_ice.sh`, `submit_bones_seed_*_ice.sh`, and `submit_sonic_*_ice.sh` surfaces that encoded the completed jobs above.

Do not restore and resubmit one of those launchers merely because the historical job is relevant. Start a new dated campaign after deciding the new protocol, persistent output path, and qualification gate; copy only the still-valid pieces from history.
