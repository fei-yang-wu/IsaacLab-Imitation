# Pruned experiment scripts

These paths were removed from the working tree in the 2026-07-23 cleanup. They
are **not lost**: every file is intact in Git history and is recoverable with
the commands below. They were removed from the live tree because a completed
one-off launcher sitting beside current code reads as a supported entrypoint,
and `experiments/README.md` requires every retained script to justify its place
in `SCRIPT_INVENTORY.md`.

Nothing here is a current entrypoint. Do not resurrect one into a live campaign
without re-reading the protocol it belonged to; several encode budgets, gates,
or data layouts that the current paper protocol has since changed.

## Recovering a pruned script

All of these paths exist at commit `862e4bd` (the last commit before the
cleanup). Restore one to inspect it:

```bash
# Read it without touching the working tree.
git show 862e4bd:experiments/interface_baselines/train_interface_planner.py

# Or restore it to a scratch location.
git show 862e4bd:experiments/interface_baselines/train_interface_planner.py \
    > /tmp/train_interface_planner.py
```

To see the full history of a pruned path, including why it changed:

```bash
git log --follow -- experiments/interface_baselines/train_interface_planner.py
```

## Pruned paths


### `experiments/` — Completed one-off cluster launchers

Each submitted a specific finished training block (BONES-SEED 100/SONIC/2B/5B resumable runs, LAFAN1 5B variants, VQ-VAE Dance102). Their job IDs and results are recorded in the wiki; none is a reusable entrypoint.

- `experiments/submit_bones_seed_100_pretrain_lowlevel_skynet.sh`
- `experiments/submit_bones_seed_100_sonic_latent_ice.sh`
- `experiments/submit_bones_seed_paper24_skynet_2b.sh`
- `experiments/submit_bones_seed_sonic_5b_resumable_ice.sh`
- `experiments/submit_lafan1_5b_resumable_ice.sh`
- `experiments/submit_lafan1_chunk_tracker_5b_resumable_ice.sh`
- `experiments/submit_lafan1_latent_variant_5b_resumable_ice.sh`
- `experiments/submit_lafan1_strict_default_sanity_ice.sh`
- `experiments/submit_sonic_latent_vram_ablation_ice.sh`
- `experiments/vqvae_submit_dance102_cluster.sh`
- `experiments/vqvae_temporal_ablation.sh`


### `experiments/interface_baselines/` — Superseded interface-comparison variants

Earlier phase-2/phase-3 gates, held-out and multi-seed Dance102/LAFAN1 variants, alternative planner families (categorical token, Future-CVAE, per-step token), and the 2026-07-15 preliminary DiffSR launcher. The paper scope is the two-row comparison in `wiki/causal-interface-paper-plan.md`.

- `experiments/interface_baselines/aggregate_interface_comparison_seeds.py`
- `experiments/interface_baselines/analyze_interface_sweep.py`
- `experiments/interface_baselines/audit_interface_comparison.py`
- `experiments/interface_baselines/audit_one_motion_causal_latent_gate.py`
- `experiments/interface_baselines/audit_phase2_shared_continuous.py`
- `experiments/interface_baselines/audit_phase3_latent_interfaces.py`
- `experiments/interface_baselines/audit_phase3_local_10m.py`
- `experiments/interface_baselines/backfill_offline_target_evals.py`
- `experiments/interface_baselines/backfill_planner_capacity_metadata.py`
- `experiments/interface_baselines/eval_latent_skill_planner_offline.py`
- `experiments/interface_baselines/preflight_interface_comparison.py`
- `experiments/interface_baselines/run_dance102_fair_interface_comparison.sh`
- `experiments/interface_baselines/run_dance102_strong_interface_multiseed.sh`
- `experiments/interface_baselines/run_diffsr_latent_qualification.sh`
- `experiments/interface_baselines/run_future_cvae_interface_comparison.sh`
- `experiments/interface_baselines/run_lafan1_from_scratch_comparison.sh`
- `experiments/interface_baselines/run_lafan1_heldout_strong_interface_comparison.sh`
- `experiments/interface_baselines/run_lafan1_heldout_strong_interface_multiseed.sh`
- `experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh`
- `experiments/interface_baselines/run_multimotion_heldout_interface_comparison.sh`
- `experiments/interface_baselines/run_one_motion_capacity_point.sh`
- `experiments/interface_baselines/run_per_step_token_interface_comparison.sh`
- `experiments/interface_baselines/run_phase2_shared_continuous_comparison.sh`
- `experiments/interface_baselines/run_phase3_latent_action_comparison.sh`
- `experiments/interface_baselines/run_phase3_local_10m_qualification.sh`
- `experiments/interface_baselines/run_vanilla_tracker_qualification_block.sh`
- `experiments/interface_baselines/select_lafan1_trajectories.py`
- `experiments/interface_baselines/smoke_test_interface_planner.py`
- `experiments/interface_baselines/split_lafan1_manifest.py`
- `experiments/interface_baselines/submit_bones_seed_diffsr_preliminary_skynet.sh`
- `experiments/interface_baselines/summarize_lafan1_motion_tracking.py`
- `experiments/interface_baselines/train_categorical_token_planner.py`
- `experiments/interface_baselines/train_interface_planner.py`


### `experiments/command_space_ablation/` — Archived command-space oracle sweep

The broad command-style sweep is out of paper scope. Only `evaluate_checkpoint.py` and `submit_cluster_oracle_ablation.sh` remain, as dependencies of current low-level gates.

- `experiments/command_space_ablation/compare_eval_tags.py`
- `experiments/command_space_ablation/evaluate_oracle_checkpoints.sh`
- `experiments/command_space_ablation/list_cluster_checkpoints.sh`
- `experiments/command_space_ablation/merge_eval_csvs.py`
- `experiments/command_space_ablation/run_local_oracle_smoke.sh`
- `experiments/command_space_ablation/submit_cluster_evaluations.sh`
- `experiments/command_space_ablation/summarize_eval_csv.py`


### `experiments/bilinear_pretrain/` — Superseded bilinear pretraining sweeps

The bilinear pretraining line was replaced by the DiffSR latent interface. Retained analysis lives in `wiki/ipmd-representation-learning.md`.

- `experiments/bilinear_pretrain/submit_cluster_ablation.sh`
- `experiments/bilinear_pretrain/submit_dance102_action_label_ablation.sh`
- `experiments/bilinear_pretrain/submit_goal_ae_ablation.sh`
- `experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh`
- `experiments/bilinear_pretrain/summarize_pretrain_wandb.py`


### `experiments/training_scale/` — Completed scale benchmark

The wall-clock screen selected the H200 16,384 x 12 profile now used by the latent-ablation campaign. Its result is recorded in `experiments/campaigns/2026-07-22-bones-h10-scale/README.md`.

- `experiments/training_scale/2026-07-17-a40-results.md`
- `experiments/training_scale/__init__.py`
- `experiments/training_scale/run_latent_scale_benchmark.py`
- `experiments/training_scale/run_local_latent_scale_benchmark.sh`
- `experiments/training_scale/submit_latent_scale_benchmark_skynet.sh`
- `experiments/training_scale/test_latent_scale_benchmark.py`


### `experiments/ipmd_stability/` — Resolved IPMD stability debugging

The instability was root-caused and fixed; the debug ablations no longer have a caller.

- `experiments/ipmd_stability/run_local_debug_ablations.sh`
- `experiments/ipmd_stability/submit_cluster_ablations.sh`


**Total pruned: 64 paths.** Recovery commit: `862e4bd`.
