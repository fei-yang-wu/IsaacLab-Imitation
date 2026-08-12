# 2026-08-10 — GR00T planner local debug loop

Purpose: produce a small, real, local dataset of oracle rollout-training
samples with the `z256_scaled` low-level tracker, cache real Cosmos language
features for the selected-ten goals, and run the GR00T action-head trainer on
them. This is a **debug campaign**: its outputs qualify the pipeline, not any
result. Every number it produces is preliminary (one seed, tiny trajectory
budget, non-frozen oracle).

## Inputs (hash-gated by `run.sh`)

| Input | Path | sha256 |
| --- | --- | --- |
| Tracker (z256_scaled, ~5.75B frames) | `logs/downloaded_checkpoints/bones129k_recent_ice/z256_scaled/model_step_5750390784.pt` | `bc4569e6...` |
| Encoder (binding-verified vs tracker) | `logs/downloaded_checkpoints/bones129k_recent_ice/encoders/z256_scaled.pt` | `862eadd7...` |
| Selected-ten manifest | `data/bones_seed_language10_v1/manifests/g1_bones_seed_language10_v1_manifest.json` | `60a5b7a5...` (identical to the 2026-08-05 campaign pin) |
| MiniLM table (pipeline gate input) | `data/bones_seed_language10_v1/language/...minilm_goal_embeddings.pt` | `04624a22...` (identical to the 2026-08-05 pin) |
| Reference arrays (root_qpos_v1) | `data/bones_seed_language10_v1/reference_arrays/root_qpos_v1` | manifest `e8996c26...` (identical to the 2026-08-05 pin) |

The tracker geometry differs from the 2026-08-05 oracle: `z256_scaled` uses
the scaled policy cells `[2048,2048,1024,1024,512,512]` and the robot anchor
(default), passed through `--policy_num_cells`. Command stays 258 = z256 +
sin/cos phase, hold 10.

## Stages

```bash
./run.sh print     # dry-run the collection command
./run.sh collect   # oracle trajectory collection (isaaclab env, Newton)
./run.sh goals     # cache Cosmos text features per goal (gr00t env, GPU)
./run.sh train     # GR00T head, warm-start stage A on the collected samples
```

- `collect` calls `imitation_experiments.pipeline.run_language_planner_oracle_pretrain
  --stage collect`: ten motions x `TRAJECTORIES_PER_MOTION` (default 5)
  environments, frame-0 starts, deterministic oracle actions, root_qpos
  sample requirement on. Samples land in
  `logs/gr00t_planner_local_debug/collection/rollout_training_samples/`.
- `goals` builds `{motion name: language_goal}` from the manifest language
  sidecar and runs the text-only `nvidia/GR00T-N1.7-3B` backbone once per
  goal. It also exports the action-head warm-start bundle.
- `train` runs `imitation_experiments.planner.train_gr00t_head` with the
  N1.7 trunk warm start, projectors-only (stage A). Stage B (trunk unfrozen)
  exceeds the local 20 GiB GPU and belongs on Skynet.

## Status

- 2026-08-10: created. Inputs verified present and hash-matched; the
  encoder-in-tracker binding was checked tensor-identical before this
  campaign was written.
- 2026-08-10: all three stages ran to completion locally (preliminary,
  debug-grade, one seed):
  - `goals`: 10 real goals cached from `nvidia/GR00T-N1.7-3B` (bf16
    backbone, select_layer from checkpoint config), warm-start bundle
    exported. `logs/gr00t_planner_local_debug/goal_features/`.
  - `collect`: formal `validate_latent_skill_checkpoint_binding` audit
    passed (`latent_skill_binding.json`); 50 envs produced 2,595 sample
    rows (`sample_step_000000.pt`, 84 MB) with the full root_qpos schema
    (`oracle_rollout_state_history` [N,930], `expert_root_qpos_future`
    [N,30,38] + valid mask, `motion_name` per row).
  - `train`: warm-start stage A (1,294.7M trunk params kept, 10.8M fresh
    projectors), 2,000 updates at ~13.5 up/s, loss converged to the
    2.0-3.5 band from the ~45 naive baseline. Checkpoint
    `gr00t_head_stage_a/checkpoints/update_0002000.pt` with normalization
    stats and the kept/fresh load manifest.
- 2026-08-11: the saved stage-A head was deployed through the native
  asynchronous pipeline against stock CPU MuJoCo for 300 paced control
  ticks with goal `walk_arc_cw_start_R_slow_001_A443`. The first behavior
  result is invalid because the native projected-gravity horizontal signs
  were opposite to the Python and Isaac convention. After this was fixed,
  the same one-motion debug run completed 295 policy ticks plus 5 startup
  WAIT ticks and 1,200 independent physics ticks with 30 planner replies,
  zero planner, control, or physics deadline misses, and no runtime fault.
  It did not fall: minimum, mean, and final base heights were 0.689 m,
  0.747 m, and 0.735 m. Maximum control tick time was 2.58 ms; maximum
  control and physics wake lateness were 0.189 ms and 0.220 ms. This
  preliminary one-motion, one-seed result qualifies the asynchronous
  construction only. It is not evidence of planner quality.
