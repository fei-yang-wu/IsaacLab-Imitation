# 2026-07-23 BONES-SEED Phase-5 language planner pilot

This campaign is an explicitly **preliminary, latent-only** test of the Phase-5
hypothesis: after planner training, an explicit language goal and causal robot
history should be sufficient to select and execute different motions from the
same initial pose, without an expert trajectory being available at deployment.

It is not a paper-facing Phase-5 result. The new H200 controller was trained on
the 91-motion SONIC-filtered manifest, while the paper gate requires a matched
latent/vanilla pair trained and audited on the fresh 100-motion Phase-5 tree.
There is no matching qualified vanilla checkpoint for this H200 model. The
campaign therefore runs only `latent_skill`, records `preliminary_unqualified`,
and must not be included in the paired paper aggregate.

## Frozen pilot

- Low-level checkpoint: `logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt`
  - SHA-256: `6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572`
- H10 skill encoder: `logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt`
  - SHA-256: `562e4f9d0cebcdeb0bdddf6fb77ea8d0b488a8e576442b7106b54a13d6eceadc`
- Encoder binding: `.../final_4975165440/latent_skill_binding.json`; the repository validator passed all 14 tensors.
- Fresh data tree used by the latent-only jobs:
  - manifest `/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json`
  - language table `/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt`
  - preparation record `/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json`
  - latent cache `/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0`
  - vanilla cache `/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0` (shared-runner compatibility input only; no vanilla samples, baseline, or comparison are run)
- Selected goals (all common to the H200 91-motion manifest and the fresh 100-motion manifest):
  `Neutral_stoop_down_001_A057`, `avoid_bump_let_go_R_003_A460`,
  `axe_cutting_tree_horizontal_R_004_A355`,
  `big_heavy_two_hands_front_high_to_front_high_R_001_A524`,
  `big_light_two_hands_pick_up_front_medium_R_001_A509`, `body_check_001_A180`,
  `burning_loop_R_001_A528`, `casual_greeting_R_001_A428`,
  `cellphone_typing_sequence_one_hand_idle_R_001_A423`,
  `cough_tuberculosis_R_001_A500`.
- 150 demonstration rows/goal, 150 planner-rollout rows/goal, ten rollout
  environments per goal, 500-step closed-loop evaluation, medium planner,
  2,000 demonstration updates, 2,000 rollout-finetuning updates,
  batch/micro-batch 256/32, learning rate `1e-4`, flow steps 16.

The demonstration stage uses the frozen latent controller and reference only to
create oracle latent labels. The rollout and final-evaluation stages use the
planner's causal history plus the explicitly supplied language goal; they do not
expose expert state or a live reference to the deployed planner. This campaign
does not run a full-body baseline, controller comparison, or parameter/GPU
ablation.

## Validation and submission

The binding was checked with:

```bash
pixi run python experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/validate_latent_skill_checkpoint_binding.py \
  --low_level_checkpoint logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt \
  --skill_checkpoint logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt \
  --output_json logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/latent_skill_binding.json
```

Render the complete dependency chain without submitting it:

```bash
DRY_RUN=1 experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit.sh
```

The wrapper defaults to dry-run and refuses a real submission unless
`CONFIRM_SUBMIT=I_UNDERSTAND_UNQUALIFIED_PHASE5` is supplied. After the user
confirms the exact rendered command and the scheduler is rechecked, submit with
the same command and `DRY_RUN=0` plus that confirmation token. The chain is
`prepare -> rollout array -> finetune -> final-eval array -> summarize`.

Use a new output root for every attempt. Do not resume the failed 100-goal
chains from July 16; those roots contain partial data from the old 1,000
rows/goal configuration and are not compatible with this pilot.
