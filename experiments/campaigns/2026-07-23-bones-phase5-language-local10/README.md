# 2026-07-23 BONES-SEED Phase-5 language planner, local ten-goal run

This campaign produces the two basic Phase-5 planner baselines **on the local
workstation**, without any Slurm queueing:

1. an **offline demonstration-pretrained** language-conditioned planner
   trained only on oracle demonstration rows; and
2. a **rollout-finetuned** planner trained on the merged demonstration plus
   robot-achieved rollout rows collected with planner 1 in the loop.

One shared planner covers all goals in both cases; the language goal is the
only per-task input. This is a preliminary, latent-only run: the H200
controller was trained on the 91-motion SONIC-filtered manifest and has no
matched qualified vanilla pair, so nothing here can enter the paired paper
aggregate. Every artifact records `preliminary_unqualified=true`.

## Scope and protocol

- Interface: `latent_skill` only (DiffSR h10/z256 latent commands at 5 Hz).
- Ten goals, identical to the 2026-07-23 H200 Skynet pilot so the two runs
  stay comparable; all ten exist in both the 91-motion training manifest and
  the corrected 100-motion Phase-5 tree.
- The Isaac stages load a derived **ten-motion subset manifest** (built once
  by `run.sh` from the frozen corrected tree, source hash recorded in the
  subset's `metadata.selection`), so batched collection and evaluation do not
  pay for the full 100-motion reference load. The derived manifest and its
  fresh latent Zarr cache live under `data/bones_seed_phase5_local10/`; the
  frozen corrected tree is never modified.
- Budgets: 150 demonstration rows/goal, 150 planner-rollout rows/goal, ten
  same-goal rollout environments, fixed 500-step (10 s) episodes, medium
  flow-matching planner, 2,000 pretrain + 2,000 finetune updates, batch
  256/32, lr `1e-4`, 16 flow steps.
- Stage semantics, sample schema, and evaluation metrics are the shared
  Phase-5 implementation in
  `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_multigoal_language_comparison.py`
  (stages `prepare -> rollout -> finetune -> final-eval -> summarize`).
  Training stages use the default Pixi environment; only collection and
  closed-loop evaluation launch Isaac.

## Frozen inputs

- Low-level checkpoint:
  `logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/final_4975165440/model_step_4975165440.pt`
  (SHA-256 `6765a324...d61d1572`, verified by `run.sh`).
- H10 skill encoder:
  `logs/bones_seed_91_h10_h200_e16384_5b_20260722/visual_check/skill_encoder_h10_z256_latest.pt`
  (SHA-256 `562e4f9d...d6eceadc`; tensor binding record must report
  `passed: true`).
- Source manifest and MiniLM language table: the corrected
  `data/bones_seed_phase5_corrected/bones_seed_100` tree.
- The `--vanilla_tracker_checkpoint` argument is a CLI-compatibility
  placeholder; latent-only mode never loads it and no vanilla row is run.

## Commands

Render the full stage plan without executing anything (default):

```bash
experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
```

Real local run, all stages in order:

```bash
DRY_RUN=0 experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
```

Stage-by-stage (for example, to stop after the demonstration-pretrained
planner exists), reusing the same `OUTPUT_ROOT`:

```bash
DRY_RUN=0 STAGE=prepare experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
DRY_RUN=0 STAGE=rollout RESUME=1 experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
DRY_RUN=0 STAGE=finetune RESUME=1 experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
DRY_RUN=0 STAGE=final-eval RESUME=1 experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
DRY_RUN=0 STAGE=summarize RESUME=1 experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh
```

Interrupted runs continue with `RESUME=1` (completed sub-steps are skipped by
their expected outputs). A fresh attempt needs a new `OUTPUT_ROOT`; the runner
refuses to overwrite an existing one.

## Outputs

Default output root: `logs/interface_baselines/bones_seed_phase5_local10_seed0`.

- Demonstration-pretrained planner:
  `<output_root>/latent_skill/planner_pretrain_demonstration/checkpoints/latest.pt`
- Rollout-finetuned planner:
  `<output_root>/latent_skill/planner_finetune_planner_rollout/checkpoints/latest.pt`
- Per-goal 500-step closed-loop summaries:
  `<output_root>/latent_skill/eval_pretrained_per_goal/` and
  `eval_finetuned_per_goal/`
- Final table: `<output_root>/summary/final_results.{json,csv}` (pretrained
  and finetuned results stay separate).

## Videos

After the pipeline writes the finetuned checkpoint, render per-goal closed-loop
videos of the single shared planner driving each goal:

```bash
experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_videos.sh
```

- One boot per goal, one environment, one explicit language goal each.
- Uses the protocol's non-terminating full-horizon diagnostic pass (all early
  terminations off, including `base_too_low`), so a fall does not truncate the
  clip. This is selected by passing neither `--disable_tracking_terminations`
  nor `--keep_early_terminations`.
- `PLANNER=finetuned` (default) visualizes the rollout-finetuned planner;
  `PLANNER=pretrained` visualizes the demonstration-only planner.
- Absolute `.mp4` paths are printed to stdout (under
  `<output_root>/videos_<planner>/<goal>/videos/play/`).

For a **side-by-side reference comparison** (expert reference in cyan vs the
planner+policy in red, in one scene, full 500-step horizon regardless of clip
length):

```bash
experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_reference_comparison.sh
SUBSET="0 2 8" .../render_reference_comparison.sh   # only those goal indices
```

Wraps `scripts/viz/compare_policy_reference.py` (env 0 = reference replay, env 1 =
policy). `--video_length 500` forces the full horizon; terminations are
disabled by the comparison script. Output under
`<output_root>/compare_reference_vs_planner_<planner>/rank_<i>_<goal>/` plus a
`video_index.md`. Note: `compare_policy_reference.py` was patched to load the
low-level checkpoint **weights-only** (optimizer state stripped) because the
SONIC-optimizer BONES-SEED controller has a different optimizer param-group
layout than a fresh eval agent.

## Status

- 2026-07-23: first run complete, exit 0, no stage failures. Output root
  `logs/interface_baselines/bones_seed_phase5_local10_seed0`. Both shared
  planners written (`planner_pretrain_demonstration/checkpoints/latest.pt`,
  `planner_finetune_planner_rollout/checkpoints/latest.pt`). Finetuned
  aggregate across 10 goals: tracking-success 0.8, mean survival 207/500
  steps, mean MPJPE ~70 mm (9 goals; one excluded as nan). Rollout finetuning
  helped several goals substantially (`Neutral_stoop` and `axe_cutting` went
  from success 0.0 to 1.0; `cellphone` 82->38 mm; `cough` 128->77 mm),
  regressed `big_heavy_two_hands` (46->138 mm) and `avoid_bump`, and left the
  two hardest (`burning_loop`, `avoid_bump`) failing early. Caveat: the short
  clips `big_light` (169 fr) and `casual_greeting` (204 fr) get a near-end
  random start under the frozen [0,200] range, so single-env final eval ends
  in 1-6 steps with trivial/nan metrics — an eval-start artifact, not a
  planner failure. Videos rendered via `render_videos.sh` under
  `videos_finetuned/`.
