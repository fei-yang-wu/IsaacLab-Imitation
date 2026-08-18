---
name: gr00t-planner
description: Train, collect data for, evaluate, and deploy the verbatim GR00T N1.7 action head as the language-conditioned high-level planner in this repo. Covers the collect - prepare - cache - train - evaluate pipeline, its Hydra configs, the arm matrix (chunk / z256 / FSQ pre-quant, mocap vs rollout), temporal ensembling, and the two known parity defects. Use when the user mentions GR00T, N1.7, the action head, gr00t_head, chunk or latent planner targets, fsq64 or z256 planner arms, goal features, Cosmos text features, temporal ensembling, RTC, or the language planner campaigns.
---

# GR00T planner

The high-level planner is the **unmodified** `Gr00tN1d7ActionHead` class from
`external/Isaac-GR00T` (pinned commit `376ba890`). Never edit the submodule.
All adapters are in `imitation_experiments.planner`.

Train in the `gr00t` Pixi environment. Evaluate in Isaac with the `isaaclab`
environment. Both are separate environments on purpose.

```bash
pixi run -e gr00t python -m imitation_experiments.planner.<module> ...
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py ...
```

## The pipeline

Five stages. Each stage writes an artifact the next stage reads. Do not skip a
stage, and do not re-use an artifact from a different tracker.

1. **Collect** — drive the frozen tracker with the oracle command and record
   one row per control step.
   `scripts/rlopt/eval_skill_commander_closed_loop.py` with the campaign's
   `collect_*.sh` wrapper. Two data modes:
   - **mocap** = `env.replay_only=true`. Kinematic replay. The state history
     carries expert kinematics.
   - **rollout** = the tracker actually drives the robot, so the state history
     carries closed-loop dynamics and that tracker's own `last_action`.
     Rollout data gives the better planner on every interface measured
     (MPJPE −12% to −32%).
2. **Prepare** — consolidate the collection into one training table.
   `imitation_experiments.planner.prepare_gr00t_dataset`, Hydra, base config
   `planner/conf_gr00t/base_prepare.yaml`.
3. **Cache goal features** — run the Cosmos-Reason2-2B text encoder once,
   offline, over the finite goal set.
   `imitation_experiments.planner.cache_gr00t_goal_features`. The backbone
   must run in bf16; the checkpoint enables flash attention, which rejects
   fp32. Read `select_layer` from the checkpoint provenance; do not assume the
   config default.
4. **Train** — `imitation_experiments.planner.train_gr00t_head`, Hydra, base
   config `planner/conf_gr00t/base_train.yaml`.
5. **Evaluate** — Isaac closed loop (the number of record), or the Embodied-
   Control rehearsal rig, or the chaining test. See "Evaluation" below.

## Configuration is Hydra, never a code pin

Campaign configs compose on the base configs:

```bash
pixi run -e gr00t python -m imitation_experiments.planner.train_gr00t_head \
    --config-dir <campaign>/conf --config-name train_<arm> [overrides...]
```

Struct mode rejects a **new** arm key given on the command line. Add an arm in
a composing YAML file, not with a CLI override.

Defaults that come from GR00T's own finetune recipe, and that you must not
change without saying so: AdamW fused, lr 1e-4, weight decay 1e-5, cosine
schedule, warmup ratio 0.05, grad clip 1.0, batch 64, bf16 + tf32, no EMA,
`state_dropout_prob=0.2`, `stage_a_updates=0` (projectors and DiT train
together — a stage A/B split is a small-GPU workaround, not their recipe).

## The target interfaces

`target: chunk | latent` in the train config, plus `latent.source` in the
prepare config.

| target | what the head predicts | notes |
| --- | --- | --- |
| `chunk` | 30 x 38 expert `root_qpos` lookahead | the explicit interface |
| `latent`, `source: stored` | 3 x 256 continuous z, joined from stored rows | z256 arms |
| `latent`, `source: fsq_prequant` | 3 x 64 **pre-quantized** FSQ vector | snap to the lattice at publication, never regress the rounded code |

Latent slots come from **joining stored rows** at control step +0, +10, +20.
Window frames are anchored at query time, so re-encoding a shifted window is
not the same thing. Re-encoding is not a substitute for a fresh collection: a
re-encoded FSQ arm scored perfect survival (1.000) and the worst MPJPE
(102.2 vs 57.1 mm) — a degenerate but safe policy.

## Evaluation

Always report both numbers of the planner metric standard: root-relative
MPJPE and fall-only survival. See the `sonic-success-eval` skill for the
low-level SONIC criterion, which is a different pass.

Isaac closed loop, current best protocol:

```bash
pixi run -e isaaclab python scripts/rlopt/eval_skill_commander_closed_loop.py \
    --headless --task Isaac-Imitation-G1-v2 --algorithm IPMD \
    --checkpoint <tracker> --skill_checkpoint <encoder> \
    --gr00t_checkpoint <head> --gr00t_goal_features <cache> --gr00t_goal <name> \
    --gr00t_consumption open_loop \
    --gr00t_temporal_ensemble exponential --gr00t_temporal_ensemble_decay 0.5 \
    --fall_only_success --disable_tracking_terminations \
    --motion_names <name> --metric_interval 10 \
    physics=newton_mjwarp ...
```

Protocol rules:

- Fall-only termination, `physics=newton_mjwarp`, 2000-step cap so
  `done_rate` is 1.000. A 500-step cap truncates about a third of the episodes
  and its numbers are **not** comparable to the 2000-step ones.
- `--fall_only_success` also disables `foot_pos_xyz`. Without it, survival
  saturates at 1.000 and step counts measure a foot-tracking termination.
- MPJPE needs `env.data.runtime_cache_body_names=[...14 tracked bodies...]`,
  or the metric silently does not appear.
- The language goal is passed explicitly. Never infer it from the reference
  cursor or the trajectory rank.
- Pin `--seed`, and pin the chunk service seed when one is used. The flow
  sampler starts from `randn`; two unseeded identical runs already differed by
  0.078 rad.

Other evaluation routes: `imitation_experiments.evaluation.eval_gr00t_ec`
(MuJoCo rehearsal through Embodied-Control — see the `ec-deployment` skill)
and `imitation_experiments.evaluation.eval_gr00t_chaining` (mid-episode goal
switch). Isaac and EC can disagree strongly; an arm that collapses in only one
simulator leans on that simulator's dynamics.

## Current baseline (2026-08-13)

`fsq64_scaled28` + exponential temporal ensembling, decay 0.5:
**46.95 mm MPJPE-L / 0.998 fall-free**, 28 motions x 20 episodes, one seed.
Recipe: verbatim N1.7 head, 12k updates, batch 64, warm-started trunk,
889,044 rows from oracle-latent rollouts of the fsq64 tracker,
`state_dropout_prob=0`, 3 latents held 10 steps.

The released NVIDIA SONIC v1.1 planner scores 46.33 / 1.000 on its own
tracker. The 0.6 mm gap is **inside evaluation noise**; do not call it a win
or a loss. Run-to-run MPJPE spread is 0.2–6.4%.

Two motions are tracker-limited and are excluded from the set:
`panic_run_away_180_R_001_A423` and `walk_big_dog_ff_225_stop_R_001_A492`.
The oracle itself falls 4/5 on both. No planner change can fix them.

## Refuted — do not retry without new evidence

- Single-motion training. The 30-goal model beats it on the same motion
  (39.76 vs 44.55 mm).
- Sample averaging, 16 ODE steps, and `fresh` consumption. All trade MPJPE for
  survival. Temporal ensembling is the only inference knob that improves both,
  because it blends estimates from different states.

**Hold 1 is not refuted as an interface** — only as a drop-in on a tracker
trained for hold 10. That test is confounded by the tracker's `sin_cos` phase
channel (`code_period=10`). To settle it, retrain the tracker at hold 1,
preferably with no phase channel, then retrain the planner against it.

## Two parity defects — name them in any parity claim

1. **Loss mask normalization.** Upstream's `action_mask` is
   `[T, max_action_dim]`; ours is `[B, H, 1]` (`gr00t_head.py:359`). The
   verbatim class sums that same mask for the denominator, so our loss divides
   by `count(B, H)` only. Our gradient magnitude is inflated by about the
   action dimension (~38x for chunk mode) against what lr 1e-4 and grad clip
   1.0 were tuned for. Fix by expanding the mask to `[B, H, D]`. Until then,
   do not compare our loss curves with upstream's loss scale.
2. **`attend_text_every_n_blocks=1`** vs the upstream default 2
   (`gr00t_head.py:167`). Necessary, because at 2 half the cross-attention
   blocks would attend to an always-empty image mask and produce NaN. But it
   changes which pretrained trunk blocks receive conditioning. It is a
   masking-schedule change, not only an "embedding source swap".

Everything else was verified at file:line as parity: model class identity,
flow matching, tokenization, q01/q99 normalization, language conditioning, and
the training recipe.

## Traps

- The `gr00t` package prints banners to **stdout**. The stdio chunk service is
  line-JSON; skip lines that do not parse.
- Checkpoints are about 4.9 GB each. `run.checkpoint_interval=1000` filled a
  916 GB disk mid-chain and killed four arms with a
  `PytorchStreamWriter` write error. Use 4000, and prune to the kept updates.
- Watch `df` before a long training chain. A full root disk killed the chain
  and all shell output twice.
- `chunk_native` on the explicit tracker cannot use
  `eval_skill_commander_closed_loop` — that script asserts a latent actor
  contract. Use `imitation_experiments.evaluation.eval_interface_planner_closed_loop`.
- Training our-encoder targets on SONIC-driven rollouts is **blocked**: one
  macro-state config per environment, and `_configure_sonic_contract`
  overwrites it after Hydra. It needs an offline two-pass rig.

## Provenance

Record in every head checkpoint: the upstream commit, the HF checkpoint SHA,
and the kept/fresh key manifest of the filtered state-dict load. Before Isaac
evaluation of a latent arm, run the binding gate — see the
`planner-submission-gate` skill.

Detail pages: `wiki/gr00t-planner-deployment.md`, and the phase log in
`experiments/campaigns/2026-08-12-gr00t-language30-compositionality/PLAN.md`.
