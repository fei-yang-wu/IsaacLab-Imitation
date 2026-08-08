# BONES language10 loco-manip H30 future-fusion retraining

Status (2026-08-06): complete. Training reached 10,000 updates after an
optimizer-preserving resume at update 7,500. All 100 quantitative evaluations
(five milestones by two fusion modes by ten goals, 100 trajectories per goal)
passed the metadata/row-count audit. All 20 non-terminating diagnostic videos
were retained at each motion's own length.

This campaign retrains one medium flow-matching planner on the frozen ten-motion
loco-manipulation selection and evaluates two execution rules from the same
planner checkpoints.

- **H30 future** means that one planner call predicts three ordered H10 latent
  commands (30 low-level control steps total). Each command is supervised in
  the future robot frame where it will be executed. The low-level policy still
  receives one H10 command for ten steps before the planner publishes again.
- **Raw exponential** aligns all still-valid forecasts for the current H10
  command and averages them with exponentially smaller weights for older
  predictions (`decay=0.5`).
- **Clipped/gated exponential** first rejects an older forecast when its
  normalized RMS distance from the fresh forecast exceeds 2.0 or its cosine
  agreement is below 0.5. It clips each accepted residual to one training
  standard deviation per feature, then applies the same exponential weights.

The comparison therefore trains one model, not two. It isolates command-fusion
behavior while holding data, model, update budget, randomization, deterministic
policy actions, no-push evaluation, and frame-0 starts fixed.

The planner receives 44,100 H30 publications from 1,000 complete oracle-policy
trajectories (100 per motion), trains for 10,000 updates, and retains checkpoints
at 2k, 4k, 6k, 8k, and 10k. Every milestone is evaluated on 100 randomized,
push-disabled trajectories per explicit goal. The final checkpoint also gets a
non-terminating one-environment diagnostic video for every motion and fusion
mode.

Run from the repository root:

```bash
STAGES=materialize experiments/campaigns/2026-08-06-bones-language10-loco-manip-h30/run.sh
STAGES=train experiments/campaigns/2026-08-06-bones-language10-loco-manip-h30/run.sh
STAGES=eval,diagnostic,aggregate experiments/campaigns/2026-08-06-bones-language10-loco-manip-h30/run.sh
STAGES=semantic experiments/campaigns/2026-08-06-bones-language10-loco-manip-h30/run.sh
```

Audited curves and final per-motion results are written to:

- `logs/bones_language10_loco_manip_v3_h30_future_seed0/aggregate/results.md`
- `logs/bones_language10_loco_manip_v3_h30_future_seed0/aggregate/results.json`

## Result

| execution | update | SONIC SR | successful MPJPE-L (mm) |
|---|---:|---:|---:|
| raw exponential | 2k | 0.752 | 61.10 |
| raw exponential | 4k | 0.835 | 58.60 |
| raw exponential | 6k | 0.817 | 58.40 |
| raw exponential | 8k | 0.813 | 57.55 |
| raw exponential | 10k | **0.880** | 54.79 |
| clipped/gated exponential | 2k | 0.758 | 61.02 |
| clipped/gated exponential | 4k | 0.835 | 58.94 |
| clipped/gated exponential | 6k | 0.816 | 58.31 |
| clipped/gated exponential | 8k | 0.809 | 57.29 |
| clipped/gated exponential | 10k | 0.867 | **54.69** |

The 10k jump means neither curve satisfies the predeclared plateau heuristic;
the apparent 6k-to-8k flattening was not convergence. Use raw exponential as
the default operating point: at 10k it completes 13 more trajectories per
1,000 while giving up only 0.10 mm of success-only MPJPE-L. The completion gap
comes mainly from stoop (40/100 versus 34/100) and passing out flyers (93/100
versus 86/100). Lift crate, light-object carry, clockwise arc, and straight
walk are 100/100 under raw exponential; stoop remains the weak motion at
40/100.

The full-horizon randomized/no-push diagnostic rendered ten motions per mode
with every termination disabled and no physical falls. Mean single-render local
MPJPE was 54.04 mm for raw exponential and 57.71 mm for clipped/gated. Videos
live under `logs/bones_language10_loco_manip_v3_h30_future_seed0/full_horizon_diagnostic/`.

## Semantic phase analysis

A **phase** is a contiguous interval with one observable task meaning, such as
approach walking, stooping to retrieve, inspecting, placing, or resuming the
walk. Boundaries are exact zero-based, end-exclusive reference control-step
indices at 50 Hz. They are manual observational annotations, not learned change
points. The frozen boundaries and five reusable binary axes live in
`semantic_phase_annotations.json`.

The comparison scene does not render crates, doors, buttons, or flyers.
Object/environment meanings come from the frozen dataset action and language
goal; the interval boundaries themselves come from visible body motion and the
reference kinematic traces.

The semantic stage cuts the ten full-horizon side-by-side videos into 39
frame-accurate H.264 clips with ffmpeg and assigns every one of the 44,100
collected latent rows to a labeled phase. It then reuses the same balanced
6,000-row PCA/t-SNE sample and adds leave-one-motion-out probes: train a binary
semantic classifier on nine motions and test it only on phases of an unseen
motion. This prevents a probe from succeeding merely by recognizing the motion
name.

The local 10-neighbor semantic-activity purity is 0.927, but the global
activity silhouette is -0.114. Cross-motion balanced accuracy is 0.571 for
locomotion and 0.556 for manipulation. Object-loaded reaches 0.643, but only
two held-out motions contain both loaded and unloaded phases. The defensible
conclusion is that the latent has locally coherent phase structure, while
shared action semantics transfer weakly across motion identities; it is not yet
evidence of a clean compositional code.

Outputs are under `logs/bones_language10_loco_manip_v3_h30_future_seed0/latent_space_analysis/`:

- `analysis.json`, `embedding.csv`, and `latent_space.png`
- `semantic_phase_clips/phase_clip_manifest.json` and 39 labeled video clips
