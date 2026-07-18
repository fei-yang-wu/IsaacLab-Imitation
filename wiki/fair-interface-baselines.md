# Fair Interface Baselines

This page is the operational guide for the focused causal-interface comparison.
The paper-facing design is intentionally not a sweep over command styles. Read
[Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md) for
the research contract and
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md)
for changing checkpoint and job history.

## Paper-Facing Rows

The planner table contains exactly two rows:

| Interface | Planner output | Low-level consumption |
| --- | --- | --- |
| `latent_skill` | DiffSR latent at 5 Hz | Frozen latent policy holds it for ten 50 Hz steps |
| `full_body_trajectory_streamed_vanilla` | Ten-frame vanilla command packet at 5 Hz | The frozen vanilla tracker consumes slots 0 through 9 at 50 Hz |

The direct `single_frame_full_body` vanilla tracker is evaluated separately as
the low-level ceiling. It receives a fresh expert command at 50 Hz and is not a
planner row.

Do not add EE chunks, alternate packet layouts, Future-CVAE, or token variants
to the main table merely because the code supports them. Those paths are
diagnostics or appendix experiments unless the project explicitly changes
scope.

## Why This Is the Fair Explicit Baseline

The explicit planner predicts exactly what a vanilla whole-body tracking policy
would normally receive over the next ten steps. The packet is not sent to a
separately trained chunk policy. Instead, an adapter sends one frame at a time
to the exact same frozen vanilla actor used by the direct ceiling.

This gives the explicit method full access to the vanilla command sequence. If
the oracle packet does not reproduce direct vanilla behavior within numerical
tolerance, the adapter or provenance is wrong. It is not evidence that explicit
whole-body commands are weak. A competent vanilla oracle should also be at
least competitive with the latent oracle because it receives the complete
expert command rather than a subset; if it is not, low-level tracker training
remains a confound.

Once equivalence holds, the planner comparison is meaningful:

- DiffSR predicts a compact learned interface.
- The explicit planner predicts a 670-value future command packet.
- Both planners use the same causal `10 x 93` robot state, backbone, data
  rows, optimizer budget, and closed-loop evaluation.

The learned explicit planner can still perform worse than its oracle because
high-dimensional prediction errors compound in closed loop. Measuring that
burden is the research question.

## Command Contract

The explicit packet contains current plus nine future vanilla command frames
and is packed term-major:

| Term | Shape | Width |
| --- | --- | ---: |
| `expert_motion` | `10 x 58` | 580 |
| `anchor_pos` | `10 x 3` | 30 |
| `anchor_ori` | `10 x 6` | 60 |
| **Total** | | **670** |

Use `COMMAND_PAST_STEPS=0`, `COMMAND_FUTURE_STEPS=9`,
`PLANNER_UPDATE_INTERVAL=10`, and `COMMAND_HOLD_STEPS=10`. Slot 0 is the
current desired frame; slot 9 is the ninth future frame. Anchors are
re-expressed against the current robot anchor before the current slot reaches
the actor.

Command renewal is per environment. An environment publishes a new packet when
its own ten-step hold expires or when it resets. Never schedule publication
from the global rollout step modulo ten.

With command-side expert noise disabled, the direct actor command tensors are
numerically identical to the corresponding critic command entries. They are
separate observation groups and the critic contains additional privileged
state, but there is no intended actor/critic command-value mismatch.

## Causal Planner Input

Both planner rows receive nine past robot frames plus current, 93 values per
frame:

- 29 relative joint positions;
- 29 joint velocities;
- 3 IMU angular velocities;
- 3 projected-gravity values;
- 29 previous-action values.

Set `STATE_HISTORY_STEPS=9`; the resulting tensor is `10 x 93`, not eleven
frames. The planner may not receive expert history, reference phase/cursor,
future reference state, trajectory rank, or a reference-derived tracking error.
The expert future is available only for oracle commands, training labels, and
metrics.

The retired `current_achieved_macro_transition_batch` helper retains
reference information and is not deployable.

## Focused Local Runner

Use
`experiments/interface_baselines/run_focused_causal_interface_comparison.sh`.
It has no cluster or Skynet submission path. It runs the equivalence
certificate first, writes direct vanilla under a separate ceiling directory,
then trains and evaluates only the two planner rows.

A dry run from the repository root:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent_low_level.pt \
LATENT_SKILL_CHECKPOINT=/path/to/diffsr_skill.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla_tracker.pt \
MANIFEST=/path/to/corrected_one_motion_manifest.json \
DATASET_PATH=/path/to/matching_corrected_latent_dataset \
DRY_RUN=1 \
experiments/interface_baselines/run_focused_causal_interface_comparison.sh
```

A small local qualification:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent_low_level.pt \
LATENT_SKILL_CHECKPOINT=/path/to/diffsr_skill.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla_tracker.pt \
MANIFEST=/path/to/corrected_one_motion_manifest.json \
DATASET_PATH=/path/to/matching_corrected_latent_dataset \
NUM_ENVS=1 \
COLLECT_SAMPLES=1200 \
SAMPLE_BUDGET=1000 \
MODEL_SIZE=medium \
PRETRAIN_UPDATES=2000 \
FINETUNE_UPDATES=2000 \
EVAL_STEPS=1000 \
experiments/interface_baselines/run_focused_causal_interface_comparison.sh
```

Important defaults and checks:

- `CHUNK_STEPS=10` and `HORIZON_STEPS=10` are fixed.
- `STATE_HISTORY_STEPS=9` creates the required ten-frame state.
- `SAMPLE_BUDGET` must be an exact positive tensor-row count.
- `FORCE_COLLECT=1` prevents accidental reuse of stale sample files.
- Equivalence uses at least two environments and enough steps to cover all hold
  phases plus asynchronous republication.
- Demonstration and planner-rollout stages select exactly N rows each.
- The same planner model size and training settings are forwarded to both rows.

The result tree is organized as:

```text
logs/interface_baselines/focused_causal_interface/
  interface_comparison_run_provenance.json
  protocol_checks/
    streamed_vanilla_equivalence.json
  ceiling/
    direct_vanilla_50hz/
  planner_rows/
    latent_skill/
    full_body_streamed_vanilla/
  focused_protocol_audit.json
```

The final audit rejects a result if the rows differ in causal observation,
planner backbone, training budgets, sample stages, seed, rates, evaluation
length, or tracker provenance. It also verifies `[580, 30, 60]` target
packing, the direct-ceiling role, and the streamed-equivalence certificate.

## Frozen Vanilla Tracker

The direct ceiling, oracle packet, sample collector, and learned explicit
planner evaluation must all identify the same tracker SHA-256. The loader must:

- restore only `policy_state_dict`;
- restore strictly;
- reject unsupported hidden normalizer state;
- freeze all parameters;
- run in evaluation mode; and
- record the exact ordered actor input keys.

The equivalence certificate compares all actor inputs and deterministic actions
for packet slots 0 through 9, checks that actor state is unchanged, and
exercises asynchronously phased environments. Do not collect paper-facing
explicit samples without a passing certificate.

## Qualification and Scaling

Run gates in this order:

1. Audit the corrected motion manifest and rebuild stale caches.
2. Evaluate direct vanilla at 50 Hz with strict terminations.
3. Certify streamed vanilla equivalence.
4. Evaluate the DiffSR oracle.
5. Run a local smoke and an approximately 10M-frame qualification.
6. Train tiny or one-motion planners only after both low-level oracles are
   credible.
7. Freeze the protocol, then use Skynet for large final verification and paper
   numbers.

Local smoke and 10M runs are not paper results. The current local 10M vanilla
tracker matched direct and streamed wiring but survived only roughly 6.4 strict
steps in the focused diagnostic. Improve the vanilla tracker before interpreting
planner quality.

## Data Requirements

Do not use legacy LAFAN1 NPZs with Isaac scene-grid offsets in
`body_pos_w`. Run:

```bash
pixi run python scripts/audit_g1_lafan1_body_frames.py \
    --manifest /path/to/corrected_manifest.json \
    --report /tmp/lafan1_body_frame_audit.json
```

Use a separately repaired or freshly exported tree, then rebuild every derived
cache. Record manifest and data hashes with the run. Cluster snapshots exclude
ordinary local data paths, so final Skynet runs must use explicit persistent
paths after local gates pass.

Phase 5 uses BONES-SEED language annotations. Follow
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md):
generate a fresh output tree, record input/output hashes and exact commands,
require body names, run the audit, and rebuild caches. Do not treat an in-place
replacement with unknown provenance as paper data.

## Reporting

Keep the direct ceiling visually separate from planner rows. For DiffSR and the
explicit packet, report:

- oracle and learned-planner success, survival, and termination causes;
- root-relative MPJPE, root, joint, and EE tracking errors;
- action smoothness;
- performance as a fraction of the row's oracle;
- exact sample rows and optimizer updates;
- planner parameters, inference latency, output dimension, and bandwidth;
- pretraining versus planner-rollout-retraining performance;
- manifest, checkpoint, and code provenance.

A strong explicit oracle and weaker learned explicit planner are not
contradictory: they show that the command is expressive but burdensome to
predict. A weak explicit oracle invalidates the adapter or low-level tracker
and must be fixed before making that argument.

## Historical Broad Workflows

The repository still contains older runners for EE chunks, native chunk
controllers, Future-CVAE, and per-step tokens. They were useful for developing
the causal sample schema, planner-driven collection, merge/retrain pipeline,
and common metrics. Earlier results using
`current_achieved_macro_transition_batch`, different history semantics,
separately trained chunk policies, or weak low-level oracle checkpoints are
diagnostic only.

Do not use the following broad wrappers to define the main paper grid:

- `run_dance102_fair_interface_comparison.sh`;
- `run_phase2_shared_continuous_comparison.sh`;
- `run_phase3_latent_action_comparison.sh`;
- broad EE/full-body capacity and held-out sweeps.

They may still support a deliberately scoped appendix or debugging task. Label
style-inspired rows as such unless the corresponding released literature
components and native settings are reproduced.

## Fast Documentation and Orchestration Checks

Check shell syntax and preview the focused commands:

```bash
bash -n experiments/interface_baselines/run_focused_causal_interface_comparison.sh

LATENT_LOW_LEVEL_CHECKPOINT=/tmp/latent_low.pt \
LATENT_SKILL_CHECKPOINT=/tmp/skill.pt \
VANILLA_TRACKER_CHECKPOINT=/tmp/vanilla.pt \
MANIFEST=/tmp/corrected_manifest.json \
DATASET_PATH=/tmp/corrected_latent_dataset \
DRY_RUN=1 \
experiments/interface_baselines/run_focused_causal_interface_comparison.sh
```

Do not commit generated logs, checkpoints, videos, sample tensors, equivalence
artifacts, or derived caches.
