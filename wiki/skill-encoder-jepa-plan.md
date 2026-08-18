# Skill Encoder as a Temporal Abstraction: JEPA + SIGReg Plan

**Status: design only, written 2026-08-13. Nothing here has been measured, and
none of the prerequisites are built.** No number in this repo currently
supports or refutes any part of it.

## Where this comes from

On 2026-08-13 we fixed *which frame* the skill encoder reads its window in:
`env.expert_macro_anchor_mode=robot_heading` expresses the live window in the
robot's heading frame, matching SONIC v1.1 (see
`experiments/campaigns/2026-08-13-sonic-frame-parity/`). That leaves two
further questions untouched:

- **which data** the encoder is fit on, and
- **what the objective is**.

This page is the plan for those two. It is deliberately separate from the
frame work, which is already qualified and must not be bundled into the same
comparison.

## Baseline to beat

The current skill encoder is pretrained offline and then frozen:

- input: the macro state at the current slot (38 wide for the root-qpos
  interface) plus the *intermediate* slots of the window — the endpoint is
  hidden from the encoder on purpose, so the code cannot copy the answer;
- objective: denoising prediction of the endpoint macro state one horizon
  later, given (current state, code). No reward, no policy gradient;
- deployment: the tracker consumes the code as its command; the planner
  regresses the code from a language goal.

Shape: **state → chunk → state**.

## The three changes

They are independent and must be sequenced, not stacked.

### 1. Data — add robot-produced windows

Fit the encoder on windows of motion the robot actually produced, alongside
expert reference windows. The command the tracker receives during training
stays the expert-encoded code; the robot data enters only the encoder's
training distribution.

Rationale: it is one transition law — a humanoid under physics. The simulated
rollout is the sample that is dynamically feasible; the mocap retarget is the
approximation. A predictive objective requires no expert-ness from its data,
only that a window and its continuation come from one continuous trajectory.
Robot trajectories satisfy that by construction, and they cover the
neighbourhood the encoder is actually queried on at rollout.

Not to be confused with `current_achieved_macro_transition_batch`, which
replaces only the current state and keeps the future window and target
expert-derived. A genuine achieved *window* does not exist in either path
today.

### 2. Structure — chunk → chunk → chunk, one encoder

Three consecutive chunks in time: the preceding chunk as context, the executed
chunk encoded into the code, the following chunk as the prediction target. One
encoder serves all three roles, so the state representation and the skill
representation become the same object: a temporal abstraction of states.

This is an options view at chunk granularity — state is the code of the
preceding chunk, action is the code of the executed chunk, next state is the
code of the following chunk. The payoff is that the predictor becomes a
**dynamics model in the space the planner acts in**, which the endpoint form
cannot provide. It also matches the control loop: one code already governs ten
control steps, so the chunk is the natural unit.

The tracker still receives a vector and the planner still regresses that
vector, so no interface moves. What moves is the meaning of the code space.

### 3. Objective — JEPA with SIGReg

Predict the *embedding* of the next chunk rather than its raw values, with
SIGReg as the anti-collapse term.

The collapse risk is real and specific: the encoder appears on both sides of
the objective, so a constant code is a global optimum. Today we are immune only
because the prediction target is raw macro state, which cannot move. Policy
gradient is deliberately excluded (tried, no benefit, and it distorts what the
code means), so that anchor is unavailable too.

SIGReg is the chosen anchor: push the embedding distribution toward an
isotropic Gaussian using sketched one-dimensional random projections plus a
characteristic-function goodness-of-fit test, at linear cost in batch and
dimension, with a single trade-off weight. It is preferred over the
alternatives because it needs no teacher network, no EMA schedule, and no
stop-gradient asymmetry — it works with one encoder used on both sides, which
is exactly our situation.

**Before implementing, confirm against the LeJEPA paper**: the exact test
statistic, the number of random projections, and how the weight interacts with
batch size. Those details decide whether the term does anything.

Alternatives kept on the shelf, not chosen:

- reconstruction of the raw chunk as the anchor — cannot collapse, keeps the
  code tied to motion, and we already have the head and metrics;
- contrastive prediction in code space — already selectable in the repo, the SR
  registry carries `speder` (a factorized NCE bilinear model) next to `diffsr`;
- EMA target encoder with stop-gradient — the standard JEPA answer, most
  machinery, hardest failure mode to see from a loss curve.

Reconstruction is cheap enough that it is worth keeping as a *diagnostic* even
when SIGReg carries the anti-collapse load.

## Prerequisite: recording robot motion

None of this runs without a record of the robot's own trajectory.

**What to store, per step, per environment:** robot joint positions (29), robot
anchor position in the environment frame (3), robot anchor orientation as a
quaternion (4) — 36 numbers.

**Store the pose raw, not pre-anchored.** A window is anchored at one slot for
all of its slots; a row already anchored to its own step cannot be re-anchored
into a shared window frame. With raw poses, the achieved window goes through
the identical heading-anchor code path as the expert window, so the two halves
match mechanically rather than by discipline.

**How much retention**, for tiled (non-overlapping) chunks of length 10:

| macro stride | steps needed for three chunks |
| --- | ---: |
| 1 | ~31 |
| 5 | ~150 |

Consequences: slicing windows out of a 24-step rollout segment does not work
for the chunk form at any stride. An observation group with `history_length`
(the mechanism the SONIC surfaces already use) is the simplest option and costs
history × 36 floats on every stored transition. A ring buffer in the data
plane, sampled at update time, is the only shape that survives stride 5 —
roughly `num_envs × ring_length × 36` floats total, independent of rollout
length.

**Validity:** a steps-since-reset counter must travel with the group. History
buffers are reset with the environment, so early-episode rows are padding.
Windows that straddle a reset or contain padding must be dropped. Windows that
end in a fall are legitimate data and should be kept.

The recording is decision-independent: it serves both online training from the
replay buffer and dumping to a collection for offline pretraining. Build it
first.

## Gates, all offline, before any tracker is retrained

- **Copy baseline.** The predicted next code must beat copy-the-previous-code
  by a clear margin on *tiled* chunks. Under sliding windows adjacent chunks
  share most of their content, so the task is nearly trivial and a good loss
  proves nothing. If it does not beat copy, the dynamics model is vacuous.
- **Informativeness.** The existing window-probe control — prediction error
  given the true code versus a shuffled code. Report it separately for expert
  windows and robot windows so a gain on one hiding a loss on the other is
  visible.
- **Embedding health.** Per-dimension variance, effective rank (already logged
  as `z_effective_rank`), and the Gaussianity statistic SIGReg itself
  computes.
- **Reconstruction quality**, as a diagnostic that the code still describes
  motion rather than only what is convenient to predict.

## Costs and consequences

- **New code space.** Existing encoders do not transfer. The planner's
  regression targets were produced by one specific encoder and must be
  recollected — currently on the order of 889k rows.
- **Provenance.** The objective, the data mix, and the chunk layout must be
  recorded in the skill checkpoint alongside the frame convention and the
  stride. The macro width does not change under any of them, so nothing
  downstream could catch a mispairing on shape. The existing pairing guard is
  the template.
- **Online training moves the code space** whichever loss does it, which costs
  the binding record between the skill checkpoint and the encoder embedded in
  the tracker. Offline pretraining followed by freezing avoids this entirely
  and is the default unless there is a reason to give it up.
- **Deliberately excluded:** policy gradient into the encoder.

## Sequencing

| phase | change | gate |
| --- | --- | --- |
| 0 | recording group + window builder | windows reproduce expert-path values on expert data; padding and reset-straddling windows are dropped |
| 1 | data only: robot windows into the current objective | offline probe metrics on both sources |
| 2 | structure: chunk → chunk → chunk, raw target | beats the copy baseline on tiled chunks |
| 3 | objective: JEPA + SIGReg | embedding health plus the phase-2 gates |

Each phase is single-variable against the one before it. Nothing here enters a
paper claim without repeated seeds, a matched protocol, and a completed run.

## Open questions

- Chunk length versus hold length. Hold 10 and chunk 10 line up naturally;
  stride 5 changes the span without changing the width.
- Mixing ratio between expert and robot windows.
- Whether the predictor trains alongside the encoder or stays frozen — today's
  finetune path freezes it by default.
- Whether reconstruction stays in the loss alongside SIGReg or is demoted to a
  diagnostic.
- Predictor form for a possible phase 4: a deterministic regressor averages
  multimodal continuations; a denoising predictor over the next code, a
  masked discrete diffusion model over fsq64 code sequences (diffusion-LM
  style, enables goal-conditioned infilling), or a diffusion-forcing
  code-sequence model are the alternatives. See
  [skill-encoder-jepa-related-work.md](skill-encoder-jepa-related-work.md)
  section E.

## Related

- [Related work for this plan](skill-encoder-jepa-related-work.md) —
  method-level comparison against latent action models, next-token
  trajectory-code models, the JEPA family, diffusion language models,
  humanoid deployed-latent systems, and the state/skill unification lineage.
- `experiments/campaigns/2026-08-13-sonic-frame-parity/` — the frame fix this
  builds on, locally qualified, cluster arm deferred.
- [IPMD Representation Learning](ipmd-representation-learning.md)
- [Latent Learning Ablation Plan](latent-learning-ablation-plan.md)
- [Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md) —
  the paper contract this must not silently change.
