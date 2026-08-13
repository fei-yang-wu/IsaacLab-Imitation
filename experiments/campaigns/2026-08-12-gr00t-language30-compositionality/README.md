# 2026-08-12 — GR00T heads on the 30-motion compositionality subset

Status: ACTIVE. Training running on the local workstation (MEL07876D).

## Purpose

Train the verbatim GR00T N1.7 action head as a language-conditioned planner
over the 30 BONES-SEED compositionality motions, in three command
interfaces, all from replay data; evaluate in Isaac (number of record) and
on Embodied-Control (deployment rehearsal); then test whether motions can
be chained through language alone.

Terms used here:

- **replay data** — collected with `env.replay_only=true`: the robot state
  is written from the kinematic reference each control step, so the causal
  sensor pipeline reports expert kinematics and no policy dynamics enter
  the state. One environment per motion, one trajectory each (replay is
  deterministic, so repeats add nothing).
- **horizon 30** — every arm's head predicts 30 control steps of future.
  For the explicit arm that is literally `[30, 38]` root_qpos frames; for
  the latent arms it is 3 published latents held 10 steps each.

## Arms

| arm | head target | tracker at eval |
| --- | --- | --- |
| `z256` | 3 consecutive DiffSR latents `[3, 256]`, hold 10 | `rollout24_gamma097` |
| `explicit` | expert `root_qpos` window `[30, 38]` | `root_qpos_explicit` (7.6B) |
| `fsq64` | 3 consecutive FSQ pre-quantization vectors `[3, 64]` | `fsq64_sonic` |

Recipe for all three: warm start from `nvidia/GR00T-N1.7-3B`, GR00T
finetune defaults (AdamW fused, lr 1e-4, cosine + warmup 0.05, batch 64,
bf16, state_dropout 0.2, projectors + DiT together), 12,000 updates,
checkpoints every 4,000. W&B group `gr00t-language30-compositionality`.

## Data

Fresh replay collections were required: the existing 30-motion tree
(`logs/bones_language30_compositionality_oracle_seed0`) is oracle-**rollout**
data, not replay.

| collection | rows | encoder | purpose |
| --- | ---: | --- | --- |
| `logs/gr00t_language30_replay_z256/collection` | 1,455 | `rollout24_gamma097` | z256 latent targets + explicit targets |
| `logs/gr00t_language30_replay_fsq64/collection` | 1,455 | `fsq64_scaled` | FSQ pre-quant targets |

Both store a 30-frame expert `root_qpos` lookahead at every publication
boundary (control step 0, 10, 20, …). Language: Cosmos-Reason2-2B features
for all 30 goals at `outputs/gr00t_language30/goal_features/`.

### Anchor / horizon-30 finding (why no extra re-collection was needed)

Each stored 30-frame window is expressed in the anchor frame of **its own
publication**, so an explicit horizon-30 target is directly available — all
30 frames already share one anchor. Latent slots at +10 and +20 need a
window anchored at *their* publication, which the row join supplies
(rows exist at exactly those control steps; slot valid fractions
1.00 / 0.979 / 0.959). Re-encoding a shifted window would be wrong.

### Known limitation: `last_action` is tracker-specific

Verified on these two collections: the reference windows, joint positions,
joint velocities, base angular velocity and projected gravity are
**byte-identical** between them, but the `last_action` block (29 of the 93
values per causal frame) differs by up to 14.8 — under replay the tracker
policy still runs, and each collection loaded a different tracker.

Consequence: a head trained on one collection sees a `last_action`
distribution from that tracker. The `explicit` arm is trained from the
z256 collection (the only source of horizon-30 explicit targets), so its
`last_action` block comes from `rollout24_gamma097`, not from the
`root_qpos_explicit` tracker it is evaluated against. `state_dropout 0.2`
partially covers this. If the explicit arm underperforms at eval, this is
the first thing to rule out — with a dedicated explicit-tracker replay
collection.

## Workflow

```bash
# 1. Language features (done)
pixi run -e gr00t python -m imitation_experiments.planner.cache_gr00t_goal_features \
    --language_sidecar data/bones_seed_language30_compositionality_v1/manifests/g1_bones_seed_language30_compositionality_v1_manifest_language.json \
    --output_dir outputs/gr00t_language30/goal_features --no-export_head_bundle

# 2. Replay collections (done)
./experiments/campaigns/2026-08-12-gr00t-language30-compositionality/collect_replay.sh z256
./experiments/campaigns/2026-08-12-gr00t-language30-compositionality/collect_replay.sh fsq64

# 3. Tables
pixi run python -m imitation_experiments.planner.prepare_gr00t_dataset \
    --config-dir <this>/conf --config-name prepare_z256    # and prepare_fsq64

# 4. Train one arm
./experiments/campaigns/2026-08-12-gr00t-language30-compositionality/train_arm.sh z256
```

## Language chaining v2: reference-free DIRECTIONAL criterion (2026-08-12)

Divergence alone cannot tell "switched to B" from "fell apart", so the
criterion now runs THREE episodes per pair — control (A throughout), switch
(A then B), target (B throughout) — and asks whether the switch run moved
toward the commanded motion. The target is aligned by time since goal onset
(switch tick `T+k` vs target tick `k`, both "k steps into B").

`identification_ratio = dist_to_A / dist_to_B`; above 1 means closer to the
commanded motion than to the one it was told to abandon. No reference motion
is read anywhere, as the goal switch leaves the reference behind.

| pair (A then B) | survived | ratio | d(A) | d(B) |
| --- | :---: | ---: | ---: | ---: |
| hurry_idle -> surrender_stop | yes | **2.99** | 0.078 | 0.026 |
| Neutral_stoop_down -> walk_arc_cw | yes | **1.70** | 0.135 | 0.079 |
| crossed_arms_idle -> jump_around | no | 1.29 | 0.255 | 0.197 |
| walk_ff_loop -> casual_greeting | no | **0.93** | 0.157 | 0.169 |

Reading:

- **3 of 4 pairs identify the commanded motion** (ratio > 1), and the two
  cleanest cases are also the two that survived — the robot ends up closer to
  what it was asked for than to what it was doing.
- **The one failure is diagnostic, not neutral.** walk -> greeting has ratio
  0.93: the robot moved away from walking WITHOUT arriving at greeting, and it
  fell. Under the old divergence-only criterion this looked like a success
  (0.157 rad of change); the directional test exposes it as a transition that
  destroyed the gait without producing the target behavior.
- Ratio and survival agree in ordering across all four pairs, which is weak
  evidence that a clean transition is also a stable one. n=4.

`dist_to_B` never reaches zero even for good chaining: the switch run enters B
from A's pose, the target run from the reset pose. The RATIO is the
interpretable quantity, not either distance.

## Language chaining v1 (2026-08-12): it works, and it costs stability

`imitation_experiments.evaluation.eval_gr00t_chaining`, arm `fsq64_rollout`
(the only arm top-two in both simulators), EC runtime, 600-step episodes,
language switched at tick 300. Each pair runs TWICE — a control that keeps
goal A throughout, and a switch run identical in every other respect. Neither
reads a reference, so post-switch divergence is attributable to the language
change alone.

| pair (A then B) | switch survived | divergence (rad) | pre-switch |
| --- | :---: | ---: | ---: |
| walk_ff_loop -> casual_greeting | no (damped) | 0.157 | 0.000 |
| crossed_arms_idle -> jump_around | no (damped) | 0.283 | 0.000 |
| Neutral_stoop_down -> walk_arc_cw | yes | 0.135 | 0.000 |
| hurry_idle -> surrender_stop | yes | 0.078 | 0.000 |

**All four CONTROL runs completed 600 steps without falling.** Combined with a
pre-switch divergence of exactly 0.000, that makes the causal claim clean: the
switch runs are bit-identical to their controls until tick 300, so both the
behavior change and the falls are caused by the language switch and nothing
else.

Two findings:

- **Language conditioning works mid-episode.** Divergence is 0.078-0.283 rad
  against an exactly-zero baseline. Changing only the goal embedding changes
  what the robot does, with no reference change and no replan trigger.
- **Chaining is not free: 2 of 4 switches caused a fall the control did not
  have.** The two that fell are also the two with the largest divergence,
  suggesting the bigger the commanded change the more destabilizing the
  transition — n=4, so that ordering is a hypothesis, not a result.

Method note, worth keeping: the first run of this experiment reported
`pre_switch = 0.078`, not zero. The head's flow sampler initializes from
`randn` and was unseeded, so two "identical" runs already differed and any
divergence number would have been noise. `gr00t_chunk_service --seed` now pins
it. The control-on-the-control is what caught this; without it the experiment
would have produced confident nonsense.

Not measured: chaining QUALITY. With the goal switched the reference does not
follow, so MPJPE and `reference_finished` would score against a motion the
robot is no longer asked to perform. That needs a paired reference schedule.

## EC rehearsal grid (2026-08-12) — and one arm that inverts

Four arms x 30 goals, batched (one process and one loaded head per variant via
`goal_sequence`), basic mode only (no RTC), `min_base_height_m=0.4` so an EC
failure is a genuine fall. **Every** failure in the grid was `base_too_low`.

| variant | EC survival | Isaac survival |
| --- | ---: | ---: |
| z256 replay | 18/30 (0.600) | 0.800 |
| z256 rollout | **8/30 (0.267)** | **0.900** |
| fsq64 rollout | 22/30 (0.733) | 0.933 |
| chunk rollout | 18/30 (0.600) | 0.967 |

EC is uniformly harsher than Isaac, which is expected and documented: MuJoCo
actuator dynamics differ from Newton by design, so EC is the rehearsal signal
and Isaac is the number of record. Absolute rates are not comparable across
the two.

What is NOT expected is `z256_rollout` **inverting**: Isaac's second-best
survivor is EC's worst by a wide margin, 22 of 30 falling. Every other arm
keeps its rough ordering. A sim-specific collapse in a single arm is the
signature of a policy leaning on Newton-specific dynamics rather than tracking
robustly — the same class of problem as the 2026-08-03 sim2sim verdict, which
traced its gap to actuator dynamics rather than command semantics. That arm
would not survive contact with a different simulator, let alone hardware,
despite looking strong on the record metric.

`fsq64_rollout` is the only arm top-two in BOTH simulators. The defensible
statement is that it is the most consistent, not that it wins: the two
simulators rank the arms differently and both are one seed.

## AVERAGED Isaac table — 150 episodes per arm (2026-08-12, current)

5 episodes per goal x 30 goals = 150 episodes, one process per arm (extra
episodes are extra ENVIRONMENTS, so averaging costs GPU memory rather than
another simulator start-up). Supersedes the 30-episode table below.

| interface | data | tracker | MPJPE | survival | falls |
| --- | --- | --- | ---: | ---: | ---: |
| z256 latent | replay | rollout24 | 84.0 | 0.800 | 30/150 |
| chunk -> encoder -> latent | replay | rollout24 | 99.8 | 0.713 | 43/150 |
| z256 latent | rollout | rollout24 | 64.7 | 0.933 | 10/150 |
| chunk -> encoder -> latent | rollout | rollout24 | 76.9 | 0.967 | 5/150 |
| fsq64 latent | replay | fsq64_sonic | 84.0 | 0.967 | 5/150 |
| fsq64 latent | rollout | fsq64_sonic | **58.0** | 0.933 | 10/150 |
| fsq64 latent | rollout re-encoded | fsq64_sonic | 94.1 | **0.973** | 4/150 |

**IMPORTANT NAMING CORRECTION.** The "chunk" rows are NOT the explicit
interface. They run `chunk_encoded`: the chunk head's 30 predicted root_qpos
frames are pushed through the frozen z256 encoder and published as a LATENT to
the z256 tracker. Verified from the run records — every chunk row's tracker is
`rollout24_gamma097`, and no run in this campaign has ever loaded
`root_qpos_explicit`. So these rows compare *predicting latents directly*
against *predicting frames that are then re-encoded into latents*, at a
matched tracker. They do NOT measure the explicit interface as SONIC or HuMI
deploy it.

Latent still leads at a matched tracker on MPJPE, on both data modes:
replay 84.0 vs 99.8 (16%), rollout 64.7 vs 76.9 (16%). Survival splits:
latent better on replay (0.800 vs 0.713), chunk better on rollout (0.967 vs
0.933).

Rollout data improves MPJPE within every interface: z256 84.0 -> 64.7 (23%),
chunk 99.8 -> 76.9 (23%), fsq64 84.0 -> 58.0 (31%).

The re-encoded arm keeps its signature: near-best survival (0.973) with the
second-worst MPJPE (94.1) — upright, not tracking.

### Why 150 episodes and not 30

Averaging changed one arm materially and left the rest alone:

| arm | 30-ep | 150-ep | shift |
| --- | ---: | ---: | ---: |
| z256 replay | 72.9 | 84.0 | **+15%** |
| chunk replay | 96.9 | 99.8 | +3% |
| z256 rollout | 64.2 | 64.7 | +1% |
| fsq64 replay | 84.2 | 84.0 | -0.2% |
| fsq64 rollout | 57.1 | 58.0 | +2% |

The single-episode figure was flattering z256 replay specifically, which is
why its replay->rollout gap doubled (12% -> 23%) once measured properly.
Repeated runs at 30 episodes had agreed to 0.5% on that arm and looked stable
— they varied the flow sampler while effectively holding the environment draw
fixed. The reset perturbation and per-environment domain randomization are the
larger variance source, and only more EPISODES sample them.

## SUPERSEDED: Isaac table at 30 episodes per arm (2026-08-12)

Protocol, set by the user on 2026-08-12 and now the launcher default: report
**root-relative MPJPE** and **success = survival = did not fall**
(`base_too_low` only). `--fall_only_success` additionally disables
`foot_pos_xyz`; without that, episodes end on foot tracking before the robot
can fall, survival saturates at 1.000, and step counts measure that
termination instead of tracking. Metrics sampled every 10 steps. 30 goals,
one environment per goal, seed 0, 500 steps.

| interface | data | tracker | MPJPE | survival | falls | steps | head calls |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| z256 latent | replay | rollout24 | 72.9 | 0.800 | 6 | 368 | 93 |
| chunk encoded | replay | rollout24 | 96.9 | 0.667 | 10 | 327 | 251 |
| z256 latent | rollout | rollout24 | 64.2 | 0.900 | 3 | 386 | 91 |
| chunk encoded | rollout | rollout24 | 78.8 | 0.967 | 1 | 401 | 197 |
| fsq64 latent | replay | fsq64_sonic | 84.2 | 0.967 | 1 | 401 | 78 |
| fsq64 latent | rollout | fsq64_sonic | **57.1** | 0.933 | 2 | 406 | 69 |
| fsq64 latent | rollout re-encoded | fsq64_sonic | 102.2 | **1.000** | 0 | 408 | 70 |

### Latent beats chunk on MPJPE at a matched tracker, both data modes

The two chunk rows share `rollout24_gamma097` with the two z256 rows, so only
the predicted target differs:

| data | latent MPJPE | chunk MPJPE | latent advantage |
| --- | ---: | ---: | ---: |
| replay | 72.9 | 96.9 | 33% |
| rollout | 64.2 | 78.8 | 23% |

Both gaps clear the ~15% noise bar and point the same way, on two independent
training sets. Latent also needs ~2.5x fewer head calls, because it consumes
three cached slots per prediction while chunk re-predicts every publication.

Survival does NOT agree: chunk is worse on replay (0.667 vs 0.800) but BETTER
on rollout (0.967 vs 0.900). So the defensible statement is that latent
tracks more accurately at equal tracker and equal data, while survival between
the two is unresolved at one seed.

### The two metrics disagree, and that is the point

`fsq64 rollout re-encoded` has **perfect survival (1.000, zero falls) and the
worst MPJPE in the table (102.2)**. Survival alone would rank it best of the
FSQ family; MPJPE shows it is the worst by 79% against fresh rollout. That is
the signature of a degenerate-but-safe policy: upright, not tracking. Either
metric on its own inverts the ranking — which is why both are required.

This also settles the re-encoding question: correct FSQ targets computed on
the z256 tracker's state distribution produce a planner that gives up on
tracking. The fresh collection was necessary, not an optimization.

### Rollout data improves tracking on every interface

Within-interface, tracker held fixed: z256 72.9 -> 64.2 (12%), fsq64 84.2 ->
57.1 (32%), chunk 96.9 -> 78.8 (19%). Consistent in sign across all three
interfaces; magnitude varies.

All of the above is one seed with one environment per goal. Seed repeats are
required before any of it becomes a claim.

## SUPERSEDED: earlier table (tracking-threshold success, no MPJPE)

30 goals, one environment per goal in a single process, 500 steps, M3
terminations, metrics sampled every 10 steps. Chunk rows use `chunk_encoded`,
so they share the latent rows' tracker and the only difference is what the
head predicts. **Survival is 1.00 with zero falls in every row** — at this
horizon the task set produces no falls, so survival carries no signal and the
comparison rests on tracking success and useful episode length.

| interface | data | completed | tracking | steps | macro MAE | head calls | p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z256 latent | replay | 0.233 | 0.400 | 240.6 | 0.347 | 202 | 28.9 ms |
| chunk | replay | 0.200 | 0.367 | 212.4 | 0.356 | 392 | 44.1 ms |
| z256 latent | rollout | 0.233 | 0.500 | 289.3 | 0.364 | 192 | 28.9 ms |
| chunk | rollout | 0.267 | 0.500 | 272.4 | 0.369 | 345 | 44.4 ms |
| fsq64 latent | replay | 0.233 | 0.500 | 275.7 | 0.351 | 184 | 28.9 ms |
| fsq64 latent | rollout | 0.333 | 0.600 | 317.1 | 0.355 | 153 | 28.9 ms |
| fsq64 latent | rollout re-encoded | 0.200 | 0.433 | 246.4 | 0.348 | 192 | 28.8 ms |

Chunk vs latent on a matched tracker, the milestone question:

- On **replay** data latent leads (0.400 vs 0.367 tracking, 241 vs 212 steps);
  on **rollout** data they tie exactly (0.500 each, 289 vs 272 steps). Both
  gaps are at or under the ~15% noise bar, so the honest statement is that
  **the interfaces are not separated by this experiment**.
- The one difference that is not noise is **cost**: chunk needs ~2x the head
  calls (392/345 vs 202/192) and ~1.5x the latency (44 vs 29 ms), because the
  latent route consumes three cached slots per prediction while chunk
  re-predicts every publication.
- Data mode moves results more than interface does: rollout beats replay for
  z256 (+25% relative) and fsq64 (+20%), and by tracking success the best row
  overall is fsq64 on fresh rollout data.

## First Isaac result: chunk vs latent on ONE tracker (2026-08-12)

Replay-trained arms, 30 goals, one environment per goal in a single process,
500 steps, seed 0, M3 terminations. Both rows drive the SAME
`rollout24_gamma097` tracker, so the only thing that differs is what the head
predicts.

| row | survival | completed success | tracking success | macro MAE (rad) | head calls | latency p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| z256 latent | 1.00 | 0.233 | 0.400 | 0.347 | 198 | 28.9 ms |
| chunk (chunk_encoded) | 1.00 | 0.200 | 0.400 | 0.356 | 394 | 44.1 ms |

Reading, all of it PRELIMINARY (one seed, one environment per goal):

- On a matched tracker the two interfaces are **indistinguishable on quality**:
  identical survival and tracking success, and a 2.7% macro-error difference
  that is far below the ~15% evaluation noise this repo treats as unresolved.
  The completed-success gap is one goal out of thirty.
- They differ on **cost**, and that difference is structural rather than
  noise: the latent arm consumes three cached slots per prediction while the
  chunk route re-predicts at every publication, so chunk pays **2x the head
  calls and ~1.5x the latency** for the same result.
- Nothing here separates the interfaces on capability. What it does show is
  that the earlier language-10 conclusion holds on a second dataset: once the
  tracker is held fixed, interface choice is second order.

Measurement note: an earlier version of this table was degenerate. With
`--metric_interval` set to the episode length, tracking metrics were sampled
once at step 0 — on the reset placement, where every error is exactly zero by
construction. The launcher now samples every 10 control steps (50 snapshots
per episode). Survival and success were never affected; they come from
termination bookkeeping.

Open gap: these macro-state joint errors are not MPJPE. The repo's headline
root-relative MPJPE is not in this summary, so cross-campaign tracking
comparisons need the EC MPJPE pass or an Isaac MPJPE metric term.

### Data mode moves the numbers; re-encoding does not substitute for it

All latent rows, 30 goals, seed 0, metrics sampled every 10 steps. Nothing
fell in any row (survival 1.00, zero falls everywhere), so survival does not
separate these arms at all — the signal is in tracking success and in how far
each episode gets.

| arm | data | completed | tracking | steps | macro MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| z256 | replay | 0.233 | 0.400 | 240.6 | 0.347 |
| z256 | rollout | 0.233 | 0.500 | 289.3 | 0.364 |
| fsq64 | replay | 0.233 | 0.500 | 275.7 | 0.351 |
| fsq64 | **rollout (fresh)** | **0.333** | **0.600** | **317.1** | 0.355 |
| fsq64 | rollout (re-encoded) | 0.200 | 0.433 | 246.4 | 0.348 |

Two readings, one that clears the noise bar and one that does not:

- **Fresh rollout data beats re-encoded rollout data**, 0.600 vs 0.433
  tracking success — a 28% relative gap, above the ~15% this repo treats as
  unresolved. Re-encoding the z256 rollout collection with the FSQ encoder
  gives correct per-row targets but leaves the state histories (and
  `last_action`) belonging to the z256 tracker's closed loop. That hybrid
  lands BELOW even the replay arm. So matched targets alone do not buy what
  matched dynamics buy, and the extra collection was worth running.
- **Rollout beats replay** on both interfaces (z256 0.400 -> 0.500, fsq64
  0.500 -> 0.600, both ~25% relative). Consistent in sign across interfaces,
  which is what makes it interesting, but it is still one seed.
- Re-encoded vs replay (0.433 vs 0.500, 13%) is **within noise** — do not
  read an ordering into it.

Macro MAE stays in a 0.347-0.364 band across every row, i.e. it does not
separate these arms; the differences live in how long an episode survives
usefully, not in instantaneous joint error.

### `chunk_native` is blocked on the entrypoint, not on the head

The publisher works — the fix needed was a vocabulary one: `ChunkCommandCfg.
components` takes COMMAND-space names (`joint_qpos`, `root_pos`, `root_ori`),
not the expert-macro-state term names (`expert_motion_qpos`, …) that
configure the encoder input. Same 38 values, different namespace.

With that corrected the run reaches a harder wall:
`eval_skill_commander_closed_loop.py` asserts a LATENT actor contract
(`supported_latent_policy_input_keys`, and a `('policy', 'latent_command')`
observation). Driving a genuinely explicit tracker such as
`root_qpos_explicit` needs the explicit path in
`imitation_experiments.evaluation.eval_interface_planner_closed_loop`, which
already understands the `root_qpos` interface. Wiring the GR00T chunk
publisher into THAT entrypoint is the remaining work for this row; nothing
about the head or the packet contract is in question.

## Evaluation plan

1. **Isaac closed-loop — the number of record.** M3 survival definition
   (`base_too_low` is a fall; `time_out` / `reference_finished` are
   successes), tracking errors as continuous metrics, plus the mandated
   full-horizon non-terminating diagnostic pass with a retained video whose
   absolute path is printed.
2. **Embodied-Control MuJoCo — deployment rehearsal**, reusing
   `imitation_experiments.evaluation.eval_gr00t_ec` with a 30-goal config.
3. **Motion chaining through language only** — last, and only after 1-2.
   Switch the language goal mid-episode with the reference held fixed, and
   measure whether the robot transitions. The runtime support
   (`Gr00tSpec.goal_schedule`, goal switching inside
   `gr00t_chunk_service`) is already in place.

## Normalization context (read before interpreting any number)

On the frozen 4,096-motion oracle board the three trackers are **not**
equal: `root_qpos_explicit` 0.9358 SR / 19.21 mm, `fsq64_sonic` 0.9038 /
25.44 mm, and `rollout24_gamma097` was never scored on that board. The
2026-08-11 language-10 campaign showed tracker choice dominates both
survival and precision, so planner numbers must be normalized by the
matching oracle before any interface claim is made.

## SONIC-style hold-1 planner (2026-08-12)

The released NVIDIA SONIC v1.1 tracker re-encodes its 64-D FSQ token on every
50 Hz control step, so a planner that replaces its encoder must emit one latent
per step. This arm predicts a 30-latent horizon and republishes every 10 control
steps, consuming one latent per step (receding horizon; slots 10-29 are
discarded). Contrast with the other 30-motion arms, which publish 3 latents each
held for 10 steps.

Target: the encoder's **bounded, lattice-scaled pre-quantization** value
(`FSQ.bound(z) / 16`), snapped onto the 32-level lattice at publication. The
snap moves a value by at most 1/32, and the deployed token is then exactly one
the encoder could have produced.

Scripts: `collect_sonic_rollout.sh` -> `conf/prepare_sonic_hold1.yaml` ->
`train_arm.sh sonic_hold1` -> `eval_sonic_planner.sh {oracle,planner}`.

Data: 900 environments (30 per motion) driven by the released tracker,
949,500 rows (one per control step), 898/900 completed, oracle MPJPE-L
26.63 mm. Collection is `logs/gr00t_language30_sonic_hold1/collection`.

Results, SONIC's own strict termination contract, 150 episodes (5 per goal):

| row | completed | MPJPE-L |
| --- | --- | --- |
| oracle (encoder-driven, 900 envs) | 898/900 | 26.63 mm |
| hold-1 planner, 12k updates | 100/150 | 43.45 mm |

Full-horizon diagnostic (all early terminations disabled, 30 episodes,
600 steps): MPJPE-L 44.53 mm over 23/30 scored environments, video at
`logs/gr00t_language30_sonic_eval/planner_diagnostic/videos/`. The diagnostic
and strict numbers agree to about 1 mm, so the strict-pass MPJPE is not
materially survivorship-biased.

These rows use SONIC's strict tracking terminations, NOT the M3 fall-only
definition used by the other 30-motion arms. Their success rates are not
comparable to the M3 survival column; the MPJPE values are comparable only in
the loose sense that both are success-only means.

### Matched-protocol comparison (M3 fall-only, Newton, 150 episodes)

Every row below: `--termination_contract fall_only` (tracking terminations off,
`base_too_low` the only failure), `physics=newton_mjwarp`, 5 episodes per goal
over the same 30 goals. This is the only table in which the SONIC rows and the
GR00T arm rows may be compared directly.

| system | tracker | hold | survival | MPJPE-L |
| --- | --- | --- | --- | --- |
| SONIC oracle (encoder-driven) | SONIC v1.1 | 1 | 150/150 | 25.41 mm |
| SONIC-style GR00T planner | SONIC v1.1 | 1 | 150/150 | 49.76 mm |
| GR00T fsq64, rollout data | ours fsq64 | 10 | 0.933 | 58.0 mm |
| GR00T z256, rollout data | ours z256 | 10 | 0.933 | 64.7 mm |
| GR00T chunk, rollout data | ours z256 | 10 | 0.967 | 76.9 mm |

The SONIC-tracker planner beats our best planner arm on both axes: 14% lower
MPJPE and no falls at all. Its oracle ceiling (25.41 mm) is HIGHER than our
z256 tracker's, so the planner advantage is not inherited from a better
low level.
