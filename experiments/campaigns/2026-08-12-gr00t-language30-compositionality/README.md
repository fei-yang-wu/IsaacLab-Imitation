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
