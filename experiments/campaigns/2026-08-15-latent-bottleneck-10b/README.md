# 2026-08-15 — Latent bottleneck at 10B frames

Twelve latent interfaces trained to 10B frames on ICE, all on the same trunk,
the same 129,785-motion data, and the same tuned low-level contract, so only the
bottleneck and its training objective vary. `campaign.yaml` carries the exact
arm contracts; `submit.sh` plans and submits them.

Two command interfaces carry the arms, both proven outside the 64-D hold-1 dead
zone (2026-08-15 grid):

- **hold 10, 64-D + `sin_cos`** — 66-D command, the SONIC token space.
- **hold 1, 256-D + `sin_cos`** — 258-D command.

## Evaluation, 2026-08-17

Every finished arm was pulled off ICE and scored locally on two boards. The
mirror keeps each checkpoint under its TRUE cumulative frame count, because the
segment step counter restarts on every chained segment.

### Isaac / Newton, 4,096 motions — the deciding board

`./eval_scoreboard4096.sh`. Protocol copied from
`2026-08-08-bones129k-4096-scoreboard/run.sh`: 4,096 environments, ranks
12288-16383 pinned, frame-0 starts, seed 0, mode actions, `no_push`,
Newton/MJWarp, released-SONIC thresholds, `foot_pos_xyz` and `base_too_low`
disabled. About 4 minutes per arm on the workstation RTX PRO 6000.

| arm | frames | SR | succ MPJPE-L | `ee_body_pos` | `anchor_ori` | `anchor_pos` |
|---|---:|---:|---:|---:|---:|---:|
| `cont_det_ln_hold1` | 10.00B | **0.9368** | 22.86 mm | 217 | 36 | 14 |
| `cont_det_hold1` | 10.00B | 0.9343 | 22.60 mm | 235 | 36 | 11 |
| `cont_det_hold1_resetramp` | 10.00B | 0.9307 | 23.84 mm | 248 | 31 | 13 |
| `jepa_sigreg_ebm_hold10_256d` | 10.00B | 0.9282 | **22.26 mm** | 234 | 61 | 12 |
| `fsq64_hold10` | 10.00B | 0.9197 | 24.93 mm | 260 | 64 | 18 |
| `jepa_sigreg_ebm_hold10_fsq64` | 10.00B | 0.9197 | 25.83 mm | 270 | 48 | 19 |
| `jepa_ntp_hold10_256d` | 8.50B | 0.9077 | 25.71 mm | 307 | 67 | 20 |
| `jepa_pure_256d_hold1` | 10.00B | 0.9050 | 27.94 mm | 336 | 43 | 20 |

Reference rows from the 2026-08-09 table, same protocol and ranks:
`root_qpos_explicit` 0.9358 / 19.21 mm at 7.60B, best previous latent arm
`critic_no_latent` 0.9062 / 24.39 mm at 5.00B, released SONIC 0.9937 /
28.65 mm.

**Every arm here beats the best previous latent arm on both axes**, and the top
four reach or pass the explicit baseline's success rate. The explicit row still
wins MPJPE by 15-18%, and it has 7.60B frames against these 10.00B, so the SR
comparison is not frame-matched in either direction: read it as "the latent
interface has caught up on falls", not as a win.

`ee_body_pos` remains the dominant failure in every row, exactly as the
2026-08-09 attribution found. The spread across the eight arms (217-336) is
larger than the spread in either tracking metric, so end-effector height is
still where the arms differ most.

Excluded on purpose: `fsq64_hold10_dyn` and `cont_det_hold1_resetramp_dyn`
fine-tune the encoder inside the tracker checkpoint (measured 0.739 max abs
divergence from the pretrained encoder), so the runner's pretrained-encoder path
would score a mismatched pair. They need the bundle path or an
encoder-from-checkpoint mode. `gumbel_multicat*` and `group_vq*` have no bundle
exporter for a learned codebook, so they have no EC row either.

### EC / MuJoCo, 10 motions — the CPU screen

Level 2 of the evaluation overhaul: `sidecar_ec_v1`, board
`selected10_repeats5_v1` (10 motions x 5 repeats = 50 episodes), sync lockstep,
SONIC sensor noise, fall-only termination at base height 0.4 m. One board takes
about a minute on 24 CPUs. Bundles are exported per checkpoint with the arm's
own hold, because the preset defaults do not match every arm.

Final-checkpoint rows (MPJPE-L over all valid transitions, then over fall-free
episodes only):

| arm | frames | MPJPE-L | MPJPE-L (fall-free only) | fall-free |
|---|---:|---:|---:|---:|
| `cont_det_ln_hold1` | 10.00B | **14.77 mm** | **13.24 mm** | 0.82 |
| `cont_det_hold1_resetramp` | 10.00B | 16.94 mm | 14.21 mm | 0.80 |
| `fsq64_hold10` | 10.00B | 19.77 mm | 17.25 mm | 0.80 |
| `jepa_sigreg_ebm_hold10_256d` | 10.00B | 23.08 mm | 17.61 mm | 0.78 |
| `cont_det_hold1` | 10.00B | 24.58 mm | 14.49 mm | 0.72 |
| `jepa_ntp_hold10_256d` | 8.50B | 24.44 mm | 18.92 mm | 0.76 |
| `jepa_sigreg_ebm_hold10_fsq64` | 10.00B | 25.43 mm | 19.40 mm | 0.84 |
| `jepa_pure_256d_hold1` | 10.00B | 27.30 mm | 21.69 mm | 0.84 |
| `fsq64_sonic_4500m` (bundle baseline) | 4.50B | 20.85 mm | 18.09 mm | 0.74 |
| `rollout24_gamma097_3500m` (bundle baseline) | 3.50B | 19.81 mm | 17.86 mm | 0.72 |

The `fsq64_sonic_4500m` row reproduces its stored number to the digit, which is
the lockstep determinism check.

`cont_det_ln_hold1` leads this board at every one of its five checkpoints from
8.0B to 10.0B (14.77-15.60 mm), so its lead is the arm and not one board draw.

### What the two boards agree and disagree about

Spearman rank correlation across the eight arms scored on both boards:

| EC quantity | 4,096 quantity | Spearman |
|---|---|---:|
| MPJPE-L, fall-free episodes | success-only MPJPE-L | +0.690 |
| MPJPE-L, all transitions | success-only MPJPE-L | +0.548 |
| MPJPE-L, all transitions | success rate (inverted) | +0.690 |
| fall-free rate | success rate | **-0.238** |

The EC board's tracking axis carries a moderate signal about the deciding
board. **Its survival axis carries none.** That is structural, not bad luck: the
EC board ends an episode only when the pelvis drops below 0.4 m, while the
scoreboard ends it on the SONIC thresholds that `ee_body_pos` dominates, and the
ten EC motions are quiet standing and manipulation clips whose falls concentrate
in three of them.

None of the ten EC reference motions goes near the fall threshold — the lowest
reference pelvis height on the whole board is 0.619 m against a 0.4 m trigger —
so the falls it reports are real collapses, not low reference poses.

### Would a different subset be a better screen?

Measured on the per-motion records of the eight scored arms, by resampling
subsets of the 4,096 ranks and re-ranking the arms from the subset alone
(median of 20 draws):

| subset | Spearman vs full board, SR | vs full board, MPJPE |
|---|---:|---:|
| random 10 | -0.333 | +0.833 |
| random 32 | +0.262 | +0.881 |
| random 64 | +0.429 | +0.905 |
| random 256 | +0.833 | +0.976 |
| **64, stratified by difficulty** | **+0.881** | **+0.881** |

3,545 of the 4,096 motions are passed by all eight arms, so a small random
subset is nearly all easy motions and says almost nothing about success rate.
Stratifying by how many arms fail a motion fixes that at 64 motions, which is
the same CPU cost class as today's 50-episode board.

Caveat: the difficulty strata come from these same eight arms, so a board built
from them is mildly fitted to them. Refresh the strata as more arms land, or
freeze them on a larger arm pool first.

### The replacement board, built and measured (2026-08-17)

`sidecar_ec_strat64_v1`: 64 motions drawn from the scoreboard ranks (seven or
eight per difficulty bucket, `random.seed(20260817)`), three noise draws each,
192 episodes, ~3.6 minutes of CPU per checkpoint. Success is the released SONIC
threshold set that Embodied-Control already evaluates per rollout
(`ec_sonic_rehearsal_v1`), not pelvis-below-0.4 m. Motions and reference arrays
live in `data/bones_seed_strat64_v1`; the rank list, the bucket populations, and
the per-case `population_weight` are frozen in `evaluation/protocol.py`.

Because the board over-samples hard motions, only the **population-weighted**
figures are comparable with anything; the raw board mean is not.

| arm (10.0B unless noted) | 4096 SR | 4096 succ MPJPE | strat64 weighted SR | strat64 weighted MPJPE |
|---|---:|---:|---:|---:|
| `cont_det_ln_hold1` | 0.9368 | 22.86 mm | 0.7814 | 24.65 mm |
| `cont_det_hold1` | 0.9343 | 22.60 mm | 0.7445 | 23.82 mm |
| `cont_det_hold1_resetramp` | 0.9307 | 23.84 mm | 0.7733 | 27.02 mm |
| `jepa_sigreg_ebm_hold10_256d` | 0.9282 | 22.26 mm | 0.7439 | 26.73 mm |
| `fsq64_hold10` | 0.9197 | 24.93 mm | **0.5511** | 29.15 mm |
| `jepa_sigreg_ebm_hold10_fsq64` | 0.9197 | 25.83 mm | 0.7355 | 32.31 mm |
| `jepa_ntp_hold10_256d` (8.5B) | 0.9077 | 25.71 mm | 0.7270 | 29.33 mm |
| `jepa_pure_256d_hold1` | 0.9050 | 27.94 mm | 0.7674 | 37.17 mm |

Agreement with the deciding board, over these eight arms:

| screen quantity | vs 4,096 board | old 10-motion board | new stratified board |
|---|---|---:|---:|
| survival axis | success rate | -0.238 | **+0.571** |
| quality axis | success-only MPJPE-L | +0.690 | **+0.929** |

The quality axis is now strong. The survival axis improved from useless to
moderate, but it did not reach the +0.88 the Isaac-only resampling predicted,
and the reason is visible in the table: **the residual disagreement is between
the two simulators, not between the board and its population.** `fsq64_hold10`
scores 0.55 weighted success in MuJoCo under sensor noise against 0.92 in Isaac
— the largest arm-level backend gap in the set, and the single row responsible
for most of the lost rank correlation. That is a finding about the arm, not a
defect of the sampling: under the pending decision to treat MuJoCo as the
deployment ground truth, "the FSQ hold-10 arm is fragile under noise" is a claim
worth testing directly rather than averaging away.

## The reset curriculum (2026-08-17)

"Reset curriculum" here is the SONIC-style adaptive-reset ramp: the share of
resets that place the robot at a uniformly random point of a random trajectory
sweeps 0.8 -> 0.2 across 2.5B frames, so the complementary share drawn from the
learned failure distribution rises 20% -> 80%, and is then pinned at 0.2. Two
arms use it: `cont_det_hold1_resetramp` (control: `cont_det_hold1`, identical in
every other respect) and `cont_det_hold1_resetramp_dyn`.

**Verdict on the ramp: not established, and left that way.** At 10.0B against
its control, one seed:

| board | ramp | control | ramp vs control |
|---|---:|---:|---|
| 4,096 SR | 0.9307 | 0.9343 | 0.4% worse |
| 4,096 success-only MPJPE-L | 23.84 mm | 22.60 mm | 5.5% worse |
| strat64 weighted SR | 0.7733 | 0.7445 | 3.9% better |
| strat64 weighted MPJPE | 27.02 mm | 23.82 mm | 13% worse |

Mixed sign across boards and every difference under the ~15% evaluation-noise
band at one seed. The ramp is not what separates the leading arms.

**Defect found in `cont_det_hold1_resetramp_dyn`, fixed by resubmission.** The
arm inherited the shared `segmented_stages` anchor, whose segments 2-4 carried
no `curriculum_hold_block`, while the base arm has custom stages that append it.
Ramp progress is `common_step_counter * num_envs`
(`mdp/commands/reference.py:538`) and nothing restores that counter across a
resume, so every segment re-swept 0.8 -> 0.2 from the start. Confirmed in the
frozen `batch_lowlevel*.sh` of the original plan, not just in the YAML.

Segment 1 ran to 5.0B, so its ramp completed there; segment 2 restarted the
sweep from 0.8. **The 5.0B-7.0B window of this arm's history was trained under a
re-swept reset schedule** — the uniform share drifted 0.8 -> about 0.32 across
those 2.0B frames while the arm it is paired with sat at 0.2. Jobs
5579717-5579719 were cancelled; the cancel's SIGTERM checkpoint landed at 7.0B,
and segments 2-4 were resubmitted as jobs 5580009-5580011 with the sampler
pinned. The resubmitted job resumed at 7,000,031,232 of the 10,000,269,312-frame
budget, so the remaining 3.0B trains on the correct schedule and the arm carries
a 2.0B contaminated window in the middle of its history. Prevention:
`vars.curriculum_hold_args` now exists, the shared segments 2-4 append it, and
any arm setting `curriculum_args` must set it too.

## Reproducing the evaluation

```bash
# 1. mirror the ICE checkpoints (file names carry the true cumulative frames)
#    -> logs/bottleneck_10b_mirror/<arm>_seed0/{encoder,tracker}
# 2. Isaac / Newton scoreboard
./experiments/campaigns/2026-08-15-latent-bottleneck-10b/eval_scoreboard4096.sh
# 3. EC / MuJoCo screen, one checkpoint
pixi run -e onnx-export python -m imitation_experiments.lowlevel.export_policy_bundle \
    --checkpoint <ckpt> --preset latent_v2 --output <bundle> \
    --skill-checkpoint <arm>/encoder/checkpoints/latest.pt --hold-steps 1
pixi run python -m imitation_experiments.evaluation.ec_tracker_sidecar run \
    --bundle <bundle> \
    --reference-root data/bones_seed_language10_v1/reference_arrays/root_qpos_v1 \
    --model external/Embodied-Control/assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
    --output-root logs/eval/bottleneck_10b --pixi-bin "$(command -v pixi)"
```

The reference arrays for the 4,096 board live at
`/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1`
(49 GB, pulled from the ICE shared allocation on 2026-08-17).

## Status

Still training on ICE as of 2026-08-17: `jepa_ntp_hold10_256d` segment 4,
`fsq64_hold10_dyn` segment 2, `cont_det_hold1_resetramp_dyn` segment 2, plus
four pending dependent segments. `group_vq64_hold10` and
`gumbel_multicat64_hold10` stopped early and have no 10B row.
