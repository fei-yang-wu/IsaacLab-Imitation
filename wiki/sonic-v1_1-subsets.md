# New common eval subset

Status: frozen and directly confirmed 2026-08-25; motion-name and targeted
visual review complete. Repeated evaluations remain open.

## Decision

The **new common eval subset** has the frozen artifact ID
`sonic_capability124_v1`. It is the only subset defined on this page. It is a
deliberately calibrated capability set, not a machine-learning validation set
and not a reconstruction of SONIC's unpublished evaluation split.

The purpose is practical:

1. The public `sonic_v1_1` checkpoint must score between 22 and 24 mm
   success-only micro MPJPE-L, near the 23.7 mm scale in SONIC Figure 2.E.
2. The clips must remain varied and physically usable instead of selecting
   only short, slow, upright motion.
3. After the set is frozen, every local tracker is scored on the identical
   ranks and protocol.
4. SONIC is the common reference for discussing baseline results that the
   SONIC authors measured but that we do not retrain.

The selection reads the public SONIC checkpoint's results on purpose. This
makes the set favorable to the anchor. That is acceptable for a calibration
set, but it must be stated whenever the set is reported.

## Frozen rank list

The 124 ranks are stored in
[`sonic_capability124_v1.json`](../experiments/campaigns/2026-08-25-sonic-paper-proxy/sonic_capability124_v1.json).

- Rank count: 124 unique ranks.
- Rank-list SHA-256:
  `19b83597f0e7bf86fb462ae691b1dad455bb6b8cc130a9a4c702062aa75de147`.
- Reference tree:
  `g1_bones_seed_sonic_full_129785_e714bbff_v1`.
- Reference content identity: `bones_seed_sonic_full_129785@e714bbff`.

Do not edit the rank list in place. A changed population requires a new board
name and a new selection record.

## Selection procedure

The source was the complete clean evaluation at
`logs/sonic_paper_proxy/sonic_v1_1_sonic_proxy_testrep4096_v1_rand_none.json`.
That parent run evaluated 4,096 environments, reached
`stop_reason=all_envs_done`, had `done_rate=1.0` and `time_out_rate=0.0`, and
used deterministic `mode` actions.

### Candidate pool

The candidate pool starts from those 4,096 evaluated ranks. A rank remains
eligible when all of the following hold:

- the reference length is 100 to 1,500 frames, or 2 to 30 seconds at 50 Hz;
- its name does not contain a scene or object token that requires absent
  terrain or a large prop;
- the clip is part of the SONIC-filtered BONES-SEED reference tree.

The exclusion tokens were the repository's
`TESTBED_EXCLUDED_NAME_TOKENS` plus:

```text
fridge cupboard obstacle plant plants baby apple eating painting item
shoe shoes dog leash feeding bird birds watering bump guitar drum bottle
glass sword gun umbrella vacuum
```

This left 3,732 candidates.

### Trial-and-error calibration

Each trial samples 124 candidates without replacement with
`random.Random(20260825 + trial_index)`. The first accepted draw must satisfy:

- SONIC success rate at least 0.98;
- success-only micro MPJPE-L from 23.5 to 23.9 mm, inside the user-approved
  22 to 24 mm band and close to 23.7 mm;
- at least 4 ground-motion clips: crouch, squat, kneel, crawl, all-fours, or
  grovel;
- at least 5 dance clips;
- at least 5 dynamic clips containing jump or kick;
- at least 5 injured-motion clips;
- at least 30 locomotion clips;
- at least 15 gesture clips;
- at least 1 boxing clip;
- at least 6 of the named deployment families recognized by
  `evaluation.sonic_paper_proxy`.

No MPJPE-G value and no result from one of our trackers enters selection.

The accepted draw was trial index 218, seed `20261043`. Its realized coverage
is:

| group | clips |
| --- | ---: |
| locomotion | 61 |
| gestures | 20 |
| dance | 12 |
| dynamic jump or kick | 12 |
| injured motion | 9 |
| ground motion | 4 |
| boxing | 1 |

These name-based groups can overlap. The six recognized deployment families
are boxing, crawl, crouch walk, hip-hop dance, kick, and kneel.

## Selection-time calibration result

The completed 4,096-clip parent result reduces to this row:

| checkpoint | clips completed | SR | MPJPE-L | MPJPE-G |
| --- | ---: | ---: | ---: | ---: |
| public `sonic_v1_1`, one clean evaluation | 124 / 124 | 1.0000 | 23.79 mm | 173.94 mm |

This is the selection-time result. The direct confirmation below reproduces
it. That run confirms:

- all 124 environments finish;
- `done_rate=1.0` and `time_out_rate=0.0`;
- `stop_reason=all_envs_done`;
- the rank hash matches the frozen JSON;
- MPJPE-L remains inside 22 to 24 mm;
- the randomization and termination metadata match the clean protocol below.

Reduce the stored parent result again with:

```bash
pixi run python -m imitation_experiments.evaluation.summarize_paper_boards \
  logs/sonic_paper_proxy/sonic_v1_1_sonic_proxy_testrep4096_v1_rand_none.json \
  --ranks_json experiments/campaigns/2026-08-25-sonic-paper-proxy/sonic_capability124_v1.json \
  --subset_label sonic_capability124_v1
```

## Direct confirmation and matched tracker result

The direct clean evaluation completed on 2026-08-25. Both rows use the same
ordered ranks, seed, frame-0 starts, SONIC termination contract, and no
randomization.

| checkpoint | successful clips | SR | MPJPE-L | MPJPE-G | qualification |
| --- | ---: | ---: | ---: | ---: | --- |
| public `sonic_v1_1` | 124 / 124 | 1.0000 | 23.79 mm | 173.92 mm | one clean evaluation |
| our `ln_hold1_sonicreset` at 30B frames | 123 / 124 | 0.9919 | 19.44 mm | 108.58 mm | seed 0, one clean evaluation, preliminary |
| our `ln_hold1_sonicreset` at 46.5B frames | 124 / 124 | 1.0000 | 19.44 mm | 122.08 mm | seed 0, one clean evaluation, mid-chain progress read |

The 46.5B row was added on 2026-08-26. It is the newest checkpoint of the
running 50B chain, not the 50B promotion row. Score the 50B checkpoint when
the chain reaches its cap and report that row separately.

Success-only errors in these rows use different successful populations. On the
123 clips that all three complete, the matched result is:

| checkpoint | clips | MPJPE-L | MPJPE-G |
| --- | ---: | ---: | ---: |
| public `sonic_v1_1` | 123 | 23.43 mm | 148.37 mm |
| our `ln_hold1_sonicreset` at 30B frames | 123 | 19.44 mm | 108.58 mm |
| our `ln_hold1_sonicreset` at 46.5B frames | 123 | 18.99 mm | 97.56 mm |

Between our own two checkpoints the matched change is 2.3% on MPJPE-L and
10.1% on MPJPE-G. Both are inside the unresolved band for one seed and one
evaluation. The firm change is rank 6364, `kneeling_loop_003_A244`: the 30B
row's only failure, completed at 46.5B.

Against public `sonic_v1_1`, the local difference is about 17% and the global
difference about 27% at 30B, and about 19% and 34% at 46.5B, each on one
evaluation. Treat the tracker comparison as preliminary until a
repeat and additional training seeds support it. The new common eval subset was
selected from the SONIC result, so it demonstrates capability on a common
calibrated population; it does not establish an unbiased advantage. The 30B
row's only failure was rank 6364, `kneeling_loop_003_A244`, on `anchor_pos`;
the 46.5B row completes every clip.

## Velocity and acceleration distance (2026-08-26)

SONIC's paper also reports velocity and acceleration tracking distances.
`evaluate_sonic_release` did not accumulate them (it snapshots only the
command term's metrics), so the evaluator gained the same per-step
accumulation `evaluate_checkpoint` uses: over the identical 14 tracked links,
`tracking_velocity_distance_mps` is the mean over links of the world-frame
linear-velocity error norm, and `tracking_acceleration_distance_mps2`
finite-differences those same link velocities at the 0.02 s control step.
Active pre-termination steps only; the post-`step` state of a finishing
environment is already reset and is excluded.

The SONIC row required a re-run (`sonic_v1_1_clean_r2_velacc.json`), which
doubles as the first repeat: 1.0000 / 23.89 / 174.23 mm against the first
run's 1.0000 / 23.79 / 173.92 mm, so repeat noise on this board is about
0.1 mm L / 0.3 mm G. All 124 clips complete in both rows, so success-only and
matched populations coincide:

| checkpoint | clips | vel dist (m/s) | acc dist (m/s^2) | qualification |
| --- | ---: | ---: | ---: | --- |
| public `sonic_v1_1` | 124 | 0.165 | 2.89 | one clean evaluation (r2) |
| our `ln_hold1_sonicreset` at 46.5B | 124 | 0.175 | 4.67 | seed 0, one clean evaluation, mid-chain |

Both numbers are our-harness measurements under one shared definition; they
are not SONIC's own reported figures and use SI units, not the paper's.
Velocity distance is close (6% apart, one evaluation, unresolved). The
acceleration gap is large and one-sided: SONIC's rollout is markedly smoother
at the link level while ours tracks position more tightly. Consistent with a
hold-1 latent command refreshed every control step; treat the mechanism as a
hypothesis, not a finding.

Machine-readable artifacts:

- `logs/sonic_capability124_v1/sonic_v1_1_clean.json`;
- `logs/sonic_capability124_v1/ln_hold1_sonicreset_30b_clean.json`;
- `logs/sonic_capability124_v1/ln_hold1_sonicreset_30b_encoder_binding.json`;
- `logs/sonic_capability124_v1/ln_hold1_sonicreset_46b5_clean.json`;
- `logs/sonic_capability124_v1/ln_hold1_sonicreset_46b5_encoder_binding.json`;
- `logs/sonic_capability124_v1/sonic_v1_1_clean_r2_velacc.json` (repeat, adds
  velocity/acceleration distance).

Every result file carries the same ordered rank list, whose frozen JSON has
file SHA-256
`19b83597f0e7bf86fb462ae691b1dad455bb6b8cc130a9a4c702062aa75de147`. Each
binding audit confirms that all 18 encoder tensors embedded in the tracker are
identical to the selected encoder checkpoint.

## Evaluation protocol

Use the public `sonic_v1_1` checkpoint with SHA-256
`af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43`.
The clean protocol is:

- `Isaac-Imitation-G1-v2`;
- Newton/MJWarp;
- seed 0;
- deterministic `mode` actions;
- frame-0 starts;
- no startup, reset, or push randomization;
- SONIC thresholds;
- `foot_pos_xyz` and `base_too_low` disabled;
- enough steps to exceed the longest reference.

Score one of our own trackers on this subset with
`experiments/campaigns/2026-08-25-sonic-paper-proxy/score_arms_capability124.sh`,
which verifies the rank-list SHA-256, runs the encoder-binding audit, and
refuses to score an arm whose binding does not pass:

```bash
ARMS="ln_hold1_sonicreset_46b5" \
  ./experiments/campaigns/2026-08-25-sonic-paper-proxy/score_arms_capability124.sh
```

Run the public-checkpoint confirmation from the repository root:

```bash
mapfile -t RANKS < <(jq -r '.[]' \
  experiments/campaigns/2026-08-25-sonic-paper-proxy/sonic_capability124_v1.json)

env TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  pixi run -e isaaclab python -u \
  -m imitation_experiments.lowlevel.evaluate_sonic_release \
  --sonic_checkpoint /mnt/hsstorage/fwu91/sonic_v1_1/last.pt \
  --sonic_version v1_1 \
  --num_envs 124 --steps 1500 --seed 0 \
  --randomization none --reference_start_frame 0 \
  --reset_schedule sequential \
  --trajectory_ranks "${RANKS[@]}" \
  --termination_contract sonic \
  --proprioception_order gravity_last --history_order oldest_first \
  --output_json logs/sonic_capability124_v1/sonic_v1_1_clean.json \
  --label sonic_v1_1_sonic_capability124_v1_clean \
  --kit_args=--/app/extensions/fsWatcherEnabled=false \
  physics=newton_mjwarp \
  env.sim.physics.solver_cfg.njmax=320 \
  env.sim.physics.solver_cfg.nconmax=200 \
  env.events.push_robot=null env.data.manifest=null \
  env.data.reference_arrays_dir=/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1 \
  env.data.persist_id='bones_seed_sonic_full_129785@e714bbff' \
  env.data.reference_arrays_resident=false \
  env.data.reference_arrays_warm_workers=8 \
  env.data.runtime_cache_device=cuda:0 \
  env.data.reference_prefetch_mode=off \
  env.data.macro_cache_device=cuda:0 \
  env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]
```

## Manual review

All 124 motion names were inspected on 2026-08-25. Nine names suggested a
possible absent object or other actor: cutting, binoculars, fixing something,
smoking, lighting a cigarette, turning a large valve, kicking trash, making
fried eggs, and high-fiving a crowd. Each was rendered with the reference
robot beside our 30B tracker, with early terminations disabled and no common
video-length cap.

All nine videos reached the trajectory end, and the policy stayed upright for
the full horizon in every case. The motions are flat-ground, mime-like
whole-body trajectories; no object contact, crowd, or special scene geometry
is part of the executable reference. They remain in the new common eval
subset. Videos and diagnostic JSON files are under
`logs/videos/sonic_capability124_v1_questionable/`.

No rank changed during review. If later evidence shows that a clip is corrupt,
unsafe, or requires a missing scene element, keep `v1` unchanged and create a
`v2` candidate with a replacement from the same broad motion group. Recheck
the full row and publish the new rank-list hash.

The manual review must not remove a clip because it raises MPJPE-L. The
22-to-24 mm calibration is complete once the rank list is frozen.

## Paper use

**This is the calibration board, not the deciding one (2026-08-26).** The
headline tracker comparison against SONIC belongs on `bones_testbed4096_v1`;
its rows are Tables A and B in
`experiments/campaigns/2026-08-17-paper-metric-canon/README.md`. On that
deciding board public `sonic_v1_1` reads 0.9888 SR / 26.25 mm L / 177.41 mm G
and our `ln_hold1_sonicreset` @46.5B reads 0.9773 / 21.95 / 92.31, matched
over 3,932 clips — a different and less favorable picture than the 124-clip
rows above, which is expected because this subset was selected by reading
SONIC's own results.

### Naming and phrasing

Always call this population the **new common eval subset**. Do not call it held
out, unbiased, SONIC's validation split, or SONIC's unpublished evaluation
set. Keep `sonic_capability124_v1` as its machine-readable artifact ID.

Report SR, MPJPE-L, and MPJPE-G together for every locally evaluated tracker.
The local direct comparison is our tracker against public `sonic_v1_1` on
these identical 124 ranks.

BeyondMimic, GMT, and Any2Track rows remain externally reported results from
SONIC Figure 2. Keep them in a separate table block and state that we did not
train or execute those methods. SONIC supplies the common reference scale;
the borrowed rows are contextual comparisons, not native reproductions.
