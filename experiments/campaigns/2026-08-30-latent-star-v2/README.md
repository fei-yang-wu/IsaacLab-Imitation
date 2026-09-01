# 2026-08-30 — Latent-Learning Star v2

How should the command latent be learned? 41 arms at 5B frames, one field apart
from a shared hub, on the question the first command-interface star answered
against a hub that four later campaigns beat.

Design of record: [wiki/latent-learning-star-v2.md](../../../wiki/latent-learning-star-v2.md).
It carries the hub rationale, the census, the gates, and what this star cannot
answer. This directory is the executable form of that page.

W&B project `g1-bs-ablation`, group `latent-star-v2`, tags
`bones-seed,129785,latent-star-v2,<arm>,seed<N>,<stage>`. One seed per arm.

## The hub

One diffusion head denoising `s[t+10..t+20]` — the merged span, with the
endpoint term folded into it — over a continuous 64-D code with the phase
channel kept.

| field | value |
|---|---|
| objective | `jepa_ntp` + `sigreg_ebm`, `--jepa_ntp_head diff_chunk --jepa_ntp_chunk_span boundary_next --jepa_endpoint_coeff 0` |
| code | continuous 64-D, LayerNorm on, command 66 (z + 2-wide sin/cos phase) |
| window | 10 consecutive frames, stride 1, `root_qpos` 380 values, `robot_heading` anchor |
| hold | 1 |
| rewards | `motion_ee_pos` 1.0, `motion_global_anchor_pos_wide` 1.0, `action_rate_l2` -0.03 |
| `feet_acc` | -2.5e-6, the corrected recipe default — NOT overridden |
| resets | SONIC selection, failure share ramped 0.8 -> 0.2 over the first 1B frames |
| scale | 20,480 envs x 24 rollout steps, **5B frames**, 10,173 iterations |
| checkpoints | every 200M frames — 25 points over the 5B budget |

**The reset preset must be `sonic`, not `random80_adaptive20`.** Under the
latter the outer `random_trajectory_sampling_ratio=0.8` wrapper stays active
and `adaptive_uniform_ratio` only rescales the remaining 20% branch, so the
effective failure share moves 4% -> 16% instead of 80% -> 20% — the near-no-op
that job 5594817 shipped with and was cancelled for. The ramp keys off
`common_step_counter`, which restarts every segment, so segment 1 carries the ramp
and segment 2 pins the landed 0.2.

**Why the merged head rather than the two-head `diffntp_chunk`.** At 64-D and
hold 1 the two-head form has never trained: `leader64_h1_nophase` was cancelled
at 0.84B with episode length stuck at 50-62 and MPJPE-L flat near 51 mm, while
its 256-D control was at 166 / 42.6 mm by 0.17B, and its encoder pretrain was
provably healthy. The merged head at the same width and hold works twice over —
`diffntp_merged64` 0.9207 / 24.54 / 91.12 at 2B and `merged64_pen_ramp_5b`
0.9543 / 23.67 / 89.33 at 5B. That stalled arm also dropped the phase channel,
so width and phase are confounded in the one failure; the hub keeps phase and
`g5_phase_none_h10` separates them.

**Why 64-D.** It width-matches Group 3's discrete cells, so
continuous-versus-discrete is a clean contrast instead of one carrying a width
confound.

**Why 20,480 environments.** The measured-safe step up; 24,576 OOM'd the H200
at the Newton graph launch (job 5580202).

## Paper sections

The 41 arms answer three questions, and every arm carries `section`,
`section_label` and sometimes `section_group` in `campaign.yaml` so the report
and the config cannot drift apart:

1. **Factorization target** — what the encoder is trained to predict. The
   default is the next chunk `s[t+10..t+20]`; the alternatives are the
   endpoint, the next latent token, the endpoint delta, successor occupancy,
   the semi-Markov factorization, the split two-head form, and triplet context.
2. **Predictive architecture** — how that prediction is estimated. Generative
   DiffSR against a deterministic conditional mean (the JEPA-like cell),
   against reconstruction, against a posterior learned inside RL, plus the
   target-asymmetry family and the online encoder finetune.
3. **Remaining design choices** — the latent prior (continuous width, FSQ,
   VQ, categorical), the encoder input space (joint velocities, stride, full
   versus intermediate window, anchor frame, horizon), and the publication
   cadence (hold 5, hold 10, and hold 10 without the phase clock).

```bash
./sections.sh            # three tables, live status from ICE
LOCAL=1 ./sections.sh    # same, from a mirrored tree, no cluster access
```

The report fills success rate, MPJPE-L and MPJPE-G from scored rows as they
land; until then those columns read `-` rather than a placeholder number.

**One cell of section 1 is missing from this campaign.** The joint next
(state, token) target — round 4's `diffntp_pair`, `--jepa_ntp_head diff_pair`
— was never added, so the target axis has chunk, endpoint, token and delta but
no pair. Adding it is one arm.

## Groups

| group | question | arms |
|---|---|---:|
| hub | — | 1 |
| 1 | reconstruction and the posterior route against prediction | 7 |
| 2a | what the latent factorization predicts | 6 |
| 2b | the predictor form, control `g2_twohead` | 10 |
| 3 | continuous versus discrete | 6 |
| 4 | what the encoder reads, and how wide the window is | 7 |
| 5 | publication cadence | 3 |
| 6 | frozen versus finetuned encoder | 1 |

34 arms need `pretrain -> lowlevel1 -> lowlevel2`; 7 need only the two lowlevel
segments. 116 ICE segments of up to 15:59 on one H200.

**Nested controls.** Group 2b does NOT read against the hub. The hub's
`jepa_endpoint_coeff=0` is meaningful only for the merged span, where the
endpoint frame lives inside the target; with any other head that setting
removes the data grounding entirely. So every 2b arm restores the split heads
and reads against `g2_twohead`, which is itself a 2a row against the hub. The
chain is hub (merged) -> `g2_twohead` (split) -> the 2b head and asymmetry
variants.

**Encoder reuse.** Group 5 changes nothing about the encoder, so its three arms
bind the hub's `encoder/checkpoints/latest.pt` and carry no pretrain stage,
which also removes encoder-initialization variance from that comparison. They
cannot be planned until the hub's pretrain has completed and written that file.

**Two arms are not where the design wanted them, and the reason is RLOpt.**

- `g2_trip` uses the MLP head. RLOpt validates that the diffusion NTP heads
  support only the chunk pair (`jepa_context_chunks=0`), so triplet context is
  unavailable with any diffusion head. Its control is `g2_mlp`.
- `g6_dyn` sits on the conditional-mean cell, not the hub. The online finetune
  requires `jepa_ntp_head='mlp'` and refuses every diffusion head, so "dyn on
  the hub" cannot run without an RLOpt change. The arm answers whether the
  finetune helps the MLP predictor, against `g2_mlp`. **The frozen-versus-
  finetuned axis is therefore NOT measured at the hub.**

Both limits were found by the smoke, not by reading the code.

## Pipeline

```bash
./smoke.sh                      # local wiring qualification, every pretrained arm
./plan_all.sh                   # resolve all 41 plans offline
./submit.sh <arm> <seed>        # plan one arm; prints the submit --confirm line
./mirror.sh                     # pull checkpoints off ICE
./eval.sh                       # milestone + clean + robust rows
./eval.sh --report
```

`eval.sh` reads every per-arm field back out of `campaign.yaml` — width, hold,
phase, command mode, macro terms, stride, anchor, **horizon, and route** — so
evaluation cannot drift from training. The last two are this campaign's
additions: `horizon` moves with `g4_h5` / `g4_h20`, and `route` selects the
posterior entry point and command source for Group 1's four posterior arms,
which have no encoder file at all.

## Partition: use `coe-gpu`, not the profile default

Every stage pins `partition: coe-gpu` + `qos: coe-ice`, overriding the ICE
profile's `ice-gpu` / `coe-ice` pair. Set both together: a partition names the
QoS values it accepts, so a partition override carrying the wrong QoS is
rejected.

The reason is capacity, measured 2026-08-31 while 36 jobs sat on `Priority`:

| partition | H200 nodes | H200 GPUs | free at the time |
|---|---:|---:|---:|
| `ice-gpu` (profile default) | 6 | 48 | **7** |
| `coe-gpu` | 18 | 144 | **60** |

Our association holds `coc-ice`, `coe-grade`, `coe-ice` and `pace-grade`.
`coe-gpu` accepts `coe-ice` and `coe-grade`, `coc-gpu` accepts `coc-ice`, and
`pace-gpu` accepts `pace-grade` — but `coc-gpu` and `pace-gpu` carry no H200 at
all (L40S, A100, V100, A40, RTX 6000, MI210), so neither is a substitute for
this workload at 20,480 environments until someone measures whether it fits in
80 GB.

Moving the queue was one command per job and needed no resubmission:
`scontrol update JobId=<id> Partition=coe-gpu`. Running jobs went from 6 to 31
immediately.

## Submission

38 of the 41 arms were submitted together on 2026-08-31 by user instruction,
rather than gated behind the hub. Only Group 5 was held.

**Group 5 is held, and cannot be submitted yet.** `g5_hold5`, `g5_hold10` and
`g5_phase_none_h10` bind
`/data/latent_star_v2/hub_seed0/encoder/checkpoints/latest.pt`, which does not
exist until the hub's `pretrain` stage completes. That path is deliberately not
in `require_container_paths`, so a plan resolves and the JOB would fail. Submit
them once job 5600005 reports COMPLETED.

**Two risks were accepted rather than resolved, both recorded before launch.**

1. **The hub is not qualified.** Its exact combination — merged head, 64-D,
   hold 1, phase on, 20,480 envs, the SONIC 0.8 -> 0.2 reset ramp, penalty on,
   5B — has never been trained. `diffntp_merged64` is the nearest measured
   relative and differs in environment count, reset schedule, penalty, budget
   and `feet_acc`. The intended gate was to train the hub alone and compare its
   episode-length slope against `diffntp_merged64`'s milestone curve
   (`logs/report/milestone_curve.csv`: 0.8594 at 0.25B rising to 0.9258 at 2B)
   before committing the rest. Do that comparison as soon as the hub's first
   checkpoints land; the other 37 arms are already spending GPU against it.
2. **Group 2b went out before the rot6d convention was decided.** Eight of its
   ten arms (`g2_mlp`, `g2_nosig`, `g2_sg`, `g2_online`, the three
   `g2_lejepa_*`, `g2_trip`, `g2_token`) carry an EMA or stop-gradient token
   target that routes through `_reanchor_heading_frames`, where the data
   plane's interleaved 6-D rotation layout meets RLOpt's column-concatenated
   parse (found 2026-08-29, unfixed). Those eight are training against a
   deterministically distorted target, so the predictor-form comparison rests
   on it. The hub is NOT affected — the merged span uses the executed anchor at
   context 0 — and neither is any Group 1, 3, 4, 5 or 6 arm. Cancel and requeue
   those eight if the convention changes.

## The table is scored at 2B, not 5B

User decision 2026-08-31: score every arm at a matched **2B** checkpoint
(`model_step_2000486400`) rather than waiting for the 5B budget. Training is
NOT cancelled and continues to 5B; 2B is simply where the comparison is read.

Three reasons this is the right screen rather than a compromise. It is matched
— every arm is compared at the same frame count, which the frozen 5B finals
would also give but many hours later. It matches precedent: the v1 star was a
2B screen, so the two studies are read on the same budget. And it is available
now, because 33 of 41 arms already have that checkpoint.

The 5B rows remain worth having later for the arms that matter, but no ablation
conclusion should wait on them. State the budget in the same sentence as any
number from this table.

## Reading the results

Two artifacts from the same runs: the 5B table on `bones_testbed4096_v1` clean
(success rate, success-only micro MPJPE-L, MPJPE-G, plus the same-board
`sonic_v1_1` row per the standing directive), and per-metric convergence line
plots against environment frames with a point every 200M.

The 2026-08-22 pareto-stack rows are NOT rows of this table. They sit four
fields away — 256-D, 16,384 environments, no action penalty, `feet_acc`
-2.5e-7 — so they support no one-variable claim about this hub. Plot them as
their own family with the regime named, and note their curves are 8 points on a
250M grid that meets this campaign's 200M grid only at 1.0B and 2.0B.

MPJPE is success-only, so any table comparing L or G across rows must recompute
both on the rows' common success set and freeze that set by name; adding a row
changes every number. Noise band: 0.016 success rate, 1.3% MPJPE-L, 6.7%
MPJPE-G. One seed per arm, so every within-band ordering is unresolved.

## Where the checkpoints live

Scratch holds only the newest three checkpoints per arm. Everything older is
ARCHIVED, not deleted, to

    /storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_checkpoints/

under the same `<arm>_seed0/tracker/<run>/models/` layout, so `mirror.sh`
reaches it with `REMOTE_ROOT=<that path>`. The newest three stay behind because
the latest may be mid-write and a resume reads it.

The encoder `best.pt` files are archived beside it under
`latent_star_v2_encoder_best/`. This campaign binds `latest.pt`, but other
workflows in the repo do read a `best.pt`, which is why they were moved rather
than removed.

Re-run the archive whenever scratch climbs: it moved 406 files and 84 GB on
2026-08-31, taking scratch from 80.9% to 54.0%.

## Storage: this campaign does not fit the ICE quota

Measured 2026-08-31 at 20,480 envs and 64-D: a tracker checkpoint is **213 MB**
and a pretrained encoder is **2.67 GB** (`latest.pt` 1.34 + `best.pt` 1.34,
of which only `latest.pt` is ever read). At the 200M interval over 5B that is
25 checkpoints per arm:

| item | per arm | x41 arms |
|---|---:|---:|
| tracker checkpoints | 5.3 GB | 218 GB |
| encoder (34 pretrained arms) | 2.67 GB | 91 GB |
| **total** | | **~309 GB** |

ICE scratch is a 300 GB hard cap shared with every other campaign. Submitting
into it without freeing space killed 38 jobs on
`OSError: [Errno 122] Disk quota exceeded`, about five minutes into each — the
job reaches the GPU and builds its CUDA graph before the first save fails.

All 41 arms together burn about **8.7 GB per 200M-frame wave**, so scratch
refills roughly every 12 waves. Archive to
`/storage/ice-shared/vip-vwt/scratch-fwu91/` (2 TB, 24% used) during the run.

## Status

- 2026-08-31 **SUBMITTED, then repaired.** All 41 arms are in the ICE queue as
  112 jobs in W&B project `g1-bs-ablation`, group `latent-star-v2`. 30 running,
  zero failures since the repair.
- Regime, by user instruction: 20,480 environments, ee + wide rewards with
  `action_rate_l2` -0.03, the SONIC 0.8 -> 0.2 reset ramp over the first 1B
  frames, 5B budget, checkpoints every 200M.
- **What went wrong and how it was fixed.** The first submission put every
  stage on the profile's `ice-gpu`, which has 6 H200 nodes against `coe-gpu`'s
  18; 36 jobs sat on `Priority` behind 7 free GPUs while 60 sat idle next door.
  Moving them (`scontrol update JobId=<id> Partition=coe-gpu`, no resubmission
  needed) took running jobs from 6 to 31. Then 38 jobs died on the scratch
  quota. Repair order: hold the 67 pending jobs, free 187 GB, release the 7
  healthy arms, cancel the 60 queued jobs whose failed pretrain made their
  dependency unsatisfiable, resubmit those 34 arms.
- Wiring smoke before launch: 35 pass, 2 fail, 4 posterior arms skipped by
  design. It caught two configuration errors that reading the code had missed,
  both fixed and re-smoked: `g2_trip` used a diffusion head with triplet
  context, and `g6_dyn` asked for the online finetune on a diffusion head.
- **VQ is fine, and the smoke's VQ gate is a false positive.** `g1_recon_vq`
  completed its full 50,000-update pretrain at **code perplexity 275-279 of
  512** with `z_dim_std_mean` 5.4, and its code is informative (reconstruction
  loss 0.0298 against 0.148 zeroed and 0.176 shuffled). The 4-update smoke and
  a 400-update probe had both read perplexity 1.0 and `z_dim_std_mean` ~4e-9;
  an EMA codebook simply needs tens of thousands of updates to spread. Do not
  read the smoke's codebook check as a verdict on a VQ arm. This also matches
  v1, where VQ failed only as a frozen PRETRAINED codebook (`bn_vq_ema`
  0.0000) and worked in the posterior route (`post_recon_vq` 0.8945,
  `post_pgrecon_vq` 0.8867, `post_pg_vq` 0.8828).
- 2026-08-31, second check: 32 arms training, 7 pretraining, 2 pending;
  39 jobs running. Depth: 16 arms inside 0-1B, 15 inside 1-2B, and
  `g1_post_ae` at 4.60B. The hub is at 1.80B.
- **One non-finite abort, and the chain healed itself.**
  `g1_post_pgrecon_fsq` lowlevel1 hit
  `RuntimeError: Training went non-finite at 682721280 cumulative frames:
  ['train/step_reward_mean']` — `IPMD._abort_on_nonfinite` firing before any
  poisoned checkpoint could be written. Its `afterany` lowlevel2 resumed from
  the last good checkpoint and has since trained past the failure point to
  1.00B. Two caveats: at 0.68B this is far earlier than the roughly one event
  per 50B frames the hazard note records, so suspect the posterior-plus-FSQ
  cell rather than the generic drift; and a rewound chain logs nothing further
  to its W&B run because the step counter is monotonic, so that arm's live
  W&B curve has a gap. The convergence figures are unaffected — they are built
  from scored checkpoints, not W&B history.
- 2026-08-31, third check: **69B of the 205B total budget trained (34%)**.
  40 of 41 arms have checkpoints, median depth 1.80B, range 0.20B to 5.00B.
  `g1_post_ae` has FINISHED its 5B; the hub is at 3.00B. 40 jobs running.
  Nothing scored yet.
- **`g1_recon_vq` is a confirmed dead cell, not a slow one.** Its step reward
  diverges — -2.3, -4.5, -20.4, -36.6, -39.8 — against `g1_recon_fsq` holding
  a steady +0.23 on the same frames, and its episode length sits at 13.8
  against 182.4. The unbounded `action_rate_l2` penalty is what drives the
  reward that far negative once the policy thrashes against a command it
  cannot use. Read together with the healthy pretrain (perplexity 275-279 of
  512, informative code), this isolates the failure to the TRACKER's use of a
  frozen VQ codebook rather than to the codebook itself.
- The encoder `best.pt` files were ARCHIVED, not deleted, to
  `/storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_encoder_best/`
  — 34 files, 29 GB, taking scratch from 78.6% to 68.8%. They are unused by
  this campaign (which binds `latest.pt`) but other workflows do read a
  `best.pt`, so moving beats deleting.
- Four pretrains have COMPLETED: `hub` (44 min, `z_dim_std_mean` 1.01,
  effective rank 48.2/64), `g1_recon_ae`, `g1_recon_fsq`, `g1_recon_vq`.
