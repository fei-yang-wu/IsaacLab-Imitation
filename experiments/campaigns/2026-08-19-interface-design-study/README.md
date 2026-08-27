# 2026-08-19 — Interface design study

What makes a learned command interface good for whole-body tracking? This
campaign is the paper's low-level section, and **the ablation is the
contribution**, not support for the planner.

Status, last verified 2026-08-20: **COMPLETE.** All 29 arms trained to 2B and
scored — 29 clean rows, 29 robust rows, 232 milestone rows, zero evaluation
failures. Results and their reading are in `wiki/current-status.md`.

Headlines: the control configuration is the design (no arm beats it on success
rate by more than noise); discrete costs ~0.03 SR and ~19% MPJPE-L against
continuous at this budget; learned codebooks lose to the fixed FSQ lattice at
equal bits and `bn_vq_ema` collapses outright (0/4096); the `sin_cos` phase
channel is load-bearing at hold 10; and the 64-D hold-1 "dead zone" from the
100M grid is a budget artifact, not a dead cell.

**Nine arms had not converged at 2B**, mostly the quantized ones, so the
discrete-versus-continuous gap is an upper bound on the gap at convergence.
Every row is single-seed and directional.

The first submission (87 jobs, `5583651`-`5583738`) lost every low-level stage
to a SIGPIPE in the control plane's W&B run-id block; the encoders survived and
the low-level stages were re-submitted alone as `5583773`-`5583802`. See
`wiki/current-status.md` and `tests/test_cluster_run_id_block.py`.

Follow with
`pixi run python -m imitation_experiments.pipeline.cluster status --campaign interface-design-study`.

**29 arms are active** (17 tier 1, 11 tier 2, 1 probe), all 29 pass
`./smoke.sh`, and all 29 plans resolve offline with ICE preflight passing.
**One seed per arm.** **5 are deferred to
tier 4** by user decision on 2026-08-19 and are neither planned nor submitted:
the three co-trained arms (`use_cotrain_pg`, `use_cotrain_sonic`,
`use_cotrain_no_pg`), which wait on an encoder-from-checkpoint eval path, and
`use_phi` / `use_z_phi`, dropped for now. They stay defined so the design is
intact and re-enabling is a one-field change; `plan_all.sh` refuses tier 4
unless `FORCE_DEFERRED=1`.

## The design

A **star**, not a grid. `ctrl` is the hub; every other arm changes **exactly
one field** from it. Cost is linear in axes instead of combinatorial, and every
arm has a stated control.

A star measures **main effects only**. The one interaction known to be real —
code width against hold, where 64-D hold-1 is a dead zone while 64-D hold-10
and 256-D hold-1 both work — gets the explicit `ix_fsq64_hold1` probe that
completes its 2x2. No other interaction claim is supported by this campaign.

### The control `ctrl`

| field | value |
|---|---|
| `transition_objective` | `endpoint` |
| `latent_mode`, `z_dim` | `deterministic`, 256 |
| `encoder_layer_norm` | on |
| `expert_macro_state_terms` | `root_qpos` (38/frame, 380-wide input) |
| `expert_macro_frame_stride` | 1 |
| `expert_macro_anchor_mode` | `robot_heading` |
| `encoder_window_mode` | `intermediate` (endpoint hidden) |
| hold / `code_period` | 10 |
| `command_phase_mode` | `sin_cos` (command 258-D) |
| `hl_skill_command_mode` | `z` |
| encoder during RL | frozen |

Every hub value is already proven at 10B, so the hub itself is not at risk:
256-D at hold 10 is `jepa_sigreg_ebm_hold10_256d` (0.9282 SR / 22.26 mm), and
the LN continuous arm is the 10B leaderboard leader.

Frozen across every arm: BONES-SEED `bones_seed_sonic_full_129785@e714bbff`,
`Isaac-Imitation-G1-v2`, `rlopt_ipmd_tuned_cfg_entry_point`, the tuned reward
set, the termination set and its 5M-30M curriculum, `random80_adaptive20`
resets, 16,384 envs x 24, gamma 0.97, `reference_prefetch_mode=next`, the
`[2048,2048,1024,1024,512,512]` SiLU networks, Newton/MJWarp, and the 50,000
offline encoder updates at batch 8,192.

## Budget is an axis, not a ladder

**Every arm trains to the same 2B frames and no arm is promoted.** Promoting
the winners would select on the outcome and would leave the table with rows of
different budgets.

Budget-dependence is measured directly instead:

1. **Continuous curves** — return, episode length and
   `TrainHealth/mpjpe_{l,g}_mm_transition_ewma` against `env_frames`, straight
   from W&B. That channel already exists
   (`mdp/commands/reference.py:628-642` through `envs/rlopt.py:101-116`); no
   code was needed.
2. **Milestone curve** — `agent.save_interval=250000000` puts eight
   checkpoints across the run, scored on `paper_milestone_testbed256_v1`:
   256 clips drawn stride-16 from `TESTBED4096_RANKS`, the **same population**
   as the headline board and the same `sonic_sr_clean_v1` protocol. A ranking
   that only holds at one end of the curve is reported as budget-dependent.
3. **Row of record** — the 2B checkpoint on `paper_testbed4096_v1` (clean) and
   `paper_testbed4096_robust_v1` (`no_push`).

This axis is worth the trouble: the 2026-07-22 10M qualification ranked
`gumbel` first and `fsq` near-flat, and **neither ordering survived to 10B**.
A design study that reports one budget is reporting an artifact.

## Arms

34 arms, of which 29 are active: 17 tier 1, 11 tier 2, 1 interaction probe.
**Tier 1 runs seeds 0 and 1**, each with its own encoder pretrain, so encoder-initialization variance
sits inside the error bar. **Tier 2 is single-seed and every tier-2 row is
labelled preliminary.** Tier 2 is written to be droppable.

Bandwidth is a reported column, never a controlled variable. Hold 10 publishes
at 5 Hz, hold 1 at 50 Hz.

### Axis 1 — what trains the encoder

| arm | change from `ctrl` | tier |
|---|---|---|
| `obj_endpoint_delta` | `transition_objective=endpoint_delta` | 1 |
| `obj_state_occupancy` | `=state_occupancy` | 1 |
| `obj_semimarkov` | `=semimarkov_chain` | 1 |
| `obj_recon` | `=reconstruction`, MSE on the exact 380-value input window | 1 |
| `obj_jepa_ntp` | `=jepa_ntp`, `jepa_loss=sigreg` | 1 |
| `obj_jepa_sigreg_ebm` | `jepa_ntp` + `jepa_loss=sigreg_ebm` | 1 |
| `obj_jepa_infonce` | `jepa_loss=infonce` | 2 |
| `obj_phi_bilinear` | `diffsr_phi_parameterization=bilinear` | 2 |

`obj_jepa_ntp` passes `--jepa_loss sigreg` explicitly even though sigreg is
the parser default. An earlier draft carried a separate `obj_jepa_sigreg` arm
that omitted the flag; the wiring smoke measured the two at loss
0.26313257217407227 apiece, to every digit — they were **the same cell**. A
star with a duplicated cell spends a full training budget twice and reports it
as two independent measurements, so a contract test now resolves parser
defaults before comparing arms.

Reconstruction has **one** target — the exact encoder input window. Future-window,
keypoint, endpoint-only, trajectory-decode and inverse-dynamics decoders are
out of scope by decision, so this campaign needs no new decoder code.

### Axis 2 — how the code is constrained

| arm | change from `ctrl` | command | bits/command | tier |
|---|---|---|---:|---|
| `bn_gaussian` | `latent_mode=gaussian` + KL | 258 | continuous | 1 |
| `bn_sonic_fsq64` | `sonic_fsq`, 64 x 32 — SONIC's token space | 66 | 320 | 1 |
| `bn_vq_ema` | `vq`, K=512 | 258 | 9 | 1 |
| `bn_gumbel_multicat` | grouped Gumbel 64 x 32 | 66 | 320 | 1 |
| `bn_cont64` | `z_dim=64` | 66 | continuous | 1 |
| `bn_cont128` | `z_dim=128` | 130 | continuous | 1 |
| `bn_sonic_fsq32` | `sonic_fsq`, 32 x 32 | 34 | 160 | 2 |
| `bn_sonic_fsq16` | `sonic_fsq`, 16 x 32 | 18 | 80 | 2 |
| `bn_sonic_fsq64_l8` | `sonic_fsq`, 64 x 8 — levels at fixed width | 66 | 192 | 2 |
| `bn_gumbel` | single Gumbel codebook, K=512 | 258 | 9 | 2 |
| `bn_categorical` | hard categorical 64 x 32 | 66 | 320 | 2 |
| `bn_no_ln` | `encoder_layer_norm=false` | 258 | continuous | 2 |

The FSQ family moves **width** (64/32/16 at 32 levels) and **levels** (64 at
32/8) separately, so the bits-per-command figure does not confound the two.
`bn_gumbel_multicat` and `bn_categorical` sit at FSQ-64's exact bit budget with
learned levels instead of the fixed 1/16 lattice.

Only `sonic_fsq` is used for the discrete arms, because it is the mode whose
width contract is explicit: it publishes the quantizer output directly and
validates `z_dim == len(sonic_fsq_levels)`
(`RLOpt/rlopt/agent/hl_skill_diffsr.py:639-653`).

### Axis 3 — what the encoder reads

| arm | change from `ctrl` | tier |
|---|---|---|
| `in_fullbody670` | `full_body` macro terms: adds 29 reference joint velocities, 380 -> 670 | 1 |
| `in_stride5` | `expert_macro_frame_stride=5`, SONIC's 0.9 s window | 1 |
| `in_window_full` | `encoder_window_mode=full` — the endpoint is visible | 2 |
| `in_anchor_robot` | `expert_macro_anchor_mode=robot` | 2 |
| `in_anchor_expert_heading` | `=expert_heading` | 2 |

`in_fullbody670` is **not** the same arm as the in-flight local 10B run
`2026-08-18-qvel-fullbody-leader`: that one sits at hold 1, so it has a
different control. Read the two separately.

`in_stride5` is a known-risk arm. Stride 5 collapsed under every latent mode
tested on 2026-08-09 (`stride5_det64` 0.7063 SR, `stride5_fsq64` 0.6785,
`stride5_gumbel64` 0.5020, against ~0.90 for their stride-1 partners), which is
why the hub keeps stride 1. It is re-tested here because those runs predate the
current recipe. State the confound with any result: SONIC pairs its 0.9 s
window with a decoder that reads **ten past frames of proprioception**, and our
tracker reads one, so this arm reproduces SONIC's encoder cadence without
SONIC's decoder context. Proprioception history is a tracker-observation axis,
not an encoder axis, and is out of this campaign's scope.

### Axis 4 — how the code reaches the tracker

| arm | change from `ctrl` | tier |
|---|---|---|
| `use_hold1` | hold 1, `code_period=1` — 50 commands/s instead of 5 | 1 |
| `use_phase_none` | `command_phase_mode=none`, command 256-D | 1 |
| `use_cotrain_pg` | encoder trained online by policy gradient only | **4, deferred** |
| `use_cotrain_sonic` | PG + offline objective + anchor term, SONIC's recipe | **4, deferred** |
| `use_cotrain_no_pg` | offline + anchor, **no** PG — isolates the PG term | **4, deferred** |
| `use_phi` | publish `phi(s)` instead of `z` | **4, deferred** |
| `use_z_phi` | publish `z` and `phi` concatenated, 512 + 2 | **4, deferred** |

With those deferred, axis 4 currently carries `use_hold1` and
`use_phase_none` only. The co-training question — SONIC's recipe against a
frozen encoder — is the largest hole this leaves in the study, and it is a
scheduling decision, not a design one.

### Interaction probe

`ix_fsq64_hold1` — 64-D at hold 1, the known dead zone. It completes the width
x hold 2x2 whose other three cells are `ctrl` (256/h10), `use_hold1` (256/h1)
and `bn_sonic_fsq64` (64/h10). **A collapse here is the result**, not a failure.

## Deferred, and why (not blocking the active 30)

1. **The co-trained arms CAN be scored. The blocker was a false premise,
   measured and retired 2026-08-20.** An arm with
   `hl_skill_finetune_enabled=true` does keep a fine-tuned encoder inside its
   tracker checkpoint, but `IPMD.load_model` already restores it from
   `hl_skill_command_sampler_state_dict` (`ipmd.py:3036-3046`);
   `agent.ipmd.hl_skill_checkpoint_path` only seeds the sampler at
   construction and is then overwritten.

   Measured on `fsq64_hold10_dyn`: the live encoder is **0.0 max abs** from the
   checkpoint's embedded encoder in every configuration tried — including one
   that deliberately passed a DIFFERENT arm's pretrained file, and one that
   forced `--skill_encoder_source pretrained`. Success rate was 0.9023 in five
   of six runs regardless of either.

   So `fsq64_hold10_dyn` and `cont_det_hold1_resetramp_dyn` were excluded from
   every board on an assumption nobody measured, and both are scoreable today.

   What was added is **provenance, not a behaviour change**:
   `--skill_encoder_source {auto,checkpoint,pretrained}` records the resolved
   source and `live_vs_checkpoint_encoder_max_abs` into `summary.json`, so this
   question is settled by a logged number instead of an assumption. One early
   run scored 0.9062 and is still unexplained; five later runs including a
   four-run repeat all gave 0.9023 at an SR spread of 0.0000 (MPJPE-L spread
   0.129 mm). Treat that outlier as unresolved, not as evidence.
2. **`use_phi` and `use_z_phi` have never been trained**
   (`RLOpt/rlopt/agent/ipmd/ipmd.py:341, 663-673`). Their published command
   widths — 258 and 514 here — are inferred from `diffsr_feature_dim=256` and
   would need a 128-frame wiring smoke to confirm.

## Blocking work before submit

3. **Every arm runs its own wiring smoke**, `./smoke.sh`. **All 29 active arms
   passed on 2026-08-19.** Two stages: the
   encoder pretrain does four real Isaac/Newton offline updates and writes a
   loadable checkpoint, then the frozen encoder drives one 128-frame IPMD
   iteration at the arm's own command width. The verdict is
   `imitation_experiments.lowlevel.smoke_verdict` and it checks:

   - the checkpoint's recorded `transition_objective`, `latent_mode` and
     `z_dim` match what the campaign declares. This is the gate's main job — a
     checkpoint that trained a different design would silently turn one
     ablation cell into a copy of another;
   - the metric stream is non-empty and every number in it is finite;
   - the code is not collapsed: `train/z_effective_rank > 1` and
     `train/z_dim_std_min > 0`. That is the mode-agnostic form of "more than
     one code level in use", and it works for FSQ, VQ, Gumbel and categorical
     alike;
   - the low-level iteration exited cleanly.

   It deliberately does **not** judge learning. At four updates the code is
   expected to be worse than a zero code — the control arm measures
   `loss_real_z_eval` 39.25 against `loss_zero_z_eval` 38.17 — so gating on the
   learning signal would fail every arm.

   The smoke runs at the **production** batch of 8,192 on purpose. At batch 256
   a 512-entry VQ codebook cannot hit enough codes and reports a fully
   collapsed code; at 8,192 it never does, and a 400-update probe watched it
   recover from `z_dim_std_mean` 0.134 to 1.365 with the loss falling 39.2 to
   4.7. Batch must not be a confound between the smoke and the run it
   qualifies.
4. **Checkpoint discipline.** Eight tracker checkpoints per arm is about
   2.2 GB, and the encoder writes `latest.pt` plus `best.pt` at **416 MB each**,
   so a run costs ~3 GB. 29 arms at one seed is ~87 GB against ICE's 300 GB
   quota. Still mirror and thin to the five reported milestones as arms finish
   rather than at the end. Mirror and
   thin to the five reported milestones as arms finish. ICE scratch filling
   already forced an emergency thinning on 2026-07-26.
5. **W&B run ids** are pinned per `(arm, seed)` output tree and W&B refuses an
   id that was ever deleted with a job-killing 410. The ids here are unique by
   construction (the generator asserts it) and all fit the 31-character cap.
   The group `interface-design-study` is **confirmed** (2026-08-19).

Out of scope and **not** an arm: a keypoint encoder input. `env.data.macro_cache_device`
serves only the `root_qpos` and `full_body` macro term sets, so a
`expert_keypoint_pos_b` arm would lose the fast macro path or fail outright.
It needs env work first.

## Paper framing (decided 2026-08-20)

**Main story: the 26 hold-10 arms, narrated by success-only MPJPE-L**, with
success rate alongside. Those two agree on this set at Spearman **+0.957**, so
the L-led narrative and the success-rate narrative are the same story — which
is what makes leading on L safe here.

The main figure is the hold-10 set on purpose. 27 of the 29 active arms are
hold 10 against a hold-10 control, so that set is the internally consistent
comparison: one cadence, one hub, every arm exactly one field from it. Putting
cadence in the main figure would mean two controls and a confounded axis.

**Cadence is an ablation**, via the width x hold 2x2 (`ctrl`, `use_hold1`,
`bn_sonic_fsq64`, `ix_fsq64_hold1`). Report it honestly: hold 10 and hold 1 tie
on success and local error — every gap there is inside the ~15% band — and
hold 1 is decisively better on global error at both widths (-29% at 256-D,
-21% at 64-D). "Hold 10 is better" is not a claim this study supports.

**MPJPE-G stays a column in every table.** `canonical-paper-metrics.md` is
frozen on "none of the three may be published alone", and this campaign is the
strongest evidence for that rule: on the 10B board L and G rank arms at
Spearman -0.071, and here `obj_recon` is +12.3% on L (inside the band) while
being +110% on G. Leading on L is a narrative choice; dropping G would hide the
drift failure mode. Within this main set L and G still disagree (+0.430), and
that disagreement belongs in the ablation and the discussion.

**One consistency trap to handle in the text.** The planner section's best row
(38.41 mm) runs on a HOLD-1 tracker. If Section 1 leads on hold 10 while
Section 2's headline pair is hold 1, say so explicitly — the cadence axis is
reported separately and the deployed pair takes hold 1 for its global-drift
advantage. Stated, that is a finding; unstated, it is an inconsistency a
reviewer will find.

### What the L-led main story says

Ranked by MPJPE-L, four arms sit nominally ahead of the control — `bn_no_ln`
(-5.4%), `in_window_full` (-3.4%), `bn_cont64` (-1.4%), `bn_cont128` (-0.9%) —
and **all four are inside the ~15% unresolved band at one seed**. So the claim
is "the control sits at the top of a five-way tie", not "nothing beats the
control". Everything below that tie separates cleanly and monotonically, from
`in_anchor_robot` (+2.7%) down to `bn_gumbel` (+182%).

## Evaluation

| purpose | profile |
|---|---|
| budget curve | `paper_milestone_testbed256_v1` |
| row of record | `paper_testbed4096_v1` |
| robustness | `paper_testbed4096_robust_v1` |
| second population, tier 1 | `bones_heldout4096_v1` |
| cross-backend, tier 1 | `sidecar_ec_strat64_v1` |

Every row prints success rate, success-only micro MPJPE-L **and** MPJPE-G
together — the rule in `wiki/canonical-paper-metrics.md` — plus velocity and
acceleration distance (the other two metrics SONIC publishes), the
per-termination counts, the true env frames, and bandwidth. Reduce with
`python -m imitation_experiments.evaluation.summarize_paper_boards`.

The eight existing 10B arms and the released SONIC checkpoint are **reference
end-points on the budget figures, not comparison rows**. No sentence compares a
2B arm to them as a win. Their canonical-testbed rows were produced on
2026-08-19 and are in `logs/testbed4096/`; the table is in
`wiki/current-status.md`.

Those rows carry a warning for this study's reporting: over the eight arms,
**MPJPE-L and MPJPE-G rank them at Spearman -0.071** — the worst arm on local
error is the best on global error by 20 mm. Rank this study's arms on both, and
never on the local metric alone.

## Commands

Nothing below submits.

```bash
# wiring gate: every active arm, two real stages each (~2-3 min per arm)
./experiments/campaigns/2026-08-19-interface-design-study/smoke.sh
ARMS="bn_vq_ema" ./experiments/campaigns/2026-08-19-interface-design-study/smoke.sh
./experiments/campaigns/2026-08-19-interface-design-study/smoke.sh --report

# resolve one arm and print its frozen command + PLAN_SHA
./experiments/campaigns/2026-08-19-interface-design-study/submit.sh ctrl 0

# resolve a whole tier offline, to catch a typo before any GPU is asked for
TIERS="1 2 3" ./experiments/campaigns/2026-08-19-interface-design-study/plan_all.sh
```

The control plane prints a separate `submit --plan ... --confirm <PLAN_SHA>`
line per plan. Do not run it until the smokes above pass. The W&B group is
already confirmed.

Scoring, once checkpoints are mirrored to
`logs/interface_design_study_mirror/<arm>_seed<seed>/{encoder,tracker}`:

```bash
# every mirrored arm: milestone curve + row of record + robustness partner
./experiments/campaigns/2026-08-19-interface-design-study/eval.sh
ARMS="ctrl obj_recon" ROWS=clean ./experiments/campaigns/2026-08-19-interface-design-study/eval.sh
./experiments/campaigns/2026-08-19-interface-design-study/eval.sh --report
```

`eval.sh` reads every per-arm interface field -- command width, hold, code
width, phase, command mode, macro terms, stride, anchor -- back out of
`campaign.yaml`, so evaluation cannot drift from training. Do not hardcode any
of them: the 2026-08-15 runner did, and needed a warning comment about the
three settings that silently differed per arm.

## Reading rules

- Relative differences under about 15% in the high-error regime are
  unresolved. **No arm carries a repeat seed**, so this applies to every row.
  Prefer a difference that holds across the milestone curve over one measured
  at the endpoint alone, and say "directional" wherever a single-seed gap is
  inside the band.
- Report success rate and success-only MPJPE together and inspect the
  per-termination counts when they disagree. Success-only MPJPE is not
  comparable across different success rates.
- Every board here is **in-distribution**: trackers train on the full
  129,785-clip tree with no rank filter. `bones_heldout4096_v1` is a different
  population, not a held-out set.
- Do not select a metric after the result is known.
