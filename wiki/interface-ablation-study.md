# Interface Ablation Study

What the 72-arm command-interface study is, how every arm was trained and
scored, and what each arm changes. This is the setup and provenance page; the
findings are in
[Results: interface ablations](results-interface-ablations.md), and the
interactive curve board is published as an artifact (link in
[current-status.md](current-status.md)).

**Evidence page.** It records how a set of numbers was produced. Read it to
learn what an arm is, not as a standing instruction.

---

## 1. The question

A learned command interface sits between a high-level module and the low-level
tracking controller. Four things about it are free, and the study measures each
against a stated control:

1. **What the encoder is trained to predict** — the offline objective that
   shapes the code.
2. **The shape of the code** — width, quantizer, normalization.
3. **What the encoder reads** — which state terms, how far apart the frames
   are, which frame the window is anchored in.
4. **How the code is used** — how many control steps one code is held, and
   whether a phase clock rides alongside it.

Three later campaigns extend that: whether findings compose, whether the code
can be learned during RL instead of pretrained, and which part of the winning
loss is load-bearing.

## 2. Design: a star, not a grid

`ctrl` is the hub. Every other arm of the design study changes **exactly one
field** from it, so each arm has a stated control and the cost is linear in axes
rather than combinatorial. A star measures main effects only; the interactions
known to be real get explicit probes (`ix_fsq64_hold5`, `ix_fsq64_hold1` for
code width × hold), and the `interface-combos` campaign tests whether the
strongest single-field findings compose.

The hub configuration:

| field | value |
|---|---|
| objective | endpoint DiffSR |
| latent mode | deterministic, continuous |
| code width | 256 (command 258 = z + 2-wide sin/cos phase) |
| encoder LayerNorm | on |
| macro state | `root_qpos` frame — 380-value window |
| frame stride | 1 (ten consecutive frames) |
| anchor mode | `robot_heading` |
| hold | 10 control steps |
| phase | `sin_cos`, sourced from the hold clock |
| command mode | publish `z` |
| encoder during RL | frozen |

A contract test holds the star property: `test_interface_design_study_campaign.py`
asserts that each arm's declared vars differ from the hub only in the fields its
own row names, and that the tier census matches. An accidental second difference
fails the test rather than producing an uninterpretable arm.

## 3. Shared training protocol

Frozen across every arm of every campaign below.

**Environment.** `Isaac-Imitation-G1-v2`, Newton MJWarp physics
(`env.sim.physics.solver_cfg.njmax=320`, `nconmax=200`), the BONES-SEED
129,785-clip reference set (`persist_id bones_seed_sonic_full_129785@e714bbff`)
held resident in RAM, `env.data.reference_prefetch_mode=next`.

**Collection and optimization.** IPMD via
`rlopt_ipmd_tuned_cfg_entry_point`; 16,384 environments × 24 rollout steps =
393,216 frames per batch; minibatch 3/4 of a batch; `agent.loss.gamma=0.97`;
expert batch 24,576; policy and value networks `[2048, 2048, 1024, 1024, 512,
512]` with SiLU.

**Rewards and terminations.** `env.rewards.action_rate_l2.weight=0.0`,
`env.rewards.tracking_reward_points.weight=4.0`, termination curriculum on and
ramped between 5M and 30M frames, reset selection
`random80_adaptive20`. Pareto-stack arms enable two otherwise-inert reward terms;
those arms name them in their row.

**Budget.** Every arm targets 2,000,000,000 environment frames — `max_iterations`
5,087 — and writes a checkpoint every 250M frames
(`agent.save_interval=250000000`), which is where the eight scored milestones
come from. No arm is promoted to a longer budget: promoting winners would select
on the outcome and leave the table with rows of different budgets.

**Encoder pretrain.** `scripts/rlopt/train_hl_skill_diffsr.py`, 50,000 updates at
batch 8,192, `--horizon_steps 10`, encoder hidden dims `[2048, 1024, 512, 512]`
with SiLU, DiffSR feature dim 256 and embed dim 1024. The tracker then binds the
frozen `encoder/checkpoints/latest.pt` through
`agent.ipmd.hl_skill_checkpoint_path`.

**Cluster.** Georgia Tech ICE, one H200 per stage, 16 CPUs, 160 GB, 15:59:00 per
segment through the control plane
(`python -m imitation_experiments.pipeline.cluster`). A run is a chain:
pretrain → lowlevel1 → lowlevel2, the second segment resuming on
`cumulative_env_frames` and exiting immediately if the budget is already met.

## 4. Shared evaluation protocol

**Board.** `bones_milestone_testbed256_v1` — `TESTBED4096_RANKS[::16]`, 256
clips drawn from the same population as the 4096-clip paper board. Every
checkpoint of every arm is scored on exactly these clips, with env *i* pinned to
rank `rank_table[i]`, a pure function of env id, so all 576 cells face the same
population.

**Rollout.** `--steps 10000`, `--num_envs 256`, `--seed 0`,
`--reference_start_frame 0`, `--action_sampling mode`, `--randomization none`,
push disabled, episode length extended to 200.04 s so a clip runs to its own
end. Terminations: `anchor_pos` 0.25 m (and 0.25 m down), `anchor_ori` 1.0 rad,
`ee_body_pos` 0.25 m (and 0.25 m down); `foot_pos_xyz` and `base_too_low`
disabled.

**Metrics.** The canonical row is three numbers that always travel together:
success rate under SONIC's termination definition, success-only frame-weighted
(micro) MPJPE-L in millimetres, and success-only MPJPE-G. An arm with no
successful episode has no MPJPE at all; it is reported at success rate 0.0000
with empty error columns, never dropped.

**Two ways a tree is scored.** `imitation_experiments.lowlevel.evaluate_checkpoint`
scores one checkpoint per process. `scripts/rlopt/eval_checkpoint_tree.py` keeps
one Isaac Sim start and swaps the policy weights across a tree's milestones —
30.9 s per cell against 47.4 s, and one container start instead of eight on the
cluster. The two paths agree within the evaluator's own nondeterminism: over
eight paired cells, mean success-rate difference +0.0010 (largest 0.0156),
MPJPE-L +0.06% mean, MPJPE-G −1.30% mean, scattered in sign.

**Reduction.** `imitation_experiments.reporting.curve_table` turns the scored
JSON into `logs/report/milestone_curve.csv`, one row per (arm, seed, budget).

## 5. The arms

Result columns are the 2B checkpoint: success rate, MPJPE-L (mm), MPJPE-G (mm).

### 5.1 Interface design study — `2026-08-19-interface-design-study`

29 arms at seed 0, plus the hold-5 pair added 2026-08-26. Five tier-4 arms are
declared but deliberately never submitted: `use_cotrain_pg`, `use_cotrain_sonic`,
`use_cotrain_no_pg` (they need an encoder-from-checkpoint evaluation path) and
`use_phi`, `use_z_phi` (dropped by user decision). They stay in the config so the
design is intact and re-enabling is a one-field change.

**Objective — what the encoder is trained to predict.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `ctrl` | nothing — the hub | `--transition_objective endpoint` | 0.9102 | 23.44 | 199.87 |
| `obj_endpoint_delta` | predict the endpoint DELTA, not the endpoint | `--transition_objective endpoint_delta` | 0.8438 | 40.75 | 249.34 |
| `obj_state_occupancy` | successor-representation occupancy instead of a single endpoint | `--transition_objective state_occupancy` | 0.8984 | 24.56 | 169.68 |
| `obj_semimarkov` | semi-Markov factorization of the skill transition | `--transition_objective semimarkov_chain` | 0.8789 | 27.32 | 181.88 |
| `obj_recon` | autoencode the exact 380-value input window; decoder offline-only | `--transition_objective reconstruction` | 0.9023 | 26.22 | 401.18 |
| `obj_jepa_ntp` | latent next-token prediction with the SIGReg energy | `--transition_objective jepa_ntp --jepa_loss sigreg` | 0.8438 | 34.68 | 167.31 |
| `obj_jepa_sigreg_ebm` | the same with the energy-based SIGReg variant | `--transition_objective jepa_ntp --jepa_loss sigreg_ebm` | 0.9023 | 27.34 | 206.70 |
| `obj_jepa_infonce` | contrastive InfoNCE energy | `--transition_objective jepa_ntp --jepa_loss infonce` | 0.8906 | 29.87 | 201.04 |
| `obj_phi_bilinear` | legacy bilinear successor head instead of the concat head | `--diffsr_phi_parameterization bilinear` | 0.8945 | 25.22 | 180.40 |

**Bottleneck — the shape of the code.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `bn_cont64` | code width 64 | `--z_dim 64`, command 66 | 0.9180 | 22.80 | 204.74 |
| `bn_cont128` | code width 128 | `--z_dim 128`, command 130 | 0.9102 | 23.39 | 205.03 |
| `bn_gaussian` | Gaussian posterior with a KL penalty | `--latent_mode gaussian` | 0.9023 | 27.68 | 212.18 |
| `bn_sonic_fsq64` | SONIC's released token space, FSQ 64 coords x 32 levels | `--latent_mode sonic_fsq --sonic_fsq_levels 32 x64` | 0.9023 | 28.86 | 177.70 |
| `bn_sonic_fsq32` | FSQ 32 x 32 — halves the width at fixed levels | `--sonic_fsq_levels 32 x32`, command 34 | 0.8945 | 31.99 | 206.43 |
| `bn_sonic_fsq16` | FSQ 16 x 32 — the narrow end of the discrete sweep | `--sonic_fsq_levels 32 x16`, command 18 | 0.6836 | 44.33 | 242.30 |
| `bn_sonic_fsq64_l8` | FSQ 64 x 8 — moves LEVELS at fixed width | `--sonic_fsq_levels 8 x64` | 0.8672 | 33.82 | 193.73 |
| `bn_gumbel_multicat` | grouped Gumbel-softmax, 64 groups x 32 categories | `--latent_mode gumbel_multicat --categorical_groups 64 --categorical_categories 32` | 0.7031 | 46.45 | 340.05 |
| `bn_categorical` | hard straight-through categorical at the same 64 x 32 | `--latent_mode categorical --categorical_groups 64 --categorical_categories 32` | 0.5703 | 54.77 | 689.24 |
| `bn_gumbel` | single Gumbel-softmax codebook, K=512 | `--latent_mode gumbel --gumbel_codebook_size 512` | 0.3750 | 68.45 | 1364.72 |
| `bn_vq_ema` | single EMA VQ codebook, K=512 | `--latent_mode vq --vq_codebook_size 512` | 0.0000 | — | — |
| `bn_no_ln` | encoder LayerNorm off | `--no_encoder_layer_norm` | 0.8945 | 22.48 | 222.26 |

**Input — what the encoder reads.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `in_fullbody670` | adds the 29 reference joint velocities: 380 -> 670 input | `env.expert_macro_state_terms=[expert_motion,expert_anchor_pos_b,expert_anchor_ori_b]` | 0.8984 | 26.92 | 225.79 |
| `in_stride5` | 10 frames spaced 5 apart — SONIC's 0.9 s window | `env.expert_macro_frame_stride=5` | 0.6992 | 45.62 | 446.67 |
| `in_window_full` | window includes the endpoint instead of hiding it | `--encoder_window_mode full` | 0.9102 | 23.20 | 224.25 |
| `in_anchor_robot` | macro anchor in the live robot frame | `env.expert_macro_anchor_mode=robot` | 0.9062 | 24.64 | 253.86 |
| `in_anchor_expert_heading` | macro anchor in the expert's heading frame | `env.expert_macro_anchor_mode=expert_heading` | 0.8945 | 30.44 | 350.25 |

**Usage — how the command is published.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `use_hold5` | publish every 5 control steps | `agent.ipmd.latent_steps_min/max=5`, `latent_learning.code_period=5` | 0.9102 | 24.79 | 146.92 |
| `use_hold1` | publish and re-encode every control step (50 commands/s) | `agent.ipmd.latent_steps_min/max=1`, `code_period=1` | 0.9180 | 25.76 | 140.94 |
| `use_phase_none` | drop the 2-wide sin/cos slot clock; command is z alone | `latent_learning.command_phase_mode=none`, command 256 | 0.4141 | 64.82 | 1344.94 |
| `ix_fsq64_hold5` | 64-D FSQ at hold 5 | FSQ 64 x 32 + `code_period=5` | 0.9023 | 28.52 | 141.36 |
| `ix_fsq64_hold1` | 64-D FSQ at hold 1 — the suspected dead zone | FSQ 64 x 32 + `code_period=1` | 0.8789 | 30.65 | 134.10 |

### 5.2 Interface combinations — `2026-08-21-interface-combos`

Do the strongest single-field findings compose, and do the two reconstruction
follow-ups move its global error?

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `jepa_ebm_hold1_256d` | best objective x best interface | `jepa_ntp/sigreg_ebm` + hold 1 | 0.9219 | 26.95 | 119.06 |
| `jepa_ebm_hold1_fsq64` | the same pair in SONIC's token space | `jepa_ntp/sigreg_ebm` + hold 1 + FSQ 64 x 32 | 0.8867 | 30.90 | 123.79 |
| `recon_endpoint` | reconstruction with the ENDPOINT decode target | `--reconstruction_target endpoint` | 0.9062 | 22.87 | 364.18 |
| `recon_full_window` | reconstruction decoding all ten future slots incl. the hidden endpoint | `--reconstruction_target full_window` | 0.9180 | 24.78 | 416.80 |
| `hold1_live_phase` | hold 1 with a LIVE clock (env steps since reset mod 10) | `command_phase_source=episode`, `command_phase_period=10` | 0.9102 | 25.52 | 143.53 |

### 5.3 Posterior route — `2026-08-20-posterior-interface`

A 3 × 3 grid: quantizer (continuous / FSQ / EMA VQ) × training signal
(reconstruction only / policy gradient only / both). These arms have **no
pretrained encoder file** — the code is learned during RL, so the encoder lives
inside the tracker checkpoint and evaluation restores it with
`--skill_encoder_source checkpoint`.

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `post_recon_ae` | reconstruction only; continuous 256-D | `recon_coeff=1.0`, `train_posterior_through_policy=false`, `quantizer=identity` | 0.8984 | 30.93 | 457.89 |
| `post_recon_fsq` | reconstruction only; FSQ | `quantizer=fsq`, `code_latent_dim=64`, `fsq_levels=[32]x8` | 0.8633 | 35.49 | 416.93 |
| `post_recon_vq` | reconstruction only; EMA VQ K=512 | `quantizer=vq_ema`, `codebook_size=512` | 0.8945 | 31.28 | 475.37 |
| `post_pg_ae` | policy gradient only; continuous 256-D | `recon_coeff=0.0`, `train_posterior_through_policy=true` | 0.8906 | 30.55 | 442.47 |
| `post_pg_fsq` | policy gradient only; FSQ | as above + `quantizer=fsq` | 0.8789 | 32.74 | 436.66 |
| `post_pg_vq` | policy gradient only; EMA VQ | as above + `quantizer=vq_ema` | 0.8828 | 32.28 | 505.50 |
| `post_pgrecon_ae` | both signals; continuous 256-D | `recon_coeff=1.0` + `through_policy=true` | 0.8789 | 32.02 | 422.85 |
| `post_pgrecon_fsq` | both signals; FSQ | as above + `quantizer=fsq` | 0.8477 | 36.05 | 498.59 |
| `post_pgrecon_vq` | both signals; EMA VQ | as above + `quantizer=vq_ema` | 0.8867 | 29.77 | 427.25 |

### 5.4 Pareto stack — `2026-08-22-pareto-stack`

27 arms in four questions: which unscreened reward terms help, whether an online
dynamics finetune stacks, how the winning interface responds to reward, and
which part of the winning loss composition is load-bearing. Every arm of the
last two groups is fixed at hold 1 with both extra reward terms enabled, so the
loss composition is the only axis.

Arms whose encoder is unchanged from a parent reuse the parent's encoder file
verbatim (`ctrl_encoder`, `hold1_encoder`, `jepa_h1_encoder`), which removes
encoder-initialization variance from those comparisons.

**Q1 — unscreened reward terms and the asymmetric critic, on the hub.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `ctrl_ee1` | wrist position reward enabled (std 0.1, pelvis-anchored) | `env.rewards.motion_ee_pos.weight=1.0` | 0.9102 | 23.49 | 192.44 |
| `ctrl_wide1` | wide global anchor position reward (std 0.5) | `env.rewards.motion_global_anchor_pos_wide.weight=1.0` | 0.9141 | 24.31 | 137.40 |
| `ctrl_ee_wide` | both reward terms together | both weights 1.0 | 0.9141 | 23.22 | 150.95 |
| `ctrl_bodyglobal` | the literal MPJPE-G integrand as a reward (std 0.1) | `env.rewards.motion_body_pos_global.weight=0.5` | 0.9141 | 23.54 | 165.53 |
| `ctrl_asymcritic` | critic reads the full explicit reference set; actor unchanged | `command_interface.reference.critic_components=[...]` | 0.9141 | 23.73 | 195.64 |

**Q2 — does the online dynamics finetune stack?**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `hold1` | the hold-1 hub re-run at SEED 1 | seed 1 | 0.8984 | 26.37 | 150.36 |
| `hold1_dyn` | online dynamics finetune on hold 1, seed 0 | `hl_skill_finetune_enabled=true`, `achieved_coeff=1.0`, `pg_coeff=0` | 0.9141 | 26.25 | 129.00 |
| `hold1_dyn_s1` | the same at seed 1, paired with `hold1` | as above, seed 1 | 0.9023 | 27.26 | 174.51 |
| `jepa_h1_dyn` | the same finetune on the best-G interface | as above on the JEPA hold-1 encoder | 0.8984 | 28.19 | 104.53 |

**Q3 — reward response of the winning interface.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `jepa_h1_ee1` | wrist reward only | `motion_ee_pos.weight=1.0` | 0.9062 | 24.25 | 114.40 |
| `jepa_h1_wide1` | wide anchor reward only | `motion_global_anchor_pos_wide.weight=1.0` | 0.9180 | 25.96 | 92.89 |
| `jepa_h1_ee_wide` | both — the mechanism square's parent cell | both weights 1.0 | 0.9141 | 25.34 | 79.69 |

**Q4 — the loss-composition square.**

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `endpoint_h1_ee_wide` | plain endpoint DiffSR — the interface-matched objective control | `--transition_objective endpoint` | 0.9023 | 26.57 | 104.71 |
| `dsrsig_h1_ee_wide` | DiffSR + SIGReg, predictor dropped from the loss | `--jepa_ntp_coeff 0` | 0.9023 | 27.13 | 113.11 |
| `jepa_nosig_h1_ee_wide` | DiffSR + NTP + EMA, SIGReg off | `--jepa_sigreg_coeff 0` | 0.9180 | 26.52 | 93.41 |
| `jepa_ol_h1_ee_wide` | the full composite with ONLINE targets instead of EMA | `--jepa_target_encoder_mode online` | 0.8984 | 28.60 | 123.70 |
| `sg_h1_ee_wide` | the full composite with a stop-grad target | `--jepa_target_encoder_mode stopgrad` | 0.9102 | 26.21 | 97.94 |
| `lejepa_h1_ee_wide` | NTP + SIGReg only, one online encoder both sides | `--jepa_loss sigreg --jepa_target_encoder_mode online` | 0.7383 | 47.58 | 691.91 |
| `lejepa_sg_h1_ee_wide` | NTP + SIGReg only, stop-grad target (SimSiam cell) | `--jepa_loss sigreg --jepa_target_encoder_mode stopgrad` | 0.8750 | 34.05 | 94.81 |
| `lejepa_ema_h1_ee_wide` | NTP + SIGReg only, EMA target | `--jepa_loss sigreg --jepa_target_encoder_mode ema` | 0.8828 | 31.14 | 96.83 |
| `trip_h1_ee_wide` | triplet NTP: predictor reads cat(z[t-H], z[t]) | `--jepa_context_chunks 1` | 0.9141 | 26.14 | 110.92 |
| `qvel_h1_ee_wide` | pair NTP on full-body frames (670-value window) | `macro_terms=[expert_motion,...]` | 0.9141 | 24.77 | 106.27 |
| `trip_qvel_h1_ee_wide` | triplet NTP AND full-body frames | both of the two above | 0.8164 | 32.65 | 112.51 |

**Q4b — generative next-chunk heads (2026-08-26).** The composite's prediction
term is a conditional mean; chunk futures are multimodal, so these arms swap the
estimator for a second diffusion head at the same slot, conditioned on (s_t, z_t).

| arm | what moves | override | SR | MPJPE-L | MPJPE-G |
|---|---|---|---|---|---|
| `diffntp_token_h1_ee_wide` | generative next-TOKEN: `p(z_next \| s_t, z_t)`, EMA target | `--jepa_ntp_head diff_token` | 0.9258 | 24.11 | 84.20 |
| `diffntp_chunk_h1_ee_wide` | generative next-CHUNK of data in the executed heading frame; no self-target | `--jepa_ntp_head diff_chunk` | 0.9297 | 23.45 | 74.85 |
| `diffntp_chunkra_h1_ee_wide` | the same with the target re-anchored onto the next frame | `--jepa_ntp_head diff_chunk --jepa_ntp_chunk_anchor next` | 0.9219 | 23.28 | 84.80 |
| `diffntp_pair_h1_ee_wide` | joint next (state, token) over the concatenated 38+256 target | `--jepa_ntp_head diff_pair` | 0.9180 | 23.29 | 83.27 |

## 6. Reproducing a row

```bash
# score one campaign's whole budget axis locally, from a mirrored tree
ROWS=milestone ./experiments/campaigns/2026-08-22-pareto-stack/eval.sh
```

```bash
# the same, one Isaac Sim start per arm instead of one per checkpoint
TREE_SCORER=1 ROWS=milestone ./experiments/campaigns/2026-08-22-pareto-stack/eval.sh
```

```bash
# rebuild the curve table from every scored directory
pixi run python -m imitation_experiments.reporting.curve_table logs/interface_design_study_eval logs/interface_combos_eval logs/pareto_stack_eval logs/posterior_interface_eval logs/hold5_curve_eval --row milestone --csv logs/report/milestone_curve.csv
```

On the cluster, `2026-08-27-hold5-curve-eval` is the worked example of scoring a
tree in place: one Slurm job per arm, about 16 minutes for eight cells, no
checkpoint transfer at all. Serialize those jobs — two overlapping eval jobs
sharing `CLUSTER_ISAAC_SIM_CACHE_DIR` crashed the second one inside Kit startup
on 2026-08-27.

## 7. Provenance and caveats

- **One seed.** Only `hold1` / `hold1_dyn_s1` carry seed 1, and they are a pair
  for the dynamics-finetune comparison, not a repeat of the star. Every ordering
  inside the noise band is unresolved until seeds are repeated.
- **The curve board is not the paper board.** 256 clips, not 4096. Numbers on
  this page are comparable to each other and are not paper rows.
- **Isaac evaluation is not deterministic**; the measured band is above.
- **No robustness claim.** Domain randomization off, no push.
- **`bn_vq_ema` never completes an episode** at any budget, so it has a success
  rate and no error metrics.
- The design study's ICE checkpoint trees were cleaned under the 300 GB quota;
  those 29 arms are scored from the local mirror
  `logs/interface_design_study_mirror`. The pareto, combos and posterior trees
  are still on ICE.
