# 2026-08-02 — RLOpt hyperparameter screen (G1 v2 low-level)

A screen of the RLOpt optimizer settings for the v2 low-level tracker. The
recipe, task, encoder and data are unchanged from the
`2026-08-02-v2-command-interface` campaign; what varies is the PPO/IPMD
optimizer configuration.

Two launchers, one arm table (`arms.sh`, sourced by both so they cannot drift):

| | `run_hp_screen_local.sh` | `submit_rlopt_hp_search_ice.sh` |
|---|---|---|
| where | workstation, sequential | ICE, one H100 job per arm, concurrent |
| wall | ~5–6 h for ten arms | ~30–45 min, all arms at once |
| physics | PhysX | `newton_mjwarp`, njmax 320 / nconmax 40 |
| encoder | local `lafan1_latent_deterministic_5b_seed0` | `/data/pretrain_store/lafan1_v2_det_sr_h10_z256_seed0` — the encoder the 5B run is on |
| metrics | CSV in each arm dir | W&B `g1-lafan1`, group `rlopt-hparam-search` |

The ICE launcher is the one to use: it is ~10× faster in wall-clock and its `b0`
is the cluster recipe verbatim, PhysX-vs-Newton included. The local launcher
remains valid for offline iteration on the harness itself.

## Why

Analysis of `wandb g1-lafan1-strict/01sktm46` (3.5B frames, 12288 × 12, the
healthiest long run to date) and the current v2 run `g1-lafan1/spi4bxdj` found
that the optimizer settings have never been searched — across all 55 runs in
`g1-lafan1-strict` and every other project, `lr=1e-3, scheduler=adaptive,
desired_kl=0.01, epochs=5, entropy_coeff=0.005` is identical in every run,
including the one project named `gpu_lr_ablation` (which varied only GPU count
and `mini_batch_size`).

Two findings motivate the arms.

### 1. The adaptive-KL rule adapts to minibatch noise, not to policy drift

`RLOpt/rlopt/base_class.py::_maybe_adjust_lr` is a bang-bang ×1.5 / ÷1.5 rule
with a dead band of `[desired_kl/2, desired_kl*2] = [0.005, 0.02]`, and
`ipmd.py` invokes it after **every minibatch** — 80 times per iteration at
12288 × 24. Measured over 10,000 logged points of the reference run:

| quantity | value |
|---|---|
| iteration-mean `kl_approx` | 0.0136 |
| fraction of iterations with mean KL below 0.005 | 0.00 |
| fraction with mean KL above 0.02 | 0.00 |
| log-LR autocorrelation (2 iterations apart) | −0.147 |
| net LR drift per logged point | +0.000 (sd 2.55, in units of log 1.5) |
| LR geomean / total spread | 2.87e-5 / 57.7× |

The iteration-mean KL is inside the dead band **100% of the time**. Evaluated
once per iteration the rule would never have fired at all; every LR movement in
that 3.5B-frame run came from per-minibatch sampling noise. The result is a
zero-drift random walk with no fixed point, running ~30× below the configured
`optim.lr=1e-3`. Note only the *actor* group adapts — the critic is pinned at
1e-3 by `adaptive_lr: False`.

### 2. The entropy bonus dominates the actor loss

`agent.ppo.entropy_coeff=0.005` is the live knob (IPMD inherits PPO's
`ClipPPOLoss`; `agent.ipmd.entropy_coeff` is dead). At 3.5B frames:

- `loss_entropy = −0.0416` versus `loss_objective = −0.0093` — the entropy term
  is **4.5× the policy-gradient term**.
- entropy 9.59 → 8.39 over the whole run: σ ≈ 0.32 rad, essentially never
  annealing.
- `action_rate_l2` penalises **raw** actions, so independent Gaussian noise
  contributes `2 · 29 · σ² = 5.94 → −0.59` at weight −0.1. The measured
  `Episode_Reward/action_rate_l2` is **−0.53**, the largest-magnitude term in
  the entire reward. Almost all of it is exploration noise rather than policy
  behaviour, and σ = 0.32 rad of injected joint noise is a floor on MPJPE.

### 3. What this costs

95% of final quality arrives in the first 5% of the budget:

| frames | ep_len | return | MPJPE | r_step |
|---|---|---|---|---|
| 1–169M | 387 | 24.9 | 39.2 | 0.0646 |
| 533M–1.07B | 416 | 27.0 | 39.6 | 0.0648 |
| 2.64–3.50B | 447 | 30.3 | 38.4 | 0.0679 |

The critic is not implicated: explained variance is 0.976–0.981 and flat, and
`value_clip_fraction` is 0.003.

### 4. Rollout length: what the r12-vs-r24 cluster comparison really shows

The 24-step v2 job (`dhjrufgp`) is clearly behind the 12-step job (`spi4bxdj`)
both per frame and per wall-clock second, over their common range:

| frames | r12 return | r24 return | r12 ep_len | r24 ep_len |
|---|---|---|---|---|
| 5.3M | 0.154 | 0.056 | 5.41 | 5.66 |
| 10.7M | 0.387 | 0.235 | 9.29 | 7.06 |
| 21.4M | 2.097 | 0.636 | 39.67 | 13.30 |

At a matched 13.0 minutes of wall-clock, r12 had collected 21.4M frames and
reached return 2.10; r24 had collected 13.3M and reached 0.287.

**But the two jobs did not differ only in rollout length.** r24 passed
`mini_batch_size=36864`, scaling the minibatch with the rollout, so it kept 8
minibatches × 5 epochs = 40 optimizer steps per *iteration* — which halves them
per *frame*:

| run | frames/iter | minibatches | updates/iter | **updates per M frames** |
|---|---|---|---|---|
| r12 | 147,456 | 8 | 40 | **271** |
| r24 | 294,912 | 8 | 40 | **136** |

So the comparison as run confounds rollout length with a 2× cut in optimizer
steps per frame, and the latter is the more likely cause. The conclusion
"24 steps is worse" holds for the configurations actually run; "24 steps is
intrinsically worse" is not established, because 24 steps at `mini_batch_size`
18432 (which would restore 271 updates/M frames) was never run.

The screen therefore uses 12 steps — the reference geometry, so `b0` is the
cluster recipe verbatim — and turns update density into an explicit axis.

## The screen

Ten arms, each 50M frames at 12288 envs, run sequentially on the workstation.
Every arm differs from `b0_baseline` in exactly one knob. Nine run at 12 steps
(340 iterations); `a8` alone rebatches to 24 steps (170 iterations) at the same
frame budget.

| arm | steps | epochs | mini_batch | reuse | updates per M frames |
|---|---|---|---|---|---|
| `b0_baseline` | 12 | 5 | 18432 | 5× | 271 |
| `a1_updates_2x` | 12 | 5 | 9216 | 5× | 542 |
| `a2_kl_per_iteration` | 12 | 5 | 18432 | 5× | 271 |
| `a3_lr_fixed_1e4` | 12 | 5 | 18432 | 5× | 271 |
| `a4_lr_fixed_3e5` | 12 | 5 | 18432 | 5× | 271 |
| `a5_entropy_1e3` | 12 | 5 | 18432 | 5× | 271 |
| `a6_entropy_0` | 12 | 5 | 18432 | 5× | 271 |
| `a7_epochs_3` | 12 | 3 | 18432 | 3× | 163 |
| `a8_r24_matched` | **24** | 5 | 18432 | 5× | 271 |
| `a9_epochs_10` | 12 | 10 | 18432 | 10× | 542 |

Optimizer steps per frame is `epochs / mini_batch_size` — it does **not** involve
`frames_per_batch`. That identity is why the r12-vs-r24 cluster comparison
confounded two axes at once, and it is what the arms above pull apart:

- **How hard the batch is worked**: `a7` (163) → `b0` (271) → `a1` (542). And
  `a9` reaches `a1`'s 542 by reusing each sample 10× rather than by halving the
  minibatch, so "more optimizer steps" and "more sample reuse" stay
  distinguishable — the two have the same step count and the same number of
  LR-controller steps, differing only in reuse.
- **How fresh the data is**: `a8` alone changes the collection cadence, holding
  update density at `b0`'s 271. This is the cell the cluster comparison never
  ran. If `a8` ties `b0`, the r24 deficit was its halved update budget and
  24-step rollouts are fine — and cheaper per frame, being fewer, larger
  iterations. Caveat: at 295k frames with `mini_batch_size` 18432 the last update
  of an iteration sits 80 steps from the collecting policy versus `b0`'s 40, so
  `a8` carries more within-iteration drift. That is inherent to larger batches,
  not a defect in the arm.

`a4` is the control that separates "the LR is too low" from "the LR *noise* is
harmful": it holds the average LR at the value the adaptive rule already
produces and removes only the noise.

The fixed-LR arms pin the LR with `min_lr = max_lr` rather than by disabling the
scheduler, so `train/kl_approx` is still computed and logged for them.

The aggregator requires arms to match on `num_envs` and `total_frames` — the
data budget — but deliberately not on `rollout_steps`, since `a8` sees the same
50M frames and only batches them differently.

## Running it

### On ICE (preferred)

```bash
# Dry run (default): gates the RLOpt working tree, prints every arm's command.
./experiments/campaigns/2026-08-02-rlopt-hp-search/submit_rlopt_hp_search_ice.sh

# Submit all ten arms as ten concurrent H100 jobs.
DRY_RUN=0 ./experiments/campaigns/2026-08-02-rlopt-hp-search/submit_rlopt_hp_search_ice.sh

# A subset; the arm -> job-id record is merged, not overwritten.
DRY_RUN=0 ARMS="b0_baseline a2_kl_per_iteration" \
  ./experiments/campaigns/2026-08-02-rlopt-hp-search/submit_rlopt_hp_search_ice.sh

# Aggregate once the group is complete.
pixi run -e default python -m \
  imitation_experiments.lowlevel.aggregate_rlopt_hp_screen \
  --wandb_group rlopt-hparam-search \
  --wandb_arm_prefix rlopt_hp_screen_20260802_ \
  --out logs/rlopt_hp_screen_20260802/screen.md
```

Each arm is a separate `cluster_interface.sh job` submission, so the loop
repacks and uploads the ~680 MB workspace archive once per arm — budget a few
minutes per arm for the submission loop itself. The jobs queue and run in
parallel regardless. Arm → job id lands in `cluster_submission.json` along with
the workspace and RLOpt SHAs and their dirty flags.

`a2_kl_per_iteration` depends on `agent.optim.kl_adapt_step`, which is a local
RLOpt change. The `ice_runtime` profile sets `CLUSTER_ARCHIVE_SYNC=1`, which tars
the working tree (RLOpt included, `.git` excluded), so the uncommitted change is
what runs; the launcher gates on it being present rather than assuming.

### Locally

```bash
# Dry run (default): prints every arm's exact command and exits.
./experiments/campaigns/2026-08-02-rlopt-hp-search/run_hp_screen_local.sh

# Full screen, ~5-6 h.
DRY_RUN=0 ./experiments/campaigns/2026-08-02-rlopt-hp-search/run_hp_screen_local.sh

# Aggregate.
pixi run -e default python -m \
  imitation_experiments.lowlevel.aggregate_rlopt_hp_screen \
  --screen_root logs/rlopt_hp_screen_20260802 \
  --out logs/rlopt_hp_screen_20260802/screen.md
```

The aggregator reads either source and scores both identically; only the loader
differs. From W&B it takes `total_frames` from the last logged step, so an arm
that died early is geometry-mismatched and rejected rather than silently ranked
against complete ones.

The launcher verifies the corrected-LAFAN1 manifest sha256
(`d972c37c…`), the dataset cache, and the encoder checkpoint before spending any
GPU time, and refuses to write into an arm directory that already holds a
completed run. Metrics go to RLOpt's CSV backend inside each arm directory, not
to W&B: a screen is local and offline-reproducible and does not belong in the
shared training project.


## Results (50M screen, complete)

Scored on the time axis: `ret/min` and `len/min` are progress per minute of
training wall-clock (`time/collecting + time/training` summed, excluding the ~7
min Isaac startup every arm pays identically).

| arm | ret/min | len/min | ep_len | MPJPE mm | verdict |
|---|---|---|---|---|---|
| `b5_term_curriculum` | 0.892 | 11.51 | 278.4 | **99.18** | rejected — see below |
| `b4_silu` | **0.528** | **6.49** | 209.3 | **66.76** | **genuine win** |
| `b2_adv_global` | 0.517 | 6.17 | 174.2 | — | mild win |
| `b3_gradclip` | 0.516 | 6.27 | 174.5 | — | mild win |
| `e1_gamma097` | 0.501 | 5.28 | 156.6 | — | mixed |
| `a12_kl_iter_entropy0_obsnorm` | 0.495 | 5.96 | 173.3 | 71.59 | prior champion |
| `a17_a12_bigcritic` | 0.458 | 5.75 | 162.5 | — | neutral/negative |
| `e2_widenet` | 0.410 | 6.13 | 171.1 | — | len/min > a12 |
| `e3_sonicnet` | 0.358 | 6.08 | 188.3 | — | len/min > a12, return still climbing |
| `a10_obs_norm` | 0.340 | 5.16 | 161.0 | — | +50% ep_len over b0 alone |
| `b0_baseline` | 0.251 | 3.74 | 115.5 | 70.63 | cluster recipe |
| `d1`/`d2` bounded sigma | 0.227/0.309 | 2.52/3.97 | 79/118 | — | rejected |
| `a13_kl_iter_nophase` | 0.022 | 0.60 | 21.0 | — | phase vector is essential |
| `c1`-`c5` low-init sigma | ~0.01 | ~0.15 | ~6.5 | — | trapped, see below |

### Both goal metrics are gameable, and one arm gamed them

`b5_term_curriculum` posts the best rates in the campaign -- 1.80x on ret/min,
1.93x on len/min -- and its MPJPE is **99.18 mm against a12's 71.59, 39%
worse**. Per-minute rates are computed over quantities that grow with episode
length, and episode length is set by the termination thresholds, so loosening a
threshold raises both rates mechanically while the policy tracks worse. It is a
relaxed test, not a result, and it is not in the recipe.

MPJPE is per-frame and length-independent, which is why it is the check. The
aggregator now emits a "Rate gained, tracking lost" section for any arm that
beats the baseline on rate while losing MPJPE.

`b4_silu` is the counter-example and the reason SiLU is in the recipe: longer
episodes (209.3 vs 173.3) **and** the best MPJPE of any arm (66.76), at
unchanged per-step reward.

### Two mechanism findings worth keeping

**`log_std_init` is a trap below the operating point.** Arms `c1`-`c5` sat at
episode length 5-8 for entire blocks. Adam moves `log_std` by about the learning
rate per step; the adaptive geomean is 2.8e-5, so a 50M block moves it ~0.2 in
log space. From sigma 0.05 that reaches 0.061 -- the policy cannot recover to the
0.36 it needs to explore, and never starts learning. `c5` changes only this knob
from a12 and reproduces the 2026-07-20 release-contract failure (episode length
6.1 against that run's 6.6), which attributes that whole bundle failure to one
knob.

**`clip_log_std` freezes sigma when the init is outside the clamp.**
`torch.clamp` passes zero gradient outside its range, so capping at log(0.5)
while initializing at 0.0 pins sigma at exactly 0.5 for the whole run, silently.
Any clipped arm must initialize inside the range.

## Recipe

```
agent.optim.kl_adapt_step=iteration
agent.ppo.entropy_coeff=0.0
agent.policy.normalize_input=true
agent.value_function.normalize_input=true
agent.policy.activation_fn=silu
agent.value_function.activation_fn=silu
```

Not carried: termination curriculum (relaxes the test), wider/deeper critic
(neutral to negative at 50M), reference EE in the critic, fixed learning rates,
doubled optimizer steps, bounded sigma, `log_std_init` below the operating
point.


## Final result (100M budget, 3 seeds)

Against `b0_baseline` — the cluster recipe verbatim — at the same 100M budget:

| | b0_baseline | **m4_gamma097** (3 seeds) | gain |
|---|---|---|---|
| return / min | 0.279 | **1.290** ± 0.026 | **4.62×** |
| ep_len / min | 3.92 | **11.12** ± 0.22 | **2.84×** |
| MPJPE mm | 69.18 | **61.74** ± 0.53 | 10.8% better |
| episode length | 181.3 | **301.2** | 1.66× |
| wall-clock | 46.2 min | **27.1 min** | 1.7× faster |

Seed spread is ~2% of the mean, so every gain here is far outside noise. The
tracking metric improves alongside the rates, which is what distinguishes this
from the relaxed-test artifact described below.

### The recipe

```
agent.optim.kl_adapt_step=iteration          # KL rule fires per iteration, not per minibatch
agent.optim.desired_kl=0.02                  # peaked; 0.04 is worse
agent.ppo.entropy_coeff=0.0                  # re-confirmed on the final base
agent.policy.normalize_input=true            # command excluded, already declared
agent.value_function.normalize_input=true
agent.policy.activation_fn=silu
agent.value_function.activation_fn=silu
agent.policy.num_cells=[1024,1024,512]
agent.value_function.num_cells=[1024,1024,512]
agent.loss.gamma=0.97
env.rewards.action_rate_l2.weight=0.0
env.enable_termination_curriculum=true
env.termination_curriculum_start_frames=5000000
env.termination_curriculum_end_frames=30000000
```

Two of these need a code change that is in this branch: `optim.kl_adapt_step`
(RLOpt) and `env.enable_termination_curriculum` plus its window fields (v2 env
config).

### What mattered, in order

1. **Termination-threshold anneal, completed early.** Largest single effect, but
   only once the anneal *finishes inside the budget*. See the trap below.
2. **Observation normalization.** +50% episode length on its own, and it was
   simply switched off with the exclusion list already correct.
3. **Removing the action-rate penalty.** It was the largest-magnitude reward
   term and 28% of positive reward mass early; at weight 0 both rates and MPJPE
   improve.
4. **KL rule per iteration**, and `desired_kl` 0.02.
5. **SiLU**, and **width** `[1024,1024,512]`.
6. **Discount 0.97** — mixed on the original base, a win on the final one.

### The trap this campaign is really about

Both goal metrics are per-minute rates over quantities that grow with episode
length, and episode length is set by the termination thresholds. Loosening a
threshold raises both rates **mechanically**, with no better policy.

`b5_term_curriculum` did exactly that: 1.80× on return/min with MPJPE at 99.18 mm
against 71.59. It looked like the best arm in the campaign and was a relaxed
test. What resolved it was compressing the anneal so it *completes* inside the
budget (`j1`, then `k3`): the scored tail then runs at strict thresholds, MPJPE
returns to the honest 61-64 band, and the speed advantage survives. The
curriculum is a genuine accelerator; the 500M window simply could not show it at
this budget.

`k3` vs `k4` isolates the mechanism directly — ending the anneal at 30M rather
than 100M buys 60.83 mm against 65.25, because more of the budget then trains
under strict thresholds.

**MPJPE is per-frame and length-independent, which is why it is the check.** The
aggregator emits a "Rate gained, tracking lost" section for any arm that beats
the baseline on rate while losing MPJPE.


### Does the curriculum window need retuning at scale? No.

The window was the last open risk for transferring this recipe: at 100M its
optimum (5M-30M) is simultaneously "5M-30M absolute" and "5%-30% of budget",
and those extrapolate to very different settings at 5B. Run at 200M, where the
two hypotheses separate:

| arm (200M) | ret/min | len/min | MPJPE mm |
|---|---|---|---|
| `n1_window_absolute` (5M-30M) | 0.865 | 7.19 | 58.24 |
| `n2_window_scaled` (10M-60M) | 0.852 | 7.12 | 58.71 |

1.5% / 1.0% / 0.8% apart, all inside the ~2% seed spread. The window is
**insensitive in this range**, so it does not need retuning with budget --
anything around 5-30M (or its proportional equivalent) works.

Two things to note about the 200M numbers. MPJPE keeps improving with budget
(58.2 at 200M against 61.4 at 100M), so tracking has not plateaued. And the
per-minute rates are *lower* at 200M than at 100M (0.865 vs 1.303) because gains
get harder as the policy matures -- **rates are budget-dependent and must only be
compared within a budget**, which is why `n1` vs `n2` is the valid comparison
here and neither should be read against the 100M table.

### Rejected, with reasons

| change | why not |
|---|---|
| `log_std_init=log(0.05)` | traps the policy: Adam moves log_std ~0.2/block at LR 2.8e-5, so sigma cannot climb from 0.05 to the 0.36 it needs. Reproduces the 2026-07-20 release-contract failure on its own |
| `clip_log_std` with init outside the clamp | zero gradient outside `torch.clamp`; sigma freezes silently |
| bounded sigma (`d1`/`d2`) | worse on every axis than deleting the bonus |
| depth at fixed width (`i1`) | neutral — so h3's edge was width, and residual/LayerNorm has nothing to fix |
| 2048x6 architecture | loses on every base tried |
| wider critic alone | neutral to negative |
| dropping the sin/cos phase | catastrophic (ep_len 21 vs 144) |
| reference EE in the critic | neutral to slightly negative |
| fixed LRs, 2x updates, fewer/more epochs, `gae_lambda` 0.98 | all lose |


## The recipe, as a default

Registered as `rlopt_ipmd_tuned_cfg_entry_point`
(`G1ImitationTunedRLOptIPMDConfig`), so the agent half needs no overrides:

```bash
--agent rlopt_ipmd_tuned_cfg_entry_point
```

It is a NEW config class, not a change to the existing local optimizer
contract. Every prior run, the in-flight 5B job and the paper-facing v2
campaign all resolve that contract, and redefining it would silently change
what those runs mean -- the same reason G1 task ids are versioned rather than
mutated.

The environment half lives on the env config and must still be passed:

```bash
env.rewards.action_rate_l2.weight=0.0
env.rewards.tracking_reward_points.weight=4.0
env.enable_termination_curriculum=true
env.termination_curriculum_start_frames=5000000
env.termination_curriculum_end_frames=30000000
```

`submit_tuned_5b_ice.sh` is the reference invocation.

### What is deliberately not in it

**Geometry: rollout 12 -> 6, and this took three attempts to read correctly.**
Scored at 23 training-minutes -- a mark EVERY arm in the campaign reached, so
nothing is clamped to its endpoint:

| arm | rollout | ep_len | return | MPJPE |
|---|---|---|---|---|
| t1_r4 | 4 | 319.9 | 47.79 | 56.04 |
| s1_envs12k_r6 | 6 | 313.5 | 46.97 | 60.55 |
| r0_champion | 12 | 293.6 | 43.77 | 60.27 |
| t2_r3 | 3 | 295.3 | 42.85 | 62.53 |

`s1` and `r0` differ in `collector.frames_per_batch` and nothing else --
verified field by field -- so this is a clean single factor worth **+7.3%
return and +6.8% episode length** at unchanged MPJPE. The optimum is broad
across 4-6 and collapses at 3, where the GAE horizon (gamma 0.97, lambda 0.95)
is too short.

It does NOT change optimizer work per frame: update density is
`epochs / mini_batch_size` and does not involve the batch size. What halving the
batch changes is how often data is recollected -- twice the iterations, so twice
the LR-controller adaptations and half the drift between the collecting policy
and the last update on a batch.

Environment COUNT is a separate axis and does not help: `s3` (20480 x 6) and
`q2` (24576 x 6) do not beat `s1` (12288 x 6). An earlier reading credited the
environments; it was the rollout.

Two wrong intermediate conclusions are recorded here because the failure mode
is easy to repeat. Scoring arms at a mark some of them had not reached clamps
the short runs to their final value and flatters them, which first made a
shorter rollout look far better than it is, and then -- when the top six were
compared only against each other -- made the whole axis look like noise. Only a
mark every arm reached is safe.

**More optimizer work.** Five arms across four bases lost by raising
updates-per-frame (`a1`, `a9`, `i5`, `m3`, `t3`). Update density is
`epochs / mini_batch_size` and does not involve the batch size.

**Bigger or deeper networks.** 2048x6 lost on every base; depth at fixed width
was neutral, so width is what helps and there is nothing for residual
connections to fix.

## Scope and limits

**This is a screen, not a result.** 50M frames at this geometry is 340
iterations; the reference run needs roughly 1200 to reach an episode length of
400. The screen resolves early-phase learning rate and optimizer health — which
is where both findings act — and says nothing about the 3B-frame plateau. Any
arm that wins here needs a longer confirmation run before it changes the
cluster recipe.

Local measurements at 12288 envs: 55.6 GB of 96 GB VRAM at 24 steps (12 steps is
lower), ~180 s of Isaac startup per arm, throughput rising as episodes lengthen.

## Supporting changes

- `RLOpt/rlopt/config_base.py`: new `optim.kl_adapt_step: "update" |
  "iteration"`, defaulting to `"update"` so existing runs are bit-identical.
- `RLOpt/rlopt/base_class.py`: `_record_kl_for_lr_adaptation` /
  `_flush_kl_lr_adaptation` implement the routing; `_maybe_adjust_lr` is
  unchanged.
- `RLOpt/rlopt/agent/{ipmd/ipmd,ppo/ppo}.py`: wired to the new routing.
- `RLOpt/rlopt/logging_utils.py`: bug fix — TorchRL's `get_logger` routes the
  file-backed loggers to `logger_name` and ignores `log_dir`, so
  `logger.backend=csv` wrote scalars to `./ipmd/` beside the process CWD instead
  of the configured run directory.

## Status

| stage | state |
|---|---|
| RLOpt change + tests | done (`pixi run test-rlopt`, 113 passed) |
| aggregator + tests | done (22 passed, incl. the W&B source) |
| 10-arm screen | submitted to ICE 2026-08-02 19:42–19:53 |

ICE array, seed 0, 50,135,040 frames each, W&B group `rlopt-hparam-search`:

| arm | job |
|---|---|
| `b0_baseline` | 5558455 |
| `a1_updates_2x` | 5558469 |
| `a2_kl_per_iteration` | 5558483 |
| `a3_lr_fixed_1e4` | 5558484 |
| `a4_lr_fixed_3e5` | 5558485 |
| `a5_entropy_1e3` | 5558487 |
| `a6_entropy_0` | 5558490 |
| `a7_epochs_3` | 5558491 |
| `a8_r24_matched` | 5558492 |
| `a9_epochs_10` | 5558493 |

Submitted against workspace `999e2f4` and RLOpt `df052d6`, both dirty — the
`kl_adapt_step` routing `a2` needs is uncommitted, and archive sync ships the
working tree. `cluster_submission.json` holds the same mapping plus those SHAs.
