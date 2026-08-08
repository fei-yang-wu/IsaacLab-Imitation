# BONES-SEED 129k ablation screen (ICE H200)

Seven arms — a control plus six one-variable deltas — against the full
129,785-motion BONES-SEED set, to find what moves tracking beyond the recipe the
local 10B run was converging toward.

An eighth arm, `physx`, was submitted and **dropped on 2026-08-05** after two
failures. It stays defined in `arms.sh` and runs on request
(`ARMS="physx" ...`); see "The `physx` arm is blocked" below before resubmitting.

**Each arm runs to the wall.** Both `ice-gpu` and `coe-gpu` cap at 16:00:00, so
that is the budget; `FRAME_CAP` is set past anything reachable and the job ends
on TIMEOUT. That is safe here because `log_dir` is under `/data`, which binds to
persistent scratch — a Slurm TIMEOUT is a hard SIGKILL that runs no final save,
so checkpoints on node-local storage would be lost, while these survive with at
most `SAVE_INTERVAL` (50M frames, ~8 min) unsaved.

Expect roughly 5.7B frames for a Newton arm at ~100k fps and ~3.4B for `physx`
at ~0.6×. **Arms therefore do not finish at equal frame counts, so score them at
a common mark** — the largest 50M multiple every arm reached — not at whatever
each one happened to stop at.

## Why: the local run was not plateaued

Its training curves said it was. MPJPE-L sat at 41–44 mm from 0.35B to 4.0B and
`reference_finished` fell monotonically 0.307 → 0.223. Scored instead on the
fixed protocol — frame 0, `--randomization none`, MODE actions, 1024 envs, 500
steps — the same checkpoints improve on every axis:

| ckpt | MPJPE | joint RMSE | survival | success | `ref_finished` | ee term | foot term |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.35B | 25.8 mm | 0.2187 | 222.6 | 0.606 | 0.497 | 0.200 | 0.198 |
| 2.00B | 23.9 | 0.2078 | 235.7 | 0.668 | 0.549 | 0.165 | 0.163 |
| 4.03B | **23.4** | **0.1980** | **241.7** | **0.692** | **0.569** | **0.162** | **0.145** |

The flat training curves were the `sonic` adaptive sampler hardening the task at
about the rate the policy improved. **Under adaptive resets, training metrics
cannot distinguish "policy plateaued" from "curriculum kept pace." Score
checkpoints on the fixed protocol or do not draw conclusions.**

### The training metric and the eval metric cannot be reconciled — measured

Attributed on 2026-08-05 against the control checkpoint at ~3.0–3.2B, varying
one factor at a time toward the training condition
(`logs/train_eval_gap/`):

| eval config | mpjpe_l | mpjpe_g | root_pos | root_ori | survival |
|---|---:|---:|---:|---:|---:|
| protocol: frame0 / rand=none / mode | 25.25 | 0.2179 | 0.2113 | 0.0458 | 251.6 |
| + `randomization=all` | 27.93 | 0.3804 | 0.3748 | 0.0525 | 235.2 |
| + full-trajectory starts | 25.54 | 0.2108 | 0.2040 | 0.0464 | 240.9 |
| + stochastic actions | 25.40 | 0.2257 | 0.2191 | 0.0464 | 248.0 |
| all three together | 28.67 | 0.3505 | 0.3444 | 0.0557 | 225.3 |
| **training log @3.23B** | **44.74** | **0.2758** | **0.4508** | **0.1407** | **~100** |

**The global/root gap is domain randomization and is fully explained** —
`root_pos` 0.2113 → 0.3748 against training's 0.4508. On `mpjpe_g` eval with
randomization is actually *worse* than training (0.3505 vs 0.2758), so global
tracking is a crossover, not a gap.

**The local and orientation gap is the averaging, not the rollout.** All three
factors together move `mpjpe_l` only 25.25 → 28.67 against 44.74. The cause is
structural: IsaacLab's `CommandTerm.reset` logs
`torch.mean(metric_value[env_ids])` — an unweighted mean over *resetting* envs
of each env's *episode* mean, **one sample per episode regardless of length** —
while evaluation averages over *transitions*. Episode length correlates almost
perfectly with quality (episodes <50 steps have success 0.000; 300+ steps have
0.934), and episodes under 50 steps are **20.0% of episodes but 1.1% of
transitions**. The training metric therefore weights the worst episodes ~18×
more heavily.

Training's adaptive sampler also has a *learned* failure-concentrated start
distribution that a fresh sampler cannot reproduce, which is why training
`ep_len` is ~100 against 225 even at matched settings. It pushes the same way.

**So do not try to make them agree, and do not read arm-vs-arm differences off
the training curves.** Rank on the fixed protocol. The diagnostic knob is
`--action_sampling stochastic`, which is not a protocol option and is recorded
in the output so a diagnostic run cannot be mistaken for a score.

So this screen beats a working recipe, not a broken one — and one with
decelerating returns: 0.35B→2.00B bought 1.9 mm over 1.65B frames, 2.00B→4.03B
bought 0.5 mm over 2.03B. An arm must beat the **control at the same frame
count**, and must clear the ~0.25 mm/B baseline drift to mean anything.

## The arms

Each is one delta from `control`, which reproduces the local run exactly. See
`arms.sh` for the reasoning attached to each.

| arm | delta | why |
|---|---|---|
| `control` | — | reproduces the local 10B recipe |
| `physx` | `physics=physx` | a policy-free oracle probe has PhysX tracking 3× better than Newton (joint MAE 0.0327 vs 0.0975 rad), with stock MuJoCo agreeing with PhysX. Every checkpoint we own is Newton-trained. |
| `reset_window50` | `adaptive_pre_failure_window` 200 → 50 | 200 frames is 4 s against a **287-frame median clip**; the credit window covers most of a median motion |
| `reset_cap50` | `adaptive_failure_rate_max_over_mean` 200 → 50 | `sonic` allows 4× the `default` preset's concentration, and it demonstrably outran the policy |
| `entropy0` | `entropy_coeff` 0.008 → 0 | constant, never annealed; the tuned recipe that won the LAFAN1 HP search used 0 |
| `encoder_finetune` | unfreeze at `hl_skill_lr=3e-5` | frozen from a 50k-update pretrain, never adapted to the controller |
| `gamma099` | `gamma` 0.97 → 0.99 | 0.97 at 50 Hz is a 0.67 s horizon; the dominant failures (`ee_body_pos` 0.162, `foot_pos_xyz` 0.145) are recovery-shaped |
| `rollout24` | rollout 6 → 24 | see below |

### The `physx` arm is blocked

Two attempts died at ~50 s: **5567113** on `gpu:h200:1` and **5567121** on
`gpu:h100:1` — the configuration the 2026-08-03 5B run proved. So the GPU is not
the cause, and neither is the GPU policy guard: the H100 log shows

```
[INFO] PhysX GPU policy accepted: NVIDIA H100 80GB HBM3
[ISAACLAB] AppLauncher initialization complete
```

**Kit starts on Hopper.** The process then exits *after* config parsing and
*before* env construction, with **status 0 and no traceback** — only the missing
workload success marker reports it. The `/isaac-sim/kit/data` write errors in
the log are non-fatal noise that Kit continued past.

Reproduce interactively at `--num_envs 64` and find the exit path before
resubmitting. A blind retry costs a 16 h slot and has already failed twice.

### On `rollout24`

The 2026-08-02 screen compared rollout 3/4/6/12 and concluded "+7.3% return and
+6.8% episode length at unchanged MPJPE". That table's MPJPE column is
`mpjpe_mm`, which the aggregator resolves to **`mpjpe_l_mm` — root-relative
only**. Global tracking was never in it, 24 was never tested, and it ran on
LAFAN1 at 23 training-minutes. Meanwhile the eval-tracking screen's own
conclusion was that *accumulating root drift* dominates eval-time failure. So
the axis the rollout screen measured is not the axis in question.

**Report `mpjpe_g` and `anchor_pos_err` for this arm, not just `mpjpe_l`.**

`minibatch` scales with `frames_per_batch` so update density
(`epochs / mini_batch_size`) is constant — otherwise `rollout24` would silently
also be an update-density arm.

## Results at 2B frames

Fixed protocol, `logs/ablation_scores_2B/`. `mpjpe_l` is root-relative;
`mpjpe_g` is `tracked_body_pos_error_m`, the global one the 2026-08-02 screen
never reported. **Anything within ±5% is noise** — that band is set by the
~12% eval standard deviation measured in the LAFAN1 mechanism study, applied to
a single seed.

| arm | mpjpe_l mm | mpjpe_g m | root_pos m | root_ori rad | survival | success |
|---|---:|---:|---:|---:|---:|---:|
| `control` | 25.78 | 0.2360 | 0.2298 | 0.0465 | 244.5 | 0.721 |
| `rollout24` | **21.92** | 0.1555 | 0.1506 | **0.0397** | **250.3** | **0.740** |
| `gamma099_rollout24` | 27.50 | **0.0937** | **0.0900** | 0.0549 | 243.7 | 0.719 |
| `gamma099` | 32.07 | 0.1420 | 0.1327 | 0.0601 | 235.9 | 0.682 |
| `reset_cap50` | 25.77 | 0.2152 | 0.2096 | 0.0478 | 238.2 | 0.690 |
| `entropy0` | 25.77 | 0.2302 | 0.2252 | 0.0470 | 238.7 | 0.690 |
| `reset_window50` | 26.20 | 0.2559 | 0.2500 | 0.0480 | 239.6 | 0.698 |
| `encoder_finetune` | 29.88 | 0.2865 | 0.2836 | 0.0585 | 195.4 | 0.456 |

Four findings.

**`rollout24` wins on every axis at 2B** — local −15.0%, global −34.1%, root
ori −14.6%, survival and success both up. It is the only arm that improves
global and local together. This is exactly what the 2026-08-02 screen could
not have seen: its MPJPE column was root-relative, so the −34% global gain was
invisible and 24 was never tested.

**Update, 4B: the global lead is a transient, the local lead is not.**
mpjpe_g vs control compresses hard as frames accumulate — 2B −34.1%, 3B −29.6%,
4B −15.6% — because control's own global tracking keeps improving (0.2360 →
0.2179 → 0.1954) while `rollout24`'s plateaus (0.1555 → 0.1533 → 0.1649).
Control is converging on `rollout24`, not the other way around. What holds
flat across all three marks: local MPJPE ~−15%, root ori ~−15 to −18%, survival
+1 to +3%, success +2 to +5%. So `rollout24`'s durable contribution is
root-relative pose and stability, plus reaching good global tracking several
billion frames sooner — not a permanent global-tracking edge.

**`gamma099` trades, it does not win.** Global −39.8% (the best in the screen)
bought with local +24.4% and root ori +29.3%. Longer credit fixes root drift and
loosens the pose. `gamma099_rollout24` exists to test whether the two compose;
read it only alongside both singles.

**`gamma099_rollout24` composes super-additively on global tracking, and the
lead is durable.** vs control: 2B mpjpe_g −60.3%, 3B −53.8% — a small, controlled
slip. Contrast `rollout24` alone, whose lead *collapsed* over the same frames:
2B −34.1%, 3B −29.6%, 4B −15.6% (see below). The composition's own absolute
numbers barely move between 2B and 3B (mpjpe_g 0.0937 → 0.1008, local
27.50 → 28.51 mm) while control keeps closing under it — a much slower
convergence than `rollout24` alone shows.

The cost side does **not** stay free, though. At 2B it looked like +6.7% local
for −60.3% global with survival/success at parity; at 3B local is +12.9%
(creeping back toward `gamma099`'s standalone +27.7%) and both survival (−2.4%)
and success (−3.9%) have moved just outside the noise band. So the honest
picture is: real, durable global-tracking win, at a real and slightly growing
local/stability cost — not the free lunch the 2B point suggested.

**Settled at 4B: the lead holds.** vs control, mpjpe_g: 2B −60.3%, 3B −53.8%,
4B **−51.5%** — a small further slip, then it holds. Its own absolute mpjpe_g
actually *improved* 2B→4B (0.1008 → 0.0947): this is not a decaying head start,
it is a genuinely different regime. Contrast `rollout24` alone, whose lead over
the same three marks *collapsed* (−34.1% → −29.6% → −15.6%) because control
converges under it. Local cost stabilized too (+6.7% → +12.9% → +19.0%, flat
3B→4B rather than still climbing), and survival/success flipped **positive**
at 4B (+3.2%, +3.3%) — the arm is now strictly better than control on
stability, not merely at parity.

**This is the strongest and most durable result in the screen.** Real,
durable global-tracking gain; real but stable ~19% local-pose cost; no
stability penalty — a net stability gain, if anything. **If world-frame
tracking is the target this is the arm to beat; if root-relative pose fidelity
is the sole target, `rollout24` alone stays cleanest** (it is the only arm
improving both axes), but its global edge alone is transient. One seed.

**`encoder_finetune` is a clear loss** — every metric 16–26% worse, survival
−20%, success −37%. Uniform degradation across all axes, not a trade-off.

The two reset arms and `entropy0` land inside the noise band on local MPJPE and
survival. `reset_cap50`'s −8.8% global is the only signal among them and is
worth one confirmation seed before it is believed.

### `encoder_finetune`: the co-adaptation is real, and so is the loss

The obvious way this result could be an artifact is if the eval silently paired
the finetuned policy with the *frozen* encoder. It does not. RLOpt persists the
finetuned encoder — not as a top-level key, which is why grepping the IPMD agent
for `skill_encoder` finds nothing, but inside
`hl_skill_command_sampler_state_dict`, written by the sampler's duck-typed
`checkpoint_state_dict()` (`ipmd.py:2916` → `hl_skill_diffsr.py:992`) and
restored by `agent.load_model()`.

Verified directly: at step 2000093184 the block carries `finetune_updates:
203460` and an encoder that differs from control's frozen copy by up to 41% in
mean-relative terms. Re-scoring the *same checkpoint with only that block
stripped* collapses it to survival 43.4 / success 0.003 against 195.4 / 0.456
intact (`logs/encoder_binding_probe/`). The policy is deeply co-adapted to its
own encoder, the binding works, and the regression is real.

**Consequence for any future finetune arm:** its checkpoints are not
interchangeable with frozen-encoder checkpoints, and a checkpoint that loses the
sampler block is worthless rather than merely degraded.

## Final result: terminal checkpoints, two protocols

The screen ran each surviving arm to its own 16:00:00 wall, so final checkpoints
land at different frame counts: `control` 5.90B, `gamma099` 5.65B, `rollout24`
6.25B, `gamma099_rollout24` 6.55B. Scored two ways.

**This repo's fixed protocol** (frame 0, `--randomization none`, mode,
1024 envs, 500 steps) ranks on world-frame tracking:

| arm | mpjpe_l mm | mpjpe_g m | root_ori rad | survival | success |
|---|---:|---:|---:|---:|---:|
| `control` | 24.78 | 0.2214 | 0.0448 | 243.9 | 0.732 |
| `rollout24` | 21.04 | 0.1507 | 0.0381 | **257.0** | **0.784** |
| `gamma099` | 31.57 | 0.1182 | 0.0618 | 236.6 | 0.694 |
| `gamma099_rollout24` | 28.38 | **0.0951** | 0.0514 | 254.0 | 0.767 |

**SONIC's own criterion** (`wiki/sonic-success-evaluation.md`: `no_push`, mode,
4096 motions, SONIC's three released termination terms only, success =
`reference_finished AND NOT any failure`) does not score world-frame drift at
all — only completion and root-relative pose:

| arm | frames | SONIC SR | success-only MPJPE-L |
|---|---:|---:|---:|
| `control` | 5.90B | 90.2% | 28.77 mm |
| `gamma099` | 5.65B | 89.7% | 35.99 mm |
| **`rollout24`** | 6.25B | **92.4%** | **24.28 mm** |
| `gamma099_rollout24` | 6.55B | 92.3% | 31.78 mm |

**The two protocols disagree, and both are reading real, different things.**
`gamma099_rollout24` wins the fixed protocol because it wins world-frame
tracking by a wide margin (mpjpe_g −57% vs control). `rollout24` wins under
SONIC because SONIC's criterion has no axis to reward that global gain, and the
composition's local-pose cost (root-relative MPJPE +14–19% relative to
`rollout24` throughout) is the only thing SONIC's metric can see. The two arms
tie on completion rate (92.4% vs 92.3%, inside noise).

**If reporting SONIC SR / SONIC MPJPE-L, `rollout24` is the correct pick,
cleanly** — better success, 31% better success-only MPJPE-L, no offsetting
axis. If world-frame/drift tracking is the target, `gamma099_rollout24` is
still the arm to beat. Do not average or otherwise combine the two protocols'
numbers; report whichever one the audience expects and say which it is.

Result files: `logs/ablation_scores_FINAL/` (fixed protocol) and
`logs/ablation_scores_SONIC/` (SONIC-compatible), both against the terminal
checkpoint of each arm.

## Data

The reference arrays: 49.4 GB holding exactly what the two derived caches
consume, memory-mapped at job start. The Zarr (196 GiB, ~5.3M files) and the
95 GiB replay are not on ICE and are not needed. Built by
`imitation_experiments.data.build_reference_arrays`, published at
`GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays`, fetched to
`/data/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1`.

Encoder pinned by hash `d191d865…f8c5e7` at
`/data/pretrain_store/bones129k_v2_root_qpos_det_sr_h10_z256_seed0/`.

## GPU

H200, no flag, either backend. **Headless PhysX runs on Hopper** — settled
2026-08-03 (`PhysX GPU policy accepted: NVIDIA H100 80GB HBM3`, 33,421 fps to
300M frames). Any text claiming Kit needs an RT-capable device, or that ICE's
PhysX-qualified parts are L40S/A40/RTX6000, is stale. Expect PhysX at roughly
0.6× Newton throughput; the budget is in frames, not hours.

## Run it

```bash
DRY_RUN=1 experiments/campaigns/2026-08-05-bones129k-ablation-h200/submit_bones129k_ablation_ice.sh
DRY_RUN=0 experiments/campaigns/2026-08-05-bones129k-ablation-h200/submit_bones129k_ablation_ice.sh
DRY_RUN=0 ARMS="physx rollout24" experiments/campaigns/2026-08-05-bones129k-ablation-h200/submit_bones129k_ablation_ice.sh
```

`DRY_RUN=0` first runs remote gates: the sidecar's motion count, transition
count, `persist_id` and body count; no `.incomplete` downloads; the encoder
hash; and that the arm-bearing code is in the working tree the archive ships.

## Scoring

Score every arm the same way the plateau was diagnosed — the fixed protocol, not
the training curve:

```bash
pixi run -e isaaclab python -u -m imitation_experiments.lowlevel.evaluate_checkpoint \
  --task Isaac-Imitation-G1-v2 --algo IPMD --checkpoint <ckpt> \
  --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  --randomization none --num_envs 1024 --steps 500 --seed 0 \
  --reference_start_frame 0 --reset_schedule sequential \
  env.data.manifest=null env.data.reference_arrays_dir=<arrays> ...
```

**Trust survival and MPJPE.** `completed_requested_horizon_rate` is not a
quality metric here: the median clip is 287 frames against a 500-step horizon,
so most episodes end because the reference ran out.
