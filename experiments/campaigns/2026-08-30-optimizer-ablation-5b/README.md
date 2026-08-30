# Optimizer ablation at 5B (`diffntp_chunk_h1_ee_wide` base)

Twelve arms, one optimizer field each except `ent_sonic`. Nothing else moves: same encoder, same
environment, same reward set, same command interface, same 20,480 x 24
geometry, same 5B budget, seed 0.

Status: **SUBMITTED 2026-08-30**, seed 0, twelve arms, 24 jobs. W&B group
`optimizer-ablation-5b`, project `g1-bones-seed`.

| arm | seg1 | seg2 | W&B id |
|---|---:|---:|---|
| `ctrl` | 5598837 | 5598838 | `oa5b-ctrl-s0` |
| `mb_half_e5` | 5598839 | 5598840 | `oa5b-mbh5-s0` |
| `mb_half_e3` | 5598841 | 5598842 | `oa5b-mbh3-s0` |
| `mb_full_e5` | 5598843 | 5598844 | `oa5b-mbf5-s0` |
| `mb_full_e3` | 5598845 | 5598846 | `oa5b-mbf3-s0` |
| `critic_lin` | 5598847 | 5598848 | `oa5b-clin-s0` |
| `wd_1e2` | 5598849 | 5598850 | `oa5b-wd2-s0` |
| `wd_1e4` | 5598851 | 5598852 | `oa5b-wd4-s0` |
| `wd_1e1` | 5598853 | 5598854 | `oa5b-wd1-s0` |
| `ent_only` | 5598855 | 5598856 | `oa5b-ent-s0` |
| `floor_late` | 5598858 | 5598859 | `oa5b-floor-s0` |
| `ent_sonic` | 5598860 | 5598861 | `oa5b-entsonic-s0` |

Code state: `2218f29` (top level) with `RLOpt 27e8741`. `submit` packs the
working tree, and the tree carried unrelated uncommitted work at submission
time (the `hl_skill_diffsr.py` `*_z_explained` diagnostic in RLOpt, the
endpoint-collapse-probe edits, the suffix-arm aggregator), so every plan
records `drift=true`. None of it is on the tracker path these arms exercise.

`mb_full_e5` / `mb_full_e3` cleared the memory question: both ran past
iteration 1900 with no OOM, so the +33% peak activation fits on the H200.

## Both entropy arms cancelled (2026-08-30, ~2h in)

`ent_sonic` (5598860/5598861) and `ent_only` (5598855/5598856) were cancelled
on user direction. Neither needs more budget.

`ent_sonic` reproduced the `smooth-ablation-5b/sigma` death exactly: episode
length pinned 16-22 from iteration one through 220M frames, against 150-180 in
every sibling. **Verdict: SONIC's `log_std_init=0.05` clamp is what kills the
contract from scratch on our optimizer stack, and `entropy_coeff=0.01` does
not rescue it.** `ent_only` carries the same bonus with our init 1.0 and had
healthy episode length throughout, so the bonus is not the cause. That closes
the question the arm was built for, and it reads against `sigma` as designed.

`ent_only` was not pinned -- episode length 156.8 and rising at cancellation --
but it lost on reward rate at every matched point, five in a row:

| frames | `ent_only` r_step | field r_step |
|---:|---:|---|
| 210M | 0.1259 | 0.150-0.185 |
| 370M | 0.1314 | 0.165-0.199 |
| 530M | 0.1324 | ~0.207 |
| 690M | 0.1317 | 0.184-0.213 |

A third of the reward rate at equal episode length is the shape of a policy
held wide by the bonus. This reproduces the 2026-08-02 verdict that retired
the entropy bonus, now in the multi-billion-frame regime the earlier screen
could not reach. PRELIMINARY: training-curve evidence at 690M of a 5B budget,
one seed, never scored on the board. Its 250M and 500M checkpoints survive on
ICE if a scored row is ever wanted.

`floor_late` continues -- the exploration question that stays open is the late
floor, not the bonus and not the init.

## Base

`2026-08-28-smooth-ablation-5b/base` reproduced verbatim — the frozen
`diffntp_chunk_h1_ee_wide` encoder
(`/data/pareto_stack/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt`),
20,480 envs, `env.rewards.action_rate_l2.weight=-0.03`, the SONIC 0.8 -> 0.2
failure-share ramp over the first 1B, corrected `feet_acc`.

`ctrl` re-runs that recipe inside this campaign rather than citing the 08-28
row. `submit` packs the working tree and this campaign adds two RLOpt knobs, so
a control from another code state is not a control.

## What is already known going in

Read from real checkpoints (`diffntp_chunk_50b_seed0`, `diffntp_pair_*`, seed 0
— same optimizer contract, not this campaign's arms):

| cumulative frames | actor lr | critic lr | sigma (mean) | sigma (per-dim min) |
|---:|---:|---:|---:|---:|
| 2.0B | 5.85e-05 | 1.0e-03 | 0.256 | 0.167 |
| 2.5B | 5.85e-05 | 1.0e-03 | 0.226 | 0.142 |
| 4.0B | 3.90e-05 | 1.0e-03 | 0.180 | 0.108 |
| 7.0B | 2.60e-05 | 1.0e-03 | 0.147 | 0.0859 |

Every actor value is exactly `1e-3 / 1.5^k`. The KL bang-bang rule steps down
about once per frame doubling, driven by the sigma anneal (KL scales as
`(delta mean / sigma)^2`). The critic never moves. At 2.5B the actor sits
`1.5^7` ~ 17x below `optim.max_lr`, so the controller has large headroom to
answer a lower update density.

## Arms

| arm | change vs `ctrl` | minibatch split | steps/iteration |
|---|---|---|---:|
| `ctrl` | none | 368,640 + 122,880 | 10 |
| `mb_half_e5` | `mini_batch_size=245760` | 2 x 245,760 | 10 |
| `mb_half_e3` | `mini_batch_size=245760`, `loss.epochs=3` | 2 x 245,760 | 6 |
| `mb_full_e5` | `mini_batch_size=491520` | 1 x 491,520 | 5 |
| `mb_full_e3` | `mini_batch_size=491520`, `loss.epochs=3` | 1 x 491,520 | 3 |
| `critic_lin` | `ipmd.critic_lr_schedule=linear`, `critic_lr_final=1.0e-5` | as `ctrl` | 10 |
| `wd_1e2` | `optim.weight_decay=1.0e-2` | as `ctrl` | 10 |
| `wd_1e4` | `optim.weight_decay=1.0e-4` | as `ctrl` | 10 |
| `wd_1e1` | `optim.weight_decay=1.0e-1` | as `ctrl` | 10 |
| `ent_only` | `ppo.entropy_coeff=0.01` | as `ctrl` | 10 |
| `floor_late` | `clip_log_std`, `log_std_min=log(0.10)`, `log_std_max=2.0` | as `ctrl` | 10 |
| `ent_sonic` | `entropy_coeff=0.01` + SONIC init log(0.05), clamp `[log 0.001, log 0.5]` | as `ctrl` | 10 |

Collected batch is 491,520 frames per iteration in every arm.

### A. Geometry (`mb_*`)

The production `minibatch = 3/4 x batch` is a non-divisor. `SamplerWithoutReplacement`
runs with `drop_last=False`, so an epoch yields 368,640 then 122,880: half the
optimizer steps see a 3x smaller batch at the same learning rate, and each
contributes equally to the per-iteration KL mean that drives the actor LR.
`mb_half` removes that asymmetry at identical update count and cuts peak
activation memory by 33%. `mb_full` raises it by 33% over anything measured on
the H200 and may OOM; that failure lands in iteration one and is cheap.

**Read the epochs arms on time-to-quality, not on frames.** Cutting epochs
lowers per-iteration policy drift, the KL controller answers by raising the
actor LR, and the measured headroom above is large enough to absorb it. What
an epochs cut actually buys is learn-phase compute (-40% at 3 epochs), and
collection was the bottleneck as of 2026-08-19 — so record fps and iteration
wall-clock, not only the frame curve.

### B. Critic learning rate (`critic_lin`)

The critic group is built `adaptive_lr=False` and never moves for the whole
run. Nobody chose the resulting 20-40x actor/critic ratio; it is what the
adaptive rule leaves behind. New RLOpt knob, linear 1e-3 -> 1e-5 over the 5B
budget, driven by `metadata.frames_processed` (cumulative, seeded with the
resume offset) so a chained segment continues the decay instead of restarting
at 1e-3.

### C. Weight decay (`wd_*`)

`optim.weight_decay` is 0.0 today, so AdamW is Adam. A global decay would also
pull `log_std` toward 0, i.e. sigma toward 1.0 rad, against the anneal that is
currently doing useful work. RLOpt now puts `log_std` in its own
`actor_log_std` parameter group at `weight_decay=0`, so these arms decay the
networks only. 1e-2 is the PyTorch AdamW default, 1e-4 the lighter common
setting, 1e-1 the LLM-scale setting that bounds the upper end of the axis. On
6-layer 2048-wide networks with no warmup, `wd_1e1` is the arm most likely to
simply lose; it is here to bound the axis, not because it is expected to win.

### D. Exploration (`ent_only`, `floor_late`, `ent_sonic`)

**SONIC's log_std contract is already measured and dead on this exact base.**
`smooth-ablation-5b/sigma` ran it verbatim (`log_std_init=log(0.05)`, clamped
`[log 0.001, log 0.5]`): episode length pinned at 15-22 from iteration one, SR
0.0151, aborted at 3.85B (job 5597007). That campaign's verdict was to test an
anneal or a late floor, not a clamp at birth. This campaign does not repeat the
init.

- `ent_only` — the entropy term alone, at SONIC's weight, on our init-1.0
  unbounded Gaussian. The 2026-08-02 ablation that retired the bonus was scored
  at ~100M frames, before sigma had annealed at all.
- `floor_late` — our init unchanged, a floor at sigma 0.10, cap left at
  `exp(2.0)`. The floor is chosen to bite: per-dim minimum sigma was 0.108 at
  4B and 0.0859 at 7B. A floor at SONIC's `log(0.001)` could never bind and
  would be a null by construction.
- `ent_sonic` — the one deliberate repeat: SONIC's exploration block entire
  (init log(0.05), clamp `[log 0.001, log 0.5]`) **plus** the entropy bonus
  `sigma` did not carry. Four fields against `ctrl`, so it attributes nothing
  on its own. Its comparison point is `smooth-ablation-5b/sigma`, whose only
  difference is the missing bonus, and the single question it answers is
  whether the bonus rescues the contract that died without it. The init sits
  strictly inside the clamp, so the parameter keeps its gradient.

`log_std` is a bare `nn.Parameter` and `torch.clamp` passes zero gradient
outside its range (`gaussian_policy.py:64`), so a `log_std_max` below the init
freezes the parameter for the entire run. Any cap arm must keep the init
strictly inside `[min, max]`; that is why `floor_late` leaves the cap at 2.0.

## RLOpt changes this campaign depends on

Both in `RLOpt/rlopt/agent/ipmd/ipmd.py`, tests in
`RLOpt/tests/test_ipmd_components.py` (already named in `pixi.toml`):

1. `ipmd.critic_lr_schedule` (`constant` | `linear` | `cosine`) and
   `ipmd.critic_lr_final`, applied once per iteration from cumulative frames.
   Config-only decay was impossible: `optim.scheduler="adaptive"` excludes
   every torch scheduler (`base_class.py:836`), and a torch scheduler steps all
   parameter groups, which would overwrite the actor's KL rule.
2. `log_std` split into an `actor_log_std` group at `weight_decay=0`. The group
   stays adaptive, so the KL rule still scales it with the rest of the actor,
   and it logs as `train/actor_log_std_lr`.

## Run

```bash
# validate offline, no ssh
pixi run python -m imitation_experiments.pipeline.cluster plan \
  --campaign experiments/campaigns/2026-08-30-optimizer-ablation-5b/campaign.yaml \
  --arm ctrl --seed 0 --skip-preflight

./experiments/campaigns/2026-08-30-optimizer-ablation-5b/submit_all.sh
./experiments/campaigns/2026-08-30-optimizer-ablation-5b/mirror.sh
./experiments/campaigns/2026-08-30-optimizer-ablation-5b/eval.sh
```

Commit before submitting if the run must be reproducible from a SHA: `submit`
packs the working tree and records `drift=true` when it is dirty, and this
campaign needs the RLOpt submodule pointer moved.

## Scoring

`bones_testbed4096_v1`, clean and robust rows, canonical 3-number row plus the
2026-08-29 smoothness metrics. Carry the `sonic_v1_1` row for that board in
every table and never mix boards in one table. Also report fps and wall-clock
per arm — for the geometry arms that is the deciding axis, not a footnote.

## Open before citing anything

- Single seed. Checkpoint variance has measured larger than evaluation noise on
  this base, so report the final checkpoint and its neighbour.
- `mb_full_*` may OOM. If it does, the cell is unanswered, not negative.
- `ent_sonic` moves four fields at once. It can only ever be read against
  `smooth-ablation-5b/sigma`, never against `ctrl`, and a death there repeats
  a known result rather than adding one.
- `train/actor_lr`, `train/critic_lr` and `train/kl_approx` must be read for
  every geometry arm. Without them an epochs result cannot be attributed
  between the update-density change and the controller's answer to it.
