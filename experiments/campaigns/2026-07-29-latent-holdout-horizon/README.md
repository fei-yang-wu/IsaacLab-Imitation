# Latent hold-out horizon ablation (2026-07-29)

**Question.** With the latent representation held *exactly* fixed, how much of
the DiffSR interface's benefit comes from the latent space itself versus from
the fact that one latent is **held for ten control steps**?

This is the GR00T-style "predict H, execute k" axis applied to our command
interface. The encoder always summarizes the same 10-step future window; only
the number of 50 Hz control steps consumed before a fresh latent is published
moves.

| Arm | Hold | Latents per 200 ms | Planner output @ 5 Hz | Status |
| --- | ---: | ---: | --- | --- |
| control | 10 | 1 | 256-d + phase | curve only — see "Control" below |
| `holdout5` | 5 | 2 | chunk of 2 latents | this campaign |
| `holdout1` | 1 | 10 | chunk of 10 latents | this campaign |

Because the encoder is shared and frozen, this ablates **command interfacing**
with **latent learning held constant** — the complement of the 2026-07-22
latent-learning ablation, which moved the bottleneck at fixed hold=10.

## What makes it single-variable

Every arm is frozen against one file:

```text
/data/pretrain_store/lafan1_strict_h10_z256_5b_seed0_20260721_jointfix_nocur_e12288_r12_nj320_nc40/checkpoints/latest.pt
```

`hl_skill_horizon_steps=10` is checkpoint-bound (`hl_skill_diffsr.py:579-584`,
`:1878-1881`), so the encoder window cannot drift across arms — a mismatch is a
hard load error, not a silent difference.

The launcher's emitted command was token-diffed against the control's recorded
`command.txt`. Of 34 Hydra overrides, 31 are byte-identical and exactly three
move:

```text
agent.ipmd.latent_steps_min      10 -> 5 / 1
agent.ipmd.latent_steps_max      10 -> 5 / 1
agent.ipmd.latent_learning.code_period  10 -> 5 / 1
```

### Why `code_period` must move with the hold

`code_period` feeds `phase_period` (`RLOpt/rlopt/agent/ipmd/ipmd.py:1248,1315`),
and the sampler computes

```text
phase = (phase_period - latent_steps) / phase_period      # hl_skill_diffsr.py:1078
```

Setting `code_period = hold` preserves the control's semantics exactly — "fraction
of my current command's hold elapsed", sweeping `0 -> (hold-1)/hold`. Leaving it
at 10 while `hold=5` would emit `0.5..0.9` instead and silently desynchronize the
clock, which would confound the ablation.

Command width stays **258** (256 code + 2 phase) for every arm, so the
observation contract never moves. At `hold=1` the phase is constant 0 — an
honest degeneracy of per-step renewal, not a width change.

## Expectation setting: hold=1 is a declared-risk arm

The 2026-07-22 controlled isolation
([`wiki/ablation-experiment-plan.md:331-347`](../../../wiki/ablation-experiment-plan.md))
already ran hold=1 against a frozen h10 encoder and it **collapsed**:

| Arm | 10M frames | 30M frames | r_step |
| --- | ---: | ---: | ---: |
| hold=10, phase=sin_cos | 10.95 | **46.38** | +0.028 |
| hold=1, phase=none | 2.72 | **2.76** | -0.039 |

That repro also switched `phase-mode` to `none`, so it confounded hold with
command width (258 -> 256). This campaign removes that confound by keeping
`sin_cos` everywhere. A collapse here is a **legitimate, citable negative
result** — it would say the latent interface's value is inseparable from the
held chunk, and that a GR00T-style 10-latent chunk is not deployable on a
frozen offline-autoencoder code. Do not tune it away.

## Control

User-decided 2026-07-29: **curve-only comparison**, no fresh hold=10 arm.

The control is job `5525664`
(`lafan1_strict_h10_z256_5b_seed0_20260721_jointfix_nocur_e12288_r12_nj320_nc40`,
W&B project `g1-lafan1-strict`, group `scaled-e12288-5b-resumable-jointfix`),
which reached ~4.56B/5B frames at ep_len 413.8.

**Its low-level policy checkpoints are gone.** Only `params/` and `command.txt`
survived under `/data/ckpt_store/<run_tag>/rlopt_train/2026-07-22_14-41-38/`;
the ICE TIMEOUT SIGKILL wiped the rest. The encoder survived on the `/data`
bind, which is what makes this campaign possible at all. Consequence: there is
**no hold=10 policy available for oracle evaluation or a planner row.** Any
downstream evaluation comparing all three holds needs a hold=10 retrain first.

## Surface

`Isaac-Imitation-G1-Latent-Strict-v0` — **not** the bare `-Latent-v0` id.

On 2026-07-27 the ids were repointed
(`config/g1/__init__.py:122-133`): `Isaac-Imitation-G1-Latent-v0` now resolves
to the Stable/SONIC reset-sampling surface. The control, its encoder, Study B
and Study C all ran on the plain strict surface, now reachable only under the
explicit `-Strict-v0` id. Using the bare id would confound hold with the
reset-sampling change.

## Segment policy

User-decided 2026-07-29: **one 16h segment per arm, no resume chain.** Whatever
frames the segment reaches is the result.

Sizing targets a clean exit just under the wall (~4.26B frames at the measured
~76k fps for latent arms at this geometry) rather than relying on the wall,
because ICE TIMEOUT is a hard SIGKILL — the final save never runs. `save_interval`
is 100M frames so even an overrun keeps a recent checkpoint. Checkpoints are
written to `/data/holdout_store/<run_tag>/rlopt_train`, never the per-submission
workspace.

Note both arms stop near ~4.26B against the control's 4.56B. The two new arms
are exactly matched to each other; the control is not frame-matched to them.

## Running

```bash
# Preview (default). Contacts nothing.
./experiments/campaigns/2026-07-29-latent-holdout-horizon/latent_holdout/submit_latent_holdout_horizon_ice.sh

# Submit both arms. Verifies the manifest sha, the 40-NPZ count, and the shared
# encoder's presence + sha256 before anything is queued.
DRY_RUN=0 ./experiments/campaigns/2026-07-29-latent-holdout-horizon/latent_holdout/submit_latent_holdout_horizon_ice.sh

# One arm only.
DRY_RUN=0 ARMS=5 ./experiments/campaigns/2026-07-29-latent-holdout-horizon/latent_holdout/submit_latent_holdout_horizon_ice.sh
```

## Reading the result

Primary signal is low-level trainability against the control curve: episode
length and per-step reward versus frames, all three holds on one plot. Episode
cap is 500 control steps.

Report bandwidth alongside performance — it is the axis being bought:

| Hold | Latents / 200 ms | Floats / 200 ms | vs explicit packet (670) |
| ---: | ---: | ---: | --- |
| 10 | 1 | 258 | 0.39x |
| 5 | 2 | 516 | 0.77x |
| 1 | 10 | 2580 | 3.85x |

hold=1 costs **more** bandwidth than the explicit full-body packet it is meant
to beat. If it also trains worse, that is the whole argument for the held chunk
in one line.

## Next phase (not in this campaign)

Planner-side GR00T-style chunk prediction — one planner forward at 5 Hz emitting
`10/hold` latents, consumed slot-by-slot with per-environment renewal. That work
is gated on an arm here training well enough to be worth a planner row, and on a
hold=10 policy existing again for the comparison.
