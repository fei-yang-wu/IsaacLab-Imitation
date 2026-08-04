# 2026-08-03 — Persistent-violation termination window (LAFAN1, 1B)

Does giving the strict tracking terminations a **time window** help? A term must
now be violated for `N` consecutive control steps before it ends the episode, so
a single contact spike, retargeting glitch, or push transient no longer destroys
the episode and the policy gets a chance to recover from it.

Thresholds are untouched. The strict 0.15 m / 0.2 bar stays exactly where it is;
only the episode boundary moves. That is the point: threshold relaxation buys
transient tolerance by lowering the bar the policy must return to, and a window
buys it without lowering the bar.

Launcher: `submit_termination_window_1b_ice.sh` (DRY_RUN=1 by default).

## The arms

| arm | window | at 50 Hz | job |
|---|---|---|---|
| `window3` | 3 consecutive steps | 60 ms | see `cluster_submission.json` |
| `window10` | 10 consecutive steps | 200 ms | see `cluster_submission.json` |
| *control* | instantaneous | — | `lafan1_v2_mjwarp_aligned_5b_seed0_e12288_r24` @ 1B |

**The control is free and must not be re-submitted.** The in-flight
`mjwarp-aligned-5b` LAFAN1 run is this exact configuration with no window, and
it passed 1B frames on 2026-08-03. Everything else is held to it: task
`Isaac-Imitation-G1-v2`, agent `rlopt_ipmd_tuned_cfg_entry_point`, 12288 × 24,
minibatch 18432, `njmax` 288 / `nconmax` 200, encoder
`lafan1_v2_det_sr_h10_z256_seed0`, corrected-LAFAN1 manifest `d972c37c…`, seed 0,
and the tuned env half including the 5M→30M termination curriculum.

## How to read the result

**MPJPE only.** A window inflates episode length, return, and every per-minute
rate *mechanically*, in exactly the way the 2026-08-02 campaign documented for
`b5_term_curriculum`. Measured locally at 256 envs:

| window | ep_len | r_step |
|---|---|---|
| 1 | 5.73 | 0.0720 |
| 25 | 30.95 | 0.0206 |

5.4× the episode length and a third of the per-step reward — no better policy,
just a later boundary. MPJPE is per-frame and length-independent, which is why
it is the check.

**Evaluate under the strict instantaneous protocol.** The window is a *training*
device. Every recorded oracle-qualification and M3 survival number is stated
against the instantaneous protocol, so a windowed checkpoint has to be scored
without the window or the comparison is meaningless. Use the same
`scripts/audit/sim2sim_backend_eval.py` strict + full-horizon pair the 5B monitor
uses.

## There is no upstream number to copy

Checked against `NVlabs/GR00T-WholeBodyControl` at `main` = `aa263a8`
(2026-07-31), the whole repository, all file types. The termination code and
configs are **byte-identical** to the `bc38f6d` release commit (2026-04-14), so
this has not moved in 3.5 months of active development.

SONIC ships the counter shape — `_CummErrorMixin` plus `CummBodyPosError` /
`CummBodyOriError` and their `_Local` variants — but **no released
configuration enables it**:

- All three release experiments (`sonic_release.yaml`, `sonic_bones_seed.yaml`,
  `sonic_h2.yaml`) select `tracking/base_adaptive_strict_ori_foot_xyz`, and the
  env default selects `tracking/base`. Both compose only instantaneous terms.
- Every term those compositions bind points `func:` at a plain **function**
  (`exceeded_anchor_height`, `exceeded_anchor_ori`, `exceeded_body_height`,
  `exceeded_body_pos`) — stateless, so no counter exists to run.
- `min_steps` appears in exactly one file in the entire repository,
  `gear_sonic/envs/manager_env/mdp/terminations.py`, and in **zero** config
  files — and that is the only `terminations*.py` in the repo, so there is no
  second implementation in `decoupled_wbc`, `gear_sonic_deploy`, or
  `motionbricks`. `_CummErrorMixin.__init__` reads it with
  `cfg.params.get("min_steps")` and no fallback, so enabling one of these terms
  without supplying a value would evaluate `self._cum_steps >= None` and raise.
- `TerminationsCfg` declares `cumm_body_pos_error` and its three siblings as
  `= None` slots that nothing ever fills.

So `_DEFAULT_WINDOW_MIN_STEPS = 3` is our choice, not an inherited constant.
Two further differences mean a SONIC value would not have transferred anyway:
their counter keys on a maintained scalar body error, ours wraps our existing
strict predicates; and their transient tolerance lives in the *threshold*
(`threshold_adaptive` with a `down_threshold` / `root_height_threshold`), not in
time.

Because there is no number to inherit, pick the window from measurement: the
probe (`--termination_window_probe`, or `G1SonicTerminationWindowProbeCfg`)
reports `Termination_Window/<term>/recovered_below_{2,3,5,10}_frac` — the
fraction of violation onsets a window of length `k` would have survived. Run it
on an existing checkpoint rather than as its own training job.

## Probe result (2026-08-03): the default window is an order of magnitude too short

Run on the instantaneous control's own 1B checkpoint
(`model_step_1000046592.pt`, sha `632679ea…`), 256 envs × 60 iterations =
368,640 frames, LR pinned to 1e-12 so the policy does not drift, curriculum
**off** so thresholds are the strict values from step one:

| term | onsets | <2 | <3 | <5 | <10 | mean recovered run | censored |
|---|---:|---:|---:|---:|---:|---:|---:|
| `foot_pos_xyz` | 1057 | 0.065 | 0.134 | 0.275 | 0.497 | 10.8 | 202 |
| `ee_body_pos` | 533 | 0.058 | 0.113 | 0.265 | 0.469 | 9.5 | 156 |
| `anchor_ori` | 429 | 0.075 | 0.131 | 0.198 | 0.296 | 13.4 | 189 |
| `anchor_pos` | 240 | 0.029 | 0.067 | 0.146 | 0.300 | 10.2 | 130 |

**A window of 3 converts only 6.7–13.4% of onsets into recoveries.** A window of
10 converts 29.6–49.7%, and a run that resolves at all takes 9.5–13.4 steps on
average. The transients this mechanism exists to absorb are roughly 10 control
steps (200 ms) long, not 3.

So `_DEFAULT_WINDOW_MIN_STEPS = 3` is under-powered, and the `window3` arm
should be expected to come back neutral. That is still worth having — it is the
shipped default and now has a measured reason for being wrong — but the
informative part of the axis is 10 and above.

`censored` is the count of runs still unresolved when the episode ended. Those
are genuine failures rather than transients, and they are correctly in the
denominator, so these fractions are not inflated. Note `base_too_low` is `None`
in `G1SonicTerminationsCfg`, so a probe episode ends only on `time_out` or
`reference_finished` and a fallen robot keeps accumulating one long unresolved
run — which is exactly why the censored counts are large.

## Running it

```bash
# Plan only (default).
./experiments/campaigns/2026-08-03-termination-window/submit_termination_window_1b_ice.sh

# Submit both arms.
DRY_RUN=0 ./experiments/campaigns/2026-08-03-termination-window/submit_termination_window_1b_ice.sh

# A different window set.
DRY_RUN=0 WINDOWS="5" ./experiments/campaigns/2026-08-03-termination-window/submit_termination_window_1b_ice.sh
```

Each arm is 1B frames in a single segment (3391 iterations at 294,912
frames/iter) under a 5 h wall — ~2.3 h at the ~120k fps the aligned run
measures, so no TIMEOUT can land on it. Checkpoints go to
`/data/term_window/<run_tag>/rlopt_train` every 100M frames, because an ICE
TIMEOUT is a hard SIGKILL that runs no final save.

The launcher sets only the deltas and delegates the actual command to
`../2026-08-02-rlopt-hp-search/submit_tuned_5b_ice.sh`, so the task, agent entry
point, data gates, encoder, and Slurm wiring have one definition. Reconstructing
that command here is the copy-drift that shipped two 5B runs at the wrong
rollout on 2026-08-03.

## Verification before submission

- `pixi run -e isaaclab python -m pytest source/isaaclab_imitation/tests/test_termination_window.py` — 15 passed.
  Pins the counter semantics, `min_steps=1` ≡ instantaneous, thresholds
  inherited unchanged, and that the default v2 surface stays instantaneous.
- Live env, `--termination_window 1` vs `25`: ep_len 5.73 → 30.95, so the
  counter demonstrably drives episode boundaries.
- Live env, window 3 **with** the termination curriculum active: runs clean.
  This is the combination the cluster job uses and the one at risk, because
  `anneal_termination_threshold_by_frames` writes `term_cfg.params["threshold"]`
  in place while the window swaps `term.func`. It looks the term up by name and
  makes no assumption about `func`, and the windowed `__call__` reads every
  predicate parameter per call, so the anneal keeps working.
- The env is **not** run-to-run deterministic at fixed seed on Newton: three
  identical no-flag runs gave ep_len 6.03 / 5.45 / 5.94. `window=1` gave 5.73,
  inside that band. Equivalence is therefore pinned at the counter level by the
  unit test, not by env-level bit-identity, which is not testable here.
