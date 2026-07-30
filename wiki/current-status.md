# Project Live Status

Experiment navigation now starts at `experiments/README.md` and its exhaustive `SCRIPT_INVENTORY.md`. One-shot launchers named in the chronology below may have been pruned on 2026-07-23; `experiments/PRUNED_SCRIPTS.md` is the authoritative deletion and recovery catalog. A historical path is not a live submission instruction.

Last verified: 2026-07-29. The current working tree repoints
`Isaac-Imitation-G1-Latent-v0` to the Stable/SONIC recipe with legacy
LAFAN1 resets; the former strict surface remains available as
`Isaac-Imitation-G1-Latent-Strict-v0`. ICE job `5542378` already trained the
new Stable surface through the requested 500M comparison point and onward to
about 1B frames, so no duplicate 500M job was submitted. The exact 500M
Stable-versus-Strict inference diagnostic is recorded below.

This is the living memory for the active research project. Read it first when
returning to the project or starting a new agent session. It answers **where we
are now**. The detailed protocol and experiment history remain in the linked
phase documents.

Human-facing launcher navigation now starts at
[`experiments/README.md`](../experiments/README.md). It marks the
2026-07-22 latent-learning ablation as the primary current campaign, keeps the
BONES h10 scale screen as its supporting campaign, and reserves
`experiments/paper/` for the eventual stable release entrypoint. Dated
campaign folders index canonical scripts rather than copying their
implementation.

Update this page after a meaningful code decision, qualification result,
cluster submission, job failure, or paper result. Verify changing external
state such as Slurm jobs before treating a status below as current. Keep old
chronology in the phase-specific pages instead of allowing this page to grow
without bound.

## Stable LAFAN1 5B convergence run submitted (2026-07-29)

ICE job `5548933` is the matched-scale follow-up to the 500M diagnostic below.
It runs `Isaac-Imitation-G1-Latent-v0`
(`ImitationG1LatentStableEnvCfg`) on one H200 with 16,384 environments x 12
rollout steps, minibatch 24,576, seed 0, and 25,431 PPO iterations =
4,999,938,048 environment frames. It uses corrected LAFAN1 manifest SHA-256
`d972c37c...c945db8`, the existing read-only dataset cache, and the same frozen
h10 DiffSR encoder (`5c84ff72...264ea`) used in the 500M comparison.

Run name: `stable_lafan1_diffsr_det_h10_e16384_s12_5b_seed0_20260729`; W&B
project/group: `g1-sonic-env-latent-det-ice` /
`stable-e16384-s12-5b-seed0`, run `yuf0st77`. The persistent ICE run directory
is `logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/`
`2026-07-29_17-06-39_wandb-yuf0st77`. Checkpoints are requested every 100M frames and
the cluster persistent-project-log bind was confirmed in the startup log. The
submitted workspace archive SHA-256 is
`560a780d23e7e4c0a1e1ea0594776bbad010601c0c70bb1b93d062a277a52be6`.
The job resolved to `ImitationG1LatentStableEnvCfg` and the SONIC actor-input
contract, completed its first PPO update cleanly at about 92.9k FPS, and used
about 73.9 GB of H200 memory.
The first persistent checkpoint landed successfully at 100,073,472 frames:
`models/model_step_100073472.pt`, 25,036,449 bytes, SHA-256
`38059555818226b6b9cc3c74b306d10a7b925151838563e8725f5338c32d4f6e`.
At that point the training logger reported mean episode length 187.91, return
13.82, and 91.6k FPS.
The matched 500M checkpoint also landed successfully:
`models/model_step_500170752.pt`, 25,036,449 bytes, SHA-256
`5a6a03059187f4cc5d81e16a9540f96f8284e4122ec6502c60ff81930ddd5a43`.
At 500,170,752 frames the logger reported mean episode length 334.96, return
26.65, and 90.6k FPS; job `5548933` remained healthy and running.

After training, inspect the late checkpoint curve and evaluate the converged
checkpoint by model inference on the exact same protocol as the 500M result:
all 40 corrected motions, seed 0, 1,000 steps, deterministic tracking, and a
non-terminating full-horizon pass for unbiased MPJPE plus the secondary strict
termination/success pass and retained video. If the 15:59 ICE walltime catches
the last fraction of the nominal budget, resume only the missing frames from
the latest persistent checkpoint.

## LAFAN1 Stable-vs-Strict 500M inference diagnostic (2026-07-29)

ICE already retained both requested checkpoints, contrary to the earlier
checkpoint-loss generalization:

- Stable/SONIC recipe, `Isaac-Imitation-G1-Latent-v0`: completed job `5542378`,
  checkpoint `model_step_500072448.pt` from run
  `2026-07-27_14-07-04_wandb-3xz1v8k1`.
- Former Strict recipe, `Isaac-Imitation-G1-Latent-Strict-v0`: checkpoint
  `model_step_500170752.pt` from run
  `2026-07-22_19-42-12_wandb-gha4nlhl`.

Both commands reference the same h10 DiffSR skill checkpoint; its encoder SHA-256
is `5c84ff7261c5a3aca732e370ca39f889d68a5d39fb498fa9fde72c653eb264ea`.
The local copies and inference artifacts live under
`logs/downloaded_checkpoints/lafan1_stable_vs_strict_500m_20260729/` and
`logs/interface_baselines/lafan1_stable_vs_strict_500m_20260729/`.

The decisive matched inference pass used all 40 corrected LAFAN1 motions, one
environment per motion, seed 0, 1,000 steps, deterministic tracking, and all
early terminations disabled. Both rows therefore contain exactly 40,000
body-frame samples:

| Recipe at ~500M frames | Root-relative MPJPE | Root XYZ | Joint RMSE | EE position | Velocity | Acceleration | Action change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stable/SONIC | 111.17 mm | 0.899 m | 0.303 rad | 0.914 m | 0.654 m/s | 11.884 m/s2 | 1.692 |
| Strict | 129.84 mm | 0.911 m | 0.275 rad | 0.928 m | 0.576 m/s | 8.009 m/s2 | 1.085 |

Stable reduced MPJPE by 18.67 mm (`14.38%`) in this single deterministic pass,
but it increased joint error, temporal errors, and action change. Treat this as
a diagnostic trend, not a win: prior repeated inference showed roughly 12%
relative MPJPE variation in this error regime, and the retained checkpoints
also have different training geometry (Stable `4096 x 24`, Strict
`16384 x 12`). A same-geometry Strict legacy-reset run stopped near 300M and
cannot provide a matched 500M row.

The strict-termination pass is retained as a secondary diagnostic. It reported
33.32 mm / 0.19 success for Stable and 40.79 mm / 0.24 success for Strict, but
its unequal termination-truncated sample counts make those MPJPE values
unsuitable for the headline comparison. Videos were retained from the same
non-terminating full-horizon passes.

## Latent hold-out horizon ablation submitted (2026-07-29)

New campaign
[`experiments/campaigns/2026-07-29-latent-holdout-horizon/`](../experiments/campaigns/2026-07-29-latent-holdout-horizon/README.md).
It ablates the **command interface at a fixed latent space**: every arm is
frozen against one shared h10 DiffSR encoder, and only the number of 50 Hz
control steps a published latent is held for moves — hold in {5, 1} against the
hold=10 control. This is the GR00T-style "predict H, execute k" axis, and the
complement of the 2026-07-22 latent-learning ablation (which moved the
bottleneck at fixed hold=10).

**Live jobs (ice-gpu H100, submitted 2026-07-29 ~14:21):** `5548369` (hold=5)
and `5548370` (hold=1). One 16h segment each, no resume chain (user-decided);
28,883 iterations ~ 4.26B frames at the measured ~76k fps. Both booted clean on
`ImitationG1LatentStrictEnvCfg` with checkpoints written to
`/data/holdout_store/<run_tag>/rlopt_train`.

Three facts worth carrying forward:

1. **The ablation is provably single-variable.** The launcher's emitted command
   was token-diffed against the control's recorded `command.txt`: of 34 Hydra
   overrides, 31 are byte-identical and only `latent_steps_min`,
   `latent_steps_max`, and `latent_learning.code_period` move. The encoder
   window (`hl_skill_horizon_steps=10`) is checkpoint-bound
   (`hl_skill_diffsr.py:579-584`), so it cannot drift silently.
2. **`code_period` must move with the hold.** It feeds `phase_period`
   (`ipmd.py:1248,1315`) and the sampler computes
   `phase = (phase_period - latent_steps)/phase_period`
   (`hl_skill_diffsr.py:1078`). Holding it at 10 while hold=5 would emit
   0.5..0.9 instead of 0..0.8 and desynchronize the clock. Command width stays
   258 for every arm, which is the confound the 2026-07-22 hold isolation had
   (it switched `phase-mode` to `none`, changing 258 -> 256).
3. **The hold=10 control's policy checkpoints are gone.** Only `params/` and
   `command.txt` survived under `/data/ckpt_store/<control_run_tag>/`; the
   TIMEOUT SIGKILL wiped the rest. Its **encoder survived** on the `/data` bind,
   which is what makes this campaign possible. Consequence: the control is a
   **curve-only** comparison (W&B `g1-lafan1-strict`, group
   `scaled-e12288-5b-resumable-jointfix`, ~4.56B frames, ep_len 413.8), and
   there is no hold=10 policy for oracle evaluation or a planner row until one
   is retrained.

Expectation: the 2026-07-22 isolation showed hold=1 collapsing (ep_len 2.76 vs
46.38 at 30M). hold=1 is submitted as a declared-risk arm; a collapse is a
citable negative result, not a bug to tune away. Note hold=1 also costs 2580
floats per 200 ms against the explicit packet's 670 — more bandwidth than the
baseline it is meant to beat.

## Reconstruction-family arms resubmitted with persistent checkpoints (2026-07-29)

The Study A `vqvae` and `fsq_recon` arms (h10 held code, sin/cos phase,
`code_period=10`, 66-d command) were resubmitted because the original
2026-07-22 H200 runs lost every checkpoint to the ICE TIMEOUT wipe — the Study
A launcher had no `agent.logger.log_dir`, so checkpoints lived in the
per-submission workspace. Only W&B curves survive from those runs, and no
run directory for any of the twelve ablation arms remains on ICE scratch.

Fix: `submit_lafan1_reconstruction_ablation_ice.sh` now writes
`agent.logger.log_dir=/data/latent_ablation_store/<exp_name>/rlopt_train`
for every arm, following the persistent-bind convention. New jobs also override
`SAVE_INTERVAL` to 100M (the launcher's 25M default is what filled scratch and
forced the 07-26 thinning).

**Live jobs (submitted 2026-07-29 ~15:10):** `5548489` vqvae (ice-gpu H200,
running) and `5548504` fsq_recon (coe-gpu H200; first attempt `5548500` was
pinned to the single unavailable ice-gpu H200 node and was cancelled/resubmitted
on coe-gpu). Approved H200 profile, 16,384 x 12, 23,071 iterations ~ 4.54B
frames in one segment, seed 0.

## Grouped-VQ capacity ablation prepared (2026-07-26)

New campaign
[`experiments/campaigns/2026-07-26-groupvq-capacity-ablation/`](../experiments/campaigns/2026-07-26-groupvq-capacity-ablation/),
documented as "Study C" in
[`wiki/latent-learning-ablation-plan.md`](latent-learning-ablation-plan.md).
It fixes the DiffSR spectral bottleneck at the grouped product codebook
(`gumbel_multicat`, hard straight-through) and sweeps only its two capacity
axes around the `G=64, C=128` anchor that previously tracked the continuous
deterministic latent: `G` in {16, 32, 64, 128} at `C=128`, and `C` in
{16, 64, 128, 512} at `G=64` — seven arms, seed 0, corrected LAFAN1, approved
H200 geometry, 5B cap.

Status: CPU pre-flight passed for all seven grid points (build, quantize,
checkpoint round-trip), and all seven passed the local 10M wiring gate in
`logs/groupvq_ablation/local_10m_gate_20260726/`.

**Live jobs (coe-gpu H100, submitted 2026-07-26 ~19:15):** `5540442`
g16_c128, `5540443` g32_c128, `5540445` g64_c128, `5540446` g128_c128,
`5540448` g64_c16, `5540449` g64_c64, `5540450` g64_c512. All seven confirmed
RUNNING with clean logs and falling DiffSR pretrain loss. Each needs one
continuation segment: 12,288 x 12 at ~80k FPS covers about 4.0B of the 5B cap
in one 14h segment.

Getting there took three submission rounds and exposed three defects, two
pre-existing:

1. `submit_hl_skill_pipeline_pace_2b.sh` computed `REPO_ROOT` from a fixed
   `..`, which broke when the 07-23 reorg moved it into
   `experiments/campaigns/<dated>/`. Now marker-based. This had also silently
   broken the 2026-07-22 campaign's launchers.
2. **Concurrent dataset-cache rebuild.** Seven arms sharing
   `/data/lafan1_corrected_8e95d557/g1_hl_diffsr` each ran with
   `env.refresh_zarr_dataset=true`; they rebuilt it underneath each other,
   four arms died on `FileNotFoundError`, and the cache was truncated to
   56 KB. Rebuilt to the full 1.2 GB / 40 motions by one-time job `5540413`
   from the intact NPZ source and the hash-matching manifest. Arms now always
   pass `refresh=false` and the cache is owned by
   `groupvq_ablation/build_lafan1_cache_ice.sh`. **The 07-22 launcher has the
   same pattern across twelve overlapping arms and is the likely explanation
   for its checkpoint-less `continuous_ae` arm.**
3. **`atl1-1-03-010-15-0` has a dead GPU** (`No devices were found`,
   `no CUDA-capable device is detected`) while Slurm still advertises it as
   `mix` with no drain reason, so it keeps accepting and killing jobs. Both
   launchers exclude it via `CLUSTER_SLURM_EXCLUDE`; worth a PACE ticket.

An initial six-arm submission to ice-gpu H200 (`5539991`-`5539999`) sat
PENDING for two hours without starting and was cancelled: every H200 GPU on
ice-gpu and coe-gpu was allocated (one free cluster-wide) and the sixth ice-gpu
H200 node has been admin-drained since 07-24. coe-gpu had about 40 free H100s,
so the grid was moved there at 12,288 envs x 12 steps, minibatch 18,432 --
the 07-22 `h100_e12288_lr1e3` geometry, because an 80 GB H100 cannot hold the
16,384-environment point.

Consequence: **all seven arms including `g64_c128` are re-run on H100.** The
finished 4.53B H200 `lafan1_diffsr_gumbel_multicat_b448_h10_z256_seed0` run
differs in env count and minibatch, so it is not a row of this grid and the
07-22 study remains a separate table.

Save interval for these arms is 100M frames, not the 25M of the 07-22 study,
because ICE scratch had roughly 20-40 GB of headroom. The 07-22 runs were
thinned to the same 100M granularity to make room (see below), so
plateau-checkpoint selection now resolves to 100M across both studies.

Two corrections to earlier entries on this page:

- **The 2026-07-22 latent-learning H200 ablation did run.** Eleven of its
  twelve arms reached 4,525,129,728 frames on ICE (the twelfth,
  `continuous_ae`, has no checkpoints in its run dir and needs a separate
  look). Any statement that no H200 jobs from that study were submitted is
  stale.
- **07-22 checkpoints were thinned on 2026-07-26.** ICE scratch hit 300/300 GB
  and blocked submission. With user approval, intermediate checkpoints were
  deleted from the eleven finished runs, keeping every ~100M-frame checkpoint
  plus each run's final one (181 -> 84 per run, 1,062 files, ~42 GB). No run
  directory, metric CSV, or final checkpoint was removed.

Caveat to carry into any table: bandwidth, per-group code dim, and encoder head
width all move together across this grid (encoder parameters span 2.4M-18.8M),
and grouped code usage/perplexity are currently pooled over groups rather than
reported per group.

## Data-loss incident: Slurm TIMEOUT destroys node-local output (2026-07-22)

**All three 2026-07-21 night segments produced zero retained checkpoints.**
Jobs `5525663` (BONES-SEED-91 h10), `5525664` (LAFAN1 h10) and `5525687`
(LAFAN1 history) each ran the full 15:59 walltime at ~80k fps to ~4.5B frames,
then ended in Slurm `TIMEOUT`. TIMEOUT is a hard SIGKILL: it kills the job step
before `run_singularity.sh`'s `sync_project_logs_back` copies the container's
node-local `$TMPDIR` workspace to shared storage, and the epilog then wipes
`/tmp`. A rescue job pinned to `atl1-1-03-013-8-0` and `atl1-1-03-013-13-0`
confirmed nothing survived. ~48 GPU-hours lost, encoders included. By contrast
the separate EE-chunk run `isaaclab_20260721_222745` finished its 5B *normally*
in 15.4h and synced back fine -- normal exit was always safe, which is why this
stayed hidden.

Root cause of the missing safety net: `scripts/rlopt/train_impl.py`
unconditionally reassigned `agent_cfg.logger.log_dir` to
`logs/rlopt/<algo>/<task>/<timestamp>`, so the `--train-override
agent.logger.log_dir=...` the launchers had been passing was **silently
discarded and had never once taken effect** (verified: no run-scoped checkpoint
directory has ever existed on ICE).

The first recovery attempted on 2026-07-22 routed the two 5B launchers through
`/data/ckpt_store` and `/data/pretrain_store`. That was durable, but it was an
unnecessarily launcher-specific layout rather than a repair of the cluster
runtime's normal logging contract. Jobs `5526545`, `5526549`, and `5526551`
used that workaround and were intentionally cancelled after 2-4 minutes.

Final cluster-wide fix, verified 2026-07-22:

- Every Apptainer/Singularity profile now binds a persistent shared project
  log root directly at `/workspace/isaaclab/project/logs`. Normal submissions
  derive it from the stable (pre-timestamp) `CLUSTER_ISAACLAB_DIR`; direct
  `run_singularity.sh` invocations fall back to their persistent workspace's
  `logs` directory. Checkpoint durability no longer depends on
  `sync_project_logs_back` or any shell exit handler.
- RLOpt therefore keeps its original layout with no log-directory override:
  `logs/rlopt/<algo>/<task>/<timestamp>/models/model_step_<N>.pt`.
- ICE job `5526584` tested the latest `Isaac-Imitation-G1-Latent-v0` surface
  with 64 environments, two rollout iterations, and `agent.save_interval=1`.
  While the job was still running, the central tree received valid 29,133,293
  byte `model_step_128.pt` and `model_step_256.pt` archives under
  `logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-22_14-49-29/models/`.
  Both archives passed ZIP integrity checks, and the first loaded successfully
  through PyTorch with policy, value, reward-estimator, optimizer, and skill
  sampler state present; the job completed in 5:16 with exit code 0.

Supporting fixes retained:

- `train_impl.py` now honors an explicit `agent.logger.log_dir` override as the
  log root; the config default is the literal `"logs"`, so runs that do not
  override keep byte-identical behavior. Verified end-to-end with a local
  1-iteration PhysX smoke run writing into an override directory.
- Each segment's iteration count is now capped to finish *before* the wall
  (`SEGMENT_TRAIN_SECONDS` 14.5h x conservative `ASSUMED_FPS` 70k = 24,780
  iterations) so jobs exit cleanly and get a final save instead of being
  SIGKILLed.

No replacement 5B low-level job is active after those cancellations. A future
resubmission should use the ordinary central log tree through the general bind;
the `/data/ckpt_store` launcher workaround is not the desired final layout.

## BONES-SEED h10 GPU/LR wall-clock ablation (2026-07-22)

A short wall-clock convergence screen is active on ICE using the post-fix
91-motion SONIC-filtered BONES-SEED manifest and one shared 50k-update h10
encoder. Encoder job `5526697` runs first on H100. Five 500M-frame controller
jobs depend on its successful completion, so every arm consumes the exact same
encoder checkpoint from the centralized project log tree:

| Job | GPU | Envs x rollout | Actor LR / adaptive cap |
| --- | --- | --- | --- |
| `5526698` | H100 | 12288 x 12 | `1e-3` |
| `5526756` | H200 | 12288 x 12 | `1e-3` |
| `5526757` | H200 | 16384 x 12 | `1e-3` |
| `5526703` | H100 | 12288 x 12 | `6e-4` |
| `5526704` | H100 | 12288 x 12 | `3e-4` |

The critic remains at `1e-3`; minibatch size is always rollout batch / 8, and
all other PPO/environment settings are fixed. Checkpoints are written every
25M frames. Compare arms by sustained wall-clock time to matched episodic
return and episodic-length levels, not final sample count alone. The W&B
project is `g1-bones-seed-h10-gpu-lr-ablation-ice`. Launcher:
`experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_bones_seed_h10_gpu_lr_ablation_ice.sh`.

PACE submission now accepts the same restricted
`CLUSTER_SLURM_DEPENDENCY=afterok:<job>[:<job>...]` contract as the general
Slurm wrapper; arbitrary dependency expressions remain rejected.

The original H200 jobs `5526700` and `5526702` started within the same second
and exposed a second concurrency bug: the standard RLOpt run directory used a
timestamp with only one-second resolution, so both wrote into
`2026-07-22_15-42-35`. They were cancelled at 16m48s and replaced by the jobs
listed above. RLOpt training now allocates the W&B run ID before logger setup,
exports it as `WANDB_RUN_ID`, and names W&B-backed run directories
`<timestamp>_wandb-<run-id>`. Non-W&B cluster runs use
`<timestamp>_slurm-<job-id>`, with a random local fallback. RLOpt's logging
manager recognizes both the legacy bare timestamp and the new suffixed form.
Replacement jobs created distinct centralized directories
`2026-07-22_16-00-03_wandb-37ozgk4i` and
`2026-07-22_16-00-28_wandb-r74i1tcr`.

After the initial wall-clock comparison, all five 500M-frame ablation arms
were cancelled at the user's request (`5526698`, `5526703`, `5526704`,
`5526756`, `5526757`). The H200 16384-env arm sustained about 90.4k FPS,
roughly 21% above the H100 12288-env arm, although this does not establish
better sample efficiency. Production job `5526830` is now running from
scratch on one H200 with 16384 envs x 12 steps, minibatch 24576, actor/critic
LR `1e-3`, the shared completed h10 encoder, checkpoints every 25M frames,
and 25431 iterations = 4,999,938,048 effective frames. Its ICE walltime is
15:59:00; at the measured screen throughput the estimated total including
initialization is about 15.5 hours, leaving only modest scheduler headroom.

## Protocol revision: no curriculum, h10 encoders, history ablation (2026-07-21 night)

The H100 h25 resubmissions (`5525266`/`5525267`) were cancelled ~40 min in
after the user traced a confusing early-metric kink to the termination
curriculum: while `G1SonicTerminationCurriculumCfg` anneals thresholds over
50M -> 300M frames, episode length/return dip as goalposts move. Decisions:

- **Curriculum removed from the default surface.**
  `ImitationG1LatentStrictEnvCfg` (behind `Isaac-Imitation-G1-Latent-v0`)
  now has `curriculum = None`; thresholds are strict from frame 0. The
  anneal remains available on the opt-in SONIC surface only. Caveat noted at
  the time: the 2026-07-19/20 investigation added the anneal precisely
  because strict-from-scratch spends the early budget on ~5-step episodes.
- **Encoder horizon moved from h25 to h10** (matches the 5 Hz planner
  publication interval: one latent per 10-step chunk at 50 Hz). Both 5B
  low-level jobs retrain their skill encoders at h10 with the full previous
  pretrain contract (50k updates, 0.9/0.1 split, groups/categories 64/128,
  gumbel_hard=true).
- **Active jobs:** `5525663` (BONES-SEED-91 h10, run tag
  `bones_seed_91_strict_h10_..._nocur_...`, ~77k fps) and `5525664`
  (corrected LAFAN1 h10, `lafan1_strict_h10_..._nocur_...`, ~83k fps), both
  5B, scaled config, Newton, H100.
- **History ablation:** new task `Isaac-Imitation-G1-Latent-History-v0` =
  strict surface + `G1SonicLatentObservationCfg` (10-step proprio
  histories, SONIC actor input set) paired with
  `G1ImitationLatentSonicRLOptIPMDConfig` on the local optimizer contract,
  so ONLY the observation/history contract differs from `Latent-v0` — a
  low-cost recurrent-policy stand-in. Running on corrected LAFAN1 as job
  `5525687` (run tag `lafan1_history_strict_h10_..._nocur_...`, W&B group
  `history10-h10-e12288-5b-jointfix-nocur`); compare against `5525664`.
  Its skill encoder is NOT retrained: it is the checkpoint `5525664`'s
  in-job pretrain produced, pulled off the compute node mid-run and staged
  to `pretrain_store/` for both run tags (sha256 `b3e23e0a...`, see the
  `PROVENANCE.txt` beside it), so the ablation pair shares tensor-identical
  encoders and differs only in policy-side history. (A first submission
  `5525682` that would have retrained its own encoder was cancelled during
  pretrain for exactly this reason.)
- **Planned next (not yet submittable):** planner training on top of the
  h10 low-level controllers once they finish and pass oracle audits — a
  language-conditioned planner for BONES-SEED (per the Phase-5 multigoal
  language workflow) and the standard no-language planner for LAFAN1. The
  h10 encoders trained here are the planner-side prerequisite; planner jobs
  must wait for qualified low-level checkpoints and the streamed-vanilla
  equivalence gates where applicable.

## Post-fix 5B resubmissions on Blackwell (2026-07-21 evening)

With the joint-order fix merged (PR #24, `900c66c`), both from-scratch 5B
low-level runs were resubmitted on `Isaac-Imitation-G1-Latent-v0` at the
scaled config (12288 envs x 12 steps, minibatch 18432, njmax=320/nconmax=40,
Newton) as ICE jobs `5525240` (BONES-SEED-91,
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh`) and `5525245`
(corrected LAFAN1, new `experiments/submit_lafan1_5b_resumable_ice.sh`).
The first submissions (`5525240` BONES, `5525245` LAFAN1) targeted the
Blackwell partition (`ice-bw-gpu`, 3x16 `rtx_pro_6000_blackwell`) as a
first Blackwell-stack test. **Result: the stack works on Blackwell**
(kernels, Newton solver init, 50k-update pretrain, and training start were
all clean) **but the cards are 48 GB (47.38 GiB visible), not 96 GB**, and
the 12288-env scaled config hit CUDA OOM at the first advantage pass in both
jobs. Both were resubmitted on `ice-gpu` H100 80 GB as jobs `5525266`
(BONES) and `5525267` (LAFAN1), reusing the completed Blackwell pretrains
(~35 min for 50k updates each) via the new pretrain store.

Two ICE plumbing facts learned in the process:

- Although `ice-bw-gpu` allows 18h, every ICE QoS
  (`coe-ice`/`coc-ice`/`pace-ice`) sets `MaxTRESMins gres/gpu=960`, capping
  any 1-GPU job at 16h regardless of partition — the initial 17:59:00
  submissions pended with `QOSMaxGRESMinutesPerJob` and were reduced in
  place via `scontrol update job`. Launchers default to 15:59:00.
- Every archive submission runs in its own `isaaclab_<timestamp>/` workspace
  dir and `run_singularity.sh` syncs job logs back into that dir only, so
  nothing accumulates under the stable `isaaclab/` root and a naive
  fixed-path resume scan never finds prior segments. The only host path all
  jobs share read-write is `CLUSTER_DATA_DIR` (bound at `/data`). Both 5B
  launchers therefore scan all `isaaclab*/logs/rlopt/ipmd/<TASK>/*/command.txt`
  (exp_name-filtered), keep cumulative-frame state in
  `<data>/resume_store/<RUN_TAG>/`, and stage the resume checkpoint and the
  skill-encoder pretrain into `/data`-visible stores that the next segment's
  container can read.

Pretrain was prolonged to the full previous contract: 50k skill-encoder
updates (the 5000 in the earlier launcher was a qualification-only value),
0.9/0.1 trajectory split (diffsr default), categorical groups/categories
64/128, and `--gumbel-hard` passed explicitly because the pipeline's argparse
default (False) silently overrides the diffsr-side default (True).

Both launchers keep checkpoints in the shared central location
(`logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/<timestamp>/`); resume
detection filters run dirs by `agent.logger.exp_name=<RUN_TAG>_oracle_low_level`
recorded in each run dir's `command.txt`, which restores per-run isolation --
required because these two jobs share one task id, and because the
invalidated pre-fix sanity checkpoints (jobs `5524387`/`5524390`) live in the
same tree.

Also removed: the `Isaac-Imitation-G1-Latent-Strict-v0` gym registration and
every repo reference to it. `Isaac-Imitation-G1-Latent-v0` is the single name
for the latest preferred latent surface; `Latent-Sonic-v0` and
`Latent-Legacy-v0` remain explicit opt-ins.

## Open Blocker: Newton joint-order leak (2026-07-21)

Cross-backend verification found that the expert command observations and the
action offset are resolved from the *live* articulation order instead of the
pinned canonical list. PhysX and Newton differ in 27 of 29 joint slots, and the
pinned list is the PhysX order, so both leaks are no-ops under PhysX and active
under Newton. Every Newton-trained checkpoint therefore encodes a
Newton-specific joint permutation, including the `reward_input` term that feeds
the IPMD reward and discriminator.

Confirmed in both directions on `L1_strict/model_step_992870400.pt`: removing
the mismatch on PhysX raises survival from 67/500 to 323/500 steps; injecting it
on Newton drops survival from 500/500 to 111/500. Joint tracking error is
~0.43-0.52 rad whenever mismatched and ~0.11-0.24 rad when matched.

**Fixed on 2026-07-21.** The command terms and the action offset are pinned,
the causal planner frame is pinned, and a latent double-scatter in
`batch_csv_to_npz.py` (live since 2026-07-16, no data affected) was removed.
The index contract now reports no leaks on either backend and the regression
test covers every command term.

**Existing Newton checkpoints are invalidated** and now fail on Newton too
(113/500 steps). `compare_policy_reference.py --emulate_joint_order_from` is a
diagnostic-only shim that restores them exactly; retraining the low-level
controllers is the real remedy.

Source reference NPZ/Zarr data is name-bound and **safe**. Policy-produced
artifacts are not: rollout NPZ state arrays, planner sample rows, and the
skill-encoder latent space are all Newton-permuted with no ordering metadata.

Two further bugs were found and fixed on 2026-07-21 while chasing the residual
gap:

- **Stale derived state after reset (both backends).** Both reset events called
  `asset.update(dt=0.0)`, which does not advance `_sim_timestamp`, so Isaac
  Lab's lazily cached body-frame buffers were never recomputed. `base_lin_vel`
  and `base_ang_vel` are policy observations, so the first observation after
  every reset came from the pre-reset state — stale under PhysX, zeros under
  Newton — throughout all training to date.
- **PhysX solver iterations.** The USD spawn copied the URDF importer's
  `articulation_props` and overrode the asset's requested 32/1 with a generic
  8/4. The override is removed; the asset now governs, verified on the live
  stage.

Neither closed the transfer gap, which is the point: with ordering matched,
Newton survives fully at 0.126 rad joint error while PhysX falls at 5.36 s with
0.242 rad. The gap is a genuine dynamics difference.

**Decision: if we randomize, we randomize for every experiment**, so the
protocol is re-frozen on the randomized event config rather than randomizing a
subset. This invalidates existing qualification artifacts, so sequence it with
the retraining already forced by the joint-order fix.

See [Sim2Sim Backend Verification](sim2sim-backend-verification.md) for the
audit tooling, evidence tables, and recorded-data status, and
[Sim2Sim Dynamics Gap and Randomization](sim2sim-dynamics-gap-and-randomization.md)
for the gap analysis and the randomization tiers.

## Research Question

We are testing whether a learned latent skill command is a better high-level
planner interface than the explicit action/state chunks used by current
humanoid VLA systems.

The main questions are:

1. Can a causal high-level planner command a frozen whole-body controller
   without future expert state leaking into its input?
2. Does the latent interface make the planner easier to learn or more
   data-efficient than an explicit full-body chunk?
3. Does the latent interface require a smaller planner to reach the same
   closed-loop performance?
4. Does language-conditioned planning work across diverse BONES-SEED motions?

## Frozen Main Comparison

The main paper comparison has exactly two planner rows:

| Interface | High-level output | Publication rate | Frozen low-level consumer |
| --- | --- | ---: | --- |
| DiffSR latent | 256-value latent code | 5 Hz | DiffSR latent tracker at 50 Hz |
| Explicit packet | Ten consecutive vanilla full-body commands, 670 values | 5 Hz | The same qualified vanilla tracker used by the direct ceiling, at 50 Hz |

The planner input is ten causal robot frames (`10 x 93`) plus an explicit task
input. Phase 5 adds the same 384-value MiniLM language embedding to both rows.
Future reference state is allowed only in oracle targets, labels, and metrics.
It is never a deployed planner input.

The direct vanilla tracker receiving a fresh expert command at 50 Hz is a
low-level ceiling, not a planner baseline. End-effector chunks and other
command styles are diagnostics or appendix work; do not start a combinatorial
command sweep.

The authoritative protocol is
[Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md).

## What Is Implemented and Verified

### Causal planner path

- Both main rows use the same ordered `10 x 93` achieved-robot history.
- The planner does not use `current_achieved_macro_transition_batch`, future
  reference state, reference rank, or reference cursor as deployed input.
- Language goals are supplied explicitly and checked against the selected
  named motion.
- Commands renew independently per environment, including after asynchronous
  resets.
- M3 disables tracking-error terminations but keeps `base_too_low`; a fall is
  defined identically for both interfaces.
- Evaluators retain success, survival, MPJPE, root, joint, end-effector,
  smoothness, velocity, acceleration, action-change, termination-cause, and
  planner-latency metrics.

### Explicit tracker equivalence

- The direct and streamed vanilla paths load the same strict frozen policy
  state and use the same ordered actor inputs.
- The streamed packet consumes slots 0 through 9 exactly once.
- The BONES-SEED certificate passed all packet phases, asynchronous renewal,
  and policy immutability. Maximum command and action differences were
  `3.02e-7` and `1.31e-6`.

### Planner families and scaling tools

Three continuous planner families are implemented with matched Transformer
parameters:

- flow matching;
- clean-target diffusion;
- deterministic chunk prediction.

The scaling reports keep demonstration-only and rollout-fine-tuned results
separate and record actual parameters, output bandwidth, and measured planner
latency. They answer both performance at the same size and the smallest tested
size that reaches a fixed performance target.

### Reproducibility gates

- Data, checkpoints, caches, language tables, workflow sources, and stage
  artifacts are hash-bound.
- Phase 4 and Phase 5 have guarded launchers, exact seed grids, stage records,
  strict aggregators, and no-overwrite behavior.
- Final paper release assembly is intentionally blocked until both complete
  audited aggregates exist.

## Current Experiment Status

### Newton joint-order bug found on an unmerged branch (2026-07-21)

**Status: all today's Newton jobs cancelled; fix not yet reviewed/merged.**

A parallel, unmerged branch (`sim2sim-verification-transfer-547ca5`, mirrored
to `origin/fix/migration`, last commit 2026-07-21 13:56 EDT) found that G1's
expert-command observation terms (`expert_motion` in `policy`, `critic`,
`expert_state`, `expert_goal`, `expert_window`, `reward_input`, plus
`expert_state.joint_pos`/`joint_vel`) and the action-offset default in
`randomize_joint_default_pos` were built from the *live* per-backend joint
enumeration instead of the pinned canonical `G1_29DOF_ISAACLAB_JOINT_NAMES`
order. This is a no-op under PhysX (which the canonical list already matches)
but active under Newton, where 27 of 29 joint slots differ from PhysX's
ordering. Every job submitted today used `physics=newton_mjwarp`.

Their own behavioral confirmation (`L1_strict/model_step_992870400.pt`,
Newton-trained, ~993M frames, `walk1_subject1`, seed 0): evaluating with a
mismatched joint/command order collapses survival and roughly doubles joint
tracking error (0.517/0.431 rad, 67/500 and 111/500 survived) versus a matched
order (0.240/0.110 rad, 323/500 and 500/500 survived) in both transfer
directions. Because `reward_input.expert_motion` carries the same leak, they
flag training itself as suspect beyond cross-backend deployment, not only a
deployment-transfer issue — though the same checkpoint evaluated Newton-native
(matched to how it trained) still survived 500/500, suggesting the permutation
is at minimum a consistent, learnable relabeling within one backend rather
than pure noise.

All 5 jobs running at the time this was discovered were cancelled rather than
left to keep training on the pre-fix ordering:
`5524182`/`5524183`/`5524338` (SONIC VRAM ablation v1/v2/v5),
`5524342` (BONES-SEED-91 5B resumable, segment 1, at 1.29B/5B frames),
`5524390` (LAFAN1 hardcoded-default sanity, at 900M/1B frames — 90% done).
`5524387` (the LAFAN1 *scaled* sanity check, exactly reproducing bn931wny's
config) had already completed before cancellation, reaching `ep_len=288.68` /
`r_ep=15.76` — beating bn931wny's `244.18`/`13.11` — but this result is
likewise pre-fix and should be treated as provisional. Between the two
completed/near-complete sanity arms, 8192 envs x 12 rollout steps reached
comparable quality to 4096 x 24 while finishing faster (~4.5h at ~65k fps vs.
~5.5h+ at ~46k fps for the same 1B frames) — a reasonable default scale
choice to carry forward once re-validated post-fix.

Full detail, the audit tool
(`scripts/dump_backend_index_contract.py`), and the fix commits are on the
unmerged branch; see `wiki/sim2sim-backend-verification.md` there. Not yet
reviewed for merge into `main`, and not yet reconciled against the
Strict/legacy-default reversal above (both branches diverged from a shared
ancestor and have not been compared for conflicts).

### Interface-ablation tracker arms submitted (2026-07-21, late evening)

**Status: all four arms submitted to ICE as jobs `5525739` (FB chunk,
running), `5525740` (EE chunk), `5525741` (FSQ), `5525742` (SONIC joint).**
Re-invoke each launcher to chain the next 16h segment; each refuses once its
5B cap is reached.

Per-step renewal decision (user, 2026-07-21): SONIC re-encodes its latent
every control step over the sliding future window, unlike our held-z
contract (which is the planner-friendly design). New pipeline knob
`--latent-hold-steps` (defaults to `--horizon-steps`, preserving every
existing run's behavior) sets `agent.ipmd.latent_steps_min/max`
independently of the encoder window. Both SONIC-flavored variants submit
with `--latent-hold-steps 1 --phase-mode none` (phase would be a constant at
hold=1; SONIC has no phase channel), so their latent command dim is 256, not
258. The main latent arm keeps the held-z contract.

Per the user's ablation-study decisions (plateau qualification instead of a
survival gate, task-index planner input, frame-0/~700-step eval, 5B budget
for every tracker, Study 1 on LAFAN1 — see
[Ablation Experiment Plan](ablation-experiment-plan.md)), four additional
LAFAN1 tracker arms are ready to join the running latent 5B job (`5525267`):

- `Isaac-Imitation-G1-Strict-v0` (new): vanilla observation/agent contract on
  the strict latent surface's protocol deltas (pelvis anchor, strict SONIC
  terminations, [0, 200] starts), so explicit-interface trackers differ from
  the latent arm only in command space.
- `experiments/submit_lafan1_chunk_tracker_5b_resumable_ice.sh`: FB-chunk and
  EE-chunk arms (`agent.command_space`, held 10-step chunks via
  `env.command_hold_steps=10`), plain `train.py`, resumable segments.
- `experiments/submit_lafan1_latent_variant_5b_resumable_ice.sh`:
  `VARIANT=fsq` (FSQ skill encoder; `FSQ_LEVELS` defaults to the
  SONIC-release token space, 64 dims x 32 levels ~= 320 bits per 5 Hz
  command, with an overflow-safe `FSQQuantizer` fix in RLOpt) and
  `VARIANT=sonic_joint` (`agent.ipmd.hl_skill_finetune_enabled=true`, PG +
  recon encoder finetuning; resume-safe via
  `hl_skill_command_sampler_state_dict`).

A 1-iteration PhysX smoke of the new Strict task with `ee_trajectory` passed
(actor in_keys confirmed as the expert-window EE terms).

### Phase 3: low-level protocol and causal planner code

**Status: complete as a code and local behavior gate.**

One-motion closed-loop experiments establish that a causal planner can command
both the latent and explicit interfaces. These runs are diagnostics, not paper
evidence across motions.

### SONIC default and policy-contract decision (2026-07-20)

**Status: code default, not yet re-validated at the new scale.**

With ICE H100 (and now H200) single-GPU access, the compute-scale objection
that paused the full SONIC surface on 2026-07-20 no longer applies at the
intended budget: 100k PPO iterations at 8192 envs x 12 rollout steps is
~9.83B (~10B) frames, matching the release's own convergence criterion
("after 100K iterations") on a single GPU instead of 64+. Decision:

- `Isaac-Imitation-G1-Latent-v0` (the SONIC surface) is the confirmed
  default latent task, not a paused/candidate one.
- `Isaac-Imitation-G1-Latent-Strict-v0` (legacy scaffolding + pelvis anchor +
  annealed strict terminations), briefly floated as the 2026-07-20 candidate
  default, is now DEPRECATED — kept only to reproduce runs already started
  on it.
- The default policy contract for the SONIC task is now the exact public
  release optimizer (`sonic_release_optimizer=True`: actor lr 2e-5, joint
  grad clip 0.1, init std 0.05, 6-layer SiLU MLPs, running input
  normalization), not the locally-validated small-scale contract — the
  release contract needs release-scale iteration counts to leave the flat
  regime, and 100k iterations now supplies that on one GPU.

Submitted 2026-07-20: the VRAM/throughput ablation
(`experiments/submit_sonic_latent_vram_ablation_ice.sh`, corrected LAFAN1,
2B-frame cap each) as ICE jobs `5523769` (v1, 8192 envs x 12 steps — njmax
95/nconmax 18), `5523770` (v2, 12288x12 — 143/27), `5523771` (v3, 16384x12 —
190/36), and `5523772` (v4, 12288x24 — 143/27); and the BONES-SEED
SONIC-latent job (`experiments/submit_bones_seed_100_sonic_latent_ice.sh`,
91/100-motion SONIC-exclusion-filtered manifest, L1 scale 8192x12, 3B-frame
cap, njmax 288/nconmax 32) as ICE job `5523773`.

**VRAM ablation result (2026-07-20): v3 and v4 failed within 10 minutes,
closed as-is (no resubmission).**

- v3 (16384 envs, `5523771`): genuine CUDA OOM, not a solver issue — 79.18 GB
  capacity, 76.82 GB already in use, failed allocating another 3 GB. The
  SONIC release network (6-layer [2048,2048,1024,1024,512,512] with running
  input normalization) plus rollout buffer at 16384 envs exceeds one H100's
  80 GB.
- v4 (12288 envs, rollout=24, `5523772`): contact-solver overflow, not VRAM —
  the proportional njmax/nconmax extrapolation (143/27) was too low for the
  longer 24-step rollout; the log shows repeated `nefc overflow` requests up
  to 196, and the run hard-crashed rather than NaN'ing. Confirms the
  extrapolation caveat noted in the ablation script: njmax/nconmax scaling by
  env count alone does not hold when rollout length also changes.
- v1 (8192 envs) and v2 (12288 envs) ran cleanly for 8.5+ h with no
  overflow/OOM. Per user direction, this is treated as sufficient signal for
  this ablation round: **12288 envs x 12 rollout steps fits one H100 and is
  the largest validated point; 16384 envs does not fit at this policy size.**
  No further arms were resubmitted.

**Correction (2026-07-21): v1/v2 "success" was contaminated by njmax
under-provisioning, not a clean result.** Log audit found `nefc overflow`
warnings throughout both "successful" arms: v1 (njmax=95) logged **7.4
million** overflow events over ~9.5h; v2 (njmax=143) logged 59,027. Peak
requested njmax was ~230-245 in BOTH arms regardless of env count (245 at
8192 envs, 232 at 12288), while the BONES-SEED job running concurrently at
njmax=288 logged zero overflow. This means njmax/nconmax is a per-step
contact-complexity budget driven by the SONIC env's domain
randomization/push events and early strict-from-scratch falling — NOT
something that scales with `num_envs`, contradicting the original
proportional-scaling assumption. All four VRAM-ablation arms were cancelled
and resubmitted with a fixed njmax=320/nconmax=40 (headroom above the
288/32 that measured zero overflow) as ICE jobs `5524182` (v1),
`5524183` (v2), `5524184` (v3), `5524185` (v4) — v3 (16384 envs) is expected
to OOM again since that failure was VRAM-related, not njmax-related.

**ICE partition walltime caps (2026-07-21): confirmed hardcoded, not a QoS
setting.** `scontrol show partition ice-gpu` shows `MaxTime=16:00:00`; `sinfo`
confirms every GPU-bearing PACE partition is capped the same way:
`ice-gpu`/`coc-gpu`/`coe-gpu`/`pace-gpu` at 16h, `ice-bw-gpu` at 18h. None of
the attached QoS (`coe-ice`, `coc-ice`, `pace-ice`) define a `MaxWall`
override, so the partition cap governs regardless of QoS choice — there is
no "long" GPU QoS on this cluster (unlike Skynet). Incidental find: H200s
are already in `ice-gpu` (`gres/gpu:h200=48`), so H200 access needs no
separate partition/QoS, just `--gres=gpu:h200:1`.

**Resumable BONES-SEED-91 SONIC-latent job (2026-07-21), 5B-frame cap.**
Since RLOpt's `save_model`/`load_model` restores weights + optimizer state
but not the frame/iteration counter (`frames_processed` resets to 0 on every
fresh `agent.train()` call), a walltime-capped job needing >16h of training
must be split into segments, and per-segment checkpoint filenames
(`model_step_<N>.pt`) are local to that segment rather than a global total.
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh` tracks true
cumulative frames itself in a remote state file keyed by the last-counted
checkpoint (crediting each segment's own contribution exactly once), and
computes the next segment's `--max_iterations` from the remaining budget;
`train_hl_skill_pipeline.py` gained `--train-checkpoint` to pass a low-level
checkpoint through to `train.py --checkpoint` for the resume case.
Re-invoking the script drives the chain forward; it refuses to resubmit once
5B frames are reached.

**v5 arm added, then the whole SONIC-default premise questioned (2026-07-21).**
v5 (`5524338`) re-tests the code's own hardcoded default shape (4096 envs x
24 rollout, mini_batch_size 24576 to match `rlopt_ipmd_cfg.py`'s literal
`4096 * 24 // 4`) under the SONIC release-optimizer contract at the
validated-safe njmax=320/nconmax=40, per the hypothesis that this exact
shape explains why earlier runs performed well. v3 and v4 were also
resubmitted at njmax=320/40 but both hit genuine CUDA OOM again (`5524184`,
`5524185`) — 12288 envs x 24 rollout doubles the collector buffer versus
v2's 12288x12 (which fits), so both are real VRAM-ceiling results, not a
solver misconfiguration; v1 (`5524182`) and v2 (`5524183`) are running
cleanly.

Pulling the actual W&B config for the run the user was comparing against
(`bn931wny`, project `g1-lafan1-strict`, group `ice3-l1-novideo`) revealed
the "L1" baseline never used the SONIC surface or release-optimizer contract
at all: `env_name=Isaac-Imitation-G1-Latent-Strict-v0`, `num_envs=8192`,
`collector.frames_per_batch=98304` (12 rollout steps),
`loss.mini_batch_size=12288`, `policy.num_cells=[512,256,128]` with
`activation_fn=elu` and `normalize_input=False` — the legacy/local optimizer
contract, not the release SiLU/[2048...512] one. It reached
`episode/length=244.18` and `episode/return=13.1`, well above anything the
new SONIC release-optimizer contract has produced so far.

**Reverted (2026-07-21): the 2026-07-20 "make SONIC the default" decision is
undone.** `Isaac-Imitation-G1-Latent-v0` resolves to the Strict/legacy
surface again (`_LATENT_STRICT_TASK_KWARGS`); `Isaac-Imitation-G1-Latent-Strict-v0`
is its back-compat alias. `Isaac-Imitation-G1-Latent-Sonic-v0` is opt-in
only and no longer aliased as `v0`.
`G1ImitationLatentSonicRLOptIPMDConfig.sonic_release_optimizer` reverts to
`False`. Every downstream script/pixi task that references the
`Isaac-Imitation-G1-Latent-v0` alias (interface_baselines scripts,
`smoke-ipmd`, etc.) automatically now gets the Strict/legacy surface again
by design — that is the whole point of using the floating alias rather than
a hardcoded surface name. The one exception fixed explicitly:
`experiments/submit_sonic_latent_vram_ablation_ice.sh` now targets
`Isaac-Imitation-G1-Latent-Sonic-v0` directly, since it specifically studies
the SONIC surface regardless of which surface is "default".

**Default-reversal sanity check submitted (2026-07-21).**
`experiments/submit_lafan1_strict_default_sanity_ice.sh` runs two arms on
`Isaac-Imitation-G1-Latent-v0` (corrected LAFAN1, ~1B frames,
njmax=320/nconmax=40): `scaled_e8192_r12` (ICE job `5524387`) exactly
reproduces bn931wny's config (8192 envs x 12 steps x minibatch 12288) as the
actual correctness check; `hardcoded_default_e4096_r24` (ICE job `5524390`)
tests the code's literal default shape (4096 envs x 24 steps, minibatch
24576) as a second, unvalidated data point on whether scale matters. Neither
has reported results yet.

Both running BONES-SEED jobs (`5523773` 3B and `5524188` 5B segment 1) were
cancelled and resubmitted with `TASK=Isaac-Imitation-G1-Latent-Strict-v0`
(matching the actual L1 config above) instead of the SONIC default; the
policy contract follows automatically since `Latent-Strict-v0`'s task kwargs
already route to the legacy-style `G1ImitationLatentRLOptIPMDConfig`, not
the Sonic one. `experiments/submit_bones_seed_100_sonic_latent_ice.sh` (the
3B one-shot) is now marked superseded/reference-only.
`experiments/submit_bones_seed_sonic_5b_resumable_ice.sh` is the live
launcher; its first segment under the corrected task is ICE job `5524342`.

### Non-paper BONES-SEED SONIC latent training

**Status: debugged locally; no active ICE job.**

Jobs `5523561`, `5523570`, `5523578`, and `5523588` are stopped or failed and
are not training results. The h25/z256 encoder checkpoint was retained, but the
Newton low-level run at the official flat-locomotion value `njmax=95` produced
NaN returns. A first-rollout finite-value trace ruled out MPJPE, the skill
encoder, and the actor: the latent and initial policy outputs were finite, then
Newton state and six independent reward terms became non-finite after contact
constraint overflow. The failing sample was `ab_bicycle_001_A359` near frame
20. In its first 200 frames, 25 of 32 body origins are below 5 cm; corrected
LAFAN1 has at most 3. An 8,192-environment LAFAN1 control had zero overflows,
while BONES-SEED at `njmax=95` had 951 in one rollout and requested up to 236
constraint rows.

A reduced debug manifest containing only `ab_bicycle_001_A359` and
`crawl_ff_loop_180_R_001_A214` isolated the interacting Newton capacities
without altering either motion. At 2,048 environments, `njmax=264` and
`nconmax=31` still overflowed (268 constraint rows requested), whereas
`272/32` and `288/32` each passed 30 rollouts across seeds 0, 1, and 2 with no
constraint/contact overflow or NaN. The retained setting is `288/32` to keep
20 rows of headroom above the observed request. Relative to the borderline
`264/31` setting, it reduced steady throughput by 0.87%; at 2,048 environments
GPU memory increased by 96 MiB (2.3%) compared with `95/18`.

The full 100-motion Newton run at `288/32` then completed 20,054,016 local
frames in 186.4 seconds with no overflow or NaN. Steady throughput was about
108--110 thousand frames/s, observed GPU use was 34,916 MiB, and the final
metrics included mean episode length 19.97 and mean episode reward 0.5444.
This qualifies the capacity change for local testing; it is not a training
result or a paper qualification. A separate PhysX local run was also finite
through 20,054,016 frames. No replacement was submitted at the user's request.

### Phase 4: corrected LAFAN1, no language

**Status: low-level prerequisites active; planner paper grid not submitted.**

Last verified on 2026-07-16:

| Purpose | Slurm job | State at last check |
| --- | ---: | --- |
| Corrected-LAFAN vanilla low level | `3500993` | Running |
| Corrected-LAFAN DiffSR low level | `3503434` | Running |
| Strict paired qualification | `3503441` | Waiting on both jobs |

The guarded Phase 4 planner grid remains blocked until both controller audits
and the matching streamed-vanilla certificate pass. The future planner grid is
fixed to seeds `0, 1, 2`, all 40 corrected motions, and sample budgets
`1k/10k/50k`.

Detailed chronology:
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

### Phase 5: BONES-SEED language study

**Status: low-level qualification passed; first planner preparation attempt
failed before training.**

The corrected, provenance-complete 100-motion BONES-SEED tree and separate
latent/vanilla caches passed their audits. Qualification job `3512041`
completed successfully:

| Controller | Strict success | Required |
| --- | ---: | ---: |
| Direct vanilla | 0.90 | 0.80 |
| DiffSR latent | 0.84 | 0.80 |

The selected skill checkpoint is tensor-bound to the encoder embedded in the
qualified latent checkpoint. The persistent qualification root is:

```text
logs/interface_baselines/bones_seed_100_low_level_qualification_seed0_retry_20260716
```

The first guarded three-seed planner chains were:

| Seed | Prepare | Rollout | Fine-tune | Final eval | Summarize |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3512092` | `3512093` | `3512094` | `3512095` | `3512096` |
| 1 | `3512097` | `3512098` | `3512099` | `3512100` | `3512101` |
| 2 | `3512113` | `3512114` | `3512115` | `3512116` | `3512117` |

All three prepare jobs failed after about 2 hours 16 minutes. Each had written
98 explicit demonstration chunks, but no latent chunks or complete prepare
stage record. The two shared failure signals were:

1. repeated `OSError: [Errno 28] No space left on device` from compute-local
   job storage; and
2. the fixed collection limit ended with four motions below the old
   1,000-row-per-goal target:
   `ab_bicycle_001_A359`, `crawl_ff_loop_180_R_001_A214`,
   `jump_sideway_135_001_A021`, and
   `sitting_legs_bend_arms_front_loop_001_A030`.

The dependent rollout arrays show `DependencyNeverSatisfied`, so no incomplete
preparation data reached planner training or evaluation. These failed chains
are not paper results and must not be resumed without auditing the partial
artifacts.

**2026-07-23 latent-only H200 language pilot submitted.** The new H200
controller is intentionally being tested outside the paper comparison: it was
trained on the 91-motion SONIC-filtered manifest and has no matched qualified
vanilla checkpoint. The campaign therefore sets `INTERFACES=latent_skill` and
does not run a full-body baseline, controller comparison, or GPU/parameter
ablation. It uses ten goals common to the 91-motion and fresh 100-motion trees,
150 demonstration rows plus 150 planner-rollout rows per goal, ten same-goal
rollout environments, and the fixed 500-step evaluation.

The guarded dependency chain is:

| Stage | Slurm job |
| --- | ---: |
| Prepare | `3560697` |
| Rollout array | `3560698` |
| Fine-tune | `3560699` |
| Final eval array | `3560700` |
| Summarize | `3560701` |

Output root: `logs/interface_baselines/bones_seed_h200_language_preliminary_seed0_20260723`.
The persistent record is on Skynet at the corresponding `cluster_submission.json`.
At submission verification, all five jobs were `PENDING`; no stage had begun.
The H200 checkpoint SHA-256 is
`6765a324a840b33a84f9a0b5a817c60303979bbec7a36ebc31242086d61d1572` and the
encoder binding passed all 14 tensor checks. This run is
`preliminary_unqualified=true` and cannot enter the paper aggregate.

**2026-07-23 local ten-goal baseline campaign added.** To obtain the two basic
planner baselines without waiting on cluster queues, the campaign
`experiments/campaigns/2026-07-23-bones-phase5-language-local10/` runs the same
shared Phase-5 workflow entirely on the local workstation: latent-only, the
same ten goals and frozen H200 checkpoints as the H200 pilot, a derived
ten-motion subset manifest (`data/bones_seed_phase5_local10/`, source-hash
recorded) so Isaac only loads the needed references, 150 demonstration plus
150 rollout rows per goal, and 500-step episodes. Its two deliverables are the
demonstration-pretrained planner and the rollout-finetuned planner under
`logs/interface_baselines/bones_seed_phase5_local10_seed0/latent_skill/`. As
part of this, the canonical stage driver
`run_bones_seed_multigoal_language_comparison.{py,sh}` (now under
`experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/`)
was made location-independent: it resolves its sibling workflow scripts and the
repository root from its own path instead of the hardcoded pre-reorganization
`experiments/interface_baselines/` prefix, which had left the pipeline
unrunnable after the 2026-07-23 script reorganization. Like the H200 pilot,
this local run is `preliminary_unqualified=true` and is not paper evidence.

Data-budget interpretation is important: one saved row is one 5 Hz planner
decision containing a ten-frame 50 Hz command chunk. The failed configuration
requested 100,000 demonstration rows plus 100,000 rollout rows per interface,
then fine-tuned on 200,000 merged rows. It was not a small dataset.

The current recommended Phase 5 budget is:

- 15,000 balanced demonstration rows total: 150 per goal;
- 15,000 planner-rollout rows total: 150 per goal;
- 30,000 unique rows in the merged fine-tuning dataset.

The old 100,000 plus 100,000 configuration should become an optional
large-data scaling point, not the default paper run. The 150-row setting is
encoded in the latent-only preliminary campaign above; it has not replaced the
guarded paper launcher. Before changing the paper launcher, verify that the
four difficult motions can reach 150 rows and increase the collection safety
limit without changing the 500-step episode protocol.

Data preparation and hashes:
[BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md).

## Preliminary Planner Evidence

The corrected one-motion `walk1_subject1` experiments show:

- causal planners work for both interfaces;
- at the tiny size, latent is stronger across flow, diffusion, and
  deterministic objectives in the current diagnostic;
- the three-seed flow diagnostic first reaches the fixed target at about
  `0.13M` parameters for latent and `4.19M` for explicit;
- explicit often catches up or obtains lower MPJPE at larger sizes;
- rollout fine-tuning frequently hurts tracking in the current one-motion
  setting, so demonstration-only and fine-tuned results must remain separate.

The working interpretation is that the latent interface may reduce the planner
capacity required for useful control, not that it always has a better
large-model tracking ceiling. None of these one-motion results is a paper claim
until repeated across motions.

Exact diagnostic tables and artifact paths are in
[LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md).

## Immediate Work Queue

1. Change the Phase 5 default paper data budget to 150 demonstration and 150
   rollout rows per goal, while preserving exact balanced counts.
2. Fix compute-local storage use and prevent repeated storage errors from
   creating gigabyte-scale logs.
3. Add an audited recovery path that either trims and reuses valid partial
   shards or deliberately starts from a fresh output root. Never silently mix
   partial seeds.
4. Run the smallest local collection/schema smoke for the revised budget.
5. Dry-run all three guarded seed launchers, then submit replacement Skynet
   chains only after the preflights pass.
6. Allow Phase 4 low-level jobs and qualification to finish; submit the Phase 4
   planner grid only after its strict gate passes.
7. Aggregate Phase 4 and Phase 5 only from complete audited seed sets, then
   build the final paper release bundle.
8. Run the bounded planner architecture/size study after the main Phase 5
   pipeline is healthy; do not multiply architecture, data, and command-style
   sweeps into one combinatorial grid.

## Execution Policy

- Use the local workstation for code debugging, inference, metrics, and video.
- Local low-level runs may reach about 10M frames for routine debugging and at
  most about 50M for a serious check. Do not run 100M locally.
- Use Skynet for long low-level convergence, large data collection, final
  verification, and paper-quality numbers.
- Preserve the frozen rewards, resets, terminations, random start range, push
  event, command cadence, and episode length unless the user explicitly
  changes the research protocol.

## Document Map

- [Project Progress Report](progress-report.md): results-facing summary in
  three fixed sections (latent encoder ablations, interface design,
  hardware), updated on every result change.
- [Experiment Navigation](../experiments/README.md): current dated campaign,
  historical launcher indexes, and staged paper-facing entrypoint.
- [Causal High-Level Interface Paper Plan](causal-interface-paper-plan.md):
  authoritative research design and phase contract.
- [Whole-Body VLA and Latent-Action Literature Review](whole-body-vla-literature-review.md):
  what current explicit-chunk and latent-action systems actually deploy, how
  they relate to our comparison, and the boundary between a native baseline
  and a literature-inspired diagnostic.
- [LAFAN1 From-Scratch Interface Comparison](lafan1-from-scratch-comparison.md):
  detailed Phase 3/4 chronology, diagnostics, checkpoints, and job history.
- [BONES-SEED Phase-5 Data Preparation](bones-seed-phase5-data-preparation.md):
  corrected data tree, hashes, caches, qualification, and Phase 5 handoff.
- [Fair Interface Baselines](fair-interface-baselines.md): operational
  two-interface runner and adapter details.
- [Context Management](context-management.md): repository ownership and where
  future context belongs.

When this page disagrees with a phase document about a frozen protocol, verify
the code and update both. When it disagrees only about current execution state,
this newer dated snapshot should be refreshed from Slurm and treated as the
status entry point.
