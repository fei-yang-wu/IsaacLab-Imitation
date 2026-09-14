# Direct PoE: capacity, batch size, and training length

The completed z64 direct-PoE sweep reduced paired held-out dynamics loss from
8.21570 to 6.87854 when the final hidden width increased from 512 to 2048.
The user requested wider/deeper networks, then explicitly added batch size,
longer training, and related optimization ablations. This campaign submits
ten fresh pretrain-only runs. No tracker or planner stages are included.

Every model remains epsilon_hat = z^T E(s,y_t,t), with z_dim=64 and 64 expert
fields. Only the pretraining decoder's internal capacity changes in the first
six runs. Reuse the completed poe_joint_wide seed-0 control (job 5769229,
previous campaign) rather than resubmitting it.

## Architecture screen

All six arms use batch8192, 50k updates, encoder/NTP LR 3e-4, seed0.
Widths are ResidualMLP block widths: each block contains two dense layers,
LayerNorm and a residual connection; this is not a count of individual layers.

| Arm | Hidden block widths | Active mu-network parameters | Contrast |
|---|---|---:|---|
| Existing control | 1024, 1024, 2048 | 68,207,744 | Completed reference |
| tail4096 | 1024, 1024, 4096 | 139,779,200 | Wider final hidden layer |
| tail8192 | 1024, 1024, 8192 | 308,087,936 | Further final-layer widening |
| uniform2048 | 2048, 2048, 2048 | 81,592,448 | Widen earlier blocks at fixed final width |
| depth5 | 1024, 1024, 2048, 2048, 2048 | 85,001,344 | Two extra residual blocks |
| depth7 | 1024, 1024, 2048, 2048, 2048, 2048, 2048 | 101,794,944 | Four extra residual blocks |
| wide_deep | 2048, 2048, 4096, 4096, 4096 | 228,696,192 | Combined capacity exploration |

Parameter counts cover mu_net only, measured from the actual 774-input and
26,752-output network geometry. Shared encoder, time embedding, unused phi
state branch, and inactive endpoint head are excluded. No parameter or FLOP
matching is claimed; compare runtime along with fixed-budget loss.

## Batch and training-length screen

All four use the completed control architecture 1024/1024/2048. Each processes
exactly 1,638,400,000 training examples, four times the original budget.
Larger batches get fewer optimizer updates, making equal-example comparisons
possible. Every run starts fresh; the constant-LR learning curve also exposes
shorter-budget behavior. No learning-rate schedule is added.

| Arm | Batch | Updates | Active encoder/NTP LR | Log interval |
|---|---:|---:|---:|---:|
| long_b8192 | 8,192 | 200,000 | 3e-4 | 1,000 |
| batch16384 | 16,384 | 100,000 | 3e-4 | 500 |
| batch32768 | 32,768 | 50,000 | 3e-4 | 250 |
| batch32768_lr2 | 32,768 | 50,000 | 6e-4 | 250 |

Logging/evaluation occurs every 8,192,000 training examples in this block.
Evaluation batch size is pinned to 8,192 for ALL ten arms, so changing training
batch size does not change diagnostic population size. Four eval batches and
the identical final paired probe are retained. The doubled-LR arm tests
square-root scaling for the 4x batch increase, as an empirical candidate.
The --encoder_lr CLI flag reaches the existing validated config field and
both the encoder and active NTP optimizer groups. The inactive endpoint
DiffSR LR remains 1e-4. The default is unchanged at 3e-4.

Compare ordinary NTP curves at equal examples (not just equal update labels):

| Examples | Batch8192 update | Batch16384 update | Batch32768 update |
|---:|---:|---:|---:|
| 409,600,000 (original budget) | 50,000 | 25,000 | 12,500 |
| 819,200,000 | 100,000 | 50,000 | 25,000 |
| 1,638,400,000 | 200,000 | 100,000 | 50,000 |

Final paired probes are available at the last row for these four arms. The
intermediate rows have ordinary held-out NTP metrics, not the final fixed-noise
probe. Checkpoints retain the existing latest/best behavior; intermediate
probe/checkpoint retention is not added. Equal-update comparisons are also
available, but expose different numbers of training examples and must be
labeled accordingly.

## Held fixed and readout

Everything else matches the prior campaign: deterministic z64 encoder,
LayerNorm, intermediate window, past-five plus current source, horizon10,
executed-heading boundary_next raw-chunk target, endpoint coefficient0,
SIGReg1 and latent L2 0.001, seed0, the same BONES-SEED 129,785-motion arrays
and 90/10 trajectory split with seed0. campaign.yaml records exact paths,
all CLI arguments, and resource settings.

Primary metric: train/jepa_ntp_loss_eval at the specified budget. Final
paired diagnostics: dynamics_probe/{train,eval}/mean/real_z and each noise
level, with shuffled/zero-z controls, seed1729 and 1,024 examples per split
per level. Loss sums over the unchanged 418-D target. Keep ordinary NTP and
paired-probe columns separate. Do not use the inactive endpoint's losses.
Report latent scale/effective rank and elapsed time alongside loss.

This is a seed-0 screen of achievable predictive fit. It does not establish
statistical significance, the best possible architecture, policy quality,
or semantic composition. Evaluate all configured arms at their declared
budgets rather than selecting a favorable training checkpoint. A later
combination of the strongest architecture and optimization settings, with
repeat seeds, can follow once this screen completes.

ICE resources: one H200, 16 CPUs, 160G host RAM and 15:59 walltime per job,
with persistent output under
/storage/ice-shared/vip-vwt/scratch-fwu91/poe_capacity_pretrain/.
Each arm has its own compiler caches. W&B: g1-bs-pareto / poe-capacity-pretrain.

## Launch and provenance

Use the repository control plane for each seed-0 arm:

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
  --campaign experiments/campaigns/2026-09-13-poe-capacity-pretrain/campaign.yaml \
  --arm <arm> --seed 0
pixi run python -m imitation_experiments.pipeline.cluster submit \
  --plan <plan-dir> --confirm <plan-sha>
```

Qualification and actual submission records are appended below.

## Qualification

- The resolved campaign contract tests passed: capacity arms preserve the
  completed control's arguments except hidden widths and run identity;
  optimization arms have equal-example totals and evaluation cadence.
- Three actual local Isaac/Newton pretrains completed two optimizer updates
  at full training batch sizes: tail8192 at batch8192, wide_deep at batch8192,
  and the reference-width model at batch32768 with LR6e-4. Every run saved its
  checkpoint and final paired probe. These establish wiring and local memory
  feasibility, not research performance.
- Checkpoint metadata confirmed z_dim=feature_dim=64 and the intended hidden
  lists/batches. Optimizer groups were [3e-4,1e-4,3e-4] for the capacity
  checks and [6e-4,1e-4,6e-4] for the large-batch learning-rate check: the
  first and last groups are the encoder and active NTP head.
- Compared 82 algorithm/entrypoint Python files against the completed
  control's submitted archive. Only train_hl_skill_diffsr.py differs, adding
  the explicit encoder_lr flag and forwarding it into the existing config.
  No algorithm implementation changed for this follow-up campaign.
- CLI and campaign test files pass Ruff; git diff --check passes. Local
  records and parameter counts are under logs/poe_capacity_smoke/.

## Submitted jobs

All ten plans passed all eight preflight checks. All ten Slurm jobs were
RUNNING when checked through the control plane at 2026-09-13 02:19 UTC.
This is launch evidence, not a convergence result.

| Arm | Job | Frozen plan SHA |
|---|---:|---|
| tail4096 | 5769838 | `45d7090bbcf3f88657a7b6c7e7ec716f3e2361002458c617bac242dddd15ed92` |
| tail8192 | 5769839 | `4749299bedf20377a99d17bcac1bbbcbf2643e27649e05411434f112781b05cb` |
| uniform2048 | 5769840 | `1d2417bece54f3212faec18859a9b7290d1c7109789055b1164c46047d4eca63` |
| depth5 | 5769841 | `d0fc7e7ed2ec0f0e7ea5570228108247855198e0aa5a68c7c2307b928067cb65` |
| depth7 | 5769842 | `4a4d81971be1e2a919c54fcd74b975d6c2144f9b41ddd35c430809cefd091810` |
| wide_deep | 5769845 | `52c30aafe87d81060c410ada5ef4a0e5615d13d756026c3cbf6eee8d7871e4da` |
| long_b8192 | 5769847 | `238757dd0f866e6c5baa4e771cce091728a9b4d87029d704cdf5472b7eb75789` |
| batch16384 | 5769848 | `9b5231cd32b63d94c007ab770ae317daa51bd10d85cda3e4605b700f9e6f83b4` |
| batch32768 | 5769850 | `6c08723bfa21aeb75e79abc0666640946b05e7e7100dc21704c570ffdc2325e1` |
| batch32768_lr2 | 5769851 | `05e8263f03f9db9c07a4239630710b216ea255f96632ce911046cace88d88a2d` |

Frozen plans under logs/cluster_control/poe-capacity-pretrain/:

- `poe-capacity-pretrain-tail4096-s0-20260913-021458-45d7090b`
- `poe-capacity-pretrain-tail8192-s0-20260913-021458-4749299b`
- `poe-capacity-pretrain-uniform2048-s0-20260913-021458-1d2417be`
- `poe-capacity-pretrain-depth5-s0-20260913-021458-d0fc7e7e`
- `poe-capacity-pretrain-depth7-s0-20260913-021458-4a4d8197`
- `poe-capacity-pretrain-wide_deep-s0-20260913-021540-52c30aaf`
- `poe-capacity-pretrain-long_b8192-s0-20260913-021540-238757dd`
- `poe-capacity-pretrain-batch16384-s0-20260913-021540-9b5231cd`
- `poe-capacity-pretrain-batch32768-s0-20260913-021540-6c08723b`
- `poe-capacity-pretrain-batch32768_lr2-s0-20260913-021540-05e8263f`

All submissions have drift=false and share verified source archive
`a38f7b998b87d21db3ebbf952fe509dbdeb6ac90e50ca9d4ad85f6c96891714d` (70.1 MB).
The submitted trainer, CLI, model, diagnostics and campaign files were
checked against the local sources. Earlier uncommitted RLOpt work was
preserved and matches the completed control archive; this follow-up adds
only CLI learning-rate plumbing to runtime code. No new submodule commit
or parent-pointer commit was made. The archived source is authoritative.

Local launch index: logs/poe_capacity_smoke/plans.json. Submission and
scheduler logs are beside it. Training-example counts include repeated
sampling of dataset windows; they are not counts of unique motions.

## Progress snapshot: 2026-09-13 02:47 UTC

All ten jobs remain Slurm RUNNING after roughly 29-31 minutes; no final
probe rows yet. Latest pulled ordinary held-out NTP losses below are at
unequal budgets and are not an architecture ranking.

| Arm | Latest / target update | Held-out NTP loss |
|---|---:|---:|
| tail4096 | 29,000 / 50,000 | 7.3909 |
| tail8192 | 4,000 / 50,000 | 11.4713 |
| uniform2048 | 36,000 / 50,000 | 7.5053 |
| depth5 | 37,000 / 50,000 | 6.9135 |
| depth7 | 28,000 / 50,000 | 7.2712 |
| wide_deep | 4,000 / 50,000 | 418.1397 |
| long_b8192 | 45,000 / 200,000 | 7.0464 |
| batch16384 | 7,500 / 100,000 | 10.3691 |
| batch32768 | 13,500 / 50,000 | 8.4009 |
| batch32768_lr2 | 9,250 / 50,000 | 8.5580 |

Concern: wide_deep has remained near loss418 from updates1000-4000,
consistent with the 418-D zero-noise-prediction baseline, while mean z std
fell from 0.00146 to 0.000434 over that interval. This is a learning-failure
signal, not a scheduler failure or established causal diagnosis. No runs
were canceled or changed. Raw metrics/configs and the status summary are
in logs/poe_capacity_pretrain_results/.
