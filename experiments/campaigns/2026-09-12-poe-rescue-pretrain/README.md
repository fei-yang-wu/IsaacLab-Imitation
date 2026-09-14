# PoE rescue: six z64 pretraining screens

The joint PoE heads lag the structured affine encoder in early matched tracker
comparisons. This campaign tests whether explicit multiplication or more joint
hidden capacity recovers DiffSR denoising fit while keeping the command 64-D.
The user authorized design and submission of pretrain-only runs on September 12.
There are no tracker, policy, or planner stages.

Let y_t be the noisy future chunk and suppress the common sqrt(embed) scaling:

| Arm | Noise prediction | Purpose |
|---|---|---|
| affine_next | (Az+b)^T F(s) M(y_t,t) | Fresh structured control |
| affine_pair | (Az+b)^T F(s) M(s,y_t,t) | Add only source input to mu |
| poe_joint | z^T D(s,y_t,t) | Fresh direct PoE control |
| poe_joint_wide | z^T D_wide(s,y_t,t) | Last mu hidden width 512 to 2048 |
| poe_modulated | z^T reshape(W[(1+gamma(s))*h(y_t,t)+beta(s)]+b) | Source-modulated expert features |
| poe_factored | z^T A^T F(s) M(y_t,t) | Remove only b from structured control |

The modulated head is the modeling extension: separate source and noisy-target
branches meet through feature-wise multiplication and shift before the 64-field
readout. Source modulation uses an MLP of width 1024/1024/512 and outputs 512
scales plus 512 shifts. Its last weights start at normal std 0.001 and bias 0,
so modulation starts near the target-only head and receives source gradients
immediately. z never enters either branch. This is inspired by
[FiLM](https://arxiv.org/abs/1709.07871) and
[low-rank bilinear pooling](https://arxiv.org/abs/1610.04325); the ingredients
are established, and their benefit for this skill-learning setup is untested.
Compared with pair concatenation, this changes both fusion and source-branch
capacity, so it is an architecture test, not an isolated proof about products.

The factored head is exactly the requested <z,E> form with E=A^T F M;
no constant code or extra latent dimensions are introduced. The multiplication
is evaluated in the efficient order (F^T Az)^T M. The joint head's 64 fields
remain 64 even in the wide arm. Its last-layer weight count then matches the
256-field/512-hidden target readout, but total parameter counts and compute
are not matched to the structured model. The wide/direct pair changes only
last hidden width. The factored/affine pair changes only the action-map bias.
A fixed Az lift was deliberately omitted because it can be absorbed into the
joint decoder's linear output layer.

## Fixed protocol

- Seed 0 for all six; fresh pretrain, 50,000 updates, batch 8,192.
- z_dim=64, deterministic encoder, LayerNorm, hidden 2048/1024/512/512, SiLU.
- Same BONES-SEED 129,785-motion reference arrays and persist ID as the
  September 12 regularized PoE campaign; see campaign.yaml for exact paths.
- Past-five plus current source, current heading anchor; intermediate encoder
  window; horizon 10; diff_chunk target boundary_next, executed heading.
- SIGReg 1.0 and latent L2 0.001, endpoint coefficient 0, NTP coefficient 1.
- Original optimizer defaults: encoder/NTP Adam group LR 3e-4; inactive
  endpoint DiffSR group LR 1e-4. No learning-rate sweep.
- Embed width 1024, F hidden 1024/1024/512; target-network hidden
  1024/1024/512, except the wide arm's last width 2048.
- Train/eval trajectory split 90/10, split seed 0. Log/evaluate every 1,000
  updates, four evaluation batches. Match final update 50,000, not best.pt.
- ICE: one H200, 16 CPUs, 160G host RAM, 15:59 walltime per job. Persistent
  checkpoints under /storage/ice-shared/vip-vwt/scratch-fwu91/poe_rescue_pretrain/.
  Each arm has independent TorchInductor and Triton caches.
- W&B project g1-bs-pareto, group poe-rescue-pretrain.

## Readout and interpretation

Primary curve: train/jepa_ntp_loss_eval, the actual active raw-chunk DiffSR
loss. train/jepa_ntp_loss is its training counterpart. Do not use generic
loss_real_z_eval or endpoint metrics: that endpoint head has coefficient zero.
All losses sum over target coordinates, so compare only identical targets.

The opt-in final dynamics probe reports dynamics_probe/{train,eval}/noise_k/
{real_z,shuffled_z,zero_z}, plus a uniform mean over noise levels. Each split
uses 4 x 256 examples per level, seed 1729, and local Gaussian generators.
Sampling resets Torch CPU/CUDA seeds independently of model construction;
all conditions reuse identical examples and Gaussian noise. A nonzero cyclic
shift avoids self-paired shuffled examples. The probe does not update models
or normalizers and restores model modes and RNG state. Results land in the
same final metrics.jsonl row and W&B record as post_train_eval at update 50k.
Probe flags are recorded in config.yaml and its recorded command.

- Wide joint closes the gap: finite joint hidden capacity is implicated.
- Factored no-bias matches affine: the pure <z,E> expression is compatible
  with the good fit when E is internally factorized.
- Affine/pair degrades versus affine/next: adding the source to mu hurts even
  when explicit multiplication is retained.
- Modulated PoE improves over direct PoE: source/target feature factorization
  is a useful candidate without expanding the command or expert count.
- Train and eval losses distinguish fitting trouble from an increased gap.
- Real/shuffled/zero losses and existing z effective rank describe command
  use; shuffled examples are counterfactual combinations and cannot establish
  useful semantic composition or causal encoder collapse.

All are jointly learned encoders: lower loss demonstrates achievable fit
under this optimizer/budget, not a proof of function-class expressivity.
One seed is a screen. A frozen common-encoder decoder comparison and repeats
can follow if these arms leave decoder fit versus co-adaptation unresolved.
Linear noise-field identities do not prove exact clean PoE sampling.

## Submission

Use the repository cluster control plane, one plan and one pretrain job per arm:

```bash
pixi run python -m imitation_experiments.pipeline.cluster plan \
  --campaign experiments/campaigns/2026-09-12-poe-rescue-pretrain/campaign.yaml \
  --arm <arm> --seed 0
pixi run python -m imitation_experiments.pipeline.cluster submit \
  --plan <plan-dir> --confirm <plan-sha>
```

Submission records, validation, and live scheduler checks are appended below.

## Validation notes

Checkpoint round-trip testing exposed an existing trainer reload omission:
load_checkpoint restored encoder and endpoint weights but omitted jepa_state_dict.
This submission restores predictor, g/f, optional target encoder, and active
NTP diffusion weights. This affects resume/eval-only; fresh pretraining's
previous in-process losses are unaffected. Both new architectures must pass
round-trip prediction parity before submission. Existing unrelated RLOpt edits
are preserved; workspace archive hashes identify the actual submitted source.

- Qualification: 15 targeted architecture/probe tests and the full 11-test
  online-JEPA checkpoint suite passed, plus the resolved six-arm campaign
  contract test. The sets overlap. Pytest dependency warnings required
  pre-importing Torch/RLOpt and disabling third-party plugin autoload; no
  repository warning policy changed.
- Both new heads completed two local pretrain updates (batch 8, full model
  widths) with 56 final probe fields and persistent checkpoint saves. These
  are wiring checks, not performance evidence. Local records:
  logs/poe_rescue_smoke/{encoder,modulated}/.
- All six plans passed all eight ICE preflight checks. Each resolves to one
  pretrain stage, z64, 50k updates, batch8192, identical trajectory split and
  active NTP objective. No tracker stage is present.
- Changed module/CLI/probe/campaign files pass Ruff; the existing large
  hl_skill_diffsr.py retains its 20 unrelated lint findings. Both repository
  diffs pass whitespace checks.

## Submitted jobs

Verified through the control-plane status command at 2026-09-13 00:03 UTC: all six
Slurm allocations RUNNING. This is launch evidence, not a convergence result.

| Arm | Pretrain job | Frozen plan SHA |
|---|---:|---|
| affine_next | 5769225 | `f257fccdd3079993e300396c5448b0d920994093e008b8d445f92f24cfe877c4` |
| affine_pair | 5769226 | `bd08d2f7032279d48c1e461c8cae54da5b3ef5c35ac60404e0c9e55d628b456d` |
| poe_joint | 5769227 | `ad92d7fe4441427bff668f237e8341a9051cc44ad5a072c9f3456495cc7a14f4` |
| poe_joint_wide | 5769229 | `e3ef84102d07d64154897e2e4e3ab28d1350963005eb35f38f8e131df3b4e79a` |
| poe_factored | 5769231 | `0291c2114aed2006444ce7b3aa2174b262be40687165a5252b48258c34099fe9` |
| poe_modulated | 5769232 | `56d0af175d9cbeb3b49cc5b36a49594930a78ed39f95b1f25ac35da340827e0c` |

Plan directories under logs/cluster_control/poe-rescue-pretrain/:

- `poe-rescue-pretrain-affine_next-s0-20260912-235931-f257fccd`
- `poe-rescue-pretrain-affine_pair-s0-20260912-235931-bd08d2f7`
- `poe-rescue-pretrain-poe_joint-s0-20260912-235931-ad92d7fe`
- `poe-rescue-pretrain-poe_joint_wide-s0-20260912-235931-e3ef8410`
- `poe-rescue-pretrain-poe_factored-s0-20260912-235931-0291c211`
- `poe-rescue-pretrain-poe_modulated-s0-20260912-235931-56d0af17`

All submissions have drift=false and share the verified 70.1 MB workspace
archive `cec98ab8171de3830205075c8d6af3c018479a99a464c3591a5d89d61a119b7e`.
The archive was checked against the local model, diagnostic, trainer, CLI,
and campaign files. Algorithm changes are in the RLOpt submodule; existing
uncommitted work was preserved. No new submodule commit or parent-pointer
commit was made; the submitted source archive is the reproduction authority.

## Partial results: September 13, 01:19 UTC

Three jobs completed with exit 0:0, update 50,000, and post_train_eval
probe rows: poe_joint, poe_joint_wide, poe_modulated. The others remain
RUNNING; the pulled snapshot ends at 42k / 41k / 40k for affine_next,
affine_pair, and poe_factored. This is a seed-0 partial grid.

Final paired probe (1,024 examples per split per level, eight noise levels;
loss sums target coordinates, lower is better):

| Completed arm | Train real z | Held-out real z | Held-out shuffled z | Held-out zero z | z std | z effective rank |
|---|---:|---:|---:|---:|---:|---:|
| poe_joint | 8.4134 | 8.2157 | 138.46 | 417.23 | 0.2176 | 48.32 |
| poe_joint_wide | 7.0624 | 6.8785 | 126.72 | 417.23 | 0.0291 | 60.15 |
| poe_modulated | 8.2908 | 8.1132 | 346.01 | 417.23 | 0.0475 | 50.89 |

The wider direct head has lower paired loss at all eight levels than the
standard direct head. Its mean held-out loss is 16.3% lower. Modulation has
1.2% lower mean loss; it is slightly higher at levels 0-2 and lower at 3-7.
These are observations from one seed, not causal proof or policy results.
Code scale changes substantially; effective rank is scale-normalized and
should not be interpreted as command quality. The affine/factored comparison
awaits its final budgets.

Pulled configs differ only on the four declared architecture fields. Local
artifacts: logs/poe_rescue_pretrain_results/{summary.csv,noise_levels.csv,
snapshot.json,summarize.py}; per-arm raw metrics.jsonl and config.yaml are
retained there. snapshot.json hashes the source files and scheduler snapshots.

## Final results: 2026-09-13 01:59 UTC

All six jobs are COMPLETED with exit 0:0, update 50,000 and final
post_train_eval probe rows. The raw metrics/configs and scheduler snapshots
were pulled again, and summary.csv, noise_levels.csv and snapshot.json
regenerated. All results below are z64, seed 0, matched pretrain budgets.

| Arm | Runtime | Regular held-out NTP | Paired train loss | Paired held-out loss | Shuffled z | Zero z | z std | Effective rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| affine_next | 01:31:16 | 7.62661 | 7.48796 | 7.29332 | 257.461 | 138.384 | 0.76441 | 41.158 |
| affine_pair | 01:31:47 | 7.55537 | 7.40901 | 7.25075 | 125.938 | 73.759 | 0.77319 | 38.667 |
| poe_joint | 00:22:08 | 8.27856 | 8.41344 | 8.21570 | 138.462 | 417.232 | 0.21757 | 48.320 |
| poe_joint_wide | 00:31:15 | 6.96711 | 7.06244 | 6.87854 | 126.720 | 417.232 | 0.02910 | 60.151 |
| poe_factored | 01:32:56 | 7.58133 | 7.43829 | 7.28290 | 233.626 | 417.232 | 0.73439 | 38.908 |
| poe_modulated | 00:58:33 | 8.21434 | 8.29076 | 8.11321 | 346.005 | 417.232 | 0.04751 | 50.889 |

Regular NTP uses the trainer evaluation batches and random noise levels.
The paired probe uses the separate fixed 1,024-example set per split and
all eight noise levels. Compare arms within one column, not the two
evaluation estimators against each other. Shuffled and zero columns refer
to the held-out paired probe.

Observations: factored no-bias and original affine have almost identical
paired loss (7.28290 versus 7.29332); adding source input to the affine
mu does not degrade this run (7.25075). The wide pure joint form has lower
paired loss (6.87854) than the affine control, so the clean linear command
form is not ruled out by predictive fit. Modulation alone does not recover
the affine loss (8.11321). These one-seed results concern achieved fit under
the chosen training budgets, not guaranteed expressivity, semantic
composition, or tracker quality. The wide head has a notably smaller latent
scale; its effective rank is scale-normalized.
