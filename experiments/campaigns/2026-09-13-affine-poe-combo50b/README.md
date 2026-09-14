# Exact affine PoE reformulation with fresh combo-50B training

Seed0, fresh encoder pretraining followed by a fresh 50B-frame tracker.
The encoder reference is combo50B's actual p5_affine encoder configuration,
retrieved from ICE and saved under logs/affine_poe_combo50/original_encoder_config.yaml.

## Model and protocol

Use phi=[1; z] and E=[b,A]^T F(s) mu(y_t,t) / sqrt(embed_dim).
A,b,F,mu and all parameter names/shapes match the existing affine model.
Contract [b,A] with F before mu to avoid the larger embed-by-target tensor.
This preserves the predictor and derivatives mathematically; floating-point
reassociation means training trajectories need not be bitwise identical.
It is a reformulation, not an additional expressivity hypothesis.

The policy commands z64 plus sin/cos phase (66 coordinates); the fixed one
is only inside the diffusion model. Internal feature width stays256,
embedding width1024. Encoder hidden widths [2048,1024,512,512], SiLU;
F hidden widths [512,512]; g configuration [1024,1024,512] with the affine
linear layer active; mu widths [1024,1024,512], target-only conditioning.
Pretrain 50000 updates, batch8192, LR0.0003 for encoder/NTP; SIGReg1,
latent L2 0.001, past-five plus current, horizon10, intermediate window,
boundary_next merged diffusion head, endpoint coefficient0, LayerNorm.
Same BONES-SEED129785 data, root_qpos input, stride1 and heading anchor.
No direct-joint network or wider/deeper architecture is introduced.

Tracker arguments match combo50B stage-for-stage except output identity and
fresh encoder binding: 16384 environments, 24-step rollouts, 50B target,
10-step actor history, single-frame critic, weight decay0.01, linear critic
decay to1e-5 over50B, hold1 and phase, same reward/network settings.
SONIC resets: uniform share0.8->0.5 in first stage,0.5->0.2 in second,
then0.2. Each ramp spans4B local stage frames, matching combo50B.

One pretrain plus seven tracker H200 allocations, each16CPU/160G/15:59.
The first tracker requires successful pretraining; later allocations use
afterany and persistent resume with the SAME full50B target. Allocation
count is walltime capacity, not a guarantee of completion if jobs fail.
W&B project g1-bs-pareto, group affine-poe-combo50b.
Output root /storage/ice-shared/vip-vwt/scratch-fwu91/affine_poe_combo50b/affine_poe_seed0/.
Existing campaigns and canceled arms are not modified or restarted.

## Qualification

13 focused model tests passed, including strict affine state-dict loading,
matching predictions, input/parameter gradients, diffusion loss and seeded
sampling. PyTorch import-time deprecation warnings were filtered for this
run; no runtime failure was suppressed. Seven-stage tracker argument parity
against combo50B passed. Smoke artifacts: logs/affine_poe_combo50/smoke/.
Implementation changes are in the RLOpt submodule and the parent pretrain
CLI. Existing unrelated submodule work is preserved; no commit or parent
submodule pointer update has been made. Submitted source archive is the
runtime provenance.

## Submitted 2026-09-13 17:01 UTC

Pretrain **5772121**, then tracker jobs **5772122 through 5772128**.
Plan `affine-poe-combo50b-affine_poe-s0-20260913-165946-ca4aa6b3`;
SHA `ca4aa6b355f8c1810f380aa36469c5c7e6a10042b3594b3affc0a8e7321d6b2e`.
Verified source archive `9abb4621c386aba6ea711eff67deac7eed52bb2150742120617b9bd29027953c`.

Full-batch two-update pretrain passed; checkpoint contains affine_poe,
z64, internal feature256 and trained NTP state. Every original recorded
trainer field matches except the new parameterization and shortened smoke
budget. One low-level update (32 environments) passed with the fresh smoke
checkpoint. Eight cluster preflight checks passed. No drift override used.
