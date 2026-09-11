# Action01 continuation: another 10B frames

Continue the completed action01 5B fine-tune from checkpoint
`model_step_55000301568.pt`. Its saved `cumulative_env_frames` was verified
locally as 55,000,301,568. Requested target 65,000,301,568 rounds to
65,000,570,880 with 16,384 environments and 24-step rollouts, adding
10,000,269,312 frames.

The training contract is inherited from the action01 arm in
`../2026-09-08-combo50b-smooth-ft5b/campaign.yaml`: action-rate penalty -0.1,
original anti-shake penalty, no action EMA, legacy normalization updates,
p5_affine z64 plus phase, ten-step actor history, original MLP, EE/wide rewards,
fullbatch three epochs, fixed 20% uniform reset mix, and loaded optimizer state.
No encoder training or curriculum restart. Checkpoints every 0.5B frames.

Two H200 segments of 15:59 each, 16 CPUs and 160G RAM; the second resumes
with afterany dependency and the same cumulative cap. Persistent output:
`/data/combo_action01_continue10b/action01_seed0/tracker`.
W&B retains the action01 campaign group and run identity for continuation.

All remote checkpoint, dataset, storage and container preflight checks passed.
Plan SHA: `dea0d48af0bd887a98b667dd30c303e2948621c968cd21c14ae2cea14ffe85c9`.

Submitted 2026-09-09: first segment `5747620`, afterany continuation `5747621`.
Source archive SHA256:
`2eba0e9868c0edc5770d4667cb27a70cee418d633ad1cf0bf0cd515e61f49d71`.
Submission record: `submission-20260909-122727.json` under the plan directory.
