# SONIC frame parity: `expert_macro_anchor_mode=robot_heading`

**Status (2026-08-13): local qualification only.** No cluster arm is submitted
and no number here is a measurement. ICE was down when this was written; the
arm that measures the mode is deferred.

## What the mode is

`env.expert_macro_anchor_mode` selects the frame the **DiffSR macro window** is
expressed in. The macro window is the reference window the skill encoder reads;
its output is the latent command the frozen low-level tracker follows. The knob
touches nothing else — not the actor's explicit command terms, not the critic,
not the rewards, not the metrics.

Upstream `gear_sonic` has exactly three canonicalizations of that window's
anchor orientation, and ships two of them:

| upstream term | frame | our mode |
| --- | --- | --- |
| `motion_anchor_ori_b_mf` | `inv(LIVE robot FULL quat) · ref` | `robot` (rollout half) |
| `motion_anchor_ori_heading_mf` | `inv(LIVE robot HEADING quat) · ref` | `robot_heading` (new) |
| `motion_anchor_ori_refheading_mf` | `inv(REFERENCE first-frame heading) · ref` | `expert_heading` |

The released SONIC checkpoint reads `b_mf`. SONIC v1.1 — the tracker behind the
25.41 mm oracle row of `2026-08-12-gr00t-language30-compositionality` — reads
`heading_mf`. No `gear_sonic` config selects `refheading`, so `expert_heading`
reproduced the one upstream variant nobody ships, and until 2026-08-13 the repo
had **no** mode for the live-robot heading frame at all.

`robot_heading` anchors the live window at
`heading_anchor_frame(live robot anchor)`: yaw-only twist, xy-only origin. The
reference keeps absolute height and its tilt relative to gravity, and the
encoder input carries the live tracking error. Pretraining has no robot, so it
keeps the expert slot-0 heading frame — the same frame in the perfect-tracking
limit, and the closest offline analogue.

## Why a frozen encoder makes this a hard error, not a preference

The macro state is **380 wide in every mode**. An encoder pretrained under one
frame and driven under another produces no shape error, only a silently
off-distribution command. So the mode is recorded into the skill checkpoint at
pretrain time and compared at low-level startup, exactly like
`macro_frame_stride`. No existing encoder can be reused under a new mode.

## What the local script does

```bash
bash experiments/campaigns/2026-08-13-sonic-frame-parity/run_local_qualification.sh
```

Four gates, in order:

1. **Pretrain accepts and records the mode.** One update at batch 8, then the
   checkpoint is read back: `macro_anchor_mode == robot_heading`,
   `macro_frame_stride == 1`, encoder input width 380.
2. **The live path runs.** One low-level iteration on `Isaac-Imitation-G1-v2`
   with `physics=newton_mjwarp`, 4 envs. This is the only gate that exercises
   the new `robot_heading` rollout context inside Isaac, where the macro window
   is rebuilt from the live robot anchor's heading frame every control step.
3. **`robot` env refuses the encoder.**
4. **`expert_heading` env refuses the encoder.**

Gates 3 and 4 are the point: they are the only evidence the pairing guard is
wired, since 380 is 380 in all three modes. Each writes `status.json` with the
source-contract hash of every file the contract depends on, so a later cluster
submit can tell whether the qualification is stale.

Output lands in `logs/sonic_frame_parity_qualification/<timestamp>/`.

The pretrain and tracker contracts reproduce the
`2026-08-08-bones129k-anchor-frame` arm exactly except for the anchor mode
(root-qpos macro 380, stride 1, `horizon_steps=10`, z256 + sin/cos phase = 258,
hold 10, tuned tracker capacity, endpoint objective), so the deferred cluster
arm is single-variable against that campaign's `robot` control, ICE 5567801.

## Deferred: the cluster arm

Not submitted. When ICE is back, the arm is a fresh encoder pretrain under
`robot_heading` plus an `afterok` tracker on it, W&B
`g1-bones-seed` / group `sonic-frame-parity` (user-confirmed 2026-08-13), same
frame cap and geometry as the 08-08 campaign. Until that runs, nothing here
says whether the frame matters — only that the code path works and the guard
holds.

## Related

- Code: `heading_anchor_frame` in
  `source/isaaclab_imitation/.../mdp/_compiled.py`; the `robot_heading` context
  in `envs/expert_data_plane.py`; the cfg field in
  `.../config/g1/common/tracking_env.py`; the guard in
  `RLOpt/rlopt/agent/hl_skill_diffsr.py`.
- Unit contract: `source/isaaclab_imitation/tests/test_expert_heading_anchor.py`
  (in `pixi run -e isaaclab test-isaaclab`).
- Prior arm on the same axis: `experiments/campaigns/2026-08-08-bones129k-anchor-frame/`.
