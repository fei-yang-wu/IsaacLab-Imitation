# 2026-08-27 — Proprioceptive history on the generative-NTP encoders

Ten-step history on every actor observation term, from scratch, on the two
round-4 winners, with the same `action_rate_l2` -0.03 penalty as the 50B
promotion chains.

## Arms (seed 0, W&B group `diffntp-history`)

| arm | encoder | jobs |
|---|---|---|
| `diffntp_chunk_hist` | `diffntp_chunk_h1_ee_wide` | 5592715 |
| `diffntp_pair_hist` | `diffntp_pair_h1_ee_wide` | 5592716 |

Change against the 50B chains' `std1` stage: `history_length=10` on
`projected_gravity`, `base_ang_vel`, `joint_pos_rel`, `joint_vel_rel`, and
`last_action`. The actor input widens; the command channel is unchanged. This
is the `smooth_block` history set from the 2026-08-15 bottleneck campaign,
never before run with a generative-NTP encoder.

## Budget: ONE segment, cap not reached by design

Each arm is a single 15:59 segment carrying the full 20B frame target. The
walltime ends it far short of the cap — expected and accepted by the user
(2026-08-27). Each arm reports whatever depth one segment buys (~7B at
~120k fps).

DO NOT read these against the 50B chains as a budget axis: they differ in
frames AND stop mid-schedule, before the `sonic` reset switch. The honest
comparison is against the 50B chains' own checkpoint at the same frame count,
inside the `random80_adaptive20` phase.

## Status

- 2026-08-27 02:52: SUBMITTED seed 0, jobs 5592715 and 5592716. Nothing
  measured.
