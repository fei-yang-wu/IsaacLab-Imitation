# BONES-SEED language-goal candidate screen

This campaign replaces the historical hand-picked `demo8`/`local10` goal sets
with a tracking-qualified ten-motion set for one shared language-conditioned
planner. The 30 candidates deliberately cover object manipulation, walking
styles and directions, gestures, raised hands, locomotion, and everyday
actions with concrete language descriptions.

The screen is one Isaac process, not 30 launches. It loads the complete
129,785-motion reference arrays once and pins an equal stable block of 128
environments to each candidate rank: 3,840 environments total. Each motion
therefore gets an independent 128-rollout success rate under startup/reset
randomization, deterministic policy actions, no push event, and the official
SONIC motion-success terminations. The raw output also retains transition-
weighted local and global tracking errors for every environment.

The initial controller is the rollout-24, gamma-0.97 checkpoint at 3.5B frames,
paired with its exact 380-wide BONES root-qpos encoder. This checkpoint scored
92.6% SONIC SR and 22.64 mm success-only MPJPE-L on the earlier 4,096-motion
screen. Re-run the candidate screen against a later checkpoint by overriding
`CHECKPOINT`, after intentionally updating its pinned hash in `run.sh`.

```bash
MODE=print experiments/campaigns/2026-08-05-bones-language10-screen/run.sh
MODE=screen experiments/campaigns/2026-08-05-bones-language10-screen/run.sh
```

Render a full-horizon side-by-side oracle-command video for each of the 30
candidates with deterministic policy actions, startup/reset randomization, no
push event, and every early termination disabled:

```bash
experiments/campaigns/2026-08-05-bones-language10-screen/render_candidate_oracle_videos.sh
```

The launcher resumes safely with `--skip_existing` and prints every retained
video's absolute path. These videos qualify the frozen low-level tracker and
support later semantic phase annotation; they are not planner rollouts.

Rank primarily by per-motion completion SR, then MPJPE-L and MPJPE-G. The final
ten are selected from the high-performing region with an explicit diversity
constraint; ten near-duplicate walk or object-transfer clips are not a useful
language-conditioning demonstration even if they occupy the numerical top ten.

This campaign fixes every planner collection and evaluation episode to
reference frame 0. The older Phase-5 random 0--200 start protocol does not
apply to this experiment, so candidate clip length is not a selection proxy.

The selected-ten planner now uses the trajectory-first protocol in
[`../2026-08-05-bones-language10-oracle-pretrain/README.md`](../2026-08-05-bones-language10-oracle-pretrain/README.md).
Collect 100 complete oracle-policy trajectories per motion in one 1,000-env
process, from frame 0 until SONIC failure or `reference_finished`. Disable foot
XYZ termination and pushes, keep the other domain randomization, and use
deterministic policy actions. Oracle-only pretraining and milestone evaluation
come before any planner-driven collection. Motion rank remains a supervised
label only; deployment always supplies an explicit language goal.

## Screening result and selected ten

The 2026-08-05 screen completed in one 69-second process. All 3,840 environments
finished their clips (`done_rate=1`, `time_out_rate=0`,
`stop_reason=all_envs_done`); aggregate candidate SR was 90.47%. Twenty-four of
the 30 candidates reached 100% SR, so semantic/action diversity, language
directability from frame 0, and tracking accuracy—not tiny metric differences
alone—determined the final set.

| motion | language goal | frames | SR | MPJPE-L | MPJPE-G |
|---|---|---:|---:|---:|---:|
| `Neutral_stoop_down_001_A057` | Walk forward, pick up an object from the ground, examine it, drop it, and continue walking. | 340 | 1.000 | 23.21 mm | 0.234 m |
| `lift_crate_walk_ff_start_180_R_001_A140` | Lift a heavy crate with both hands and start walking forward. | 247 | 1.000 | 21.13 mm | 0.148 m |
| `drinking_standing_mug_R_001_A282` | While standing, lift a mug with your right hand, drink from it, and lower it. | 551 | 1.000 | 11.21 mm | 0.069 m |
| `fishing_standing_loop_R_001_A500` | Stand and fish, occasionally checking the rod for a catch. | 1055 | 1.000 | 12.36 mm | 0.070 m |
| `cellphone_typing_sequence_one_hand_idle_R_001_A423` | Take out a phone, type on it with your right hand, then put it away. | 814 | 1.000 | 14.15 mm | 0.205 m |
| `feeding_birds_start_R_001_A456` | Feed the birds by repeatedly scattering seeds with your right hand. | 792 | 1.000 | 18.25 mm | 0.127 m |
| `walk_arc_cw_start_R_slow_001_A443` | Start walking slowly in a clockwise, rightward arc. | 467 | 1.000 | 18.26 mm | 0.118 m |
| `mosquito_drive_away_R_001_A500` | Raise your right hand and swat side to side to drive away a mosquito. | 309 | 1.000 | 13.67 mm | 0.075 m |
| `casual_greeting_R_001_A428` | Raise your right hand in a casual greeting. | 203 | 1.000 | 15.91 mm | 0.172 m |
| `surrender_stop_R_001_A468` | Lower both raised hands to your sides, ending a surrender pose. | 369 | 1.000 | 19.13 mm | 0.088 m |

Across the selected ten, mean SR is 1.000, mean MPJPE-L is 16.73 mm, and mean
MPJPE-G is 0.131 m. The machine-readable frozen selection is `selected10.json`.

The rejected performance outliers are useful negative controls but poor first
language demos: `panic_run_away` scored 0% SR, `walk_big_dog` 8.6%, `rock_out`
21.9%, and `jump_around` 93.8% with 0.763 m MPJPE-G.

The v2 collection port and oracle-pretrain front door are complete. A real
end-to-end smoke collected all ten trajectories at 100% oracle SONIC SR, wrote
the 30-frame root-qpos lookahead contract, trained a medium planner for 20
updates, and evaluated both 10- and 20-update checkpoints against every named
goal. Those tiny planner results are a wiring gate, not a performance result.

## Task-active replacement ten (2026-08-06)

The completed planner comparison showed that low-level tracking qualification
alone is insufficient for language-planner selection. A second selection,
frozen as `selected10_taskactive_v2.json`, adds an explicit anti-holding gate
and favors actions that can be placed in a concrete task context.

A **hold frame** is a frame with root linear speed below 0.03 m/s, root angular
speed below 0.10 rad/s, and joint RMS speed below 0.15 rad/s. Hold fraction
counts only continuous segments lasting at least 0.20 seconds. Reject a motion
when more than 20% of its frames are holds or any one hold exceeds 1.50 seconds.
This is a selection heuristic, not a SONIC metric. The preparation code now
enforces the declared limits whenever a selection contains `hold_screen`.

| motion | role or task context | low-level SR | hold fraction | longest hold | prior planner evidence |
|---|---|---:|---:|---:|---|
| `Neutral_stoop_down_001_A057` | retrieve a ground object while walking | 1.000 | 6.7% | 0.46 s | conditional, best SR 0.40 |
| `lift_crate_walk_ff_start_180_R_001_A140` | lift and carry a crate | 1.000 | 0.0% | 0.00 s | robust, best SR 1.00 |
| `feeding_birds_start_R_001_A456` | repeated directed scattering | 1.000 | 7.4% | 1.18 s | conditional, best SR 0.77 |
| `walk_arc_cw_start_R_slow_001_A443` | direction/curvature-conditioned walking | 1.000 | 0.0% | 0.00 s | robust, best SR 1.00 |
| `mosquito_drive_away_R_001_A500` | reactive directed hand gesture | 1.000 | 8.7% | 0.30 s | conditional, best SR 0.51 |
| `walk_ff_loop_180_R_slow_001_A443` | speed/direction-conditioned walking | 1.000 | 0.0% | 0.00 s | not yet planner-trained |
| `medium_big_light_one_hand_walk_ff_start_360_R_001_A504` | one-hand carrying while walking backward | 1.000 | 0.0% | 0.00 s | not yet planner-trained |
| `inside_door_handle_right_side_open_walk_turn_close_R_001_A514` | open, traverse, and close a door | 1.000 | 0.0% | 0.04 s | not yet planner-trained |
| `injured_torso_walk_ff_start_315_R_001_A214` | style/direction-conditioned walking | 1.000 | 0.0% | 0.00 s | not yet planner-trained |
| `big_heavy_one_hand_front_high_to_front_low_R_001_A524` | transfer a heavy object between heights | 1.000 | 3.6% | 0.30 s | not yet planner-trained |

The new set retains five previously evaluated motions and adds five active
replacements. It removes drinking, fishing, phone typing, casual greeting, and
surrender. Fishing is removed despite 0.99--1.00 planner SR because 38.9% of
its clip is held and its longest uninterrupted hold is 4.44 seconds. The other
four have both poor planner SR and large hold fractions.

This is a frozen selection, not a completed planner result. The five replacement
motions have 1.0 low-level SONIC SR but still require a fresh ten-motion data
collection, planner fit, and per-motion evaluation. Do not reuse the v1 sample
set, language table, planner checkpoint, or reference-array identity for that
run.
