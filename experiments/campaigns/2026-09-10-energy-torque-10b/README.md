# 2026-09-10 energy / torque penalty fine-tunes (10B, two hubs)

Four arms, each 10B frames past its hub checkpoint, one variable each
(see `campaign.yaml` header): `e1` / `e5` add SONIC's whole-body
mechanical-power penalty `energy_consumption` at -1.0e-4 to the action01 55.5B
hub and the action01rate05 60B hub; `t1` / `t5` add `joint_torques_l2` at
-1.0e-5 (our term, not SONIC's) to the same hubs. W&B group
`combo50b-smooth-ablation`. Submitted 2026-09-11 02:4x UTC: e1 5757020/5757021,
e5 5757023/5757025, t1 5757036/5757038, t5 5757055/5757056.

Why: `wiki/sonic-training-parity-2026-09-10.md` — every SONIC reward term is
in our recipe at SONIC's weight except the energy term, and the plant/hardware
evidence (`external/Embodied-Control/docs/evidence/noise_20260910`) puts the
ankle/knee "stiffness" in the commanded targets, not the PD gains.

Regular evaluation, two channels:

- `submit_live_eval.sh`: newest checkpoint of each arm on `bones_testbed4096_v1`
  (clean row) through the `latest-eval` campaign arms `energy_<arm>_live`;
  rows land as `latest_eval/energy_<arm>_seed0_clean_f<frames>.json`.
- `ec_grade_live.sh`: downloads the newest checkpoint, exports a `combo_v3`
  hold-1 bundle, verifies it, runs the Embodied-Control async lifecycle
  rehearsal on the MuJoCo plant with the robot's measured sensor noise (video
  per motion under `external/Embodied-Control/artifacts/energy_<arm>_f<frames>_measured/<motion>/sim/seed_0/video.mp4`)
  and the foot-dither statistic; one row per checkpoint in
  `logs/energy_torque_ec/results.tsv`. `ARMS="hub_action01 hub_rate05"` grades
  the hubs through the same runtime for the baseline rows. Sequential, locked:
  a rehearsal never shares the plant lanes with another run.

The EC runtime these rows use is the 2026-09-10 one: joint targets are NOT
clamped to the soft joint limits, plant joint limits are Isaac-stiff, and the
first-action distance gate is off (`docs/evidence/isaac_replay_20260910` in
Embodied-Control). Rows from before that change are not comparable.
