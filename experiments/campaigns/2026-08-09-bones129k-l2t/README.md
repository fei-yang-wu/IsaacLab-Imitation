# 2026-08-09 — BONES-SEED 129k L2T

This campaign trains one learning-to-track (L2T) controller on all 129,785
BONES-SEED motions. The privileged teacher receives the explicit reference
command and full critic state. The deployable student receives the frozen
DiffSR latent command and proprioception. It learns the action that the teacher
executes on the same rollout.

The shared contract is `Isaac-Imitation-G1-v2`, Newton/MJWarp, seed 0, 16,384
environments x 24 rollout steps, minibatch 294,912, gamma 0.97,
`random80_adaptive20` resets, and the tuned v2 rewards. The required run has a
10B-frame cap. ICE has a 16-hour allocation limit, and all checkpoints write to
persistent `/data` every 50M frames.

The frozen encoder is the BONES-SEED root-qpos encoder at
`/data/bones129k_anchor_frame/expert_heading_encoder/checkpoints/latest.pt`,
SHA-256 `be6d533f1d1ca4aa6b1e819af1d3ef63eb033125018c8309c7448384b6a9583e`.
Its contract is expert-heading anchor, stride 1, horizon 10, z256, and 380
encoder inputs. The student command stays 258 values: z256 plus sine/cosine
phase.

W&B project: `g1-bones-seed`. Group: `l2t`.

## Runs

ICE job `5573723` started on an H200 on 2026-08-09. It was incorrectly
submitted with a 1B-frame cap instead of the required 10B-frame cap. It
completed 1,000,341,504 frames at iteration 2,544 with exit code 0. W&B run
`2znme7lg` ended with student loss 0.2205 and student RMSE 0.4696. This run is
an incomplete 1B qualification and is not the requested 10B result. It was not
used as a resume point.
The workspace archive SHA-256 is
`bc3c4f66853c55b49f2f4bdbccf87d756854c6827c8b57fc15f286680ff74717`.
The complete submission record is in `cluster_submission.json`.

The corrected fresh run is ICE job `5574140`. It started on an H200 on
2026-08-09 with 25,432 iterations: 10,000,269,312 actual frames for the 10B
cap. W&B run `ycmodfu3` is in project/group `g1-bones-seed` / `l2t`. It
writes to the new persistent path
`/data/bones129k_l2t_10b/l2t_teacher_explicit_student_latent/rlopt_train`.
Its workspace archive SHA-256 is
`d153c3d784ae322a0d7790b40cd3005e9d2f5b0bc7352c417e0165bd2e3e9449`.
The complete corrected submission record is in
`cluster_submission_10b.json`.

```bash
MODE=print ./experiments/campaigns/2026-08-09-bones129k-l2t/run.sh
MODE=smoke ./experiments/campaigns/2026-08-09-bones129k-l2t/run.sh
MODE=validate LOCAL_SMOKE_ROOT=<smoke-dir> \
  ./experiments/campaigns/2026-08-09-bones129k-l2t/run.sh
MODE=submit LOCAL_SMOKE_ROOT=<smoke-dir> CONFIRM_SUBMIT=l2t \
  ./experiments/campaigns/2026-08-09-bones129k-l2t/run.sh
```

The submit gate verifies the local smoke source hash, full remote reference
identity, exact encoder SHA-256 and interface metadata, a fresh persistent
output path, and the explicit W&B group confirmation. The launcher uses a
self-contained workspace archive so the dirty RLOpt L2T implementation is in
the job snapshot.

## One-billion-frame student evaluation

The final checkpoint from the incomplete 1B qualification was downloaded to
`logs/downloaded_checkpoints/bones129k_l2t_1b/model_step_1000341504.pt`.
Its SHA-256 is
`0eee920744d854a294c94e8d209314527c6296d41cb1f01658dd277241eff293`.
Checkpoint metadata declares `IPMD_L2T` with the student as the primary policy;
deployment evaluation therefore uses ordinary `IPMD` with
`rlopt_ipmd_tuned_cfg_entry_point` and loads only the 351-input student policy.

A fresh selected-ten manifest was prepared at
`data/bones_seed_language10_l2t_eval_v1/manifests/`
`g1_bones_seed_language10_l2t_eval_v1_manifest.json`, SHA-256
`0625849fb1b895c0a381449a4d3958f29b064879637429338be9a31f78f6cfb1`.
It contains the canonical ordered language10 motions and uses its own small
Zarr cache.

The SONIC-compatible pass used one environment per selected motion, frame-0
starts, seed 0, mode actions, startup/reset randomization, no push, and the
released 0.25 m / 1 rad / 0.25 m termination criterion. It ended with
`all_envs_done` after 287 steps, with 0/10 completed motions. Termination events
were `ee_body_pos=8`, `anchor_ori=2`, and `anchor_pos=1`; one motion fired two
terms on the same step. The result is in
`logs/bones129k_l2t_1b_selected10_eval/sonic_eval.json`.

The required non-terminating diagnostic rendered all ten motions for 5,137
total control transitions. Nine motions fell below the 0.4 m diagnostic root
height; only `Neutral_stoop_down_001_A057` stayed upright. Step-weighted
pre-fall MPJPE-L was 135.98 mm. Including the required post-fall frames,
full-horizon errors were 328.75 mm MPJPE-L, 3.210 m MPJPE-G, and 3.172 m EE
XYZ. Per-motion metrics and retained videos are under
`logs/bones129k_l2t_1b_selected10_eval/`
`nonterminating_videos_randomized_no_push/`.

## One-billion-frame teacher evaluation

The same checkpoint's privileged 286-input teacher was evaluated on the same
ordered motions. The evaluator loaded the full IPMD-L2T checkpoint with
`--ipmd_l2t_policy_role teacher`; this preserves the teacher's explicit
full-body reference and privileged robot-state inputs. The teacher is a
training ceiling and is not deployable because those inputs include the live
reference.

The SONIC-compatible pass ended with `all_envs_done` after 1,055 steps. All
10/10 motions completed, no tracking-failure term fired, and success-only
MPJPE-L was 14.71 mm. The result is in
`logs/bones129k_l2t_1b_selected10_teacher_eval/sonic_eval.json`.

The separate non-terminating diagnostic retained one video per motion and
covered all 5,137 transitions. No motion fell below 0.4 m. Frame-weighted
full-horizon errors were 14.00 mm MPJPE-L, 138.59 mm MPJPE-G, 0.140 m EE XYZ,
and 0.0926 rad joint-position MAE. The teacher artifacts are under
`logs/bones129k_l2t_1b_selected10_teacher_eval/`
`nonterminating_videos_randomized_no_push/`.

This isolates the 1B result: the teacher is already strong, while the student
completed 0/10 motions. The failure is in student distillation or its latent
interface, not in the teacher policy.

## Latest complete checkpoint evaluation: 5.65B frames

The corrected 10B job was still running when this evaluation started. The
latest complete checkpoint was therefore used:
`logs/downloaded_checkpoints/bones129k_l2t_10b/model_step_5650120704.pt`
(5,650,120,704 frames), SHA-256
`5d92a8b12843bf35a66e6242208914d5990a4639b43d4f8757dfdfd6e48b0f0b`.
This is an interim snapshot, not the timeout endpoint. The frozen
`expert_heading` encoder is the verified matching 380-input, stride-1 encoder.

Both roles used the same SONIC-compatible protocol as the 1B pass: task v2,
Newton, seed 0, frame-0 sequential starts, mode actions, startup/reset
randomization, no push, and the released 0.25 m / 1 rad / 0.25 m terms.

### Selected ten

| Role | Input contract | Completed | Success-only MPJPE-L |
| --- | --- | ---: | ---: |
| Student (deployable) | 351-D actor: 258-D latent plus phase and proprioception | 0/10 | not defined |
| Teacher (privileged ceiling) | 286-D explicit reference plus privileged robot state | 10/10 | 12.31 mm |

The student failed with `anchor_ori=4` and `ee_body_pos=6`. The teacher had no
tracking-failure terms and ended all ten motions through `reference_finished`.
Results are in
`logs/bones129k_l2t_10b_5650m_selected10_{student,teacher}_eval/sonic_eval.json`.

### 4,096-motion scoreboard

The fixed rank block is 12,288--16,383. Both passes completed all environments,
had zero timeouts, and used rank SHA-256
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`.

| Role | Completed | Success-only MPJPE-L | Failure terms |
| --- | ---: | ---: | --- |
| Student (deployable) | 2,873/4,096 (0.7014) | 47.94 mm | `ee_body_pos=892`, `anchor_ori=319`, `anchor_pos=53` |
| Teacher (privileged ceiling) | 3,893/4,096 (0.9504) | 17.83 mm | `ee_body_pos=165`, `anchor_ori=32`, `anchor_pos=13` |

The scoreboard artifacts are in
`logs/bones129k_l2t_10b_5650m_scoreboard4096_{student,teacher}/`.
The teacher remains a training ceiling: it consumes the live explicit
reference and privileged state, so its score is not a deployable policy result.

## Newer interim checkpoint evaluation: 5.80B frames

ICE job `5574140` was still `RUNNING` during this pass (`12:41:43` elapsed of
the `15:59:00` limit). It has not reached the timeout endpoint. The newest
complete checkpoint used here was
`logs/downloaded_checkpoints/bones129k_l2t_10b/model_step_5800329216.pt`
(5,800,329,216 frames), SHA-256
`cb990686240d67bf0add995268bc49f85bdf4d577a3b4ea87cd88dde473d2d49`.

The frozen `expert_heading` encoder remains tensor-identical to the encoder
embedded in this checkpoint. Both roles used the same SONIC-compatible
protocol, selected-ten ranks `0--9`, and the fixed 4,096-motion rank block
`12,288--16,383`. The selected-ten rank SHA-256 is
`9c60d3f5d54fd3a81c3e77aa3dba60ff518b2e7cea69c8b18a849d2938ad4f12`; the
scoreboard rank SHA-256 is
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`.

### Selected ten

| Role | Completed | Success-only MPJPE-L |
| --- | ---: | ---: |
| Student (deployable) | 0/10 | not defined |
| Teacher (privileged ceiling) | 10/10 | 12.74 mm |

The student stopped with `anchor_pos=3`, `anchor_ori=1`, and `ee_body_pos=6`.
The teacher ended all ten motions through `reference_finished` with no
tracking-failure term.

### 4,096-motion scoreboard

| Role | Completed | Success-only MPJPE-L | Failure terms |
| --- | ---: | ---: | --- |
| Student (deployable) | 2,851/4,096 (0.6960) | 47.71 mm | `ee_body_pos=954`, `anchor_ori=270`, `anchor_pos=53` |
| Teacher (privileged ceiling) | 3,904/4,096 (0.9531) | 17.92 mm | `ee_body_pos=159`, `anchor_ori=27`, `anchor_pos=15` |

The detailed artifacts are in
`logs/bones129k_l2t_10b_5800m_selected10_{student,teacher}_eval/` and
`logs/bones129k_l2t_10b_5800m_scoreboard4096_{student,teacher}/`. These are
interim results. They must not be described as the final 10B result until the
ICE job reaches its terminal state.
