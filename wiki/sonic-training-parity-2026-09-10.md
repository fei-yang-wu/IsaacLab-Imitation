# SONIC vs. our tracker: training-setup comparison (2026-09-10)

Field-by-field comparison of the released SONIC v1.1 training configuration
(`/mnt/hsstorage/fwu91/sonic_v1_1/config.yaml`, Hydra dump of
`sonic_release_3pt_heading_wrist_81`, 128 GPUs) against the recipe our
combo / action01 / rate arms use (`Isaac-Imitation-G1-v2` with
`--agent rlopt_ipmd_tuned_fullbatch_cfg_entry_point` and the campaign
overrides in `experiments/campaigns/2026-09-10-smooth-ablation-55b/campaign.yaml`).
"Same" means the value was read from both sources; "verify" means one side
could not be read from the files on this machine.

## 1. Simulation and episode

| field | SONIC | ours | note |
|---|---|---|---|
| physics dt x decimation | 0.005 x 4 (50 Hz) | 0.005 x 4 | same |
| episode length | 10 s | 10 s | same |
| terrain | `trimesh` (`force_flat_terrain: false`) | `plane` | **differs**; the released trimesh generator is not in the dump (verify whether it is flat) |
| robot actuators | `g1_model_12_dex`, hip pitch on 7520-22 | `UNITREE_G1_29DOF_SONIC_CFG` (same table) | same |
| action term | `JointPositionActionCfg`, `use_default_offset` | same term | same |
| action scale | not in the yaml (term default 1.0) | 0.25 x effort / kp per joint | **verify**: our scale is "induced by SONIC's released actuator config"; the training yaml itself carries no `scale` key |
| raw action clip | `action_clip_value: 20.0` | none | minor at our action magnitudes (rms 1.5) |

## 2. Domain randomization (events)

| event | SONIC | ours (`G1SonicEventCfg`) |
|---|---|---|
| friction (startup) | static 0.3-1.6, dynamic 0.3-1.2, restitution 0-0.5, 64 buckets | same |
| joint default pos | +-0.01 rad add | same |
| torso COM | x +-0.025, y +-0.05, z +-0.05 | same |
| mass scale | wrists + torso, 0.8-2.5 | same |
| push | every 4-6 s, lin +-0.5 / +-0.5 / +-0.2, ang +-0.52 / +-0.52 / +-0.78 | same |
| reset pose noise | x,y +-0.05, z +-0.01, roll/pitch +-0.1, yaw +-0.2 | same |
| reset velocity noise | same ranges as the push | same |
| reset joint noise | pos +-0.1, vel 0 | same |
| actuator gain randomization | none | none |
| `init_at_random_ep_len` | true | reference start frame random (0-200 at eval) |

Nothing on this axis explains a behaviour difference.

## 3. Observations

| group / term | SONIC | ours |
|---|---|---|
| actor proprio | gravity dir, base ang vel, joint pos rel, joint vel rel, last action; history 10 each | same five, history 10 (campaign override) |
| actor noise (uniform) | gravity +-0.05, ang vel +-0.2, joint pos +-0.01, joint vel +-0.5, action none | same values (`G1V2ObservationCfg`) |
| actor command | 64-D token (tokenizer online) | 64-D z from the frozen `p5_affine` encoder + sin/cos phase (constant at hold 1) |
| actor input normalization | `running_mean_std: false` (raw inputs) | `normalize_input: true`; frozen after the parent run (`update_normalizers_after_rollout=false`) |
| critic | privileged: future command, anchor pos/ori in body frame, body pos/ori, base lin vel, proprio history 10; `running_mean_std: true` | `SONIC_LATENT_CRITIC_INPUT_KEYS` (latent + expert window terms), normalized |
| command lookahead | `num_future_frames: 10` at `dt_future_ref_frames: 0.1` = **1.0 s** ahead | horizon 10 frames at stride 1 = **0.2 s** ahead |
| tokenizer input | joint qpos + qvel + anchor ori (heading frame), noise +-0.05 on orientations | `root_qpos` (29 qpos + anchor pos 3 + rot6d 6), pelvis, `robot_heading` |

The lookahead difference (1.0 s against 0.2 s) is the largest structural
difference on the input side.

## 4. Rewards

| term | SONIC | ours (`G1V2TunedRewardsCfg` + campaign) |
|---|---|---|
| anchor pos (std 0.3) | 0.5 | 0.5 |
| anchor ori (std 0.4) | 0.5 | 0.5 |
| relative body pos (std 0.3) | 1.0 | 1.0 |
| relative body ori (std 0.4) | 1.0 | 1.0 |
| body lin vel (std 1.0) | 1.0 | 1.0 |
| body ang vel (std 3.14) | 1.0 | 1.0 |
| VR 5-point local (std 0.1) | 2.0 | `tracking_reward_points` **4.0** (campaign) |
| wrist ori (std 0.1) | 0.4 | present |
| `motion_body_pos` (std 0.05) | -- | 2.0 |
| `motion_global_anchor_pos` (std 0.1) / `_wide` | -- | 2.0 / 1.0 (campaign) |
| `motion_global_anchor_ori` (std 0.15) | -- | 2.0 |
| `motion_ee_pos` | -- | 1.0 (campaign) |
| `action_rate_l2` | -0.1 | -0.1 (rate arms -0.2 ... -0.5) |
| `joint_pos_limits` | -10 | -10 |
| `undesired_contacts` (thr 1.0) | -0.1 | -0.1 |
| `anti_shake_ang_vel` (thr 1.5) | -0.005 | -0.005 |
| `feet_acc` (ankles) | -2.5e-6 | -2.5e-6 |
| `energy_consumption` (\|tau qdot\|, all joints) | **-1.0e-4** | **0.0** (e1/e5 arms turn it on) |
| `joint_torques_l2` | none | 0.0 (t1/t5 arms turn it on) |
| `action_acc_l2` | none | 0.0 (a2/c1 arms) |

Ours carries every SONIC term at SONIC's weight, doubles the keypoint term,
adds three global-anchor / body terms SONIC does not have (introduced for the
root-drift problem), and lacked the energy term until today's arms.

## 5. Terminations (training)

| term | SONIC | ours (`G1SonicTerminationsCfg`) |
|---|---|---|
| anchor pos | 0.15 adaptive, down 0.75, root height 0.5 | same |
| anchor ori | 0.2 | same |
| EE body pos (ankles, wrists) | 0.15 adaptive, down 0.75 | same |
| foot pos xyz | 0.2 | verify (the eval board disables it) |
| time out | tracking time-out | same |

## 6. Reference sampling and data

| field | SONIC | ours |
|---|---|---|
| dataset | `train_pyroki_v081_0414_26` mix + SMPL retarget (`smpl_retarget/1215`) | `bones_seed_sonic_full_129785` (the released SONIC clip set converted; no SMPL) |
| augmentation | upper-body augment on listed prefixes, `cat_upper_body_poses` p 0.5, freeze-frame aug, wrist pose randomization, three encoders (g1 / teleop / smpl) sampled 1:1:1 | none |
| adaptive sampling | bin 50, uniform 0.1, init failures 1, pre-failure window 200, max/mean 200 | bin 50, uniform **0.2**, init failures 1, window 200, max/mean **50** |
| tokenizer training | online, aux losses on the actor (`g1_recon` 0.01, latent losses 1.0) | frozen pretrained encoder (`hl_skill_finetune_enabled=false`) |

## 7. Optimizer (the largest gap)

| field | SONIC (`PPOIM`) | ours (`rlopt_ipmd_tuned_fullbatch`, local contract, `sonic_release_optimizer=False`) |
|---|---|---|
| envs x steps per iteration | 4096 x 24 x **128 GPUs** = 12.6M frames | 16384 x 24 = 393k frames |
| epochs x minibatches | 5 x 4 | 3 x 1 (full batch) |
| clip | 0.2 | 0.2 |
| gamma / lambda | 0.99 / 0.95 | **0.97** / 0.95 |
| entropy coef | 0.01 | **0.0** |
| actor lr | 2e-5, adaptive on KL 0.01 in [1e-5, 2e-4] | **1e-3**, adaptive on KL **0.02** per iteration in [1e-5, 1e-3] |
| critic lr | 1e-3 | 1e-3 constant (campaign), weight decay 1e-2 (campaign) |
| max grad norm | 0.1 | **1.0** |
| value loss | clipped, coef 1.0 | l2, clipped, coef 1.0 |
| advantage normalization | global (sync across GPUs) | per minibatch (`normalize_advantage_global=False`) |
| policy std | init 0.05, clamped [0.001, 0.5] | init **1.0** (`log_std_init 0.0`), **unclamped** |
| actor net | 2048 / 1024 / 512 / 512 SiLU, no input norm | 2048 / 2048 / 1024 / 1024 / 512 / 512 SiLU (campaign), input norm |
| critic net | 4096 / 4096 / 2048 / 2048 / 1024 / 1024 / 512 / 512 SiLU | 2048 / 2048 / 1024 / 1024 / 512 / 512 SiLU |
| aux losses on the actor | yes (tokenizer) | none |

The repo has SONIC's exact optimizer contract available
(`sonic_release_optimizer=True`: lr 2e-5 in [1e-5, 2e-4], grad clip 0.1,
std 0.05 clamped, global advantage norm, entropy 0.01) but the tuned recipe
selects the local contract, which the 2026-08-02 screen found 4.6x faster per
minute (`wiki` rlopt-hp-search notes). The exploration noise is the most
plausible carrier of the "stiff" behaviour on the reward side: an unclamped
std that starts at 1.0 rewards policies that survive large action noise, and
`action_rate_l2` at -0.1 is the only thing pulling the mean action back.

## 8. What this says for the smoothness question

Identical: actuator gains, DR, proprio noise, terminations, reset protocol,
every SONIC reward term. Different, in order of likely effect on ankle/knee
"stiffness": (1) exploration std 1.0 unclamped against 0.05 clamped, (2) no
energy penalty, (3) 0.2 s against 1.0 s command lookahead, (4) 4x keypoint
weight plus three global-anchor terms, (5) frozen encoder against an online
tokenizer with aux losses, (6) local optimizer contract (lr 1e-3, grad clip
1.0, no global advantage norm), (7) no upper-body / freeze-frame augmentation.
Items 2 and 1 are the ones a fine-tune can move; the energy/torque arms
(`2026-09-10-energy-torque-10b`) cover 2. Item 1 would be a
`sonic_release_optimizer=True` fine-tune off the same hub, one variable.
