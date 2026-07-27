# 2026-07-27 — SONIC environment × continuous deterministic latent

One-arm screen. It asks a single question: **does the continuous deterministic
DiffSR latent train better under SONIC's environment design than under our
current `Isaac-Imitation-G1-Latent-v0` surface?**

Exactly one axis moves against the 2026-07-22 Study B `deterministic` row
(`wiki/latent-learning-ablation-plan.md`): the environment. Same corrected
LAFAN1 data, same seed, same optimizer contract, and the *same frozen skill
encoder checkpoint* — this job does not re-pretrain.

## Scripts

```bash
./run.sh local     # MODE=print by default; MODE=run for the wiring gate
./run.sh ice       # MODE=print by default; MODE=validate then MODE=submit
```

## What "SONIC's environment" means here

New task id `Isaac-Imitation-G1-Latent-Sonic-NoHist-v0`
(`ImitationG1LatentSonicNoHistoryEnvCfg`). Reference is
`NVlabs/GR00T-WholeBodyControl` at `4141c34` (`gear_sonic/`), release config
`config/exp/manager/universal_token/all_modes/sonic_release.yaml`.

| SONIC release | This task | Match |
| --- | --- | --- |
| `terminations: tracking/base_adaptive_strict_ori_foot_xyz` — `anchor_pos` adaptive 0.15 (down 0.75, root-height 0.5), `anchor_ori_full` 0.2, `ee_body_pos` adaptive 0.15, `foot_pos_xyz` 0.2, `motion_time_out`; no base-height term | `G1SonicTerminationsCfg` — identical thresholds and functions, `base_too_low = None` | ✅ |
| `rewards: tracking/base_5point_local_feet_acc` — anchor pos 0.5/std 0.3, anchor ori 0.5/std 0.4, rel body pos 1.0/std 0.3, rel body ori 1.0/std 0.4, body linvel 1.0/std 1.0, body angvel 1.0/std 3.14, 5-point local 2.0/std 0.1, action rate −0.1, joint limit −10, undesired contacts −0.1 (ankle/wrist/elbow exempt), anti-shake −5e-3 @1.5, feet joint acc −2.5e-6 | `G1SonicRewardsCfg` — same terms, weights, stds, and contact-exclusion regex | ✅ |
| `events: tracking/level0_4` — startup physics material, joint default pos, base COM, body-mass scale 0.8–2.5 on wrists+torso; interval push 4–6 s | `G1SonicEventCfg` | ✅ |
| Threshold curriculum from base/eval values to strict | `G1SonicTerminationCurriculumCfg`, 50M → 500M frames | ✅ |
| Pelvis anchor, full-trajectory adaptive-failure reset sampling, `adp_samp_failure_rate_max_over_mean: 200` | `expert_anchor_body_name="pelvis"`, `random_reset_full_trajectory=True`, `…max_over_mean=200.0` | ✅ |
| `decimation: 4`, `sim_dt: 0.005`, `episode_length_s: 10.0` | same | ✅ |
| `num_envs: 4096`, `algo.config.num_steps_per_env: 24` | 4096 × 24, minibatch 12288 | ✅ |
| `observations/policy: local_dir_hist`, `critic: privileged_mf_hist` — 10-step proprioceptive histories | **single-frame `G1LatentObservationCfg`** | ❌ deliberate |
| `terrain_type: trimesh` | plane | ❌ |
| `robot.type: g1_model_12_dex`; `head_link` in anti-shake | bundled 29-DoF G1 (`G1SonicRobotCfg`); `torso_link` proxy for the absent head body | ❌ asset |
| Release optimizer contract (6-layer SiLU, actor lr 2e-5, joint clip 0.1) | local contract (512/256/128 ELU, actor lr 1e-3) | ❌ deliberate |
| Universal-token backbone, 3 encoders, FSQ, upper-body/SMPL/teleop augmentation | DiffSR continuous deterministic latent, z256 + sin/cos phase, held 10 steps | ❌ out of scope |

The history departure is the user's call and is backed by the 2026-07-21
isolated history ablation (`ImitationG1LatentStrictHistoryEnvCfg` vs. the
single-frame strict surface), which found the 10-step histories buy little at
our scale. Term *names* are unchanged, so
`G1ImitationLatentSonicRLOptIPMDConfig`'s SONIC input-key selection still
resolves; only per-term history length differs.

## Comparability caveat

The 2026-07-22 Study B `deterministic` row ran at **16,384 envs × 12 steps** on
an H200 (minibatch 24,576). This arm runs at **4,096 × 24** because that is
SONIC's own release geometry, which is what the run is meant to replicate. The
two therefore differ in batch geometry as well as environment, and this row is
**not** a drop-in cell of the Study B table. Treat it as a screen; a strictly
controlled A/B would need a matched-geometry `Isaac-Imitation-G1-Latent-v0`
companion at 4096 × 24.

## Fixed protocol

- Task `Isaac-Imitation-G1-Latent-Sonic-NoHist-v0`, seed 0.
- Corrected LAFAN1 only: manifest sha256 `d972c37c…c945db8`,
  cache `/data/lafan1_corrected_8e95d557/g1_hl_diffsr`. Both the local and the
  ICE-side copies are hash-gated before submission.
- `env.refresh_zarr_dataset=false` always. Seven grouped-VQ arms share this
  cache; a concurrent refresh truncated it to 56 KB on 2026-07-26.
- Frozen encoder, reused verbatim:
  `logs/latent_ablation/lafan1_diffsr_deterministic_continuous_h10_z256_seed0/skill_encoder/checkpoints/latest.pt`
  (168,410,412 bytes). `--skip-pretrain`.
- h10, `--phase-mode sin_cos`, `--latent-hold-steps 10`, z256.
- 5B frame cap to line up with the 07-22 / 07-26 checkpoint grid; ICE caps GPU
  walltime at 16–18 h, so ~2.27B lands per segment at the assumed 45k FPS and
  a continuation segment is needed for the rest.
- Save interval 100M, matching both existing studies after the 2026-07-26
  checkpoint thinning.
- H100 on `coe-gpu`, excluding `atl1-1-03-010-15-0` (dead GPU, still advertised
  as healthy).
