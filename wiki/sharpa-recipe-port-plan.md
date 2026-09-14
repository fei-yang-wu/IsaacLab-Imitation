# Sharpa recipe port plan

Status date: 2026-09-05

This page is the plan to replicate the video-to-data (CHORD) Sharpa
dual floating-hand training recipe inside this workspace: ILTools manages
data and retargeting, `isaaclab_imitation` defines the task, RLOpt trains.
The RSL-RL port stays registered as the A/B baseline. The living status is
in [sharpa-progress.md](sharpa-progress.md); the runbook is in
[sharpa-floating-hand-v1.md](sharpa-floating-hand-v1.md).

## Decisions (2026-09-02 to 2026-09-05)

| Decision | Choice |
| --- | --- |
| Data | **ARCTIC, downloaded and loaded 2026-09-06.** The synthbox fixture stays only as a schema smoke. The LOAD stage runs natively in a sibling Pixi project because Docker is not installed here. |
| Fidelity | Exact source parity first, as a separate task `Isaac-Sharpa-V2D-Source-v0` pinned by a contract test. The object-goal variant `Isaac-Sharpa-V2D-PhysX-v0` stays. |
| Submodules | Edits in RLOpt and ImitationLearningTools are allowed; bump both pointers with the change set. |
| Compute | Local workstation only (RTX PRO 5000, 48 GB). Qualification-scale runs up to about 50M frames. No cluster submission; a campaign spec may be prepared for later. |

## Terms

- **Source-parity variant**: the task `Isaac-Sharpa-V2D-Source-v0`. It
  reproduces the released observation, command, reward, curriculum, event,
  and termination contract. It differs from the released environment only
  where this page says so.
- **Task variant**: the existing `Isaac-Sharpa-V2D-PhysX-v0` with the
  object-goal and success rewards of this repository.
- **Released playback**: the reference resampled with `motion_speed 0.5`,
  which the released code advances once per 20 Hz control step (half speed).
- **Settling hold**: the 20 control steps after a reset in which the
  virtual object controller stays at full scale and the reference frame does
  not advance.

## Phases

### A. Data (ILTools)

Done:
- Pink MANO-to-Sharpa retargeter with proven source parity
  (`compare_sharpa_references.py`, solver test, provenance metadata).
- Resample provenance (`metadata.resample.motion_speed`) written by the
  loader; the parity task refuses references without `motion_speed 0.5`.
- Half-speed synthbox dataset `data/dexmanip/sharpa/synthbox_mano_pink_rigid_v2_speed05`.

Done 2026-09-06 (real capture):
- ARCTIC raw sequences downloaded and checksum-verified against the official
  manifest; MANO models extracted to `data/dexmanip/human_motion_data/mano`.
- Native LOAD stage. `/home/fwu91/Documents/DexManip/v2d_loader` is a sibling
  Pixi project pinned to python 3.10, numpy 1.23, and mujoco below 3.7, with
  manotorch and chumpy installed there and nowhere else. It carries a package
  shim and a viser stub, and it produced
  `data/dexmanip/human_motion_data/arctic/arctic_loaded` for
  `dataset_s07_box_grab_01`: 725 frames at 30 Hz, a 17.7 cm hand span, and
  fingertips within 0.6 mm of the object.
- Rigid proxy asset for the hinged ARCTIC box
  (`scripts/data/make_rigid_object_urdf.py`). The lid moves at most 0.012 rad
  in this sequence, so the bottom is a fair proxy.
- Retargeted set `data/dexmanip/sharpa/arctic_box_grab_speed05`: 965 frames at
  the released half-speed playback, contact geometry on 764 frames.
- Comparison videos under `logs/rsl_rl/sharpa_v2d/retarget_videos/`.

Open (needed for a real capture):
- Support-surface reconstruction port (`support_recon.py` logic, pxr-free
  USDA writer) into `iltools/retarget/support_surfaces.py`.
- Hand-object and hand-hand penetration check (`filter_penetrations.py`
  logic, 2 cm limit, stride 3) and a quality audit script.
- Sharpa promotion: assisted replay (virtual object control at full scale)
  must complete the horizon; attach `TrainingQualification`; the parity env
  verifies it.

### B. Source-parity task (isaaclab_imitation)

Done 2026-09-05:
- `sharpa_source_command.py`: 65-value command, per-link contact tensors,
  friction-cone wrench supports (512 basis directions, 8 cone edges,
  friction 0.1), step-mode decay with the settling hold, released start-frame
  rule, unique wrist quaternions, limit-scaled fingers in the live order.
- `sharpa_source_mdp.py`: released action penalty (sum of squares over both
  hands) and thin bindings of the shared CHORD kernels.
- `sharpa_source_env_cfg.py`: twelve observation terms with inert noise,
  released rewards and weights, terminations with the orientation limit,
  hand material randomization, six-term curriculum, released-direction PhysX
  contact sensor (object sensed, 17 hand links filtered by kinematic-tree
  prim paths).
- Contract test `test_sharpa_source_parity_contract.py`.
- Two defects found in the task variant while porting, both fixed: the
  finger observation scaled reference-ordered positions with Isaac-ordered
  limits, and the PhysX contact sensor resolved only the palm body, so the
  boolean contact reward never saw finger contact.

Known departures from the released environment:
- Support surfaces are static colliders from the reference USDA, not
  re-spawned kinematic cylinders; hand-to-support collisions stay enabled.
- PhysX contact positions are Isaac Lab's averaged contact points, else the
  link origins. Newton uses exact contact points.
- `scene.replicate_physics` stays `True` (released: `False`) because the
  Newton contact-pair adapter requires it; the MDP is unchanged.
- The wrench basis is seeded, so rewards are comparable across runs here but
  not bit-equal to the released code.

Open:
- **The parity gate does not pass on real data at the synthbox thresholds.**
  See the progress page for the numbers. Decide the gate for redundant IK
  before citing the port as source-parity on real captures.
- Released checkpoint replay certificate. The Hugging Face repository is
  login-gated; without it, equivalence rests on the contract test and the
  RSL-RL versus RLOpt comparison.

### C. RLOpt training

Done 2026-09-05:
- `sharpa_agents.py`: `SharpaRLOptPPOConfig`, field-by-field equal to the
  RSL-RL recipe, registered for both Sharpa tasks; recipe restatement at any
  environment count.
- RLOpt `ppo.truncation_bootstrap = "rsl_rl"`: adds `gamma * V(s_t)` at
  time-outs and treats them as terminal before GAE, with a unit test against
  the RSL-RL return loop. Advantage normalization over the whole rollout,
  Adam, MSE value loss, learning-rate bounds `[1e-5, 1e-2]`.
- PhysX smokes of the task variant: 4 envs x 2 iterations and 8 envs x 6
  iterations.

Open:
- Input normalization clamp (RLOpt clamps to plus or minus 5; RSL-RL does
  not). Expose `normalization_clip=None`.
- Matched local blocks, RSL-RL versus RLOpt, same seed and frame count.

### D. Local qualification ladder

Curriculum thresholds are PPO updates times 24 control steps, compared with
the global step counter, so they are independent of the environment count.

| Rung | Shape | Frames | Curriculum | Gate |
| --- | --- | --- | --- | --- |
| D0 | 256/512/1024 envs x 20 iterations, PhysX and Newton | - | - | steps per second recorded |
| D1 | 1 env x 1 iteration, both backends | 24 | 0 | exit 0, finite rollout |
| D2 | 16 envs x 2 iterations, both trainers | 768 | 0 | metrics present, checkpoint written |
| D3 | 128 envs x 3300 iterations | 10.1M | stage 1 at update 2000 | scale drops at 2000, no NaN |
| D4 | 512 envs x 4000 iterations, two seeds, both trainers | 49.2M | stages 1 and 2 | both arms complete, evaluation summary written |

Never run a 100M block locally.

### F. Cluster (Skynet)

Campaign: `experiments/campaigns/2026-09-06-sharpa-arctic-source-parity/`.
Preflight passes on every check as of 2026-09-06; nothing is submitted.

Slurm facts verified on the login node: `wu-lab` caps jobs at 4 hours and
carries `gpu:l40s:8` on two nodes; `overcap` allows 2 days on the same nodes
but is preemptible; QOS `short` and `long` allow 2 and 7 days, but the
partition cap binds. Segments run 3:59:00 and chain with `afterany`.

ILTools now relocates a moved scene asset by content hash, so a Reference set
stays valid when its tree is copied to a cluster bind. This was the blocker
for any cluster run.

Two control-plane defects were found and fixed by this first submission:
ILTools could not resolve a relocated scene asset (fixed by content-hash
relocation), and the Skynet profile issued bare Slurm commands over a
non-login ssh shell, which exits 127 (fixed by a declared `slurm_bin_dir`).

Calibration job 3771894 is submitted. It measures the 4096-environment memory
fit on a 40 GB L40S before the long chain is queued; the extrapolation is
36.4 GB.

### E. Docs and status

Keep [sharpa-progress.md](sharpa-progress.md) as the status page and this
page as the plan. Record every measured number with its qualification in
the same sentence.
