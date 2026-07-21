# IsaacLab 3 CU130 Runtime Migration

Status: the runtime-aware RLOpt entrypoints are implemented, and local
one-iteration G1 PhysX and strict kit-less Newton runs both pass. The immutable
SIF, manifests, data, and G1 USD assets are staged on ICE. ICE H100 Newton also
ran and produced a headless rollout video. Runtime startup is therefore not the
current blocker. The dynamics/training gate that paused this work on 2026-07-19
is now resolved: the flat curves were caused by an RLOpt advantage-normalization
defect on the IPMD path plus running the public SONIC release optimizer contract
at 0.1% of its intended scale, not by the migration or by either physics
backend. See "Training-gate resolution (2026-07-19)" below. The default
`Isaac-Imitation-G1-Latent-Sonic-v0` agent config now uses a locally-validated
optimizer contract that learns at single-GPU scale; the exact release contract
remains available for cluster-scale reproduction via
`--agent rlopt_ipmd_sonic_release_cfg_entry_point`. ICE training job `5521466`
was cancelled before these fixes; resubmission should use the fixed default
contract. Local qualification checks must use about 50M frames; 10M-frame
blocks are too short to distinguish a learning recipe from a flat one.

## Goals

- Use one immutable IsaacLab 3 runtime SIF based on
  `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`.
- Keep PyTorch 2.11 and its CUDA 13.0 packages in the immutable runtime.
- Bind-mount this repository, RLOpt, and ImitationLearningTools so ordinary
  Python source changes do not rebuild the image.
- Run the SIF read-only with targeted writable binds for caches, data, logs,
  home, and `/tmp`; do not create an Apptainer overlay.
- Support both strict kit-less Newton execution and Kit-based PhysX execution.
- Qualify compute-only GPUs for Newton and RT-capable GPUs for either Newton or
  PhysX without silently selecting an unsupported renderer.

## Runtime layout

The runtime deliberately has two Python contexts with different owners:

| Context | Interpreter | Owns | Intended use |
| --- | --- | --- | --- |
| CU130 runtime | `/opt/isaaclab-imitation-runtime/bin/python` (stable target; the first built image uses `/opt/isaaclab-imitation-runtime-spec/.pixi/envs/container-runtime/bin/python`) | PyTorch 2.11 CU130, TorchRL, TensorDict, RLOpt/ILTools dependencies | Newton kit-less training, data tools, validation |
| Isaac Sim Kit | `/isaac-sim/python.sh` | SimulationApp, Kit extensions, PhysX, USD and bundled native dependencies | PhysX and any Kit/RTX path |

The editable source roots are supplied through normal bind mounts:

- `/workspace/project/source/isaaclab_imitation`
- `/workspace/project/RLOpt`
- `/workspace/project/ImitationLearningTools`

Dependency changes rebuild the SIF. Source-only changes require only a new
source snapshot or bind mount.

## Why there is no overlay

A SIF root filesystem is immutable by default. The training process only needs
writes in known locations: logs, data, caches, home, and `/tmp`. Each location
can be a normal writable bind. An overlay is only needed when a job must persist
changes to the container root filesystem, which this workflow explicitly
avoids.

The previous 20 GB overlay size was an arbitrary empty-filesystem allocation,
not a measured runtime requirement. Recreating the local Pixi environment in an
overlay would also duplicate Isaac Sim and CUDA/PyTorch packages already stored
in the image.

## What upstream IsaacLab 3 changes

The current upstream RSL-RL entrypoint is the structural model for the RLOpt
migration:

1. A small unified dispatcher selects the RL library.
2. The library entrypoint exposes `run(argv)` rather than doing all work during
   module import.
3. CLI parsing and task-config resolution are explicit functions.
4. Training runs inside `with launch_simulation(env_cfg, args_cli):`.
5. Environment creation, runner creation, training, and cleanup are scoped to
   that runtime lifecycle.

Relevant upstream files in the pinned submodule are:

- `IsaacLab/scripts/reinforcement_learning/train.py`
- `IsaacLab/scripts/reinforcement_learning/common.py`
- `IsaacLab/scripts/reinforcement_learning/rsl_rl/train_rsl_rl.py`
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/sim_launcher.py`

RLOpt should adopt the same explicit `run(argv)` and lifecycle organization.
The old `@hydra_task_config` wrapper is a poor boundary for this runtime because
it registers the task and resolves Hydra configuration before the decorated
function enters `launch_simulation`.

## Required RLOpt startup split

Copying the upstream RSL-RL entrypoint verbatim is insufficient for the SIF.
Upstream normally runs one coherent Python installation. This SIF intentionally
combines Kit from the NGC image with newer CU130 ML packages, so native package
load order must be controlled.

### Newton kit-less path

The Newton path should execute `scripts/rlopt/train.py` directly with the CU130
runtime Python.

1. Inspect raw CLI arguments for `--assert-kitless` before importing Isaac Sim.
2. Install the strict import guard.
3. Parse the normal IsaacLab preset arguments.
4. Resolve the task and agent configs.
5. Require `compute_kit_requirements=False` and an active `NewtonCfg`.
6. Enter `launch_simulation`; it is a no-op for the Kit lifecycle.
7. Import RLOpt/TorchRL training components and run training.
8. Assert that `isaacsim`, `omni.kit`, and `SimulationApp` were never loaded.

No `/isaac-sim/python.sh`, Kit cache, renderer bootstrap, or overlay is needed
for this path.

### PhysX and Kit path

The PhysX path must use `/isaac-sim/python.sh` and start `AppLauncher` before
importing the project task registry, Hydra task configuration, W&B, TorchRL, or
RLOpt. This follows the long-standing IsaacLab rule used by Kit-first examples:
launch SimulationApp first, then import modules that may load Omniverse, USD, or
native plugins.

Use a small Kit bootstrap entrypoint rather than scattering conditional imports
through the training implementation:

1. Parse only the AppLauncher-compatible arguments with `parse_known_args`.
2. Construct `AppLauncher` immediately.
3. Import and call the RLOpt `run(argv)` implementation.
4. Let the implementation resolve task/agent configuration and enter
   `launch_simulation`; it detects the already-running Kit app and must not
   create a second one.
5. Close the environment first and the bootstrap-owned SimulationApp last, in a
   `finally` block.

The cluster runner should select this bootstrap for PhysX. It should select the
direct CU130 interpreter for strict Newton. Backend selection must happen before
the Python process begins; changing interpreters after native packages have
loaded is unsafe.

## CU130 bridge for the Kit interpreter

The Kit interpreter needs access to the immutable runtime dependencies, but the
entire runtime `site-packages` directory must not be placed ahead of Isaac Sim's
own paths. Doing so can replace Kit's protobuf/gRPC/native packages and crash
during plugin shutdown.

The bridge should therefore:

- keep Isaac Sim/Kit paths first;
- append the CU130 runtime `site-packages` directory;
- prepend only a narrow `nvidia` namespace redirect to the CU130 runtime so
  PyTorch loads the matching NCCL/CUDA libraries;
- expose project, RLOpt, and ILTools source directories through `PYTHONPATH`;
- fail fast unless `torch.__version__` is 2.11, `torch.version.cuda` is `13.0`,
  and Torch resolves from `/opt/isaaclab-imitation-runtime`.

This narrow bridge is necessary because allowing the Kit interpreter to load
the older bundled NCCL caused a CU130 Torch import failure with an undefined
`ncclDevCommDestroy` symbol. Conversely, placing every runtime package before
Kit produced duplicate protobuf/gRPC registration and an invalid-pointer crash.

## GPU policy

| GPU class | Newton compute | Newton-Warp renderer | PhysX | Isaac RTX / OVRTX | Kit visualizer |
| --- | ---: | ---: | ---: | ---: | ---: |
| A100/H100/H200 compute-only | yes | yes | reject | reject | reject |
| A40/L40S/RTX workstation | yes | yes | yes | yes | yes |

The compute-only rejection must occur before launching Kit. Local tests on an
RTX workstation still need to prove that strict Newton did not import or launch
Kit; later A100/H100 execution is the hardware confirmation.

## Current qualification evidence

The read-only SIF was tested on Skynet without an overlay. The nodes used Ubuntu
24.04 with `kernel.apparmor_restrict_unprivileged_userns=1`. Plain unprivileged
`unshare -Ur` failed, while the installed non-setuid Apptainer executables still
executed the SIF successfully:

- Dendrite A40: system Apptainer 1.4.5.
- Bishop L40S: system Apptainer 1.5.2.
- Host NVIDIA driver: 580.159.03.
- Container PyTorch: 2.11.0 CU130.
- CUDA allocations and kernels passed on both GPU types.

Newton qualification passed on both A40 and L40S. The A40 run completed one PPO
iteration, reported `compute_kit_requirements=False`, retained the strict no-Kit
invariant, and passed changing 64 x 64 RGB frame validation with the Newton-Warp
renderer.

Minimal PhysX qualification also passed on both A40 and L40S: SimulationApp
started, `omni.physx` loaded, CU130 Torch executed, Kit updates ran, and the app
shut down cleanly.

The RLOpt startup split is now implemented in
`scripts/rlopt/train.py`, `scripts/rlopt/train_physx.py`, and
`scripts/rlopt/train_impl.py`. On 2026-07-19, the local RTX PRO 6000 run started
Kit first, created the G1 PhysX environment, completed one PPO iteration over
384 frames, and closed the environment and bootstrap-owned app without the
former invalid-pointer shutdown. The CPU runtime tests also prove import-light
dispatch, backend conflict checks, compute-only GPU rejection, the no-Kit
import guard, and exactly one PhysX app owner.

G1 initially used a backend-specific asset preset: PhysX retained the packaged
URDF converter while Newton loaded a preconverted USD directly, avoiding the
Kit-backed converter. On 2026-07-19, strict Newton created 16 environments,
completed one PPO iteration over 384 frames, and retained the no-Kit invariant
through shutdown. The local USD tree is 28 MiB and contains the root layer plus
its `configuration/` sublayers. Cluster profiles expose the persistent tree as
`/data/unitree/usd/g1_29dof_rev_1_0.usd`; a missing root layer fails before
simulation with preparation instructions.

For matched dynamics work, both solvers can and should use one USD. The current
explicit diagnostic/training override `--shared-g1-usd` selects Isaac Sim's
bundled G1 USD at
`.pixi/envs/isaaclab/lib/python3.12/site-packages/isaacsim/exts/isaacsim.asset.transformer.rules/data/tests/G1/g1.usda`
for either backend. This removes URDF conversion and asset-content differences
from the comparison without changing the contested default preset. A common
USD alone is not sufficient, however: Newton enumerates the articulation in a
depth-first limb order while PhysX exposes a breadth-first/interleaved order.
The action term and actor/critic joint observations now request the exact
29-joint IsaacLab order with `preserve_order=True`, so an action index and a
checkpoint input have the same meaning on both solvers.

## Paused dynamics and training investigation

The local comparison now uses the public SONIC release settings where they are
unambiguous:

- pelvis anchor;
- SONIC PD gains and action scale;
- SONIC rewards and strict terminations;
- 10-step proprioception and action histories;
- full-trajectory random starting frames;
- rigid-body mass scale range `[0.8, 2.5]`;
- PPO gradient clipping at `0.1`;
- one shared G1 USD and one canonical 29-joint input/output order.

The resolved RLOpt actor inputs are already correct. RLOpt selects only
`latent_command`, projected gravity, base angular velocity, joint position,
joint velocity, and previous-action histories. Other terms exposed by the
environment observation group are not consumed by the actor. Removing those
exposed terms was investigated as a false lead and reverted.

The full-trajectory reset path is active. A 128-environment Newton probe sampled
128 unique initial frames, with minimum 56, maximum 12,841, and no frame-zero
starts. A separate 256-environment probe over the full manifest sampled 249
unique frames from 0 through 13,095, with 98.0% nonzero starts.

With randomization disabled, a 16-environment, 100-step reference-action probe
using the same USD reported:

| Metric | Newton | PhysX | Difference |
| --- | ---: | ---: | ---: |
| Sum of mean reward terms per second | 5.1793 | 5.2114 | -0.62% |
| Mean joint-position error | 0.02777 rad | 0.02869 rad | -3.2% |
| Mean end-effector XYZ error | 0.03417 m | 0.03358 m | +1.8% |

Both solvers also hit the same strict wrist/end-effector termination roughly
every ten steps in that probe. With all SONIC startup/reset randomization and
full random starts enabled over 256 environments and 100 steps, Newton versus
PhysX remained close in physical tracking: joint error was 0.06697 versus
0.06949 rad, end-effector error was 0.09711 versus 0.09723 m, and completed
episode length was 12.12 versus 12.06 steps. Startup-only and reset-only probes
also stayed close. The largest backend-dependent diagnostic terms were the
action-rate and feet-acceleration penalties, not pose tracking.

The remaining discrepancy appears during optimization. Matched 9.83M-frame
runs (4096 environments, 24 rollout steps, 100 iterations) used the shared USD,
canonical joint order, full random starts, and exact release overrides. The
last-ten-iteration smoothed results were:

| Backend | Mean reward | Mean episode length | Throughput |
| --- | ---: | ---: | ---: |
| Newton | 0.0324 | 6.31 steps | 53.6k frames/s |
| PhysX | 0.0543 | 7.65 steps | 33.8k frames/s |

Both curves were essentially flat and neither is a promising controller. This
means deterministic solver parity is substantially better than training parity;
the next investigation should compare fixed initial policy weights, rollout
tensors, advantages, losses, and optimizer updates across backends before any
more cluster training.

## Training-gate resolution (2026-07-19)

The flat-training investigation concluded the same day. Three compounding
causes were identified; none is backend-specific, which is why Newton and
PhysX were flat together.

1. **RLOpt correctness defect (fixed).** With
   `ppo.normalize_advantage_global=True`, `PPO._construct_loss_module`
   disables the per-minibatch normalization inside `ClipPPOLoss` and
   `PPO.pre_iteration_compute` normalizes the advantage once over the whole
   rollout. `IPMD.pre_iteration_compute` overrides that method and omitted the
   global-normalization block, so the SONIC IPMD runs trained on raw GAE
   advantages at a ~0.03 reward scale. The block is now applied identically in
   the IPMD override (`RLOpt/rlopt/agent/ipmd/ipmd.py`).
2. **Release-scale optimizer contract at local scale.** The port was audited
   line-by-line against the actual public release
   (`NVlabs/GR00T-WholeBodyControl`, config
   `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml`).
   The transcription is faithful: strict adaptive terminations from iteration
   zero, `init_noise_std 0.05` clamped to [0.001, 0.5], actor lr 2e-5 adaptive
   in [1e-5, 2e-4], joint actor+critic gradient-norm clip 0.1, default-offset
   actions (the release does NOT use reference-residual actions), 6-layer SiLU
   MLPs, global per-rollout advantage normalization, and running observation
   normalization. That recipe is tuned for 64+ GPUs x 4096 envs x 100k
   iterations; NVIDIA's own docs state convergence targets "after 100K
   iterations". A 100-iteration or even 500-iteration single-GPU run of that
   contract is designed-in flat: rerunning the exact flat command with only
   the advantage fix still ended at episode length 6.6 after 100 iterations.
   Release quirks worth recording: the released trainer never consumes its
   configured `critic_learning_rate` (single optimizer at the actor lr), and
   the release's own `base.yaml`/`eval.yaml` termination thresholds are the
   looser 0.25 / 1.0 / 0.25 — the strict 0.15 / 0.2 / 0.15 set is an
   experiment override.
3. **Missing release-parity feature (added).** The release normalizes both
   actor and critic observation inputs with running mean/std; the port only
   normalized the critic. `policy.normalize_input` is now honored in
   `PPO._construct_policy`.

Local 50M-frame validations (RTX PRO 6000, 4096 envs x 24 steps, seed 0,
LAFAN1 40-motion manifest, h10 pelvis skill encoder unless noted; about 600 s
per 50M-frame Newton run):

| Run | Backend | Config | Result |
| --- | --- | --- | --- |
| Nominal repro | PhysX | `Latent-v0`, pre-SONIC recipe, h25 torso encoder | ep_len 4 -> 19.3 at 9.8M frames; matches the 2026-07-13 pre-migration per-frame pace, so the migrated stack itself trains correctly |
| Release contract + advantage fix | Newton | `Latent-Sonic-v0`, release optimizer | flat at ep_len 6.6 (100 iters), as expected at this scale |
| SONIC env strict + local contract | Newton | strict release terminations, lr 1e-3 adaptive, grad clip 1.0, std init 1.0, ELU [512,256,128] | ep_len 4 -> 14.6, r_step -0.07 -> +0.037 over 50M, monotonic |
| SONIC env base thresholds + local contract | Newton | anchor 0.25, ori 1.0, EE 0.25, feet 0.3 | ep_len 4 -> 25.9, r_ep 0.71 over 50M, fastest local learner |
| Nominal repro | Newton | `Latent-v0`, pre-SONIC recipe, h25 torso encoder | ep_len 113.8, r_ep 3.57 at 50M — on/above the pre-migration PhysX curve (49 at 18M, ~390 at 150M); Newton training parity confirmed |

Newton needed more frames than PhysX to leave the initial short-episode regime
(ep_len 6 vs 19 at 10M) but caught up by ~20M and finished far ahead of the
10M-frame snapshot, which is why 10M-frame checks misdiagnosed the backend as
non-learning. Under the local contract the policy exceeds reference-replay
survival
(~10-12 steps under full randomization) on the strict protocol and keeps
climbing, so the strict SONIC terminations are learnable locally, just slower
than the release's own base thresholds.

Config outcome: `G1ImitationLatentSonicRLOptIPMDConfig` now defaults to the
locally-validated contract, and the exact release contract is preserved as
`G1ImitationLatentSonicReleaseRLOptIPMDConfig`
(`--agent rlopt_ipmd_sonic_release_cfg_entry_point`, or flip
`sonic_release_optimizer=True` in code). The environment side (pelvis anchor,
SONIC rewards including the 3-point local reward, strict adaptive
terminations, adaptive failure sampling, domain randomization, SONIC
actuators and action scale, 10-step histories) is unchanged and remains
faithful to the release.

Decision (2026-07-19): training uses the strict termination protocol — the
env default — matching the release's train-strict convention, since that is
the protocol behind the released state-of-the-art policy. The base
0.25 / 1.0 / 0.25 thresholds are reserved for evaluation and success
definitions, which is also how the release uses them. Do not relax the
training thresholds for convenience; the expected local signature of a
healthy strict-protocol run is episode length climbing from ~4 to ~15 over a
50M-frame block with monotonic step-reward growth, not the ~26 the relaxed
thresholds produce. Local qualification gates should be calibrated to the
strict numbers.

Second decision (2026-07-19): the SONIC surface is now the default latent
task. `Isaac-Imitation-G1-Latent-v0` resolves to the SONIC env and agent
configs, `Isaac-Imitation-G1-Latent-Sonic-v0` remains as a back-compat alias
with identical kwargs, and the pre-migration beyondmimic-style surface is
deprecated but preserved as `Isaac-Imitation-G1-Latent-Legacy-v0` for old
checkpoints and frozen paper-protocol reproductions. Two design points keep
this simple:

- The SONIC policy observation group still EXPOSES the expert reference terms
  (`expert_motion`, `expert_anchor_pos_b`, `expert_anchor_ori_b`); the agent
  config's `input_keys` decide what actually feeds each network. This keeps
  posterior-mode training available as a baseline on the same task, which the
  `smoke-ipmd` pixi task now exercises against the default surface.
- Running observation normalization (both networks in the release contract)
  never touches the latent command: `normalize_input_exclude_keys` passes
  `latent_command` through raw, because the skill code and sin/cos phase
  scale/geometry are part of the pretrained encoder contract.

Pre-migration workflows and any expected-curve tables written against the old
`Latent-v0` surface must now reference `Isaac-Imitation-G1-Latent-Legacy-v0`.

## Termination-threshold curriculum (2026-07-19)

Even with the local optimizer contract, strict-from-scratch training spends
most of its early budget on ~5-step episodes (the 2026-07-19 strict 1B run
needed ~420M frames to reach episode length ~25). Since the release brute-
forces this phase with 64+ GPUs, the local default now uses a termination
curriculum: `G1SonicTerminationCurriculumCfg` linearly anneals the four
tracking-failure thresholds from the release's base/eval values
(anchor 0.25, ori 1.0, EE 0.25, feet 0.3) to the strict release values
(0.15 / 0.2 / 0.15 / 0.2) between 50M and 500M collected frames, via
`mdp.anneal_termination_threshold_by_frames` (termination params are read
live per evaluation, so the curriculum term mutates them in place; annealed
values are logged under `Curriculum/*`). Every frame after 500M — and the
final policy's protocol — is identical to strict-from-scratch training.
Adjust the window per term with
`env.curriculum.<term>.params.{start_frames,end_frames}`; remove the terms
for release-fidelity strict-from-scratch runs. First validation: at 10M
frames the curriculum run reaches episode length ~11.6 versus ~4.7 for
strict-from-scratch (same seed, 4096 envs, Newton).

## ICE ablation array (submitted 2026-07-19)

Local 1B attempts exposed a terminations-versus-asset confound: the strict
no-curriculum run on the bundled `g1.usda` climbed steadily (ep_len ~29 at
528M frames, wandb `yvqdy6i2`), while the curriculum run on the preconverted
USD stalled at ep_len ~11 through 110M frames. Both local runs were cancelled
in favor of a six-arm ICE H100 ablation (1B frames each, 4096 envs, seed 0,
strict kit-less Newton, corrected LAFAN1 tree
`/data/lafan1_corrected_8e95d557` — all 40 NPZ MD5s verified against local —
wandb project `g1-lafan1-sonic-ice-ablation`, follow-cam video):

| Arm | Job | Task / terminations | G1 asset |
| --- | --- | --- | --- |
| a0 | 5522494 | `Latent-Legacy-v0` recovery anchor (torso h25 encoder) | bundled (`--shared-g1-usd`) |
| a1 | 5522496 | SONIC strict from scratch | bundled |
| a2 | 5522498 | SONIC strict from scratch | preconverted (ICE default) |
| a3 | 5522500 | SONIC loose full-run (0.25/1.0/0.25/0.3) | bundled |
| a4 | 5522502 | SONIC curriculum 50M -> 500M | bundled |
| a5 | 5522504 | SONIC loose full-run | preconverted |

**Wave-3 correction (2026-07-20, decisive asset finding).** All wave-3 arms
silently ran the preconverted USD: the 2026-07-19 resolver change made the
cache tree shadow the bundled asset, and on ICE
`ISAACLAB_IMITATION_UNITREE_USD_PATH` (exported unconditionally by
`run_singularity.sh`) took precedence, so `--shared-g1-usd` no longer meant
"bundled". A local re-run of arm a1's exact command reproduced the ICE
numbers (ep_len 8.5 at 100M) rather than v1's 19.2, proving the gap is the
asset, not the cluster stack. Reconciled across every run to date, on
identical protocols the **preconverted URDF-conversion asset learns
drastically worse than Isaac Sim's bundled variant-aware `g1.usda`**
(strict: 8.5 vs 19.2 ep_len at 100M; legacy: ~11 vs ~114 at 50M; loose:
~11.5 vs ~25.9 at 50M). Fixes: `_force_shared_g1_usd` now hard-selects the
bundled asset with its `SimplifiedPhysX` variants regardless of env
overrides, and the resolver again prefers bundled over the preconverted
cache (the cache remains for explicit overrides and GL video rendering).
Corrected array: kept a2=5522498 (strict+preconverted) and a4=5522502
(curriculum+preconverted) as the preconverted record; resubmitted the
bundled arms as b0=5522569 (legacy), b1=5522570 (strict), b3=5522571
(loose), b4=5522572 (curriculum), wandb groups `ablate-a*`. The preconverted
tree should be treated as deprecated for training until it is regenerated
with sane collision/inertia data; its GL-viewer "zero-volume mesh" warnings
were an early symptom.

**Final 1B results (all arms completed 2026-07-20, 999,948,288 frames each):**

| Arm | Protocol / asset | Final ep_len | Final r_ep |
| --- | --- | ---: | ---: |
| b0 (5522569) | legacy latent, bundled | **421.8** | 24.01 |
| b3 (5522571) | SONIC loose, bundled | 57.8 | 2.24 |
| b1 (5522570) | SONIC strict-from-scratch, bundled | 44.1 | 2.17 |
| b4 (5522572) | SONIC curriculum 50->500M, bundled | 37.0 | 1.74 |
| a2 (5522498) | SONIC strict, preconverted | 8.7 | 0.55 |
| a4 (5522502) | SONIC curriculum, preconverted | 8.2 | 0.47 |

Verdicts: (1) the bundled asset outperforms the preconverted one ~5x at
matched protocol — preconverted is retired for training; (2) on the SONIC
surface at a 1B budget, strict-from-scratch matches or beats the curriculum
(44.1 vs 37.0, with strict's per-episode return nearly equal to loose's
despite the harder exam), so the curriculum is not required there; (3) the
legacy recovery anchor finished at 421.8, above the pre-migration reference
— previous-best performance is fully recovered on ICE H100 Newton. SONIC
training ep_len is depressed by design (adaptive failure sampling
concentrates resets on the hardest motion bins), so cross-surface ep_len
comparisons are not meaningful; checkpoint-level success-rate evaluation is
the honest cross-surface metric. Follow-up direction (2026-07-20): the
SONIC surface is paused as unsuitable at current compute scale; the new
candidate default is `Isaac-Imitation-G1-Latent-Strict-v0` — legacy
scaffolding + pelvis anchor + strict termination functions annealed
50M->300M — gated locally at 50M before promotion. A recurrent policy is
planned to replace stacked observation histories as a separate track.

**Controlled asset A/B (2026-07-20, local, single variable).** Two
`Isaac-Imitation-G1-Latent-Strict-v0` runs with identical seed, encoder,
anneal (10M -> 40M compressed), machine, and code — only the G1 asset
differs:

| Frames | Bundled `g1.usda` | Preconverted tree |
| ---: | ---: | ---: |
| 10M | 6.3 | 10.7 |
| 20M | 28.1 | 10.0 |
| 30M | 39.5 | 9.0 |
| 50M | **58.9** | **8.0** |

The preconverted asset's episode length *falls* as thresholds tighten while
its per-step reward stays high (0.057 vs 0.039) — the policy tracks well on
average but physically cannot stay inside precise thresholds.

**Official-asset-only cleanup and protocol ablation v2 (2026-07-20).** Per
the user's direction the bundled Isaac Sim test fixture and the modified
URDF-conversion tree are removed from the codebase entirely: the resolver
returns only the repo-packaged official USD (plus the
`ISAACLAB_IMITATION_UNITREE_USD_PATH` escape hatch), `--shared-g1-usd` is
deleted from `train.py` and `diagnose_g1_dynamics.py`, and
`run_singularity.sh` defaults to no USD override (`CLUSTER_G1_USD_PATH=repo`
in both runtime profiles). With the asset axis gone, the ablation collapses
to protocols. Local 50M results on the official asset (seed 0, identical
recipes; compressed 10M->40M anneal for the curriculum arm):

| Arm | ep_len @50M | r_ep |
| --- | ---: | ---: |
| L0 legacy anchor (torso h25 encoder) | **143.9** | 4.37 |
| L3 loose full-run | 102.7 | 3.31 |
| L1 strict-from-scratch | 67.7 | 2.67 |
| L4 curriculum | 63.5 | - |

Strict-from-scratch beats the curriculum on the official asset, so the
curriculum is no longer necessary at this scale (kept as a config option).
In-training Newton GL video now works natively on the training asset (L1 ran
with `--video`; real meshes + follow camera confirmed). The first 1B ICE
re-run (4096 envs: l0=5522875/5522854, l1=5522855, l3=5522856, l4=5522857;
5522854 died on a failed GPU on `atl1-1-03-010-15-0`) was superseded at
~50%: per user direction the four arms were cancelled and resubmitted with
the production-scale config (8192 envs x 12 steps, mini-batch 12288) and
the new Warp bird-view video (300 frames every 25k steps) as
l0=5522924, l1=5522926, l3=5522927, l4=5522928 (wandb `g1-lafan1-strict`,
groups `ice2-*`).

Cluster Newton-GL video path (historical, superseded 2026-07-20): `--video`
on the Newton backend
attaches a **headless Newton visualizer** (`NewtonVisualizerCfg` with
`headless=True`, bird-view `eye`/`lookat`, and `visible_env_indices` set to
a 4x4 block at the env-grid center) via `_enable_bird_video_visualizer` in
`scripts/rlopt/train.py`. The video recorder prefers a live Newton
visualizer as its capture backend and follows its camera, so this uses only
official machinery. Measured at 4096 envs: ~74k fps outside capture
windows, ~11k fps inside them (~7 ms/frame; ~1% amortized at 300 frames per
25k steps), and the captured video shows the full robot grid with ground —
the intended whole-population bird view. Gotchas discovered on the way and
kept for the record: `enable_shadows=False` crashes newton's `RendererGL`
(`_light_space_matrix` only initialized with shadows on); a Warp-raycast
`TiledCamera` was tried first but renders only its own env's bodies with no
ground (single floating robot) and `TiledCameraCfg.OffsetCfg.rot` is
scalar-last (x, y, z, w) — that approach was reverted in favor of the
visualizer. This proved that the rendering path can produce a correct image,
but it is not suitable for large-environment cluster training.

**Current cluster rendering policy (2026-07-20).** Do not pass `--video`,
`--video_length`, or `--video_interval` to cluster training jobs, and keep
`agent.logger.video=false`. The only exception is a job that explicitly uses
the Isaac RTX/OVRTX rendering backend on an RT-capable allocation; merely
running Newton GL on an RTX-class GPU is not that exception. Strict kit-less
Newton training on ICE H100 is metrics-and-checkpoints only.

Use local inference to quality-check motion. Sync or download an early,
middle, and final checkpoint, then render a short deterministic playback with
the matching task, skill encoder, manifest, and dataset configuration. Prefer
the local PhysX + RTX path for the video so rendering is isolated from the
cluster training process. For example:

```bash
pixi run -e isaaclab python scripts/rlopt/play.py \
    --task <matching-task-id> \
    --algo IPMD \
    --checkpoint /absolute/path/to/checkpoint.pt \
    --num_envs 16 \
    --headless \
    --video \
    --video_length 500 \
    physics=physx \
    agent.ipmd.hl_skill_checkpoint_path=/absolute/path/to/skill_encoder.pt \
    env.lafan1_manifest_path=$PWD/data/lafan1/manifests/g1_lafan1_manifest.json \
    env.refresh_zarr_dataset=false
```

The playback must use the training arm's task/configuration; for example, L0
uses `Isaac-Imitation-G1-Latent-Legacy-v0`, while L1/L3/L4 use
`Isaac-Imitation-G1-Latent-Strict-v0` with their corresponding threshold
overrides. The video is a qualitative diagnostic, not a replacement for
checkpoint metrics.

**Resolution (2026-07-20): the official Unitree USD wins outright.** The
failing "preconverted" cache turned out to be a *modified derivative* of
Unitree's official asset (different root layer, all sublayers touched, an
extra `_robot.usd` sublayer); the pristine official
`unitree_model/G1/29dof/usd/g1_29dof_rev_1_0` tree, run through the same
50M protocol, scored **ep_len 63.5** — beating even the bundled test asset
(58.9) — and it carries real visual meshes, so Newton GL video rendering
works directly on the training asset. Three-way A/B at 50M, identical
protocol/seed/encoder: official 63.5 / bundled 58.9 / modified-derivative
8.0. The official tree (byte-identical to the validated copy) is now
committed into the repo at
`source/isaaclab_imitation/isaaclab_imitation/assets/unitree/g1_description/`
(git-lfs, from `unitreerobotics/unitree_model` commit `b6a8942b`, matching
the lab's G1 model 11: 29-DoF rev 1.0, waist unlocked, no hands), where the
resolver picks it up as the packaged default — no env vars or flags needed
locally or on clusters. The modified derivative in
`~/.cache/isaaclab_imitation/unitree_usd` and on cluster `/data/unitree/usd`
is retired and should not be referenced by new runs; the bundled test asset
remains only as the `--shared-g1-usd` diagnostic and last-resort fallback.
Local checkpoint playback is now the standard visual-quality check for
cluster runs. It also remains useful for rendering old bundled-asset
checkpoints.

Submission-wave history (current):

1. Wave 1 (5522340-48) failed in ~80 s with `CUDA error: device(s) busy or
   unavailable` at the first tensor-to-device copy — every failure on H100
   node `atl1-1-03-011-23-0`, which reports idle but rejects new CUDA
   contexts. Excluded via `CLUSTER_SLURM_EXCLUDE=atl1-1-03-011-23-0`;
   consider reporting the node to PACE support.
2. Wave 2 (5522404-09, a1 resubmitted as 5522448 after losing a shared
   TorchInductor-cache rename race when four arms cold-started on one node)
   trained but at ~2k fps: every arm was stuck in its first 500-step
   `--video` capture window. Initial diagnosis blamed CPU software
   rasterization; that was WRONG — a 2026-07-20 in-container check on an ICE
   H100 node confirmed `apptainer --nv` injects `libEGL_nvidia.so.0`
   correctly and pyglet/ViewerGL obtains a hardware H100 EGL context
   natively (RHEL 9.6 hosts carry the full driver GL stack in /usr/lib64;
   the SIF ships the glvnd ICD). The measured cost (job 5522902, 8192 envs,
   hardware context): ~18 s per captured frame, ~450 fps during capture
   windows — the bottleneck is the Newton GL viewer's per-frame scene
   submission at ~250k body instances, which scales with env count and hits
   any renderer. Locally the same path costs ~0.6 s/frame at 4096 envs
   (tolerable); at cluster scale it is prohibitive. Standing rule: keep
   `--video` off for cluster training unless the job explicitly uses the RTX
   rendering backend, and render checkpoints locally instead.
3. A later 8192-environment video retry submitted L0/L1/L3/L4 as
   5522953/5522954/5522955/5522956. L0, L1, and L4 hit NVIDIA Xid 109 while
   the Newton GL viewer was copying rendered state from CUDA; they were
   cancelled. L3 completed its initial capture and trained to about 206M
   frames, but a later OpenGL capture also became unusable and the run was
   interrupted. Slurm recorded `COMPLETED` because the trainer handled
   `Ctrl+C` and returned zero; it did not reach the 1B-frame target.
4. Wave 3 drops video and W&B video syncing. The no-video replacements are
   L0=5523400, L1=5523402, L3=5523449, and L4=5523407.

Threshold pinning is done through the curriculum terms
(`env.curriculum.<term>.params.{start,end}_value`), so all SONIC arms share
one config surface. Frozen encoders staged at `/data/checkpoints/`:
pelvis h25 `388d3e826f50...73b3d` (arms a1-a5), torso h25 `6ea0f826089c...cd55f`
(arm a0). Profile changes: `CLUSTER_SLURM_GPU_GRES=gpu:h100:1`,
`CLUSTER_SLURM_TIME_LIMIT=15:59:00` (ICE H100 partitions cap at 16 h; 1B
needs ~4-6 h). Launcher:
`scratchpad/launch_ice_ablation.sh` via
`cluster_interface.sh -c ice_runtime job ...` (note: `-c` selects the cluster
env file; passing `ice_runtime` as the positional profile silently routes to
Skynet). Decision rule: a1 and a0 must reproduce their local trajectories
(recovery); a2 decides whether the preconverted asset is trainable; a3/a4
pick the loose-versus-curriculum protocol on the winning asset.

## Headless Newton video: framing and asset resolution (2026-07-19)

Two independent defects made local `--video` Newton training runs record
empty checkerboard instead of robots. Neither existed on the ICE
qualification job, which is why "it worked on ICE":

1. **Asset resolution.** `run_singularity.sh` exports
   `ISAACLAB_IMITATION_UNITREE_USD_PATH` to the preconverted 28 MiB G1 USD
   tree, so cluster jobs render real meshes. Locally that variable was unset
   and the resolver fell back to Isaac Sim's bundled variant-aware
   `g1.usda`, whose simplified geometry collapses into zero-volume fallback
   shapes in the Newton GL viewer (the robots render as millimeter-scale
   specks; physics is unaffected). `resolve_unitree_g1_29dof_usd_path()` now
   prefers an existing preconverted cache tree
   (`~/.cache/isaaclab_imitation/unitree_usd/`) over the bundled asset, so
   local and cluster runs use the same physics asset by default and the GL
   recorder shows real robots. The bundled asset remains the last-resort
   fallback and the explicit `--shared-g1-usd` diagnostic override.
2. **Static recorder camera.** The kit-less Newton GL recorder
   (`NewtonGlPerspectiveVideoCfg`) supports only a static world-frame
   eye/lookat; `ViewerCfg.origin_type` is ignored on this path, and the
   camera is only re-synced when an interactive Newton visualizer window
   exists. With SONIC full-trajectory random starts the robots leave their
   env origins within seconds, so a static camera films empty ground.
   `ImitationRLEnv.render()` now re-aims the recorder at one robot root
   every captured frame via the recorder's public `update_camera()` hook.
   Config fields on the G1 base cfg: `video_follow_robot` (default True),
   `video_follow_env_index`, `video_follow_eye_offset` (3.5, 3.5, 2.0 m,
   world-axis-aligned), `video_follow_lookat_offset`.

A 16-env Newton probe with both fixes produces correctly framed 720p video
that tracks the followed robot. Caveat: the 2026-07-19 local 1B LAFAN1 run
(`logs/rlopt/ipmd/Isaac-Imitation-G1-Latent-v0/2026-07-19_17-53-25`) was
launched before these fixes, so it trains on the bundled `g1.usda` and its
in-training videos show empty ground; evaluate its checkpoints with
`ISAACLAB_IMITATION_UNITREE_USD_PATH` pointed at the bundled asset for
asset-matched playback, or re-render videos from checkpoints with the fixed
stack.

## ICE expectations

- H100 should use strict Newton compute or Newton-Warp rendering. The workflow
  must reject PhysX, Isaac RTX, OVRTX, and Kit visualizers before launch.
- L40S can use Newton or PhysX after the Kit-first RLOpt entrypoint and narrow
  CU130 bridge pass qualification.
- CUDA 13.0 requires a sufficiently new host driver; driver 580.159.03 already
  passed the observed Skynet CUDA tests, and driver 610 is above that baseline.
- System Apptainer is acceptable when it can execute the read-only SIF and
  provide `--nv` plus the required bind mounts. A user-local Apptainer binary
  blocked by Ubuntu AppArmor is not required by this workflow.

A read-only ICE audit on 2026-07-19 found the user-local Apptainer and Slurm
commands, and confirmed that `gpu:l40s` is available. The Dance102 manifest is
already staged. The 28 MiB preconverted G1 USD tree was subsequently synced to
both Skynet and ICE; its complete sorted checksum manifest hashes to
`6814baa09479e1735811448ae9936dd55bed7387d2cddd551b64938898791d3c` on all
three systems. The 14.8 GB immutable SIF and its manifest are also staged on
ICE. The SIF is 14,775,566,336 bytes and matches the Skynet SHA-256
`41f4082fb37a3575f340a1187751aa0c5c80a195173648e79adc921b272ee9d7`.
ICE Apptainer inspection succeeds, and direct read-only execution reports
Python 3.12.13 and PyTorch 2.11.0+cu130 with CUDA 13.0.
The Dance102 manifest's referenced NPZ is present on both clusters and matches
the local 2,612,738-byte file SHA-256
`eb84eb283a1c7d692cad03dc6ecd6c04e6fb54e3d1dd4b7eefc966e81f429707`.

ICE H100 Newton video qualification job `5521373` completed far enough to
produce a retained headless MP4; the local copy is
`/tmp/ice-h100-newton-video-5521373.mp4` (361,196 bytes). This confirms that
recording a video does not require an interactive viewer. Later latent
pretraining plus low-level job `5521466` passed runtime startup and Warp's
initial compilation phase, but it was deliberately cancelled because the
learning performance was below the previous Isaac Sim runs. That cancellation
is a dynamics/training decision, not an ICE, Apptainer, CUDA-toolkit, or
renderer failure.

No Slurm submission is implicit in this migration note.

### BONES-SEED SONIC latent training (2026-07-20)

`experiments/submit_bones_seed_100_sonic_latent_ice.sh` submits one
non-Phase-5 sequential job: rebuild the 100-motion cache, train a fresh
h25/z256 DiffSR skill encoder for 5,000 updates, then train the frozen-encoder
SONIC oracle low-level policy for 999,948,288 frames. The low-level block uses
8,192 environments by 12 rollout steps, mini-batches of 12,288, the official
repo asset, strict-from-scratch terminations, the locally validated optimizer,
and no video.

No BONES-SEED job from this wave remains active. Attempts `5523556` and
`5523559` failed during startup; `5523561` reached low-level training but
returned NaN; later Newton diagnostics were stopped or failed before producing
a candidate checkpoint. Local finite-value tracing found that the official G1
flat-locomotion `njmax=95` is sufficient for corrected LAFAN1 but not for the
valid many-body ground contacts in BONES-SEED. One BONES rollout emitted 951
constraint overflows and requested up to 236 rows; the first invalid simulator
state was associated with `ab_bicycle_001_A359` near frame 20. MPJPE, the
encoder latent, and initial actor outputs were finite before the solver state
failed. A two-motion sweep kept the model and source data fixed and exposed an
interaction between `njmax` and `nconmax`: `264/31` still requested 268 rows,
while both `272/32` and `288/32` passed 30 rollouts across three seeds. The
launcher retains `env.sim.physics.solver_cfg.njmax=288` and
`env.sim.physics.solver_cfg.nconmax=32` for headroom. This cost 0.87% steady
throughput relative to the borderline `264/31` setting and 96 MiB (2.3%) at
2,048 environments relative to `95/18`. The final full-manifest local run
completed 20,054,016 frames with no overflow or NaN at roughly 108--110
thousand steady frames/s. No replacement job was submitted.

### SONIC default and policy-contract decision (2026-07-20)

ICE H100 (and now H200) single-GPU access removes the compute-scale
objection that paused the full SONIC surface earlier the same day: 8192
envs x 12 rollout steps x 100k PPO iterations is ~9.83B (~10B) frames,
matching the release's own "after 100K iterations" convergence budget on one
GPU instead of 64+. `Isaac-Imitation-G1-Latent-v0` (the SONIC surface) is now
the confirmed default latent task rather than paused/candidate;
`Isaac-Imitation-G1-Latent-Strict-v0` is DEPRECATED and kept only to
reproduce runs already started on it. `G1ImitationLatentSonicRLOptIPMDConfig`
now defaults `sonic_release_optimizer=True` (the exact public-release
optimizer contract: actor lr 2e-5, joint grad clip 0.1, init std 0.05,
6-layer SiLU MLPs, running input normalization) instead of the
locally-validated small-scale contract, since 100k iterations is the scale
the release contract needs to leave the flat regime. Nothing has trained
end-to-end under this default+contract combination yet; the VRAM/throughput
ablation and BONES-SEED SONIC-latent submissions below are its first test.

## Migration sequence

1. **Complete:** refactor RLOpt training into an import-light dispatcher plus
   `run(argv)` implementation, following upstream `train_rsl_rl.py`.
2. **Complete:** replace the Hydra decorator boundary with explicit task/agent
   config resolution while preserving existing Hydra/preset CLI behavior.
3. **Complete:** add the Kit-first bootstrap and make it own the SimulationApp
   lifecycle.
4. **Implemented, Skynet validation pending:** add the narrow CU130 bridge used
   only by `/isaac-sim/python.sh`.
5. **Complete:** teach the SIF/cluster runner to choose direct Newton versus
   Kit-first PhysX before starting Python and run without an overlay by default.
6. **Complete:** add CPU tests for selection, argument forwarding, single
   ownership/cleanup, invalid backend/GPU combinations, and Newton asset
   preflight.
7. **Complete:** use the existing hash-pinned immutable SIF and rebuild only if
   a dependency or immutable runtime change is required.
8. **Deferred:** the exact archive-backed Skynet A40 PhysX Slurm dry run passes,
   but no additional cluster qualification is needed while local training is
   unstable.
9. **Partially complete:** ICE has the exact SIF, runtime manifest, data, and G1
   USD tree. H100 Newton plus headless video ran successfully. The long latent
   training attempt was cancelled due to learning quality; re-submit only after
   the local training gate passes.

Both runtime profiles use a 56 MiB self-contained source archive rather than a
Git clone or recursive workspace copy on cluster NFS. The submitters verify the
archive hash, retain a repository provenance manifest, and extract the archive
into compute-local `/tmp` before invoking `run_singularity.sh`. They also
support `CLUSTER_SLURM_DRY_RUN=1`, which leaves the generated job script for
inspection and never calls `sbatch`.

## Acceptance criteria

- Strict Newton completes one RLOpt PPO iteration without loading Kit.
- Newton-Warp RGB validation passes without RT cores or Kit.
- Full RLOpt PhysX completes one PPO iteration with exactly one SimulationApp
  owner and clean shutdown.
- Both paths use PyTorch 2.11 with `torch.version.cuda == "13.0"`.
- Project, RLOpt, and ILTools source edits are visible without rebuilding the
  SIF or installing packages.
- Both paths pass through `apptainer exec --nv` with a read-only SIF and normal
  bind mounts only.
- No overlay, remote upload, or cluster submission is implicit in source-only
  development.
