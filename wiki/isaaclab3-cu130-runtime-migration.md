# IsaacLab 3 CU130 Runtime Migration

Status: runtime image and Newton qualification are implemented; the remaining
blocking item is the full RLOpt PhysX startup path in the split Isaac Sim/CU130
Python runtime.

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
| CU130 runtime | `/opt/isaaclab-imitation-runtime/bin/python` | PyTorch 2.11 CU130, TorchRL, TensorDict, RLOpt/ILTools dependencies | Newton kit-less training, data tools, validation |
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
shut down cleanly. Full RLOpt PhysX remains blocked by the startup/import-order
and package-bridge issue described above; this is a training-entry migration
problem, not an Apptainer overlay requirement.

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

No ICE upload or Slurm submission is part of this migration note.

## Migration sequence

1. Refactor RLOpt training into an import-light dispatcher plus `run(argv)`
   implementation, following upstream `train_rsl_rl.py`.
2. Replace the Hydra decorator boundary with explicit task/agent config
   resolution while preserving existing Hydra/preset CLI behavior.
3. Add the Kit-first bootstrap and make it own the SimulationApp lifecycle.
4. Add the narrow CU130 bridge used only by `/isaac-sim/python.sh`.
5. Teach the SIF/cluster runner to choose direct Newton versus Kit-first PhysX
   before starting Python.
6. Add CPU tests for selection, argument forwarding, single ownership/cleanup,
   and invalid backend/GPU combinations.
7. Rebuild the immutable SIF and rerun local read-only SIF acceptance tests.
8. Rerun one-iteration Newton and PhysX qualification on A40 and L40S.
9. After local and Skynet qualification, prepare—but do not submit—the ICE H100
   Newton and ICE L40S Newton/PhysX commands.

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
