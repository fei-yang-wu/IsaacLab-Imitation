---
name: ec-deployment
description: Run the Embodied-Control deployment rehearsal rig for a G1 tracker — export a policy bundle, run the MuJoCo or DDS plant, measure MPJPE, and read the R0-R2 timing certificates. Use when the user mentions Embodied-Control, EC, the tracker runtime, policy bundles, sim2sim or hardware rehearsal, DDS, unitree_sdk2, MuJoCo native evaluation, SLO or tick certificates, or asks to deploy a checkpoint outside Isaac.
---

# Embodied-Control deployment rig

`external/Embodied-Control` is the **deployment rehearsal rig**. Synced
evaluation stays in IsaacLab-Imitation; EC evaluation is asynchronous and
statistical. A number from EC and a number from Isaac are not the same
measurement, and they have disagreed strongly.

State: R0–R2 of the v2 plan are done and certified. R3 (real hardware) is
gated on a robot. Design page: `wiki/tracker-runtime-v2-architecture.md`;
running log: `wiki/embodied-control-tracker-runtime.md`.

## Always change directory first

```bash
cd external/Embodied-Control
pixi run -e native ...
```

The main-repo manifest has **no** `native` environment. Running
`pixi run -e native` from the repository root resolves the wrong manifest and
can fail silently through a grep filter.

## The policy bundle is the only training-to-runtime interface

Export in the `isaaclab` (or `onnx-export`) environment:

```bash
pixi run -e isaaclab python -m imitation_experiments.lowlevel.export_policy_bundle ...
```

A bundle carries the manifest, the TorchScript policy and encoder, the
observation and action contracts, the normalizer, a golden trace, and the
provenance: checkpoint SHA, encoder stride, and anchor mode. The runtime
hardcodes nothing about the policy.

- Encoder checkpoints from before the stride/anchor/activation fields exist do
  not carry them. Pass them explicitly.
- Golden traces must be batch 1. fp32 matmul order differs by batch size, and
  a batched trace produces a false mismatch of about 1.6e-5.
- For an FSQ arm, the planner target is the **pre-quantized** bounded vector;
  the runtime snaps it at consume time. Never regress the rounded code.

## Running

`ec lowlevel mujoco-native --mpjpe` runs the MuJoCo backend; the sweep driver
is `scripts/oracle_mpjpe_eval.py`. The Digit-style `MujocoDdsPlant` serves the
exact G1 DDS protocol, so `NativeUnitreeLoop` is the one hardware path:
`--network lo` for simulation, the NIC for the robot. `unitree_sdk2` is pinned
at `native/thirdparty` and is git-ignored; CMake finds it automatically.

GR00T heads reach the rig through
`imitation_experiments.evaluation.eval_gr00t_ec` (Hydra) and the stdio chunk
service. See the `gr00t-planner` skill.

## Traps that each cost real debugging time

- Record `active_reference_tick + slot` as the reference frame, **never** the
  window slot. MPJPE needs the robot initialized on reference frame 0, and it
  needs the frame-0 anchor rigid alignment — without it the metric measures
  the spawn offset.
- The G1 straight-leg zero pose is only marginally stable in MuJoCo. Any ramp
  transient tips it about 2 s later. Use the bent-knee stance for standing
  tests, and do not chase a phantom command bug first.
- The plant and the controller must be separate processes. One DDS
  `ChannelFactory` init per process. CycloneDDS on `lo` prints
  "not multicast-capable"; that message is benign.
- `MotionSwitcherClient` RPC times out after about 5 s against the simulated
  plant inside `begin_initialization`. Expected simulation/real difference.
- `ec_native/python/ec_native/__init__.py` re-exports a fixed symbol list. A
  new binding stays invisible until it is added there.
- Publisher sequence numbers must be monotonic over the process lifetime, not
  per episode; otherwise the buffer drops episode 2.
- Set `MUJOCO_GL=egl`. Pin `ffmpeg>=6`; a bare `*` solves to 2.8.6.
- Free-joint `qvel` angular components are in the **body** frame.
- Safety needs `SafetySpec.min_base_height_m`. Without it a fallen robot
  "completes" the episode and a real fall is masked.
- Never run anything else on the host during an SLO evaluation. The
  certificate catches host load: a concurrent test suite raised the measured
  wake latency to 3.5 ms and contaminated the run.

## Certificates

`logs/policy_bundles/` holds `r1_slo_certificate.json` and
`r2_async_certificate_v2.json`. `logs/` stays uncommitted; the wiki cites the
paths.

Simulation-rehearsal targets and hardware targets are different profiles in
`slo.py`. Wake-late of about 3.3 ms on this platform is idle-state exit
latency, not a fault — it is identical under quiet, loaded, and FIFO+mlock
conditions. The hardware profile (1 ms) needs a tuned robot host: capped
C-states and no co-hosted GPU planner.

## Known documentation defect

`source/isaaclab_imitation/CONTEXT.md:68-70` states the quaternion invariant
backwards. The code is XYZW at runtime (Isaac Lab 3.0) and WXYZ in the NPZ
datasets.
