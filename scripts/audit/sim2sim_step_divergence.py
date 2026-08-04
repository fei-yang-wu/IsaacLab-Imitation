#!/usr/bin/env python3
# ruff: noqa: E402
"""Where do CPU MuJoCo, Newton MJWarp, and PhysX first disagree?

Every earlier probe compared end-of-rollout aggregates, which cannot say
*where* a difference enters. This one drives all three engines with a
**state-independent** joint-position target sequence from an identical initial
state and records the full state after every control step, so the comparison is
per-step and the first divergence is localizable to a step and a joint.

Reference point (fixed by decision, 2026-08-03): **CPU MuJoCo + the rev_1_0
model + the SONIC actuator parameters**. Both Isaac backends are then measured
against that single reference rather than against each other, which is what
makes a disagreement attributable.

Three properties make this a controlled comparison, and all three matter:

* **The command is a joint-position target, not an action.** Isaac's action term
  applies ``target = offset + scale * action``; MuJoCo's position servo takes
  the target directly. Sending the same *action* would send different physical
  commands. This script specifies targets and inverts Isaac's affine map, so the
  physical command is identical by construction.
* **The targets are precomputed and state-independent.** An oracle action is a
  function of the live state, so once two engines diverge they also receive
  different commands and the cause is unrecoverable. A fixed sequence keeps the
  input identical no matter how far the states drift.
* **The initial state is verified, not assumed.** The trace records step 0 before
  any physics, and ``--compare`` refuses to interpret the rollout if the two
  runs did not start from the same state.

Usage (from the repository root):

.. code-block:: bash

    # The reference.
    pixi run -e isaaclab python scripts/audit/sim2sim_step_divergence.py \\
        --engine mujoco --output logs/step_divergence/mujoco.json

    # The two Isaac backends.
    ... --engine isaac --output logs/step_divergence/physx.json  physics=physx <data>
    ... --engine isaac --output logs/step_divergence/newton.json physics=newton_mjwarp <data>

    python scripts/audit/sim2sim_step_divergence.py --compare \\
        logs/step_divergence/mujoco.json logs/step_divergence/newton.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/unitree"
    / "g1_description/g1_29dof_rev_1_0.xml"
)

# Joint angle every engine is initialized to and every target sinusoid is
# centered on. Zero is the G1's straight-legged, arms-down pose -- valid, held
# clear of the floor, and unambiguous across engines.
REFERENCE_POSE = 0.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compare", nargs=2, default=None, metavar=("LEFT", "RIGHT"))
parser.add_argument("--engine", choices=("isaac", "mujoco"), default=None)
parser.add_argument("--task", default="Isaac-Imitation-G1-v2")
parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--amplitude", type=float, default=0.10, help="Target sweep, rad.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--start-height",
    type=float,
    default=1.30,
    help=(
        "Pelvis height at release. 1.30 holds the robot clear of the floor "
        "(actuator-only). ~0.80 stands it on the floor so the trace is "
        "contact-loaded -- the regime the free-flight run cannot reach."
    ),
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=0,
    help=(
        "Control steps holding the reference pose before the measured sequence. "
        "Needed with a ground start so contact is established and transient; the "
        "post-settle state difference is reported rather than assumed zero."
    ),
)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--tolerance",
    type=float,
    default=1.0e-4,
    help="rad / rad-per-s below which a per-step difference is float noise.",
)


# --------------------------------------------------------------------------
# The command sequence. Deterministic, state-independent, engine-independent.
# --------------------------------------------------------------------------
def target_sequence(
    joint_names: list[str], default_pos: np.ndarray, steps: int, amplitude: float
) -> np.ndarray:
    """``(steps, num_joints)`` joint-position targets around the default pose.

    A per-joint sinusoid rather than noise: every joint moves smoothly and
    continuously, so the actuators are genuinely exercised, and the sequence is
    reproducible from its arguments alone without sharing an RNG across
    processes.
    """
    num = len(joint_names)
    # Phase and rate are derived from each joint's rank in the NAME-sorted
    # ordering, never from its index in the caller's array. The engines
    # enumerate joints differently, so index-derived parameters would send a
    # different physical command to the same joint on each engine -- which is
    # the one thing this probe must not do.
    rank = np.empty(num, dtype=np.int64)
    rank[np.argsort(np.array(joint_names))] = np.arange(num)
    phase = rank[None, :] * (2.0 * np.pi / max(num, 1))
    # Rate varies per joint so they are not all moving in lockstep, which would
    # let a whole-body error hide as a common-mode offset.
    rate = 0.15 + 0.05 * (rank[None, :] % 5)
    t = np.arange(steps)[:, None]
    return default_pos[None, :] + amplitude * np.sin(rate * t + phase)


def targets_sha(joint_names: list[str], targets: np.ndarray) -> str:
    """Fingerprint the command sequence independently of joint ORDER.

    The two Isaac backends and MuJoCo each enumerate joints differently, so
    hashing the raw array would report "different commands" for identical
    physical commands. Columns are sorted by joint name first.
    """
    import hashlib

    order = np.argsort(np.array(joint_names))
    canonical = np.ascontiguousarray(np.asarray(targets)[:, order].round(9))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _record(joint_names, joint_pos, joint_vel, torque, root_pos, root_quat) -> dict:
    return {
        "joint_pos": {n: round(float(v), 9) for n, v in zip(joint_names, joint_pos)},
        "joint_vel": {n: round(float(v), 9) for n, v in zip(joint_names, joint_vel)},
        "applied_torque": (
            None
            if torque is None
            else {n: round(float(v), 9) for n, v in zip(joint_names, torque)}
        ),
        "root_pos": [round(float(v), 9) for v in root_pos],
        "root_quat": [round(float(v), 9) for v in root_quat],
    }


# --------------------------------------------------------------------------
# Comparison.
# --------------------------------------------------------------------------
def _worst(left: dict, right: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for name, lv in left.items():
        delta = abs(lv - right.get(name, lv))
        if delta > worst:
            worst, where = delta, name
    return worst, where


def compare(left_path: Path, right_path: Path, tolerance: float) -> int:
    left = json.loads(Path(left_path).read_text())
    right = json.loads(Path(right_path).read_text())
    print(f"left  : {left_path}  ({left['engine_label']})")
    print(f"right : {right_path}  ({right['engine_label']})")

    if left["joint_names"] != right["joint_names"]:
        print("\n[INFO] joint orders differ; comparing by name (as intended).")
    if left["targets_sha"] != right["targets_sha"]:
        print("\n[FAIL] the two runs were driven by different target sequences.")
        return 2

    lt, rt = left["trace"], right["trace"]
    if not lt or not rt:
        print("\n[FAIL] empty trace.")
        return 2

    # Step 0 is pre-physics. If it differs, nothing after it is interpretable.
    p0, _ = _worst(lt[0]["joint_pos"], rt[0]["joint_pos"])
    v0, _ = _worst(lt[0]["joint_vel"], rt[0]["joint_vel"])
    print(f"\ninitial state (step 0, pre-physics): joint_pos {p0:.2e}  vel {v0:.2e}")
    if max(p0, v0) > tolerance:
        print(
            "[FAIL] the runs did not start from the same state, so any later\n"
            "       divergence is not attributable to the engines. Fix the reset\n"
            "       before reading the table below."
        )
        return 2
    print("[PASS] identical initial state.\n")

    print(
        f"{'step':>4s} {'max|dq| rad':>12s} {'joint':>26s} "
        f"{'max|dqd| rad/s':>15s} {'|droot| m':>10s}"
    )
    first = None
    for index in range(min(len(lt), len(rt))):
        dq, where = _worst(lt[index]["joint_pos"], rt[index]["joint_pos"])
        dv, _ = _worst(lt[index]["joint_vel"], rt[index]["joint_vel"])
        droot = float(
            np.linalg.norm(
                np.array(lt[index]["root_pos"]) - np.array(rt[index]["root_pos"])
            )
        )
        if first is None and index > 0 and dq > tolerance:
            first = (index, where, dq)
        print(f"{index:4d} {dq:12.3e} {where:>26s} {dv:15.3e} {droot:10.3e}")

    print()
    if first is None:
        print(f"[PASS] the engines agree within {tolerance:g} for every step.")
        return 0
    index, where, dq = first
    print(
        f"[DIVERGENCE] first exceeds {tolerance:g} at control step {index}, "
        f"on {where} ({dq:.3e} rad).\n"
        "             Same initial state, same joint-position targets, so this is\n"
        "             the engines' own integration / actuator / contact response."
    )
    return 1


args_cli, hydra_args = parser.parse_known_args()
if args_cli.compare:
    raise SystemExit(
        compare(
            Path(args_cli.compare[0]), Path(args_cli.compare[1]), args_cli.tolerance
        )
    )
if args_cli.engine is None:
    parser.error("--engine is required unless --compare is used")


def run_mujoco() -> dict:
    """CPU MuJoCo with the rev_1_0 model and the SONIC actuator parameters."""
    import mujoco

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "bench"))
    from mujoco_reference_tracking_baseline import _isaaclab_joint_params

    spec = mujoco.MjSpec.from_file(str(args_cli.mjcf))
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        floor = spec.worldbody.add_geom()
        floor.name = "probe_floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
    model = spec.compile()

    # Match Isaac Lab's integration contract exactly: sim.dt 0.005, decimation 4,
    # implicitfast. Anything else compares two timesteps, not two engines.
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    decimation = 4

    params = _isaaclab_joint_params()
    jnames = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    for actuator in range(model.nu):
        joint = jnames[model.actuator_trnid[actuator, 0]]
        entry = params[joint]
        model.actuator_gaintype[actuator] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator, :] = 0.0
        model.actuator_biasprm[actuator, :] = 0.0
        model.actuator_gainprm[actuator, 0] = entry["stiffness"]
        model.actuator_biasprm[actuator, 1] = -entry["stiffness"]
        model.actuator_biasprm[actuator, 2] = -entry["damping"]
        model.actuator_forcelimited[actuator] = 1
        model.actuator_forcerange[actuator] = [-entry["effort"], entry["effort"]]
        model.dof_armature[model.jnt_dofadr[model.actuator_trnid[actuator, 0]]] = entry[
            "armature"
        ]

    actuated = [jnames[model.actuator_trnid[a, 0]] for a in range(model.nu)]
    data = mujoco.MjData(model)
    # The reference pose is DECLARED (see REFERENCE_POSE), not inherited from
    # whatever each engine happens to default to. Isaac's `default_joint_pos` for
    # this task is all zeros while the SONIC `init_state` is a crouched stance;
    # centering the two engines on different poses silently makes them different
    # experiments, which is exactly what the first run of this probe did.
    default_pos = np.full(len(actuated), REFERENCE_POSE)
    data.qpos[7:] = default_pos
    data.qvel[:] = 0.0
    # Held aloft: the divergence under test appears within a few control steps,
    # long before ground contact, and removing the floor interaction removes the
    # one term whose two implementations are known to be different pipelines.
    data.qpos[0:3] = [0.0, 0.0, float(args_cli.start_height)]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Settle under gravity/contact holding the reference pose, so a ground start
    # begins from a resting state instead of a drop transient. Omitted here in an
    # earlier revision while the Isaac side had it, which made CPU MuJoCo the only
    # engine reporting its un-stepped initial state -- the probe's step-0 gate
    # caught it, which is what the gate is for.
    for _ in range(int(args_cli.settle_steps)):
        data.ctrl[:] = default_pos
        for _ in range(decimation):
            mujoco.mj_step(model, data)

    targets = target_sequence(actuated, default_pos, args_cli.steps, args_cli.amplitude)
    trace = [
        _record(
            actuated,
            np.array(data.qpos[7:]),
            np.array(data.qvel[6:]),
            None,
            data.qpos[0:3],
            data.qpos[3:7],
        )
    ]
    for step in range(args_cli.steps):
        data.ctrl[:] = targets[step]
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        trace.append(
            _record(
                actuated,
                np.array(data.qpos[7:]),
                np.array(data.qvel[6:]),
                np.array(data.actuator_force),
                data.qpos[0:3],
                data.qpos[3:7],
            )
        )
    return {
        "engine_label": "cpu-mujoco (rev_1_0 + SONIC params)",
        "joint_names": actuated,
        "targets_sha": targets_sha(actuated, targets),
        "timestep": float(model.opt.timestep),
        "decimation": decimation,
        "trace": trace,
    }


def main() -> int:
    if args_cli.engine == "mujoco":
        report = run_mujoco()
    else:
        report = run_isaac()
    print(f"\nengine     : {report['engine_label']}")
    print(f"targets sha: {report['targets_sha'][:16]}")
    print(f"steps      : {len(report['trace']) - 1}")
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] wrote {args_cli.output}")
    return 0


if args_cli.engine == "isaac":
    sys.argv = [sys.argv[0]] + hydra_args
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import torch

    import isaaclab_imitation.tasks  # noqa: F401
    from imitation_experiments.audit.backend_determinism import (
        apply_randomization_profile,
        pin_reference_start,
    )

    def run_isaac() -> dict:
        holder: dict = {}

        @hydra_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
        def _build(env_cfg, agent_cfg):
            env_cfg.scene.num_envs = 1
            env_cfg.seed = args_cli.seed
            pin_reference_start(env_cfg, start_frame=0)
            apply_randomization_profile(env_cfg, "none")
            env_cfg.observations.policy.enable_corruption = False
            # Held aloft, matching the MuJoCo reference: no reference teleport,
            # no contact.
            env_cfg.events.reset_reference_state = None
            env_cfg.scene.robot.init_state.pos = (
                0.0,
                0.0,
                float(args_cli.start_height),
            )

            env = gym.make(args_cli.task, cfg=env_cfg)
            base = env.unwrapped
            env.reset(seed=args_cli.seed)
            robot = base.scene["robot"]
            term = base.action_manager.get_term("joint_pos")
            names = list(term._joint_names)
            offset = term._offset[0].detach().cpu().numpy().astype(np.float64)
            scale = term._scale
            scale = (
                scale[0].detach().cpu().numpy().astype(np.float64)
                if hasattr(scale, "ndim") and getattr(scale, "ndim", 0) == 2
                else float(scale)
            )

            targets = target_sequence(
                names,
                np.full(len(names), REFERENCE_POSE),
                args_cli.steps,
                args_cli.amplitude,
            )

            def snapshot(torque):
                jp = robot.data.joint_pos.torch[0].detach().cpu().numpy()
                jv = robot.data.joint_vel.torch[0].detach().cpu().numpy()
                live = list(robot.joint_names)
                order = [live.index(n) for n in names]
                return _record(
                    names,
                    jp[order],
                    jv[order],
                    None if torque is None else torque[order],
                    robot.data.root_pos_w.torch[0].detach().cpu().numpy(),
                    robot.data.root_quat_w.torch[0].detach().cpu().numpy(),
                )

            with torch.inference_mode():
                for _ in range(int(args_cli.settle_steps)):
                    hold = (np.full(len(names), REFERENCE_POSE) - offset) / scale
                    env.step(
                        torch.as_tensor(
                            hold, dtype=torch.float32, device=base.device
                        ).unsqueeze(0)
                    )
            trace = [snapshot(None)]
            with torch.inference_mode():
                for step in range(args_cli.steps):
                    # Invert `target = offset + scale * action` so the physical
                    # command equals MuJoCo's ctrl exactly.
                    action = (targets[step] - offset) / scale
                    env.step(
                        torch.as_tensor(
                            action, dtype=torch.float32, device=base.device
                        ).unsqueeze(0)
                    )
                    tau = robot.data.applied_torque.torch[0].detach().cpu().numpy()
                    trace.append(snapshot(tau))
            holder.update(
                {
                    "engine_label": f"isaac-{type(env_cfg.sim.physics).__name__}",
                    "joint_names": names,
                    "targets_sha": targets_sha(names, targets),
                    "timestep": float(env_cfg.sim.dt),
                    "decimation": int(env_cfg.decimation),
                    "trace": trace,
                }
            )
            env.close()

        _build()
        return holder

    if __name__ == "__main__":
        import traceback

        try:
            raise SystemExit(main())
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            simulation_app.close()
else:
    if __name__ == "__main__":
        raise SystemExit(main())
