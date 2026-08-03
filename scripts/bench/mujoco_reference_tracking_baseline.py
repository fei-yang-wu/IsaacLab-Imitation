#!/usr/bin/env python3
"""Third-party reference for G1 joint tracking: plain MuJoCo, no Isaac Lab.

`diagnose_g1_dynamics.py` compares two Isaac Lab backends against each other.
When they disagree it cannot say which one is wrong, because both are the thing
under test. This probe adds an outside reference: stock MuJoCo, driven by the
same oracle law over the same reference motion, so a disagreement can be
attributed instead of merely observed.

MuJoCo is the right third party specifically because Isaac Lab's
``newton_mjwarp`` backend *is* MuJoCo Warp. Anything stock MuJoCo reproduces is
a property of the model and its parameters, not of Isaac Lab's wrapping.

**What the training run actually loads.** Isaac Lab spawns the G1 from
``g1_description/g1_29dof_rev_1_0.usd``; the vendored MJCF is read only by
``unitree_joint_order.py``, for its actuator name order. So the MJCF is *not*
the simulated model, and this probe does not claim it is. What makes the
comparison meaningful is that the runtime physical parameters are supplied by
``ImplicitActuatorCfg`` in ``assets/robots/unitree.py`` -- identically on both
backends -- and this probe injects those same numbers into MuJoCo
(``--actuator-params isaaclab``, the default). The USD contributes link inertia
and colliders, which the MJCF carries too.

**Why the sweep.** Unitree's own MuJoCo model (``unitree_mujoco``,
``unitree_robots/g1/g1_29dof.xml``) declares per-joint ``armature=0.01``,
``damping=0.05``, and ``frictionloss=0.2`` (``0.1`` for wrists), and runs at
MuJoCo's default ``timestep=0.002``. The repo's runtime configuration has
frictionloss 0, passive damping 0, wrist armature 0.00361-0.00425, and
``sim.dt=0.005``. Those are four independent differences against the vendor's
own validated setup, so the flags below change one at a time and the caller
reads which one moves the number.

Usage (from the repository root):

.. code-block:: bash

    # The repo's runtime parameters, as MJWarp sees them.
    pixi run -e isaaclab python scripts/bench/mujoco_reference_tracking_baseline.py \\
        --motion data/lafan1/npz/g1/dance1_subject1.npz --steps 300 \\
        --output logs/mujoco_baseline/repo.json

    # One variable changed.
    ... --frictionloss 0.2 --wrist-frictionloss 0.1 --output .../frictionloss.json
    ... --timestep 0.002 --output .../timestep.json

    # Every arm at once, into one table.
    ... --sweep --output logs/mujoco_baseline/sweep.json
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

# Unitree's own MuJoCo simulation model, for reference in comments and in the
# sweep's `unitree_all` arm. Values read from
# github.com/unitreerobotics/unitree_mujoco, unitree_robots/g1/g1_29dof.xml
# (default classes leg_motor / arm_motor / wrist_motor / ankle_motor /
# torso_motor), commit ae6a840.
UNITREE_ARMATURE = 0.01
UNITREE_JOINT_DAMPING = 0.05
UNITREE_FRICTIONLOSS = 0.2
UNITREE_WRIST_FRICTIONLOSS = 0.1
UNITREE_TIMESTEP = 0.002

# The repo's Isaac Lab runtime: sim.dt 0.005 with decimation 4 -> 50 Hz control.
REPO_TIMESTEP = 0.005
REPO_DECIMATION = 4

EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
# Below this pelvis height the robot has fallen; matches the spirit of the
# `base_too_low` termination rather than its exact threshold, which is defined
# against the reference rather than the floor.
FALL_HEIGHT_M = 0.4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--decimation", type=int, default=REPO_DECIMATION)
    parser.add_argument("--timestep", type=float, default=REPO_TIMESTEP)
    parser.add_argument(
        "--integrator",
        choices=("implicitfast", "implicit", "euler", "rk4"),
        default="implicitfast",
        help="MJWarp runs implicitfast; that is the default here.",
    )
    parser.add_argument(
        "--actuator-params",
        choices=("isaaclab", "model"),
        default="isaaclab",
        help=(
            "isaaclab: inject the repo's ImplicitActuatorCfg gains/armature/effort "
            "limits, which is what both Isaac backends run. model: keep whatever "
            "the MJCF declares."
        ),
    )
    parser.add_argument(
        "--armature",
        type=float,
        default=None,
        help="Override every joint's armature (Unitree's model uses 0.01).",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=None,
        help="Passive joint damping, on top of the actuator's kv (Unitree: 0.05).",
    )
    parser.add_argument(
        "--frictionloss",
        type=float,
        default=None,
        help="Coulomb joint friction (Unitree: 0.2).",
    )
    parser.add_argument(
        "--wrist-frictionloss",
        type=float,
        default=None,
        help="Frictionloss for wrist joints only (Unitree: 0.1).",
    )
    parser.add_argument(
        "--base",
        choices=("free", "fixed"),
        default="fixed",
        help=(
            "fixed welds the pelvis and isolates the actuator + integrator, which "
            "is what differs between the backends. free keeps the floating base, "
            "where an open-loop humanoid falls within ~1 s and the fall, not the "
            "actuator, dominates every metric."
        ),
    )
    parser.add_argument("--sweep", action="store_true", help="Run every arm.")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _isaaclab_joint_params() -> dict[str, dict[str, float]]:
    """Per-joint stiffness / damping / armature / effort from the repo config.

    Imported rather than transcribed so the probe cannot drift from the config
    the training runs actually use. This import does not need a running Kit.
    """
    import re

    from isaaclab_imitation.assets.robots.unitree import UNITREE_G1_29DOF_SONIC_CFG
    from isaaclab_imitation.assets.robots.unitree_joint_order import (
        UNITREE_G1_29DOF_SDK_JOINT_NAMES,
    )

    def _resolve(value, joint_name: str):
        """Mirror Isaac Lab's actuator resolution: dict of regex, or a scalar."""
        if not isinstance(value, dict):
            return value
        for pattern, entry in value.items():
            if re.fullmatch(pattern, joint_name):
                return entry
        return None

    params: dict[str, dict[str, float]] = {}
    for actuator in UNITREE_G1_29DOF_SONIC_CFG.actuators.values():
        expressions = list(actuator.joint_names_expr)
        for joint_name in UNITREE_G1_29DOF_SDK_JOINT_NAMES:
            if not any(re.fullmatch(p, joint_name) for p in expressions):
                continue
            entry = {
                "stiffness": _resolve(actuator.stiffness, joint_name),
                "damping": _resolve(actuator.damping, joint_name),
                "armature": _resolve(actuator.armature, joint_name),
                "effort": _resolve(actuator.effort_limit_sim, joint_name),
            }
            if any(v is None for v in entry.values()):
                raise RuntimeError(
                    f"Actuator config does not resolve every field for {joint_name}: "
                    f"{entry}"
                )
            params[joint_name] = {k: float(v) for k, v in entry.items()}
    missing = [n for n in UNITREE_G1_29DOF_SDK_JOINT_NAMES if n not in params]
    if missing:
        raise RuntimeError(f"No actuator group covers: {missing}")
    return params


def _compile_model(args: argparse.Namespace):
    """Compile the MJCF, adding a ground plane and optionally welding the base.

    ``--base fixed`` deletes the pelvis free joint. That is the mode that
    actually isolates the actuator: a free-base humanoid tracking a dance
    motion open loop falls within a second in *any* simulator, and once it is
    on the floor the joint error is dominated by the fall rather than by the
    actuator response under test. Welding the base removes falling, contact,
    and balance from the measurement and leaves the PD loop and the integrator.
    """
    import mujoco

    spec = mujoco.MjSpec.from_file(str(args.mjcf))
    has_floor = any(
        geom.type == mujoco.mjtGeom.mjGEOM_PLANE for geom in spec.worldbody.geoms
    )
    if not has_floor:
        # The vendored description is a bare URDF conversion with no scene;
        # Unitree's own simulation model ships one, so only add when absent.
        floor = spec.worldbody.add_geom()
        floor.name = "probe_floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 0.05]
    if args.base == "fixed":
        free_joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
        if not free_joints:
            raise RuntimeError("--base fixed needs a free joint to delete.")
        for joint in free_joints:
            spec.delete(joint)
    return spec.compile()


def _apply_parameters(model, args: argparse.Namespace) -> dict:
    """Write timestep, integrator, actuator law, and joint physics onto a model.

    Returns the record of what was applied, for the output JSON.
    """
    import mujoco

    model.opt.timestep = float(args.timestep)
    model.opt.integrator = {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[args.integrator]

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    applied: dict = {"actuator_params": args.actuator_params}

    if args.actuator_params == "isaaclab":
        params = _isaaclab_joint_params()
        for actuator_id in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            joint_id = model.actuator_trnid[actuator_id, 0]
            joint_name = joint_names[joint_id]
            entry = params.get(joint_name) or params.get(name)
            if entry is None:
                raise RuntimeError(f"No Isaac Lab actuator entry for {joint_name!r}.")
            # Isaac Lab's ImplicitActuator evaluates tau = kp*(q* - q) - kd*qd
            # inside the solver at every physics step. MuJoCo's `position`
            # actuator with kp/kv is exactly that law, and under an implicit
            # integrator it is also solved implicitly -- so this matches the
            # semantics rather than approximating it with a held torque.
            kp, kv = entry["stiffness"], entry["damping"]
            model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[actuator_id, :] = 0.0
            model.actuator_biasprm[actuator_id, :] = 0.0
            model.actuator_gainprm[actuator_id, 0] = kp
            model.actuator_biasprm[actuator_id, 1] = -kp
            model.actuator_biasprm[actuator_id, 2] = -kv
            model.actuator_forcelimited[actuator_id] = 1
            model.actuator_forcerange[actuator_id] = [-entry["effort"], entry["effort"]]
            model.actuator_ctrllimited[actuator_id] = 0
            # Armature is a joint (dof) property, not an actuator one.
            dof = model.jnt_dofadr[joint_id]
            model.dof_armature[dof] = entry["armature"]
        applied["armature_source"] = "isaaclab_actuator_cfg"

    # Overrides, applied after the base parameters so a sweep arm wins.
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] in (
            mujoco.mjtJoint.mjJNT_FREE,
            mujoco.mjtJoint.mjJNT_BALL,
        ):
            continue
        dof = model.jnt_dofadr[joint_id]
        name = joint_names[joint_id] or ""
        if args.armature is not None:
            model.dof_armature[dof] = args.armature
        if args.joint_damping is not None:
            model.dof_damping[dof] = args.joint_damping
        is_wrist = "wrist" in name
        if is_wrist and args.wrist_frictionloss is not None:
            model.dof_frictionloss[dof] = args.wrist_frictionloss
        elif args.frictionloss is not None:
            model.dof_frictionloss[dof] = args.frictionloss

    applied.update(
        {
            "timestep": float(model.opt.timestep),
            "integrator": args.integrator,
            "armature_override": args.armature,
            "joint_damping_override": args.joint_damping,
            "frictionloss_override": args.frictionloss,
            "wrist_frictionloss_override": args.wrist_frictionloss,
        }
    )
    return applied


def _reference(motion_path: Path, model) -> dict:
    """Load the NPZ and map its joint columns onto the MuJoCo joint order."""
    import mujoco

    data = np.load(motion_path, allow_pickle=True)
    npz_joint_names = [str(n) for n in data["joint_names"]]

    actuated = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    missing = [n for n in actuated if n not in npz_joint_names]
    if missing:
        raise RuntimeError(f"Motion {motion_path} lacks joints: {missing}")
    # Map by NAME. The MuJoCo joint order and the dataset order are unrelated,
    # and this is exactly the mistake the Isaac backends were audited for.
    columns = [npz_joint_names.index(n) for n in actuated]
    return {
        "joint_names": actuated,
        "joint_pos": np.asarray(data["joint_pos"], np.float64)[:, columns],
        "joint_vel": np.asarray(data["joint_vel"], np.float64)[:, columns],
        "root_pos": np.asarray(data["root_pos"], np.float64),
        # NPZ quaternions are WXYZ, which is also MuJoCo's convention, so the
        # free joint takes them unchanged (Isaac Lab 3.0 is the XYZW outlier).
        "root_quat": np.asarray(data["root_quat"], np.float64),
        "num_frames": int(np.asarray(data["joint_pos"]).shape[0]),
    }


def _run(model, reference: dict, args: argparse.Namespace) -> dict:
    import mujoco

    data = mujoco.MjData(model)
    start = int(args.start_frame)
    joint_names = reference["joint_names"]
    fixed_base = args.base == "fixed"
    # With the base welded there is no free joint, so the joint block starts at
    # qpos[0]; with a floating base it starts after pos(3) + quat(4).
    q0, v0 = (0, 0) if fixed_base else (7, 6)

    # Teleport onto the reference frame, matching the Isaac reset event.
    if not fixed_base:
        data.qpos[0:3] = reference["root_pos"][start]
        data.qpos[3:7] = reference["root_quat"][start]
    data.qpos[q0:] = reference["joint_pos"][start]
    data.qvel[:] = 0.0
    data.qvel[v0:] = reference["joint_vel"][start]
    mujoco.mj_forward(model, data)

    steps = int(args.steps)
    last = reference["num_frames"] - 1
    error_sum = np.zeros(len(joint_names))
    error_max = np.zeros(len(joint_names))
    chatter_sum = np.zeros(len(joint_names))
    torque_sum = 0.0
    samples = 0
    fell_at = None
    diverged = False
    previous_q = np.array(data.qpos[q0:], dtype=np.float64)

    for step in range(steps):
        frame = min(start + step, last)
        target_frame = min(frame + 1, last)
        # The oracle action: command the reference's NEXT pose, the same law
        # `imitation_experiments.lowlevel.oracle_action.live_oracle_action`
        # feeds the Isaac probe.
        data.ctrl[:] = reference["joint_pos"][target_frame]
        for _ in range(int(args.decimation)):
            mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)):
            diverged = True
            break

        if not fixed_base and fell_at is None and float(data.qpos[2]) < FALL_HEIGHT_M:
            fell_at = step
        # Stop accumulating once the robot is on the floor: past that point the
        # error measures the fall, not the actuator, and it swamps every
        # parameter difference the sweep is trying to separate.
        if fell_at is not None:
            continue

        q = np.array(data.qpos[q0:], dtype=np.float64)
        error = np.abs(q - reference["joint_pos"][frame])
        error_sum += error
        error_max = np.maximum(error_max, error)
        # Step-to-step motion of the joint itself. A PD loop that is marginally
        # stable at this timestep shows up here as chatter long before it shows
        # up as mean error.
        chatter_sum += np.abs(q - previous_q)
        previous_q = q
        torque_sum += float(np.abs(np.asarray(data.actuator_force)).mean())
        samples += 1

    if samples == 0:
        raise RuntimeError(
            "No measured steps: the robot fell or diverged immediately. Use "
            "--base fixed to isolate the actuator, or a later --start-frame."
        )

    mean_error = error_sum / samples
    mean_chatter = chatter_sum / samples
    return {
        "joint_pos_mae_mean_rad": float(mean_error.mean()),
        "joint_pos_mae_by_joint_rad": {
            n: float(v) for n, v in zip(joint_names, mean_error, strict=True)
        },
        "joint_pos_error_max_rad": float(error_max.max()),
        "joint_pos_error_max_by_joint_rad": {
            n: float(v) for n, v in zip(joint_names, error_max, strict=True)
        },
        "joint_chatter_mean_rad": float(mean_chatter.mean()),
        "joint_chatter_by_joint_rad": {
            n: float(v) for n, v in zip(joint_names, mean_chatter, strict=True)
        },
        "applied_torque_abs_mean_nm": torque_sum / samples,
        "steps_measured": samples,
        "steps_completed": samples,
        "diverged": diverged,
        "fell_at_step": fell_at,
        "final_pelvis_height_m": float(data.qpos[2]),
    }


SWEEP_ARMS: tuple[tuple[str, dict], ...] = (
    ("repo_runtime", {}),
    ("plus_timestep_0.002", {"timestep": UNITREE_TIMESTEP}),
    (
        "plus_frictionloss",
        {
            "frictionloss": UNITREE_FRICTIONLOSS,
            "wrist_frictionloss": UNITREE_WRIST_FRICTIONLOSS,
        },
    ),
    ("plus_joint_damping", {"joint_damping": UNITREE_JOINT_DAMPING}),
    ("plus_armature_0.01", {"armature": UNITREE_ARMATURE}),
    (
        "unitree_all",
        {
            "timestep": UNITREE_TIMESTEP,
            "frictionloss": UNITREE_FRICTIONLOSS,
            "wrist_frictionloss": UNITREE_WRIST_FRICTIONLOSS,
            "joint_damping": UNITREE_JOINT_DAMPING,
            "armature": UNITREE_ARMATURE,
        },
    ),
)


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    if not args.motion.is_file():
        raise SystemExit(f"[FAIL] motion not found: {args.motion}")
    if not args.mjcf.is_file():
        raise SystemExit(f"[FAIL] MJCF not found: {args.mjcf}")

    arms = SWEEP_ARMS if args.sweep else (("single", {}),)
    results: dict[str, dict] = {}
    for arm_name, overrides in arms:
        arm_args = argparse.Namespace(**vars(args))
        for key, value in overrides.items():
            setattr(arm_args, key, value)
        # Decimation follows the timestep so every arm holds 50 Hz control:
        # changing the physics rate must not silently change the control rate.
        if overrides.get("timestep"):
            arm_args.decimation = int(
                round(args.decimation * args.timestep / overrides["timestep"])
            )
        model = _compile_model(arm_args)
        applied = _apply_parameters(model, arm_args)
        outcome = _run(model, _reference(args.motion, model), arm_args)
        applied["decimation"] = arm_args.decimation
        applied["control_hz"] = 1.0 / (applied["timestep"] * arm_args.decimation)
        results[arm_name] = {"applied": applied, **outcome}

    report = {
        "mjcf": str(args.mjcf),
        "motion": str(args.motion),
        "start_frame": int(args.start_frame),
        "steps": int(args.steps),
        "arms": results,
    }

    width = max(len(name) for name in results)
    print(
        f"\n{'arm':<{width}}  {'jointMAE':>9s} {'chatter':>9s} {'torqueNm':>9s} "
        f"{'wristRollL':>11s} {'wrChatter':>10s} {'steps':>6s} {'ctrlHz':>7s}"
    )
    for name, entry in results.items():
        wrist = entry["joint_pos_mae_by_joint_rad"].get("left_wrist_roll_joint", 0.0)
        wrist_chatter = entry["joint_chatter_by_joint_rad"].get(
            "left_wrist_roll_joint", 0.0
        )
        print(
            f"{name:<{width}}  {entry['joint_pos_mae_mean_rad']:9.4f} "
            f"{entry['joint_chatter_mean_rad']:9.4f} "
            f"{entry['applied_torque_abs_mean_nm']:9.2f} {wrist:11.4f} "
            f"{wrist_chatter:10.4f} {entry['steps_measured']:6d} "
            f"{entry['applied']['control_hz']:7.1f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\n[INFO] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
