#!/usr/bin/env python3
# ruff: noqa: E402
"""Dump the MuJoCo model Isaac Lab's Newton backend actually simulates.

`scripts/bench/mujoco_reference_tracking_baseline.py` established that stock
MuJoCo, given the repo's ``ImplicitActuatorCfg`` numbers, tracks the G1
reference about as well as PhysX does -- and nothing like Isaac Lab's
``newton_mjwarp`` backend, which is 14-31x worse on the light joints. Since
that backend *is* MuJoCo Warp, the parameters it ends up simulating cannot be
the ones the config asks for.

This reaches into ``NewtonManager._solver.mjw_model`` after the environment is
built and prints, per joint name, the values MuJoCo Warp holds: armature,
damping, frictionloss, joint range, and the actuator gain / bias / force range
that encode the PD law. Compare them against the config (printed alongside) and
against the stock-MuJoCo model to find which one is wrong.

Usage (from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/audit/dump_mjwarp_model_contract.py \\
        --task Isaac-Imitation-G1-v2 --num_envs 2 \\
        --output logs/mjwarp_contract/newton.json \\
        physics=newton_mjwarp \\
        env.data.manifest=./data/lafan1/manifests/g1_lafan1_manifest.json \\
        env.data.cache_dir=./data/lafan1/zarr/g1_hl_diffsr env.data.cache_refresh=false
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Imitation-G1-v2")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=None)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
import numpy as np

import isaaclab_imitation.tasks  # noqa: F401
from imitation_experiments.audit.backend_determinism import apply_randomization_profile


def _as_numpy(value):
    """Warp array, CUDA torch tensor, or ndarray -> host ndarray."""
    if hasattr(value, "detach"):  # torch, possibly on CUDA
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):  # warp
        return np.asarray(value.numpy())
    return np.asarray(value)


def _per_world(array: np.ndarray, count: int) -> np.ndarray:
    """MuJoCo Warp batches most model fields over worlds; take world 0.

    Fields that were not batched come back already the right length, so this
    only strips a leading world axis when there is one.
    """
    array = np.asarray(array)
    if array.ndim >= 2 and array.shape[0] != count:
        return array[0]
    return array


@hydra_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
def main(env_cfg, agent_cfg):
    from isaaclab_newton.physics.newton_manager import NewtonManager

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Startup mass randomization scales torso and both wrist_yaw links; with it
    # on, the body-mass table below reports the dice roll rather than the asset.
    apply_randomization_profile(env_cfg, "none")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset(seed=args_cli.seed)

    solver = NewtonManager._solver
    if solver is None or getattr(solver, "mjw_model", None) is None:
        raise RuntimeError(
            "No MuJoCo Warp model on the solver. This audit requires "
            "physics=newton_mjwarp."
        )
    model = solver.mjw_model

    robot = base.scene["robot"]
    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)

    dof_armature = _per_world(_as_numpy(model.dof_armature), num_joints)
    dof_damping = _per_world(_as_numpy(model.dof_damping), num_joints)
    dof_frictionloss = _per_world(_as_numpy(model.dof_frictionloss), num_joints)
    gainprm = _per_world(_as_numpy(model.actuator_gainprm), num_joints)
    biasprm = _per_world(_as_numpy(model.actuator_biasprm), num_joints)
    forcerange = _per_world(_as_numpy(model.actuator_forcerange), num_joints)

    # A floating-base articulation contributes 6 leading dofs that are not
    # joints. Anything else means the layout is not what this audit assumes,
    # and a silently misaligned dump is worse than none.
    offset = len(dof_armature) - num_joints
    if offset not in (0, 6):
        raise RuntimeError(
            f"Cannot align {len(dof_armature)} MuJoCo dofs to {num_joints} named "
            f"joints (offset {offset}); refusing to emit a misaligned dump."
        )

    # What Isaac Lab believes it configured, read back off the live articulation.
    cfg_stiffness = _as_numpy(robot.data.joint_stiffness.torch)[0]
    cfg_damping = _as_numpy(robot.data.joint_damping.torch)[0]
    cfg_armature = _as_numpy(robot.data.joint_armature.torch)[0]

    per_joint = {}
    for index, name in enumerate(joint_names):
        dof = index + offset
        actuator = index if index < len(gainprm) else None
        entry = {
            "mjwarp_armature": float(dof_armature[dof]),
            "mjwarp_damping": float(dof_damping[dof]),
            "mjwarp_frictionloss": float(dof_frictionloss[dof]),
            "isaaclab_armature": float(cfg_armature[index]),
            "isaaclab_stiffness": float(cfg_stiffness[index]),
            "isaaclab_damping": float(cfg_damping[index]),
        }
        if actuator is not None:
            # MuJoCo position servo: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kv.
            entry["mjwarp_actuator_gainprm0"] = float(gainprm[actuator][0])
            entry["mjwarp_actuator_biasprm1"] = float(biasprm[actuator][1])
            entry["mjwarp_actuator_biasprm2"] = float(biasprm[actuator][2])
            entry["mjwarp_actuator_forcerange"] = [
                float(v) for v in forcerange[actuator]
            ]
        per_joint[name] = entry

    # The solver overwrites `opt.timestep` from the dt handed to `step()`, so
    # the value sitting on a freshly built model is only MuJoCo's default.
    # Read it again after real stepping: that is the timestep actually
    # integrated, and the one that has to equal sim.dt / num_substeps.
    timestep_before = float(np.ravel(_as_numpy(model.opt.timestep))[0])
    import torch

    zero_action = torch.zeros(
        (args_cli.num_envs, base.action_manager.total_action_dim), device=base.device
    )
    for _ in range(3):
        env.step(zero_action)
    timestep_after = float(np.ravel(_as_numpy(model.opt.timestep))[0])

    # Link inertia, per body. An error here would scale with how little inertia
    # a link has of its own, i.e. worst at the wrists and mildest at the hips --
    # which is the exact gradient the Newton-vs-PhysX tracking error follows, so
    # it has to be checked rather than assumed.
    body_names = list(robot.body_names)
    isaaclab_mass = _as_numpy(robot.data.default_mass)[0]
    isaaclab_inertia = _as_numpy(robot.data.default_inertia)[0]
    per_body = {
        name: {
            "isaaclab_mass": float(isaaclab_mass[index]),
            "isaaclab_inertia": [
                float(v) for v in np.ravel(isaaclab_inertia[index])[:9]
            ],
        }
        for index, name in enumerate(body_names)
    }
    mjw_mass = _per_world(_as_numpy(model.body_mass), len(body_names))
    mjw_inertia = _per_world(_as_numpy(model.body_inertia), len(body_names))
    report_bodies_aligned = len(mjw_mass) - len(body_names)
    if report_bodies_aligned in (0, 1):  # MuJoCo adds a world body at index 0
        for index, name in enumerate(body_names):
            entry = per_body[name]
            entry["mjwarp_mass"] = float(mjw_mass[index + report_bodies_aligned])
            entry["mjwarp_inertia"] = [
                float(v) for v in np.ravel(mjw_inertia[index + report_bodies_aligned])
            ]

    # Whether MuJoCo generates its own contacts, or Newton's collision pipeline
    # feeds them in. Reported because a `use_mujoco_contacts` override that never
    # reaches the solver looks exactly like "contacts make no difference".
    solver_cfg = getattr(env_cfg.sim.physics, "solver_cfg", None)
    contact_mode = {
        "cfg_use_mujoco_contacts": bool(
            getattr(solver_cfg, "use_mujoco_contacts", False)
        ),
        "mjw_run_collision_detection": bool(
            np.ravel(_as_numpy(model.opt.run_collision_detection))[0]
        )
        if hasattr(model.opt, "run_collision_detection")
        else None,
        "newton_needs_collision_pipeline": bool(
            NewtonManager._needs_collision_pipeline
        ),
    }
    print(f"\ncontact mode: {contact_mode}")

    # Every solver option MuJoCo Warp is running with, so it can be diffed
    # field-by-field against the CPU MuJoCo reference rather than eyeballed.
    mjw_opt = {}
    for field in dir(model.opt):
        if field.startswith("_"):
            continue
        try:
            value = getattr(model.opt, field)
        except Exception:  # noqa: BLE001 - option tables vary by version
            continue
        if callable(value):
            continue
        try:
            flat = np.ravel(_as_numpy(value))
        except Exception:  # noqa: BLE001
            continue
        if flat.size == 0 or flat.size > 8:
            continue
        mjw_opt[field] = [float(v) for v in flat]
    print("mjw opt: " + json.dumps(mjw_opt, sort_keys=True))

    report = {
        "task": args_cli.task,
        "physics_cfg": type(env_cfg.sim.physics).__name__,
        "contact_mode": contact_mode,
        "mjw_opt": mjw_opt,
        "per_body": per_body,
        "body_names": body_names,
        "sim_dt": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "mjwarp_timestep_before_stepping": timestep_before,
        "mjwarp_timestep_after_stepping": timestep_after,
        "mjwarp_timestep": timestep_after,
        "mjwarp_integrator": int(np.ravel(_as_numpy(model.opt.integrator))[0]),
        "num_mujoco_dofs": int(len(dof_armature)),
        "dof_offset": int(offset),
        "joint_names": joint_names,
        "per_joint": per_joint,
    }

    print(
        f"\n{'joint':28s} {'arm(mjw)':>9s} {'arm(cfg)':>9s} {'kp(mjw)':>9s} "
        f"{'kp(cfg)':>9s} {'kv(mjw)':>9s} {'kv(cfg)':>9s} {'fric':>6s}"
    )
    for name, entry in per_joint.items():
        print(
            f"{name:28s} {entry['mjwarp_armature']:9.5f} "
            f"{entry['isaaclab_armature']:9.5f} "
            f"{entry.get('mjwarp_actuator_gainprm0', float('nan')):9.3f} "
            f"{entry['isaaclab_stiffness']:9.3f} "
            f"{-entry.get('mjwarp_actuator_biasprm2', float('nan')):9.3f} "
            f"{entry['isaaclab_damping']:9.3f} "
            f"{entry['mjwarp_frictionloss']:6.3f}"
        )
    substeps = getattr(env_cfg.sim.physics, "num_substeps", 1) or 1
    expected = float(env_cfg.sim.dt) / float(substeps)
    print(
        f"\ntimestep: sim.dt={report['sim_dt']} num_substeps={substeps} "
        f"-> expected {expected:g}"
    )
    print(
        f"          mjwarp before stepping={timestep_before:g} "
        f"after stepping={timestep_after:g} integrator={report['mjwarp_integrator']}"
    )
    if abs(timestep_after - expected) > 1e-9:
        print(
            f"[FAIL] MuJoCo Warp integrates at {timestep_after:g} s while Isaac Lab "
            f"advances its clock by {expected:g} s per physics step.\n"
            f"       Physics then runs {expected / timestep_after:.2f}x slower than "
            "the environment believes, so the robot falls behind a reference that\n"
            "       keeps advancing at the full rate."
        )
    else:
        print("[PASS] MuJoCo Warp integrates at the timestep Isaac Lab expects.")

    print(f"\n{'body':28s} {'mass(mjw)':>10s} {'mass(IL)':>10s} {'inertia(mjw)':>34s}")
    for name in body_names:
        entry = per_body[name]
        inertia = entry.get("mjwarp_inertia")
        rendered = (
            " ".join(f"{v:10.6f}" for v in inertia[:3]) if inertia else "<unaligned>"
        )
        print(
            f"{name:28s} {entry.get('mjwarp_mass', float('nan')):10.4f} "
            f"{entry['isaaclab_mass']:10.4f} {rendered:>34s}"
        )

    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] wrote {args_cli.output}")
    env.close()


if __name__ == "__main__":
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
