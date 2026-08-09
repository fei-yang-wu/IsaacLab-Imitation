#!/usr/bin/env python3
# ruff: noqa: E402
"""Run NVIDIA's released SONIC G1 tracker closed-loop in our env (Tier 2).

This drives the public GEAR-SONIC checkpoint (``sonic_release/last.pt``) as a
low-level tracker inside ``Isaac-Imitation-G1-Sonic-v0`` on our BONES-SEED
references. It bypasses the IPMD command interface: SONIC's actor is a separate
encoder -> FSQ -> decoder module, so this harness reads the reference window and
proprioception from the env, calls ``SonicReleaseActor``, and applies the
resulting joint targets directly.

Verified alignment (see wiki/sonic-release-checkpoint-tier2.md):
actuators/action-scale match SONIC (via the overrides below), the encoder input
is qpos+qvel+anchor_ori(pelvis) at stride 5, and proprioception is term-major
[ang_vel, joint_pos_rel, joint_vel_rel, last_action, gravity] x10. SONIC's
29-vectors use the interleaved IsaacLab joint order; the reference and action
term already use it, but proprioception read from the live Newton articulation
(grouped SDK order) is permuted into it.

Three calibrations only close on a live rollout and this script fails loud on
each: the stride-5 future-frame origin, the history flatten direction, and the
pelvis-anchor gravity convention.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

# CU130 split-runtime bootstrap (ICE only); no-op elsewhere. Mirrors the
# SkillCommander evaluator so this runs on the same compute-only GPUs.
if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1":
    _KIT_PY = "/isaac-sim/kit/python"
    sys.path[:] = [
        _p
        for _p in sys.path
        if not (
            os.path.realpath(_p or ".").startswith(_KIT_PY)
            and "site-packages" not in os.path.realpath(_p or ".")
        )
    ]

from runtime_bootstrap import assert_kit_not_loaded, install_kit_import_guard

STRICT_KITLESS = "--assert-kitless" in sys.argv[1:]
if STRICT_KITLESS:
    install_kit_import_guard()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Imitation-G1-Sonic-v0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/hsstorage/fwu91/sonic_release/last.pt"),
    )
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor_body", default="pelvis")
    parser.add_argument("--action_clip", type=float, default=20.0)
    parser.add_argument("--encoder_frames", type=int, default=10)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--motion_names", nargs="*", default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--assert-kitless", action="store_true", dest="assert_kitless")
    # Own flags are parsed here; the remaining ``key=value`` Hydra overrides are
    # left on sys.argv for ``resolve_task_config``'s Hydra pass (e.g.
    # physics=newton_mjwarp). This mirrors the SkillCommander evaluator.
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _apply_sonic_actuator_overrides(env_cfg, anchor_body: str) -> None:
    """Set SONIC's action scale, hip-pitch actuator gains, and the anchor body.

    ``ImitationG1SonicSurfaceEnvCfg`` supplies SONIC's resets, curriculum,
    rewards, and 10-step histories but inherits v2's action scale (MIMIC) and
    the torso anchor. We do **not** swap the robot cfg: our
    ``UNITREE_G1_29DOF_SONIC_CFG`` spawns from URDF, which needs Kit's converter
    and cannot load under `--assert-kitless` Newton. The USD robot preset is
    otherwise identical, so we patch in place:

    - action scale -> SONIC's (only the hip-pitch entry differs from MIMIC);
    - hip-pitch actuator gains -> 7520-22 on the newton/physx robot variants;
    - anchor body -> pelvis on the anchor-relative observation terms.
    """
    from isaaclab_imitation.assets.robots import UNITREE_G1_29DOF_SONIC_ACTION_SCALE
    from isaaclab_imitation.assets.robots.unitree import (
        ARMATURE_7520_22,
        DAMPING_7520_22,
        STIFFNESS_7520_22,
    )

    env_cfg.actions.joint_pos.scale = UNITREE_G1_29DOF_SONIC_ACTION_SCALE

    # Patch hip-pitch to SONIC's 7520-22 on whichever robot variants exist
    # (the preset carries default/physx/newton_mjwarp sub-configs).
    robot = env_cfg.scene.robot
    variants = [
        getattr(robot, name, None) for name in ("default", "physx", "newton_mjwarp")
    ]
    variants = [v for v in variants if v is not None] or [robot]
    for variant in variants:
        actuators = getattr(variant, "actuators", None)
        legs = actuators.get("legs") if isinstance(actuators, dict) else None
        if legs is None:
            continue
        for attr, value in (
            ("effort_limit_sim", 139.0),
            ("velocity_limit_sim", 20.0),
            ("stiffness", STIFFNESS_7520_22),
            ("damping", DAMPING_7520_22),
            ("armature", ARMATURE_7520_22),
        ):
            table = getattr(legs, attr, None)
            if isinstance(table, dict) and ".*_hip_pitch_joint" in table:
                table[".*_hip_pitch_joint"] = value

    # Anchor-relative observation terms default to torso; SONIC uses pelvis.
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name, None)
        if group is None:
            continue
        for term_name in ("expert_anchor_ori_b", "expert_anchor_pos_b"):
            term = getattr(group, term_name, None)
            if term is not None and "anchor_body_name" in getattr(term, "params", {}):
                term.params["anchor_body_name"] = anchor_body


def main() -> None:
    args = _parse_args()

    from isaaclab_tasks.utils import (
        compute_kit_requirements,
        launch_simulation,
        resolve_task_config,
    )

    # Import for its side effect: the ``gym.register`` calls for the
    # Isaac-Imitation-G1-* tasks run on package import.
    import isaaclab_imitation.tasks  # noqa: F401

    env_cfg, _agent_cfg = resolve_task_config(args.task, None)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)

    needs_kit, _, _ = compute_kit_requirements(env_cfg, args)
    if args.assert_kitless and needs_kit:
        raise RuntimeError("--assert-kitless requires physics=newton_mjwarp.")
    _apply_sonic_actuator_overrides(env_cfg, args.anchor_body)

    with launch_simulation(env_cfg, args):
        _run(env_cfg, args)
    if args.assert_kitless:
        assert_kit_not_loaded()


def _run(env_cfg, args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch

    from imitation_experiments.lowlevel.sonic_release_actor import (
        assemble_proprioception,
        load_sonic_release_actor,
        pack_encoder_window,
    )

    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.constants import (
        G1_29DOF_ISAACLAB_JOINT_NAMES,
    )

    raw = gym.make(args.task, cfg=env_cfg, render_mode=None).unwrapped
    device = raw.device
    actor = load_sonic_release_actor(args.checkpoint).to(device)
    num_envs = raw.num_envs

    # Joint-order alignment. SONIC's 29-vectors are in the interleaved IsaacLab
    # order (== G1_29DOF_ISAACLAB_JOINT_NAMES). Under Newton the live
    # articulation buffer (robot.data.joint_pos) is the *grouped* SDK order, so
    # proprioception must be permuted into the interleaved order. The reference
    # (expert_motion) and the action term already use the interleaved target
    # order, so those need no manual permutation. See
    # wiki/sonic-release-checkpoint-tier2.md and the Newton joint-order memory.
    prop_perm_list, _ = raw.robot.find_joints(
        list(G1_29DOF_ISAACLAB_JOINT_NAMES), preserve_order=True
    )
    prop_perm = torch.as_tensor(prop_perm_list, device=device, dtype=torch.long)
    print(
        f"[INFO] live physics joint order: {list(raw.robot.joint_names)}",
        flush=True,
    )
    print(
        f"[INFO] proprioception permutation (physics->SONIC): {prop_perm_list}",
        flush=True,
    )

    frames = int(args.encoder_frames)
    stride = int(args.frame_stride)
    # oldest -> newest ring buffers for the five proprioception terms
    history: dict[str, deque] = {
        name: deque(maxlen=frames)
        for name in ("gravity", "ang_vel", "jpos", "jvel", "act")
    }
    last_action = torch.zeros(num_envs, 29, device=device)

    def _proprioception() -> torch.Tensor:
        data = raw.robot.data
        jpos = (data.joint_pos - data.default_joint_pos)[:, prop_perm]
        jvel = (data.joint_vel - data.default_joint_vel)[:, prop_perm]
        current = {
            "gravity": data.projected_gravity_b,
            "ang_vel": data.root_ang_vel_b,
            "jpos": jpos,
            "jvel": jvel,
            "act": last_action,  # already SONIC/interleaved order
        }
        for name, value in current.items():
            buf = history[name]
            while len(buf) < frames:  # warm the ring with the reset state
                buf.append(value.clone())
            buf.append(value.clone())
        stacked = {name: torch.stack(list(buf), dim=1) for name, buf in history.items()}
        return assemble_proprioception(
            stacked["gravity"],
            stacked["ang_vel"],
            stacked["jpos"],
            stacked["jvel"],
            stacked["act"],
        )

    def _encoder_token() -> torch.Tensor:
        motion = raw.get_current_expert_window_term(
            term_name="expert_motion",
            past_steps=0,
            future_steps=frames - 1,
            frame_stride=stride,
        ).reshape(num_envs, frames, 58)
        anchor_ori = raw.get_current_expert_window_term(
            term_name="expert_anchor_ori_b",
            past_steps=0,
            future_steps=frames - 1,
            frame_stride=stride,
            anchor_body_name=args.anchor_body,
        ).reshape(num_envs, frames, 6)
        window = pack_encoder_window(motion[..., :29], motion[..., 29:58], anchor_ori)
        return actor.encode(window)

    raw.reset()
    survived = torch.zeros(num_envs, dtype=torch.bool, device=device)
    step_count = torch.zeros(num_envs, device=device)
    with torch.no_grad():
        for step in range(int(args.max_steps)):
            token = _encoder_token()
            proprioception = _proprioception()
            action = actor.decode(token, proprioception)
            action = action.clamp(-float(args.action_clip), float(args.action_clip))
            last_action = action
            _obs, _rew, terminated, truncated, _info = raw.step(action)
            done = terminated | truncated
            alive = ~done
            step_count = step_count + alive.float()
            survived = survived | truncated  # reached the horizon without falling
            if bool(done.all().item()):
                break

    fall_free = float((~terminated).float().mean().item())
    mean_steps = float(step_count.mean().item())
    print(
        f"[RESULT] envs={num_envs} mean_survival_steps={mean_steps:.1f} "
        f"fall_free_rate={fall_free:.3f}",
        flush=True,
    )
    raw.close()


if __name__ == "__main__":
    main()
