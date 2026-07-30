#!/usr/bin/env python3
"""Streaming-equivalence certificate for the reduced explicit interfaces.

Before spending ~1B frames of controller training on a new command space, prove
that the space's packet survives the 5 Hz publish / 50 Hz consume round trip.

Method (one live env, one pass, no trained policy needed):

    streamed = the phase-aligned slot of the held 10-frame packet -- exactly the
               tensor the actor will read under ``full_body_chunk_current_slot``
    direct   = the live single-frame reference command in the robot's *current*
               anchor frame -- what the actor would read at 50 Hz with no
               chunking at all

Both are read every control step from the same env. ``get_current_expert_window_term``
does not consult ``command_observation_source``, so it bypasses the packet and is
a genuine independent ground truth rather than a second read of the same buffer.

They must agree to float precision at EVERY hold phase 0..9, including across
asynchronous resets. A packet frame published at renewal time t0 for time
t0+p, re-expressed from the t0 anchor frame into the t0+p anchor frame, IS the
reference at t0+p in the current frame -- so any disagreement is a real defect
in the adapter (a phase misalignment, a stale anchor, a wrong body/joint order,
or a width mismatch), not an approximation.

Random actions are used deliberately: the robot must actually move between
renewal and consumption, otherwise the anchor re-expression is the identity and
the test proves nothing.

Usage:
    pixi run -e isaaclab python \\
        experiments/campaigns/2026-07-23-lafan1-planner-capacity/smoke_test_reduced_interface_streaming.py \\
        --motion_manifest data/lafan1/manifests/g1_lafan1_walk1_subject1_manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Imitation-G1-Strict-v0")
parser.add_argument("--motion_manifest", type=Path, required=True)
parser.add_argument("--motion_name", default="walk1_subject1")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--hold_steps", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
# This script does not go through Hydra, so `physics=newton_mjwarp` is not
# available as a CLI override; the preset is selected here instead. Newton is
# the default because it is the backend the study runs on, and joint-order
# resolution is backend-specific.
parser.add_argument(
    "--physics", default="newton_mjwarp", choices=("default", "newton_mjwarp")
)
parser.add_argument("--njmax", type=int, default=320)
parser.add_argument("--nconmax", type=int, default=40)
parser.add_argument(
    "--allow_resets",
    action="store_true",
    help="Keep terminations active so asynchronous renewal is exercised. Off by "
    "default: the core certificate isolates chunk consumption from reset "
    "transients.",
)
parser.add_argument(
    "--action_scale",
    type=float,
    default=0.1,
    help="Std of the random joint-position offsets. Non-zero on purpose: the "
    "robot must drift so the anchor re-expression is not the identity.",
)
# Float32 anchor-frame round trips accumulate a little error; 1e-4 (0.1 mm on
# positions) is far below anything that could matter and far above float noise.
parser.add_argument("--tol", type=float, default=1e-4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_imitation.tasks  # noqa: E402, F401
import isaaclab_tasks  # noqa: E402, F401
import torch  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    load_cfg_from_registry,
    resolve_presets,
)

from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (  # noqa: E402
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1_KEYPOINT5_BODY_NAMES,
    G1_OBS_ANCHOR_BODY_NAME,
)


# term name -> (body list, expected per-frame width, which interface needs it)
CASES: list[tuple[str, tuple[str, ...], int, str]] = [
    # Control: the already-qualified full-body interface. If this fails the
    # harness is wrong, not the new interfaces.
    ("expert_motion", (), 58, "full_body (control)"),
    ("expert_anchor_pos_b", (), 3, "shared root"),
    ("expert_anchor_ori_b", (), 6, "shared root"),
    ("expert_motion_qpos", (), 29, "root_qpos"),
    ("expert_keypoint_pos_b", tuple(G1_KEYPOINT5_BODY_NAMES), 15, "root_points5"),
]


def main() -> int:
    # parse_env_cfg() resolves every PresetCfg to its `default` (PhysX) before
    # returning, so the backend has to be chosen on the unresolved tree. This
    # mirrors what Hydra does for `physics=newton_mjwarp` on the launcher path.
    env_cfg = load_cfg_from_registry(
        args_cli.task.split(":")[-1], "env_cfg_entry_point"
    )
    env_cfg = resolve_presets(env_cfg, (str(args_cli.physics),))
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = True
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.physics == "newton_mjwarp":
        env_cfg.sim.physics.solver_cfg.njmax = int(args_cli.njmax)
        env_cfg.sim.physics.solver_cfg.nconmax = int(args_cli.nconmax)

    env_cfg.lafan1_manifest_path = str(args_cli.motion_manifest.expanduser().resolve())
    env_cfg.motion_name_filter = [str(args_cli.motion_name)]
    env_cfg.seed = int(args_cli.seed)

    hold = int(args_cli.hold_steps)
    # The streamed contract under test: 10-frame future-only packet, published
    # once per hold window, consumed one slot per control step.
    env_cfg.command_observation_source = "planner_oracle"
    env_cfg.command_hold_steps = hold
    env_cfg.latent_patch_past_steps = 0
    env_cfg.latent_patch_future_steps = hold - 1
    env_cfg.policy_command_mode = "full_body_chunk_current_slot"

    if not args_cli.allow_resets:
        # A reset mid-window is a separate phenomenon from chunk consumption: it
        # jumps the reference cursor while the packet still holds the previous
        # episode's frames. Measuring both at once yields an uninterpretable
        # number, so the core certificate runs reset-free and --allow_resets
        # exercises asynchronous renewal separately.
        for term_name in (
            "anchor_pos",
            "anchor_ori",
            "ee_body_pos",
            "foot_pos_xyz",
            "base_too_low",
        ):
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)
        env_cfg.episode_length_s = 1000.0

    resolve_manifest = getattr(env_cfg, "_resolve_manifest_config", None)
    if callable(resolve_manifest):
        resolve_manifest()
    sync_window = getattr(env_cfg, "_sync_expert_window_observation_params", None)
    if callable(sync_window):
        sync_window()

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    # Read the contract from the env, never from a constant. The packet is
    # published by the observation manager during env.step() using the env's own
    # anchor body and pinned joint order; comparing against any other anchor or
    # ordering measures that difference instead of the adapter. This surface is
    # pelvis-anchored (`_set_anchor_body("pelvis")`), not torso-anchored.
    anchor = str(base_env._expert_anchor_body_name)
    joint_cfg = SceneEntityCfg(
        "robot", joint_names=G1_29DOF_ISAACLAB_JOINT_NAMES, preserve_order=True
    )
    joint_cfg.resolve(base_env.scene)
    pinned_joint_ids = joint_cfg.joint_ids
    print(f"[contract] anchor_body={anchor} (env), joint order=pinned 29-DoF")
    if anchor == G1_OBS_ANCHOR_BODY_NAME:
        print(f"[contract] note: this surface anchors on {G1_OBS_ANCHOR_BODY_NAME}")

    # max abs error per (term, phase), and the scale of the signal being matched
    worst: dict[str, list[float]] = {name: [0.0] * hold for name, _, _, _ in CASES}
    scale: dict[str, float] = {name: 0.0 for name, _, _, _ in CASES}
    widths: dict[str, int] = {}
    phase_counts = [0] * hold
    resets_seen = 0

    try:
        env.reset()
        action_dim = int(env.action_space.shape[-1])
        generator = torch.Generator(device="cpu").manual_seed(int(args_cli.seed))

        for _ in range(int(args_cli.steps)):
            actions = (
                float(args_cli.action_scale)
                * torch.randn((int(args_cli.num_envs), action_dim), generator=generator)
            ).to(base_env.device)
            _, _, terminated, truncated, _ = env.step(actions)
            resets_seen += int((terminated | truncated).sum().item())

            phase = base_env._command_hold_phase()
            for phase_value in phase.tolist():
                phase_counts[int(phase_value)] += 1

            for term_name, bodies, expected_width, _label in CASES:
                joint_ids = (
                    pinned_joint_ids
                    if term_name in ("expert_motion", "expert_motion_qpos")
                    else slice(None)
                )
                streamed = base_env.current_full_body_tracker_command_term(
                    term_name,
                    joint_ids=joint_ids,
                    anchor_body_name=anchor,
                    reference_body_names=bodies,
                )
                direct = base_env.get_current_expert_window_term(
                    term_name,
                    past_steps=0,
                    future_steps=0,
                    joint_ids=joint_ids,
                    anchor_body_name=anchor,
                    reference_body_names=bodies,
                )
                widths.setdefault(term_name, int(streamed.shape[1]))
                if int(streamed.shape[1]) != expected_width:
                    raise AssertionError(
                        f"{term_name}: streamed slot width {int(streamed.shape[1])}, "
                        f"expected {expected_width}."
                    )
                if streamed.shape != direct.shape:
                    raise AssertionError(
                        f"{term_name}: streamed {tuple(streamed.shape)} vs "
                        f"direct {tuple(direct.shape)}."
                    )
                err = (streamed - direct).abs().amax(dim=1)
                scale[term_name] = max(
                    scale[term_name], float(direct.abs().amax().item())
                )
                for env_index, phase_value in enumerate(phase.tolist()):
                    slot = int(phase_value)
                    worst[term_name][slot] = max(
                        worst[term_name][slot], float(err[env_index].item())
                    )
    finally:
        env.close()

    tol = float(args_cli.tol)
    print("\n=== reduced-interface streaming equivalence ===")
    print(
        f"task={args_cli.task} motion={args_cli.motion_name} "
        f"envs={args_cli.num_envs} steps={args_cli.steps} hold={hold}"
    )
    print(
        f"resets during the run: {resets_seen}"
        f"{' (async renewal exercised)' if resets_seen else ' (reset-free)'}"
    )
    print(f"hold phases visited  : {phase_counts}")

    missing_phase = [i for i, count in enumerate(phase_counts) if count == 0]
    print(
        f"\n{'term':24}{'width':>7}{'interface':>22}"
        f"{'max |err|':>12}{'signal':>10}  verdict"
    )
    failures: list[str] = []
    for term_name, _bodies, _expected, label in CASES:
        worst_err = max(worst[term_name])
        ok = worst_err <= tol
        if not ok:
            failures.append(term_name)
        print(
            f"{term_name:24}{widths[term_name]:>7}{label:>22}"
            f"{worst_err:>12.2e}{scale[term_name]:>10.3f}  "
            f"{'ok' if ok else 'MISMATCH'}"
        )

    # Per-phase breakdown. The shape of the residual across the hold window is
    # what identifies the defect: flat-and-large means the packet content is
    # wrong; zero at phase 0 and growing means a phase/anchor drift; a single
    # bad phase means an off-by-one in the slot shift.
    print(f"\nmax |err| by hold phase (0 = renewal step):\n{'term':24}", end="")
    print("".join(f"{p:>10}" for p in range(hold)))
    for term_name, _bodies, _expected, _label in CASES:
        print(
            f"{term_name:24}"
            + "".join(f"{worst[term_name][p]:>10.2e}" for p in range(hold))
        )

    if missing_phase:
        print(
            f"\nRESULT: INCONCLUSIVE -- hold phases {missing_phase} were never "
            "observed, so the certificate is not phase-complete. Raise --steps."
        )
        return 2
    if failures:
        print(
            f"\nRESULT: FAIL -- {', '.join(failures)} do not survive the "
            f"publish/consume round trip (tolerance {tol:.1e})."
        )
        print(
            "        Do not train a controller on this command space: the "
            "packet the actor reads is not the command that was published."
        )
        return 1
    print(
        f"\nRESULT: PASS -- every term matches the unchunked reference at all "
        f"{hold} hold phases (tolerance {tol:.1e})."
    )
    print(
        "        The adapter is sound. Whether the space carries ENOUGH "
        "information is a separate question, answered by qualify_interface.sh"
    )
    print("        once a controller exists.")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
