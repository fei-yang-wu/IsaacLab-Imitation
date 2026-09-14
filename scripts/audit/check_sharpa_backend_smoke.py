#!/usr/bin/env python3
"""Check that the Sharpa task can actually run on one physics backend.

A training job that starts, logs, and exits cleanly is not evidence that the
task works. The Skynet calibration job did all three and still produced
``r_step=nan`` with one-step episodes, because every environment terminated on
the first control step. This gate catches that in about a minute on a
workstation, so it runs before a cluster submission, not after.

It resets the task, steps it with zero actions, and reports four quantities
that a working scene must satisfy. Zero actions are the right probe: the
residual policy contributes nothing, so the hands and the object are carried by
the reference tracking controllers alone and should stay on the Reference.

  terminations     no environment may terminate while tracking the Reference
  contact force    the reset pose must not inject a penetration impulse
  rewards          finite, never NaN
  episode length   more than one control step

Example, both backends:

    export SHARPA_REFERENCE_MANIFEST=$PWD/data/dexmanip/sharpa/<set>/manifest.json
    pixi run -e isaaclab python scripts/audit/check_sharpa_backend_smoke.py \\
        --num-envs 4 --steps 8 physics=newton_mjwarp
    pixi run -e isaaclab python scripts/rlopt/train_physx.py --help  # PhysX needs Kit first

Exit codes: 0 all checks passed, 1 a check failed, 2 the task could not start.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCRIPTS_DIR, "rlopt"))

# The contact force a reset pose may legitimately carry. The object weighs a
# few newtons; anything far above that is penetration recovery, not a grasp.
DEFAULT_MAX_CONTACT_FORCE = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Sharpa-V2D-Source-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-contact-force",
        type=float,
        default=DEFAULT_MAX_CONTACT_FORCE,
        help="Largest contact force, in newtons, a healthy reset may show.",
    )
    parser.add_argument(
        "--allow-terminations",
        action="store_true",
        help="Report terminations without failing (for a deliberately hard set).",
    )
    parser.add_argument("--video", action="store_true", default=False)
    return parser


def _launch_kit_if_needed(argv: list[str]):
    """Start Kit before the config is resolved, for every backend but Newton.

    PhysX needs the Kit runtime, and resolving the task config first imports
    ``pxr``, after which ``SimulationApp`` crashes during startup. Newton needs
    no Kit at all, so it is left alone.
    """

    from runtime_bootstrap import requested_backend

    if requested_backend(argv) == "newton":
        return None
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--video", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args, _ = parser.parse_known_args(argv)
    launcher_args.headless = True
    return AppLauncher(launcher_args)


def main() -> int:
    app_launcher = _launch_kit_if_needed(sys.argv[1:])

    from isaaclab_tasks.utils import setup_preset_cli

    args_cli, hydra_args = setup_preset_cli(build_parser(), sys.argv[1:])
    sys.argv = [sys.argv[0]] + hydra_args
    args_cli.headless = True

    import gymnasium as gym
    import torch

    import isaaclab_imitation.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    env_cfg, _ = resolve_task_config(args_cli.task, "rlopt_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    failures: list[str] = []
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        action = torch.zeros(env.action_space.shape, device=env.device)

        peak_contact = 0.0
        terminated_steps = 0
        nan_steps = 0
        lengths: list[float] = []

        for step in range(args_cli.steps):
            _, reward, terminated, _, _ = env.step(action)
            if torch.isnan(reward).any():
                nan_steps += 1
            fired = int(terminated.sum().item())
            if fired:
                terminated_steps += 1
            for side in ("right", "left"):
                try:
                    sensor = env.scene[f"{side}_hand_object_contacts"]
                except KeyError:
                    continue
                for name in ("force_matrix_w", "net_forces_w"):
                    buffer = getattr(sensor.data, name, None)
                    if buffer is not None:
                        peak_contact = max(
                            peak_contact, float(buffer.norm(dim=-1).max())
                        )
            lengths.append(float(env.episode_length_buf.float().mean()))
            print(
                f"step {step}: terminated={fired}/{env.num_envs} "
                f"reward={float(reward.mean()):+.4f} "
                f"peak_contact={peak_contact:.2f} N"
            )
        env.close()

    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    print("\n--- verdict ---")
    print(f"steps run                 : {args_cli.steps}")
    print(f"steps with a termination  : {terminated_steps}")
    print(f"steps with a NaN reward   : {nan_steps}")
    print(f"peak contact force        : {peak_contact:.2f} N")
    print(f"mean episode length       : {mean_length:.2f} control steps")

    if nan_steps:
        failures.append(f"{nan_steps} of {args_cli.steps} steps had a NaN reward")
    if peak_contact > args_cli.max_contact_force:
        failures.append(
            f"peak contact force {peak_contact:.2f} N is above the "
            f"{args_cli.max_contact_force:.2f} N limit; the reset pose is "
            f"penetrating the object"
        )
    if terminated_steps and not args_cli.allow_terminations:
        failures.append(
            f"{terminated_steps} of {args_cli.steps} steps terminated an "
            f"environment while it was tracking the Reference"
        )
    if mean_length <= 1.0:
        failures.append(
            f"mean episode length {mean_length:.2f} means episodes end on the "
            f"first control step; no learning signal is possible"
        )

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nPASSED: the task tracks the Reference on this backend.")
    sys.stdout.flush()

    # Closing Kit can end the process, so say the verdict first.
    if app_launcher is not None:
        app_launcher.app.close()
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # pragma: no cover - startup failures
        print(f"could not start the task: {error}", file=sys.stderr)
        raise SystemExit(2) from error
