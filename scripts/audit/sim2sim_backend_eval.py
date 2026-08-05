#!/usr/bin/env python3
# ruff: noqa: E402
"""Behavioural cross-backend check for one G1 low-level checkpoint.

`dump_backend_index_contract.py` proves the *index contract* matches across
backends -- that every order-sensitive term resolves to the same joints. It
cannot prove the policy behaves the same, because it never steps the
environment: an index audit is blind to anything written inside an event
function, to actuator differences, and to solver dynamics.

This runs the identical deterministic checkpoint under two backends and
compares the environment's own MPJPE, which the reference command term
maintains at `metrics["mpjpe_mm"]`.

Reading the result:

* MPJPE close across backends -> the joint contract is genuinely backend
  agnostic and the remaining gap is solver realism. A checkpoint may then be
  rendered or evaluated on either backend.
* MPJPE wildly apart -> something order-sensitive still leaks somewhere the
  index audit cannot see, and cross-backend playback is invalid.

Early terminations are disabled and the rollout is fixed length, so both
backends are scored over exactly the same frames. Without that, a backend whose
policy falls sooner reports MPJPE only over the easy opening frames and looks
*better*, which is the survivor bias that has already produced one wrong
conclusion in this campaign.

Everything else that could differ between the two processes is pinned for the
same reason: the same trajectory per environment, the same start frame, no
domain randomization, no interval pushes, and the distribution mode of the
actor rather than a sample. Any of those left free makes the two backends score
*different rollouts*, and the difference is then indistinguishable from a
solver gap. ``--keep_randomization`` and ``--stochastic_actions`` restore the
training-time distributions when that is what is being measured; the recorded
JSON says which was used.

Usage:

    pixi run -e isaaclab python scripts/audit/sim2sim_backend_eval.py \
        --checkpoint <policy.pt> --steps 300 --num_envs 32 \
        --output logs/sim2sim/result.json \
        <hydra overrides...>

The backend is NOT a flag: pass `physics=...` per invocation and compare the
two JSON files with --compare. Making it a flag would invite running one
process and believing it covered both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_HELPER_DIR = SCRIPT_DIR.parent / "rlopt"
if str(RUNTIME_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_HELPER_DIR))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compare", nargs=2, default=None, metavar=("LEFT", "RIGHT"))
parser.add_argument("--task", default="Isaac-Imitation-G1-v2")
parser.add_argument("--algo", "--algorithm", dest="algorithm", default="IPMD")
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--agent_entry_point",
    type=str,
    default=None,
    help=(
        "Agent config entry point to build the network from, e.g. "
        "rlopt_ipmd_tuned_cfg_entry_point. Must match what the checkpoint was "
        "TRAINED with -- the tuned recipe adds input normalization and changes "
        "widths, so a tuned checkpoint fails on a state-dict mismatch under the "
        "default contract. Defaults to the task's algorithm entry point."
    ),
)
parser.add_argument(
    "--keep_terminations",
    action="store_true",
    default=False,
    help=(
        "Leave the tracking-error terminations active. OFF by default because a "
        "fixed-length rollout is what makes two backends comparable. Turn it ON "
        "to reproduce the standard oracle protocol, whose MPJPE is lower and NOT "
        "comparable to the disabled-termination number: an episode that resets on "
        "drift only ever scores its own easy frames."
    ),
)
parser.add_argument(
    "--keep_randomization",
    action="store_true",
    default=False,
    help=(
        "Leave startup domain randomization, reset perturbations, and interval "
        "pushes active. OFF by default: each draws from the RNG, and the two "
        "backends do not consume it at the same points, so the rollouts diverge "
        "for reasons that are not the solver."
    ),
)
parser.add_argument(
    "--start_frame",
    type=int,
    default=0,
    help="Reference frame every environment resets onto (pinned on both sides).",
)
parser.add_argument(
    "--stochastic_actions",
    action="store_true",
    default=False,
    help=(
        "Sample from the actor instead of using its mode. OFF by default: the "
        "actor is a ProbabilisticActor whose default interaction type is "
        "`random`, so the comparison would otherwise score two different action "
        "sequences."
    ),
)


# Float32 accumulated through two different solvers will not agree bit for bit
# even on a quantity both compute from the same reference. 1e-4 rad / m is far
# below any ordering error (which moves whole joints, ~0.1-1.0) and far above
# single-precision noise.
_RESET_PARITY_TOL = 1.0e-4


def _reset_parity(left: dict, right: dict) -> tuple[list[str], list[str]]:
    """Per-term max |difference| of the post-reset observation.

    Returns ``(differing, missing)``. A term listed in ``differing`` is
    order-sensitive or backend-dependent *by definition*, before any physics
    has run -- which is the one thing a solver difference cannot explain.
    """
    left_obs = left.get("reset_observation") or {}
    right_obs = right.get("reset_observation") or {}
    differing: list[str] = []
    missing = sorted(set(left_obs) ^ set(right_obs))
    for term in sorted(set(left_obs) & set(right_obs)):
        lv, rv = left_obs[term], right_obs[term]
        if len(lv) != len(rv):
            differing.append(f"{term}: width {len(lv)} vs {len(rv)}")
            continue
        worst = max((abs(a - b) for a, b in zip(lv, rv)), default=0.0)
        if worst > _RESET_PARITY_TOL:
            permutation = sorted(round(v, 5) for v in lv) == sorted(
                round(v, 5) for v in rv
            )
            verdict = "SAME VALUES, DIFFERENT ORDER" if permutation else "values differ"
            differing.append(f"{term}: max|d|={worst:.6f}  ({verdict})")
    return differing, missing


def compare(left_path: Path, right_path: Path) -> int:
    """Diff two runs and say plainly whether cross-backend playback is valid."""
    left = json.loads(Path(left_path).read_text())
    right = json.loads(Path(right_path).read_text())
    print(f"left  : {left_path}  ({left['backend']})")
    print(f"right : {right_path}  ({right['backend']})")
    if left["checkpoint_sha256"] != right["checkpoint_sha256"]:
        print("\n[FAIL] Different checkpoints; this comparison is meaningless.")
        return 2
    print(f"checkpoint sha256 : {left['checkpoint_sha256'][:16]} (identical)")
    print(f"steps / envs      : {left['steps']} / {left['num_envs']}")

    # Two runs of different protocols are two different experiments. Say so
    # before printing a ratio that would be read as a solver measurement.
    left_protocol, right_protocol = left.get("protocol"), right.get("protocol")
    if left_protocol is None or right_protocol is None:
        print(
            "\n[FAIL] One side predates protocol recording, so it may have used "
            "random reference starts and training randomization. Re-run both."
        )
        return 2
    if left_protocol != right_protocol:
        print("\n[FAIL] The two runs used different protocols:")
        print(f"  left : {json.dumps(left_protocol, sort_keys=True)}")
        print(f"  right: {json.dumps(right_protocol, sort_keys=True)}")
        return 2
    print(f"protocol          : {json.dumps(left_protocol, sort_keys=True)}\n")

    # Stage 1: does the environment present the same command and proprioception
    # to the policy before any physics has run?
    differing, missing = _reset_parity(left, right)
    if missing:
        print(f"[FAIL] observation terms present on only one side: {missing}")
    if differing:
        print("[FAIL] post-reset observation differs BEFORE the first step:")
        for line in differing:
            print(f"    {line}")
        print(
            "\n       No solver difference can explain this: it is the same "
            "reference\n       frame rendered through the index contract. Fix "
            "the listed term(s)\n       before reading the rollout numbers below."
        )
    else:
        print(
            "[PASS] post-reset observation identical across backends "
            f"(all terms within {_RESET_PARITY_TOL:g})."
        )
        print("       Any rollout gap below is solver behaviour, not indexing.\n")

    # Stage 2: the rollout itself.
    print(f"{'metric':22s} {'left':>10s} {'right':>10s} {'ratio':>8s}")
    worst = 1.0
    for key in ("mpjpe_mm_first", "mpjpe_mm_mean", "mpjpe_mm_final", "survived_frac"):
        if key not in left["metrics"] or key not in right["metrics"]:
            continue
        lv, rv = left["metrics"][key], right["metrics"][key]
        ratio = (max(lv, rv) / min(lv, rv)) if min(lv, rv) > 1e-9 else float("inf")
        worst = max(worst, ratio)
        print(f"{key:22s} {lv:10.3f} {rv:10.3f} {ratio:8.2f}x")
    # 1.25x is a judgement call, not a derived threshold: solver differences of
    # a few percent are expected, while a surviving order leak permutes joints
    # and produces a gross divergence. Anything between is worth a human look,
    # so it is reported rather than silently passed.
    print()
    if differing or missing:
        return 2
    if worst < 1.25:
        print(f"[PASS] Backends agree within {worst:.2f}x. Joint contract is portable.")
        return 0
    print(f"[WARN] Backends differ by {worst:.2f}x -- inspect before trusting")
    print("       cross-backend playback or evaluation of this checkpoint.")
    return 1


args_cli, hydra_args = parser.parse_known_args()
if args_cli.compare:
    raise SystemExit(compare(Path(args_cli.compare[0]), Path(args_cli.compare[1])))
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required unless --compare is used")

sys.argv = [sys.argv[0]] + hydra_args

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import hashlib

import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Importing the task package is what registers the Gym ids; gym.spec below
# raises NameNotFound without it.
import isaaclab_imitation.tasks  # noqa: F401
from imitation_experiments.audit.backend_determinism import (
    apply_randomization_profile,
    describe_reference_selection,
    pin_reference_start,
)


def _observation_snapshot(td) -> dict[str, list[float]]:
    """Environment 0's value for every observation term, keyed by group.term.

    Full values, not a summary: the point is to localize a disagreement to one
    term, and a mean would hide a permutation entirely.
    """
    snapshot: dict[str, list[float]] = {}
    # Mixed str / tuple keys are not mutually orderable, so filter first and
    # sort the rendered names.
    keys = [
        key
        for key in td.keys(include_nested=True, leaves_only=True)
        if isinstance(key, tuple) and key[0] in ("policy", "critic")
    ]
    for key in sorted(keys, key=lambda k: tuple(str(part) for part in k)):
        value = td.get(key)
        if not torch.is_tensor(value) or value.ndim < 1:
            continue
        snapshot[".".join(str(part) for part in key)] = [
            round(float(v), 6) for v in value[0].reshape(-1).tolist()
        ]
    return snapshot


def _robot_state_snapshot(base) -> dict[str, object]:
    """Environment 0's articulation state, joint quantities keyed BY NAME.

    Joint vectors are dicts rather than lists because the live articulation
    order is exactly what differs between backends; comparing them positionally
    would report a difference for every backend pair regardless of correctness.
    """
    robot = base.scene["robot"]
    names = list(robot.joint_names)
    joint_pos = robot.data.joint_pos.torch[0].tolist()
    joint_vel = robot.data.joint_vel.torch[0].tolist()
    return {
        "joint_pos": {n: round(float(v), 6) for n, v in zip(names, joint_pos)},
        "joint_vel": {n: round(float(v), 6) for n, v in zip(names, joint_vel)},
        "root_pos_w": [round(float(v), 6) for v in robot.data.root_pos_w.torch[0]],
        "root_quat_w": [round(float(v), 6) for v in robot.data.root_quat_w.torch[0]],
    }


def _agent_entry_point(task_name: str, algorithm: str) -> str:
    """Same rule the training and evaluation entry points use.

    Duplicated deliberately rather than imported: the canonical copy lives
    inside `evaluate_checkpoint.py`, a 1700-line module whose import pulls in
    the whole evaluation stack. Three lines is cheaper than that coupling.
    """
    entry = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    spec = gym.spec(task_name.split(":")[-1])
    if spec.kwargs.get(entry) is None:
        raise ValueError(f"Task {task_name!r} exposes no {entry!r}.")
    return entry


agent_entry_point = args_cli.agent_entry_point or _agent_entry_point(
    args_cli.task, args_cli.algorithm
)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(env_cfg, agent_cfg):
    from isaaclab_imitation.envs.rlopt import (
        IsaacLabTerminalObsReader,
        IsaacLabWrapper,
    )
    from rlopt.agent import IPMD
    from torchrl.envs import TransformedEnv

    algorithm_classes = {"IPMD": IPMD}

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Fixed-length rollout: both backends must be scored over the same frames.
    if not args_cli.keep_terminations:
        for term in ("anchor_pos", "anchor_ori", "ee_body_pos", "foot_pos_xyz"):
            if getattr(env_cfg.terminations, term, None) is not None:
                setattr(env_cfg.terminations, term, None)

    # Same motion per environment, same start frame, no randomization. Raises
    # rather than silently doing nothing if the config exposes no reset-sampling
    # surface it recognizes.
    selection_surface = pin_reference_start(env_cfg, start_frame=args_cli.start_frame)
    randomization_kept = apply_randomization_profile(
        env_cfg, "all" if args_cli.keep_randomization else "none"
    )
    env_cfg.observations.policy.enable_corruption = False

    agent_cfg.env.num_envs = args_cli.num_envs
    agent_cfg.logger.backend = ""

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    backend = type(env_cfg.sim.physics).__name__

    wrapped = IsaacLabWrapper(env)
    wrapped = wrapped.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=wrapped.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(base_env=wrapped)
    agent = algorithm_classes[args_cli.algorithm](env=env, config=agent_cfg)
    agent.load_model(str(args_cli.checkpoint))
    policy = getattr(agent, "deployment_policy", None) or agent.collector_policy
    policy.eval()

    ref_term = base.command_manager.get_term("reference")
    interaction = (
        ExplorationType.RANDOM if args_cli.stochastic_actions else ExplorationType.MODE
    )

    mpjpe_trace: list[float] = []
    td = env.reset()
    # The post-reset observation, before any physics has run. Both backends have
    # just been teleported onto the same reference frame, so every term here is
    # a pure function of the reference and the pinned index contract: a term
    # that differs is an ordering or definition leak, not a solver difference.
    # This is the measurement that separates the two, and it cannot be made
    # after stepping, when the states have legitimately diverged.
    reset_observation = _observation_snapshot(td)
    reset_robot_state = _robot_state_snapshot(base)

    with torch.no_grad(), set_exploration_type(interaction):
        for _ in range(args_cli.steps):
            td = policy(td)
            td = env.step(td)
            td = td["next"] if "next" in td.keys() else td
            mpjpe_trace.append(float(ref_term.metrics["mpjpe_mm"].mean().item()))

    alive = 1.0 - float(base.termination_manager.dones.float().mean().item())
    result = {
        "backend": backend,
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            Path(args_cli.checkpoint).read_bytes()
        ).hexdigest(),
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "seed": args_cli.seed,
        "terminations_active": bool(args_cli.keep_terminations),
        # The protocol, recorded so `--compare` can refuse two runs that were
        # not actually run the same way.
        "protocol": {
            "start_frame": int(args_cli.start_frame),
            "reference_selection_surface": selection_surface,
            "reference_selection": describe_reference_selection(env_cfg),
            "randomization_kept": randomization_kept,
            "action_sampling": "random" if args_cli.stochastic_actions else "mode",
        },
        "robot_joint_names": list(base.scene["robot"].joint_names),
        "reset_observation": reset_observation,
        "reset_robot_state": reset_robot_state,
        "metrics": {
            "mpjpe_mm_mean": (
                sum(mpjpe_trace) / len(mpjpe_trace) if mpjpe_trace else 0.0
            ),
            "mpjpe_mm_final": mpjpe_trace[-1] if mpjpe_trace else 0.0,
            "mpjpe_mm_first": mpjpe_trace[0] if mpjpe_trace else 0.0,
            "survived_frac": alive,
        },
        "mpjpe_mm_trace": mpjpe_trace,
    }
    print(json.dumps(result, indent=2))
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[INFO] wrote {args_cli.output}")
    env.close()


if __name__ == "__main__":
    # The traceback is printed HERE, not left to propagate. Isaac's
    # `simulation_app.close()` terminates the process from the finally block
    # before the interpreter gets to print an unhandled exception, so the
    # standard `try/finally` idiom turns every failure into a silent exit 0 --
    # which is how `evaluate_checkpoint.py` appears to succeed while doing
    # nothing at all.
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
