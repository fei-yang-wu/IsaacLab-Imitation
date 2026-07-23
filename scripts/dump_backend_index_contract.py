#!/usr/bin/env python3
"""Dump every joint/body index mapping an imitation task resolves at runtime.

Isaac Lab physics backends do not enumerate an articulation in the same order:
PhysX exposes a breadth-first order while Newton uses a depth-first per-limb
order. Any observation, action, reward, or termination term that resolves its
indices from the *live* articulation instead of a pinned canonical name list
therefore changes meaning when the backend changes, which silently invalidates
a checkpoint trained on the other backend.

This probe builds the task, resolves every manager term, and writes a JSON
"index contract". Run it once per backend and diff the two files: any term whose
ordering is backend dependent shows up as a differing ``joint_ids`` /
``body_ids`` entry, or as an identity list paired with differing
``robot_joint_names``.

Examples (run from the repository root):

.. code-block:: bash

    pixi run -e isaaclab python scripts/dump_backend_index_contract.py \
        --task Isaac-Imitation-G1-Latent-v0 \
        --output logs/index_contract/newton.json \
        physics=newton_mjwarp

    pixi run -e isaaclab python scripts/dump_backend_index_contract.py \
        --task Isaac-Imitation-G1-Latent-v0 \
        --output logs/index_contract/physx.json \
        physics=physx

    python scripts/dump_backend_index_contract.py --compare \
        logs/index_contract/newton.json logs/index_contract/physx.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_HELPER_DIR = SCRIPT_DIR / "rlopt"
if str(RUNTIME_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_HELPER_DIR))

from runtime_bootstrap import (  # noqa: E402
    assert_kit_not_loaded,
    config_contains_type_name,
    detect_gpu_names,
    install_kit_import_guard,
    requested_backend,
    validate_gpu_policy,
)


logger = logging.getLogger(__name__)
DEFAULT_TASK = "Isaac-Imitation-G1-Latent-v0"
DEFAULT_OUTPUT_ROOT = Path("logs/index_contract")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    from isaaclab_tasks.utils import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--assert-kitless",
        action="store_true",
        help="Refuse to run unless Kit is absent and the backend is Newton.",
    )
    add_launcher_args(parser)
    args, _ = parser.parse_known_args(argv)
    return args


def _normalize_ids(value: object) -> object:
    """Render a resolved index selection in a diff-friendly, JSON-safe form."""
    if isinstance(value, slice):
        if value == slice(None):
            return "slice(None)"
        return f"slice({value.start}, {value.stop}, {value.step})"
    if value is None:
        return None
    try:
        return [int(v) for v in value]
    except TypeError:
        return repr(value)


def _scene_entity_report(entity_cfg: object) -> dict:
    return {
        "asset": getattr(entity_cfg, "name", None),
        "preserve_order": bool(getattr(entity_cfg, "preserve_order", False)),
        "joint_names": getattr(entity_cfg, "joint_names", None),
        "joint_ids": _normalize_ids(getattr(entity_cfg, "joint_ids", None)),
        "body_names": getattr(entity_cfg, "body_names", None),
        "body_ids": _normalize_ids(getattr(entity_cfg, "body_ids", None)),
    }


def _collect_scene_entity_cfgs(container: object) -> dict[str, dict]:
    """Find every resolved SceneEntityCfg reachable from a manager cfg object."""
    found: dict[str, dict] = {}
    if container is None:
        return found
    for term_name in dir(container):
        if term_name.startswith("_"):
            continue
        term_cfg = getattr(container, term_name, None)
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict):
            continue
        for param_name, param_value in params.items():
            if not hasattr(param_value, "joint_ids"):
                continue
            found[f"{term_name}.{param_name}"] = _scene_entity_report(param_value)
    return found


def _build_report(env_cfg: object, args_cli: argparse.Namespace) -> dict:
    import gymnasium as gym

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.observations.policy.enable_corruption = False

    # This is an ordering audit, so remove every source of run-to-run variation.
    # Otherwise two backends start from different reference frames and different
    # randomized joint defaults, and the sampled planner frame below would differ
    # for reasons that have nothing to do with joint order.
    env_cfg.random_reset_step_min = 0
    env_cfg.random_reset_step_max = 0
    env_cfg.random_reset_full_trajectory = False
    events_cfg = getattr(env_cfg, "events", None)
    for event_name in dir(events_cfg or ()):
        if event_name.startswith("_"):
            continue
        params = getattr(getattr(events_cfg, event_name), "params", None)
        if not isinstance(params, dict):
            continue
        # The joint-default randomization perturbs the action offset, which is
        # one of the order-dependent quantities being audited. It documents
        # None as "no randomization".
        if "pos_distribution_params" in params:
            params["pos_distribution_params"] = None
        # The reference reset draws pose, velocity, and joint noise. Two
        # backends consume the RNG differently, so leaving these on makes the
        # two runs start from genuinely different states and any sampled state
        # below would differ for reasons unrelated to indexing. Empty ranges
        # mean "no offset"; None is not accepted here.
        for range_key in ("pose_range", "velocity_range"):
            if range_key in params:
                params[range_key] = {}
        if "joint_position_range" in params:
            params["joint_position_range"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        unwrapped = env.unwrapped
        env.reset(seed=int(args_cli.seed))
        robot = unwrapped.scene["robot"]

        report: dict[str, object] = {
            "task": args_cli.task,
            "physics_cfg": type(env_cfg.sim.physics).__name__,
            "robot_usd_path": getattr(
                getattr(env_cfg.scene.robot, "spawn", None), "usd_path", None
            ),
            "robot_joint_names": list(robot.joint_names),
            "robot_body_names": list(robot.body_names),
        }

        # The canonical name lists the env believes it is pinned to.
        trajectory_manager = getattr(unwrapped, "trajectory_manager", None)
        report["canonical_names"] = {
            "cfg_target_joint_names": list(
                getattr(env_cfg, "target_joint_names", []) or []
            ),
            "cfg_reference_joint_names": list(
                getattr(env_cfg, "reference_joint_names", []) or []
            ),
            "cfg_reference_body_names": list(
                getattr(env_cfg, "reference_body_names", []) or []
            ),
            "tm_target_joint_names": list(
                getattr(trajectory_manager, "target_joint_names", []) or []
            ),
            "env_reference_body_names": list(
                getattr(unwrapped, "reference_body_names", []) or []
            ),
        }

        # Action terms: names, resolved ids, and the realized offset/scale
        # vectors, since a mismatched offset is the classic silent failure.
        action_terms: dict[str, dict] = {}
        for term_name in unwrapped.action_manager.active_terms:
            term = unwrapped.action_manager.get_term(term_name)
            entry: dict[str, object] = {
                "joint_names": list(getattr(term, "_joint_names", []) or []),
                "joint_ids": _normalize_ids(getattr(term, "_joint_ids", None)),
                "preserve_order": bool(
                    getattr(getattr(term, "cfg", None), "preserve_order", False)
                ),
            }
            for attr_name, key in (
                ("_offset", "offset_env0"),
                ("_scale", "scale_env0"),
            ):
                attr = getattr(term, attr_name, None)
                if hasattr(attr, "shape") and attr.ndim == 2:
                    entry[key] = [round(float(v), 8) for v in attr[0].tolist()]
                elif hasattr(attr, "item"):
                    entry[key] = round(float(attr.item()), 8)
                else:
                    entry[key] = attr
            action_terms[term_name] = entry
        report["action_terms"] = action_terms

        # Observation terms, grouped, with their resolved SceneEntityCfgs.
        obs_entities: dict[str, dict] = {}
        obs_cfg = unwrapped.observation_manager.cfg
        for (
            group_name,
            term_names,
        ) in unwrapped.observation_manager.active_terms.items():
            group_cfg = getattr(obs_cfg, group_name, None)
            for term_name in term_names:
                term_cfg = getattr(group_cfg, term_name, None)
                params = getattr(term_cfg, "params", None)
                if not isinstance(params, dict):
                    continue
                for param_name, param_value in params.items():
                    if not hasattr(param_value, "joint_ids"):
                        continue
                    key = f"{group_name}.{term_name}.{param_name}"
                    obs_entities[key] = _scene_entity_report(param_value)
        report["observation_scene_entity_cfgs"] = obs_entities

        report["observation_term_dims"] = {
            group_name: {
                term_name: list(dim)
                for term_name, dim in zip(
                    unwrapped.observation_manager.active_terms[group_name],
                    unwrapped.observation_manager.group_obs_term_dim[group_name],
                    strict=True,
                )
            }
            for group_name in unwrapped.observation_manager.active_terms
        }

        for label in ("rewards", "terminations", "events", "curriculum"):
            report[f"{label}_scene_entity_cfgs"] = _collect_scene_entity_cfgs(
                getattr(unwrapped.cfg, label, None)
            )

        contact_sensor = unwrapped.scene.sensors.get("contact_forces")
        if contact_sensor is not None:
            report["contact_sensor_body_names"] = list(contact_sensor.body_names)

        # Sample the causal planner frame straight after reset, when the robot
        # has been placed on the reference and the two backends should agree.
        # This is the recorded-data path, which is ordered independently of the
        # observation manager and so is not covered by the index diffs above.
        for label, accessor in (
            ("causal", "current_causal_planner_observation"),
            ("offline_demo", "current_offline_demo_planner_observation"),
        ):
            builder = getattr(unwrapped, accessor, None)
            if builder is None:
                continue
            try:
                frame = builder(env_ids=[0])
            except Exception as exc:  # noqa: BLE001 - recorded as a diagnostic
                report[f"planner_frame_{label}"] = f"<unavailable: {exc}>"
                continue
            values = None
            for key in (("planner", "state_history"), "state_history"):
                try:
                    values = frame.get(key)
                except KeyError:
                    values = None
                if values is not None:
                    break
            if values is None:
                keys = sorted(str(k) for k in frame.keys(include_nested=True))
                report[f"planner_frame_{label}"] = f"<no state_history; keys={keys}>"
                continue
            flat = values.reshape(-1).tolist()
            report[f"planner_frame_{label}"] = {
                "shape": list(values.shape),
                # Rounded so solver-level noise does not mask an ordering
                # difference, which is a whole-slot permutation.
                "values_round3": [round(float(v), 3) for v in flat],
            }

        return report
    finally:
        env.close()


def run(argv: list[str], *, require_running_kit: bool = False) -> int:
    strict_kitless = "--assert-kitless" in argv
    if strict_kitless:
        install_kit_import_guard()

    args_cli = _parse_args(argv)
    backend = requested_backend(argv)

    import isaaclab_imitation.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab.utils import has_kit
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    # ``register_task`` reads ``sys.argv`` directly and forwards every token
    # without an ``=`` to Hydra, which rejects this script's own flags. Expose
    # only the ``key=value`` overrides for the duration of the call.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]] + [token for token in argv if "=" in token]
    try:
        env_cfg, _ = resolve_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
    finally:
        sys.argv = saved_argv
    if strict_kitless:
        if not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError("--assert-kitless requires physics=newton_mjwarp.")
        assert_kit_not_loaded()
    if require_running_kit and not has_kit():
        raise RuntimeError("The PhysX index contract requires a running Kit app.")

    with launch_simulation(env_cfg, args_cli):
        report = _build_report(env_cfg, args_cli)

    if strict_kitless:
        assert_kit_not_loaded()

    output = args_cli.output or (DEFAULT_OUTPUT_ROOT / f"{backend}.json")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"[INFO] Index contract written to {output}", flush=True)
    return 0


def compare(left_path: Path, right_path: Path) -> int:
    """Diff two contracts and flag terms whose ordering is backend dependent."""
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))

    left_joints = left["robot_joint_names"]
    right_joints = right["robot_joint_names"]
    same_enumeration = left_joints == right_joints
    print(f"left  : {left_path}  ({left['physics_cfg']})")
    print(f"right : {right_path}  ({right['physics_cfg']})")
    print(f"articulation joint order identical: {same_enumeration}")
    if not same_enumeration:
        moved = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(left_joints, right_joints, strict=True))
            if a != b
        ]
        print(f"  {len(moved)}/{len(left_joints)} joint slots differ, e.g.:")
        for index, a, b in moved[:5]:
            print(f"    slot {index:2d}: left={a} right={b}")

    # A term is only *wrong* if the values it produces land in a different
    # semantic order on the two backends. A term with ``preserve_order=True``
    # whose ids differ is the remap working as intended, not a defect.
    leaks: list[str] = []
    remapped: list[str] = []
    benign: list[str] = []
    state_differences: list[str] = []

    def _is_wildcard(names: object) -> bool:
        """True when a selection cannot encode an ordering."""
        if names is None:
            return True
        if isinstance(names, str):
            names = [names]
        # A single name selects one element, so its order cannot be wrong.
        return len(names) <= 1 or all(n in ("*", ".*") for n in names)

    for section in (
        "observation_scene_entity_cfgs",
        "rewards_scene_entity_cfgs",
        "terminations_scene_entity_cfgs",
        "events_scene_entity_cfgs",
        "curriculum_scene_entity_cfgs",
    ):
        for key, left_entry in (left.get(section) or {}).items():
            label = f"{section}.{key}"
            right_entry = (right.get(section) or {}).get(key)
            if right_entry is None:
                leaks.append(f"{label}: present only on the left")
                continue
            if left_entry["preserve_order"]:
                if left_entry["joint_ids"] != right_entry["joint_ids"] or (
                    left_entry["body_ids"] != right_entry["body_ids"]
                ):
                    remapped.append(label)
                continue
            # preserve_order=False: ids are ascending in the live enumeration,
            # so the selected values stay in *live* order on both backends.
            named = not (
                _is_wildcard(left_entry["joint_names"])
                and _is_wildcard(left_entry["body_names"])
            )
            if same_enumeration:
                continue
            if named:
                leaks.append(
                    f"{label}: preserve_order=False over a named selection "
                    f"{left_entry['joint_names'] or left_entry['body_names']!r:.60} "
                    "-> values stay in live order, which differs per backend"
                )
            else:
                benign.append(label)

    for key, left_entry in (left.get("action_terms") or {}).items():
        label = f"action_terms.{key}"
        right_entry = (right.get("action_terms") or {}).get(key)
        if right_entry is None:
            leaks.append(f"{label}: present only on the left")
            continue
        if left_entry["joint_names"] != right_entry["joint_names"]:
            leaks.append(f"{label}: joint_names differ across backends")
        for field in ("offset_env0", "scale_env0"):
            lv, rv = left_entry.get(field), right_entry.get(field)
            if not isinstance(lv, list) or not isinstance(rv, list):
                continue
            # Slot i means the same joint on both sides once joint_names match,
            # so these vectors must agree up to per-env randomization noise.
            if len(lv) != len(rv):
                leaks.append(f"{label}.{field}: length differs")
            elif max(abs(a - b) for a, b in zip(lv, rv, strict=True)) > 0.05:
                leaks.append(
                    f"{label}.{field}: slot values disagree by more than "
                    "randomization noise -> written in the wrong joint order"
                )
        if left_entry["joint_ids"] != right_entry["joint_ids"]:
            remapped.append(label)

    if left.get("contact_sensor_body_names") != right.get("contact_sensor_body_names"):
        remapped.append("contact_sensor_body_names")

    for label in ("causal", "offline_demo"):
        key = f"planner_frame_{label}"
        lv, rv = left.get(key), right.get(key)
        if not isinstance(lv, dict) or not isinstance(rv, dict):
            continue
        if lv["shape"] != rv["shape"]:
            leaks.append(f"{key}: shape differs across backends")
            continue
        left_values, right_values = lv["values_round3"], rv["values_round3"]
        diffs = sum(1 for a, b in zip(left_values, right_values) if a != b)
        if not diffs:
            continue
        # An ordering leak permutes values within a frame, so the two sides hold
        # the same multiset. Genuinely different values mean the two backends
        # are in different physical states, which is a solver/reset-fidelity
        # question and not an index-contract failure.
        joints = 29
        is_permutation = all(
            sorted(left_values[base : base + joints])
            == sorted(right_values[base : base + joints])
            for base in range(0, len(left_values) - joints + 1, joints)
        )
        if is_permutation:
            leaks.append(
                f"{key}: {diffs}/{len(left_values)} values differ and each block "
                "is a permutation of the other -> joint-order leak"
            )
        else:
            state_differences.append(
                f"{key}: {diffs}/{len(left_values)} values differ but the blocks "
                "are not permutations -> backend state/solver difference, not "
                "an ordering leak"
            )

    if remapped:
        print(
            f"\n{len(remapped)} term(s) correctly remapped (pinned order, ids differ):"
        )
        for label in remapped:
            print(f"  . {label}")
    if benign:
        print(f"\n{len(benign)} wildcard selection(s), order-insensitive:")
        for label in benign:
            print(f"  . {label}")

    if state_differences:
        print(
            f"\n{len(state_differences)} backend state difference(s) "
            "(NOT ordering; solver/reset fidelity):"
        )
        for difference in state_differences:
            print(f"  ~ {difference}")

    if not leaks:
        print("\nNo backend-dependent index leak found.")
        return 0
    print(f"\n{len(leaks)} BACKEND-DEPENDENT LEAK(S):")
    for leak in leaks:
        print(f"  ! {leak}")
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--compare" in argv:
        rest = [a for a in argv if a != "--compare"]
        if len(rest) != 2:
            raise ValueError("--compare takes exactly two contract JSON paths.")
        return compare(Path(rest[0]), Path(rest[1]))

    backend = requested_backend(argv)
    if backend == "newton":
        if "--assert-kitless" not in argv:
            argv.append("--assert-kitless")
        return run(argv)

    if backend != "physx":
        raise ValueError(f"Unsupported index-contract backend: {backend!r}")
    validate_gpu_policy("physx", detect_gpu_names())

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args(argv)
    app_launcher = AppLauncher(launcher_args)
    try:
        return run(argv, require_running_kit=True)
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Index contract dump failed.")
        raise
