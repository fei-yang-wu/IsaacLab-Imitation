#!/usr/bin/env python3
# ruff: noqa: E402
"""Check the spawned G1 articulation's link masses against the URDF.

The URDF is the authority. It is the only G1 description in this repo that is
complete (it carries the sensor/hand/head links and the fixed joints that
attach them), both vendored MJCFs agree with it to 1e-4 on total mass, and it
is what the USD was supposed to be converted from.

Verified 2026-08-03: the shipped ``g1_29dof_rev_1_0.usd`` **passes** on both
backends -- every link within 1e-3 kg of the URDF, total 33.3411 kg. The asset
is also byte-identical (sha256, all four layers) to Unitree's own publication at
``unitreerobotics/unitree_model``, ``G1/29dof/usd/g1_29dof_rev_1_0``. There is
no asset defect.

What this audit is really for is the trap that produced a false positive on the
way to that conclusion. ``G1SonicEventCfg.randomize_rigid_body_mass`` scales
``.*wrist_yaw.*|torso_link`` by a factor drawn from ``(0.8, 2.5)``, at startup.
Run this check with that event live and it reports torso and both wrist_yaw
links "wrong" by up to +4.4 kg -- a plausible-looking asset defect on exactly
three links, which is really one dice roll. Two backends draw different factors
because they consume the RNG at different points, so it also looks like a
backend disagreement. This script therefore forces
``apply_randomization_profile(env_cfg, "none")`` rather than offering it as a
flag: there is no version of this measurement that wants randomized mass.

A fixed-joint child is merged into its parent by the URDF importer, so the
expected mass of a link is its own plus every fixed descendant's. That merge is
computed here rather than assumed, which is what makes the comparison exact.

Usage (from the repository root):

.. code-block:: bash

    # What we ship today.
    pixi run -e isaaclab python scripts/audit/audit_g1_link_mass_contract.py \\
        --spawn usd physics=physx <data overrides...>

    # The same check against a spawn converted from the URDF at runtime.
    pixi run -e isaaclab python scripts/audit/audit_g1_link_mass_contract.py \\
        --spawn urdf physics=physx <data overrides...>

Exits non-zero when any link disagrees with the URDF beyond --tolerance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Imitation-G1-v2")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument(
    "--spawn",
    choices=("usd", "urdf"),
    default="usd",
    help="usd: the prebuilt asset the repo ships. urdf: convert at spawn time.",
)
parser.add_argument(
    "--tolerance",
    type=float,
    default=1.0e-3,
    help="Absolute kg tolerance per link.",
)
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
from isaaclab_imitation.assets.robots.unitree_joint_order import (
    UNITREE_G1_29DOF_URDF_FILE,
)


def urdf_expected_masses(urdf_path: Path) -> dict[str, float]:
    """Per-link mass after the importer merges every fixed-joint child.

    Fixed chains can nest (``torso_link <- head_link`` alongside
    ``torso_link <- d435_link``), so children are folded into their nearest
    non-fixed ancestor rather than only one level up.
    """
    root = ET.parse(urdf_path).getroot()
    mass = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        node = inertial.find("mass") if inertial is not None else None
        mass[link.attrib["name"]] = (
            float(node.attrib["value"]) if node is not None else 0.0
        )

    fixed_parent = {}
    movable_children = set()
    for joint in root.findall("joint"):
        child = joint.find("child").attrib["link"]
        parent = joint.find("parent").attrib["link"]
        if joint.attrib.get("type") == "fixed":
            fixed_parent[child] = parent
        else:
            movable_children.add(child)

    def anchor(link: str) -> str:
        seen = set()
        while link in fixed_parent:
            if link in seen:
                raise RuntimeError(f"Cycle in fixed joints at {link!r}.")
            seen.add(link)
            link = fixed_parent[link]
        return link

    merged: dict[str, float] = {}
    for link, link_mass in mass.items():
        merged[anchor(link)] = merged.get(anchor(link), 0.0) + link_mass
    # Only links that survive as bodies (root plus every movable joint's child).
    surviving = {anchor(link) for link in mass} - set(fixed_parent)
    return {name: value for name, value in merged.items() if name in surviving}


@hydra_task_config(args_cli.task, "rlopt_ipmd_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    # MANDATORY, not optional: `randomize_rigid_body_mass` scales exactly
    # `.*wrist_yaw.*|torso_link` by a factor drawn from (0.8, 2.5). Leaving it on
    # makes this audit read a randomized robot and report a phantom asset defect
    # on precisely those three links -- which is what it did before this line
    # existed.
    apply_randomization_profile(env_cfg, "none")

    if args_cli.spawn == "urdf":
        # `unitree_g1_29dof_usd_articulation_cfg` is what swaps the URDF spawn
        # for the prebuilt USD; rebuilding the cfg without it converts the URDF
        # at spawn time instead, which is the whole point of the comparison.
        from isaaclab_imitation.assets.robots.unitree import (
            UNITREE_G1_29DOF_SONIC_CFG,
        )

        robot = env_cfg.scene.robot
        env_cfg.scene.robot = UNITREE_G1_29DOF_SONIC_CFG.replace(
            prim_path=robot.prim_path,
            init_state=robot.init_state,
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    robot = base.scene["robot"]

    expected = urdf_expected_masses(UNITREE_G1_29DOF_URDF_FILE)
    body_names = list(robot.body_names)
    actual = np.asarray(robot.data.default_mass.detach().cpu().numpy()[0]).ravel()

    rows, failures = [], []
    for index, name in enumerate(body_names):
        want = expected.get(name)
        got = float(actual[index])
        delta = None if want is None else got - want
        rows.append({"body": name, "urdf": want, "spawned": got, "delta": delta})
        if want is None:
            failures.append(f"{name}: present in the spawn but not in the URDF")
        elif abs(delta) > args_cli.tolerance:
            failures.append(
                f"{name}: URDF {want:.4f} kg, spawned {got:.4f} kg ({delta:+.4f})"
            )

    missing = sorted(set(expected) - set(body_names))
    total_urdf = sum(expected.values())
    total_spawn = float(actual.sum())

    print(f"\nspawn source : {args_cli.spawn}")
    print(f"{'body':28s} {'URDF kg':>10s} {'spawned kg':>11s} {'delta':>10s}")
    for row in rows:
        want = "-" if row["urdf"] is None else f"{row['urdf']:.4f}"
        delta = "-" if row["delta"] is None else f"{row['delta']:+.4f}"
        mark = (
            ""
            if row["delta"] is not None and abs(row["delta"]) <= args_cli.tolerance
            else "  <=="
        )
        print(
            f"{row['body']:28s} {want:>10s} {row['spawned']:11.4f} {delta:>10s}{mark}"
        )
    print(
        f"{'TOTAL':28s} {total_urdf:10.4f} {total_spawn:11.4f} {total_spawn - total_urdf:+10.4f}"
    )
    if missing:
        print(f"\nURDF links with no spawned body: {missing}")

    report = {
        "task": args_cli.task,
        "spawn": args_cli.spawn,
        "tolerance_kg": args_cli.tolerance,
        "urdf": str(UNITREE_G1_29DOF_URDF_FILE),
        "total_mass_urdf": total_urdf,
        "total_mass_spawned": total_spawn,
        "rows": rows,
        "failures": failures,
    }
    if args_cli.output:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] wrote {args_cli.output}")

    env.close()
    if failures:
        print(f"\n[FAIL] {len(failures)} link(s) disagree with the URDF:")
        for line in failures:
            print(f"    {line}")
        print(
            f"\n       Total {total_spawn:.4f} kg against the URDF's {total_urdf:.4f} kg "
            f"({total_spawn - total_urdf:+.4f}).\n"
            "       Every dynamics measurement taken on this asset is a measurement\n"
            "       of a different robot."
        )
        raise SystemExit(1)
    print(f"\n[PASS] every link matches the URDF within {args_cli.tolerance} kg.")


if __name__ == "__main__":
    import traceback

    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
