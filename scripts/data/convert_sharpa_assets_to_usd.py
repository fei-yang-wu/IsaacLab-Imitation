#!/usr/bin/env python3
"""Pre-convert the Sharpa URDF assets to USD so training can run without Kit.

Isaac Lab converts a ``UrdfFileCfg`` to USD at spawn time, and that conversion
needs Kit. On a cluster that forces the Kit-first entrypoint, which starts a
full Omniverse app on a headless compute node. The G1 task avoids this by
shipping an already-converted USD; this tool does the same for Sharpa.

It reuses the task's own spawn configuration, so the generated USD carries the
identical import settings (base fixity, joint drives, contact reporting,
rigid and articulation properties). Run it once, locally, where Kit is
available:

    OMNI_KIT_ACCEPT_EULA=YES pixi run -e isaaclab python \\
        scripts/data/convert_sharpa_assets_to_usd.py --headless

Afterwards ``ISAACLAB_IMITATION_SHARPA_USD_DIR`` (or the default directory
below) makes the task spawn from USD, and training can run with
``--assert-kitless``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT / "source/isaaclab_imitation/isaaclab_imitation/assets/sharpa_wave/usd"
)


def parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--object-urdf",
        type=Path,
        action="append",
        default=None,
        help="Also convert one rigid object URDF; may be repeated.",
    )
    parser.add_argument(
        "--contact-stiffness",
        type=float,
        nargs=2,
        metavar=("KE", "KD"),
        default=(0.0, 0.0),
        help=(
            "Optional Newton per-shape contact stiffness and damping, written "
            "to every collider of every asset this run touches, hands "
            "included. Off by default (0 0), which keeps Newton's own "
            "(2500, 100), MuJoCo solref (0.02, 1.0), and matches the released "
            "recipe. 400 40 is solref (0.05, 1.0), a softer contact that cuts "
            "the reset impulse on a penetrating pose from 54 N to 9 N; it was "
            "only needed while Newton misframed external wrenches."
        ),
    )
    parser.add_argument(
        "--skip-hands",
        action="store_true",
        help="Convert only the object URDFs passed with --object-urdf.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even when the USD is newer than its URDF.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _collision_prim_specs(layer):
    """Yield every prim spec in ``layer`` that applies ``PhysicsCollisionAPI``."""

    from pxr import Sdf

    found = []

    def visit(path: Sdf.Path) -> None:
        if not path.IsPrimPath():
            return
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return
        schemas = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
        if schemas is None:
            return
        items = list(schemas.GetAddedOrExplicitItems())
        if "PhysicsCollisionAPI" in items:
            found.append(spec)

    layer.Traverse(Sdf.Path("/"), visit)
    return found


def _usd_layers(usd_path: Path):
    """Return every layer file of one converted asset, payloads included."""

    from pxr import Sdf

    for path in sorted(usd_path.parent.rglob("*")):
        if path.suffix.lower() not in (".usd", ".usda", ".usdc"):
            continue
        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is not None:
            yield layer


def set_mesh_approximation(usd_path: Path, approximation: str) -> int:
    """Force every collision mesh in a converted USD to one approximation.

    The URDF importer's ``collision_type`` does not survive this repository's
    asset-transformer and multi-physics conversion steps: they rewrite
    ``physics:approximation`` back to ``convexHull``. A convex hull fills a
    container's cavity, so a hand reaching into the ARCTIC box reads as
    centimetres of penetration and the solver fights to expel it.

    The collision prims live in payload layers that the root stage does not
    compose, so the token is set on each layer directly rather than through a
    composed stage. That also makes the result greppable in the written files.

    Returns the number of attributes changed.
    """

    changed = 0
    for layer in _usd_layers(usd_path):
        touched = False
        for spec in _collision_prim_specs(layer):
            attribute = layer.GetAttributeAtPath(
                spec.path.AppendProperty("physics:approximation")
            )
            if attribute is None or attribute.default == approximation:
                continue
            attribute.default = approximation
            changed += 1
            touched = True
        if touched:
            layer.Save()
    return changed


def set_contact_stiffness(usd_path: Path, ke: float, kd: float) -> int:
    """Author Newton's per-shape contact stiffness and damping on every collider.

    Newton's MuJoCo solver turns a shape's ``ke``/``kd`` into MuJoCo's
    ``solref``: ``timeconst = 2 / kd`` and ``dampratio = kd / 2 * sqrt(1 / ke)``.
    Its default (2500, 100) is MuJoCo's default ``(0.02, 1.0)``, which on the
    retargeted reset pose produces a 54 N impulse that throws the hands and
    the object off the Reference. PhysX caps the same event with
    ``max_depenetration_velocity``; Newton has no such cap, so the contact time
    constant is the knob that plays its role. ``ke=400, kd=40`` is
    ``(0.05, 1.0)`` and measured at 9 N, PhysX's magnitude.

    Isaac Lab's ``NewtonShapeCfg`` exposes only ``margin`` and ``gap``, so the
    value is authored on the collision prims, where Newton's USD importer reads
    ``newton:contact_ke`` and ``newton:contact_kd`` by name.

    Returns the number of collision prims written.
    """

    from pxr import Sdf

    written = 0
    for layer in _usd_layers(usd_path):
        touched = False
        for spec in _collision_prim_specs(layer):
            for name, value in (("newton:contact_ke", ke), ("newton:contact_kd", kd)):
                attribute = layer.GetAttributeAtPath(spec.path.AppendProperty(name))
                if attribute is None:
                    attribute = Sdf.AttributeSpec(spec, name, Sdf.ValueTypeNames.Float)
                if attribute.default != float(value):
                    attribute.default = float(value)
                    touched = True
            written += 1
        if touched:
            layer.Save()
    return written


def main() -> int:
    args = parse_args()
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter

    from isaaclab_imitation.assets.sharpa import (
        left_sharpa_wave_urdf_path,
        right_sharpa_wave_urdf_path,
        sharpa_hand_urdf_spawn,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path]] = []

    if args.object_urdf and args.skip_hands:
        written_sides: tuple[tuple[str, str], ...] = ()
    else:
        written_sides = (
            ("right", right_sharpa_wave_urdf_path),
            ("left", left_sharpa_wave_urdf_path),
        )
    for side, urdf_path in written_sides:
        # Always import from the URDF: the task prefers an already-converted
        # USD, and asking it for a spawn would hand back that USD instead.
        spawn = sharpa_hand_urdf_spawn(urdf_path).replace(
            usd_dir=str(output_dir),
            usd_file_name=f"{side}_sharpa_wave.usd",
            force_usd_conversion=True,
        )
        converter = UrdfConverter(spawn)
        written.append((f"{side} hand", Path(converter.usd_path)))
    if args.skip_hands:
        # The hands were not reconverted, but Newton's contact stiffness lives
        # on their colliders too, so the existing USDs are patched in place.
        from isaaclab_imitation.assets.sharpa import sharpa_hand_usd_path

        for side in ("right", "left"):
            existing = sharpa_hand_usd_path(side)
            if existing is not None:
                written.append((f"{side} hand (existing)", existing))

    for urdf in args.object_urdf or ():
        urdf = urdf.expanduser().resolve()
        if not urdf.is_file():
            raise SystemExit(f"Object URDF does not exist: {urdf}")
        # Mirror the object spawn settings the task uses for a rigid proxy.
        spawn = sim_utils.UrdfFileCfg(
            asset_path=str(urdf),
            fix_base=False,
            joint_drive=None,
            merge_fixed_joints=True,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
            activate_contact_sensors=True,
            # Match the released spawn. A convex hull fills a container's
            # cavity, so a hand reaching into the box would read as deep
            # penetration and the solver would fight to expel it.
            collision_type="Convex Decomposition",
            usd_dir=str(urdf.parent),
            usd_file_name=f"{urdf.stem}.usd",
            force_usd_conversion=True,
        )
        converter = UrdfConverter(spawn)
        usd_path = Path(converter.usd_path)
        count = set_mesh_approximation(usd_path, "convexDecomposition")
        print(f"set convexDecomposition on {count} collision meshes")
        written.append((f"object {urdf.name}", usd_path))

    ke, kd = args.contact_stiffness
    for label, path in written:
        if ke > 0.0 and kd > 0.0:
            count = set_contact_stiffness(path, ke, kd)
            print(
                f"set newton:contact_ke={ke:g} kd={kd:g} on {count} colliders of {label}"
            )
        print(f"wrote {label}: {path}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
