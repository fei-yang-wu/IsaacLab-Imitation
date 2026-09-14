#!/usr/bin/env python3
"""Audit the Wuji finger joints of a stored Vega-Wuji Reference NPZ.

A kinematic replay shows exactly what the IK optimized, so it cannot fail
visually.  This joint-space audit is the gate; videos are confirmation only.
Per side and joint it reports min/max against limits, the fraction of frames
near a limit, interphalangeal hyperextension, the maximum frame-to-frame
jump, and DIP-vs-PIP coupling, then applies the pre-registered gates
(hyperextension 0 %, jump <= 35 deg per frame). Fingertip and wrist residuals
come from the converter's final-FK acceptance record when it is embedded in the
Reference metadata. A required gate fails closed when that evidence is absent.

Run from the repository root:

    pixi run python scripts/audit/audit_wuji_finger_joints.py \
        --reference <reference.npz> [--anatomical] [--out <report.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from iltools.core import load_dexterous_reference_npz
from iltools.retarget.wuji_finger import (
    apply_anatomical_finger_limits,
    audit_finger_joints,
    evaluate_finger_gates,
)
from imitation_experiments.paths import REPO_ROOT


DEFAULT_MODEL = (
    REPO_ROOT
    / "source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji"
    / "vega_u_wuji_v2_beta1_with_mount.xml"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_float32_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    shape = ",".join(str(size) for size in array.shape).encode("ascii")
    return hashlib.sha256(b"float32:" + shape + b"\0" + array.tobytes()).hexdigest()


def _embedded_evidence_binding(
    metadata: Any,
    *,
    qpos: np.ndarray,
    joint_names: tuple[str, ...],
    model_path: Path,
) -> tuple[bool, str]:
    """Verify that embedded final-FK summaries belong to this qpos and model."""

    if not isinstance(metadata, Mapping):
        return False, "Reference metadata is missing."
    finger_mapping = metadata.get("finger_mapping")
    final_acceptance = (
        finger_mapping.get("final_acceptance")
        if isinstance(finger_mapping, Mapping)
        else None
    )
    binding = (
        final_acceptance.get("evidence_binding")
        if isinstance(final_acceptance, Mapping)
        else None
    )
    if not isinstance(binding, Mapping):
        return False, "Final-FK evidence binding is missing."
    expected = {
        "schema": "isaaclab_imitation_wuji_final_fk_evidence/v1",
        "qpos_float32_sha256": _canonical_float32_array_sha256(qpos),
        "model_sha256": _sha256_file(model_path),
        "joint_names_sha256": hashlib.sha256(
            json.dumps(list(joint_names), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    mismatches = [
        name for name, value in expected.items() if binding.get(name) != value
    ]
    if mismatches:
        return False, f"Final-FK evidence binding mismatch: {mismatches}."
    target_hash = binding.get("fingertip_targets_float32_sha256")
    if not isinstance(target_hash, str) or len(target_hash) != 64:
        return False, "Final-FK fingertip-target hash is missing or malformed."
    return True, "verified"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--anatomical",
        action="store_true",
        help=(
            "Audit against the anatomical finger envelope (interphalangeal "
            "lower bounds 0 deg) instead of the model's mechanical limits. "
            "A trajectory produced without the anatomical clamp usually "
            "fails this stricter audit through hyperextension."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Exit non-zero when any required gate fails or is not measured.",
    )
    return parser


def _embedded_gate_measurements(
    metadata: Any, side: str
) -> tuple[float | None, float | None]:
    """Read final-FK tip/wrist measurements from Reference metadata."""

    if not isinstance(metadata, Mapping):
        return None, None
    finger_mapping = metadata.get("finger_mapping")
    if not isinstance(finger_mapping, Mapping):
        return None, None
    final_acceptance = finger_mapping.get("final_acceptance")
    if not isinstance(final_acceptance, Mapping):
        return None, None
    sides = final_acceptance.get("sides")
    if not isinstance(sides, Mapping) or side not in sides:
        return None, None
    side_record = sides[side]
    if not isinstance(side_record, Mapping):
        raise ValueError(f"Embedded final acceptance for {side} is malformed.")
    fingertip_record = side_record.get("fingertip_residuals")
    if not isinstance(fingertip_record, Mapping):
        raise ValueError(f"Embedded fingertip residuals for {side} are malformed.")
    tip_p95_mm = float(fingertip_record["tip_p95_mm"])
    wrist_mean_mm = float(side_record["wrist_mean_mm"])
    if (
        not np.isfinite((tip_p95_mm, wrist_mean_mm)).all()
        or tip_p95_mm < 0.0
        or wrist_mean_mm < 0.0
    ):
        raise ValueError(f"Embedded final measurements for {side} are invalid.")
    return tip_p95_mm, wrist_mean_mm


def _print_side_table(side: str, audit: dict[str, Any]) -> None:
    print(
        f"\n=== {side}: {audit['num_frames']} frames; frames with any IP "
        f"hyperextension > 5 deg: {audit['pct_frames_any_hyperext_gt5deg']}% ==="
    )
    header = (
        f"{'joint':26s} {'min':>7s} {'max':>7s} {'limits':>16s} "
        f"{'%lo':>5s} {'%hi':>5s} {'jump':>6s} {'%hyper':>7s}"
    )
    print(header)
    for row in audit["joints"]:
        print(
            f"{row['joint']:26s} {row['min_deg']:7.1f} {row['max_deg']:7.1f} "
            f"{str(row['limit_deg']):>16s} {str(row['pct_near_lower']):>5s} "
            f"{str(row['pct_near_upper']):>5s} {row['max_jump_deg']:6.1f} "
            f"{str(row.get('pct_hyperext_gt5deg', '')):>7s}"
        )
    if audit["worst_frames_by_hyperextension"]:
        print("worst frames:", audit["worst_frames_by_hyperextension"][:5])
    print(
        "coupling (official target slope 0.7):",
        {
            name: (value["slope_dip_vs_pip"], f"{value['pct_within_15deg_of_0.7pip']}%")
            for name, value in audit["coupling_dip_vs_pip"].items()
        },
    )


def main() -> int:
    args = build_parser().parse_args()
    reference = load_dexterous_reference_npz(args.reference)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    limit_rows: list[dict[str, Any]] = []
    if args.anatomical:
        limit_rows = apply_anatomical_finger_limits(model)

    qpos = np.asarray(reference.qpos, dtype=np.float64)
    joint_names = tuple(reference.joint_names)
    report: dict[str, Any] = {
        "reference": str(args.reference),
        "model": str(args.model),
        "limits": "anatomical" if args.anatomical else "model_mechanical",
        "tightened_joints": limit_rows,
        "sides": {},
        "gates": {},
    }
    evidence_bound, evidence_reason = _embedded_evidence_binding(
        reference.metadata,
        qpos=qpos,
        joint_names=joint_names,
        model_path=args.model,
    )
    report["embedded_final_fk_evidence"] = {
        "bound": evidence_bound,
        "reason": evidence_reason,
    }
    if not evidence_bound:
        print(f"[audit] embedded final-FK evidence not trusted: {evidence_reason}")
    required_gate_failures: list[str] = []
    for side, prefix in (("left", "l_"), ("right", "r_")):
        columns = [
            index for index, name in enumerate(joint_names) if name.startswith(prefix)
        ]
        names = [joint_names[index] for index in columns]
        if not names:
            raise ValueError(f"Reference has no {side} Wuji finger joints.")
        limits = {
            name: (
                float(model.jnt_range[model.joint(name).id, 0]),
                float(model.jnt_range[model.joint(name).id, 1]),
            )
            for name in names
        }
        audit = audit_finger_joints(qpos[:, columns], names, limits)
        tip_p95_mm, wrist_mean_mm = (
            _embedded_gate_measurements(reference.metadata, side)
            if evidence_bound
            else (None, None)
        )
        gates = evaluate_finger_gates(
            audit=audit,
            tip_p95_mm=tip_p95_mm,
            wrist_mean_mm=wrist_mean_mm,
        )
        report["sides"][side] = audit
        report["gates"][side] = gates
        _print_side_table(side, audit)
        measured = {
            name: value for name, value in gates["values"].items() if value is not None
        }
        print(
            f"[gates {side}] measured: "
            + " ".join(f"{name}={value:.1f}" for name, value in measured.items())
            + f" | failures: {gates['failures'] or 'none'}"
            + f" | not measured offline: {gates['not_measured']}"
        )
        required_gate_failures.extend(
            f"{side}:{name}" for name in (*gates["failures"], *gates["not_measured"])
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"[audit] wrote {args.out}")
    if required_gate_failures:
        print(f"[audit] required gate failures: {required_gate_failures}")
        if args.require_gates:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
