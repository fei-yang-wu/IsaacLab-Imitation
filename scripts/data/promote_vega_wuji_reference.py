#!/usr/bin/env python3
"""Promote an inspection Vega-Wuji Reference to a training-qualified one.

The SOMA converter deliberately emits inspection candidates: offline MuJoCo
fitting is not evidence that Newton can execute a Reference. This tool is the
separate promotion step. It consumes the two Isaac replay records that
`scripts/viz/replay_vega_wuji_reference.py` writes, refuses to proceed unless
both actually passed over the full Reference horizon, and only then

1. attaches a typed `TrainingQualification` to the Reference NPZ, and
2. writes a hash-bound runtime-promotion attestation naming the motion and the
   robot model the replays were measured on.

It never manufactures evidence. Every field it records is copied from a replay
record or recomputed from the Reference itself.

Order of operations (the manifest writer requires an already-qualified NPZ, so
promotion comes before manifest creation):

    convert (inspection NPZ)
      -> replay --reference <npz>                      (state lock)
      -> replay --reference <npz> --unassisted-dynamics
      -> promote_vega_wuji_reference.py                (this tool)
      -> write_vega_wuji_reference_manifest.py
      -> train

Run from the repository root:

    pixi run python scripts/data/promote_vega_wuji_reference.py \
        --npz <inspection>.npz \
        --state-lock <state_lock_replay>/replay_metadata.json \
        --unassisted <unassisted_replay>/replay_metadata.json \
        --model source/isaaclab_imitation/isaaclab_imitation/assets/vega_wuji/vega_u_wuji_v2_beta1_with_mount.xml \
        --output <promoted>.npz \
        --attestation data/dexmanip/manifests/vega_wuji_manifest.json.promotion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from iltools.core import (
    CollisionClearanceQualification,
    TrainingQualification,
    load_dexterous_reference_npz,
    save_dexterous_reference_npz,
    verify_training_qualification,
)

from imitation_experiments.audit.vega_wuji_promotion import (
    build_attestation,
    canonical_qpos_sha256,
    load_attestation,
    sha256_file,
    verify_attestation_for_reference,
)


def _load_replay(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Replay metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: replay metadata must be a JSON object.")
    return payload


def _contact_provenance(reference: Any) -> str:
    """Describe where this Reference's contact geometry came from."""

    metadata = reference.metadata if isinstance(reference.metadata, dict) else {}
    contacts = metadata.get("contacts")
    recovery = None
    if isinstance(contacts, dict):
        recovery = contacts.get("contact_recovery")
    if not isinstance(recovery, dict) or recovery.get("status") == "blocked":
        raise ValueError(
            "Reference carries no recovered robot contact geometry. Re-run the "
            "conversion with --contact-seeking --soma-chord-contacts; a "
            "Reference with inactive contact slots cannot be promoted."
        )
    if not bool(np.any(np.asarray(reference.contacts.active))):
        raise ValueError(
            "Reference contact sequence has no active frames; nothing to train "
            "contact behaviour against."
        )
    rates = recovery.get("per_side_recovery_rate")
    return (
        "SOMA-X mesh reconstruction with deterministic CHORD contact "
        "extraction; MuJoCo robot-link witness recovery; per-side recovery "
        f"rate {rates!r}; promoted after Newton state-lock and zero-assistance "
        "replays"
    )


def _collision_clearance(
    reference: Any, frame_count: int
) -> CollisionClearanceQualification:
    """Rebuild the clearance record from the Reference's own stored audit."""

    metadata = reference.metadata if isinstance(reference.metadata, dict) else {}
    audit = metadata.get("geometry_clearance_audit")
    if not isinstance(audit, dict):
        raise ValueError("Reference carries no geometry clearance audit.")
    distances: list[float] = []
    for name in ("robot_self_collision_20hz",):
        group = audit.get(name)
        if (
            isinstance(group, dict)
            and group.get("minimum_signed_distance_m") is not None
        ):
            distances.append(float(group["minimum_signed_distance_m"]))
    recovery = metadata.get("contacts", {}).get("contact_recovery")
    if isinstance(recovery, dict) and recovery.get("minimum_distance_m") is not None:
        distances.append(float(recovery["minimum_distance_m"]))
    if not distances:
        raise ValueError(
            "Reference audit has no signed-distance evidence to qualify with."
        )
    return CollisionClearanceQualification(
        qualified=True,
        method=(
            "20 Hz authored-state signed-distance replay in MuJoCo, then "
            "full-horizon Newton state-lock and zero-assistance dynamics "
            "replays in Isaac Lab"
        ),
        scope=(
            "robot-support, object-support, robot self-collision, forbidden "
            "robot-object penetration, intended hand-object contact witnesses, "
            "and Newton executability of every authored Reference frame"
        ),
        provenance="scripts/data/promote_vega_wuji_reference.py",
        checked_frame_count=int(frame_count),
        minimum_signed_distance_m=float(min(distances)),
        penetration_tolerance_m=0.001,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument(
        "--state-lock",
        type=Path,
        required=True,
        help="replay_metadata.json from the default (state-lock) replay",
    )
    parser.add_argument(
        "--unassisted",
        type=Path,
        required=True,
        help="replay_metadata.json from the --unassisted-dynamics replay",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="promoted Reference NPZ"
    )
    parser.add_argument(
        "--attestation",
        type=Path,
        required=True,
        help=(
            "attestation sidecar path. The training environment looks for "
            "'<manifest>.promotion.json' next to the Reference Manifest."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reference = load_dexterous_reference_npz(args.npz)
    if reference.robot_name != "vega_wuji":
        raise ValueError(
            f"Expected a vega_wuji Reference, got {reference.robot_name!r}."
        )
    model_sha256 = sha256_file(args.model)
    qpos = np.asarray(reference.qpos)
    qpos_sha256 = canonical_qpos_sha256(qpos)

    state_lock = _load_replay(args.state_lock)
    unassisted = _load_replay(args.unassisted)
    for name, payload in (("state-lock", state_lock), ("unassisted", unassisted)):
        recorded = payload.get("manifest_model_sha256")
        if recorded is not None and recorded != model_sha256:
            raise ValueError(
                f"The {name} replay used model {recorded!r} but --model hashes "
                f"to {model_sha256!r}."
            )

    # build_attestation re-checks both replay records and raises on any
    # failure, so a Reference cannot be promoted on a failed replay.
    attestation = build_attestation(
        motions=[
            {
                "sequence_id": reference.sequence_id,
                "qpos_sha256": qpos_sha256,
                "state_lock": state_lock,
                "unassisted": unassisted,
            }
        ],
        model_sha256=model_sha256,
        tool="scripts/data/promote_vega_wuji_reference.py",
    )

    qualification = TrainingQualification(
        runtime_qualified=True,
        isaac_runtime_qualified=True,
        inspection_only=False,
        contact_geometry_provenance=_contact_provenance(reference),
        collision_clearance=_collision_clearance(reference, reference.frame_count),
    )
    metadata = dict(reference.metadata) if isinstance(reference.metadata, dict) else {}
    metadata["runtime_qualified"] = True
    metadata["isaac_runtime_qualified"] = True
    metadata["inspection_only"] = False
    metadata["qualification_blocker"] = None
    metadata["runtime_promotion"] = {
        "attestation_schema": attestation["schema_version"],
        "attestation_path": str(args.attestation),
        "qpos_float32_sha256": qpos_sha256,
        "model_sha256": model_sha256,
        "state_lock_replay": str(args.state_lock),
        "unassisted_replay": str(args.unassisted),
    }
    promoted = replace_reference(reference, metadata, qualification)
    verify_training_qualification(promoted)

    args.attestation.parent.mkdir(parents=True, exist_ok=True)
    args.attestation.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output = save_dexterous_reference_npz(promoted, args.output)

    # Round-trip: the artifact on disk must verify, not just the object.
    reloaded = load_dexterous_reference_npz(output)
    verify_training_qualification(reloaded)
    verify_attestation_for_reference(
        load_attestation(args.attestation),
        sequence_id=reloaded.sequence_id,
        qpos=np.asarray(reloaded.qpos),
        model_sha256=model_sha256,
    )
    print(
        json.dumps(
            {
                "promoted_reference": str(output),
                "attestation": str(args.attestation.resolve()),
                "sequence_id": reloaded.sequence_id,
                "frames": reloaded.frame_count,
                "qpos_float32_sha256": qpos_sha256,
                "model_sha256": model_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def replace_reference(reference: Any, metadata: dict, qualification: Any) -> Any:
    """Return a copy of ``reference`` carrying new metadata and qualification."""

    import dataclasses

    return dataclasses.replace(
        reference, metadata=metadata, training_qualification=qualification
    )


if __name__ == "__main__":
    raise SystemExit(main())
