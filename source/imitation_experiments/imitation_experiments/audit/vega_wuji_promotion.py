"""Hash-bound Vega-Wuji runtime-promotion attestation.

A retargeted Reference is fitted offline in MuJoCo. That is not evidence that
Newton can execute it. Two Isaac replays supply the missing evidence, and
`scripts/viz/replay_vega_wuji_reference.py` already produces both:

- **state lock** — teleport the articulation and object onto every stored
  Reference frame and measure how far the simulator drifts. It answers "is
  this Reference physically writable into Newton at all".
- **unassisted dynamics** — replay the same Reference with zero policy
  residual and the virtual object controller forced to exactly zero. It
  answers "does the scene hold together without the invisible wrench that
  normally carries the object".

This module turns those two replay records into one typed attestation and
verifies it again at training load. The binding is deliberately on Reference
*content* rather than on file bytes: promotion rewrites the NPZ to attach a
``TrainingQualification``, which changes the file hash but not the motion. The
canonical float32 qpos digest survives that rewrite, so it is the identity
that ties a replay record to the Reference the environment actually loaded.

Free-form Reference metadata is not accepted as a substitute; the attestation
is a separate artifact whose records must name the model and the motion they
were measured on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ATTESTATION_SCHEMA_VERSION = "isaaclab_imitation_vega_wuji_runtime_promotion/v1"
"""Schema of the attestation sidecar. Bump when a required field changes."""

STATE_LOCK_MODE = "reference_state_lock"
UNASSISTED_MODE = "unassisted_zero_residual_dynamics"

#: Sidecar filename for a manifest at ``<path>``.
ATTESTATION_SUFFIX = ".promotion.json"


def attestation_path_for_manifest(manifest_path: str | Path) -> Path:
    """Return the attestation sidecar path for one Reference Manifest."""

    path = Path(manifest_path)
    return path.with_name(path.name + ATTESTATION_SUFFIX)


def canonical_qpos_sha256(qpos: np.ndarray) -> str:
    """Digest Reference joint trajectories independently of NPZ packaging.

    Promotion re-saves the NPZ to attach a typed qualification, so a file
    digest cannot bind a replay record to a Reference. The motion itself can.
    """

    array = np.ascontiguousarray(np.asarray(qpos), dtype="<f4")
    shape = ",".join(str(size) for size in array.shape).encode("ascii")
    return hashlib.sha256(b"float32:" + shape + b"\0" + array.tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _replay_record(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Extract and check the fields a replay record must carry."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: replay metadata must be a JSON object.")
    for field in ("mode", "task", "physics", "selected_motion"):
        _require(
            isinstance(payload.get(field), str) and bool(payload[field]),
            f"{source}: replay metadata is missing {field!r}.",
        )
    _require(
        bool(payload.get("full_reference_horizon_requested")),
        f"{source}: replay did not request the full Reference horizon.",
    )
    _require(
        bool(payload.get("full_reference_horizon_completed")),
        f"{source}: replay did not complete the full Reference horizon "
        f"(termination_step={payload.get('termination_step')!r}, "
        f"fired={payload.get('fired_termination_terms')!r}).",
    )
    return dict(payload)


def _state_lock_evidence(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    record = _replay_record(payload, source=source)
    _require(
        record["mode"] == STATE_LOCK_MODE,
        f"{source}: expected a {STATE_LOCK_MODE!r} replay, got {record['mode']!r}.",
    )
    state_lock = record.get("state_lock")
    _require(
        isinstance(state_lock, Mapping),
        f"{source}: state-lock replay carries no state_lock record.",
    )
    _require(
        bool(state_lock.get("passed")),
        f"{source}: state-lock replay did not pass its tolerances: {state_lock!r}.",
    )
    return {
        "mode": STATE_LOCK_MODE,
        "task": record["task"],
        "physics": record["physics"],
        "completed_replay_steps": record.get("completed_replay_steps"),
        "state_lock": dict(state_lock),
    }


def _unassisted_evidence(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    record = _replay_record(payload, source=source)
    _require(
        record["mode"] == UNASSISTED_MODE,
        f"{source}: expected a {UNASSISTED_MODE!r} replay, got {record['mode']!r}.",
    )
    _require(
        bool(record.get("zero_virtual_object_controller_verified")),
        f"{source}: the virtual object controller was not verified to be zero.",
    )
    _require(
        record.get("assistance_violation_step") is None,
        f"{source}: virtual object assistance re-engaged at step "
        f"{record.get('assistance_violation_step')!r}.",
    )
    _require(
        bool(record.get("completed_without_termination")),
        f"{source}: unassisted replay terminated early at step "
        f"{record.get('termination_step')!r}.",
    )
    dynamics = record.get("unassisted_dynamics")
    _require(
        isinstance(dynamics, Mapping),
        f"{source}: unassisted replay carries no unassisted_dynamics record.",
    )
    # Drop the per-step table; the attestation keeps the summary only.
    summary = {key: value for key, value in dynamics.items() if key != "per_step"}
    return {
        "mode": UNASSISTED_MODE,
        "task": record["task"],
        "physics": record["physics"],
        "completed_replay_steps": record.get("completed_replay_steps"),
        "unassisted_dynamics": summary,
    }


def build_attestation(
    *,
    motions: Sequence[Mapping[str, Any]],
    model_sha256: str,
    tool: str,
) -> dict[str, Any]:
    """Assemble the attestation document from per-motion evidence.

    Each entry of ``motions`` must provide ``sequence_id``, ``qpos_sha256``,
    ``state_lock`` and ``unassisted`` payloads (raw replay metadata).
    """

    _require(bool(motions), "An attestation must cover at least one motion.")
    _require(
        isinstance(model_sha256, str) and len(model_sha256) == 64,
        "model_sha256 must be a 64 character digest.",
    )
    records = []
    seen: set[str] = set()
    for entry in motions:
        sequence_id = str(entry["sequence_id"])
        qpos_sha256 = str(entry["qpos_sha256"])
        _require(
            sequence_id not in seen,
            f"Motion {sequence_id!r} appears more than once in the attestation.",
        )
        seen.add(sequence_id)
        _require(
            len(qpos_sha256) == 64,
            f"Motion {sequence_id!r} has a malformed qpos digest.",
        )
        state_lock = _state_lock_evidence(
            entry["state_lock"], source=f"{sequence_id} state-lock"
        )
        unassisted = _unassisted_evidence(
            entry["unassisted"], source=f"{sequence_id} unassisted"
        )
        _require(
            state_lock["task"] == unassisted["task"],
            f"Motion {sequence_id!r}: the two replays used different tasks.",
        )
        records.append(
            {
                "sequence_id": sequence_id,
                "qpos_float32_sha256": qpos_sha256,
                "model_sha256": model_sha256,
                "state_lock": state_lock,
                "unassisted": unassisted,
            }
        )
    return {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "tool": str(tool),
        "model_sha256": model_sha256,
        "motions": records,
    }


def load_attestation(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Vega-Wuji runtime-promotion attestation is missing: {source}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, Mapping),
        f"{source}: attestation must be a JSON object.",
    )
    _require(
        payload.get("schema_version") == ATTESTATION_SCHEMA_VERSION,
        f"{source}: unsupported attestation schema "
        f"{payload.get('schema_version')!r}; expected "
        f"{ATTESTATION_SCHEMA_VERSION!r}.",
    )
    _require(
        isinstance(payload.get("motions"), list) and bool(payload["motions"]),
        f"{source}: attestation covers no motions.",
    )
    return dict(payload)


def verify_attestation_for_reference(
    attestation: Mapping[str, Any],
    *,
    sequence_id: str,
    qpos: np.ndarray,
    model_sha256: str,
) -> dict[str, Any]:
    """Verify one loaded Reference against the attestation. Returns its record.

    Raises with a specific reason when the Reference is not covered, when its
    motion content does not match what was replayed, or when the replays were
    measured against a different robot model.
    """

    records = [
        record
        for record in attestation["motions"]
        if str(record.get("sequence_id")) == str(sequence_id)
    ]
    _require(
        bool(records),
        f"Reference {sequence_id!r} has no runtime-promotion record in the "
        "attestation. Every motion must be replayed and promoted.",
    )
    _require(
        len(records) == 1,
        f"Reference {sequence_id!r} has {len(records)} attestation records.",
    )
    record = records[0]
    expected_qpos = canonical_qpos_sha256(qpos)
    _require(
        record.get("qpos_float32_sha256") == expected_qpos,
        f"Reference {sequence_id!r} does not match its promotion record: the "
        "attested motion digest is "
        f"{record.get('qpos_float32_sha256')!r} but the loaded Reference is "
        f"{expected_qpos!r}. Re-run the replays and promote again.",
    )
    _require(
        record.get("model_sha256") == model_sha256,
        f"Reference {sequence_id!r} was promoted against model "
        f"{record.get('model_sha256')!r}, but the runtime model is "
        f"{model_sha256!r}.",
    )
    # Re-check the evidence itself, so editing the sidecar by hand cannot
    # promote a Reference whose replays actually failed.
    _require(
        bool(record.get("state_lock", {}).get("state_lock", {}).get("passed")),
        f"Reference {sequence_id!r} carries a failing state-lock record.",
    )
    unassisted = record.get("unassisted", {})
    _require(
        unassisted.get("mode") == UNASSISTED_MODE
        and isinstance(unassisted.get("unassisted_dynamics"), Mapping),
        f"Reference {sequence_id!r} carries no zero-assistance replay record.",
    )
    return dict(record)
