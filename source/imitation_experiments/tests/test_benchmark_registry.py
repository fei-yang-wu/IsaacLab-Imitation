"""Append-only guarantees and fairness bookkeeping of the benchmark registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imitation_experiments.provenance.benchmark_registry import (
    INDEX_NAME,
    register,
    render_table,
)

_INTERFACE = {
    "command_dim": 258,
    "z_dim": 256,
    "hold": 1,
    "phase_mode": "sin_cos",
    "anchor_mode": "robot_heading",
    "bottleneck": "none",
    "objective": "endpoint",
}


def _summary(**overrides) -> dict:
    payload = {
        "metric_means": {"tracking_mpjpe_mm": 19.59, "tracking_mpjpe_g_mm": 114.4},
        "aggregate": {"fall_free_rate": 0.947, "survival_steps_mean": 472.2},
        "num_metric_rows": 106,
        "num_envs": 150,
        "max_steps": 2000,
        "seed": 0,
        "survival_definition": "no_base_too_low_termination",
        "tracking_terminations_enabled": False,
        "stop_reason": "all_envs_done",
        "steps_run": 1055,
        "metadata": {"label": "bneck_test_oracle"},
        "command": "python eval.py env.data.persist_id=bones_seed_language30_compositionality_v1@f31fd755",
    }
    payload.update(overrides)
    return payload


def _eval_dir(tmp_path: Path, name: str = "eval", **overrides) -> Path:
    # Distinct names matter: `_add` builds its default eval dir eagerly, so a
    # shared path would be overwritten with the complete summary.
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    (d / "summary.json").write_text(json.dumps(_summary(**overrides)))
    return d


def _add(tmp_path: Path, registry: Path, **kwargs):
    defaults = dict(
        registry=registry,
        eval_dir=_eval_dir(tmp_path),
        campaign="bneck10b",
        arm="cont_det_hold1",
        seed=0,
        interface=_INTERFACE,
        checkpoint_path="/data/x/model_step_1.pt",
        checkpoint_tag="seg2_2000289792",
        frames_global=None,
        encoder_path="/data/x/encoder.pt",
        binding_json=None,
        train_dataset="bones_seed_sonic_full_129785@e714bbff",
        protocol="oracle30_fallonly_newton",
        notes="",
        store_weights=False,
    )
    defaults.update(kwargs)
    return register(**defaults)


def test_it_records_metrics_and_appends_to_the_index(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    record = _add(tmp_path, registry)
    assert record.metrics["mpjpe_l_mm"] == pytest.approx(19.59)
    assert record.metrics["mpjpe_g_mm"] == pytest.approx(114.4)
    rows = (registry / INDEX_NAME).read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["record_id"] == record.record_id
    assert (registry / "runs" / record.record_id / "summary.json").is_file()
    assert (registry / "runs" / record.record_id / "record.json").is_file()


def test_a_duplicate_record_is_refused(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    _add(tmp_path, registry)
    # Append-only: re-registering the same eval must not silently overwrite the
    # stored evidence, which is the whole point of the registry.
    with pytest.raises(FileExistsError, match="append-only"):
        _add(tmp_path, registry)


def test_a_different_protocol_is_a_distinct_record(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    _add(tmp_path, registry)
    other = _add(tmp_path, registry, protocol="oracle30_pushed")
    assert other.record_id.endswith("oracle30_pushed")
    assert len((registry / INDEX_NAME).read_text().strip().splitlines()) == 2


def test_a_missing_metric_refuses_registration(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    # MPJPE-G absent = the eval predates the global metric; registering it would
    # put an unverifiable row next to complete ones.
    partial = _eval_dir(
        tmp_path, name="eval_partial", metric_means={"tracking_mpjpe_mm": 22.0}
    )
    with pytest.raises(ValueError, match="mpjpe_g_mm"):
        _add(tmp_path, registry, eval_dir=partial)


def test_segment_local_frames_are_recorded_as_unknown(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    record = _add(tmp_path, registry, frames_global=None)
    assert record.checkpoint["frames_global"] is None
    assert "segment-local" in record.checkpoint["frames_note"]
    # The table must not print a segment-local step as if it were a budget.
    assert "seg-local" in render_table(registry)


def test_the_table_sorts_by_mpjpe_l(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    _add(tmp_path, registry, arm="worse", eval_dir=_eval_dir(tmp_path))
    better = tmp_path / "eval_better"
    better.mkdir()
    (better / "summary.json").write_text(
        json.dumps(
            _summary(
                metric_means={
                    "tracking_mpjpe_mm": 12.0,
                    "tracking_mpjpe_g_mm": 90.0,
                }
            )
        )
    )
    _add(tmp_path, registry, arm="better", eval_dir=better, frames_global=10_000_000_000)
    table = render_table(registry)
    assert table.index("`better`") < table.index("`worse`")
    assert "10.00B" in table


def _weights(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_weights_are_stored_content_addressed(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    tracker = _weights(tmp_path, "t.pt", b"tracker-bytes")
    encoder = _weights(tmp_path, "e.pt", b"encoder-bytes")
    record = _add(
        tmp_path,
        registry,
        local_checkpoint=tracker,
        local_encoder=encoder,
        store_weights=True,
    )
    for kind, src in (("checkpoint", tracker), ("encoder", encoder)):
        blob = registry / getattr(record, kind)["blob"]
        assert blob.is_file()
        assert blob.read_bytes() == src.read_bytes()
        assert getattr(record, kind)["bytes"] == len(src.read_bytes())
    # The store is a model store, never git content.
    assert (registry / "blobs" / ".gitignore").is_file()


def test_an_identical_encoder_is_stored_once(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    encoder = _weights(tmp_path, "e.pt", b"shared-encoder")
    first = _add(
        tmp_path,
        registry,
        checkpoint_tag="seg1",
        local_checkpoint=_weights(tmp_path, "t1.pt", b"tracker-1"),
        local_encoder=encoder,
        store_weights=True,
    )
    second = _add(
        tmp_path,
        registry,
        checkpoint_tag="seg2",
        eval_dir=_eval_dir(tmp_path, name="eval2"),
        local_checkpoint=_weights(tmp_path, "t2.pt", b"tracker-2"),
        local_encoder=encoder,
        store_weights=True,
    )
    # Two records of one arm share an encoder: dedupe by content, not by name.
    assert first.encoder["blob"] == second.encoder["blob"]
    assert first.checkpoint["blob"] != second.checkpoint["blob"]
    assert len(list((registry / "blobs").glob("*.pt"))) == 3


def test_storing_weights_without_the_files_is_refused(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    # Silently registering a row with no backup would defeat the model store.
    with pytest.raises(ValueError, match="local-checkpoint"):
        _add(tmp_path, registry, store_weights=True)


def test_metrics_only_registration_is_explicit(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    record = _add(tmp_path, registry, store_weights=False)
    assert record.checkpoint["blob"] is None
    assert record.encoder["blob"] is None


def test_dataset_provenance_comes_from_the_frozen_eval_command(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    record = _add(tmp_path, registry)
    assert record.dataset["train"] == "bones_seed_sonic_full_129785@e714bbff"
    # Read off the eval's own command line, not a launcher that may drift.
    assert record.dataset["eval"] == "bones_seed_language30_compositionality_v1@f31fd755"


def test_an_eval_without_a_persist_id_is_marked_unknown(tmp_path: Path) -> None:
    registry = tmp_path / "reg"
    registry.mkdir()
    d = _eval_dir(tmp_path, name="eval_nodata", command="python eval.py --headless")
    record = _add(tmp_path, registry, eval_dir=d)
    # Better an explicit "unknown" than a plausible-looking default.
    assert record.dataset["eval"] == "unknown"
