"""Tests for the GR00T action-head integration glue.

Pure-logic tests (key classification, quantile normalization, collation
shape checks) run in the default environment. Tests that construct the real
head skip cleanly when the GR00T dependency set is absent — run those through
``pixi run -e gr00t test-gr00t``.
"""

from __future__ import annotations

import pytest
import torch

from imitation_experiments.planner.gr00t_head import (
    BACKBONE_EMBEDDING_DIM,
    DEBUG_TRUNK_CONFIG,
    classify_head_keys,
    compute_quantile_stats,
    denormalize_minmax,
    filtered_pretrained_load,
    normalize_minmax,
)


def _gr00t_available() -> bool:
    try:
        from imitation_experiments.planner.gr00t_head import import_head_classes

        import_head_classes(stub_model_package=True)
    except Exception:
        return False
    return True


def test_classify_head_keys_partitions_trunk_and_projectors() -> None:
    keys = [
        "model.transformer_blocks.0.attn1.to_q.weight",
        "model.timestep_encoder.timestep_embedder.linear_1.weight",
        "vlln.weight",
        "position_embedding.weight",
        "state_encoder.layer1.W",
        "action_encoder.W1.W",
        "action_decoder.layer2.b",
    ]
    result = classify_head_keys(keys)
    assert result["model.transformer_blocks.0.attn1.to_q.weight"] == "keep"
    assert result["vlln.weight"] == "keep"
    assert result["position_embedding.weight"] == "keep"
    assert result["state_encoder.layer1.W"] == "fresh"
    assert result["action_encoder.W1.W"] == "fresh"
    assert result["action_decoder.layer2.b"] == "fresh"
    assert "other" not in result.values()


def test_classify_head_keys_flags_unknown_keys() -> None:
    assert classify_head_keys(["mystery.weight"])["mystery.weight"] == "other"


def test_quantile_normalization_round_trip_and_mask() -> None:
    values = torch.randn(50, 7, 4) * 3.0 + 1.0
    valid = torch.ones(50, 7, dtype=torch.bool)
    valid[:, -2:] = False
    q01, q99 = compute_quantile_stats(values, valid)
    normalized = normalize_minmax(values, q01, q99)
    restored = denormalize_minmax(normalized, q01, q99)
    assert torch.allclose(restored, values, atol=1.0e-4)
    # Valid-frame values map overwhelmingly into [-1, 1].
    inside = normalized[valid].abs() <= 1.0 + 1.0e-6
    assert float(inside.float().mean()) > 0.97


def test_compute_quantile_stats_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="No valid frames"):
        compute_quantile_stats(
            torch.randn(4, 3, 2), torch.zeros(4, 3, dtype=torch.bool)
        )


@pytest.mark.skipif(not _gr00t_available(), reason="GR00T deps not in this env")
def test_build_batch_and_forward_loss() -> None:
    from imitation_experiments.planner.gr00t_head import (
        build_batch,
        build_g1_head_config,
        import_head_classes,
    )

    config = build_g1_head_config(trunk=DEBUG_TRUNK_CONFIG, action_horizon=6)
    _, head_cls = import_head_classes(stub_model_package=True)
    head = head_cls(config)
    backbone_output, action_input = build_batch(
        state=torch.randn(2, 10, 93),
        action=torch.randn(2, 6, 38),
        action_mask=torch.ones(2, 6, dtype=torch.bool),
        language_features=torch.randn(2, 5, BACKBONE_EMBEDDING_DIM),
        language_attention_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    out = head(backbone_output, action_input)
    assert torch.isfinite(out["loss"])


@pytest.mark.skipif(not _gr00t_available(), reason="GR00T deps not in this env")
def test_filtered_load_round_trip_and_shape_mismatch() -> None:
    from imitation_experiments.planner.gr00t_head import (
        build_g1_head_config,
        import_head_classes,
    )

    config = build_g1_head_config(trunk=DEBUG_TRUNK_CONFIG, action_horizon=6)
    _, head_cls = import_head_classes(stub_model_package=True)
    source_head = head_cls(config)
    target_head = head_cls(config)
    source_state = source_head.state_dict()
    manifest = filtered_pretrained_load(target_head, source_state)
    assert manifest["num_kept_params"] > 0
    assert manifest["fresh"], "expected fresh projector keys"
    target_state = target_head.state_dict()
    for key in manifest["kept"]:
        assert torch.equal(target_state[key], source_state[key])
    # A trunk tensor with the wrong shape must fail loudly, not skip.
    bad = dict(source_state)
    key = next(iter(manifest["kept"]))
    bad[key] = torch.zeros(3, 3)
    fresh_head = head_cls(config)
    with pytest.raises(ValueError, match="shape mismatch"):
        filtered_pretrained_load(fresh_head, bad)
