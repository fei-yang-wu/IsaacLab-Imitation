from __future__ import annotations

from pathlib import Path

import torch

from interface_planner_common import (
    CausalInterfaceTransformerDeterministicPlanner,
    CausalInterfaceTransformerDiffusionPlanner,
    ChunkedTransformerFlowPlanner,
    InterfaceTargetSpec,
    load_planner_checkpoint,
    parameter_counts,
    save_planner_checkpoint,
)


MODEL_KWARGS = {
    "state_dim": 12,
    "target_dim": 10,
    "term_widths": (6, 4),
    "d_model": 16,
    "num_layers": 1,
    "num_heads": 4,
    "feedforward_dim": 32,
    "patch_dim": 4,
    "num_state_tokens": 1,
    "language_dim": 5,
    "num_language_tokens": 1,
    "dropout": 0.0,
}


def _models() -> list[torch.nn.Module]:
    return [
        ChunkedTransformerFlowPlanner(**MODEL_KWARGS),
        CausalInterfaceTransformerDiffusionPlanner(**MODEL_KWARGS),
        CausalInterfaceTransformerDeterministicPlanner(**MODEL_KWARGS),
    ]


def test_continuous_families_have_identical_parameter_counts_and_shapes() -> None:
    models = _models()
    counts = [parameter_counts(model)["parameter_count"] for model in models]
    assert len(set(counts)) == 1
    state = torch.randn(3, MODEL_KWARGS["state_dim"])
    language = torch.randn(3, MODEL_KWARGS["language_dim"])
    for model in models:
        prediction = model(
            state,
            num_inference_steps=3,
            inference_noise_std=0.0,
            language=language,
        )
        assert prediction.shape == (3, MODEL_KWARGS["target_dim"])
        assert torch.isfinite(prediction).all()


def test_family_objectives_are_finite_and_differentiable() -> None:
    state = torch.randn(3, MODEL_KWARGS["state_dim"])
    target = torch.randn(3, MODEL_KWARGS["target_dim"])
    language = torch.randn(3, MODEL_KWARGS["language_dim"])
    flow, diffusion, deterministic = _models()
    losses = [
        flow.flow_matching_loss(state, target, language=language),
        diffusion.diffusion_loss(state, target, language=language),
        deterministic.deterministic_loss(state, target, language=language),
    ]
    for loss, model in zip(losses, (flow, diffusion, deterministic), strict=True):
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_new_family_checkpoints_round_trip(tmp_path: Path) -> None:
    target_spec = InterfaceTargetSpec(
        interface="latent_skill", term_names=("z",), term_widths=(10,)
    )
    for model in _models()[1:]:
        checkpoint = tmp_path / model.planner_type / "checkpoints" / "latest.pt"
        save_planner_checkpoint(
            checkpoint,
            planner=model,
            optimizer=None,
            target_spec=target_spec,
            metadata={"test": True},
        )
        loaded, loaded_spec, metadata = load_planner_checkpoint(checkpoint)
        assert type(loaded) is type(model)
        assert loaded_spec == target_spec
        assert metadata == {"test": True}
        assert loaded.config_dict()["planner_type"] == model.planner_type
