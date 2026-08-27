"""Gates for the controlled BONES-SEED latent-design ablation."""

from __future__ import annotations

import pytest

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign

CAMPAIGN_YAML = (
    REPO_ROOT
    / "experiments/campaigns/2026-08-18-bones129k-latent-design-ablation/campaign.yaml"
)

ARMS = {
    "spectral_cont256": ("endpoint", "deterministic", 256, 258),
    "spectral_fsq64": ("endpoint", "sonic_fsq", 64, 66),
    "recon_cont256": ("reconstruction", "deterministic", 256, 258),
    "recon_fsq64": ("reconstruction", "sonic_fsq", 64, 66),
}


def _flag_value(args: tuple[str, ...], flag: str) -> str:
    return args[args.index(flag) + 1]


@pytest.mark.parametrize(("arm", "contract"), ARMS.items())
@pytest.mark.parametrize("seed", [0, 2])
def test_ablation_arm_resolves_with_only_two_design_axes(
    arm: str,
    contract: tuple[str, str, int, int],
    seed: int,
) -> None:
    objective, latent_mode, z_dim, command_dim = contract
    jobset = load_campaign(CAMPAIGN_YAML, arm=arm, seed=seed)
    assert [stage.name for stage in jobset.stages] == [
        "pretrain",
        "lowlevel1",
        "lowlevel2",
        "lowlevel3",
        "lowlevel4",
    ]

    pretrain = jobset.stages[0]
    assert _flag_value(pretrain.args, "--transition_objective") == objective
    assert _flag_value(pretrain.args, "--latent_mode") == latent_mode
    assert _flag_value(pretrain.args, "--z_dim") == str(z_dim)
    assert _flag_value(pretrain.args, "--horizon_steps") == "10"
    assert _flag_value(pretrain.args, "--encoder_window_mode") == "intermediate"
    assert "--no_encoder_layer_norm" in pretrain.args
    assert _flag_value(pretrain.args, "--num_updates") == "50000"
    assert (
        "env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]"
        in pretrain.args
    )
    assert "env.expert_macro_frame_stride=1" in pretrain.args
    assert "env.expert_macro_anchor_mode=robot_heading" in pretrain.args
    expected_objective_tag = (
        "reconstruction" if objective == "reconstruction" else "spectral"
    )
    assert expected_objective_tag in pretrain.env["CLUSTER_WANDB_TAGS"]
    expected_bottleneck_tag = "fsq" if latent_mode == "sonic_fsq" else "continuous"
    assert expected_bottleneck_tag in pretrain.env["CLUSTER_WANDB_TAGS"]

    for index, stage in enumerate(jobset.stages[1:], start=1):
        assert _flag_value(stage.args, "--max_iterations") == "25432"
        assert f"env.command_interface.actor.dim={command_dim}" in stage.args
        assert f"agent.ipmd.latent_dim={command_dim}" in stage.args
        assert "agent.ipmd.latent_steps_min=10" in stage.args
        assert "agent.ipmd.latent_steps_max=10" in stage.args
        assert "agent.ipmd.hl_skill_finetune_enabled=false" in stage.args
        assert "env.data.reference_prefetch_mode=next" in stage.args
        assert stage.env["WANDB_RESUME"] == "allow"
        assert "WANDB_MODE" not in stage.env
        assert len(stage.env["WANDB_RUN_ID"]) <= 31
        assert f"segment{index}" in stage.env["CLUSTER_WANDB_TAGS"]
        if index > 1:
            assert stage.dependency_kind == "afterany"
            assert "env.enable_termination_curriculum=false" in stage.args

    assert jobset.output_container_path == (
        f"/data/bones_latent_ablation/{arm}_seed{seed}"
    )
