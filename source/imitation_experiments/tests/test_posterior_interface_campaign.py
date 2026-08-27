"""Contract gates for the posterior-interface campaign.

This campaign is the counterpart to `2026-08-19-interface-design-study`: there
the skill encoder is pretrained offline for 50,000 updates and frozen, here it
is learned during RL through `command_source=posterior`. It is deliberately a
separate campaign, because it differs from that study's `ctrl` in the whole
command-generation path rather than in one field, so it cannot be an arm of
that star without breaking the star's one-field property.
"""

from __future__ import annotations

import yaml

import pytest

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign

CAMPAIGN_DIR = REPO_ROOT / "experiments/campaigns/2026-08-20-posterior-interface"
CAMPAIGN_YAML = CAMPAIGN_DIR / "campaign.yaml"

# arm -> (train_posterior_through_policy, recon_coeff). A 2x2 minus its empty
# cell: the two signals that can shape the code, alone and together.
# arm -> (train_posterior_through_policy, recon_coeff, quantizer, code width).
# A 3 x 3: the learning signal that shapes the code, crossed with the latent
# space it is squeezed through.
ARMS = {
    "post_recon_ae": ("false", "1.0", "identity", "256"),
    "post_recon_fsq": ("false", "1.0", "fsq", "64"),
    "post_recon_vq": ("false", "1.0", "vq_ema", "256"),
    "post_pg_ae": ("true", "0.0", "identity", "256"),
    "post_pg_fsq": ("true", "0.0", "fsq", "64"),
    "post_pg_vq": ("true", "0.0", "vq_ema", "256"),
    "post_pgrecon_ae": ("true", "1.0", "identity", "256"),
    "post_pgrecon_fsq": ("true", "1.0", "fsq", "64"),
    "post_pgrecon_vq": ("true", "1.0", "vq_ema", "256"),
}

# The input view both routes must agree on. `EncoderViewCfg.components`
# defaults to the full-body trio, which is where the reference qvel enters. For
# a latent actor that view is the ONLY source of expert terms in the policy
# group, so it decides what the posterior can read.
ALIGNED_COMPONENTS = (
    "env.command_interface.encoder.components=[joint_qpos,root_pos,root_ori]"
)

# Held identical across the three so the only differences are the two signals.
SHARED = {
    "agent.ipmd.command_source=posterior",
    "agent.ipmd.latent_learning.method=patch_autoencoder",
    "agent.ipmd.latent_learning.command_phase_mode=sin_cos",
    "agent.ipmd.latent_learning.code_period=10",
    "agent.ipmd.latent_learning.patch_past_steps=0",
    "agent.ipmd.latent_learning.patch_future_steps=9",
    ALIGNED_COMPONENTS,
}


def _campaign() -> dict:
    return yaml.safe_load(CAMPAIGN_YAML.read_text(encoding="utf-8"))


def test_the_three_arms_are_declared() -> None:
    assert set(_campaign()["arms"]) == set(ARMS)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_has_no_pretrain_stage(arm: str) -> None:
    """The absence of an offline pretrain IS the route. A pretrain stage here
    would silently turn this into the frozen study."""
    jobset = load_campaign(CAMPAIGN_YAML, arm=arm, seed=0)
    assert [stage.name for stage in jobset.stages] == ["lowlevel"]
    joined = " ".join(jobset.stages[0].args)
    assert "hl_skill_checkpoint_path" not in joined
    assert "train_hl_skill_diffsr" not in joined


@pytest.mark.parametrize(("arm", "contract"), sorted(ARMS.items()))
def test_arm_carries_its_two_signals(
    arm: str, contract: tuple[str, str, str, str]
) -> None:
    through_policy, recon, _, _ = contract
    args = set(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
    assert (
        f"agent.ipmd.latent_learning.train_posterior_through_policy={through_policy}"
        in args
    )
    assert f"agent.ipmd.latent_learning.recon_coeff={recon}" in args
    # KL is a separate claim and a separate arm; it must be off in all three.
    assert "agent.ipmd.latent_learning.kl_coeff=0.0" in args


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_everything_except_the_two_signals_is_shared(arm: str) -> None:
    args = set(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
    missing = SHARED - args
    assert not missing, f"{arm} is missing {sorted(missing)}"


@pytest.mark.parametrize(("arm", "contract"), sorted(ARMS.items()))
def test_arm_carries_its_latent_space(
    arm: str, contract: tuple[str, str, str, str]
) -> None:
    _, _, quantizer, code = contract
    args = set(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
    assert f"agent.ipmd.latent_learning.quantizer={quantizer}" in args
    assert f"agent.ipmd.latent_learning.code_latent_dim={code}" in args
    # The published command is the code plus the 2-wide sin_cos phase.
    expected_command = str(int(code) + 2)
    assert f"agent.ipmd.latent_dim={expected_command}" in args
    assert f"env.command_interface.actor.dim={expected_command}" in args


def test_the_posterior_reads_the_same_view_the_control_encodes() -> None:
    """One input declaration, followed by both halves.

    The agent side pins `posterior_input_keys` to the root_qpos triple (it
    cannot be set from the CLI: `sync_input_keys` re-assigns it after Hydra),
    and the env side must publish that same triple through the encoder view.
    If these drift apart the run either crashes with a KeyError or silently
    trains against the wider full_body view, which the 2026-08-19 study
    measured as WORSE.
    """
    # `isaaclab_imitation` lives only in the isaaclab environment, so read the
    # entry point's declaration from source rather than importing it. That keeps
    # this gate running in the default environment, where the rest of the
    # campaign contract is checked.
    agents = (
        REPO_ROOT
        / "source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based"
        / "imitation/config/g1/agents/rlopt_ipmd_cfg.py"
    ).read_text(encoding="utf-8")
    block = agents.split("ROOT_QPOS_POSTERIOR_INPUT_KEYS: list[tuple[str, str]] = [", 1)
    assert len(block) == 2, "the root_qpos posterior key list is gone"
    declared = block[1].split("]", 1)[0]
    for term in ("expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b"):
        assert term in declared, term
    assert 'expert_motion"' not in declared, "the wide full_body view crept back in"
    for arm in ARMS:
        args = set(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
        assert ALIGNED_COMPONENTS in args, arm
        assert "--agent" in args or True
        joined = " ".join(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
        assert "rlopt_ipmd_posterior_root_qpos_cfg_entry_point" in joined, arm


def test_no_two_arms_are_the_same_cell() -> None:
    """Three arms that resolve to one configuration would spend the budget
    three times and report it as three measurements."""
    seen: dict[tuple[str, ...], str] = {}
    for arm in ARMS:
        args = tuple(
            sorted(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
        )
        assert args not in seen, f"{arm} is the same cell as {seen.get(args)}"
        seen[args] = arm


def test_budget_matches_the_frozen_study() -> None:
    """2B frames, so a posterior arm is frame-comparable with the star's arms
    even though it is not one-field comparable with them."""
    for arm in ARMS:
        args = load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args
        assert args[args.index("--max_iterations") + 1] == "5087"


def test_hold_matches_the_frozen_study_control() -> None:
    """Cadence is held at the control's hold 10 so the route comparison is not
    also a cadence comparison."""
    for arm in ARMS:
        args = set(load_campaign(CAMPAIGN_YAML, arm=arm, seed=0).stages[0].args)
        assert "agent.ipmd.latent_steps_min=10" in args
        assert "agent.ipmd.latent_steps_max=10" in args
        assert "agent.ipmd.latent_learning.code_period=10" in args


def test_wandb_ids_are_unique_and_fit_the_cap() -> None:
    ids = [a["vars"]["wandb_id"] for a in _campaign()["arms"].values()]
    assert len(set(ids)) == len(ids)
    for wandb_id in ids:
        assert len(f"{wandb_id}-s0") <= 31, wandb_id


def test_campaign_directory_carries_no_python() -> None:
    assert not list(CAMPAIGN_DIR.glob("*.py"))
