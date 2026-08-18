"""Gate: the pilot campaign.yaml resolves correctly for every arm.

This campaign was cut over to the control plane 2026-08-15 (real-ICE jobs
5577564/5577565, full pretrain -> afterok -> lowlevel chain, exit 0);
run.sh is now a deprecation shim (see test_cluster_legacy_deprecated.py) and
no longer produces comparable command lines, so the transitional parity test
against its MODE=print output was removed along with it.
"""

from __future__ import annotations

import pytest

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign

CAMPAIGN_DIR = REPO_ROOT / "experiments/campaigns/2026-08-14-latent-quant-ice-repeats"
CAMPAIGN_YAML = CAMPAIGN_DIR / "campaign.yaml"

ARMS = (
    "fsq64",
    "cont_det",
    "cont_det_ln",
    "group_vq",
    "jepa_pure",
    "jepa_sigreg_ebm",
    "fsq64_s5",
    "fsq64_ln",
    "fsq64_dyn",
    "fsq64_curriculum",
    "fsq64_dyn_smooth_curriculum",
    "fsq64_smooth",
    "fsq64_ln_dyn_smooth",
    "cont_det_ln_dyn",
)


@pytest.mark.skipif(not CAMPAIGN_YAML.exists(), reason="untracked campaign spec absent")
@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("seed", [0, 3])
def test_campaign_yaml_resolves_all_arms(arm: str, seed: int) -> None:
    jobset = load_campaign(CAMPAIGN_YAML, arm=arm, seed=seed)
    assert [s.name for s in jobset.stages] == ["pretrain", "lowlevel"]
    assert jobset.stages[1].depends_on == "pretrain"
    pretrain_args = jobset.stages[0].args
    has_ln = "--encoder_layer_norm" in pretrain_args
    assert has_ln == ("_ln" in arm), f"LayerNorm mismatch for arm {arm}"
    assert jobset.output_container_path == f"/data/quant_repeats/{arm}_seed{seed}"

    # W&B naming convention (2026-08-16): functional arm-seed names only;
    # RLOpt adds the generated logdir timestamp back as a ``logdir:...`` tag.
    run_name_flag = pretrain_args.index("--wandb_run_name")
    assert (
        pretrain_args[run_name_flag + 1] == f"{arm.replace('_', '-')}-pretrain-s{seed}"
    )
    lowlevel_args = jobset.stages[1].args
    assert f"agent.logger.exp_name={arm.replace('_', '-')}-s{seed}" in lowlevel_args


@pytest.mark.skipif(not CAMPAIGN_YAML.exists(), reason="untracked campaign spec absent")
def test_curriculum_ramp_uses_frame_cap() -> None:
    # Regression coverage for the FRAME_CAP-used-before-defined bug (fixed
    # 2026-08-15); previously exercised through run.sh, now the arm logic
    # lives only in this YAML.
    jobset = load_campaign(
        CAMPAIGN_YAML,
        arm="fsq64_curriculum",
        seed=0,
        overrides=["vars.frame_cap=2000000000"],
    )
    lowlevel_args = jobset.stages[1].args
    assert (
        "env.command_interface.reference.selection.adaptive_ratio_ramp_frames=2000000000"
        in lowlevel_args
    )


@pytest.mark.skipif(not CAMPAIGN_YAML.exists(), reason="untracked campaign spec absent")
@pytest.mark.parametrize(
    "arm",
    [a for a in ARMS if "_dyn" in a],
)
def test_dyn_arms_carry_online_dynamics_block(arm: str) -> None:
    jobset = load_campaign(CAMPAIGN_YAML, arm=arm, seed=0)
    lowlevel_args = jobset.stages[1].args
    assert "agent.ipmd.hl_skill_finetune_enabled=true" in lowlevel_args
    assert "agent.ipmd.hl_skill_pg_coeff=0" in lowlevel_args
