"""Protect the fixed-command, pretraining-only comparison contract."""

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign


def test_poe_rescue_protocol_and_single_variable_pairs():
    path = REPO_ROOT / "experiments/campaigns/2026-09-12-poe-rescue-pretrain/campaign.yaml"
    expected = {
        "affine_next": ("affine", "next", "256", "512"),
        "affine_pair": ("affine", "pair", "256", "512"),
        "poe_joint": ("identity", "pair", "64", "512"),
        "poe_joint_wide": ("identity", "pair", "64", "2048"),
        "poe_factored": ("affine_no_bias", "next", "256", "512"),
        "poe_modulated": ("identity", "film", "64", "512"),
    }
    normalized = []
    for arm, (phi, mu, width, hidden) in expected.items():
        campaign = load_campaign(path, arm=arm, seed=0)
        assert len(campaign.stages) == 1
        stage = campaign.stages[0]
        assert stage.name == "pretrain"
        assert stage.executable == "scripts/rlopt/train_hl_skill_diffsr.py"
        args = list(stage.args)
        for option, value in {
            "--z_dim": "64", "--num_updates": "50000", "--batch_size": "8192",
            "--source_history_steps": "5", "--jepa_endpoint_coeff": "0",
            "--jepa_sigreg_coeff": "1.0", "--reg_coeff": "0.001",
            "--jepa_ntp_chunk_span": "boundary_next", "--dynamics_probe_seed": "1729",
            "--diffsr_phi_parameterization": phi, "--diffsr_mu_conditioning": mu,
            "--diffsr_feature_dim": width,
        }.items():
            assert args[args.index(option) + 1] == value
        assert args[args.index("--diffsr_mu_hidden_dims") + 3] == hidden
        assert "--dynamics_probe" in args
        for option in ("--diffsr_phi_parameterization", "--diffsr_mu_conditioning",
                       "--diffsr_feature_dim"):
            args[args.index(option) + 1] = "<architecture>"
        args[args.index("--diffsr_mu_hidden_dims") + 3] = "<hidden>"
        normalized.append([a.replace(arm, "ARM").replace(arm.replace("_", "-"), "ARM")
                           for a in args])
    assert all(args == normalized[0] for args in normalized)
