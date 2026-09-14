"""The capacity sweep changes only decoder hidden widths and run identity."""

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign


ARCHITECTURES = {
    "tail4096": [1024, 1024, 4096],
    "tail8192": [1024, 1024, 8192],
    "uniform2048": [2048, 2048, 2048],
    "depth5": [1024, 1024, 2048, 2048, 2048],
    "depth7": [1024, 1024, 2048, 2048, 2048, 2048, 2048],
    "wide_deep": [2048, 2048, 4096, 4096, 4096],
}


def _normalize(args):
    args = list(args)
    for option, expected in (("--encoder_lr", 3e-4), ("--eval_batch_size", 8192)):
        if option in args:
            index = args.index(option)
            assert float(args[index + 1]) == expected
            del args[index:index + 2]
    start = args.index("--diffsr_mu_hidden_dims") + 1
    end = args.index("--batch_size")
    widths = [int(value) for value in args[start:end]]
    args[start:end] = ["HIDDEN_WIDTHS"]
    for option in ("--output_dir", "--wandb_group", "--wandb_run_name"):
        args[args.index(option) + 1] = "RUN_IDENTITY"
    return args, widths


def test_capacity_sweep_matches_completed_z64_control():
    campaigns = REPO_ROOT / "experiments/campaigns"
    control = load_campaign(campaigns / "2026-09-12-poe-rescue-pretrain/campaign.yaml",
                            arm="poe_joint_wide", seed=0)
    baseline, dims = _normalize(control.stages[0].args)
    assert dims == [1024, 1024, 2048]
    for arm, expected in ARCHITECTURES.items():
        campaign = load_campaign(campaigns / "2026-09-13-poe-capacity-pretrain/campaign.yaml",
                                 arm=arm, seed=0)
        assert len(campaign.stages) == 1
        stage = campaign.stages[0]
        assert stage.name == "pretrain"
        assert stage.executable == control.stages[0].executable
        args, dims = _normalize(stage.args)
        assert args == baseline, arm
        assert dims == expected, arm
        for option, value in {"--z_dim": "64", "--diffsr_feature_dim": "64",
                              "--num_updates": "50000", "--batch_size": "8192",
                              "--diffsr_phi_parameterization": "identity",
                              "--diffsr_mu_conditioning": "pair"}.items():
            assert stage.args[stage.args.index(option) + 1] == value
        for field in ("gres", "partition", "qos", "cpus_per_task", "mem", "time_limit"):
            assert getattr(stage, field) == getattr(control.stages[0], field)


def test_batch_sweep_matches_examples_and_evaluation_cadence():
    path = REPO_ROOT / "experiments/campaigns/2026-09-13-poe-capacity-pretrain/campaign.yaml"
    expected = {
        "long_b8192": (8192, 200000, 1000, 3e-4),
        "batch16384": (16384, 100000, 500, 3e-4),
        "batch32768": (32768, 50000, 250, 3e-4),
        "batch32768_lr2": (32768, 50000, 250, 6e-4),
    }
    normalized = []
    for arm, (batch, updates, interval, lr) in expected.items():
        campaign = load_campaign(path, arm=arm, seed=0)
        assert len(campaign.stages) == 1
        args = list(campaign.stages[0].args)
        for option, value in {"--batch_size": batch, "--num_updates": updates,
                              "--log_interval": interval, "--encoder_lr": lr,
                              "--eval_batch_size": 8192}.items():
            assert float(args[args.index(option) + 1]) == value
        assert batch * updates == 1638400000
        assert batch * interval == 8192000
        for examples in (409600000, 819200000, 1638400000):
            assert examples % (batch * interval) == 0
        for option, baseline in (("--batch_size", "8192"), ("--num_updates", "50000"),
                                 ("--log_interval", "1000"), ("--encoder_lr", "0.0003")):
            args[args.index(option) + 1] = baseline
        canonical, dims = _normalize(args)
        assert dims == [1024, 1024, 2048]
        normalized.append(canonical)
    assert all(args == normalized[0] for args in normalized)
