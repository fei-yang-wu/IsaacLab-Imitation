"""The base-field arm must remain a one-variable comparison to PoE."""

from dataclasses import asdict
import json

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign


def test_poe_base_matches_control_except_base_field_and_run_identity():
    campaigns = REPO_ROOT / "experiments/campaigns"
    control = load_campaign(
        campaigns / "2026-09-11-poe-z64-10b/campaign.yaml", arm="z64_poe", seed=0
    )
    treatment = load_campaign(
        campaigns / "2026-09-12-poe-base-z64-10b/campaign.yaml",
        arm="z64_poe_base", seed=0,
    )
    assert treatment.wandb_group == control.wandb_group == "latent64-probe-10b"
    args = treatment.stages[0].args
    assert args[args.index("--z_dim") + 1] == "64"
    args = treatment.stages[1].args
    assert args[args.index("--max_iterations") + 1] == "20346"
    assert len(treatment.stages) == len(control.stages) == 3
    for base_stage, control_stage in zip(treatment.stages, control.stages, strict=True):
        base_spec = json.loads(json.dumps(asdict(base_stage)))
        if base_stage.name == "pretrain":
            args = base_spec["args"]
            phi_index = args.index("--diffsr_phi_parameterization") + 1
            dim_index = args.index("--diffsr_feature_dim") + 1
            assert args[phi_index] == "identity_bias" and args[dim_index] == "65"
            args[phi_index], args[dim_index] = "identity", "64"
        normalized = json.dumps(base_spec).replace("z64_poe_base", "z64_poe")
        normalized = normalized.replace("z64-poe-base", "z64-poe")
        normalized = normalized.replace("l64p-z64poe-base", "l64p-z64poe")
        assert json.loads(normalized) == json.loads(json.dumps(asdict(control_stage)))
