"""Gate: the 10B bottleneck campaign resolves with functional W&B naming.

The campaign was added 2026-08-15 after the 64-D hold-1 dead-zone diagnosis.
Each arm is one pretrain plus four walltime-segmented low-level jobs that all
resume ONE W&B run; RLOpt names that run from ``agent.logger.exp_name`` and
adds each segment's logdir timestamp as a tag.
"""

from __future__ import annotations

import pytest

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign

CAMPAIGN_DIR = REPO_ROOT / "experiments/campaigns/2026-08-15-latent-bottleneck-10b"
CAMPAIGN_YAML = CAMPAIGN_DIR / "campaign.yaml"


@pytest.mark.skipif(not CAMPAIGN_YAML.exists(), reason="untracked campaign spec absent")
def test_functional_wandb_name_is_stable_across_resume_segments() -> None:
    jobset = load_campaign(CAMPAIGN_YAML, arm="fsq64_hold10", seed=1)

    assert [s.name for s in jobset.stages] == [
        "pretrain",
        "lowlevel1",
        "lowlevel2",
        "lowlevel3",
        "lowlevel4",
    ]

    pretrain_args = jobset.stages[0].args
    run_name_flag = pretrain_args.index("--wandb_run_name")
    assert pretrain_args[run_name_flag + 1] == "fsq64-hold10-pretrain-s1"

    for index, stage in enumerate(jobset.stages[1:], start=1):
        assert "agent.logger.exp_name=fsq64-hold10-s1" in stage.args
        assert stage.env["WANDB_RUN_ID"] == "fsq64-hold10-s1"
        assert f"segment{index}" in stage.env["CLUSTER_WANDB_TAGS"]
