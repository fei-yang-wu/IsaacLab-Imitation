"""The early-phase rerun must copy the training campaign and change only the
budget, the checkpoint interval, the output tree, the encoder source, the
stage list, and the W&B identity."""

from __future__ import annotations

import pytest
import yaml

from imitation_experiments.pipeline.derive_early_phase_campaign import derive

ARCHIVE = "/shared/archive"
OUT = "/shared/early"


def _stage(name, depends=None):
    stage = {
        "name": name,
        "executable": "scripts/rlopt/train.py",
        "args": "${concat:${vars.lowlevel_head},${vars.lowlevel_tail}}",
        "time_limit": "15:59:00",
        "env": {
            "CLUSTER_WANDB_TAGS": "bones-seed,latent-star-v2,${vars.arm},lowlevel",
            "WANDB_RUN_ID": "${vars.wandb_id}-s${vars.seed}",
        },
    }
    if depends:
        stage["depends_on"] = depends
        stage["dependency_kind"] = "afterany"
    return stage


def _campaign(tmp_path, arms):
    doc = {
        "name": "latent-star-v2",
        "profile": "ice",
        "wandb_project": "g1-bs-ablation",
        "wandb_group": "latent-star-v2",
        "vars": {
            "wandb_id": "lsv2-arm",
            "frame_cap": 5000000000,
            "save_interval": 200000000,
            "output_root": "/data/latent_star_v2/${vars.arm}_seed${vars.seed}",
            "encoder_ckpt": "${vars.output_root}/encoder/checkpoints/latest.pt",
            "hub_encoder": "/data/latent_star_v2/hub_seed0/encoder/checkpoints/latest.pt",
            "lowlevel_head": [
                "--task",
                "Isaac-Imitation-G1-v2",
                "agent.logger.group_name=latent-star-v2",
                "agent.ipmd.hl_skill_checkpoint_path=${vars.encoder_ckpt}",
            ],
            "pretrain_head": ["--wandb_group", "latent-star-v2"],
            "lowlevel_tail": ["agent.save_interval=${vars.save_interval}"],
        },
        "preflight": {
            "require_container_paths": ["/shared/ref_arrays"],
            "output_container_path": "/data/latent_star_v2/${vars.arm}_seed${vars.seed}",
        },
        "arms": arms,
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _derive(path, **kw):
    kw.setdefault("name", "latent-star-v2-early")
    kw.setdefault("wandb_group", "latent-star-v2-early")
    kw.setdefault("frame_cap", 200000000)
    kw.setdefault("save_interval", 9830400)
    kw.setdefault("output_root", OUT)
    kw.setdefault("archive_root", ARCHIVE)
    return yaml.safe_load(derive(path, **kw))


def _three_stage_arm(wandb_id):
    return {
        "vars": {"wandb_id": wandb_id},
        "stages": [
            _stage("pretrain"),
            _stage("lowlevel1", "pretrain"),
            _stage("lowlevel2", "lowlevel1"),
        ],
    }


def test_only_the_first_tracker_segment_survives_without_dependencies(tmp_path):
    doc = _derive(_campaign(tmp_path, {"hub": _three_stage_arm("lsv2-hub")}))
    stages = doc["arms"]["hub"]["stages"]
    assert [s["name"] for s in stages] == ["lowlevel1"]
    assert "depends_on" not in stages[0] and "dependency_kind" not in stages[0]
    assert stages[0]["time_limit"] == "03:00:00"


def test_budget_interval_output_and_encoder_binding_change(tmp_path):
    doc = _derive(_campaign(tmp_path, {"hub": _three_stage_arm("lsv2-hub")}))
    v = doc["vars"]
    assert v["frame_cap"] == 200000000
    assert v["save_interval"] == 9830400
    assert v["output_root"] == f"{OUT}/${{vars.arm}}_seed${{vars.seed}}"
    assert v["archive_root"] == ARCHIVE
    assert v["encoder_ckpt"] == (
        "${vars.archive_root}/${vars.arm}_seed${vars.seed}/encoder/checkpoints/latest.pt"
    )
    assert (
        v["hub_encoder"]
        == "${vars.archive_root}/hub_seed0/encoder/checkpoints/latest.pt"
    )
    # The tracker still binds through the same indirection, so no arm's
    # interface field was touched.
    assert (
        "agent.ipmd.hl_skill_checkpoint_path=${vars.encoder_ckpt}" in v["lowlevel_head"]
    )
    assert doc["preflight"]["output_container_path"] == v["output_root"]
    assert ARCHIVE in doc["preflight"]["require_container_paths"]
    assert "/shared/ref_arrays" in doc["preflight"]["require_container_paths"]


def test_wandb_identity_moves_to_a_new_group_and_id_prefix(tmp_path):
    doc = _derive(_campaign(tmp_path, {"hub": _three_stage_arm("lsv2-hub")}))
    assert doc["name"] == "latent-star-v2-early"
    assert doc["wandb_group"] == "latent-star-v2-early"
    assert doc["arms"]["hub"]["vars"]["wandb_id"] == "lsv2e-hub"
    assert (
        "agent.logger.group_name=latent-star-v2-early" in doc["vars"]["lowlevel_head"]
    )
    assert doc["vars"]["pretrain_head"] == ["--wandb_group", "latent-star-v2-early"]
    tags = doc["arms"]["hub"]["stages"][0]["env"]["CLUSTER_WANDB_TAGS"].split(",")
    assert "latent-star-v2-early" in tags and "latent-star-v2" not in tags
    assert "early-dense" in tags


def test_an_arm_without_pretrain_and_a_hub_bound_arm_pass_through(tmp_path):
    arms = {
        "g1_post_ae": {
            "vars": {"wandb_id": "lsv2-g1postae", "route": "posterior"},
            "stages": [_stage("lowlevel1"), _stage("lowlevel2", "lowlevel1")],
        },
        "g5_hold5": {
            "vars": {"wandb_id": "lsv2-g5hold5", "encoder_ckpt": "${vars.hub_encoder}"},
            "stages": [_stage("lowlevel1"), _stage("lowlevel2", "lowlevel1")],
        },
    }
    doc = _derive(_campaign(tmp_path, arms))
    assert doc["arms"]["g1_post_ae"]["vars"]["route"] == "posterior"
    assert doc["arms"]["g5_hold5"]["vars"]["encoder_ckpt"] == "${vars.hub_encoder}"


def test_refuses_an_arm_without_the_first_tracker_segment(tmp_path):
    arms = {"odd": {"vars": {"wandb_id": "lsv2-odd"}, "stages": [_stage("pretrain")]}}
    with pytest.raises(ValueError, match="lowlevel1"):
        _derive(_campaign(tmp_path, arms))


def test_refuses_an_over_long_run_id_and_a_foreign_prefix(tmp_path):
    arms = {"hub": _three_stage_arm("lsv2-" + "x" * 26)}
    with pytest.raises(ValueError, match="exceeds 31"):
        _derive(_campaign(tmp_path, arms))
    arms = {"hub": _three_stage_arm("other-hub")}
    with pytest.raises(ValueError, match="lacks prefix"):
        _derive(_campaign(tmp_path, arms))


def test_refuses_a_checkpoint_interval_above_the_budget(tmp_path):
    with pytest.raises(ValueError, match="save_interval"):
        _derive(
            _campaign(tmp_path, {"hub": _three_stage_arm("lsv2-hub")}),
            save_interval=300000000,
        )
