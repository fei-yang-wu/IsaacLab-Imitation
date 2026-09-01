"""The curve-eval campaign must copy the training interface, never guess it."""

from __future__ import annotations

import yaml

from imitation_experiments.reporting.build_curve_eval_campaign import build

TREE = "/shared/archive"
EVAL = "/data/eval/curves"


def _training_campaign(tmp_path, arms):
    doc = {
        "name": "train",
        "vars": {
            "task": "Isaac-Imitation-G1-v2",
            "z_dim": 64,
            "command_dim": 66,
            "code_latent_dim": "${vars.z_dim}",
            "hold": 1,
            "horizon": 10,
            "phase_mode": "sin_cos",
            "phase_source": "hold",
            "phase_period": 0,
            "command_mode": "z",
            "macro_terms": "[expert_motion_qpos]",
            "stride": 1,
            "anchor_mode": "robot_heading",
            "route": "pretrained",
            "extra_args": [],
        },
        "arms": arms,
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _args(text, arm):
    doc = yaml.safe_load(text)
    return doc["arms"][arm]["stages"][0]["args"]


def test_pretrained_arm_binds_its_own_encoder_and_hl_skill_source(tmp_path):
    path = _training_campaign(tmp_path, {"hub": {"vars": {}}})
    args = _args(build(path, TREE, EVAL, 0), "hub")
    assert "agent.ipmd.command_source=hl_skill" in args
    assert (
        f"agent.ipmd.hl_skill_checkpoint_path={TREE}/hub_seed0/encoder/checkpoints/latest.pt"
        in args
    )
    assert args[args.index("--skill_encoder_source") + 1] == "pretrained"
    assert args[args.index("--tree") + 1] == f"{TREE}/hub_seed0"


def test_posterior_arm_switches_entry_point_and_encoder_source(tmp_path):
    path = _training_campaign(
        tmp_path,
        {
            "post": {
                "vars": {
                    "route": "posterior",
                    "quantizer": "vq_ema",
                    "through_policy": "true",
                    "recon_coeff": "0.0",
                    "posterior_space_args": [
                        "agent.ipmd.latent_learning.codebook_size=512"
                    ],
                }
            }
        },
    )
    args = _args(build(path, TREE, EVAL, 0), "post")
    assert (
        args[args.index("--agent_entry_point") + 1]
        == "rlopt_ipmd_posterior_root_qpos_cfg_entry_point"
    )
    assert args[args.index("--skill_encoder_source") + 1] == "checkpoint"
    assert "agent.ipmd.command_source=posterior" in args
    assert "agent.ipmd.latent_learning.quantizer=vq_ema" in args
    assert "agent.ipmd.latent_learning.recon_coeff=0.0" in args
    assert "agent.ipmd.latent_learning.codebook_size=512" in args
    # A posterior arm has no pretrained encoder file to bind.
    assert not any(a.startswith("agent.ipmd.hl_skill_checkpoint_path") for a in args)


def test_per_arm_interface_overrides_are_carried_not_defaulted(tmp_path):
    path = _training_campaign(
        tmp_path,
        {
            "wide": {"vars": {"z_dim": 256, "command_dim": 258, "hold": 10}},
            "h20": {"vars": {"horizon": 20, "stride": 5, "anchor_mode": "robot"}},
        },
    )
    text = build(path, TREE, EVAL, 0)
    wide = _args(text, "wide")
    assert "env.command_interface.actor.dim=258" in wide
    assert "agent.ipmd.latent_steps_min=10" in wide
    assert "agent.ipmd.latent_learning.code_latent_dim=256" in wide

    h20 = _args(text, "h20")
    # The regression this guards: horizon and stride were hardcoded to 10 and 1
    # in the runner this generator replaces.
    assert "agent.ipmd.hl_skill_horizon_steps=20" in h20
    assert "env.expert_macro_frame_stride=5" in h20
    assert "env.expert_macro_anchor_mode=robot" in h20


def test_code_latent_dim_var_indirection_is_resolved(tmp_path):
    path = _training_campaign(tmp_path, {"hub": {"vars": {}}})
    args = _args(build(path, TREE, EVAL, 0), "hub")
    # `${vars.z_dim}` must become 64, not reach the command line verbatim.
    assert "agent.ipmd.latent_learning.code_latent_dim=64" in args
    assert not any("${vars." in a for a in args)


def test_only_env_extra_args_pass_through(tmp_path):
    path = _training_campaign(
        tmp_path,
        {
            "x": {
                "vars": {
                    "extra_args": [
                        "env.rewards.feet_acc.weight=-2.5e-7",
                        "agent.ipmd.something=1",
                    ]
                }
            }
        },
    )
    args = _args(build(path, TREE, EVAL, 0), "x")
    assert "env.rewards.feet_acc.weight=-2.5e-7" in args
    assert "agent.ipmd.something=1" not in args


def test_every_arm_gets_exactly_one_serialized_scoring_stage(tmp_path):
    path = _training_campaign(
        tmp_path, {"a": {"vars": {}}, "b": {"vars": {}}, "c": {"vars": {}}}
    )
    doc = yaml.safe_load(build(path, TREE, EVAL, 0))
    assert set(doc["arms"]) == {"a", "b", "c"}
    for arm in doc["arms"].values():
        assert len(arm["stages"]) == 1
        stage = arm["stages"][0]
        assert stage["executable"] == "scripts/rlopt/eval_checkpoint_tree.py"
        assert stage["partition"] == "coe-gpu"
        assert stage["qos"] == "coe-ice"


def test_curves_default_to_the_same_board_as_the_tables(tmp_path):
    # Curves and tables must come from ONE corpus: the 2B point of a curve IS
    # the table row. Scoring curves on the 256-clip board would make the two
    # disagree by more than a rounding step and could not be plotted together.
    path = _training_campaign(tmp_path, {"hub": {"vars": {}}})
    args = _args(build(path, TREE, EVAL, 0), "hub")
    assert args[args.index("--board") + 1] == "bones_testbed4096_v1"
    assert args[args.index("--row") + 1] == "clean"


def test_board_and_row_are_overridable(tmp_path):
    path = _training_campaign(tmp_path, {"hub": {"vars": {}}})
    args = _args(
        build(path, TREE, EVAL, 0, board="other_board", row="milestone"), "hub"
    )
    assert args[args.index("--board") + 1] == "other_board"
    assert args[args.index("--row") + 1] == "milestone"


def test_header_warns_against_hand_editing(tmp_path):
    path = _training_campaign(tmp_path, {"hub": {"vars": {}}})
    text = build(path, TREE, EVAL, 0)
    assert "GENERATED" in text
    assert "concurrent-eval crash" in text


def test_each_arm_gets_a_private_isaac_cache(tmp_path):
    # The regression this guards: a shared Isaac Sim cache crashed the second
    # of two concurrent evaluation jobs inside Kit startup (2026-08-27).
    path = _training_campaign(tmp_path, {"a": {"vars": {}}, "b": {"vars": {}}})
    doc = yaml.safe_load(build(path, TREE, EVAL, 0, cache_root="/cache"))
    caches = {
        name: arm["stages"][0]["env"]["CLUSTER_ISAAC_SIM_CACHE_DIR"]
        for name, arm in doc["arms"].items()
    }
    assert caches == {"a": "/cache/a", "b": "/cache/b"}
    assert len(set(caches.values())) == len(caches)
