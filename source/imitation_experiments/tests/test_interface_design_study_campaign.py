"""Contract gates for the interface design study.

The study is a STAR: `ctrl` is the hub and every other arm changes exactly one
field. These tests hold that property, because a second accidental difference
turns an ablation arm into an uninterpretable one, and hold the width contract
the trainer enforces at startup, because violating it kills a job after the
queue wait rather than here.
"""

from __future__ import annotations

import yaml

import pytest

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.pipeline.cluster.config import load_campaign

CAMPAIGN_DIR = REPO_ROOT / "experiments/campaigns/2026-08-19-interface-design-study"
CAMPAIGN_YAML = CAMPAIGN_DIR / "campaign.yaml"

# The hub, as the README states it. An arm may differ from this in exactly the
# fields its own row names, and in nothing else.
CONTROL = {
    "objective": "endpoint",
    "latent_mode": "deterministic",
    "z_dim": 256,
    "command_dim": 258,
    "window_mode": "intermediate",
    "macro_terms": "[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]",
    "stride": 1,
    "anchor_mode": "robot_heading",
    "hold": 10,
    "phase_mode": "sin_cos",
    "command_mode": "z",
}

# arm -> the design fields it is ALLOWED to move. `z_dim`, `command_dim` and
# `code_latent_dim` travel together, so a width change counts as one field.
WIDTH = {"z_dim", "command_dim", "code_latent_dim"}
ALLOWED: dict[str, set[str]] = {
    "ctrl": set(),
    "obj_endpoint_delta": {"objective"},
    "obj_state_occupancy": {"objective"},
    "obj_semimarkov": {"objective"},
    "obj_recon": {"objective"},
    "obj_jepa_ntp": {"objective", "objective_args"},
    "obj_jepa_sigreg_ebm": {"objective", "objective_args"},
    "obj_jepa_infonce": {"objective", "objective_args"},
    "obj_phi_bilinear": {"objective_args"},
    "bn_gaussian": {"latent_mode", "mode_args"},
    "bn_sonic_fsq64": {"latent_mode", "mode_args"} | WIDTH,
    "bn_vq_ema": {"latent_mode", "mode_args"},
    "bn_gumbel_multicat": {"latent_mode", "mode_args"} | WIDTH,
    "bn_cont64": WIDTH,
    "bn_cont128": WIDTH,
    "bn_sonic_fsq32": {"latent_mode", "mode_args"} | WIDTH,
    "bn_sonic_fsq16": {"latent_mode", "mode_args"} | WIDTH,
    "bn_sonic_fsq64_l8": {"latent_mode", "mode_args"} | WIDTH,
    "bn_gumbel": {"latent_mode", "mode_args"},
    "bn_categorical": {"latent_mode", "mode_args"} | WIDTH,
    "bn_no_ln": {"ln_args"},
    "in_fullbody670": {"macro_terms"},
    "in_stride5": {"stride"},
    "in_window_full": {"window_mode"},
    "in_anchor_robot": {"anchor_mode"},
    "in_anchor_expert_heading": {"anchor_mode"},
    "use_hold1": {"hold"},
    "use_hold5": {"hold"},
    "use_phase_none": {"phase_mode"} | WIDTH,
    "use_cotrain_pg": {"finetune_args"},
    "use_cotrain_sonic": {"finetune_args"},
    "use_cotrain_no_pg": {"finetune_args"},
    "use_phi": {"command_mode"},
    "use_z_phi": {"command_mode"} | WIDTH,
    # The interaction probe is the ONE arm allowed to move two fields, because
    # measuring the width x hold interaction is exactly its purpose.
    "ix_fsq64_hold1": {"latent_mode", "mode_args", "hold"} | WIDTH,
    "ix_fsq64_hold5": {"latent_mode", "mode_args", "hold"} | WIDTH,
}

BOOKKEEPING = {"tier", "wandb_id"}

# Deferred by user decision on 2026-08-19. The co-trained arms wait on an
# encoder-from-checkpoint eval path; phi/z_phi are dropped for now. They stay
# DEFINED so the design is intact, but tier 4 is never planned or submitted.
DEFERRED = {
    "use_cotrain_pg",
    "use_cotrain_sonic",
    "use_cotrain_no_pg",
    "use_phi",
    "use_z_phi",
}


def _campaign() -> dict:
    return yaml.safe_load(CAMPAIGN_YAML.read_text(encoding="utf-8"))


def _merged_vars(campaign: dict, arm: str) -> dict:
    merged = {**campaign["vars"], **campaign["arms"][arm]["vars"]}
    code = merged.get("code_latent_dim", merged["z_dim"])
    if isinstance(code, str) and code.startswith("${"):
        code = merged["z_dim"]
    merged["code_latent_dim"] = int(code)
    return merged


def _flag_value(args: tuple[str, ...], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_every_arm_is_declared_in_the_allowed_map() -> None:
    """A new arm must state which field it is allowed to move."""
    assert set(_campaign()["arms"]) == set(ALLOWED)


@pytest.mark.parametrize("arm", sorted(ALLOWED))
def test_arm_changes_only_the_fields_it_declares(arm: str) -> None:
    """The star property. A second accidental difference makes the arm
    uninterpretable, because it would no longer have a stated control."""
    overrides = set(_campaign()["arms"][arm]["vars"]) - BOOKKEEPING
    # `code_latent_dim` is written on every arm as an interpolation of z_dim,
    # so it only counts as a change when the arm sets it to something else.
    arm_vars = _campaign()["arms"][arm]["vars"]
    if str(arm_vars.get("code_latent_dim", "")).startswith("${"):
        overrides.discard("code_latent_dim")
    assert overrides <= ALLOWED[arm], (
        f"{arm} moves {sorted(overrides - ALLOWED[arm])}, which its row does not declare"
    )


@pytest.mark.parametrize("arm", sorted(ALLOWED))
def test_width_contract_holds(arm: str) -> None:
    """`ipmd.py` requires latent_dim == code_latent_dim + phase_dim and raises
    at startup otherwise -- after the queue wait, which is the expensive place
    to find out."""
    merged = _merged_vars(_campaign(), arm)
    phase_dim = 2 if merged["phase_mode"] == "sin_cos" else 0
    assert int(merged["command_dim"]) == int(merged["code_latent_dim"]) + phase_dim


def test_sonic_fsq_arms_size_their_level_list_to_the_code_width() -> None:
    """`latent_mode=sonic_fsq` publishes the quantizer output directly, so the
    trainer validates z_dim == len(sonic_fsq_levels)."""
    campaign = _campaign()
    checked = 0
    for arm in campaign["arms"]:
        merged = _merged_vars(campaign, arm)
        if merged["latent_mode"] != "sonic_fsq":
            continue
        levels = list(merged["mode_args"])
        assert levels[0] == "--sonic_fsq_levels"
        assert len(levels) - 1 == int(merged["z_dim"]), arm
        checked += 1
    assert checked >= 4


def test_control_arm_is_the_declared_hub() -> None:
    merged = _merged_vars(_campaign(), "ctrl")
    for field, value in CONTROL.items():
        assert merged[field] == value, field
    assert _campaign()["vars"]["ln_args"] == ["--encoder_layer_norm"]
    assert _campaign()["vars"]["finetune_args"] == [
        "agent.ipmd.hl_skill_finetune_enabled=false"
    ]


def test_wandb_ids_are_unique_and_fit_the_cap() -> None:
    """W&B caps a run id at 31 characters, and refuses an id that was ever
    deleted with a 410 that kills the job."""
    campaign = _campaign()
    ids = [arm["vars"]["wandb_id"] for arm in campaign["arms"].values()]
    assert len(set(ids)) == len(ids)
    for wandb_id in ids:
        assert len(f"{wandb_id}-s0") <= 31, wandb_id


@pytest.mark.parametrize(
    "arm",
    [
        "ctrl",
        "bn_sonic_fsq64",
        "use_hold1",
        "use_hold5",
        "ix_fsq64_hold1",
        "ix_fsq64_hold5",
    ],
)
# The study runs seed 0 only (user decision, 2026-08-19). Seed 1 is still
# exercised here so the campaign stays re-seedable if a row later needs a
# repeat -- a resolve failure would otherwise only surface at that point.
@pytest.mark.parametrize("seed", [0, 1])
def test_arm_resolves_to_the_matched_screen_budget(arm: str, seed: int) -> None:
    """Every arm trains to the SAME 2B frames: the study reads budget off the
    curve, so a differing cap would confound the axis it is measuring."""
    jobset = load_campaign(CAMPAIGN_YAML, arm=arm, seed=seed)
    assert [stage.name for stage in jobset.stages] == [
        "pretrain",
        "lowlevel1",
        "lowlevel2",
    ]
    merged = _merged_vars(_campaign(), arm)

    pretrain = jobset.stages[0]
    assert _flag_value(pretrain.args, "--num_updates") == "50000"
    assert _flag_value(pretrain.args, "--horizon_steps") == "10"
    assert _flag_value(pretrain.args, "--z_dim") == str(merged["z_dim"])
    assert _flag_value(pretrain.args, "--latent_mode") == merged["latent_mode"]
    assert _flag_value(pretrain.args, "--seed") == str(seed)

    lowlevel = jobset.stages[1]
    # 5087 iterations x 16384 envs x 24 steps = 2.000B frames.
    assert _flag_value(lowlevel.args, "--max_iterations") == "5087"
    assert "agent.save_interval=250000000" in lowlevel.args
    assert f"agent.ipmd.latent_dim={merged['command_dim']}" in lowlevel.args
    assert f"agent.ipmd.latent_learning.code_period={merged['hold']}" in lowlevel.args
    assert f"agent.ipmd.hl_skill_command_mode={merged['command_mode']}" in lowlevel.args


# Flags whose parser default makes an omitted flag equivalent to an explicit
# one. Two arms that differ only by such an omission are the SAME cell.
PARSER_DEFAULTS = {"--jepa_loss": "sigreg"}


def _resolved_design(merged: dict) -> tuple:
    """One arm's design with parser defaults filled in.

    `obj_jepa_ntp` and a would-be `obj_jepa_sigreg` differ textually -- one
    passes `--jepa_loss sigreg`, the other omits it -- but resolve to the same
    run. The wiring smoke measured both at loss 0.26313257217407227, to every
    digit. Comparing raw overrides cannot see that; this can.
    """
    args = [str(a) for a in merged["objective_args"]]
    for flag, default in PARSER_DEFAULTS.items():
        if flag not in args:
            args += [flag, default]
    code = merged.get("code_latent_dim", merged["z_dim"])
    if isinstance(code, str) and code.startswith("${"):
        code = merged["z_dim"]
    return (
        merged["objective"],
        tuple(sorted(zip(args[::2], args[1::2]))),
        merged["latent_mode"],
        tuple(str(a) for a in merged["mode_args"]),
        int(merged["z_dim"]),
        int(code),
        tuple(str(a) for a in merged["ln_args"]),
        merged["window_mode"],
        merged["macro_terms"],
        int(merged["stride"]),
        merged["anchor_mode"],
        int(merged["hold"]),
        merged["phase_mode"],
        merged["command_mode"],
        tuple(str(a) for a in merged["finetune_args"]),
    )


def test_no_two_arms_are_the_same_cell_once_defaults_resolve() -> None:
    """A star with a duplicated cell spends a full training budget twice and
    reports it as two independent measurements."""
    campaign = _campaign()
    seen: dict[tuple, str] = {}
    for name in campaign["arms"]:
        design = _resolved_design(_merged_vars(campaign, name))
        assert design not in seen, f"{name} is the same cell as {seen.get(design)}"
        seen[design] = name


def test_the_jepa_arm_names_its_energy_explicitly() -> None:
    """Its identity must be on the command line, not in a parser default."""
    merged = _merged_vars(_campaign(), "obj_jepa_ntp")
    assert "--jepa_loss" in [str(a) for a in merged["objective_args"]]


def test_deferred_arms_are_tier_four_and_nothing_else_is() -> None:
    """Tier 4 is the deferred set, exactly. An arm that quietly lands there
    would silently drop out of the study."""
    campaign = _campaign()
    tier_four = {
        name
        for name, arm in campaign["arms"].items()
        if int(arm["vars"].get("tier", 1)) == 4
    }
    assert tier_four == DEFERRED


def test_the_active_study_is_the_non_deferred_arms() -> None:
    campaign = _campaign()
    tiers: dict[int, int] = {}
    for arm in campaign["arms"].values():
        tier = int(arm["vars"].get("tier", 1))
        tiers[tier] = tiers.get(tier, 0) + 1
    # 18 core + 11 supporting + 2 interaction probes = 31 active, 5 deferred.
    # The hold-5 pair joined on 2026-08-26, so the hold axis reads 10 / 5 / 1
    # at both code widths instead of only at its ends.
    assert tiers == {1: 18, 2: 11, 3: 2, 4: 5}


def test_plan_all_refuses_the_deferred_tier() -> None:
    """Planning a deferred arm would freeze a command for something that
    cannot yet be scored."""
    plan_all = (CAMPAIGN_DIR / "plan_all.sh").read_text(encoding="utf-8")
    assert "tier 4 is deferred" in plan_all
    assert "FORCE_DEFERRED" in plan_all


def test_wandb_group_is_the_confirmed_one() -> None:
    assert _campaign()["wandb_group"] == "interface-design-study"


def test_plan_all_defaults_to_one_seed() -> None:
    """One seed per arm. With no repeat, every difference inside the evaluation
    band is directional, which the README has to say and this pins."""
    plan_all = (CAMPAIGN_DIR / "plan_all.sh").read_text(encoding="utf-8")
    assert 'SEEDS="${SEEDS:-0}"' in plan_all


def test_the_launchers_exist_and_never_submit() -> None:
    for name in ("submit.sh", "plan_all.sh", "eval.sh", "smoke.sh"):
        path = CAMPAIGN_DIR / name
        assert path.is_file(), name
    # `submit.sh` plans; the control plane prints the submit line separately.
    submit = (CAMPAIGN_DIR / "submit.sh").read_text(encoding="utf-8")
    assert "cluster plan" in submit
    assert "cluster submit" not in submit


def test_campaign_directory_carries_no_python() -> None:
    """A campaign directory is thin: README, config, launchers. Shared
    experiment Python lives in the package with a test."""
    assert not list(CAMPAIGN_DIR.glob("*.py"))
