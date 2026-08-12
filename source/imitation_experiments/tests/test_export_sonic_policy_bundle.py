from imitation_experiments.lowlevel.export_sonic_policy_bundle import (
    action_contract,
    command_contract,
    observation_contract,
)


def test_sonic_bundle_records_term_major_history_contract():
    contract = observation_contract()
    assert contract["total_width"] == 994
    assert [term["name"] for term in contract["terms"]] == [
        "latent_command",
        "base_ang_vel",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "projected_gravity",
    ]
    assert [term["history_length"] for term in contract["terms"]] == [
        1,
        10,
        10,
        10,
        10,
        10,
    ]
    assert all(
        term["history_order"] == "oldest_first" and term["reset_fill"] == "repeat_first"
        for term in contract["terms"]
    )


def test_sonic_bundle_records_reference_encoder_time_contract():
    contract = command_contract(encoder_sha256="a" * 64)
    assert contract["hold_steps"] == 10
    assert contract["encoder_trigger"] == "every_control_tick"
    assert contract["encoder_state_interface"] == "joint_qpos_qvel_anchor_ori"
    assert contract["macro_frame_stride"] == 5
    assert contract["macro_anchor_mode"] == "robot_heading"


def test_sonic_bundle_records_action_clip_and_physical_arrays():
    contract = action_contract()
    assert contract["raw_action_clip"] == 20.0
    for name in (
        "default_joint_pos",
        "action_scale",
        "stiffness",
        "damping",
        "armature",
        "effort_limit",
    ):
        assert len(contract[name]) == 29
