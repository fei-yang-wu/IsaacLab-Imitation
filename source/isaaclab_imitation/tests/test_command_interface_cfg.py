"""Contract tests for the declared command interface (``command_interface.py``).

The interface is the single authority on what the actor, the critic, and the
skill encoder read, so these tests pin the two invariants that used to be spread
across five environment knobs and a hand-maintained agent key list:

1. the ACTOR consumes exactly one command source (never the encoder view that
   shares its observation group), while the CRITIC may read several;
2. an incoherent selection fails at config resolution, not at rollout time.
"""

from __future__ import annotations

import pytest

from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
    ChunkCommandCfg,
    CommandInterfaceCfg,
    EncoderViewCfg,
    ExplicitCommandCfg,
    LatentCommandCfg,
    ReferenceChannelCfg,
    ReferenceSelectionCfg,
    actor_command_keys,
    actor_input_keys,
    command_space_components,
    critic_command_keys,
    critic_input_keys,
    encoder_command_keys,
    normalize_command_components,
)

PROPRIO_KEYS = [
    ("policy", "base_ang_vel"),
    ("policy", "joint_pos_rel"),
    ("policy", "joint_vel_rel"),
    ("policy", "last_action"),
]
PRIVILEGED_KEYS = [
    ("critic", "body_pos"),
    ("critic", "body_ori"),
    ("critic", "base_lin_vel"),
]
TRACKED_BODIES = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]
EE_BODIES = ["left_rubber_hand", "right_rubber_hand"]
KEYPOINT_BODIES = ["pelvis", "left_rubber_hand", "right_rubber_hand"]


def _reference(**kwargs) -> ReferenceChannelCfg:
    params = {
        "mpjpe_body_names": list(TRACKED_BODIES),
        "ee_body_names": list(EE_BODIES),
        "keypoint_body_names": list(KEYPOINT_BODIES),
    }
    params.update(kwargs)
    return ReferenceChannelCfg(**params)


def _latent_interface(**kwargs) -> CommandInterfaceCfg:
    params = {
        "reference": _reference(),
        "actor": LatentCommandCfg(dim=258),
        "encoder": EncoderViewCfg(),
        "critic_channels": ("actor", "reference"),
    }
    params.update(kwargs)
    cfg = CommandInterfaceCfg(**params)
    cfg.resolve()
    return cfg


def _explicit_interface(**actor_kwargs) -> CommandInterfaceCfg:
    cfg = CommandInterfaceCfg(
        reference=_reference(),
        actor=ExplicitCommandCfg(**actor_kwargs),
        encoder=None,
        critic_channels=("reference",),
    )
    cfg.resolve()
    return cfg


# ---------------------------------------------------------------------------
# Component vocabulary.
# ---------------------------------------------------------------------------


def test_components_are_canonically_ordered_not_as_written():
    assert normalize_command_components(
        ["root_ori", "joint_qpos_qvel", "root_pos"]
    ) == (
        "joint_qpos_qvel",
        "root_pos",
        "root_ori",
    )


def test_components_accept_the_hydra_string_form_and_aliases():
    assert normalize_command_components("[qpos,root_position,root_orientation]") == (
        "joint_qpos",
        "root_pos",
        "root_ori",
    )


@pytest.mark.parametrize(
    "components",
    [
        [],
        ["not_a_component"],
        ["root_pos", "root_pos"],
        ["joint_qpos_qvel", "joint_qpos"],
    ],
)
def test_incoherent_component_sets_are_rejected(components):
    with pytest.raises(ValueError):
        normalize_command_components(components)


def test_named_command_spaces_are_component_tuples():
    assert command_space_components("root_qpos") == (
        "joint_qpos",
        "root_pos",
        "root_ori",
    )
    with pytest.raises(ValueError):
        command_space_components("not_a_space")


# ---------------------------------------------------------------------------
# The one-actor-source invariant.
# ---------------------------------------------------------------------------


def test_latent_actor_reads_only_the_latent_command():
    cfg = _latent_interface()
    assert cfg.actor_kind() == "latent"
    assert actor_command_keys(cfg) == [("policy", "latent_command")]
    assert actor_input_keys(cfg, proprio_keys=PROPRIO_KEYS) == [
        ("policy", "latent_command"),
        *PROPRIO_KEYS,
    ]


def test_encoder_view_shares_the_group_but_never_the_actor_contract():
    cfg = _latent_interface(encoder=EncoderViewCfg(past_steps=8))
    encoder_keys = encoder_command_keys(cfg)
    assert encoder_keys == [
        ("policy", "expert_motion"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]
    # The group is a superset; the actor contract is not.
    assert set(encoder_keys).issubset(
        {("policy", name) for name in cfg.policy_command_terms()}
    )
    assert not set(encoder_keys) & set(actor_command_keys(cfg))


def test_explicit_actor_reads_its_components_in_canonical_order():
    cfg = _explicit_interface(components=("root_ori", "keypoint_pos", "root_pos"))
    assert cfg.actor_kind() == "explicit"
    assert actor_command_keys(cfg) == [
        ("policy", "expert_keypoint_pos_b"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]


def test_chunk_actor_is_a_published_explicit_packet():
    cfg = CommandInterfaceCfg(
        reference=_reference(),
        actor=ChunkCommandCfg(horizon=10, hold_steps=10),
        critic_channels=("reference",),
    )
    cfg.resolve()
    assert cfg.actor_kind() == "chunk"
    assert cfg.actor.past_steps == 0
    assert cfg.actor.future_steps == 9
    assert actor_command_keys(cfg) == [
        ("policy", "expert_motion"),
        ("policy", "expert_anchor_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]


# ---------------------------------------------------------------------------
# The critic may read several channels.
# ---------------------------------------------------------------------------


def test_latent_critic_reads_both_channels():
    cfg = _latent_interface()
    assert critic_command_keys(cfg) == [
        ("critic", "latent_command"),
        ("critic", "expert_motion"),
        ("critic", "expert_anchor_pos_b"),
        ("critic", "expert_anchor_ori_b"),
    ]
    assert critic_input_keys(cfg, privileged_keys=PRIVILEGED_KEYS)[-3:] == (
        PRIVILEGED_KEYS
    )


def test_latent_critic_can_be_restricted_to_the_reference_channel():
    cfg = _latent_interface(critic_channels=("reference",))
    assert critic_command_keys(cfg) == [
        ("critic", "expert_motion"),
        ("critic", "expert_anchor_pos_b"),
        ("critic", "expert_anchor_ori_b"),
    ]


def test_explicit_critic_mirrors_the_actor_components_by_default():
    cfg = _explicit_interface(components=("joint_qpos", "root_pos", "root_ori"))
    assert cfg.critic_components() == ("joint_qpos", "root_pos", "root_ori")
    assert critic_command_keys(cfg) == [
        ("critic", "expert_motion_qpos"),
        ("critic", "expert_anchor_pos_b"),
        ("critic", "expert_anchor_ori_b"),
    ]


def test_critic_components_can_be_declared_explicitly():
    cfg = CommandInterfaceCfg(
        reference=_reference(critic_components=("joint_qpos_qvel", "root_pos")),
        actor=ExplicitCommandCfg(components=("ee_pos", "ee_ori")),
        critic_channels=("reference",),
    )
    cfg.resolve()
    assert critic_command_keys(cfg) == [
        ("critic", "expert_motion"),
        ("critic", "expert_anchor_pos_b"),
    ]


def test_unknown_or_empty_critic_channels_are_rejected():
    with pytest.raises(ValueError):
        _latent_interface(critic_channels=("planner",))
    with pytest.raises(ValueError):
        _latent_interface(critic_channels=())


# ---------------------------------------------------------------------------
# Fail-fast validation.
# ---------------------------------------------------------------------------


def test_reference_channel_requires_mpjpe_bodies():
    cfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(),
        actor=LatentCommandCfg(),
    )
    with pytest.raises(ValueError, match="mpjpe_body_names"):
        cfg.resolve()


def test_components_needing_bodies_fail_when_the_body_set_is_empty():
    cfg = CommandInterfaceCfg(
        reference=ReferenceChannelCfg(
            mpjpe_body_names=list(TRACKED_BODIES), ee_body_names=[]
        ),
        actor=ExplicitCommandCfg(components=("ee_pos", "ee_ori")),
        critic_channels=("reference",),
    )
    with pytest.raises(ValueError, match="ee_body_names"):
        cfg.resolve()


def test_chunk_needs_one_command_frame_per_held_step():
    cfg = CommandInterfaceCfg(
        reference=_reference(),
        actor=ChunkCommandCfg(horizon=5, hold_steps=10),
        critic_channels=("reference",),
    )
    with pytest.raises(ValueError, match="one command frame per held control step"):
        cfg.resolve()


def test_chunk_rejects_an_agent_publisher():
    cfg = CommandInterfaceCfg(
        reference=_reference(),
        actor=ChunkCommandCfg(source="agent"),
        critic_channels=("reference",),
    )
    with pytest.raises(ValueError):
        cfg.resolve()


def test_latent_dim_must_be_positive():
    cfg = CommandInterfaceCfg(reference=_reference(), actor=LatentCommandCfg(dim=0))
    with pytest.raises(ValueError):
        cfg.resolve()


def test_custom_selection_requires_a_selector():
    cfg = CommandInterfaceCfg(
        reference=_reference(selection=ReferenceSelectionCfg(schedule="custom")),
        actor=LatentCommandCfg(),
    )
    with pytest.raises(ValueError, match="custom_fn"):
        cfg.resolve()


def test_custom_selection_pins_rank_and_frame():
    def pin(env_ids, num_trajectories):  # pragma: no cover - not called here
        return env_ids % num_trajectories

    cfg = CommandInterfaceCfg(
        reference=_reference(
            selection=ReferenceSelectionCfg(
                schedule="custom", custom_fn=pin, start_mode="fixed", start_frame=0
            )
        ),
        actor=LatentCommandCfg(),
    )
    cfg.resolve()
    assert cfg.reference.selection.resolved_start_mode() == "fixed"


def test_auto_start_mode_resolves_from_the_declared_range():
    wide = _reference(selection=ReferenceSelectionCfg(random_step_max=200))
    wide.resolve()
    assert wide.selection.resolved_start_mode() == "random"
    pinned = _reference(
        selection=ReferenceSelectionCfg(random_step_min=0, random_step_max=0)
    )
    pinned.resolve()
    assert pinned.selection.resolved_start_mode() == "fixed"


def test_adaptive_weight_fn_conflicts_with_full_trajectory_sampling():
    cfg = CommandInterfaceCfg(
        reference=_reference(
            selection=ReferenceSelectionCfg(
                full_trajectory=True, adaptive_weight_fn=lambda ranks, steps: ranks
            )
        ),
        actor=LatentCommandCfg(),
    )
    with pytest.raises(ValueError, match="full_trajectory"):
        cfg.resolve()


def test_resolve_is_idempotent():
    cfg = _latent_interface()
    before = (
        actor_command_keys(cfg),
        critic_command_keys(cfg),
        encoder_command_keys(cfg),
    )
    cfg.resolve()
    cfg.resolve()
    assert (
        actor_command_keys(cfg),
        critic_command_keys(cfg),
        encoder_command_keys(cfg),
    ) == before
