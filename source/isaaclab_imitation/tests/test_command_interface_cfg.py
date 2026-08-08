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
    ReferenceSelectionPreset,
    actor_command_keys,
    bind_command_interface,
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


def test_encoder_view_frame_stride_reaches_windows_and_expert_batch():
    """SONIC-style strided windows: the view's stride must flow to both the
    live observation window and the offline expert-batch window, and stay 1
    for the actor's own terms and the critic."""
    cfg = _latent_interface(
        encoder=EncoderViewCfg(
            components=("keypoint_pos", "root_ori"),
            past_steps=0,
            future_steps=9,
            frame_stride=5,
        )
    )
    assert cfg.expert_batch_window() == (0, 9, 5)
    assert cfg.policy_window_for("expert_keypoint_pos_b") == (0, 9, 5)
    assert cfg.policy_window_for("latent_command") == (0, 0, 1)
    assert encoder_command_keys(cfg) == [
        ("policy", "expert_keypoint_pos_b"),
        ("policy", "expert_anchor_ori_b"),
    ]


def test_explicit_actor_frame_stride_reaches_the_expert_batch_window():
    """An explicit actor may be trained and evaluated on a strided window too.

    Without an encoder the offline expert batch is shaped by the actor, so its
    stride has to survive the trip; a stride-1 batch paired with a stride-5
    policy would be silently off-distribution.
    """
    cfg = _explicit_interface(
        components=("joint_qpos_qvel", "root_ori"),
        past_steps=0,
        future_steps=9,
        frame_stride=5,
    )
    assert cfg.expert_batch_window() == (0, 9, 5)


def test_explicit_actor_rejects_non_positive_frame_stride():
    with pytest.raises(ValueError, match="frame_stride"):
        _explicit_interface(future_steps=9, frame_stride=0)


def test_chunk_actor_expert_batch_window_comes_from_the_packet():
    """``ChunkCommandCfg`` carries no past/future fields, only a horizon."""
    cfg = CommandInterfaceCfg(
        reference=_reference(),
        actor=ChunkCommandCfg(source="reference", horizon=10, hold_steps=10),
        encoder=None,
        critic_channels=("reference",),
    )
    cfg.resolve()
    assert cfg.expert_batch_window() == (0, 9, 1)


def test_encoder_view_rejects_non_positive_frame_stride():
    with pytest.raises(ValueError, match="frame_stride"):
        _latent_interface(
            encoder=EncoderViewCfg(past_steps=0, future_steps=9, frame_stride=0)
        )


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"random_trajectory_sampling_ratio": -0.1}, "in \\[0, 1\\]"),
        ({"random_trajectory_sampling_ratio": 1.1}, "in \\[0, 1\\]"),
        ({"random_trajectory_start_fraction": 0.0}, "in \\(0, 1\\]"),
        ({"random_trajectory_start_fraction": 1.1}, "in \\(0, 1\\]"),
        (
            {"random_trajectory_sampling_ratio": 0.8, "full_trajectory": False},
            "requires full_trajectory=true",
        ),
    ],
)
def test_random_adaptive_mixture_selection_is_validated(kwargs, message):
    cfg = CommandInterfaceCfg(
        reference=_reference(selection=ReferenceSelectionCfg(**kwargs)),
        actor=LatentCommandCfg(),
    )
    with pytest.raises(ValueError, match=message):
        cfg.resolve()


def test_random80_adaptive20_preset_is_an_exact_top_level_mixture():
    selection = ReferenceSelectionPreset().random80_adaptive20
    selection.resolve()

    assert selection.full_trajectory
    assert selection.random_trajectory_sampling_ratio == 0.8
    assert selection.random_trajectory_start_fraction == 0.5
    # No hidden uniform-bin share inside the 20% adaptive branch.
    assert selection.adaptive_uniform_ratio == 0.0


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


class _StubAgentCfg:
    """Minimal agent stand-in: records the interface and derives critic keys.

    Mirrors what the real IPMD config does at bind time -- the point of the test
    is *when* normalization happens, not which agent does the deriving.
    """

    def __init__(self):
        self._command_interface = None
        self.critic_keys = None

    def sync_input_keys(self):
        self.critic_keys = critic_input_keys(
            self._command_interface, privileged_keys=PRIVILEGED_KEYS
        )


class _StubEnvCfg:
    def __init__(self, interface):
        self.command_interface = interface


def test_bind_normalizes_a_cli_string_component_list_before_deriving_keys():
    """A Hydra override reaches bind as a raw string, and must still work.

    Binding happens in the training entry point, long before the env
    constructor runs `resolve_late_overrides`, so `critic_components` is still
    whatever Isaac Lab's config updater assigned -- for a `None`-default field
    that is the literal string "[a,b,c]". Without normalization at bind time it
    is iterated character by character and dies with `KeyError: '['` several
    frames away from the override that caused it.
    """
    interface = CommandInterfaceCfg(
        reference=_reference(),
        actor=LatentCommandCfg(dim=258),
        critic_channels=("reference",),
    )
    # Exactly what `env.command_interface.reference.critic_components=[...]`
    # leaves behind: a string, not a sequence.
    interface.reference.critic_components = "[joint_qpos_qvel,root_pos,ee_pos]"

    agent = _StubAgentCfg()
    bound = bind_command_interface(agent, _StubEnvCfg(interface))

    assert bound is interface
    # Canonically ordered, not as written -- ee_pos precedes root_pos.
    assert interface.reference.critic_components == (
        "joint_qpos_qvel",
        "ee_pos",
        "root_pos",
    )
    assert ("critic", "expert_ee_pos_b") in agent.critic_keys


def test_bind_is_idempotent_with_the_environments_own_resolution():
    """Binding resolves, and the env resolving again afterwards is a no-op."""
    interface = CommandInterfaceCfg(
        reference=_reference(critic_components="[root_pos,joint_qpos_qvel]"),
        actor=LatentCommandCfg(dim=258),
        critic_channels=("reference",),
    )
    agent = _StubAgentCfg()
    bind_command_interface(agent, _StubEnvCfg(interface))
    first = agent.critic_keys

    interface.resolve()
    agent.sync_input_keys()
    assert agent.critic_keys == first
