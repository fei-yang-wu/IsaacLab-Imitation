"""Backend-independent G1 policy input/output ordering tests.

Isaac Lab physics backends enumerate the G1 articulation differently: PhysX is
breadth-first, Newton/MJWarp is depth-first per limb, and 27 of 29 joint slots
differ. ``G1_29DOF_ISAACLAB_JOINT_NAMES`` is the pinned canonical order and it
happens to be the PhysX order, so a term that resolves indices from the *live*
articulation silently works under PhysX and permutes under Newton.

A ``SceneEntityCfg`` only pins an ordering when ``preserve_order=True``;
otherwise the resolved indices are ascending in the live enumeration and the
selected values stay in live order. Every joint-space term the policy consumes
must therefore be pinned, not just proprioception.

See ``wiki/sim2sim-backend-verification.md``.
"""

import pytest

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1SonicActionsCfg,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_latent_env_cfg import (
    G1SonicLatentObservationCfg,
)


# Every joint-space observation term, per group, that must be pinned. These are
# the terms whose values are paired slot-for-slot with proprioception or with
# the action vector by the policy, the latent posterior, or the IPMD reward.
JOINT_SPACE_TERMS = {
    "policy": ("joint_pos_rel", "joint_vel_rel", "expert_motion"),
    "critic": (
        "joint_pos_rel",
        "joint_vel_rel",
        "joint_pos",
        "joint_vel",
        "expert_motion",
    ),
    "expert_state": ("expert_motion", "joint_pos", "joint_vel"),
    "expert_goal": ("expert_motion",),
    "expert_window": ("expert_motion",),
    "reward_input": ("expert_motion",),
}


def test_sonic_action_uses_canonical_joint_order() -> None:
    action_cfg = G1SonicActionsCfg().joint_pos
    assert action_cfg.joint_names == G1_29DOF_ISAACLAB_JOINT_NAMES
    assert action_cfg.preserve_order is True


def test_joint_space_observations_are_pinned() -> None:
    """No joint-space policy input may resolve indices from the live articulation.

    Regression guard for the leak fixed on 2026-07-21, where the expert command
    terms were selected by name but not pinned by order, so a Newton-trained
    checkpoint encoded a 27-of-29-slot permutation and collapsed under PhysX.
    """
    observations = G1SonicLatentObservationCfg()
    checked = 0
    for group_name, term_names in JOINT_SPACE_TERMS.items():
        group = getattr(observations, group_name, None)
        if group is None:
            continue
        for term_name in term_names:
            term = getattr(group, term_name, None)
            if term is None:
                continue
            asset_cfg = term.params["asset_cfg"]
            assert asset_cfg.joint_names == G1_29DOF_ISAACLAB_JOINT_NAMES, (
                f"{group_name}.{term_name} must select the canonical joint "
                f"list, got {asset_cfg.joint_names}"
            )
            assert asset_cfg.preserve_order is True, (
                f"{group_name}.{term_name} sets preserve_order=False, so its "
                "values stay in the live articulation order and permute between "
                "physics backends"
            )
            checked += 1
    assert checked, "no joint-space terms were checked"


def test_no_unpinned_joint_selection_anywhere() -> None:
    """Catch any future joint-space term that forgets ``preserve_order=True``.

    Broader than the explicit table above: it also covers terms that do not
    exist yet, which is how the original leak escaped the previous version of
    this test.
    """
    observations = G1SonicLatentObservationCfg()
    offenders = []
    for group_name in dir(observations):
        if group_name.startswith("_"):
            continue
        group = getattr(observations, group_name)
        for term_name in dir(group):
            if term_name.startswith("_"):
                continue
            params = getattr(getattr(group, term_name), "params", None)
            if not isinstance(params, dict):
                continue
            for param_value in params.values():
                joint_names = getattr(param_value, "joint_names", None)
                if not joint_names or len(joint_names) <= 1:
                    continue
                if not getattr(param_value, "preserve_order", False):
                    offenders.append(f"{group_name}.{term_name}")
    assert not offenders, (
        f"joint selections without preserve_order=True: {sorted(set(offenders))}"
    )


@pytest.mark.parametrize("term_name", ["expert_motion", "joint_pos_rel"])
def test_command_and_proprioception_share_one_order(term_name: str) -> None:
    """The command and proprioception must be pairable slot-for-slot.

    This is the property the policy actually depends on; pinning both to the
    same list is only the means.
    """
    policy = G1SonicLatentObservationCfg().policy
    reference = policy.joint_pos_rel.params["asset_cfg"].joint_names
    assert getattr(policy, term_name).params["asset_cfg"].joint_names == reference
