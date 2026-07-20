"""Backend-independent G1 policy input/output ordering tests."""

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_29DOF_ISAACLAB_JOINT_NAMES,
    G1SonicActionsCfg,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_latent_env_cfg import (
    G1SonicLatentObservationCfg,
)


def test_sonic_action_uses_canonical_joint_order() -> None:
    action_cfg = G1SonicActionsCfg().joint_pos
    assert action_cfg.joint_names == G1_29DOF_ISAACLAB_JOINT_NAMES
    assert action_cfg.preserve_order is True


def test_sonic_proprioception_uses_canonical_joint_order() -> None:
    observations = G1SonicLatentObservationCfg()
    for group in (observations.policy, observations.critic):
        for term_name in ("joint_pos_rel", "joint_vel_rel"):
            asset_cfg = getattr(group, term_name).params["asset_cfg"]
            assert asset_cfg.joint_names == G1_29DOF_ISAACLAB_JOINT_NAMES
            assert asset_cfg.preserve_order is True
