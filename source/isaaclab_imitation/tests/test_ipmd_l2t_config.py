"""Configuration and task-registration contracts for IPMD-L2T."""

from __future__ import annotations

import gymnasium as gym
import pytest

import isaaclab_imitation.tasks  # noqa: F401
from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
    bind_command_interface,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents.rlopt_ipmd_l2t_cfg import (
    G1ImitationRLOptIPMDL2TConfig,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v2 import (
    ImitationG1V2EnvCfg,
)
from isaaclab_imitation.tasks.manager_based.imitation.config.g1.surfaces import (
    ImitationG1ChunkSurfaceEnvCfg,
    ImitationG1ExplicitSurfaceEnvCfg,
)


@pytest.mark.parametrize(
    ("env_cfg_cls", "kind", "uses_latent"),
    [
        (ImitationG1V2EnvCfg, "latent", True),
        (ImitationG1ExplicitSurfaceEnvCfg, "explicit", False),
        (ImitationG1ChunkSurfaceEnvCfg, "chunk", False),
    ],
)
def test_ipmd_l2t_v2_input_contracts(env_cfg_cls, kind, uses_latent) -> None:
    agent_cfg = G1ImitationRLOptIPMDL2TConfig()
    env_cfg = env_cfg_cls()

    bind_command_interface(agent_cfg, env_cfg)

    teacher_keys = agent_cfg.policy.get_input_keys()
    student_keys = agent_cfg.ipmd_l2t.student_policy.get_input_keys()
    assert agent_cfg.value_function is not None
    critic_keys = agent_cfg.value_function.get_input_keys()

    assert env_cfg.command_interface.actor_kind() == kind
    assert agent_cfg.ipmd.use_latent_command is uses_latent
    assert teacher_keys == critic_keys
    assert teacher_keys != student_keys
    assert all(key[0] == "critic" for key in teacher_keys)
    assert all(key[0] == "policy" for key in student_keys)

    # Student and teacher retain the established actor architecture even though
    # they consume different observation groups.
    student = agent_cfg.ipmd_l2t.student_policy
    teacher = agent_cfg.policy
    assert student.num_cells == teacher.num_cells
    assert student.activation_fn == teacher.activation_fn
    assert student.output_dim == teacher.output_dim
    assert student.normalize_input == teacher.normalize_input

    teacher_latent = ("critic", "latent_command")
    student_latent = ("policy", "latent_command")
    if uses_latent:
        explicit_teacher_command = {
            ("critic", "expert_motion"),
            ("critic", "expert_anchor_pos_b"),
            ("critic", "expert_anchor_ori_b"),
        }
        assert explicit_teacher_command.issubset(teacher_keys)
        assert teacher_latent not in teacher_keys
        assert student_latent not in teacher_keys
        assert student_latent in student_keys
        assert ("policy", "expert_motion") not in student_keys
        assert ("policy", "expert_anchor_pos_b") not in student_keys
        assert ("policy", "expert_anchor_ori_b") not in student_keys
        assert agent_cfg.ipmd.latent_key == student_latent
        assert agent_cfg.ipmd_l2t.student_latent_key == student_latent
        assert teacher.normalize_input_exclude_keys == []
        assert agent_cfg.value_function.normalize_input_exclude_keys == []
        assert student.normalize_input_exclude_keys == [student_latent]
    else:
        assert teacher_latent not in teacher_keys
        assert student_latent not in student_keys
        assert teacher.normalize_input_exclude_keys == []
        assert student.normalize_input_exclude_keys == []


def test_ipmd_l2t_uses_the_current_tuned_recipe() -> None:
    agent_cfg = G1ImitationRLOptIPMDL2TConfig()

    assert agent_cfg.policy.num_cells == [1024, 1024, 512]
    assert agent_cfg.ipmd_l2t.student_policy.num_cells == [1024, 1024, 512]
    assert agent_cfg.policy.activation_fn == "silu"
    assert agent_cfg.ipmd_l2t.student_policy.activation_fn == "silu"
    assert agent_cfg.loss.gamma == pytest.approx(0.97)
    assert agent_cfg.optim.kl_adapt_step == "iteration"
    assert agent_cfg.ppo.entropy_coeff == pytest.approx(0.0)


def test_ipmd_l2t_entry_point_is_registered_only_on_current_v2_surfaces() -> None:
    entry_point = (
        "isaaclab_imitation.tasks.manager_based.imitation.config.g1.agents."
        "rlopt_ipmd_l2t_cfg:G1ImitationRLOptIPMDL2TConfig"
    )
    for task_id in (
        "Isaac-Imitation-G1-v2",
        "Isaac-Imitation-G1-Explicit-v2",
        "Isaac-Imitation-G1-Chunk-v2",
    ):
        assert gym.spec(task_id).kwargs["rlopt_ipmd_l2t_cfg_entry_point"] == entry_point

    for frozen_task_id in (
        "Isaac-Imitation-G1-v0",
        "Isaac-Imitation-G1-v1",
        "Isaac-Imitation-G1-Latent-v0",
    ):
        assert "rlopt_ipmd_l2t_cfg_entry_point" not in gym.spec(frozen_task_id).kwargs
