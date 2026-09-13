"""Sub-step actuation delay, gain DR term, and anchor-jitter DR (2026-09-13)."""

from __future__ import annotations

import types

import pytest
import torch

from isaaclab_imitation.tasks.manager_based.imitation.mdp.actions import (
    EMAJointPositionAction,
    EMAJointPositionActionCfg,
)


class _Asset:
    def __init__(self, num_envs, dim):
        self.targets = []
        self.data = types.SimpleNamespace(
            default_joint_pos=types.SimpleNamespace(torch=torch.zeros(num_envs, dim))
        )

    def set_joint_position_target_index(self, target, joint_ids):
        self.targets.append(target.clone())


def _term(num_envs=4, dim=3, decimation=4, **overrides):
    """Build the term without an Isaac Lab scene: patch the base __init__."""
    cfg = EMAJointPositionActionCfg(
        asset_name="robot", joint_names=[f"j{i}" for i in range(dim)], **overrides
    )
    term = EMAJointPositionAction.__new__(EMAJointPositionAction)
    asset = _Asset(num_envs, dim)
    term.cfg = cfg
    term._env = types.SimpleNamespace(cfg=types.SimpleNamespace(decimation=decimation))
    term._asset = asset
    term._joint_ids = list(range(dim))
    term._num_envs = num_envs
    term._device = "cpu"
    term._raw_actions = torch.zeros(num_envs, dim)
    term._scale = 1.0
    term._offset = torch.zeros(num_envs, dim)
    term._processed_actions = torch.zeros(num_envs, dim)
    # Replay the parts of __init__ this test exercises.
    term._ema_alpha = 1.0
    term._ema_state = None
    low, high = int(cfg.delay_substeps_min), int(cfg.delay_substeps_max)
    assert 0 <= low <= high <= decimation
    term._delay_range = (low, high)
    term._delay_enabled = high > 0
    term._delay_steps = torch.zeros(num_envs, dtype=torch.long)
    term._previous_target = None
    term._substep = 0
    if term._delay_enabled:
        term._sample_delays(slice(None))
    return term, asset


@pytest.fixture(autouse=True)
def _patch_base(monkeypatch):
    from isaaclab.envs.mdp.actions import joint_actions

    def process(self, actions):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids=None):
        self._raw_actions[env_ids if env_ids is not None else slice(None)] = 0.0

    def apply(self):
        self._asset.set_joint_position_target_index(
            target=self.processed_actions, joint_ids=self._joint_ids
        )

    monkeypatch.setattr(joint_actions.JointAction, "process_actions", process)
    monkeypatch.setattr(joint_actions.JointAction, "reset", reset)
    monkeypatch.setattr(joint_actions.JointPositionAction, "apply_actions", apply)
    monkeypatch.setattr(
        EMAJointPositionAction, "num_envs", property(lambda self: self._num_envs)
    )
    monkeypatch.setattr(EMAJointPositionAction, "device", property(lambda self: self._device))
    monkeypatch.setattr(
        EMAJointPositionAction, "action_dim", property(lambda self: self._raw_actions.shape[1])
    )
    monkeypatch.setattr(
        EMAJointPositionAction,
        "processed_actions",
        property(lambda self: self._processed_actions),
    )


def test_zero_delay_is_the_identity():
    term, asset = _term()
    term.process_actions(torch.ones(4, 3))
    for _ in range(4):
        term.apply_actions()
    assert all(torch.equal(t, torch.ones(4, 3)) for t in asset.targets)


def test_delayed_environments_keep_the_previous_target_for_their_substeps():
    term, asset = _term(delay_substeps_min=0, delay_substeps_max=2)
    term._delay_steps = torch.tensor([0, 1, 2, 2])
    term.process_actions(torch.ones(4, 3))  # previous target: the offset (zeros)
    for _ in range(4):
        term.apply_actions()
    term.process_actions(torch.full((4, 3), 2.0))
    for _ in range(4):
        term.apply_actions()
    second = asset.targets[4:]
    # env 0: new target from sub-step 0; env 1: old for 1 sub-step; env 2/3: old for 2.
    assert second[0][0, 0] == 2.0 and second[0][1, 0] == 1.0 and second[0][2, 0] == 1.0
    assert second[1][1, 0] == 2.0 and second[1][2, 0] == 1.0
    assert second[2][2, 0] == 2.0 and second[3][3, 0] == 2.0


def test_reset_resamples_delay_and_restarts_from_the_offset():
    term, asset = _term(delay_substeps_min=1, delay_substeps_max=1)
    term.process_actions(torch.ones(4, 3))
    term.reset(torch.tensor([1, 3]))
    assert term._previous_target[1].abs().sum() == 0.0
    assert term._previous_target[0].sum() == 0.0  # the pre-first-step offset
    assert torch.equal(term._delay_steps, torch.ones(4, dtype=torch.long))


def test_delay_range_is_validated_against_decimation():
    with pytest.raises(AssertionError):
        _term(delay_substeps_min=0, delay_substeps_max=5)


def test_sonic_events_carry_an_identity_gain_randomization_term():
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.events import (
        G1SonicEventCfg,
    )

    term = G1SonicEventCfg().randomize_actuator_gains
    assert term.mode == "startup"
    assert term.params["stiffness_distribution_params"] == (1.0, 1.0)
    assert term.params["damping_distribution_params"] == (1.0, 1.0)
    assert term.params["operation"] == "scale"


def test_anchor_jitter_offsets_only_xy_and_resets_the_walk():
    from isaaclab_imitation.envs.expert_data_plane import ExpertDataPlane

    plane = ExpertDataPlane.__new__(ExpertDataPlane)
    plane._env = types.SimpleNamespace(
        num_envs=6,
        cfg=types.SimpleNamespace(
            expert_macro_anchor_walk_std=0.01,
            expert_macro_anchor_walk_max=0.02,
            expert_macro_anchor_jitter=0.005,
            expert_macro_anchor_jump_prob=1.0,
            expert_macro_anchor_jump=0.03,
        ),
    )
    env_ids = torch.arange(6)
    base = torch.zeros(6, 3)
    torch.manual_seed(0)
    out = plane._jittered_live_anchor(base, env_ids)
    assert out.shape == (6, 3)
    assert torch.all(out[:, 2] == 0.0)
    assert out[:, :2].abs().max() <= 0.02 + 0.005 + 0.03 + 1e-6
    assert out[:, :2].abs().sum() > 0.0
    for _ in range(50):
        plane._jittered_live_anchor(base, env_ids)
    assert plane._anchor_walk_state.abs().max() <= 0.02 + 1e-6
    plane.reset_anchor_jitter(torch.tensor([0, 1]))
    assert plane._anchor_walk_state[:2].abs().sum() == 0.0
    plane._env.cfg.expert_macro_anchor_walk_std = 0.0
    plane._env.cfg.expert_macro_anchor_jitter = 0.0
    plane._env.cfg.expert_macro_anchor_jump_prob = 0.0
    assert plane._jittered_live_anchor(base, env_ids) is base
