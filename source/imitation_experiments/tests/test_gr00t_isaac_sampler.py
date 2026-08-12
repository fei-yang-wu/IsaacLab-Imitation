"""Slot consumption, FSQ snapping, and reset semantics of the Isaac sampler."""

from __future__ import annotations

import torch

from imitation_experiments.planner.gr00t_isaac_sampler import Gr00tSkillCommandSampler


class _Sampler(Gr00tSkillCommandSampler):
    """Bypass head loading; drive `gr00t_z` with a scripted prediction."""

    def __init__(self, num_envs: int, horizon: int, dim: int, consumption: str,
                 fsq_half: torch.Tensor | None = None):
        device = torch.device("cpu")
        self._gr00t_device = device
        self._gr00t_consumption = consumption
        self._gr00t_horizon = horizon
        self._gr00t_slots = horizon
        self._gr00t_action_dim = dim
        self._gr00t_fsq_half = fsq_half
        self._gr00t_cache = torch.zeros((num_envs, horizon, dim))
        self._gr00t_cursor = torch.full((num_envs,), horizon, dtype=torch.long)
        self._gr00t_calls = 0
        self._gr00t_latency_ms = []
        self.predictions: list[int] = []

    def _gr00t_predict(self, planner_state: torch.Tensor) -> torch.Tensor:
        rows = int(planner_state.shape[0])
        self.predictions.append(rows)
        base = float(self._gr00t_calls) * 100.0
        self._gr00t_calls += 1
        out = torch.empty(rows, self._gr00t_horizon, self._gr00t_action_dim)
        for slot in range(self._gr00t_horizon):
            out[:, slot] = base + slot
        return out


def _state(rows: int) -> torch.Tensor:
    return torch.zeros(rows, 930)


def test_open_loop_consumes_cached_slots_before_re_planning():
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="open_loop")
    env_ids = torch.tensor([0, 1])
    seen = [float(sampler.gr00t_z(_state(2), env_ids)[0, 0]) for _ in range(6)]
    # One head call per 3 publications; slots consumed in order.
    assert seen == [0.0, 1.0, 2.0, 100.0, 101.0, 102.0]
    assert sampler.predictions == [2, 2]


def test_fresh_mode_calls_the_head_every_publication():
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="fresh")
    env_ids = torch.tensor([0, 1])
    seen = [float(sampler.gr00t_z(_state(2), env_ids)[0, 0]) for _ in range(3)]
    assert seen == [0.0, 100.0, 200.0]
    assert sampler.predictions == [2, 2, 2]


def test_only_expired_environments_are_re_planned():
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="open_loop")
    sampler.gr00t_z(_state(2), torch.tensor([0, 1]))
    # Env 0 renews twice more (cursor expires); env 1 stays cached.
    sampler.gr00t_z(_state(1), torch.tensor([0]))
    sampler.gr00t_z(_state(1), torch.tensor([0]))
    sampler.predictions.clear()
    value = sampler.gr00t_z(_state(1), torch.tensor([0]))
    assert sampler.predictions == [1]  # only env 0 re-planned
    assert float(value[0, 0]) == 100.0


def test_reset_forces_replanning():
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="open_loop")
    sampler.gr00t_z(_state(2), torch.tensor([0, 1]))
    sampler.gr00t_reset(torch.tensor([1]))
    sampler.predictions.clear()
    sampler.gr00t_z(_state(2), torch.tensor([0, 1]))
    assert sampler.predictions == [1]  # env 1 only


def test_fsq_snapping_lands_on_the_lattice():
    half = torch.full((4,), 16.0)
    sampler = _Sampler(num_envs=1, horizon=1, dim=4, consumption="fresh", fsq_half=half)
    z = sampler.gr00t_z(_state(1), torch.tensor([0]))
    lattice = torch.round(z * half) / half
    assert torch.equal(z, lattice)
    assert bool((z <= 15.0 / 16.0).all()) and bool((z >= -1.0).all())


class _ChunkTerm:
    """Minimal ChunkActorCommand stand-in recording published packets."""

    def __init__(self, horizon: int):
        self.window_steps = horizon
        self.packets: list[dict] = []

    def publish(self, env_ids, payload):
        self.packets.append({k: v.clone() for k, v in payload.items()})


def test_chunk_publisher_splits_frames_term_major():
    from imitation_experiments.planner.gr00t_chunk_publisher import (
        ROOT_QPOS_COMPONENT_WIDTHS,
        Gr00tChunkPublisher,
    )

    horizon, width = 30, 38
    publisher = Gr00tChunkPublisher.__new__(Gr00tChunkPublisher)
    term = _ChunkTerm(horizon)
    publisher._chunk_term = term
    publisher._components = ROOT_QPOS_COMPONENT_WIDTHS
    publisher._causal_history_steps = 9
    publisher._gr00t_device = torch.device("cpu")
    publisher._gr00t_horizon = horizon
    publisher._gr00t_action_dim = width
    publisher.publications = 0

    # Frame f, dim d carries the value f*100 + d, so the split is checkable.
    prediction = torch.arange(width, dtype=torch.float32).reshape(1, 1, width)
    prediction = prediction + (
        torch.arange(horizon, dtype=torch.float32).reshape(1, horizon, 1) * 100.0
    )
    publisher._gr00t_predict = lambda state: prediction
    publisher._causal_observation_fn = lambda env_ids, history_steps: _Batch()

    publisher.publish(torch.tensor([0]))
    packet = term.packets[0]
    assert set(packet) == {name for name, _ in ROOT_QPOS_COMPONENT_WIDTHS}
    qpos = packet["expert_motion_qpos"].reshape(1, horizon, 29)
    assert float(qpos[0, 0, 0]) == 0.0
    assert float(qpos[0, 1, 0]) == 100.0  # frame 1, dim 0
    pos = packet["expert_anchor_pos_b"].reshape(1, horizon, 3)
    assert float(pos[0, 0, 0]) == 29.0  # frame 0, first anchor-pos dim
    ori = packet["expert_anchor_ori_b"].reshape(1, horizon, 6)
    assert float(ori[0, 0, 0]) == 32.0
    assert publisher.publications == 1


class _Batch:
    def get(self, key):
        return torch.zeros(1, 10, 93)


def test_packet_planner_returns_term_major_full_horizon():
    from imitation_experiments.planner.gr00t_chunk_publisher import (
        Gr00tChunkPublisher,
        Gr00tPacketPlanner,
    )

    horizon, width = 30, 38
    publisher = Gr00tChunkPublisher.__new__(Gr00tChunkPublisher)
    publisher._gr00t_device = torch.device("cpu")
    publisher._gr00t_horizon = horizon
    publisher._gr00t_action_dim = width
    prediction = torch.arange(width, dtype=torch.float32).reshape(1, 1, width) + (
        torch.arange(horizon, dtype=torch.float32).reshape(1, horizon, 1) * 100.0
    )
    publisher._gr00t_predict = lambda state: prediction

    planner = Gr00tPacketPlanner(publisher)
    packet = planner(torch.zeros(1, 930), num_inference_steps=4)
    assert packet.shape == (1, horizon * width)
    # Term-major: the whole qpos block precedes the anchor-pos block.
    qpos = packet[0, : horizon * 29].reshape(horizon, 29)
    assert float(qpos[0, 0]) == 0.0 and float(qpos[1, 0]) == 100.0
    pos = packet[0, horizon * 29 : horizon * 32].reshape(horizon, 3)
    assert float(pos[0, 0]) == 29.0
    ori = packet[0, horizon * 32 :].reshape(horizon, 6)
    assert float(ori[0, 0]) == 32.0
