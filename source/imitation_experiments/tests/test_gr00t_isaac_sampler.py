"""Slot consumption, FSQ snapping, and reset semantics of the Isaac sampler."""

from __future__ import annotations

import torch

from imitation_experiments.planner.gr00t_isaac_sampler import Gr00tSkillCommandSampler


class _Sampler(Gr00tSkillCommandSampler):
    """Bypass head loading; drive `gr00t_z` with a scripted prediction."""

    def __init__(
        self,
        num_envs: int,
        horizon: int,
        dim: int,
        consumption: str,
        fsq_half: torch.Tensor | None = None,
    ):
        device = torch.device("cpu")
        self._gr00t_device = device
        self._gr00t_consumption = consumption
        self._gr00t_horizon = horizon
        self._gr00t_slots = horizon
        self._gr00t_action_dim = dim
        self._gr00t_fsq_half = fsq_half
        self._gr00t_cache = torch.zeros((num_envs, horizon, dim))
        self._gr00t_cursor = torch.full((num_envs,), horizon, dtype=torch.long)
        self._gr00t_goal_index = torch.zeros(num_envs, dtype=torch.long)
        self._gr00t_goal_names = ["goal"] * num_envs
        self._gr00t_calls = 0
        self._gr00t_latency_ms = []
        self.predictions: list[int] = []

    def _gr00t_predict(self, planner_state, goal_index=None) -> torch.Tensor:
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


def test_inference_provenance_records_every_arm_distinguishing_knob():
    """The knobs that separate published arms must reach the summary.

    Ensembling, sample averaging, and ODE-step count each move the reported
    MPJPE without touching the checkpoint. When they lived only in the output
    directory name, no consumer could read an arm's configuration back off its
    artifact.
    """
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="open_loop")
    sampler._gr00t_ensemble = "exponential"
    sampler._gr00t_ensemble_decay = 0.5
    sampler._gr00t_inference_steps = 16
    sampler._gr00t_samples = 4

    record = sampler.gr00t_inference_provenance()

    assert record == {
        "temporal_ensemble": "exponential",
        "temporal_ensemble_decay": 0.5,
        "num_inference_timesteps": 16,
        "samples_per_publication": 4,
        "consume_slots": 3,
    }


def test_unused_ensemble_decay_is_not_reported_as_a_setting():
    sampler = _Sampler(num_envs=2, horizon=3, dim=4, consumption="open_loop")
    sampler._gr00t_ensemble = "none"
    sampler._gr00t_ensemble_decay = 0.5

    record = sampler.gr00t_inference_provenance()

    assert record["temporal_ensemble"] == "none"
    assert record["temporal_ensemble_decay"] is None


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
    publisher._gr00t_goal_index = torch.zeros(4, dtype=torch.long)
    publisher.publications = 0

    # Frame f, dim d carries the value f*100 + d, so the split is checkable.
    prediction = torch.arange(width, dtype=torch.float32).reshape(1, 1, width)
    prediction = prediction + (
        torch.arange(horizon, dtype=torch.float32).reshape(1, horizon, 1) * 100.0
    )
    publisher._gr00t_predict = lambda state, goal_index=None: prediction
    publisher._causal_observation_fn = lambda env_ids, history_steps: _Batch()

    publisher.publish(torch.tensor([0]))
    packet = term.packets[0]
    assert set(packet) == {name for name, _ in ROOT_QPOS_COMPONENT_WIDTHS}
    qpos = packet["joint_qpos"].reshape(1, horizon, 29)
    assert float(qpos[0, 0, 0]) == 0.0
    assert float(qpos[0, 1, 0]) == 100.0  # frame 1, dim 0
    pos = packet["root_pos"].reshape(1, horizon, 3)
    assert float(pos[0, 0, 0]) == 29.0  # frame 0, first anchor-pos dim
    ori = packet["root_ori"].reshape(1, horizon, 6)
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
    publisher._gr00t_goal_index = torch.zeros(4, dtype=torch.long)
    prediction = torch.arange(width, dtype=torch.float32).reshape(1, 1, width) + (
        torch.arange(horizon, dtype=torch.float32).reshape(1, horizon, 1) * 100.0
    )
    publisher._gr00t_predict = lambda state, goal_index=None: prediction

    planner = Gr00tPacketPlanner(publisher)
    planner.note_env_ids(torch.tensor([0]))
    packet = planner(torch.zeros(1, 930), num_inference_steps=4)
    assert packet.shape == (1, horizon * width)
    # Term-major: the whole qpos block precedes the anchor-pos block.
    qpos = packet[0, : horizon * 29].reshape(horizon, 29)
    assert float(qpos[0, 0]) == 0.0 and float(qpos[1, 0]) == 100.0
    pos = packet[0, horizon * 29 : horizon * 32].reshape(horizon, 3)
    assert float(pos[0, 0]) == 29.0
    ori = packet[0, horizon * 32 :].reshape(horizon, 6)
    assert float(ori[0, 0]) == 32.0


def _per_env_sampler(goals, num_envs, horizon=3, dim=4):
    s = _Sampler(num_envs=num_envs, horizon=horizon, dim=dim, consumption="fresh")
    s._gr00t_goal_names = list(goals)
    s._gr00t_goal_index = torch.tensor([0, 1, 2][: len(goals)], dtype=torch.long)
    s._gr00t_features = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5)
    s._gr00t_feature_mask = torch.ones(3, 2, dtype=torch.bool)
    return s


def test_per_env_goal_index_selects_that_envs_language():
    seen = {}

    class _S(_Sampler):
        def _gr00t_predict(self, planner_state, goal_index=None):
            seen["idx"] = None if goal_index is None else goal_index.tolist()
            return torch.zeros(
                planner_state.shape[0], self._gr00t_horizon, self._gr00t_action_dim
            )

    s = _S(num_envs=3, horizon=3, dim=4, consumption="fresh")
    s._gr00t_goal_index = torch.tensor([2, 0, 1], dtype=torch.long)
    s.gr00t_z(_state(2), torch.tensor([1, 2]))
    assert seen["idx"] == [0, 1], "each row must carry its own env's goal"


def test_goal_reference_mismatch_is_a_hard_error():
    s = _per_env_sampler(["walk", "greet", "stoop"], num_envs=3)
    s.gr00t_assert_goal_matches(torch.tensor([0, 1]), ["walk", "greet"])
    try:
        s.gr00t_assert_goal_matches(torch.tensor([0, 1]), ["walk", "stoop"])
    except ValueError as error:
        assert "goal/reference mismatch" in str(error)
    else:
        raise AssertionError("a reassigned trajectory must fail loudly")


def test_root_qpos_vocabularies_agree_on_widths_but_not_names():
    """The two routes name the same 38 values differently; keep them distinct.

    Mixing them fails only at run time (one validates against the chunk actor
    term, the other against the env's macro frame layout), so pin both here.
    """
    from imitation_experiments.planner.gr00t_chunk_publisher import (
        ROOT_QPOS_COMMAND_COMPONENTS,
        ROOT_QPOS_MACRO_TERMS,
    )

    command_widths = [w for _, w in ROOT_QPOS_COMMAND_COMPONENTS]
    macro_widths = [w for _, w in ROOT_QPOS_MACRO_TERMS]
    assert command_widths == macro_widths == [29, 3, 6]
    assert sum(command_widths) == 38
    command_names = [n for n, _ in ROOT_QPOS_COMMAND_COMPONENTS]
    macro_names = [n for n, _ in ROOT_QPOS_MACRO_TERMS]
    assert command_names == ["joint_qpos", "root_pos", "root_ori"]
    assert macro_names == [
        "expert_motion_qpos",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ]
    assert not set(command_names) & set(macro_names)
