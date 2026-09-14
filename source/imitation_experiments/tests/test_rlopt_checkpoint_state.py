from types import SimpleNamespace

import pytest

from imitation_experiments.lowlevel.rlopt_checkpoint import attach_isaac_training_state


def test_checkpoint_restores_curriculum_clock_and_rejects_other_data(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("first dataset")
    calls = []
    env = SimpleNamespace(
        common_step_counter=2400,
        num_envs=64,
        cfg=SimpleNamespace(reference=SimpleNamespace(manifest=str(manifest))),
        spec=SimpleNamespace(id="mounted"),
        curriculum_manager=SimpleNamespace(
            compute=lambda env_ids: calls.append(env_ids)
        ),
    )
    agent = SimpleNamespace(
        _extra_model_state_dict=lambda: {"ppo_updates_completed": 2000},
        _load_extra_model_state_dict=lambda state: calls.append(
            state["ppo_updates_completed"]
        ),
    )
    attach_isaac_training_state(agent, env)
    checkpoint = agent._extra_model_state_dict()
    env.common_step_counter = 0
    agent._load_extra_model_state_dict(checkpoint)
    assert env.common_step_counter == 2400
    assert calls == [2000, None]
    manifest.write_text("another dataset")
    with pytest.raises(ValueError, match="Reference differs"):
        agent._load_extra_model_state_dict(checkpoint)
    # A final rollout can close the simulator before checkpoint recording.
    del env.num_envs
    del env.cfg
    env.common_step_counter = 2500
    final = agent._extra_model_state_dict()["isaaclab_training_state"]
    assert final["num_envs"] == 64
    assert final["common_step_counter"] == 2500
    assert (
        final["reference_manifest_sha256"]
        == checkpoint["isaaclab_training_state"]["reference_manifest_sha256"]
    )


def test_checkpoint_restores_stateful_term_before_curriculum_compute():
    term_state = {"stage": 2, "episodes": 17}
    observed = []
    term = SimpleNamespace(
        training_state_dict=lambda: dict(term_state),
        load_training_state_dict=lambda state: term_state.update(state),
    )
    manager = SimpleNamespace(
        _term_names=["assistance"],
        _term_cfgs=[SimpleNamespace(func=term)],
        compute=lambda env_ids: observed.append(dict(term_state)),
    )
    env = SimpleNamespace(
        common_step_counter=2400, num_envs=4, curriculum_manager=manager
    )
    agent = SimpleNamespace(
        _extra_model_state_dict=lambda: {},
        _load_extra_model_state_dict=lambda state: None,
    )
    attach_isaac_training_state(agent, env)
    checkpoint = agent._extra_model_state_dict()
    term_state.update(stage=0, episodes=0)
    agent._load_extra_model_state_dict(checkpoint)
    assert observed == [{"stage": 2, "episodes": 17}]
    checkpoint["isaaclab_training_state"]["curriculum_terms"] = {}
    with pytest.raises(ValueError, match="Stateful curriculum terms differ"):
        agent._load_extra_model_state_dict(checkpoint)
