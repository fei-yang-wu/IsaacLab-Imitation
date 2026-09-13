import dataclasses

import pytest

from imitation_experiments.provenance.config_keys import (
    assert_no_unknown_config_keys,
    unknown_config_keys,
)


@dataclasses.dataclass
class _Inner:
    entropy_coeff: float = 0.0


@dataclasses.dataclass
class _Agent:
    ppo: _Inner = dataclasses.field(default_factory=_Inner)
    seed: int = 0
    _runtime_binding: object = None


def test_declared_fields_pass():
    agent = _Agent()
    agent.ppo.entropy_coeff = 0.01
    agent._runtime_binding = object()
    assert unknown_config_keys(agent) == []
    assert_no_unknown_config_keys(agent)


def test_a_setattr_override_on_an_undeclared_key_is_reported_with_its_path():
    agent = _Agent()
    setattr(agent.ppo, "update_normalizers_after_rollout", False)
    setattr(agent, "typo_seed", 1)
    assert unknown_config_keys(agent) == [
        "agent.ppo.update_normalizers_after_rollout",
        "agent.typo_seed",
    ]
    with pytest.raises(ValueError, match="update_normalizers_after_rollout"):
        assert_no_unknown_config_keys(agent)


def test_non_dataclass_configs_are_ignored():
    assert unknown_config_keys({"a": 1}) == []
