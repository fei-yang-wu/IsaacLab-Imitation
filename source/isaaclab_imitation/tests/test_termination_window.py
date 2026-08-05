"""Contract for windowed (persistent-violation) terminations.

Two things are being pinned here:

1. The counter semantics. ``min_steps=1`` must be bit-identical to the
   instantaneous terms it replaces, a window must require *consecutive*
   violations, and ``diagnostic_only`` must never terminate while still
   measuring run lengths.
2. That the windowed config changes only episode boundaries. Its thresholds
   must stay equal to the strict SONIC protocol's, because a window is meant
   to be orthogonal to the error geometry, and the default v2 surface must
   keep using the instantaneous protocol so recorded gate numbers stay valid.
"""

from __future__ import annotations

import torch
from isaaclab.managers import TerminationTermCfg

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.terminations import (  # noqa: E501
    SONIC_WINDOW_TERM_NAMES,
    G1SonicTerminationsCfg,
    G1SonicTerminationWindowProbeCfg,
    G1TerminationsCfg,
    G1SonicWindowedTerminationsCfg,
    apply_termination_window,
)
from isaaclab_imitation.tasks.manager_based.imitation.mdp.terminations import (
    PersistentViolation,
)

NUM_ENVS = 3

_WINDOWED_FUNCS = {
    getattr(G1SonicWindowedTerminationsCfg(), name).func
    for name in SONIC_WINDOW_TERM_NAMES
}


class _StubEnv:
    """The only environment surface ``PersistentViolation`` touches."""

    def __init__(self, num_envs: int = NUM_ENVS):
        self.num_envs = num_envs
        self.device = "cpu"
        self.extras: dict = {"log": {}}


class _ScriptedViolation(PersistentViolation):
    """Drive the counter from an injected mask instead of robot state."""

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        env,
        violated: torch.Tensor,
        min_steps: int = 1,
        diagnostic_only: bool = False,
    ) -> torch.Tensor:
        return self._resolve(violated, min_steps, diagnostic_only)


def _term(min_steps: int = 1, diagnostic_only: bool = False):
    env = _StubEnv()
    cfg = TerminationTermCfg(
        func=_ScriptedViolation,  # ty: ignore[invalid-argument-type]
        params={"min_steps": min_steps, "diagnostic_only": diagnostic_only},
    )
    return _ScriptedViolation(cfg=cfg, env=env), env  # ty: ignore[invalid-argument-type]


def _mask(*values: bool) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def test_min_steps_one_matches_instantaneous_termination():
    term, _ = _term(min_steps=1)
    for step_mask in (_mask(True, False, False), _mask(False, True, True)):
        assert torch.equal(term(None, violated=step_mask, min_steps=1), step_mask)


def test_window_requires_consecutive_violations():
    term, _ = _term(min_steps=3)
    always = _mask(True, True, True)
    assert not term(None, violated=always, min_steps=3).any()
    assert not term(None, violated=always, min_steps=3).any()
    assert term(None, violated=always, min_steps=3).all()


def test_single_good_step_resets_the_counter():
    term, _ = _term(min_steps=3)
    violated = _mask(True, True, True)
    recovered = _mask(True, False, True)
    term(None, violated=violated, min_steps=3)
    term(None, violated=recovered, min_steps=3)  # env 1's run ends at length 1
    # Envs 0 and 2 reach three consecutive violations; env 1 is only on its
    # first, so the same raw error pattern is fatal for two envs and not the
    # third purely because of the interruption.
    assert torch.equal(
        term(None, violated=violated, min_steps=3), _mask(True, False, True)
    )
    # The fired envs restarted from zero, so nothing fires again immediately.
    assert not term(None, violated=violated, min_steps=3).any()


def test_min_steps_is_read_per_call_so_a_curriculum_can_anneal_it():
    term, _ = _term(min_steps=5)
    always = _mask(True, True, True)
    assert not term(None, violated=always, min_steps=5).any()
    # Same term instance, tightened window: the run so far still counts.
    assert term(None, violated=always, min_steps=2).all()


def test_diagnostic_mode_never_terminates_but_records_run_lengths():
    term, env = _term(diagnostic_only=True)
    sequence = [
        _mask(True, True, False),
        _mask(True, False, False),
        _mask(False, False, False),  # env 0 recovers at 2, env 1 recovered at 1
    ]
    for step_mask in sequence:
        assert not term(None, violated=step_mask, diagnostic_only=True).any()
    term.reset(torch.arange(NUM_ENVS))

    log = env.extras["log"]
    assert log["Termination_Window/_ScriptedViolation/runs_total"] == 2.0
    assert log["Termination_Window/_ScriptedViolation/runs_fatal"] == 0.0
    assert log["Termination_Window/_ScriptedViolation/runs_censored"] == 0.0
    assert log["Termination_Window/_ScriptedViolation/recovered_mean_steps"] == 1.5
    # A window of 2 would have saved only the length-1 run; a window of 3 both.
    assert log["Termination_Window/_ScriptedViolation/recovered_below_2_frac"] == 0.5
    assert log["Termination_Window/_ScriptedViolation/recovered_below_3_frac"] == 1.0


def test_fatal_run_is_not_double_counted_as_censored():
    term, env = _term(min_steps=2)
    always = _mask(True, True, True)
    term(None, violated=always, min_steps=2)
    assert term(None, violated=always, min_steps=2).all()
    term.reset(torch.arange(NUM_ENVS))

    log = env.extras["log"]
    assert log["Termination_Window/_ScriptedViolation/runs_fatal"] == 3.0
    assert log["Termination_Window/_ScriptedViolation/runs_censored"] == 0.0


def test_run_in_progress_at_episode_end_is_counted_as_censored():
    term, env = _term(min_steps=10)
    term(None, violated=_mask(True, True, False), min_steps=10)
    term.reset(torch.arange(NUM_ENVS))

    log = env.extras["log"]
    assert log["Termination_Window/_ScriptedViolation/runs_censored"] == 2.0
    assert log["Termination_Window/_ScriptedViolation/runs_total"] == 2.0


def test_reset_clears_only_the_given_environments():
    term, _ = _term(min_steps=2)
    always = _mask(True, True, True)
    term(None, violated=always, min_steps=2)
    term.reset(torch.tensor([1]))
    # Env 1 restarted its run; envs 0 and 2 are still one step from firing.
    assert torch.equal(
        term(None, violated=always, min_steps=2), _mask(True, False, True)
    )


def test_windowed_cfg_changes_only_episode_boundaries():
    # Construct the windowed config first: it edits terms in place, so this
    # also catches the config classes sharing one mutable default between them.
    windowed = G1SonicWindowedTerminationsCfg()
    strict = G1SonicTerminationsCfg()
    for term_name in SONIC_WINDOW_TERM_NAMES:
        strict_params = dict(getattr(strict, term_name).params)
        windowed_params = dict(getattr(windowed, term_name).params)
        assert windowed_params.pop("min_steps") > 1
        assert windowed_params.pop("diagnostic_only") is False
        assert windowed_params == strict_params, term_name
        assert "min_steps" not in strict_params
        assert getattr(strict, term_name).func is not getattr(windowed, term_name).func
    # A window must never be attached to the fall condition.
    assert windowed.base_too_low is None


def test_window_helper_is_idempotent():
    cfg = G1SonicTerminationsCfg()
    apply_termination_window(cfg, min_steps=4)
    once = {name: getattr(cfg, name).func for name in SONIC_WINDOW_TERM_NAMES}
    apply_termination_window(cfg, min_steps=7)
    for term_name in SONIC_WINDOW_TERM_NAMES:
        term = getattr(cfg, term_name)
        assert term.func is once[term_name]
        assert term.params["min_steps"] == 7


def test_window_helper_refuses_predicates_it_cannot_wrap():
    import pytest

    # The v0 surface uses the z-only predicates, which have no windowed form;
    # silently leaving them instantaneous would be a half-applied protocol.
    with pytest.raises(ValueError, match="no windowed equivalent"):
        apply_termination_window(G1TerminationsCfg())  # ty: ignore[invalid-argument-type]


def test_probe_cfg_disables_every_tracking_termination():
    probe = G1SonicTerminationWindowProbeCfg()
    for term_name in SONIC_WINDOW_TERM_NAMES:
        assert getattr(probe, term_name).params["diagnostic_only"] is True
        assert getattr(probe, term_name).func in _WINDOWED_FUNCS
    assert probe.base_too_low is None
    # The horizon must still end the episode, or nothing ever resets.
    assert probe.time_out is not None
    assert probe.reference_finished is not None


def test_manager_accepts_every_windowed_term():
    """Run IsaacLab's own term-signature gate, which otherwise fires only at
    environment construction: each class term's ``__call__`` must declare every
    configured parameter as a defaulted keyword."""
    from isaaclab.managers.manager_base import ManagerBase

    class _SimStub:
        @staticmethod
        def is_playing() -> bool:
            # Keep the check static: the runtime half needs a live scene.
            return False

    class _EnvStub:
        sim = _SimStub()

    stub = _EnvStub()
    for cfg in (G1SonicWindowedTerminationsCfg(), G1SonicTerminationWindowProbeCfg()):
        for term_name in SONIC_WINDOW_TERM_NAMES:
            ManagerBase._resolve_common_term_cfg(
                type("_S", (), {"_env": stub})(),  # ty: ignore[invalid-argument-type]
                term_name,
                getattr(cfg, term_name),
                min_argc=1,
            )


def test_default_v2_surface_keeps_the_instantaneous_protocol():
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1 import (
        imitation_g1_env_v2,
    )

    cfg = imitation_g1_env_v2.ImitationG1V2EnvCfg()
    assert type(cfg.terminations) is G1SonicTerminationsCfg


def test_train_cli_flags_select_the_protocol():
    """The launcher flags are the only way a run leaves the instantaneous
    protocol, so they must apply before the config is dumped and must refuse
    the combination that would silently measure nothing."""
    import argparse
    import importlib
    import sys
    from pathlib import Path

    import pytest

    scripts_rlopt = Path(__file__).resolve().parents[3] / "scripts" / "rlopt"
    sys.path.insert(0, str(scripts_rlopt))
    try:
        train_impl = importlib.import_module("train_impl")
    finally:
        sys.path.remove(str(scripts_rlopt))
    apply_args = train_impl._apply_termination_window_args

    class _EnvCfgStub:
        def __init__(self):
            self.terminations = G1SonicTerminationsCfg()

    def _args(**kwargs) -> argparse.Namespace:
        defaults = {"termination_window": None, "termination_window_probe": False}
        return argparse.Namespace(**{**defaults, **kwargs})

    # Unset flags must leave the recorded protocol byte-for-byte alone.
    untouched = _EnvCfgStub()
    apply_args(_args(), untouched)
    for term_name in SONIC_WINDOW_TERM_NAMES:
        assert "min_steps" not in getattr(untouched.terminations, term_name).params

    windowed = _EnvCfgStub()
    apply_args(_args(termination_window=4), windowed)
    for term_name in SONIC_WINDOW_TERM_NAMES:
        params = getattr(windowed.terminations, term_name).params
        assert params["min_steps"] == 4
        assert params["diagnostic_only"] is False

    probed = _EnvCfgStub()
    apply_args(_args(termination_window_probe=True), probed)
    for term_name in SONIC_WINDOW_TERM_NAMES:
        assert getattr(probed.terminations, term_name).params["diagnostic_only"] is True

    with pytest.raises(ValueError, match="cannot be combined"):
        apply_args(
            _args(termination_window=3, termination_window_probe=True), _EnvCfgStub()
        )
    with pytest.raises(ValueError, match="must be >= 1"):
        apply_args(_args(termination_window=0), _EnvCfgStub())


def test_window_can_be_scoped_to_a_single_term():
    """Scoping matters: foot_pos_xyz is ~2/3 of non-timeout terminations.

    A window on it alone targets the dominant failure while the height and
    orientation terms stay strict, so the horizontal bar the policy must return
    to is unchanged.
    """
    terminations = G1SonicTerminationsCfg()
    instantaneous = {
        name: getattr(terminations, name).func for name in SONIC_WINDOW_TERM_NAMES
    }

    apply_termination_window(terminations, min_steps=4, term_names=("foot_pos_xyz",))

    foot = getattr(terminations, "foot_pos_xyz")
    assert foot.func in _WINDOWED_FUNCS
    assert foot.params["min_steps"] == 4
    for name in SONIC_WINDOW_TERM_NAMES:
        if name == "foot_pos_xyz":
            continue
        term = getattr(terminations, name)
        assert term.func is instantaneous[name], name
        assert "min_steps" not in term.params, name


def test_scoped_window_keeps_the_thresholds_untouched():
    strict = G1SonicTerminationsCfg()
    windowed = G1SonicTerminationsCfg()
    apply_termination_window(windowed, min_steps=4, term_names=("foot_pos_xyz",))
    for name in SONIC_WINDOW_TERM_NAMES:
        before = getattr(strict, name).params.get("threshold")
        after = getattr(windowed, name).params.get("threshold")
        assert before == after, name


def test_foot_termination_allowance_defaults_to_off():
    """`down_threshold=None` must leave the strict single-threshold behaviour.

    The allowance is opt-in so that adding it cannot silently change any
    recorded gate number.
    """
    import inspect

    from isaaclab_imitation.tasks.manager_based.imitation.mdp.terminations import (
        bad_reference_body_pos_relative,
    )

    sig = inspect.signature(bad_reference_body_pos_relative)
    assert sig.parameters["down_threshold"].default is None
    assert sig.parameters["root_height_threshold"].default == 0.5
    # The default config must not enable it.
    term = G1SonicTerminationsCfg().foot_pos_xyz
    assert term.params.get("down_threshold") is None


def test_windowed_foot_term_forwards_the_allowance():
    """The windowed wrapper must not drop the new parameters.

    It forwards explicit kwargs rather than **kwargs, so a parameter added to
    the predicate and not to the wrapper would be silently ignored whenever a
    window is active.
    """
    import inspect

    from isaaclab_imitation.tasks.manager_based.imitation.mdp.terminations import (
        PersistentBadReferenceBodyPosRelative,
        bad_reference_body_pos_relative,
    )

    predicate = set(inspect.signature(bad_reference_body_pos_relative).parameters)
    wrapper = set(
        inspect.signature(PersistentBadReferenceBodyPosRelative.__call__).parameters
    )
    missing = predicate - wrapper - {"env"}
    assert not missing, f"windowed wrapper drops {sorted(missing)}"
