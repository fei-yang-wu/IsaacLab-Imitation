"""Contract tests for the G1 training metric channel.

A metric cannot be expressed as a reward term with ``weight=0.0``:
:meth:`RewardManager.compute` skips zero-weight terms without calling them, so
such a term logs a constant zero rather than the quantity it names. Metrics are
therefore driven by ``cfg.mpjpe_metric_body_names`` and logged by the env on a
dedicated ``Metrics/`` channel.

See ``wiki/sim2sim-dynamics-gap-and-randomization.md``.
"""

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_TRACKED_BODY_NAMES,
    ImitationG1EnvCfg,
)


def test_reward_manager_skips_zero_weight_terms() -> None:
    """Pin the upstream behaviour that makes a zero-weight metric term inert.

    If Isaac Lab ever starts evaluating zero-weight terms, the workaround in
    the env becomes redundant and this test is the reminder to revisit it.
    """
    import inspect

    from isaaclab.managers.reward_manager import RewardManager

    source = inspect.getsource(RewardManager.compute)
    assert "weight == 0.0" in source and "continue" in source, (
        "RewardManager.compute no longer short-circuits zero-weight terms; "
        "the Metrics/ channel in ImitationRLEnv may no longer be necessary."
    )


def test_g1_configures_the_mpjpe_metric() -> None:
    cfg = ImitationG1EnvCfg()
    assert cfg.mpjpe_metric_body_names, (
        "G1 must configure mpjpe_metric_body_names; without it the env logs no "
        "MPJPE metric and Episode_Reward/mpjpe_m is a constant zero."
    )


def test_mpjpe_metric_matches_the_evaluated_body_set() -> None:
    """The training curve and the reported evaluation number must agree.

    The metric is only useful if it is the same quantity the closed-loop
    evaluators report, which means the same bodies in the same order.
    """
    cfg = ImitationG1EnvCfg()
    assert list(cfg.mpjpe_metric_body_names) == list(G1_TRACKED_BODY_NAMES)


def test_mpjpe_metric_is_reported_in_millimetres() -> None:
    """The training metric must use the same unit as the paper aggregators.

    The evaluators emit ``tracking_mpjpe_mm`` and every aggregator consumes it,
    so a training curve in metres would differ from the reported number by
    1000x and invite a silent misreading.
    """
    from isaaclab_imitation.envs.imitation_rl_env import (
        _METRES_TO_MM,
        ImitationRLEnv,
    )

    assert _METRES_TO_MM == 1000.0
    for method in (
        ImitationRLEnv._accumulate_mpjpe_metric,
        ImitationRLEnv._emit_mpjpe_episode_metric,
    ):
        names = method.__code__.co_consts
        keys = [c for c in names if isinstance(c, str) and c.startswith("Metrics/")]
        assert keys, f"{method.__name__} emits no Metrics/ key"
        for key in keys:
            assert key.startswith("Metrics/mpjpe_mm"), (
                f"{method.__name__} emits {key!r}; MPJPE must be logged in "
                "millimetres to match tracking_mpjpe_mm"
            )


def test_mpjpe_metric_bodies_exist_in_the_reference() -> None:
    """The metric compares robot and reference bodies of the same name."""
    cfg = ImitationG1EnvCfg()
    missing = [
        name
        for name in cfg.mpjpe_metric_body_names
        if name not in set(cfg.reference_body_names)
    ]
    assert not missing, f"MPJPE metric bodies absent from the reference: {missing}"


def test_terminal_mpjpe_is_folded_in_before_trajectory_reassignment() -> None:
    """The terminal (pre-reset) frame must count toward the ending episode.

    ``_reset_idx`` reassigns the tracked trajectory (and later overwrites the
    physical state) for every env it resets. Once that happens, neither the
    robot's terminal pose nor the reference it was being scored against are
    recoverable, so ``_accumulate_terminal_mpjpe_metric`` must run first --
    otherwise the last transition of every episode is silently dropped from
    its MPJPE average, and instead misattributed to the *next* episode once
    ``_accumulate_mpjpe_metric`` runs again in ``step()``.
    """
    import inspect

    from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv

    source = inspect.getsource(ImitationRLEnv._reset_idx)
    terminal_call = source.find("_accumulate_terminal_mpjpe_metric")
    assert terminal_call != -1, (
        "_reset_idx no longer folds the terminal frame into the MPJPE episode "
        "sum; the last transition of every episode will be dropped."
    )
    reassignment_call = source.find("trajectory_manager.reset_envs")
    assert reassignment_call != -1, "trajectory reassignment call not found"
    assert terminal_call < reassignment_call, (
        "_accumulate_terminal_mpjpe_metric must run before the trajectory is "
        "reassigned, while the robot's terminal state and the reference it "
        "was scored against still belong to the ending episode."
    )


def test_step_excludes_just_reset_envs_from_the_new_episode_sum() -> None:
    """The post-step accumulation must not double-count into a fresh episode.

    By the time ``step()`` calls ``_accumulate_mpjpe_metric`` after
    ``super().step()``, any env that reset this step already had its
    terminal frame folded into the ending episode by ``_reset_idx``. The
    state visible at this point for those envs is the fresh post-reset pose,
    not something the policy produced, so it must be excluded here or it
    would be misattributed as the new episode's first sample.
    """
    import inspect

    from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv

    source = inspect.getsource(ImitationRLEnv.step)
    assert "exclude_env_ids" in source and "reset_terminated" in source, (
        "step() no longer excludes just-reset envs from the post-step MPJPE "
        "accumulation; their post-reset pose would be misattributed to the "
        "new episode."
    )
