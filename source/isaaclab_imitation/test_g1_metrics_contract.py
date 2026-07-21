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


def test_mpjpe_metric_bodies_exist_in_the_reference() -> None:
    """The metric compares robot and reference bodies of the same name."""
    cfg = ImitationG1EnvCfg()
    missing = [
        name
        for name in cfg.mpjpe_metric_body_names
        if name not in set(cfg.reference_body_names)
    ]
    assert not missing, f"MPJPE metric bodies absent from the reference: {missing}"
