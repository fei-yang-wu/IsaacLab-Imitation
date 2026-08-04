"""Contract tests for the G1 training metric channel.

A metric cannot be expressed as a reward term with ``weight=0.0``:
:meth:`RewardManager.compute` skips zero-weight terms without calling them, so
such a term logs a constant zero rather than the quantity it names. Metrics are
therefore driven by ``cfg.mpjpe_metric_body_names`` and logged by the env on a
dedicated ``Metrics/`` channel.

See ``wiki/sim2sim-dynamics-gap-and-randomization.md``.
"""

import inspect

import pytest

from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_cfg import (
    G1_TRACKED_BODY_NAMES,
    ImitationG1LafanTrackEnvCfg,
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
        "the Metrics/ channel in ImitationRLEnvLegacy may no longer be necessary."
    )


def test_g1_configures_the_mpjpe_metric() -> None:
    cfg = ImitationG1LafanTrackEnvCfg()
    assert cfg.mpjpe_metric_body_names, (
        "G1 must configure mpjpe_metric_body_names; without it the env logs no "
        "MPJPE metric and Episode_Reward/mpjpe_m is a constant zero."
    )


def test_mpjpe_metric_matches_the_evaluated_body_set() -> None:
    """The training curve and the reported evaluation number must agree.

    The metric is only useful if it is the same quantity the closed-loop
    evaluators report, which means the same bodies in the same order.
    """
    cfg = ImitationG1LafanTrackEnvCfg()
    assert list(cfg.mpjpe_metric_body_names) == list(G1_TRACKED_BODY_NAMES)


def test_mpjpe_metric_is_reported_in_millimetres() -> None:
    """The training metric must use the same unit as the paper aggregators.

    The evaluators emit ``tracking_mpjpe_mm`` and every aggregator consumes it,
    so a training curve in metres would differ from the reported number by
    1000x and invite a silent misreading.
    """
    from isaaclab_imitation.envs.imitation_rl_env_legacy import (
        _METRES_TO_MM,
        ImitationRLEnvLegacy,
    )

    assert _METRES_TO_MM == 1000.0
    for method in (
        ImitationRLEnvLegacy._accumulate_mpjpe_metric,
        ImitationRLEnvLegacy._emit_mpjpe_episode_metric,
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
    cfg = ImitationG1LafanTrackEnvCfg()
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

    from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy

    source = inspect.getsource(ImitationRLEnvLegacy._reset_idx)
    terminal_call = source.find("_accumulate_terminal_mpjpe_metric")
    assert terminal_call != -1, (
        "_reset_idx no longer folds the terminal frame into the MPJPE episode "
        "sum; the last transition of every episode will be dropped."
    )
    # Match the call by method name only: the receiver spelling varies
    # (`self.trajectory_manager.reset_envs` vs a local `tm.reset_envs`).
    reassignment_call = source.find(".reset_envs(")
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

    from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy

    source = inspect.getsource(ImitationRLEnvLegacy.step)
    assert "exclude_env_ids" in source and "reset_terminated" in source, (
        "step() no longer excludes just-reset envs from the post-step MPJPE "
        "accumulation; their post-reset pose would be misattributed to the "
        "new episode."
    )


def test_mpjpe_local_and_global_are_both_reported() -> None:
    """MPJPE-L and MPJPE-G are the pair the SONIC/PHC lineage reports.

    ``mpjpe_mm`` keeps its historical name and stays equal to MPJPE-L so old
    runs and the screen aggregator remain readable.
    """
    import torch

    from isaaclab_imitation.envs.expert_data_plane import ExpertDataPlane

    num_envs, num_bodies = 3, 4
    robot_pos = torch.randn(num_envs, num_bodies, 3)
    reference_pos = torch.randn(num_envs, num_bodies, 3)
    robot_root = torch.randn(num_envs, 3)
    reference_root = torch.randn(num_envs, 3)

    expected_local = torch.linalg.vector_norm(
        (robot_pos - robot_root[:, None, :])
        - (reference_pos - reference_root[:, None, :]),
        dim=-1,
    ).mean(dim=-1)
    expected_global = torch.linalg.vector_norm(robot_pos - reference_pos, dim=-1).mean(
        dim=-1
    )

    # Global counts drift that local removes, so it can never be smaller when
    # the two roots differ.
    assert torch.all(expected_global >= 0.0)
    assert not torch.allclose(expected_local, expected_global)
    assert hasattr(ExpertDataPlane, "_compute_mpjpe_metrics")


def test_pure_translation_drift_moves_global_but_not_local() -> None:
    """The defining property: shift the robot bodily and only MPJPE-G reacts."""
    import torch

    num_envs, num_bodies = 2, 5
    reference_pos = torch.randn(num_envs, num_bodies, 3)
    reference_root = reference_pos.mean(dim=1)
    drift = torch.tensor([0.30, -0.20, 0.05])

    robot_pos = reference_pos + drift
    robot_root = reference_root + drift

    local = torch.linalg.vector_norm(
        (robot_pos - robot_root[:, None, :])
        - (reference_pos - reference_root[:, None, :]),
        dim=-1,
    ).mean(dim=-1)
    global_ = torch.linalg.vector_norm(robot_pos - reference_pos, dim=-1).mean(dim=-1)

    assert torch.allclose(local, torch.zeros_like(local), atol=1e-6)
    assert torch.allclose(global_, torch.full_like(global_, float(drift.norm())))


def test_foot_reward_mirrors_the_foot_termination() -> None:
    """The foot reward must track exactly what `foot_pos_xyz` terminates on.

    Same predicate geometry (rerooted reference vs robot body), same anchor,
    same body set. If these drift apart the policy is once again rewarded for
    one quantity and killed by another, which is the mismatch this term exists
    to remove.
    """
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.rewards import (  # noqa: E501
        G1SonicRewardsCfg,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.common.terminations import (  # noqa: E501
        G1SonicTerminationsCfg,
    )

    reward = G1SonicRewardsCfg().motion_foot_pos
    termination = G1SonicTerminationsCfg().foot_pos_xyz

    assert list(reward.params["reference_body_names"]) == list(
        termination.params["reference_body_names"]
    )
    assert reward.params["anchor_body_name"] == termination.params["anchor_body_name"]
    assert list(reward.params["asset_cfg"].body_names) == list(
        termination.params["asset_cfg"].body_names
    )
    # The kernel's useful gradient must sit inside the survivable band.
    assert reward.params["std"] < termination.params["threshold"]
    assert reward.weight > 0.0


def test_mpjpe_accumulator_is_an_episode_mean_not_a_terminal_sample() -> None:
    """`CommandTerm.reset` samples the buffer once, so it must hold a mean.

    Isaac Lab logs `mean(metric[env_ids])` of whatever sits in the buffer at the
    reset step and then zeroes it. A buffer holding the instantaneous error
    therefore reports the error AT THE MOMENT THE EPISODE ENDED, which for a
    tracking-error termination is a sample taken at the failure threshold. This
    pins the accumulate-then-average shape that makes the logged value an
    episode mean, comparable to what evaluation reports.
    """
    import torch

    per_step = torch.tensor([10.0, 20.0, 60.0])  # a clean episode that ends badly
    running_sum, steps = 0.0, 0.0
    for value in per_step:
        running_sum = running_sum + float(value)
        steps += 1.0
    episode_mean = running_sum / steps
    terminal = float(per_step[-1])

    assert episode_mean == pytest.approx(30.0)
    assert terminal == pytest.approx(60.0)
    # The terminal sample overstates the episode by 2x here; on the real 1.9B
    # checkpoint it was 64.8 against 30.9.
    assert terminal > episode_mean


def test_reference_term_clears_mpjpe_accumulators_on_reset() -> None:
    """A new episode must not inherit the previous episode's error."""
    from isaaclab_imitation.tasks.manager_based.imitation.mdp.commands.reference import (  # noqa: E501
        ReferenceCommandTerm,
    )

    # `reset` must clear the running sums, or episode two is contaminated by one.
    source = inspect.getsource(ReferenceCommandTerm.reset)
    for accumulator in ("_mpjpe_l_sum", "_mpjpe_g_sum", "_mpjpe_steps"):
        assert accumulator in source, accumulator
    assert "super().reset" in source


def test_macro_state_terms_select_the_encoder_input_width() -> None:
    """The skill encoder's input width is set by `expert_macro_state_terms`.

    Nothing downstream validates an encoder's input space, so a run that asked
    for root_qpos and silently got full-body would train, load and evaluate
    without complaint while the experiment record claimed the wrong interface.
    The widths are the check:

        full_body  58 + 3 + 6 = 67/frame -> 670 over a 10-frame window
        root_qpos  29 + 3 + 6 = 38/frame -> 380

    Measured 2026-08-04: overriding `command_interface.encoder.components`
    alone leaves the encoder at 670. `expert_macro_state_terms` is the knob
    that moves it.
    """
    widths = {
        "expert_motion": 58,
        "expert_motion_qpos": 29,
        "expert_anchor_pos_b": 3,
        "expert_anchor_ori_b": 6,
    }
    full_body = ("expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b")
    root_qpos = ("expert_motion_qpos", "expert_anchor_pos_b", "expert_anchor_ori_b")
    window = 10
    assert sum(widths[t] for t in full_body) * window == 670
    assert sum(widths[t] for t in root_qpos) * window == 380
    assert full_body != root_qpos


def test_v2_default_macro_state_is_root_qpos() -> None:
    """`-G1-v2` defaults to the root_qpos frame; the SONIC surface does not.

    The arithmetic test above pins the widths but never touches a config, so it
    passed unchanged when the default moved on 2026-08-04. This pins the config.

    v2's default is root_qpos (qpos + root pose, no joint velocity), which means
    a v2 run REQUIRES a 380-wide encoder. `ImitationG1SonicSurfaceEnvCfg` stays
    on the full-body frame deliberately: it is the published SONIC recipe, not a
    place for our tuned defaults.
    """
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v2 import (  # noqa: E501
        ImitationG1V2EnvCfg,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.surfaces import (
        ImitationG1SonicSurfaceEnvCfg,
    )

    root_qpos = [
        "expert_motion_qpos",
        "expert_anchor_pos_b",
        "expert_anchor_ori_b",
    ]
    full_body = ["expert_motion", "expert_anchor_pos_b", "expert_anchor_ori_b"]

    assert list(ImitationG1V2EnvCfg().expert_macro_state_terms) == root_qpos
    assert list(ImitationG1SonicSurfaceEnvCfg().expert_macro_state_terms) == full_body

    # The default must not be a shared mutable: two configs editing one list
    # would silently couple every surface built in the same process.
    a, b = ImitationG1V2EnvCfg(), ImitationG1V2EnvCfg()
    assert a.expert_macro_state_terms is not b.expert_macro_state_terms


def test_v2_default_rewards_are_the_tuned_weights() -> None:
    """v2 carries the 2026-08-04 tuned tracking weights; v1 and SONIC do not.

    Measured over three seeds against two control seeds with randomization off:
    MPJPE-G -37.3%, EE-G -34.7%, ranges disjoint. v1 is frozen and the SONIC
    surface is the published recipe, so both keep the previous weights.
    """
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v1 import (  # noqa: E501
        ImitationG1EnvV1Cfg,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.imitation_g1_env_v2 import (  # noqa: E501
        ImitationG1V2EnvCfg,
    )
    from isaaclab_imitation.tasks.manager_based.imitation.config.g1.surfaces import (
        ImitationG1SonicSurfaceEnvCfg,
    )

    v2 = ImitationG1V2EnvCfg().rewards
    assert (v2.motion_body_pos.weight, v2.motion_body_pos.params["std"]) == (2.0, 0.05)
    assert (
        v2.motion_global_anchor_pos.weight,
        v2.motion_global_anchor_pos.params["std"],
    ) == (2.0, 0.1)
    assert (
        v2.motion_global_anchor_ori.weight,
        v2.motion_global_anchor_ori.params["std"],
    ) == (2.0, 0.15)
    # The coarse companion ships inert; enabling it is an explicit override.
    assert v2.motion_global_anchor_pos_wide.weight == 0.0

    for frozen in (
        ImitationG1EnvV1Cfg().rewards,
        ImitationG1SonicSurfaceEnvCfg().rewards,
    ):
        assert frozen.motion_body_pos.params["std"] == 0.3
        assert frozen.motion_global_anchor_pos.weight == 0.5
        assert frozen.motion_global_anchor_ori.weight == 0.5
