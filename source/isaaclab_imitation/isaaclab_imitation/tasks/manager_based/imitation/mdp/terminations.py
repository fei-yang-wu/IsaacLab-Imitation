from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_imitation.envs import ImitationRLEnv

from ._compiled import (
    quat_error_squared,
    reroot_body_positions,
    rms_error,
    xy_error_norm,
)


def _select_last_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(-1, ids)


def _select_body_dim(values: torch.Tensor, ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(ids, slice):
        return values
    return values.index_select(1, ids)


def reference_joint_pos_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 0.75,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = env._get_joint_ids_tensor_fast(asset_cfg.joint_ids)
    joint_pos_actual = _select_last_dim(asset.data.joint_pos.torch, joint_ids)
    joint_pos_reference = _select_last_dim(
        env.current_expert_frame["joint_pos"], joint_ids
    )
    return rms_error(joint_pos_actual, joint_pos_reference) > threshold


def reference_root_position_xy_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_actual = asset.data.root_state_w.torch[:, :3]
    root_pos_reference_w = env._get_reference_root_state_w_fast()[0]
    return (
        xy_error_norm(root_pos_actual[:, :2], root_pos_reference_w[:, :2]) > threshold
    )


def reference_root_quat_deviation_too_much(
    env: ImitationRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_actual = asset.data.root_state_w.torch[:, 3:7]
    root_quat_reference_w = env._get_reference_root_state_w_fast()[1]
    angular_error = torch.sqrt(
        quat_error_squared(root_quat_actual, root_quat_reference_w)
    )
    return angular_error > threshold


def bad_anchor_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot_anchor_pos_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[0]
    ref_anchor_pos_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[0][
        :, 0, :
    ]
    return torch.abs(ref_anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(
    env: ImitationRLEnv,
    threshold: float = 0.8,
    anchor_body_name: str = "torso_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[1]
    ref_anchor_quat_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[1][
        :, 0, :
    ]
    reference_projected_gravity_b = quat_apply_inverse(
        ref_anchor_quat_w, asset.data.GRAVITY_VEC_W.torch
    )
    robot_projected_gravity_b = quat_apply_inverse(
        robot_anchor_quat_w, asset.data.GRAVITY_VEC_W.torch
    )
    return (
        reference_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
    ).abs() > threshold


def bad_reference_body_pos_z_only(
    env: ImitationRLEnv,
    threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_body_names: Sequence[str] = (),
) -> torch.Tensor:
    actual_pos_w = env._get_robot_body_pose_w_fast(asset_cfg.body_ids)[0]
    ref_pos_w = env._get_reference_body_pose_w_fast(reference_body_names)[0]
    return torch.any(
        torch.abs(ref_pos_w[..., 2] - actual_pos_w[..., 2]) > threshold, dim=1
    )


def _reference_root_height(env: ImitationRLEnv) -> torch.Tensor:
    root_pos = env.current_expert_frame.get("root_pos")
    if root_pos is None:
        return env._get_reference_body_pose_w_fast(("pelvis",))[0][:, 0, 2]
    return root_pos[:, 2]


def bad_anchor_pos_z_adaptive(
    env: ImitationRLEnv,
    threshold: float = 0.15,
    down_threshold: float = 0.75,
    root_height_threshold: float = 0.5,
    anchor_body_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SONIC anchor-height termination with a crouching-motion allowance."""
    robot_anchor_pos_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[0]
    ref_anchor_pos_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[0][
        :, 0, :
    ]
    height_error = torch.abs(ref_anchor_pos_w[:, 2] - robot_anchor_pos_w[:, 2])
    thresholds = torch.full_like(height_error, threshold)
    thresholds = torch.where(
        _reference_root_height(env) < root_height_threshold,
        torch.full_like(thresholds, down_threshold),
        thresholds,
    )
    return height_error > thresholds


def bad_anchor_ori_full(
    env: ImitationRLEnv,
    threshold: float = 0.2,
    anchor_body_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate on squared full-quaternion anchor error, matching SONIC."""
    robot_quat_w = env._get_robot_anchor_state_w_fast(anchor_body_name)[1]
    ref_quat_w = env._get_reference_body_pose_w_fast((anchor_body_name,))[1][:, 0, :]
    return quat_error_squared(ref_quat_w, robot_quat_w) > threshold


def bad_reference_body_pos_z_adaptive(
    env: ImitationRLEnv,
    threshold: float = 0.15,
    down_threshold: float = 0.75,
    root_height_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_body_names: Sequence[str] = (),
) -> torch.Tensor:
    """SONIC end-effector height termination with a crouching allowance."""
    actual_pos_w = env._get_robot_body_pose_w_fast(asset_cfg.body_ids)[0]
    ref_pos_w = env._get_reference_body_pose_w_fast(reference_body_names)[0]
    height_error = torch.abs(ref_pos_w[..., 2] - actual_pos_w[..., 2])
    thresholds = torch.full_like(height_error, threshold)
    low_reference = _reference_root_height(env) < root_height_threshold
    thresholds[low_reference] = down_threshold
    return torch.any(height_error > thresholds, dim=1)


def bad_reference_body_pos_relative(
    env: ImitationRLEnv,
    threshold: float = 0.2,
    anchor_body_name: str = "pelvis",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_body_names: Sequence[str] = (),
    down_threshold: float | None = None,
    root_height_threshold: float = 0.5,
    swing_threshold: float | None = None,
    swing_height_threshold: float = 0.15,
) -> torch.Tensor:
    """Terminate when a rerooted reference body is too far from the robot body.

    ``down_threshold`` is the crouching allowance the other two position
    terminations already carry: when the REFERENCE root is below
    ``root_height_threshold`` the bar relaxes to ``down_threshold``. Default
    ``None`` keeps the single strict threshold, so existing configs are
    unchanged.

    Why it matters here. This is the only termination in the config that
    constrains horizontal position, and the only one without the allowance --
    `bad_anchor_pos_z_adaptive` and `bad_reference_body_pos_z_adaptive` both
    have it. Measured on the 2026-08-03 checkpoint, every `fallAndGetUp` clip
    failed and every `dance2` clip survived the full horizon, with
    `foot_pos_xyz` the cause in 4 of 6 failures. A fall-and-recover reference
    spends its hard phase exactly where the root is low, which is the condition
    the allowance exists to detect -- so the strict 0.2 m horizontal bar was
    being applied precisely where the other terms had already decided it should
    not be.

    ``swing_threshold`` is the symmetric allowance for the opposite regime.
    Measured on the same checkpoint, the failing clips are the DYNAMIC ones --
    jumps, runs, sprints, fights and falls -- while every dance clip and 11 of
    12 walks survive. The low-root allowance covers the falls; the airborne
    cases have a HIGH root, so it cannot fire there. During a flight phase the
    foot's horizontal position is not correctable at that instant and
    self-corrects on landing, so terminating for it punishes what the policy
    cannot fix.

    Applied PER BODY, keyed on the reference foot's own height: one foot is
    usually airborne while the other is planted, and a per-environment test
    would relax the stance foot too. Both allowances default to ``None``.
    """
    robot_anchor_pos_w, robot_anchor_quat_w = env._get_robot_anchor_state_w_fast(
        anchor_body_name
    )
    ref_pos_w, _ = env._get_reference_body_pose_w_fast(reference_body_names)
    ref_anchor_pos_w, ref_anchor_quat_w = env._get_reference_body_pose_w_fast(
        (anchor_body_name,)
    )
    target_pos_w = reroot_body_positions(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        ref_pos_w,
        ref_anchor_pos_w[:, 0, :],
        ref_anchor_quat_w[:, 0, :],
    )
    actual_pos_w = env._get_robot_body_pose_w_fast(asset_cfg.body_ids)[0]
    error = torch.linalg.vector_norm(target_pos_w - actual_pos_w, dim=-1)
    if down_threshold is None and swing_threshold is None:
        return torch.any(error > threshold, dim=1)
    thresholds = torch.full_like(error, threshold)
    if down_threshold is not None:
        low_reference = _reference_root_height(env) < root_height_threshold
        thresholds[low_reference] = down_threshold
    if swing_threshold is not None:
        # PER BODY, not per environment: one foot is usually airborne while the
        # other is planted, so a whole-environment test would relax the stance
        # foot too. `ref_pos_w` is the reference body height above the ground
        # plane, so this asks "is the REFERENCE foot in flight right now".
        airborne = ref_pos_w[..., 2] > swing_height_threshold
        thresholds = torch.where(
            airborne, torch.full_like(thresholds, swing_threshold), thresholds
        )
    return torch.any(error > thresholds, dim=1)


def reference_trajectory_finished(env: ImitationRLEnv) -> torch.Tensor:
    return env.current_reference_is_final_frame()


class PersistentViolation(ManagerTermBase):
    """End the episode only after a predicate holds for ``min_steps`` steps.

    The SONIC release ships this shape (``_CummErrorMixin``: a consecutive
    violation counter that any in-threshold step resets), but none of the
    release termination compositions we mirror enable it -- ``tracking_base``,
    ``tracking_base_adaptive_strict_ori_foot_xyz``, and ``tracking_eval`` all
    use the instantaneous predicates. Subclasses here wrap our existing
    predicates unchanged, so a window is purely an opt-in change to where
    episode boundaries fall, never a change to the error geometry.

    ``min_steps`` and every predicate parameter are read from the per-step call
    arguments rather than cached at construction, so
    :func:`~...mdp.curriculums.anneal_termination_threshold_by_frames`, which
    writes ``term_cfg.params["threshold"]`` in place, keeps working.

    With ``diagnostic_only=True`` the term never terminates and only records
    run-length statistics. That is the shadow measurement: it answers "what
    fraction of violation onsets resolve on their own within k steps", which is
    the same as "what fraction of today's one-step terminations a window of
    length k would convert into a recovery". Measuring it requires *not*
    terminating, because an instantaneous term destroys the episode before the
    run length it would have had is observable.
    """

    _MAX_TRACKED_RUN = 32
    """Run lengths at or above this land in a single overflow bucket."""

    _RECOVERY_BUCKETS = (2, 3, 5, 10)
    """Window lengths reported as "would have recovered" fractions."""

    def __init__(self, cfg: TerminationTermCfg, env: ImitationRLEnv):
        super().__init__(cfg, env)
        device = env.device
        self._run_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        # Bucket i counts completed runs of exactly i steps; bucket 0 is never
        # written (a run has length >= 1) and the last bucket is the overflow.
        self._run_hist = torch.zeros(
            self._MAX_TRACKED_RUN + 1, dtype=torch.long, device=device
        )
        self._fatal_runs = torch.zeros((), dtype=torch.long, device=device)
        self._censored_runs = torch.zeros((), dtype=torch.long, device=device)
        self._log_prefix: str | None = None

    def _resolve(
        self,
        violated: torch.Tensor,
        min_steps: int,
        diagnostic_only: bool,
    ) -> torch.Tensor:
        """Advance the per-env counter and return the done mask.

        Every operation stays on device: this runs once per term per control
        step, so a host sync here would cost more than the term itself.
        """
        previous = self._run_steps.clone()
        self._run_steps.add_(1).mul_(violated)
        # A run that ends without terminating is a recovery; record its length.
        recovered = torch.logical_and(torch.logical_not(violated), previous > 0)
        self._run_hist.index_add_(
            0, previous.clamp_(max=self._MAX_TRACKED_RUN), recovered.long()
        )
        if diagnostic_only:
            return torch.zeros_like(violated)
        done = self._run_steps >= max(int(min_steps), 1)
        self._fatal_runs += done.sum()
        # Those environments reset next step; clearing now keeps `reset` from
        # counting a fatal run a second time as censored.
        self._run_steps.mul_(torch.logical_not(done))
        return done

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected: Sequence[int] | slice = slice(None) if env_ids is None else env_ids
        self._censored_runs += (self._run_steps[selected] > 0).sum()
        self._run_steps[selected] = 0
        self._publish_stats()

    def _resolve_log_prefix(self) -> str:
        if self._log_prefix is None:
            name = type(self).__name__
            manager = getattr(self._env, "termination_manager", None)
            if manager is not None:
                for term_name in manager.active_terms:
                    if manager.get_term_cfg(term_name).func is self:
                        name = term_name
                        break
            self._log_prefix = f"Termination_Window/{name}"
        return self._log_prefix

    def _publish_stats(self) -> None:
        """Write cumulative run-length statistics into the episode log.

        Called from ``reset``, which the termination manager runs after
        ``extras["log"]`` is recreated for this reset, so the entries survive
        into whatever the training runner reports.
        """
        extras = getattr(self._env, "extras", None)
        log = extras.get("log") if isinstance(extras, dict) else None
        if not isinstance(log, dict):
            return
        # One host transfer for the whole term rather than one per statistic.
        packed = torch.cat(
            (
                self._run_hist,
                self._fatal_runs.reshape(1),
                self._censored_runs.reshape(1),
            )
        ).tolist()
        hist = packed[: self._MAX_TRACKED_RUN + 1]
        fatal, censored = packed[-2], packed[-1]
        recovered = sum(hist)
        total = recovered + fatal + censored
        prefix = self._resolve_log_prefix()
        log[f"{prefix}/runs_total"] = float(total)
        log[f"{prefix}/runs_fatal"] = float(fatal)
        log[f"{prefix}/runs_censored"] = float(censored)
        if recovered > 0:
            steps = sum(index * count for index, count in enumerate(hist))
            log[f"{prefix}/recovered_mean_steps"] = steps / recovered
        if total > 0:
            for window in self._RECOVERY_BUCKETS:
                # Runs shorter than `window` are exactly the ones a window of
                # that length would have survived.
                log[f"{prefix}/recovered_below_{window}_frac"] = (
                    sum(hist[1:window]) / total
                )


class PersistentBadAnchorPosZAdaptive(PersistentViolation):
    """Windowed :func:`bad_anchor_pos_z_adaptive`."""

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        env: ImitationRLEnv,
        threshold: float = 0.15,
        down_threshold: float = 0.75,
        root_height_threshold: float = 0.5,
        anchor_body_name: str = "pelvis",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_steps: int = 1,
        diagnostic_only: bool = False,
    ) -> torch.Tensor:
        violated = bad_anchor_pos_z_adaptive(
            env,
            threshold=threshold,
            down_threshold=down_threshold,
            root_height_threshold=root_height_threshold,
            anchor_body_name=anchor_body_name,
            asset_cfg=asset_cfg,
        )
        return self._resolve(violated, min_steps, diagnostic_only)


class PersistentBadAnchorOriFull(PersistentViolation):
    """Windowed :func:`bad_anchor_ori_full`."""

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        env: ImitationRLEnv,
        threshold: float = 0.2,
        anchor_body_name: str = "pelvis",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_steps: int = 1,
        diagnostic_only: bool = False,
    ) -> torch.Tensor:
        violated = bad_anchor_ori_full(
            env,
            threshold=threshold,
            anchor_body_name=anchor_body_name,
            asset_cfg=asset_cfg,
        )
        return self._resolve(violated, min_steps, diagnostic_only)


class PersistentBadReferenceBodyPosZAdaptive(PersistentViolation):
    """Windowed :func:`bad_reference_body_pos_z_adaptive`."""

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        env: ImitationRLEnv,
        threshold: float = 0.15,
        down_threshold: float = 0.75,
        root_height_threshold: float = 0.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_body_names: Sequence[str] = (),
        min_steps: int = 1,
        diagnostic_only: bool = False,
    ) -> torch.Tensor:
        violated = bad_reference_body_pos_z_adaptive(
            env,
            threshold=threshold,
            down_threshold=down_threshold,
            root_height_threshold=root_height_threshold,
            asset_cfg=asset_cfg,
            reference_body_names=reference_body_names,
        )
        return self._resolve(violated, min_steps, diagnostic_only)


class PersistentBadReferenceBodyPosRelative(PersistentViolation):
    """Windowed :func:`bad_reference_body_pos_relative`."""

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        env: ImitationRLEnv,
        threshold: float = 0.2,
        anchor_body_name: str = "pelvis",
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_body_names: Sequence[str] = (),
        down_threshold: float | None = None,
        root_height_threshold: float = 0.5,
        swing_threshold: float | None = None,
        swing_height_threshold: float = 0.15,
        min_steps: int = 1,
        diagnostic_only: bool = False,
    ) -> torch.Tensor:
        violated = bad_reference_body_pos_relative(
            env,
            threshold=threshold,
            anchor_body_name=anchor_body_name,
            asset_cfg=asset_cfg,
            reference_body_names=reference_body_names,
            down_threshold=down_threshold,
            root_height_threshold=root_height_threshold,
            swing_threshold=swing_threshold,
            swing_height_threshold=swing_height_threshold,
        )
        return self._resolve(violated, min_steps, diagnostic_only)
