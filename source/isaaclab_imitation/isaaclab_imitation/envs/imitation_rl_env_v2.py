"""ImitationRLEnv: the clean, composed env of the v2 fork (flagship).

The 4,997-line ``envs/imitation_rl_env_legacy.py`` is the byte-frozen LEGACY
env for the v0/v1 tasks (class :class:`ImitationRLEnvLegacy`); this module
holds the flagship env class :class:`ImitationRLEnv`, used only by the
``-G1-v2`` task registration (the default latent task since 2026-08-01). It
composes two owned components instead of carrying a ~5,000-line monolith:

- :class:`~isaaclab_imitation.envs.expert_data_plane.ExpertDataPlane`:
  dataset load, reference metadata / joint alignment, all MDP fast paths,
  frame refresh, alignment transforms, expert window / goal building, expert
  batch / macro-transition sampling, the parked reward-input cache, the MPJPE
  metric computation, and the offline-dataset mapper params.
- the two command terms of the declared command interface
  (``tasks/manager_based/imitation/command_interface.py``): the ``reference``
  channel (selection, reset-start sampling, tracking metrics, component
  emission) and the single ``actor`` channel (latent, explicit, or chunk),
  which own their own buffers and publish surfaces.

The env keeps every public name the mdp funcs, the command terms, the RLOpt
wrapper, and the ``imitation_experiments`` planners call (``trajectory_manager``,
``current_expert_frame``, the ``_get_*_fast`` accessors, the expert samplers,
...) as thin delegators or as thinly orchestrated methods.

Excluded from the v2 env (legacy-only): the hand-rolled ``Metrics/mpjpe_mm*``
logging channel (the reference channel owns ``Metrics/reference/*``), the
diagnostic command trace, and the marker/visualizer tooling. Reference
replay (``replay_reference`` / ``replay_only`` and the replay-target cursor
sync) IS supported, ported from the legacy env.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab_imitation.contracts.command_channels import (
    ACTOR_TERM_NAME,
    REFERENCE_TERM_NAME,
)
from isaaclab_imitation.contracts.causal_planner_observation import (
    CAUSAL_PLANNER_FRAME_DIM,
    CausalPlannerHistory,
    build_causal_planner_frame,
    causal_planner_observation_spec,
)
from tensordict import TensorDict

from .expert_data_plane import ExpertDataPlane
from .imitation_interface import ImitationEnvInterface

logger = logging.getLogger(__name__)

_CAUSAL_PLANNER_HISTORY_STEPS = 9


class ImitationRLEnv(ManagerBasedRLEnv):
    """Reference-tracking env composing the ExpertDataPlane and command planes.

    Config attributes: identical to :class:`ImitationRLEnvLegacy` (both
    classes accept the same v2 cfg); see the legacy class docstring for the
    full field reference. The lifecycle order mirrors the legacy env so that
    observations, rewards, terminations, and command values are
    behaviorally identical under the same cfg.
    """

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initialize the v2 env: planes first, then the base managers."""
        # Get device
        device = torch.device(cfg.sim.device)
        num_envs = int(cfg.scene.num_envs)

        # Isaac Lab 3.0 applies `env.*` CLI overrides and preset selections
        # after the config is constructed, so this is the first moment the
        # config's field values are final. The v2 configs own exactly one
        # resolution step, and it needs nothing from here: every derived value
        # is a function of the fields it can already see.
        cfg.resolve()

        # The declared command interface is the whole command surface. The
        # environment derives only what its data plane and reset path need;
        # the terms read the interface themselves.
        command_interface = cfg.command_interface
        reference_channel = command_interface.reference
        self._reference_start_frame = int(reference_channel.selection.start_frame)
        self._command_ee_body_names = tuple(
            str(name) for name in reference_channel.ee_body_names
        )
        self._command_keypoint_body_names = tuple(
            str(name) for name in reference_channel.keypoint_body_names
        )
        # Window the offline expert-batch mapper serves policy command keys at:
        # the skill encoder's view when there is one, else the actor's own.
        past_steps, future_steps, frame_stride = command_interface.expert_batch_window()
        self._latent_patch_past_steps = int(past_steps)
        self._latent_patch_future_steps = int(future_steps)
        self._latent_patch_frame_stride = int(frame_stride)
        self._latent_goal_steps = int(getattr(cfg, "latent_goal_steps", 0))
        if self._latent_goal_steps < 0:
            raise ValueError("latent_goal_steps must be >= 0.")

        # Phase 1 of the data plane: dataset load, trajectory manager, first
        # reference frame. Runs before `super().__init__` (which builds the
        # managers), exactly where the legacy env loads its reference data.
        self.expert_data_plane = ExpertDataPlane(cfg, self)

        self.replay_reference = getattr(cfg, "replay_reference", False)
        self.replay_only = getattr(cfg, "replay_only", False)
        if self.replay_only and not self.replay_reference:
            self.replay_reference = True
        self._reference_replay_targets_enabled = False
        self._reference_replay_source_env_ids: torch.Tensor | None = None
        self._reference_replay_target_env_ids: torch.Tensor | None = None
        self._last_tracked_root_pos_w = torch.zeros((num_envs, 3), device=device)
        self._last_tracked_root_pos_valid = torch.zeros(
            (num_envs,), device=device, dtype=torch.bool
        )

        # Initialize parent class (builds the managers; the `motion` term
        # constructs its owned reset-start samplers inside `load_managers`).
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        # Phase 2 of the data plane: retarget the trajectory manager, build
        # the fast paths against the live scene, resolve the metric bodies.
        self.expert_data_plane.finalize(self.scene, self.robot)
        self._initialize_causal_planner_history()

        # Reference selection, reset-start sampling, and the adaptive-failure
        # bookkeeping live on the reference channel; the environment only calls
        # them at the timing-critical points below.
        self._reference_term = self.reference_command

    # ------------------------------------------------------------------
    # ExpertDataPlane delegators (name-identical to the legacy env).
    # ------------------------------------------------------------------

    @property
    def trajectory_manager(self):
        return self.expert_data_plane.trajectory_manager

    @property
    def current_expert_frame(self) -> TensorDict | None:
        return self.expert_data_plane.current_expert_frame

    @property
    def _current_reference_local_step(self) -> torch.Tensor:
        return self.expert_data_plane._current_reference_local_step

    @property
    def reference_joint_names(self) -> list[str]:
        return self.expert_data_plane.reference_joint_names

    @property
    def reference_body_names(self) -> list[str]:
        return self.expert_data_plane.reference_body_names

    @property
    def reference_site_names(self) -> list[str]:
        return self.expert_data_plane.reference_site_names

    @property
    def _mpjpe_metric_body_names(self) -> list[str]:
        return self.expert_data_plane._mpjpe_metric_body_names

    @property
    def _mdp_reset_pose_bounds(self) -> torch.Tensor | None:
        return self.expert_data_plane._mdp_reset_pose_bounds

    @_mdp_reset_pose_bounds.setter
    def _mdp_reset_pose_bounds(self, value: torch.Tensor | None) -> None:
        self.expert_data_plane._mdp_reset_pose_bounds = value

    @property
    def _mdp_reset_velocity_bounds(self) -> torch.Tensor | None:
        return self.expert_data_plane._mdp_reset_velocity_bounds

    @_mdp_reset_velocity_bounds.setter
    def _mdp_reset_velocity_bounds(self, value: torch.Tensor | None) -> None:
        self.expert_data_plane._mdp_reset_velocity_bounds = value

    @property
    def _mdp_reset_root_pose_source(self) -> str:
        return self.expert_data_plane._mdp_reset_root_pose_source

    @property
    def _mdp_reset_root_velocity_source(self) -> str:
        return self.expert_data_plane._mdp_reset_root_velocity_source

    def _invalidate_mdp_cache(self) -> None:
        self.expert_data_plane._invalidate_mdp_cache()

    def _get_reference_alignment_fast(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_alignment_fast()

    def _get_body_ids_tensor_fast(
        self, body_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        return self.expert_data_plane._get_body_ids_tensor_fast(body_ids)

    def _get_joint_ids_tensor_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        return self.expert_data_plane._get_joint_ids_tensor_fast(joint_ids)

    def _get_reference_root_state_w_fast(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_root_state_w_fast()

    def _get_reference_cvel_fast(self) -> torch.Tensor:
        return self.expert_data_plane._get_reference_cvel_fast()

    def _get_reference_body_ids_fast(
        self, reference_body_names: Sequence[str]
    ) -> torch.Tensor:
        return self.expert_data_plane._get_reference_body_ids_fast(reference_body_names)

    def _get_robot_body_ids_by_name_fast(
        self, body_names: Sequence[str]
    ) -> torch.Tensor:
        return self.expert_data_plane._get_robot_body_ids_by_name_fast(body_names)

    def _get_reference_body_pose_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_body_pose_w_fast(
            reference_body_names
        )

    def _get_reference_body_velocity_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_body_velocity_w_fast(
            reference_body_names
        )

    def _get_robot_anchor_body_id_fast(self, anchor_body_name: str) -> int:
        return self.expert_data_plane._get_robot_anchor_body_id_fast(anchor_body_name)

    def _get_robot_anchor_state_w_fast(
        self, anchor_body_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_anchor_state_w_fast(anchor_body_name)

    def _get_robot_body_pose_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_pose_w_fast(body_ids)

    def _get_robot_body_velocity_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_velocity_w_fast(body_ids)

    def _get_robot_body_state_in_anchor_frame_fast(
        self,
        body_ids: Sequence[int] | torch.Tensor | slice,
        anchor_body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_robot_body_state_in_anchor_frame_fast(
            body_ids, anchor_body_name
        )

    def get_expert_motion_qpos_command(
        self, joint_ids: Sequence[int] | slice = slice(None)
    ) -> torch.Tensor:
        return self.expert_data_plane.get_expert_motion_qpos_command(joint_ids)

    def _get_expert_motion_command_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor:
        return self.expert_data_plane._get_expert_motion_command_fast(joint_ids)

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expert_data_plane._get_reference_alignment_transform(env_ids)

    def _transform_reference_pose_to_world(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.expert_data_plane._transform_reference_pose_to_world(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _transform_reference_body_pose_to_init_alignment(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.expert_data_plane._transform_reference_body_pose_to_init_alignment(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _index_copy_reference_rows_(
        self, dst: TensorDict, src: TensorDict, env_ids: torch.Tensor
    ) -> None:
        self.expert_data_plane._index_copy_reference_rows_(dst, src, env_ids)

    def _refresh_current_expert_frame(
        self, env_ids: torch.Tensor | None = None, *, advance: bool = False
    ) -> None:
        self.expert_data_plane._refresh_current_expert_frame(env_ids, advance=advance)

    def current_reference_is_final_frame(self) -> torch.Tensor:
        return self.expert_data_plane.current_reference_is_final_frame()

    def _current_local_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.expert_data_plane._current_local_steps(env_ids)

    def _compute_mpjpe_metric(self) -> torch.Tensor | None:
        return self.expert_data_plane._compute_mpjpe_metric()

    def _compute_mpjpe_metrics(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.expert_data_plane._compute_mpjpe_metrics()

    def _compute_rollout_reference_state_log(self) -> dict[str, float]:
        return self.expert_data_plane._compute_rollout_reference_state_log()

    def get_offline_dataset_mapper_params(self) -> dict[str, Any]:
        return self.expert_data_plane.get_offline_dataset_mapper_params()

    def get_expert_trajectory_data(
        self, key: str | None = None, joint_indices: Sequence[int] | None = None
    ) -> TensorDict | torch.Tensor:
        """Get the current reference data.

        Args:
            key: Specific key to extract. If None, returns full TensorDict.

        Returns:
            Reference data for all environments
        """
        if self.current_expert_frame is None:
            raise RuntimeError("No reference data available. Call reset() first.")

        if key is None:
            return self.current_expert_frame

        data: torch.Tensor | TensorDict | None = None
        if key in self.current_expert_frame:
            data = self.current_expert_frame[key]
        elif key == "xpos" and "body_pos_w" in self.current_expert_frame:
            data = self.current_expert_frame["body_pos_w"]
        elif key == "xquat" and "body_quat_w" in self.current_expert_frame:
            data = self.current_expert_frame["body_quat_w"]
        elif key == "cvel":
            data = self._get_reference_cvel_fast()

        if data is None:
            available_keys = [str(k) for k in self.current_expert_frame.keys()]
            raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")

        if joint_indices is not None:
            return data[..., joint_indices]
        else:
            return data  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Expert sampling delegators (RLOpt discovers these by name).
    # ------------------------------------------------------------------

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[Any]
    ) -> TensorDict | None:
        return self.expert_data_plane.sample_expert_batch(batch_size, required_keys)

    def expert_macro_frame_stride(self) -> int:
        """Reference frames between consecutive DiffSR macro-window slots.

        Published so a consumer that pairs a pretrained skill encoder with this
        environment can refuse a stride the encoder was not trained on. The
        macro state's width is identical at every stride, so this is the only
        way that mismatch can be detected.
        """
        return self.expert_data_plane._expert_macro_frame_stride()

    def expert_macro_anchor_mode(self) -> str:
        """Frame convention of the DiffSR macro window.

        Published so a consumer that pairs a pretrained skill encoder with this
        environment can refuse a frame convention the encoder was not trained
        on. The macro state's width is identical in every mode, so this is the
        only way that mismatch can be detected.
        """
        return self.expert_data_plane._expert_macro_anchor_mode()

    def sample_expert_macro_transition_batch(
        self,
        batch_size: int,
        horizon_steps: int,
        split: str | None = None,
        eval_fraction: float = 0.1,
        split_seed: int = 0,
        trajectory_ranks: Sequence[int] | torch.Tensor | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.sample_expert_macro_transition_batch(
            batch_size,
            horizon_steps,
            split=split,
            eval_fraction=eval_fraction,
            split_seed=split_seed,
            trajectory_ranks=trajectory_ranks,
            state_history_steps=state_history_steps,
        )

    def current_expert_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.current_expert_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def current_achieved_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        return self.expert_data_plane.current_achieved_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )

    def expert_macro_feature_slices(
        self,
        horizon_steps: int,
    ) -> dict[str, tuple[int, int]]:
        return self.expert_data_plane.expert_macro_feature_slices(horizon_steps)

    def expert_trajectory_motion_names(self) -> list[str]:
        return self.expert_data_plane.expert_trajectory_motion_names()

    def get_current_expert_window_term(
        self,
        term_name: str,
        *,
        past_steps: int,
        future_steps: int,
        frame_stride: int = 1,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expert_data_plane.get_current_expert_window_term(
            term_name,
            past_steps=past_steps,
            future_steps=future_steps,
            frame_stride=frame_stride,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
            env_ids=env_ids,
        )

    def get_current_expert_goal_term(
        self,
        term_name: str,
        *,
        goal_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expert_data_plane.get_current_expert_goal_term(
            term_name,
            goal_steps=goal_steps,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            env_ids=env_ids,
        )

    # ------------------------------------------------------------------
    # Command-channel access (the two terms own their state).
    # ------------------------------------------------------------------

    @property
    def imitation_interface(self) -> ImitationEnvInterface:
        """The capability surface RLOpt resolves (expert data + publication)."""
        interface = getattr(self, "_imitation_interface", None)
        if interface is None:
            interface = ImitationEnvInterface(self)
            self._imitation_interface = interface
        return interface

    @property
    def reference_command(self):
        """The always-present reference channel term."""
        return self.command_manager.get_term(REFERENCE_TERM_NAME)

    @property
    def actor_command(self):
        """The single actor command term (latent, explicit, or chunk)."""
        return self.command_manager.get_term(ACTOR_TERM_NAME)

    def set_adaptive_reset_weight_fn(self, weight_fn: Any) -> None:
        """Install a custom adaptive start-frame weight function.

        Delegates to the reference channel, which owns the ``StartFrameSampler``.
        The callable takes ``(trajectory_ranks, frame_steps)`` and returns one
        non-negative weight per pair, as expected by
        ``iltools.datasets.reset_sampling.StartFrameSampler``.
        """
        self.reference_command.set_weight_fn(weight_fn)

    # ------------------------------------------------------------------
    # Causal planner observation surface (robot-only, no expert reads).
    # ------------------------------------------------------------------

    def _pinned_joint_ids(self) -> torch.Tensor:
        """Live articulation indices in the action term's pinned joint order.

        Physics backends enumerate the articulation differently, so any joint
        vector that a policy or a recorded dataset consumes must be expressed
        in a fixed order. The action term already pins one via
        ``preserve_order=True``; reusing its mapping keeps joint state, the
        action, and the expert command mutually pairable on every backend.
        """
        cached = getattr(self, "_pinned_joint_ids_cache", None)
        if cached is not None:
            return cached
        term_joint_ids = self.action_manager.get_term("joint_pos")._joint_ids
        if isinstance(term_joint_ids, slice):
            term_joint_ids = range(self.robot.num_joints)[term_joint_ids]
        pinned = torch.as_tensor(
            list(term_joint_ids), dtype=torch.long, device=self.device
        )
        self._pinned_joint_ids_cache = pinned
        return pinned

    def _current_causal_planner_frame(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> torch.Tensor:
        """Build one planner frame exclusively from the live robot and action state."""
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        action = self.action_manager.action.index_select(0, env_ids_t)
        joint_ids = self._pinned_joint_ids()
        return build_causal_planner_frame(
            {
                "joint_pos_rel": (
                    self.robot.data.joint_pos.index_select(0, env_ids_t)
                    - self.robot.data.default_joint_pos.index_select(0, env_ids_t)
                ).index_select(1, joint_ids),
                "joint_vel_rel": (
                    self.robot.data.joint_vel.index_select(0, env_ids_t)
                    - self.robot.data.default_joint_vel.index_select(0, env_ids_t)
                ).index_select(1, joint_ids),
                "base_ang_vel": self.robot.data.root_ang_vel_b.index_select(
                    0, env_ids_t
                ),
                "projected_gravity": self.robot.data.projected_gravity_b.index_select(
                    0, env_ids_t
                ),
                "last_action": action,
            }
        )

    def _initialize_causal_planner_history(self) -> None:
        """Initialize ten 50 Hz frames using repeat-first reset padding."""
        frame = self._current_causal_planner_frame()
        if tuple(frame.shape) != (self.num_envs, CAUSAL_PLANNER_FRAME_DIM):
            raise ValueError(
                "Unexpected causal planner frame shape during initialization: "
                f"{tuple(frame.shape)}."
            )
        self._causal_planner_history = CausalPlannerHistory(
            frame, history_steps=_CAUSAL_PLANNER_HISTORY_STEPS
        )

    def _reset_causal_planner_history(self, env_ids: torch.Tensor) -> None:
        history = getattr(self, "_causal_planner_history", None)
        if history is None:
            return
        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        ).reshape(-1)
        if int(env_ids_t.numel()) == 0:
            return
        frame = self._current_causal_planner_frame(env_ids_t)
        history.reset(env_ids_t, frame)

    def _append_causal_planner_history(self) -> None:
        history = getattr(self, "_causal_planner_history", None)
        if history is None:
            return
        history.append(self._current_causal_planner_frame())

    def causal_planner_observation_spec(
        self, history_steps: int = _CAUSAL_PLANNER_HISTORY_STEPS
    ) -> dict[str, Any]:
        """Return the exact feature and history contract used by the planner."""
        history_steps = int(history_steps)
        if history_steps > _CAUSAL_PLANNER_HISTORY_STEPS:
            raise ValueError(
                "history_steps exceeds the live causal history capacity: "
                f"{history_steps} > {_CAUSAL_PLANNER_HISTORY_STEPS}."
            )
        return causal_planner_observation_spec(history_steps=history_steps)

    def current_causal_planner_observation(
        self,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        history_steps: int = _CAUSAL_PLANNER_HISTORY_STEPS,
    ) -> TensorDict:
        """Return robot-only planner state without reading any expert reference."""
        history_steps = int(history_steps)
        spec = self.causal_planner_observation_spec(history_steps)
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        if int(env_ids_t.numel()) == 0:
            raise ValueError("env_ids must select at least one environment.")
        history = self._causal_planner_history.select(
            env_ids_t, history_steps=history_steps
        )
        expected_history = (
            int(env_ids_t.numel()),
            int(spec["history_frames"]),
            int(spec["frame_dim"]),
        )
        if tuple(history.shape) != expected_history:
            raise RuntimeError(
                "Causal planner history shape mismatch: expected "
                f"{expected_history}, got {tuple(history.shape)}."
            )
        planner = TensorDict(
            {"state": history[:, -1], "state_history": history},
            batch_size=[int(env_ids_t.numel())],
            device=self.device,
        )
        return TensorDict(
            {"planner": planner},
            batch_size=[int(env_ids_t.numel())],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Reference replay (ported from the legacy env; vis stays legacy-only).
    # ------------------------------------------------------------------

    def configure_reference_replay_targets(
        self,
        *,
        source_env_ids: Sequence[int] | torch.Tensor,
        target_env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Configure target envs to replay the reference cursor of source envs."""

        source_env_ids_t = torch.as_tensor(
            source_env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        target_env_ids_t = torch.as_tensor(
            target_env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)
        if source_env_ids_t.shape != target_env_ids_t.shape:
            raise ValueError(
                "source_env_ids and target_env_ids must have the same shape."
            )

        self._reference_replay_source_env_ids = source_env_ids_t
        self._reference_replay_target_env_ids = target_env_ids_t
        self._reference_replay_targets_enabled = True

    def apply_reference_replay_targets(self) -> None:
        """Public hook to synchronize and replay configured reference target envs."""

        self._apply_reference_replay_targets()

    def _apply_reference_replay_targets(self) -> None:
        """Replay target envs from their paired source env trajectory cursors."""

        if not self._reference_replay_targets_enabled:
            return
        if (
            self._reference_replay_source_env_ids is None
            or self._reference_replay_target_env_ids is None
        ):
            return

        self.sync_reference_cursor_from_source_envs(
            source_env_ids=self._reference_replay_source_env_ids,
            target_env_ids=self._reference_replay_target_env_ids,
        )
        self._replay_reference(env_ids=self._reference_replay_target_env_ids)

    def sync_reference_cursor_from_source_envs(
        self,
        *,
        source_env_ids: Sequence[int] | torch.Tensor,
        target_env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Copy trajectory cursor state from source envs to target envs."""

        tm = self.trajectory_manager
        source_env_ids_tm = torch.as_tensor(
            source_env_ids, dtype=torch.long, device=tm._state_device
        ).reshape(-1)
        target_env_ids_tm = torch.as_tensor(
            target_env_ids, dtype=torch.long, device=tm._state_device
        ).reshape(-1)
        if source_env_ids_tm.shape != target_env_ids_tm.shape:
            raise ValueError(
                "source_env_ids and target_env_ids must have the same shape."
            )
        if source_env_ids_tm.numel() == 0:
            return

        source_ranks = tm.env_traj_rank.index_select(0, source_env_ids_tm)
        source_steps = tm.env_step.index_select(0, source_env_ids_tm)
        tm.set_env_cursor(
            env_ids=target_env_ids_tm,
            ranks=source_ranks,
            steps=source_steps,
        )

        source_env_ids = source_env_ids_tm.to(device=self.device)
        target_env_ids = target_env_ids_tm.to(device=self.device)

        self._refresh_current_expert_frame(target_env_ids, advance=False)

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
        if tracked_root_pos_w is not None:
            self._last_tracked_root_pos_w.index_copy_(
                0,
                target_env_ids,
                tracked_root_pos_w.index_select(0, target_env_ids),
            )
            self._last_tracked_root_pos_valid.index_fill_(0, target_env_ids, True)

    def _replay_reference(
        self, env_ids: torch.Tensor | None = None, reference: TensorDict | None = None
    ):
        """Replay the reference data.

        If env_ids is provided, only replay the reference data for the given
        environments. If env_ids is not provided, replay the reference data
        for all environments.
        """

        if env_ids is None:
            ref = self.current_expert_frame if reference is None else reference
            defaults_pos = self.robot.data.default_joint_pos.torch
            defaults_vel = self.robot.data.default_joint_vel.torch
        else:
            env_ids_tensor = env_ids
            full_reference = (
                self.current_expert_frame if reference is None else reference
            )
            ref = full_reference[env_ids_tensor]
            defaults_pos = self.robot.data.default_joint_pos.torch[env_ids_tensor]
            defaults_vel = self.robot.data.default_joint_vel.torch[env_ids_tensor]

        root_pos, root_quat_opt = self._transform_reference_pose_to_world(
            ref["root_pos"], ref["root_quat"], env_ids=env_ids
        )
        if root_quat_opt is None:
            raise RuntimeError(
                "Failed to transform reference root quaternion for replay."
            )
        root_quat = root_quat_opt
        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = self._estimate_reference_root_lin_vel_w_from_pos(
            ref["root_pos"], env_ids=env_ids
        )
        root_ang_vel = math_utils.quat_apply(align_quat, ref["root_ang_vel"])
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        root_vel = torch.cat([root_lin_vel, root_ang_vel], dim=-1)
        # Extract joint data from reference TensorDict
        # ref is a TensorDict, so accessing keys returns tensors
        joint_pos_raw = ref["joint_pos"]  # type: ignore[assignment]
        joint_vel_raw = ref["joint_vel"]  # type: ignore[assignment]
        joint_pos = joint_pos_raw.clone()
        joint_vel = joint_vel_raw.clone()

        # Replace NaN positions with default values
        joint_pos = torch.where(torch.isnan(joint_pos), defaults_pos, joint_pos)
        joint_vel = torch.where(torch.isnan(joint_vel), defaults_vel, joint_vel)
        # Use link/com-specific writers so all articulation data buffers stay
        # coherent. `base_lin_vel` uses root_com_vel_w + root_link_quat_w
        # internally.
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()
        # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct
        # state writes.
        self.scene.update(dt=0.0)
        self.robot.update(dt=0.0)
        self._invalidate_mdp_cache()

    def _get_tracked_reference_root_pos_w(self) -> torch.Tensor | None:
        """Return tracked reference root positions in world frame for all environments."""
        if self.current_expert_frame is None:
            return None

        reference_root_pos = self.current_expert_frame.get("root_pos")
        if reference_root_pos is None:
            return None

        # Apply the full per-episode rigid transform (R, t) from reset frame to
        # world frame.
        tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
            reference_root_pos
        )
        return tracked_root_pos_w

    def _estimate_reference_root_lin_vel_w_from_pos(
        self,
        reference_root_pos: torch.Tensor,
        env_ids: torch.Tensor | None = None,
        update_cache: bool = False,
    ) -> torch.Tensor:
        """Estimate reference root linear velocity in world frame from finite differences of root position."""
        if env_ids is None:
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos
            )
            previous_pos_w = self._last_tracked_root_pos_w
            previous_valid = self._last_tracked_root_pos_valid
        else:
            env_ids_tensor = env_ids.to(dtype=torch.int64)
            tracked_root_pos_w, _ = self._transform_reference_pose_to_world(
                reference_root_pos, env_ids=env_ids_tensor
            )
            previous_pos_w = self._last_tracked_root_pos_w[env_ids_tensor]
            previous_valid = self._last_tracked_root_pos_valid[env_ids_tensor]

        reference_root_lin_vel_w = torch.zeros_like(tracked_root_pos_w)
        dt = float(self.step_dt)
        if dt > 0.0:
            reference_root_lin_vel_w[previous_valid] = (
                tracked_root_pos_w[previous_valid] - previous_pos_w[previous_valid]
            ) / dt

        if update_cache:
            if env_ids is None:
                self._last_tracked_root_pos_w.copy_(tracked_root_pos_w)
                self._last_tracked_root_pos_valid.fill_(True)
            else:
                env_ids_tensor = env_ids.to(dtype=torch.int64)
                self._last_tracked_root_pos_w[env_ids_tensor] = tracked_root_pos_w
                self._last_tracked_root_pos_valid[env_ids_tensor] = True

        return reference_root_lin_vel_w

    # ------------------------------------------------------------------
    # Lifecycle: step / reset.
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the specified environments.

        Notes:
            IsaacLab managers, events, and sensors accept tensor indices and
            internally move them to the appropriate device. We normalize
            ``env_ids`` to a CUDA long tensor so that all internal buffers
            (which live on ``self.device``) and the trajectory manager see
            consistent indexing.
        """
        # Isaac Lab 3.0 hands out int32 env indices; normalize once here.
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # Reset reference tracking (reassigns trajectories and resets steps).
        # The reference channel owns the reset-start samplers AND the
        # adaptive-failure bookkeeping; failure bins are recorded here, the
        # last point at which the cursor still belongs to the ending episode,
        # and selection runs before `super()._reset_idx`, whose reset-mode
        # events read the reference at the new cursor. The actor command's own
        # reset (zeroing a stale published command) happens inside
        # `command_manager.reset`, which `super()._reset_idx` triggers.
        self._reference_term.record_failures(env_ids)
        prefetched_reset_count = self._reference_term.resample_reference(env_ids)

        # Refresh only the resetting rows before reset events consume
        # current_expert_frame.
        if prefetched_reset_count > 0:
            self.expert_data_plane.consume_predicted_reset_reference(
                env_ids, prefetched_count=prefetched_reset_count
            )
        else:
            self._refresh_current_expert_frame(env_ids, advance=False)

        # Trigger the reset events (curriculum, sensors, managers, etc.) using
        # tensor indices.
        result = super()._reset_idx(env_ids)  # type: ignore[arg-type]

        if self.replay_reference:
            self._replay_reference(env_ids)

        tracked_root_pos_w = self._get_tracked_reference_root_pos_w()
        if tracked_root_pos_w is not None:
            self._last_tracked_root_pos_w.index_copy_(
                0, env_ids, tracked_root_pos_w.index_select(0, env_ids)
            )
            self._last_tracked_root_pos_valid.index_fill_(0, env_ids, True)

        self._reset_causal_planner_history(env_ids)

        return result

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data.

        Thin single-compute step: unlike the legacy env (which runs the base
        env's mid-step observation compute at the reward frame and then
        recomputes at the next frame), this env runs the physics /
        terminations / rewards / resets / command lifecycle directly and
        computes observations exactly ONCE, at the returned next-reference
        frame. The returned values follow the same frame contract (rewards at
        frame T, obs at frame T+1); the difference is the stochastic stream:
        the discarded mid-step compute also drew observation noise, so the
        v2 stream is fresh and v2 is deliberately NOT bit-equivalent to the
        legacy env.
        """
        # Replay-only path: ignore physics stepping and evaluate rewards
        # exactly on the replayed reference state (ported from the legacy
        # env's replay-only branch).
        if self.replay_only:
            return self._step_replay_only(action)

        # The returned frame from the preceding step is already the reward
        # frame for this one. Advance the logical cursors without rereading it,
        # and (when configured) stage the next rows while physics runs.
        self.expert_data_plane.begin_next_reference()
        # Record the pre-step cursor as a visit in the SONIC failure sampler
        # (the reference channel owns the sampler; the timing-critical call
        # site stays here, before the physics step).
        self._reference_term.record_visits()
        self._reference_term.prepare_predicted_resets()
        self._step_core(action)
        rollout_state_log = self._compute_rollout_reference_state_log()
        if rollout_state_log:
            self.extras.setdefault("log", {}).update(rollout_state_log)
        self._apply_reference_replay_targets()
        # Match IsaacLab command timing: reward/logging used the pre-step frame,
        # while returned observations expose the next frame. Reset rows were
        # refreshed synchronously inside `_reset_idx`; the data plane patches
        # those over any stale sequential rows in the prefetched full batch.
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        override_env_ids = reset_env_ids
        if (
            self._reference_replay_targets_enabled
            and self._reference_replay_target_env_ids is not None
        ):
            override_env_ids = torch.unique(
                torch.cat(
                    (
                        reset_env_ids,
                        self._reference_replay_target_env_ids.to(
                            device=self.device, dtype=torch.long
                        ),
                    )
                )
            )
        self.expert_data_plane.finish_next_reference(override_env_ids)
        self._reference_term.finish_predicted_reset_step()
        prefetch_log = self.expert_data_plane.reference_prefetch_metrics()
        if prefetch_log:
            self.extras.setdefault("log", {}).update(prefetch_log)
        self._append_causal_planner_history()
        # The single observation compute of the step, at the returned frame.
        self.obs_buf = self.observation_manager.compute(update_history=True)
        if len(self.recorder_manager.active_terms) > 0:
            self.recorder_manager.record_post_step()
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _step_core(self, action: torch.Tensor) -> None:
        """The base env step body without its mid-step observation compute.

        Replicates ``ManagerBasedRLEnv.step``'s lifecycle (action process,
        physics loop with decimation, counters, terminations, rewards, reset
        handling with terminal obs / recorder hooks / re-renders, command
        compute, interval events) so the caller can run the single
        observation compute at the correct frame. The recorder-gated obs
        compute is dropped: with active recorder terms the post-step record
        hook runs after the caller's compute instead.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: uses cached property to avoid settings lookup every step
        is_rendering = self.sim.is_rendering

        # perform physics stepping
        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self.action_manager.apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                # render between steps only if the GUI or an RTX sensor needs it.
                if (
                    self._sim_step_counter % self.cfg.sim.render_interval == 0
                    and is_rendering
                ):
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            reset_env_ids_list = reset_env_ids.tolist()
            # Populate Gymnasium-style terminal observation info for vector
            # envs. final_obs/final_info are object arrays with None for
            # non-reset envs.
            final_obs = np.empty(self.num_envs, dtype=object)
            final_obs[:] = None
            final_info = np.empty(self.num_envs, dtype=object)
            final_info[:] = None

            def _slice_obs(obs: dict | torch.Tensor, env_id: int):
                if isinstance(obs, dict):
                    return {k: _slice_obs(v, env_id) for k, v in obs.items()}
                return obs[env_id].clone()

            for env_id in reset_env_ids_list:
                final_obs[env_id] = _slice_obs(self.obs_buf, env_id)
                final_info[env_id] = {}

            self.extras["final_obs"] = final_obs
            self.extras["final_info"] = final_info

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids_list)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect
            # changes in reset
            if (
                self.render_enabled
                and is_rendering
                and self.has_rtx_sensors
                and self.cfg.num_rerenders_on_reset > 0
            ):
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids_list)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

    def _step_replay_only(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Replay-only stepping: no physics, rewards on the replayed reference."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # The current frame is already resident from the preceding returned
        # observation. Advance without rereading it and stage t+1.
        self.expert_data_plane.begin_next_reference()
        self._reference_term.record_visits()
        self._reference_term.prepare_predicted_resets()
        self._replay_reference()
        self.scene.update(dt=0.0)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Clear any stale terminal info from previous steps.
        for key in ("final_obs", "final_info"):
            if key in self.extras:
                del self.extras[key]

        if len(reset_env_ids) > 0:
            reset_env_ids_list = reset_env_ids.tolist()
            # Populate Gymnasium-style terminal observation info for vector
            # envs. final_obs/final_info are object arrays with None for
            # non-reset envs.
            final_obs = np.empty(self.num_envs, dtype=object)
            final_obs[:] = None
            final_info = np.empty(self.num_envs, dtype=object)
            final_info[:] = None

            def _slice_obs(obs: dict | torch.Tensor, env_id: int):
                if isinstance(obs, dict):
                    return {k: _slice_obs(v, env_id) for k, v in obs.items()}
                return obs[env_id].clone()

            for env_id in reset_env_ids_list:
                final_obs[env_id] = _slice_obs(self.obs_buf, env_id)
                final_info[env_id] = {}

            self.extras["final_obs"] = final_obs
            self.extras["final_info"] = final_info

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids_list)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect
            # changes in reset
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids_list)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # Expose post-step reference (frame t+1) for observations/outputs,
        # matching ManagerBasedRLEnv command timing after command computation.
        self.expert_data_plane.finish_next_reference(reset_env_ids)
        self._reference_term.finish_predicted_reset_step()
        prefetch_log = self.expert_data_plane.reference_prefetch_metrics()
        if prefetch_log:
            self.extras.setdefault("log", {}).update(prefetch_log)
        self._append_causal_planner_history()
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        # return observations, rewards, resets and extras
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def close(self) -> None:
        """Drain the reference worker before releasing the simulator."""
        try:
            self.expert_data_plane.close()
        finally:
            super().close()


# Back-compat alias: the pre-flip (2026-08-01) `-G1-v2` registration and any
# serialized config recorded against `isaaclab_imitation.envs:ImitationRLEnvV2`
# resolve to the same class.
ImitationRLEnvV2 = ImitationRLEnv
