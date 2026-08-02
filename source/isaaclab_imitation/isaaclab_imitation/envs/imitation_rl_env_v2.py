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
- :class:`~isaaclab_imitation.envs.command_plane.LatentCommandBuffer` and
  :class:`~isaaclab_imitation.envs.command_plane.HeldCommandPlane`: the
  agent-published latent buffer and the held explicit-chunk publish surface.

The env keeps every public name the mdp funcs, the command terms, the RLOpt
wrapper, and the ``imitation_experiments`` planners call (``trajectory_manager``,
``current_expert_frame``, the ``_get_*_fast`` accessors, the publish surface,
the three RLOpt-discovered expert samplers, ...) as thin delegators or as
thinly orchestrated methods, with the same lifecycle ordering as the legacy
env so the v2 surface is behaviorally equivalent.

Excluded from the v2 env (legacy-only): the hand-rolled ``Metrics/mpjpe_mm*``
logging channel (the ``motion`` command term owns ``Metrics/motion/*``), the
diagnostic command trace, and the marker/visualizer tooling. Reference
replay (``replay_reference`` / ``replay_only`` and the replay-target cursor
sync) IS supported, ported from the legacy env.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab_imitation.contracts.causal_planner_observation import (
    CAUSAL_PLANNER_FRAME_DIM,
    CausalPlannerHistory,
    build_causal_planner_frame,
    causal_planner_observation_spec,
)
from tensordict import TensorDict

from .command_plane import HeldCommandPlane, LatentCommandBuffer
from .expert_data_plane import ExpertDataPlane

logger = logging.getLogger(__name__)

_CAUSAL_PLANNER_HISTORY_STEPS = 9

_COMMAND_OBSERVATION_SOURCES = frozenset({"reference", "planner", "planner_oracle"})
_POLICY_COMMAND_MODES = frozenset(
    {
        "reference",
        "explicit_chunk_current_slot",
        "full_body_chunk_current_slot",
        "ee_chunk_current_slot",
    }
)


def _normalize_command_observation_source(value: str) -> str:
    source = str(value).strip().lower().replace("-", "_")
    if source not in _COMMAND_OBSERVATION_SOURCES:
        raise ValueError(
            f"Unsupported command_observation_source={value!r}; "
            f"expected one of {sorted(_COMMAND_OBSERVATION_SOURCES)}."
        )
    return source


def _normalize_policy_command_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _POLICY_COMMAND_MODES:
        raise ValueError(
            f"Unsupported policy_command_mode={value!r}; "
            f"expected one of {sorted(_POLICY_COMMAND_MODES)}."
        )
    return mode


class ImitationRLEnv(ManagerBasedRLEnv):
    """Reference-tracking env composing the ExpertDataPlane and command planes.

    Config attributes: identical to :class:`ImitationRLEnvLegacy` (both
    classes accept the same v2 cfg); see the legacy class docstring for the
    full field reference. The lifecycle order mirrors the legacy env so that
    observations, rewards, terminations, and command values are
    behaviorally identical under the same cfg.
    """

    @staticmethod
    def _lafan_source_entries_from_loader_kwargs(
        loader_kwargs: Any,
    ) -> list[dict[str, Any]]:
        try:
            entries = loader_kwargs["dataset"]["trajectories"]["lafan1_csv"]
        except Exception:
            return []
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        """Initialize the v2 env: planes first, then the base managers."""
        # Get device
        device = torch.device(cfg.sim.device)
        num_envs = int(cfg.scene.num_envs)

        # Isaac Lab 3.0's hydra integration applies `env.*` CLI overrides with
        # a plain setattr on the config (no `from_dict` round-trip), so a
        # `env.lafan1_manifest_path=...` override may arrive here without the
        # manifest-derived loader config having been resolved yet.
        if getattr(cfg, "lafan1_manifest_path", None) is not None and not (
            self._lafan_source_entries_from_loader_kwargs(
                getattr(cfg, "loader_kwargs", {})
            )
        ):
            manifest_resolver = getattr(cfg, "_resolve_manifest_config", None)
            if callable(manifest_resolver):
                # Preserve values applied after config construction. Isaac
                # Lab's Hydra integration uses plain setattr for late CLI
                # overrides, so default resolution would silently replace an
                # explicit cache and motion subset.
                configured_dataset_path = getattr(cfg, "dataset_path", None)
                default_dataset_path = getattr(
                    type(cfg), "dataset_path", "data/lafan1/g1/"
                )
                manifest_resolver(
                    dataset_path_explicit=(
                        configured_dataset_path is not None
                        and configured_dataset_path != default_dataset_path
                    ),
                    motions_explicit=getattr(cfg, "motions", None) is not None,
                )

        # Same plain-setattr gotcha for the command configuration:
        # `env.command_mode=...` / `env.command_observation_terms=[...]` can
        # arrive as direct field writes after `__post_init__` already pruned
        # the observation groups with the class defaults. Re-derive the
        # pruned command-term set from the final field values before any
        # manager reads the observation config. Idempotent for defaults.
        refresh_command_terms = getattr(cfg, "_refresh_command_observation_terms", None)
        if callable(refresh_command_terms):
            refresh_command_terms()

        # Reset / start-frame configuration (cluster K: the env parses these
        # once; on v2 the `motion` command term owns the reset-start samplers
        # and reads the parsed knobs back through the env attributes).
        reference_start_frame = int(getattr(cfg, "reference_start_frame", 0))
        if reference_start_frame < 0:
            raise ValueError("reference_start_frame must be >= 0.")
        self._reference_start_frame = reference_start_frame
        self._latent_patch_past_steps = int(getattr(cfg, "latent_patch_past_steps", 0))
        self._latent_patch_future_steps = int(
            getattr(cfg, "latent_patch_future_steps", 0)
        )
        if self._latent_patch_past_steps < 0 or self._latent_patch_future_steps < 0:
            raise ValueError("latent patch window steps must be >= 0.")
        self._latent_goal_steps = int(getattr(cfg, "latent_goal_steps", 0))
        if self._latent_goal_steps < 0:
            raise ValueError("latent_goal_steps must be >= 0.")
        self._random_reset_step_min = int(getattr(cfg, "random_reset_step_min", 0))
        self._random_reset_step_max = int(getattr(cfg, "random_reset_step_max", 0))
        self._random_reset_full_trajectory = bool(
            getattr(cfg, "random_reset_full_trajectory", False)
        )
        self._reset_start_mode = (
            str(getattr(cfg, "reset_start_mode", "auto")).strip().lower()
        )
        if self._reset_start_mode not in ("auto", "fixed", "random", "adaptive"):
            raise ValueError(
                "reset_start_mode must be one of 'auto', 'fixed', 'random', "
                f"'adaptive'; got {self._reset_start_mode!r}."
            )
        # Optional custom adaptive weight function (see StartFrameSampler). A
        # callable (trajectory_ranks, frame_steps) -> non-negative weights,
        # attached as `cfg.adaptive_reset_weight_fn`; falls back to the SONIC
        # failure-weight function when None.
        self._adaptive_reset_weight_fn = getattr(cfg, "adaptive_reset_weight_fn", None)
        self._adaptive_failure_reset_uniform_ratio = float(
            getattr(cfg, "adaptive_failure_reset_uniform_ratio", 0.1)
        )
        self._adaptive_failure_reset_bin_size = int(
            getattr(cfg, "adaptive_failure_reset_bin_size", 50)
        )
        self._adaptive_failure_reset_sequence_length_agnostic = bool(
            getattr(cfg, "adaptive_failure_reset_sequence_length_agnostic", True)
        )
        self._adaptive_failure_reset_init_num_failures = float(
            getattr(cfg, "adaptive_failure_reset_init_num_failures", 1.0)
        )
        self._adaptive_failure_reset_pre_failure_window = int(
            getattr(cfg, "adaptive_failure_reset_pre_failure_window", 200)
        )
        self._adaptive_failure_reset_failure_rate_max_over_mean = float(
            getattr(
                cfg,
                "adaptive_failure_reset_failure_rate_max_over_mean",
                50.0,
            )
        )
        if self._random_reset_step_min < 0:
            raise ValueError("random_reset_step_min must be >= 0.")
        if self._random_reset_step_max < self._random_reset_step_min:
            raise ValueError("random_reset_step_max must be >= random_reset_step_min.")
        if not 0.0 <= self._adaptive_failure_reset_uniform_ratio <= 1.0:
            raise ValueError("adaptive_failure_reset_uniform_ratio must be in [0, 1].")

        # Phase 1 of the data plane: dataset load, trajectory manager, first
        # reference frame. Runs before `super().__init__` (which builds the
        # managers), exactly where the legacy env loads its reference data.
        self.expert_data_plane = ExpertDataPlane(cfg, self)

        # Command-plane buffers (agent-published latent + held chunk windows).
        # Allocated before `super().__init__` so the command terms constructed
        # inside `load_managers()` can validate against them.
        self._agent_latent_dim = int(getattr(cfg, "latent_command_dim", 16))
        self.latent_command_buffer = LatentCommandBuffer(
            num_envs, self._agent_latent_dim, device
        )
        self._command_ee_body_names = tuple(
            str(name) for name in getattr(cfg, "command_ee_body_names", ())
        )
        self._command_keypoint_body_names = tuple(
            str(name) for name in getattr(cfg, "command_keypoint_body_names", ())
        )
        self._command_observation_source = _normalize_command_observation_source(
            getattr(cfg, "command_observation_source", "reference")
        )
        self._policy_command_mode = _normalize_policy_command_mode(
            getattr(cfg, "policy_command_mode", "reference")
        )
        self._command_hold_steps = int(getattr(cfg, "command_hold_steps", 0))
        if self._command_hold_steps < 0:
            raise ValueError("command_hold_steps must be >= 0.")
        if self._command_hold_steps > 0 and self._latent_patch_past_steps > 0:
            raise ValueError(
                "command_hold_steps requires latent_patch_past_steps == 0; "
                "held chunk consumption is only defined for future-only windows."
            )
        self._agent_trajectory_command_window_steps = (
            HeldCommandPlane.command_window_steps_from_offsets(
                self._latent_patch_past_steps,
                self._latent_patch_future_steps,
            )
        )
        self.held_command_plane = HeldCommandPlane(
            self,
            num_envs=num_envs,
            device=device,
            window_steps=self._agent_trajectory_command_window_steps,
            num_joints=len(self.reference_joint_names),
            num_ee_bodies=len(self._command_ee_body_names),
            num_keypoint_bodies=len(self._command_keypoint_body_names),
        )

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

        # The v2 env's reset/sampling path is fully absorbed into the
        # `motion` command term; require it so `step` / `_reset_idx` never
        # fall into the legacy inline branches.
        self._motion_reset_owner = self._motion_command_reset_owner()
        if self._motion_reset_owner is None:
            raise RuntimeError(
                "ImitationRLEnvV2 requires the `motion` command term with "
                "owns_reset=True: the term owns the reset-start samplers and "
                "the adaptive-failure bookkeeping (see MotionCommand)."
            )

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
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expert_data_plane.get_current_expert_window_term(
            term_name,
            past_steps=past_steps,
            future_steps=future_steps,
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
    # Command-plane delegators (publish surface).
    # ------------------------------------------------------------------

    @property
    def _agent_latent_command(self) -> torch.Tensor:
        return self.latent_command_buffer.get()

    def get_agent_latent_command(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.latent_command_buffer.get(env_ids)

    def set_agent_latent_command(
        self, latent_command: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        self.latent_command_buffer.set(latent_command, env_ids=env_ids)

    def reset_agent_latent_command(self, env_ids: torch.Tensor | None = None) -> None:
        self.latent_command_buffer.reset(env_ids)

    @property
    def _agent_trajectory_command_terms(self) -> dict[str, torch.Tensor]:
        return self.held_command_plane.terms

    def get_agent_trajectory_command_term(
        self, term_name: str, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.held_command_plane.get_term(term_name, env_ids=env_ids)

    def set_agent_trajectory_command(
        self,
        command_terms: Mapping[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.held_command_plane.set_command(command_terms, env_ids=env_ids)

    def set_agent_full_body_trajectory_command(
        self,
        *,
        expert_motion: torch.Tensor,
        expert_anchor_pos_b: torch.Tensor,
        expert_anchor_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.held_command_plane.set_full_body_command(
            expert_motion=expert_motion,
            expert_anchor_pos_b=expert_anchor_pos_b,
            expert_anchor_ori_b=expert_anchor_ori_b,
            env_ids=env_ids,
        )

    def set_agent_ee_trajectory_command(
        self,
        *,
        expert_ee_pos_b: torch.Tensor,
        expert_ee_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.held_command_plane.set_ee_command(
            expert_ee_pos_b=expert_ee_pos_b,
            expert_ee_ori_b=expert_ee_ori_b,
            env_ids=env_ids,
        )

    def reset_agent_trajectory_command(
        self, env_ids: torch.Tensor | None = None
    ) -> None:
        self.held_command_plane.reset(env_ids)

    def set_planner_command_provider(self, provider: Any) -> None:
        self.held_command_plane.set_planner_command_provider(provider)

    def capture_held_command_anchor(
        self,
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.held_command_plane.capture_held_command_anchor(
            anchor_body_name, env_ids=env_ids
        )

    def _command_hold_phase(self) -> torch.Tensor:
        return self.held_command_plane.hold_phase()

    @property
    def policy_command_mode(self) -> str:
        """Return the adapter used for low-level policy command observations."""
        return self._policy_command_mode

    # ------------------------------------------------------------------
    # Command-window orchestration (composes the planes).
    # ------------------------------------------------------------------

    def _validate_command_window_request(
        self,
        *,
        past_steps: int,
        future_steps: int,
    ) -> None:
        requested_steps = HeldCommandPlane.command_window_steps_from_offsets(
            past_steps,
            future_steps,
        )
        if requested_steps != self._agent_trajectory_command_window_steps:
            raise ValueError(
                "Planner command window mismatch. "
                f"Configured planner command has {self._agent_trajectory_command_window_steps} steps, "
                f"but observation requested {requested_steps} steps."
            )

    def get_current_command_window_term(
        self,
        term_name: str,
        *,
        past_steps: int,
        future_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = self._command_observation_source
        hold_steps = int(self._command_hold_steps)
        if source == "reference":
            if hold_steps <= 0:
                return self.get_current_expert_window_term(
                    term_name=term_name,
                    past_steps=past_steps,
                    future_steps=future_steps,
                    joint_ids=joint_ids,
                    anchor_body_name=anchor_body_name,
                    reference_body_names=reference_body_names,
                    env_ids=env_ids,
                )
            return self._held_reference_command_window_term(
                term_name,
                past_steps=past_steps,
                future_steps=future_steps,
                joint_ids=joint_ids,
                anchor_body_name=anchor_body_name,
                reference_body_names=reference_body_names,
                env_ids=env_ids,
            )

        self._validate_command_window_request(
            past_steps=past_steps,
            future_steps=future_steps,
        )
        if source == "planner_oracle":
            value = self.get_current_expert_window_term(
                term_name=term_name,
                past_steps=past_steps,
                future_steps=future_steps,
                joint_ids=joint_ids,
                anchor_body_name=anchor_body_name,
                reference_body_names=reference_body_names,
            )
            if hold_steps <= 0:
                self.set_agent_trajectory_command({term_name: value})
            else:
                renew_ids = torch.nonzero(
                    self._command_hold_phase() == 0, as_tuple=False
                ).flatten()
                if renew_ids.numel() > 0:
                    self.set_agent_trajectory_command(
                        {term_name: value.index_select(0, renew_ids)},
                        env_ids=renew_ids,
                    )
        elif source == "planner" and hold_steps > 0:
            # Give a registered planner the same in-step publication contract
            # as planner_oracle above, so its packet is expressed in the
            # anchor frame of the step that consumes it rather than one step
            # early.
            self.held_command_plane.maybe_fill_from_planner_provider(
                self._command_hold_phase()
            )

        # Observation tensors must not alias the mutable planner command
        # buffers: resets and subsequent planner publishes update those
        # buffers in-place.
        value = self.get_agent_trajectory_command_term(term_name).clone()
        if hold_steps > 0:
            phase = self._command_hold_phase()
            self.held_command_plane.update_held_command_anchor_pose(
                anchor_body_name, phase
            )
            value = HeldCommandPlane.shift_window_by_phase(
                value,
                phase,
                window_steps=self._agent_trajectory_command_window_steps,
            )
            value = self.held_command_plane.reexpress_window_in_current_anchor_frame(
                value,
                term_name=term_name,
                anchor_body_name=anchor_body_name,
                window_steps=self._agent_trajectory_command_window_steps,
            )
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def _held_reference_command_window_term(
        self,
        term_name: str,
        *,
        past_steps: int,
        future_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reference command windows under the held-chunk contract.

        Matches what standard VLA-WBC middleware would provide from a chunk
        planner running once per ``command_hold_steps``: command *content* is
        limited to the frames known at the last renewal (fresh lookahead
        shrinks toward the hold boundary, tail-padded with the boundary
        frame), while coordinates are re-expressed in the robot's current
        anchor frame every control step, exactly like odometry-based target
        re-expression on a real stack.
        """
        if int(past_steps) != 0:
            raise ValueError(
                "command_hold_steps requires past_steps == 0 command windows."
            )
        fresh = self.get_current_expert_window_term(
            term_name=term_name,
            past_steps=0,
            future_steps=int(future_steps),
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
        )
        value = HeldCommandPlane.clamp_window_to_hold_boundary(
            fresh, self._command_hold_phase(), window_steps=int(future_steps) + 1
        )
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def current_full_body_tracker_command_term(
        self,
        term_name: str,
        *,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume the current frame of a held full-body command chunk.

        The existing window path first shifts the held packet by each
        environments command phase and re-expresses anchor terms in the live
        robot frame. Selecting slot zero here therefore exposes the exact
        67-D command contract expected by the vanilla 50 Hz tracker.
        """
        if self._policy_command_mode not in (
            "explicit_chunk_current_slot",
            "full_body_chunk_current_slot",
            "ee_chunk_current_slot",
        ):
            raise RuntimeError(
                "Chunk slot adapter requested while policy_command_mode="
                f"{self._policy_command_mode!r}."
            )
        if self._command_observation_source not in {"planner", "planner_oracle"}:
            raise RuntimeError(
                "full_body_chunk_current_slot requires command_observation_source "
                "to be planner or planner_oracle."
            )
        if self._latent_patch_past_steps != 0:
            raise RuntimeError(
                "full_body_chunk_current_slot requires latent_patch_past_steps=0."
            )
        if self._command_hold_steps <= 0:
            raise RuntimeError(
                "full_body_chunk_current_slot requires command_hold_steps>0."
            )

        future_steps = int(self._latent_patch_future_steps)
        window_steps = future_steps + 1
        if window_steps < self._command_hold_steps:
            raise RuntimeError(
                "full_body_chunk_current_slot requires at least one command "
                "frame per held control step: "
                f"window_steps={window_steps}, hold_steps={self._command_hold_steps}."
            )
        value = self.get_current_command_window_term(
            term_name=term_name,
            past_steps=0,
            future_steps=future_steps,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
            env_ids=env_ids,
        )
        if value.ndim != 2 or int(value.shape[1]) % window_steps != 0:
            raise RuntimeError(
                f"Invalid streamed command term {term_name!r} shape "
                f"{tuple(value.shape)} for window_steps={window_steps}."
            )
        frame_width = int(value.shape[1]) // window_steps
        if term_name in ("expert_motion", "expert_motion_qpos"):
            # ``joint_ids`` defaults to slice(None), for which the fast lookup
            # returns the slice itself rather than an index tensor -- that is
            # every reference joint, which is what the window path selects too.
            joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
            num_joints = (
                len(self.reference_joint_names)
                if isinstance(joint_ids_t, slice)
                else int(joint_ids_t.numel())
            )
            # expert_motion carries positions AND velocities; expert_motion_qpos
            # is the position half only.
            expected_frame_width = num_joints * (
                2 if term_name == "expert_motion" else 1
            )
        elif term_name == "expert_anchor_pos_b":
            expected_frame_width = 3
        elif term_name == "expert_anchor_ori_b":
            expected_frame_width = 6
        elif term_name in ("expert_ee_pos_b", "expert_ee_ori_b"):
            # One 3-vector / rot6d per referenced end-effector body.
            n_bodies = max(int(len(reference_body_names)), 1)
            expected_frame_width = n_bodies * (
                3 if term_name == "expert_ee_pos_b" else 6
            )
        elif term_name in ("expert_keypoint_pos_b", "expert_keypoint_ori_b"):
            # Position and orientation are independent components, so a config
            # may select point targets or complete keypoint poses.
            n_bodies = max(int(len(reference_body_names)), 1)
            expected_frame_width = n_bodies * (
                3 if term_name == "expert_keypoint_pos_b" else 6
            )
        else:
            raise KeyError(f"Unsupported chunk tracker command term {term_name!r}.")
        if frame_width != expected_frame_width:
            raise RuntimeError(
                f"Streamed command term {term_name!r} has per-frame width "
                f"{frame_width}, expected {expected_frame_width}."
            )
        return value.reshape(value.shape[0], window_steps, frame_width)[:, 0, :]

    # ------------------------------------------------------------------
    # Reset / adaptive-failure sampling (cluster K, env-owned).
    # ------------------------------------------------------------------

    def _motion_command_reset_owner(self) -> Any | None:
        """The active command term that owns reset-start sampling (v2).

        Any command term with ``owns_reset=True`` qualifies: the lean
        ``command`` term on the default surface, or the ``motion`` term on the
        full explicit surface. Exactly one such term exists per config.
        """
        command_manager = getattr(self, "command_manager", None)
        if command_manager is None:
            return None
        for term_name in getattr(command_manager, "active_terms", ()):
            term = command_manager.get_term(term_name)
            if getattr(term.cfg, "owns_reset", False):
                return term
        return None

    def set_adaptive_reset_weight_fn(self, weight_fn: Any) -> None:
        """Provide a custom adaptive starting-frame weight function.

        Delegates to the owning ``motion`` command term, which owns the
        ``StartFrameSampler`` instance. The callable must accept
        ``(trajectory_ranks, frame_steps)`` tensors and return one
        non-negative weight per (rank, step) pair, as expected by
        ``iltools.datasets.reset_sampling.StartFrameSampler``. It replaces
        the SONIC failure-weight function and switches the term to
        ``reset_start_mode='adaptive'`` (trajectory ranks still come from the
        manager's reset schedule).
        """
        owner = getattr(self, "_motion_reset_owner", None)
        if owner is None:
            raise RuntimeError(
                "set_adaptive_reset_weight_fn requires the `motion` command "
                "term with owns_reset=True."
            )
        owner.set_weight_fn(weight_fn)

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

        # Reset trajectory tracking (reassigns trajectories and resets steps).
        # The `motion` command term owns the reset-start samplers AND the
        # adaptive-failure bookkeeping; failure bins are recorded here, at
        # the same point the legacy env-inline path recorded them (before
        # `super()._reset_idx`, whose reset-mode events read the reference at
        # the new cursor), then the term applies its samplers.
        motion_reset_owner = self._motion_reset_owner
        motion_reset_owner.record_failures(env_ids)
        motion_reset_owner.resample_reference(env_ids)
        self.reset_agent_latent_command(env_ids)
        self.reset_agent_trajectory_command(env_ids)

        # Refresh only the resetting rows before reset events consume
        # current_expert_frame.
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

        # Get next reference data point (advance=True to move to next step)
        self._refresh_current_expert_frame(advance=True)
        # Record the pre-step cursor as a visit in the SONIC failure sampler
        # (the reset-owning command term owns the sampler; the timing-critical
        # call site stays here, before the physics step).
        self._motion_reset_owner.record_visits()
        self._step_core(action)
        rollout_state_log = self._compute_rollout_reference_state_log()
        if rollout_state_log:
            self.extras.setdefault("log", {}).update(rollout_state_log)
        self._apply_reference_replay_targets()
        # Match IsaacLab command timing: reward/logging use the pre-step
        # reference frame, while returned observations expose the next frame.
        # The pre-step sample already advanced the trajectory cursor, so this
        # refresh must not advance again.
        self._refresh_current_expert_frame(advance=False)
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

        # Sample the current reference frame and advance the internal step by
        # exactly one. `sample(advance=True)` returns frame t and then
        # increments to t+1. This avoids double-advance while keeping reward
        # computation aligned with frame t.
        self._refresh_current_expert_frame(advance=True)
        self._motion_reset_owner.record_visits()
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
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
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
        # matching ManagerBasedRLEnv command timing after
        # command_manager.compute().
        self._refresh_current_expert_frame(advance=False)
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


# Back-compat alias: the pre-flip (2026-08-01) `-G1-v2` registration and any
# serialized config recorded against `isaaclab_imitation.envs:ImitationRLEnvV2`
# resolve to the same class.
ImitationRLEnvV2 = ImitationRLEnv
