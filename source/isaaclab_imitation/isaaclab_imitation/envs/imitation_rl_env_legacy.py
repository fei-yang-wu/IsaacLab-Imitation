import logging
import os as _os
import shutil
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import isaaclab.utils.math as math_utils
import numpy as np
import torch
import zarr
from isaaclab.assets import Articulation
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import (
    FRAME_MARKER_CFG,
)
from isaaclab_imitation.assets.robots import UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES
from isaaclab_imitation.contracts.causal_planner_observation import (
    CAUSAL_PLANNER_FRAME_DIM,
    CausalPlannerHistory,
    build_causal_planner_frame,
    causal_planner_observation_spec,
)
from iltools.datasets.reset_sampling import (
    SonicAdaptiveResetSampler,
    StartFrameSampler,
)
from tensordict import TensorDict

logger = logging.getLogger(__name__)
NestedKey: TypeAlias = str | tuple[str, ...]
_MDP_COMPILED: Any | None = None

_COMMAND_OBSERVATION_SOURCES = frozenset({"reference", "planner", "planner_oracle"})
# ``*_chunk_current_slot`` all mean the same thing: the actor reads its command
# from the phase-aligned slot of the held packet instead of from the live
# reference. Only the packet's *content* differs by command space, so the
# reduced explicit interfaces (root_qpos, root_points5) reuse
# ``full_body_chunk_current_slot`` rather than adding names that behave
# identically. ``ee_chunk_current_slot`` is retained for the abandoned EE
# tracker's recorded contract.
_POLICY_COMMAND_MODES = frozenset(
    {
        "reference",
        "explicit_chunk_current_slot",
        "full_body_chunk_current_slot",
        "ee_chunk_current_slot",
    }
)
_CAUSAL_PLANNER_HISTORY_STEPS = 9
# Tracking maths stays in metres; MPJPE is reported in millimetres because that
# is the unit the closed-loop evaluators and the paper aggregators use.
_METRES_TO_MM = 1000.0


def _get_mdp_compiled_module() -> Any:
    global _MDP_COMPILED
    if _MDP_COMPILED is None:
        from isaaclab_imitation.tasks.manager_based.imitation.mdp import _compiled

        _MDP_COMPILED = _compiled
    return _MDP_COMPILED


_REFERENCE_QUAT_KEYS = (
    "root_quat",
    "xquat",
    "body_quat_w",
    "next_root_quat",
    "next_xquat",
    "next_body_quat_w",
)
_WXYZ_TO_XYZW = [1, 2, 3, 0]


def _convert_reference_quats_to_xyzw(reference: TensorDict) -> TensorDict:
    """Convert dataset quaternions from WXYZ to Isaac Lab 3.0's XYZW layout.

    Reference datasets (NPZ/zarr built by ImitationLearningTools) store
    quaternions scalar-first (w, x, y, z). Isaac Lab 3.0 switched the
    simulation state and all ``isaaclab.utils.math`` helpers to scalar-last
    (x, y, z, w), so every reference frame is converted once at the
    trajectory-manager boundary. Conversion is out-of-place so shared or lazy
    replay-buffer storage is never mutated.
    """
    next_td = reference.get("next", None)
    holders = (reference, next_td) if isinstance(next_td, TensorDict) else (reference,)
    for holder in holders:
        for key in _REFERENCE_QUAT_KEYS:
            quat = holder.get(key, None)
            if quat is None:
                continue
            holder.set(key, quat[..., _WXYZ_TO_XYZW])
    return reference


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


# Import the new manager and utilities
try:
    from iltools.datasets.lafan1.loader import Lafan1CsvLoader
    from iltools.datasets.loaders import load_dataset_loader
    from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
    from iltools.datasets.utils import make_rb_from
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules from iltools_datasets: {e}. Make sure ImitationLearningTools is installed."
    ) from e


def _load_loco_mujoco_loader() -> type[Any]:
    """Import the optional Loco-MuJoCo loader only when requested."""
    try:
        loader_cls = load_dataset_loader("loco_mujoco")
    except ImportError as exc:
        raise ImportError(
            "loader_type='loco_mujoco' requires the optional loco-mujoco "
            "dependencies. Install ImitationLearningTools with its "
            "`loco-mujoco` extra or select a different loader such as "
            "'lafan1_csv'."
        ) from exc
    return loader_cls


def _normalize_dataset_keys(raw: Any) -> list[str] | None:
    """Normalize `env.dataset_keys` into an explicit list of Zarr array names.

    A Hydra override such as ``env.dataset_keys=[qpos,qvel]`` can arrive as the
    literal string ``"[qpos,qvel]"`` rather than a parsed list. Iterating that
    string yields single characters, which surfaces much later as a confusing
    ``KeyError: "Key '[' not found"``, so normalize it here instead.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        names = [part.strip().strip("'\"") for part in text.split(",")]
    else:
        names = [str(part).strip().strip("'\"") for part in raw]
    names = [name for name in names if name]
    if not names:
        raise ValueError(
            "env.dataset_keys was provided but resolved to an empty selection."
        )
    return names


class ImitationRLEnvLegacy(ManagerBasedRLEnv):
    """
    Simplified RL environment for imitation learning with clean dataset interface.

    LEGACY env: byte-frozen reference implementation for the v0/v1 tasks.
    The v2 fork (:class:`~isaaclab_imitation.envs.imitation_rl_env_v2.ImitationRLEnv`)
    composes the ExpertDataPlane and the command planes instead of carrying
    this monolith; the two are equivalence-certified under the same cfg
    (``scripts/audit/certify_v2_env_equivalence.py``).

    Config attributes (cfg):
        dataset_path: str, path to Zarr dataset directory (or directory containing trajectories.zarr)
        reset_schedule: str, trajectory reset schedule ("random", "sequential", "round_robin", "custom")
        wrap_steps: bool, if True, wrap steps within trajectory (default: False)
        replay_only: bool, if True, ignore actions and force reference root/joint state each step
        loader_type: str, required if Zarr does not exist
            (supported: "loco_mujoco", "lafan1_csv", "lafan1")
        loader_kwargs: dict, required if Zarr does not exist (e.g., {"env_name": "UnitreeG1", "cfg": ...})
        reference_joint_names: list[str], joint names in reference data order
        target_joint_names: list[str], optional, joint names in target robot order (for mapping)
        datasets: str | list[str] | None, optional, dataset names to load from Zarr
        motions: str | list[str] | None, optional, motion names to load from Zarr
        trajectories: str | list[str] | None, optional, trajectory names to load from Zarr
        keys: str | list[str] | None, optional, keys to load from Zarr (default: all keys)
        refresh_zarr_dataset: bool, if True, delete existing zarr and rebuild it using the loader each run
        dataset_storage_device: str, torch device that holds the reference replay buffer
            ("cuda:0" by default). Set to "cpu" to use TorchRL's LazyMemmapStorage instead
            of a GPU-resident LazyTensorStorage, so a reference set larger than VRAM can be
            trained against. The trajectory manager already indexes on the storage device
            and copies each sampled batch to the compute device, so only throughput changes.
        dataset_storage_persist_dir: str | None, reusable directory for the CPU memmap buffer.
            Only used when dataset_storage_device is a CPU device. A matching build there is
            memory-mapped in milliseconds instead of refilled from Zarr, which otherwise costs
            hours for a reference set of this size.
        dataset_storage_persist_id: str | None, content identity for that buffer, making it
            relocatable so it can be built once and copied to a compute node.
        dataset_storage_persist_rebuild: bool, force a refill of dataset_storage_persist_dir.
        reference_start_frame: int, trajectory-local frame index used after each reset (default: 0)
        random_reset_step_min/random_reset_step_max: int, uniform random starting-frame range on
            reset (inclusive). Ignored when reset_start_mode selects another mode.
        reset_start_mode: str, starting-frame selection on reset -- "auto" (default, legacy:
            random when a [min, max] range is configured, else the fixed reference_start_frame),
            "fixed", "random", or "adaptive" (weighted by the SONIC failure sampler or by a custom
            callable attached as `cfg.adaptive_reset_weight_fn`). Trajectory selection always stays
            with reset_schedule; only random_reset_full_trajectory overrides both via SONIC.
        adaptive_reset_weight_fn: Callable|None, optional (trajectory_ranks, frame_steps) -> weights
            callable used by reset_start_mode="adaptive"; defaults to the SONIC failure sampler.
        visualize_reference_arrows: bool, if True show reference velocity/position/heading arrows and
            desired/current frame markers for root and tracked bodies (default: False)

    Example config:
        dataset_path = '/path/to/zarr'
        reset_schedule = 'random'  # or 'sequential', 'round_robin', 'custom'
        wrap_steps = False
        loader_type = 'lafan1_csv'  # or 'loco_mujoco' when its optional dependency is installed
        loader_kwargs = {'dataset': {'trajectories': {'lafan1_csv': [...]}}}
        reference_joint_names = ['left_hip_pitch_joint', ...]
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
        """Initialize the simplified legacy ImitationRLEnv.

        DEPRECATED (2026-08-01): this env class backs only the frozen v0/v1
        task ids (``-G1-v0`` / ``-G1-v1`` / ``-G1-Latent-v0`` / Strict pins
        and friends) for reproducibility. New work uses the flat v2 configs
        with the v2 env (``ImitationRLEnv``, ``Isaac-Imitation-G1-v2``).
        """
        warnings.warn(
            "ImitationRLEnvLegacy is DEPRECATED; it backs only the frozen "
            "v0/v1 task ids. Use Isaac-Imitation-G1-v2 (ImitationRLEnv, flat "
            "ImitationG1EnvCfg) for new work.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Get device
        device = cfg.sim.device
        num_envs = cfg.scene.num_envs

        # Isaac Lab 3.0's hydra integration applies `env.*` CLI overrides with a
        # plain setattr on the config (no `from_dict` round-trip), so a
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

        # Get dataset path and determine if we need to create it
        dataset_path = getattr(cfg, "dataset_path", None)
        loader_type = getattr(cfg, "loader_type", None)
        loader_kwargs = getattr(cfg, "loader_kwargs", {})
        refresh_zarr_dataset = bool(getattr(cfg, "refresh_zarr_dataset", False))
        if loader_type in ("lafan1_csv", "lafan1"):
            lafan_source_entries = self._lafan_source_entries_from_loader_kwargs(
                loader_kwargs
            )
            manifest_path = getattr(cfg, "lafan1_manifest_path", None)
            has_manifest_loader = (
                manifest_path is not None and len(lafan_source_entries) > 0
            )
            has_explicit_loader_setup = (
                dataset_path is not None and len(lafan_source_entries) > 0
            )
            if not has_manifest_loader and not has_explicit_loader_setup:
                raise ValueError(
                    "G1 LAFAN tracking tasks now require "
                    "`env.lafan1_manifest_path=/path/to/manifest.json` for normal use. "
                    "If you are configuring the env programmatically, provide explicit "
                    "`loader_kwargs.dataset.trajectories.lafan1_csv` entries and "
                    "`dataset_path` before env creation."
                )

        # Build or load the replay buffer and trajectory info
        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            # Check if it's a directory containing trajectories.zarr or the zarr itself
            if dataset_path.is_dir():
                zarr_path = dataset_path / "trajectories.zarr"
                if not zarr_path.exists():
                    zarr_path = dataset_path  # Assume the directory itself is the zarr
            else:
                zarr_path = dataset_path

            # For debugging, optionally force dataset refresh on every run.
            if refresh_zarr_dataset:
                if loader_type is None:
                    raise ValueError(
                        "refresh_zarr_dataset=True requires loader_type + loader_kwargs "
                        "so the zarr dataset can be rebuilt."
                    )
                if zarr_path.exists():
                    if zarr_path.is_dir():
                        shutil.rmtree(zarr_path)
                    else:
                        zarr_path.unlink()

            # If zarr doesn't exist and loader is provided, create it
            if not zarr_path.exists() and loader_type is not None:
                if loader_type == "loco_mujoco":
                    from omegaconf import DictConfig

                    loader_cfg = DictConfig(loader_kwargs)
                    loader_cls = _load_loco_mujoco_loader()
                    _ = loader_cls(
                        env_name=loader_kwargs["env_name"],
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                    )
                elif loader_type in ("lafan1_csv", "lafan1"):
                    from omegaconf import DictConfig

                    loader_cfg = DictConfig(loader_kwargs)
                    loader_build_kwargs = {
                        key: int(loader_kwargs[key])
                        for key in ("chunk_size", "shard_size")
                        if key in loader_kwargs and loader_kwargs[key] is not None
                    }
                    _ = Lafan1CsvLoader(
                        cfg=loader_cfg,
                        build_zarr_dataset=True,
                        zarr_path=str(zarr_path),
                        **loader_build_kwargs,
                    )
                else:
                    raise ValueError(
                        f"Unsupported loader_type: {loader_type}. "
                        "Supported loader types: loco_mujoco, lafan1_csv, lafan1."
                    )

            # Load replay buffer from Zarr
            datasets = getattr(cfg, "datasets", None)
            motions = getattr(cfg, "motions", None)
            traj_names = getattr(cfg, "trajectories", None)
            # `dataset_keys`, not `keys`: on a dict-like config object `cfg.keys`
            # resolves to the bound `keys()` method, so the old lookup could
            # never select a subset and silently loaded every array.
            keys = _normalize_dataset_keys(getattr(cfg, "dataset_keys", None))

            # The reference replay buffer normally lives in VRAM. A reference
            # set larger than the GPU (e.g. the 129,785-clip BONES-SEED tree,
            # about 135 GB of transitions) needs CPU storage instead;
            # `make_rb_from` then builds a LazyMemmapStorage, and
            # ParallelTrajectoryManager already indexes on the storage device
            # and copies each sampled batch to the compute device.
            storage_device = torch.device(
                str(getattr(cfg, "dataset_storage_device", "cuda:0"))
            )
            storage_persist_dir = getattr(cfg, "dataset_storage_persist_dir", None)
            rb, traj_info = make_rb_from(
                zarr_path=str(zarr_path),
                datasets=datasets,
                motions=motions,
                trajectories=traj_names,
                keys=keys,
                device=storage_device,
                persist_dir=storage_persist_dir,
                persist_id=getattr(cfg, "dataset_storage_persist_id", None)
                if storage_persist_dir is not None
                else None,
                persist_rebuild=bool(
                    getattr(cfg, "dataset_storage_persist_rebuild", False)
                ),
                verbose_tree=False,
                # Prefetch threads only help the CPU-storage path; on a
                # GPU-resident buffer the gather is already ~15 us.
                prefetch=3,
                # Pinning a >100 GB CPU buffer would exhaust pinned memory and
                # is unnecessary: samples are copied one small batch at a time.
                pin_memory=storage_device.type == "cuda",
            )
        else:
            raise ValueError(
                "Either dataset_path must be provided, or loader_type + loader_kwargs "
                "must be provided to create a new dataset."
            )

        # Map assignment_strategy to reset_schedule (for backward compatibility)
        assignment_strategy = getattr(cfg, "assignment_strategy", None)
        reset_schedule = getattr(cfg, "reset_schedule", None)
        if reset_schedule is None and assignment_strategy is not None:
            # Map old assignment_strategy to new reset_schedule
            mapping = {
                "random": ResetSchedule.RANDOM,
                "sequential": ResetSchedule.SEQUENTIAL,
                "round_robin": ResetSchedule.ROUND_ROBIN,
            }
            reset_schedule = mapping.get(assignment_strategy, ResetSchedule.RANDOM)
        if reset_schedule is None:
            reset_schedule = ResetSchedule.RANDOM
        # Get other config options
        wrap_steps = getattr(cfg, "wrap_steps", False)
        reference_start_frame = int(getattr(cfg, "reference_start_frame", 0))
        if reference_start_frame < 0:
            raise ValueError("reference_start_frame must be >= 0.")
        self._reference_start_frame = reference_start_frame
        self._latent_patch_past_steps = int(getattr(cfg, "latent_patch_past_steps", 0))
        self._latent_patch_future_steps = int(
            getattr(cfg, "latent_patch_future_steps", 0)
        )
        self._expert_anchor_body_name = str(
            getattr(cfg, "expert_anchor_body_name", "torso_link")
        ).strip()
        if not self._expert_anchor_body_name:
            raise ValueError("expert_anchor_body_name must be non-empty.")
        if self._latent_patch_past_steps < 0 or self._latent_patch_future_steps < 0:
            raise ValueError("latent patch window steps must be >= 0.")
        # The legacy macro window is consecutive-frame only. Refuse a strided
        # request here rather than silently serving the wrong cadence to a
        # skill encoder pretrained at that stride; the v2 surface owns it.
        if int(getattr(cfg, "expert_macro_frame_stride", 1)) != 1:
            raise ValueError(
                "expert_macro_frame_stride is a v2 surface feature; the legacy "
                "environment serves consecutive macro frames only."
            )
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
        reference_joint_names = list(getattr(cfg, "reference_joint_names", []))
        target_joint_names = list(getattr(cfg, "target_joint_names", []))
        dataset_joint_names = self._read_reference_joint_names_from_zarr(zarr_path)
        if len(dataset_joint_names) > 0:
            # The dataset (zarr) is authoritative for the reference joint order.
            # The zarr is written in canonical (articulation) order at build time,
            # so this normally equals the configured order; adopt it whenever it
            # differs so `reference -> target` remaps correctly for any source.
            if (
                len(reference_joint_names) == 0
                or reference_joint_names != dataset_joint_names
            ):
                reference_joint_names = dataset_joint_names

        first_transition = rb[0]
        first_qpos = first_transition.get("qpos")
        if first_qpos is not None:
            expected_reference_joint_dim = int(first_qpos.shape[-1]) - 7
            if len(reference_joint_names) != expected_reference_joint_dim:
                raise ValueError(
                    "reference_joint_names length mismatch with replay buffer qpos. "
                    f"Expected {expected_reference_joint_dim} joints from qpos, got "
                    f"{len(reference_joint_names)} reference names."
                )

        assert len(reference_joint_names) > 0 and len(target_joint_names) > 0, (
            "Reference and target joint names must have the length greater than 0"
        )
        self._reference_has_aligned_next = (
            first_transition.get("next_qpos") is not None
            and first_transition.get("next_qvel") is not None
        )

        # Initialize the trajectory manager
        self.trajectory_manager = ParallelTrajectoryManager(
            rb=rb,
            traj_info=traj_info,
            num_envs=num_envs,
            reset_schedule=reset_schedule,
            reset_start_step=reference_start_frame,
            wrap_steps=wrap_steps,
            device=device,
            reference_joint_names=reference_joint_names,
            target_joint_names=target_joint_names,
        )
        self._setup_adaptive_failure_reset_sampler(cfg)

        # Get initial reference data (this also initializes env assignments)
        self.current_expert_frame: TensorDict = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(advance=False)
        )
        self._current_reference_local_step = self.trajectory_manager.env_step.to(
            device=device, dtype=torch.long
        ).clone()
        # The reward_input observation group is opt-in (parked IPMD reward
        # estimation; see cfg.enable_reward_input_observations). The refresh
        # above already applied the toggle, so the group's presence on the
        # observation config decides whether the expert-side cache exists.
        self._reward_input_group_present = (
            getattr(getattr(cfg, "observations", None), "reward_input", None)
            is not None
        )
        self._build_reward_input_cache(device=torch.device(device))
        self._agent_latent_dim = int(getattr(cfg, "latent_command_dim", 16))
        self._agent_latent_command = torch.zeros(
            (num_envs, self._agent_latent_dim),
            device=device,
            dtype=torch.float32,
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
        # Anchor pose at each env's last command renewal, per anchor body:
        # published chunks are expressed in the publish-time anchor frame and
        # re-expressed into the current frame each step (odometry middleware).
        self._held_command_anchor_pose: dict[
            str, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Diagnostic command-pipeline trace (see _maybe_trace_command_window).
        # Inert unless ISAACLAB_COMMAND_TRACE names an output path.
        self._command_trace_enabled = bool(_os.environ.get("ISAACLAB_COMMAND_TRACE"))
        self._command_trace_records: list[dict[str, Any]] = []
        # Set once an external publisher calls capture_held_command_anchor();
        # from then on it owns the held reference pose and the automatic phase-0
        # recapture is suppressed. See capture_held_command_anchor().
        self._external_command_anchor_owner = False
        # Optional in-step planner publication hook (set_planner_command_provider).
        self._planner_command_provider: Any = None
        self._planner_command_provider_token: int | None = None

        # Store reference joint mapping
        self.reference_joint_names = reference_joint_names
        self.reference_body_names: list[str] = []
        self.reference_site_names: list[str] = []
        self._agent_trajectory_command_window_steps = (
            self._command_window_steps_from_offsets(
                self._latent_patch_past_steps,
                self._latent_patch_future_steps,
            )
        )
        self._agent_trajectory_command_terms = (
            self._allocate_agent_trajectory_command_terms(
                window_steps=self._agent_trajectory_command_window_steps,
                num_joints=len(self.reference_joint_names),
                num_ee_bodies=len(self._command_ee_body_names),
                num_keypoint_bodies=len(self._command_keypoint_body_names),
                device=torch.device(device),
            )
        )
        self._joint_mapping_cache: torch.Tensor | None = None
        self._reference_vel_vis_enabled = bool(
            getattr(
                cfg,
                "visualize_reference_arrows",
                getattr(cfg, "visualize_reference_velocity", False),
            )
        )
        self._reference_vel_marker: VisualizationMarkers | None = None
        self._reference_pos_delta_marker: VisualizationMarkers | None = None
        self._initial_heading_marker: VisualizationMarkers | None = None
        self._goal_root_frame_marker: VisualizationMarkers | None = None
        self._current_root_frame_marker: VisualizationMarkers | None = None
        self._goal_body_frame_markers: list[VisualizationMarkers] = []
        self._current_body_frame_markers: list[VisualizationMarkers] = []
        self._vis_reference_body_ids: torch.Tensor | None = None
        self._vis_robot_body_ids: torch.Tensor | None = None
        self._vis_body_names: list[str] = []
        self._last_tracked_root_pos_w = torch.zeros((num_envs, 3), device=device)
        self._last_tracked_root_pos_valid = torch.zeros(
            (num_envs,), device=device, dtype=torch.bool
        )
        self.replay_reference = getattr(cfg, "replay_reference", False)
        self.replay_only = getattr(cfg, "replay_only", False)
        if self.replay_only and not self.replay_reference:
            self.replay_reference = True
        self._reference_replay_targets_enabled = False
        self._reference_replay_source_env_ids: torch.Tensor | None = None
        self._reference_replay_target_env_ids: torch.Tensor | None = None
        self._expert_sampler_warned_action_fallback = False
        self._expert_sampler_warned_unknown_terms: set[str] = set()

        self._load_reference_metadata(zarr_path)

        # Initialize parent class
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self._align_reference_target_joints_to_articulation()
        self._expert_env_origins = self.scene.env_origins.clone()
        self._expert_default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self._expert_default_joint_vel = self.robot.data.default_joint_vel.torch.clone()
        self._finalize_reference_body_names()
        self._initialize_mdp_fast_paths()
        self._setup_reference_velocity_visualizer()
        self._initialize_causal_planner_history()
        self._initialize_mpjpe_metric()

    def _align_reference_target_joints_to_articulation(self) -> None:
        """Retarget the trajectory manager to the live articulation joint order.

        ``target_joint_names`` in the config describes one backend's joint
        enumeration (PhysX is breadth-first). Other physics backends (Newton
        is depth-first per limb) enumerate the same joints in a different
        order, so the reference->target remap must be rebuilt against the
        actual robot once the scene exists — otherwise reference joint
        targets are silently scattered across the wrong joints.
        """
        tm = self.trajectory_manager
        robot_joint_names = list(self.robot.joint_names)
        if list(tm.target_joint_names) == robot_joint_names:
            return
        if sorted(tm.target_joint_names) != sorted(robot_joint_names):
            raise RuntimeError(
                "Configured target_joint_names and articulation joint names are "
                "different sets; cannot retarget the reference. Difference: "
                f"{sorted(set(tm.target_joint_names) ^ set(robot_joint_names))}"
            )
        logger.warning(
            "Articulation joint order differs from configured target_joint_names "
            "(physics-backend-specific enumeration); rebuilding the "
            "reference->target joint remap for %d joints.",
            len(robot_joint_names),
        )
        from iltools.datasets.utils import _map_reference_to_target

        tm.target_joint_names = robot_joint_names
        tm.ref_to_target_map, tm.target_to_ref_map = _map_reference_to_target(
            tm.reference_joint_names,
            tm.target_joint_names,
            tm._state_device,
        )
        tm.target_mask = torch.zeros(
            len(robot_joint_names), dtype=torch.bool, device=tm._state_device
        )
        tm.target_mask[tm.ref_to_target_map] = True
        # Refresh everything that already captured target-ordered joint data.
        self.current_expert_frame = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(advance=False)
        )
        self._build_reward_input_cache(device=torch.device(self.device))

    @staticmethod
    def _read_reference_joint_names_from_zarr(zarr_path: Path) -> list[str]:
        """Read reference joint names from zarr metadata if available."""
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception:
            return []

        try:
            for key in list(root.group_keys()):  # type: ignore[attr-defined]
                group = root[key]
                joint_names = group.attrs.get("joint_names", None)
                if joint_names is not None:
                    return list(joint_names)
        except Exception:
            return []

        return []

    def _load_reference_metadata(self, zarr_path: Path) -> None:
        """Load reference body/site names from zarr metadata if available."""
        try:
            root = zarr.open(str(zarr_path), mode="r")
        except Exception:
            return

        dataset_group = None
        try:
            group_keys = list(root.group_keys())  # type: ignore[attr-defined]
            for key in group_keys:
                group = root[key]
                if "body_names" in group.attrs:
                    dataset_group = group
                    break
        except Exception:
            dataset_group = None

        if dataset_group is None:
            return

        body_names = dataset_group.attrs.get("body_names", [])
        site_names = dataset_group.attrs.get("site_names", [])
        self.reference_body_names = list(body_names) if body_names is not None else []
        self.reference_site_names = list(site_names) if site_names is not None else []

    def _gather_action_term_parameter(
        self,
        value: torch.Tensor | float,
        *,
        env_ids: torch.Tensor,
        template: torch.Tensor,
    ) -> torch.Tensor:
        """Gather an action-term parameter for the sampled env ids."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 2:
                return value.index_select(0, env_ids).to(
                    device=template.device, dtype=template.dtype
                )
            return value.to(device=template.device, dtype=template.dtype)
        return torch.full_like(template, float(value))

    @staticmethod
    def _resolve_offline_static_action_vector(
        value: torch.Tensor | float,
        *,
        name: str,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve an env-invariant action parameter for offline dataset mapping."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=device, dtype=torch.float32)
            if tensor.ndim == 2:
                reference = tensor[0]
                if tensor.shape[0] > 1 and not torch.allclose(
                    tensor, reference.unsqueeze(0)
                ):
                    raise ValueError(
                        f"offline_dataset mapper requires env-invariant {name}."
                    )
                tensor = reference
            elif tensor.ndim != 1:
                raise ValueError(f"Unexpected {name} shape {tuple(tensor.shape)}.")
        else:
            tensor = torch.full((width,), float(value), device=device)
        if tuple(tensor.shape) != (width,):
            raise ValueError(
                f"{name} must have shape ({width},), got {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _resolve_offline_action_vector_pool(
        value: torch.Tensor | float,
        *,
        name: str,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve one or more env-indexed action vectors for offline mapping."""
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(device=device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim != 2:
                raise ValueError(f"Unexpected {name} shape {tuple(tensor.shape)}.")
        else:
            tensor = torch.full((1, width), float(value), device=device)
        if tensor.shape[0] <= 0 or tuple(tensor.shape[1:]) != (width,):
            raise ValueError(
                f"{name} must have shape (N, {width}), got {tuple(tensor.shape)}."
            )
        return tensor

    def get_offline_dataset_mapper_params(self) -> dict[str, Any]:
        """Return G1 action inversion constants for offline TensorDict mapping."""
        action_term = self.action_manager.get_term("joint_pos")
        if not isinstance(action_term, JointPositionAction):
            raise TypeError(
                "offline_dataset G1 WBT mapper requires JointPositionAction."
            )

        action_joint_names = list(action_term._joint_names)
        action_width = len(action_joint_names)
        if action_width != 29:
            raise ValueError(
                "offline_dataset unitree_g1_wbt_29dof mapper requires 29 action "
                f"joints, got {action_width}."
            )
        self.robot.find_joints(action_joint_names, preserve_order=True)
        action_offset_pool = self._resolve_offline_action_vector_pool(
            action_term._offset,
            name="JointPositionAction offset",
            width=action_width,
            device=self.device,
        )
        action_scale = self._resolve_offline_static_action_vector(
            action_term._scale,
            name="JointPositionAction scale",
            width=action_width,
            device=self.device,
        )
        if torch.any(action_scale.abs() <= 1.0e-8):
            raise ValueError("JointPositionAction scale must not contain zeros.")
        default_root_height = float(
            self.robot.data.default_root_state.torch[0, 2].detach().cpu().item()
        )
        return {
            "default_joint_pos": action_offset_pool[0].cpu().tolist(),
            "default_joint_pos_pool": action_offset_pool.cpu().tolist(),
            "action_scale": action_scale.cpu().tolist(),
            "default_root_height": default_root_height,
            "align_root_z_to_default": True,
            "dataset_joint_names": list(UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES),
            "target_joint_names": action_joint_names,
            "joint_names": action_joint_names,
        }

    @staticmethod
    def _compute_rollout_state_alignment_metrics(
        actual_state: torch.Tensor,
        reference_state: torch.Tensor,
        *,
        prefix: str,
    ) -> dict[str, float]:
        """Aggregate next-state tracking metrics against the aligned reference transition."""
        if actual_state.ndim == 1:
            actual_state = actual_state.unsqueeze(0)
        if reference_state.ndim == 1:
            reference_state = reference_state.unsqueeze(0)

        actual_state = actual_state.detach().to(dtype=torch.float32)
        reference_state = reference_state.detach().to(dtype=torch.float32)
        reference_nan_frac = float(
            (~torch.isfinite(reference_state)).float().mean().item()
        )

        actual_state = torch.nan_to_num(actual_state, nan=0.0, posinf=0.0, neginf=0.0)
        reference_state = torch.nan_to_num(
            reference_state, nan=0.0, posinf=0.0, neginf=0.0
        )
        diff = actual_state - reference_state
        per_env_abs_mean = diff.abs().reshape(diff.shape[0], -1).mean(dim=-1)
        per_env_mse = diff.square().reshape(diff.shape[0], -1).mean(dim=-1)
        return {
            f"{prefix}_mae": float(per_env_abs_mean.mean().item()),
            f"{prefix}_mse": float(per_env_mse.mean().item()),
            f"{prefix}_rmse": float(per_env_mse.sqrt().mean().item()),
            f"{prefix}_max_abs": float(
                diff.abs().reshape(diff.shape[0], -1).amax(dim=-1).mean().item()
            ),
            f"{prefix}_reference_nan_frac": reference_nan_frac,
        }

    def _initialize_mpjpe_metric(self) -> None:
        """Resolve the bodies used for the root-relative MPJPE training metric.

        This exists because a metric cannot be expressed as a ``RewTerm`` with
        ``weight=0.0``: :meth:`RewardManager.compute` skips zero-weight terms
        without calling them, so such a term logs a constant zero.
        """
        body_names = list(getattr(self.cfg, "mpjpe_metric_body_names", []) or [])
        self._mpjpe_metric_body_names: list[str] = []
        self._mpjpe_metric_body_ids: torch.Tensor | None = None
        self._mpjpe_metric_sum: torch.Tensor | None = None
        self._mpjpe_metric_count: torch.Tensor | None = None
        if not body_names:
            return

        missing = [
            name for name in body_names if name not in set(self.reference_body_names)
        ]
        if missing:
            raise ValueError(
                "mpjpe_metric_body_names contains bodies absent from the "
                f"reference: {missing}. The metric compares robot bodies against "
                "reference bodies of the same name, so every entry must exist in "
                "both."
            )
        body_ids, resolved = self.robot.find_bodies(body_names, preserve_order=True)
        if list(resolved) != body_names:
            raise RuntimeError(
                "Could not resolve the MPJPE metric bodies in order: "
                f"expected={body_names}, got={list(resolved)}."
            )
        self._mpjpe_metric_body_names = body_names
        self._mpjpe_metric_body_ids = torch.as_tensor(
            body_ids, dtype=torch.long, device=self.device
        )
        self._mpjpe_metric_sum = torch.zeros(self.num_envs, device=self.device)
        self._mpjpe_metric_count = torch.zeros(self.num_envs, device=self.device)

    def _motion_command_owns_metrics(self) -> bool:
        """True when a manager-based ``motion`` command term owns the metrics.

        The v2 surface's ``MotionCommand`` term logs ``Metrics/motion/...``
        natively (delegating to :meth:`_compute_mpjpe_metric`), so the env-side
        ``Metrics/mpjpe_mm*`` channel is skipped to avoid logging the same
        quantity twice. Tasks without the term (v0/v1) keep the env channel.
        """
        command_manager = getattr(self, "command_manager", None)
        if command_manager is None:
            return False
        return "motion" in getattr(command_manager, "active_terms", ())

    def _compute_mpjpe_metric(self) -> torch.Tensor | None:
        """Per-environment root-relative MPJPE in metres.

        Mirrors ``mdp.mpjpe_relative_body_pos_m`` and the closed-loop
        evaluators: both sides are expressed relative to their own root, so the
        value measures pose error rather than global drift.

        Kept in metres to match those two, matching ``tracking_mpjpe_m``; the
        conversion to millimetres happens once at the logging boundary.

        Note that only root *position* is subtracted, not root *orientation*, so
        a rotated root rigidly rotates every body within the root-relative frame
        and contributes an error of roughly (distance from root) x (rotation) per
        body. That is why this is non-zero on the first frame of an episode: the
        ``reset_reference_state`` event perturbs the initial root orientation by
        up to 0.1/0.1/0.2 rad, which alone measures about 39 mm on the G1's
        14-body set, with a further 6 mm from the +/-0.1 rad joint noise.
        Measured with all reset randomization disabled the value is exactly
        0.00 mm, so there is no systematic reference-versus-URDF body-frame
        offset underneath it.
        """
        if self._mpjpe_metric_body_ids is None:
            return None
        robot_pos_w = self._get_robot_body_pose_w_fast(self._mpjpe_metric_body_ids)[0]
        reference_pos_w = self._get_reference_body_pose_w_fast(
            self._mpjpe_metric_body_names
        )[0]
        robot_root_w = self.robot.data.root_state_w.torch[:, :3]
        reference_root_w = self._get_reference_root_state_w_fast()[0]
        robot_relative = robot_pos_w - robot_root_w[:, None, :]
        reference_relative = reference_pos_w - reference_root_w[:, None, :]
        return torch.linalg.vector_norm(
            robot_relative - reference_relative, dim=-1
        ).mean(dim=-1)

    def _accumulate_mpjpe_metric(
        self, exclude_env_ids: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Accumulate the episode sum and return the instantaneous log entry.

        Reported in millimetres, the unit the closed-loop evaluators and every
        paper aggregator use for ``tracking_mpjpe_mm``.

        Args:
            exclude_env_ids: Environments to skip in the *episode* accumulator
                this call (their instantaneous value is still included in the
                returned/logged mean). This is for envs that were reset earlier
                in the same ``step()`` call: by this point their state is the
                fresh post-reset pose, not something the policy produced, and
                their terminal-frame contribution to the episode that just
                ended was already folded in by :meth:`_reset_idx` before the
                reset write. Accumulating here too would double-count the
                terminal frame into the wrong (new) episode.
        """
        if self._motion_command_owns_metrics():
            return {}
        mpjpe = self._compute_mpjpe_metric()
        if mpjpe is None:
            return {}
        assert self._mpjpe_metric_sum is not None
        assert self._mpjpe_metric_count is not None
        if exclude_env_ids is not None and exclude_env_ids.numel() > 0:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            active.index_fill_(0, exclude_env_ids, False)
            self._mpjpe_metric_sum[active] += mpjpe[active]
            self._mpjpe_metric_count[active] += 1.0
        else:
            self._mpjpe_metric_sum += mpjpe
            self._mpjpe_metric_count += 1.0
        return {"Metrics/mpjpe_mm": float(mpjpe.mean().item()) * _METRES_TO_MM}

    def _accumulate_terminal_mpjpe_metric(self, env_ids: torch.Tensor) -> None:
        """Fold the pre-reset terminal frame into the ending episode's sum.

        Must run before any trajectory reassignment or reset write for
        ``env_ids`` (i.e. at the very top of :meth:`_reset_idx`): the robot's
        physical state is still the terminal one from the step that just
        triggered the reset, and the tracked reference is still the one that
        terminating episode was scored against. Without this, the terminal
        transition's error is never counted in any episode -- by the time
        :meth:`_emit_mpjpe_episode_metric` runs, the accumulator holds every
        frame except the last one.
        """
        if self._mpjpe_metric_sum is None or env_ids.numel() == 0:
            return
        if self._motion_command_owns_metrics():
            return
        mpjpe = self._compute_mpjpe_metric()
        if mpjpe is None:
            return
        assert self._mpjpe_metric_count is not None
        self._mpjpe_metric_sum.index_add_(0, env_ids, mpjpe.index_select(0, env_ids))
        self._mpjpe_metric_count.index_add_(
            0, env_ids, torch.ones(env_ids.numel(), device=self.device)
        )

    def _emit_mpjpe_episode_metric(self, env_ids: torch.Tensor) -> None:
        """Log the completed episodes' mean MPJPE, then clear their accumulators.

        Emitted per episode rather than per step so the value is comparable
        with the evaluators, which average over a whole rollout. The
        instantaneous key is logged every step as well, so the curve never has
        gaps on steps where nothing reset.
        """
        if self._mpjpe_metric_sum is None or env_ids.numel() == 0:
            return
        if self._motion_command_owns_metrics():
            return
        counts = self._mpjpe_metric_count.index_select(0, env_ids)
        valid = counts > 0
        if bool(valid.any()):
            sums = self._mpjpe_metric_sum.index_select(0, env_ids)
            episode_mean = (sums[valid] / counts[valid]).mean()
            self.extras.setdefault("log", {})["Metrics/mpjpe_mm_per_episode"] = (
                float(episode_mean.item()) * _METRES_TO_MM
            )
        self._mpjpe_metric_sum.index_fill_(0, env_ids, 0.0)
        self._mpjpe_metric_count.index_fill_(0, env_ids, 0.0)

    def _compute_rollout_reference_state_log(self) -> dict[str, float]:
        """Compare the post-step robot state against the aligned reference next state."""
        if self.current_expert_frame is None:
            return {}

        next_joint_pos = self.current_expert_frame.get(("next", "joint_pos"))
        next_joint_vel = self.current_expert_frame.get(("next", "joint_vel"))
        if next_joint_pos is None or next_joint_vel is None:
            return {}

        metrics = self._compute_rollout_state_alignment_metrics(
            self.robot.data.joint_pos.torch,
            next_joint_pos.to(device=self.device, dtype=torch.float32),
            prefix="rollout_state/joint_pos",
        )
        metrics.update(
            self._compute_rollout_state_alignment_metrics(
                self.robot.data.joint_vel.torch,
                next_joint_vel.to(device=self.device, dtype=torch.float32),
                prefix="rollout_state/joint_vel",
            )
        )
        return metrics

    def _finalize_reference_body_names(self) -> None:
        """Improve reference body-name mapping for datasets that only provide generic names."""
        ref_body_pos = self.current_expert_frame.get("xpos")
        if ref_body_pos is None:
            ref_body_pos = self.current_expert_frame.get("body_pos_w")
        if ref_body_pos is None or ref_body_pos.ndim < 3:
            return

        num_reference_bodies = int(ref_body_pos.shape[1])
        robot_body_names = list(self.robot.body_names)

        has_generic_names = len(self.reference_body_names) == 0 or all(
            name.startswith("body_") and name[5:].isdigit()
            for name in self.reference_body_names
        )
        if not has_generic_names:
            return
        # Prefer the config-declared dataset body order: the live robot's body
        # enumeration is physics-backend-specific (PhysX is breadth-first,
        # Newton is depth-first), while the recorded body arrays have one
        # fixed order.
        cfg_body_names = list(getattr(self.cfg, "reference_body_names", []) or [])
        if len(cfg_body_names) >= num_reference_bodies:
            self.reference_body_names = cfg_body_names[:num_reference_bodies]
        elif len(robot_body_names) >= num_reference_bodies:
            # Legacy fallback: assumes the dataset was recorded with the same
            # backend (and thus body order) as the running simulation.
            logger.warning(
                "Reference dataset has no body-name metadata and the env cfg "
                "declares no reference_body_names; falling back to the live "
                "robot's body order, which is only correct if the dataset was "
                "recorded with the same physics backend."
            )
            self.reference_body_names = robot_body_names[:num_reference_bodies]

    @staticmethod
    def _normalize_body_name_for_matching(name: str) -> str:
        """Normalize body names for tolerant cross-dataset matching."""
        lowered = name.lower()
        if lowered.endswith("_link"):
            lowered = lowered[:-5]
        return lowered

    def _initialize_mdp_fast_paths(self) -> None:
        if not hasattr(self, "robot"):
            self.robot = self.scene["robot"]
        self._finalize_reference_body_names()
        self._mdp_cache_step = -1
        self._mdp_align_quat: torch.Tensor | None = None
        self._mdp_align_pos: torch.Tensor | None = None
        self._mdp_reference_root_cache: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._mdp_reference_cvel_cache: torch.Tensor | None = None
        self._mdp_expert_motion_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_expert_window_obs_cache: dict[
            tuple[int, int, str, object, tuple[str, ...]], dict[str, torch.Tensor]
        ] = {}
        self._mdp_expert_goal_obs_cache: dict[
            tuple[int, str, object], dict[str, torch.Tensor]
        ] = {}
        self._mdp_reference_body_id_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self._mdp_reference_body_pose_cache: dict[
            tuple[str, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_reference_body_velocity_cache: dict[
            tuple[str, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_anchor_id_cache: dict[str, int] = {}
        self._mdp_robot_anchor_state_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_pose_w_cache: dict[
            object, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_velocity_w_cache: dict[
            object, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_robot_body_anchor_frame_cache: dict[
            tuple[int, object], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._mdp_body_name_to_id = {
            name: idx for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_body_name_to_id_lower = {
            name.lower(): idx for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_body_name_to_id_normalized = {
            self._normalize_body_name_for_matching(name): idx
            for idx, name in enumerate(self.robot.body_names)
        }
        self._mdp_reference_body_name_to_id = {
            name: idx for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_reference_body_name_to_id_lower = {
            name.lower(): idx for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_reference_body_name_to_id_normalized = {
            self._normalize_body_name_for_matching(name): idx
            for idx, name in enumerate(self.reference_body_names)
        }
        self._mdp_body_id_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_joint_id_tensor_cache: dict[tuple[int, ...], torch.Tensor] = {}
        self._mdp_all_body_ids_key = tuple(range(len(self.robot.body_names)))
        self._mdp_reset_pose_bounds: torch.Tensor | None = None
        self._mdp_reset_velocity_bounds: torch.Tensor | None = None

        reference = self.current_expert_frame
        self._mdp_reference_body_pos_key = (
            "xpos" if "xpos" in reference else "body_pos_w"
        )
        self._mdp_reference_body_quat_key = (
            "xquat" if "xquat" in reference else "body_quat_w"
        )
        self._mdp_reference_body_count = int(
            reference[self._mdp_reference_body_pos_key].shape[1]
        )
        self._mdp_reset_root_pose_source = (
            "root" if "root_pos" in reference and "root_quat" in reference else "body"
        )
        if "root_lin_vel" in reference and "root_ang_vel" in reference:
            self._mdp_reset_root_velocity_source = "root"
        elif "body_lin_vel_w" in reference and "body_ang_vel_w" in reference:
            self._mdp_reset_root_velocity_source = "body"
        else:
            self._mdp_reset_root_velocity_source = "zeros"

    def _ensure_mdp_fast_paths(self) -> None:
        if hasattr(self, "_mdp_cache_step"):
            return
        self._initialize_mdp_fast_paths()

    def _invalidate_mdp_cache(self) -> None:
        self._ensure_mdp_fast_paths()
        self._mdp_cache_step = -1
        self._mdp_align_quat = None
        self._mdp_align_pos = None
        self._mdp_reference_root_cache = None
        self._mdp_reference_cvel_cache = None
        self._mdp_expert_motion_cache.clear()
        self._mdp_expert_window_obs_cache.clear()
        self._mdp_expert_goal_obs_cache.clear()
        self._mdp_reference_body_pose_cache.clear()
        self._mdp_reference_body_velocity_cache.clear()
        self._mdp_robot_anchor_state_cache.clear()
        self._mdp_robot_body_pose_w_cache.clear()
        self._mdp_robot_body_velocity_w_cache.clear()
        self._mdp_robot_body_anchor_frame_cache.clear()

    def _ensure_mdp_step_cache(self) -> None:
        self._ensure_mdp_fast_paths()
        if (
            self._mdp_cache_step == self.common_step_counter
            and self._mdp_align_quat is not None
        ):
            return
        align_quat, align_pos = self._get_reference_alignment_transform()
        self._mdp_align_quat = align_quat
        self._mdp_align_pos = align_pos
        self._mdp_reference_root_cache = None
        self._mdp_reference_cvel_cache = None
        self._mdp_expert_motion_cache.clear()
        self._mdp_expert_window_obs_cache.clear()
        self._mdp_expert_goal_obs_cache.clear()
        self._mdp_reference_body_pose_cache.clear()
        self._mdp_reference_body_velocity_cache.clear()
        self._mdp_robot_anchor_state_cache.clear()
        self._mdp_robot_body_pose_w_cache.clear()
        self._mdp_robot_body_velocity_w_cache.clear()
        self._mdp_robot_body_anchor_frame_cache.clear()
        self._mdp_cache_step = self.common_step_counter

    def _get_reference_alignment_fast(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        return self._mdp_align_quat, self._mdp_align_pos  # type: ignore[return-value]

    def _get_body_ids_tensor_fast(
        self, body_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        self._ensure_mdp_fast_paths()
        if isinstance(body_ids, slice):
            return body_ids
        key = tuple(int(body_id) for body_id in body_ids)
        body_ids_t = self._mdp_body_id_tensor_cache.get(key)
        if body_ids_t is None:
            body_ids_t = torch.tensor(key, dtype=torch.long, device=self.device)
            self._mdp_body_id_tensor_cache[key] = body_ids_t
        return body_ids_t

    def _get_joint_ids_tensor_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor | slice:
        self._ensure_mdp_fast_paths()
        if isinstance(joint_ids, slice):
            return joint_ids
        key = tuple(int(joint_id) for joint_id in joint_ids)
        joint_ids_t = self._mdp_joint_id_tensor_cache.get(key)
        if joint_ids_t is None:
            joint_ids_t = torch.tensor(key, dtype=torch.long, device=self.device)
            self._mdp_joint_id_tensor_cache[key] = joint_ids_t
        return joint_ids_t

    def _get_reference_root_state_w_fast(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        if self._mdp_reference_root_cache is None:
            compiled = _get_mdp_compiled_module()
            reference = self.current_expert_frame
            root_pos_w, root_quat_w = compiled.transform_root_pose_to_world(
                self._mdp_align_quat,
                self._mdp_align_pos,
                reference["root_pos"],
                reference["root_quat"],
            )
            root_lin_vel_w, root_ang_vel_w = compiled.transform_root_velocity_to_world(
                self._mdp_align_quat,
                reference["root_lin_vel"],
                reference["root_ang_vel"],
            )
            self._mdp_reference_root_cache = (
                root_pos_w,
                root_quat_w,
                root_lin_vel_w,
                root_ang_vel_w,
            )
        return self._mdp_reference_root_cache

    def _get_reference_cvel_fast(self) -> torch.Tensor:
        self._ensure_mdp_step_cache()
        if self._mdp_reference_cvel_cache is None:
            reference = self.current_expert_frame
            self._mdp_reference_cvel_cache = torch.cat(
                [reference["body_ang_vel_w"], reference["body_lin_vel_w"]], dim=-1
            )
        return self._mdp_reference_cvel_cache

    def _get_reference_body_ids_fast(
        self, reference_body_names: Sequence[str]
    ) -> torch.Tensor:
        self._ensure_mdp_fast_paths()
        cache_key = tuple(reference_body_names)
        body_ids = self._mdp_reference_body_id_cache.get(cache_key)
        if body_ids is not None:
            return body_ids

        ref_indices: list[int] = []
        for name in cache_key:
            body_id = self._mdp_reference_body_name_to_id.get(name)
            if body_id is None:
                body_id = self._mdp_reference_body_name_to_id_lower.get(name.lower())
            if body_id is None:
                body_id = self._mdp_reference_body_name_to_id_normalized.get(
                    self._normalize_body_name_for_matching(name)
                )
            if body_id is None:
                body_id = self._mdp_body_name_to_id.get(name)
            if body_id is None:
                body_id = self._mdp_body_name_to_id_lower.get(name.lower())
            if body_id is None:
                body_id = self._mdp_body_name_to_id_normalized.get(
                    self._normalize_body_name_for_matching(name)
                )
            if body_id is not None and body_id >= self._mdp_reference_body_count:
                body_id = None
            if body_id is None:
                raise KeyError(
                    f"Reference body '{name}' not found in reference metadata."
                )
            ref_indices.append(body_id)

        body_ids = torch.tensor(ref_indices, dtype=torch.long, device=self.device)
        self._mdp_reference_body_id_cache[cache_key] = body_ids
        return body_ids

    def _get_robot_body_ids_by_name_fast(
        self, body_names: Sequence[str]
    ) -> torch.Tensor:
        """Resolve live robot body names with the dataset-tolerant name rules."""
        self._ensure_mdp_fast_paths()
        body_ids: list[int] = []
        for name in body_names:
            body_id = self._mdp_body_name_to_id.get(name)
            if body_id is None:
                body_id = self._mdp_body_name_to_id_lower.get(name.lower())
            if body_id is None:
                body_id = self._mdp_body_name_to_id_normalized.get(
                    self._normalize_body_name_for_matching(name)
                )
            if body_id is None:
                raise KeyError(f"Robot body {name!r} not found in live articulation.")
            body_ids.append(int(body_id))
        resolved = self._get_body_ids_tensor_fast(body_ids)
        if isinstance(resolved, slice):
            raise RuntimeError("Named robot body lookup unexpectedly returned a slice.")
        return resolved

    def _get_reference_body_pose_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        cache_key = tuple(reference_body_names)
        body_pose = self._mdp_reference_body_pose_cache.get(cache_key)
        if body_pose is None:
            compiled = _get_mdp_compiled_module()
            ref_body_ids = self._get_reference_body_ids_fast(cache_key)
            reference = self.current_expert_frame
            ref_pos = reference[self._mdp_reference_body_pos_key].index_select(
                1, ref_body_ids
            )
            ref_quat = reference[self._mdp_reference_body_quat_key].index_select(
                1, ref_body_ids
            )
            body_pose = compiled.transform_body_pose_to_world(
                self._mdp_align_quat,
                self._mdp_align_pos,
                ref_pos,
                ref_quat,
            )
            self._mdp_reference_body_pose_cache[cache_key] = body_pose
        return body_pose

    def _get_reference_body_velocity_w_fast(
        self, reference_body_names: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        cache_key = tuple(reference_body_names)
        body_velocity = self._mdp_reference_body_velocity_cache.get(cache_key)
        if body_velocity is None:
            compiled = _get_mdp_compiled_module()
            ref_body_ids = self._get_reference_body_ids_fast(cache_key)
            ref_cvel = self._get_reference_cvel_fast().index_select(1, ref_body_ids)
            body_velocity = compiled.transform_body_velocity_to_world(
                self._mdp_align_quat, ref_cvel
            )
            self._mdp_reference_body_velocity_cache[cache_key] = body_velocity
        return body_velocity

    def _get_robot_anchor_body_id_fast(self, anchor_body_name: str) -> int:
        self._ensure_mdp_fast_paths()
        anchor_body_id = self._mdp_robot_anchor_id_cache.get(anchor_body_name)
        if anchor_body_id is None:
            anchor_body_id = self._mdp_body_name_to_id.get(anchor_body_name)
            if anchor_body_id is None:
                anchor_body_id = self._mdp_body_name_to_id_lower.get(
                    anchor_body_name.lower()
                )
            if anchor_body_id is None:
                anchor_body_id = self._mdp_body_name_to_id_normalized[
                    self._normalize_body_name_for_matching(anchor_body_name)
                ]
            self._mdp_robot_anchor_id_cache[anchor_body_name] = anchor_body_id
        return anchor_body_id

    def _body_ids_cache_key(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> object:
        if isinstance(body_ids, slice):
            return self._mdp_all_body_ids_key
        if isinstance(body_ids, torch.Tensor):
            if body_ids.device.type == "cpu":
                return tuple(int(body_id) for body_id in body_ids.tolist())
            return ("tensor", int(body_ids.data_ptr()), int(body_ids.numel()))
        return tuple(int(body_id) for body_id in body_ids)

    def _get_robot_anchor_state_w_fast(
        self, anchor_body_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        anchor_body_id = self._get_robot_anchor_body_id_fast(anchor_body_name)
        anchor_state = self._mdp_robot_anchor_state_cache.get(anchor_body_id)
        if anchor_state is None:
            anchor_state = (
                self.robot.data.body_pos_w.torch[:, anchor_body_id],
                self.robot.data.body_quat_w.torch[:, anchor_body_id],
            )
            self._mdp_robot_anchor_state_cache[anchor_body_id] = anchor_state
        return anchor_state

    def _get_robot_body_pose_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        body_ids_key = self._body_ids_cache_key(body_ids)
        body_pose = self._mdp_robot_body_pose_w_cache.get(body_ids_key)
        if body_pose is not None:
            return body_pose
        body_ids_t = self._get_body_ids_tensor_fast(body_ids)
        if isinstance(body_ids_t, slice):
            body_pose = (
                self.robot.data.body_pos_w.torch,
                self.robot.data.body_quat_w.torch,
            )
        else:
            body_pose = (
                self.robot.data.body_pos_w.torch.index_select(1, body_ids_t),
                self.robot.data.body_quat_w.torch.index_select(1, body_ids_t),
            )
        self._mdp_robot_body_pose_w_cache[body_ids_key] = body_pose
        return body_pose

    def _get_robot_body_velocity_w_fast(
        self, body_ids: Sequence[int] | torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        body_ids_key = self._body_ids_cache_key(body_ids)
        body_velocity = self._mdp_robot_body_velocity_w_cache.get(body_ids_key)
        if body_velocity is not None:
            return body_velocity
        body_ids_t = self._get_body_ids_tensor_fast(body_ids)
        if isinstance(body_ids_t, slice):
            body_velocity = (
                self.robot.data.body_ang_vel_w.torch,
                self.robot.data.body_lin_vel_w.torch,
            )
        else:
            body_velocity = (
                self.robot.data.body_ang_vel_w.torch.index_select(1, body_ids_t),
                self.robot.data.body_lin_vel_w.torch.index_select(1, body_ids_t),
            )
        self._mdp_robot_body_velocity_w_cache[body_ids_key] = body_velocity
        return body_velocity

    def _get_robot_body_state_in_anchor_frame_fast(
        self,
        body_ids: Sequence[int] | torch.Tensor | slice,
        anchor_body_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_mdp_step_cache()
        anchor_body_id = self._get_robot_anchor_body_id_fast(anchor_body_name)
        body_ids_key = self._body_ids_cache_key(body_ids)
        cache_key = (anchor_body_id, body_ids_key)
        body_state = self._mdp_robot_body_anchor_frame_cache.get(cache_key)
        if body_state is not None:
            return body_state

        compiled = _get_mdp_compiled_module()
        robot_anchor_pos_w, robot_anchor_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        body_pos_w, body_quat_w = self._get_robot_body_pose_w_fast(body_ids)
        body_state = compiled.body_pose_in_anchor_frame(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            body_pos_w,
            body_quat_w,
        )
        self._mdp_robot_body_anchor_frame_cache[cache_key] = body_state
        return body_state

    def get_expert_motion_qpos_command(
        self, joint_ids: Sequence[int] | slice = slice(None)
    ) -> torch.Tensor:
        """Expert joint POSITIONS only, without the velocity half.

        The Heracles-style ``root_qpos`` interface (its 38D config: 29 joint
        positions + 3D root position + 6D root orientation) carries no joint
        velocities at all. Because its controller is trained on this command
        space directly, the velocities are simply absent rather than
        reconstructed -- there is nothing to finite-difference.
        """
        self._ensure_mdp_step_cache()
        qpos = self.current_expert_frame["joint_pos"]
        if isinstance(joint_ids, slice):
            return qpos
        return qpos.index_select(-1, self._get_joint_ids_tensor_fast(joint_ids))

    def _get_expert_motion_command_fast(
        self, joint_ids: Sequence[int] | slice
    ) -> torch.Tensor:
        self._ensure_mdp_step_cache()
        if isinstance(joint_ids, slice):
            return torch.cat(
                [
                    self.current_expert_frame["joint_pos"],
                    self.current_expert_frame["joint_vel"],
                ],
                dim=-1,
            )

        joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
        cache_key = tuple(int(joint_id) for joint_id in joint_ids)
        motion_command = self._mdp_expert_motion_cache.get(cache_key)
        if motion_command is None:
            motion_command = torch.cat(
                [
                    self.current_expert_frame["joint_pos"].index_select(
                        -1, joint_ids_t
                    ),
                    self.current_expert_frame["joint_vel"].index_select(
                        -1, joint_ids_t
                    ),
                ],
                dim=-1,
            )
            self._mdp_expert_motion_cache[cache_key] = motion_command
        return motion_command

    @staticmethod
    def _command_window_steps_from_offsets(past_steps: int, future_steps: int) -> int:
        past_steps = int(past_steps)
        future_steps = int(future_steps)
        if past_steps < 0 or future_steps < 0:
            raise ValueError("Command window steps must be >= 0.")
        return past_steps + future_steps + 1

    @staticmethod
    def _allocate_agent_trajectory_command_terms(
        *,
        window_steps: int,
        num_joints: int,
        num_ee_bodies: int,
        num_keypoint_bodies: int = 0,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        def zeros(width: int) -> torch.Tensor:
            return torch.zeros((0, width), device=device, dtype=torch.float32)

        window_steps = int(window_steps)
        return {
            "expert_motion": zeros(window_steps * 2 * int(num_joints)),
            # root_qpos: the position half only, so the packet carries no joint
            # velocities at all rather than zero-filling them.
            "expert_motion_qpos": zeros(window_steps * int(num_joints)),
            "expert_anchor_pos_b": zeros(window_steps * 3),
            "expert_anchor_ori_b": zeros(window_steps * 6),
            "expert_ee_pos_b": zeros(window_steps * int(num_ee_bodies) * 3),
            "expert_ee_ori_b": zeros(window_steps * int(num_ee_bodies) * 6),
            # Keypoint positions and orientations share a body set of their
            # own so their packet slots never collide with the EE interface's.
            "expert_keypoint_pos_b": zeros(window_steps * int(num_keypoint_bodies) * 3),
            "expert_keypoint_ori_b": zeros(window_steps * int(num_keypoint_bodies) * 6),
        }

    def _ensure_agent_trajectory_command_terms(self) -> None:
        num_envs = int(self.num_envs)
        device = torch.device(self.device)
        for term_name, term in tuple(self._agent_trajectory_command_terms.items()):
            if term.shape[0] == num_envs and term.device == device:
                continue
            self._agent_trajectory_command_terms[term_name] = torch.zeros(
                (num_envs, int(term.shape[1])),
                device=device,
                dtype=torch.float32,
            )

    def _validate_command_window_request(
        self,
        *,
        past_steps: int,
        future_steps: int,
    ) -> None:
        requested_steps = self._command_window_steps_from_offsets(
            past_steps,
            future_steps,
        )
        if requested_steps != self._agent_trajectory_command_window_steps:
            raise ValueError(
                "Planner command window mismatch. "
                f"Configured planner command has {self._agent_trajectory_command_window_steps} steps, "
                f"but observation requested {requested_steps} steps."
            )

    def get_agent_trajectory_command_term(
        self,
        term_name: str,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._ensure_agent_trajectory_command_terms()
        try:
            value = self._agent_trajectory_command_terms[str(term_name)]
        except KeyError as err:
            raise KeyError(f"Unknown trajectory command term: {term_name!r}.") from err
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def set_agent_trajectory_command(
        self,
        command_terms: Mapping[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._ensure_agent_trajectory_command_terms()
        if env_ids is not None:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        for term_name, command in command_terms.items():
            key = str(term_name)
            if key not in self._agent_trajectory_command_terms:
                raise KeyError(f"Unknown trajectory command term: {key!r}.")
            target = self._agent_trajectory_command_terms[key]
            command = command.to(device=self.device, dtype=torch.float32)
            if env_ids is None:
                if command.ndim != 2 or command.shape != target.shape:
                    raise ValueError(
                        f"Trajectory command term {key!r} shape mismatch. "
                        f"Expected {tuple(target.shape)}, got {tuple(command.shape)}."
                    )
                target.copy_(command)
                continue
            expected_shape = (int(env_ids.shape[0]), int(target.shape[1]))
            if command.ndim != 2 or tuple(command.shape) != expected_shape:
                raise ValueError(
                    f"Trajectory command term {key!r} indexed shape mismatch. "
                    f"Expected {expected_shape}, got {tuple(command.shape)}."
                )
            target.index_copy_(0, env_ids, command)

    def set_agent_full_body_trajectory_command(
        self,
        *,
        expert_motion: torch.Tensor,
        expert_anchor_pos_b: torch.Tensor,
        expert_anchor_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.set_agent_trajectory_command(
            {
                "expert_motion": expert_motion,
                "expert_anchor_pos_b": expert_anchor_pos_b,
                "expert_anchor_ori_b": expert_anchor_ori_b,
            },
            env_ids=env_ids,
        )

    def set_agent_ee_trajectory_command(
        self,
        *,
        expert_ee_pos_b: torch.Tensor,
        expert_ee_ori_b: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self.set_agent_trajectory_command(
            {
                "expert_ee_pos_b": expert_ee_pos_b,
                "expert_ee_ori_b": expert_ee_ori_b,
            },
            env_ids=env_ids,
        )

    def set_planner_command_provider(self, provider: Any) -> None:
        """Register a callback that produces planner command packets in-step.

        ``planner_oracle`` fills the command buffer from the expert *inside* the
        observation pass, so the packet is expressed in the anchor frame of the
        very step that consumes it and its re-expression is exactly the identity
        at publication. An external publisher writing between steps cannot match
        that: it fetches body-frame quantities one physics step early, which
        silently biases the root command.

        Registering a provider gives a planner the same in-step contract. The
        callback receives the environment ids being renewed and returns a
        mapping of command term name to tensor for those environments.
        """
        self._planner_command_provider = provider
        self._planner_command_provider_token = None

    def _maybe_fill_from_planner_provider(self, phase: torch.Tensor) -> None:
        """Publish the registered planner packet for envs at hold phase zero."""
        provider = getattr(self, "_planner_command_provider", None)
        if provider is None:
            return
        # get_current_command_window_term runs once per command term; the
        # planner must be evaluated once per control step, not once per term.
        token = self.common_step_counter
        if getattr(self, "_planner_command_provider_token", None) == token:
            return
        renew_ids = torch.nonzero(phase == 0, as_tuple=False).flatten()
        self._planner_command_provider_token = token
        if renew_ids.numel() == 0:
            return
        terms = provider(renew_ids)
        if terms:
            self.set_agent_trajectory_command(terms, env_ids=renew_ids)

    def capture_held_command_anchor(
        self,
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Pin the held command anchor pose to the robot's current anchor.

        Published chunks are stored in the anchor frame at publish time and
        re-expressed into the current anchor frame on every consumption step.
        The env-filled (``planner_oracle``) path writes the chunk and captures
        that reference pose atomically inside observation computation, so its
        re-expression is exactly the identity at publication.

        An external publisher writes the buffer at a different instant, so its
        chunk is interpreted against a stale reference pose and the root command
        is systematically wrong. Calling this immediately after
        :meth:`set_agent_trajectory_command` restores the atomicity.
        """
        anchor_pos_w, anchor_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        anchor_pos_w = anchor_pos_w.reshape(-1, 3)
        anchor_quat_w = anchor_quat_w.reshape(-1, 4)
        # From here on the publisher owns this reference pose: the automatic
        # phase-0 recapture must not clobber it, or the packet (expressed in the
        # publish-time anchor frame) would be re-expressed as if it were already
        # in the consuming step's frame, losing exactly one step of robot motion.
        self._external_command_anchor_owner = True
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if stored is None:
            self._held_command_anchor_pose[anchor_body_name] = (
                anchor_pos_w.clone(),
                anchor_quat_w.clone(),
            )
            return
        if env_ids is None:
            stored[0].copy_(anchor_pos_w)
            stored[1].copy_(anchor_quat_w)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        stored[0].index_copy_(0, env_ids, anchor_pos_w.index_select(0, env_ids))
        stored[1].index_copy_(0, env_ids, anchor_quat_w.index_select(0, env_ids))

    def reset_agent_trajectory_command(
        self, env_ids: torch.Tensor | None = None
    ) -> None:
        self._ensure_agent_trajectory_command_terms()
        if env_ids is None:
            for command in self._agent_trajectory_command_terms.values():
                command.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        for command in self._agent_trajectory_command_terms.values():
            command.index_fill_(0, env_ids, 0.0)

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
            # Give a registered planner the same in-step publication contract as
            # planner_oracle above, so its packet is expressed in the anchor
            # frame of the step that consumes it rather than one step early.
            self._maybe_fill_from_planner_provider(self._command_hold_phase())

        # Observation tensors must not alias the mutable planner command buffers:
        # resets and subsequent planner publishes update those buffers in-place.
        value = self.get_agent_trajectory_command_term(term_name).clone()
        if hold_steps > 0:
            raw_traced = value if self._command_trace_enabled else None
            phase = self._command_hold_phase()
            self._update_held_command_anchor_pose(anchor_body_name, phase)
            value = self._shift_window_by_phase(
                value,
                phase,
                window_steps=self._agent_trajectory_command_window_steps,
            )
            shifted_traced = value if self._command_trace_enabled else None
            value = self._reexpress_window_in_current_anchor_frame(
                value,
                term_name=term_name,
                anchor_body_name=anchor_body_name,
                window_steps=self._agent_trajectory_command_window_steps,
            )
            self._maybe_trace_command_window(
                term_name=term_name,
                source=source,
                raw=raw_traced,
                shifted=shifted_traced,
                consumed=value,
                anchor_body_name=anchor_body_name,
                past_steps=past_steps,
                future_steps=future_steps,
                joint_ids=joint_ids,
                reference_body_names=reference_body_names,
            )
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def _maybe_trace_command_window(
        self,
        *,
        term_name: str,
        source: str,
        raw: torch.Tensor | None,
        shifted: torch.Tensor | None,
        consumed: torch.Tensor,
        anchor_body_name: str,
        past_steps: int,
        future_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice,
        reference_body_names: Sequence[str],
    ) -> None:
        """Append one command-pipeline record when tracing is enabled.

        Diagnostic only. Enabled by setting ``ISAACLAB_COMMAND_TRACE`` to an
        output ``.pt`` path; otherwise this is a single dict lookup and returns
        immediately, so the normal control path is unchanged.

        Records the buffer at each pipeline stage (raw -> phase-shifted ->
        re-expressed in the current anchor frame) alongside the expert window
        for the same control step. Comparing traces from two publication
        sources that carry identical ground-truth content isolates whether a
        divergence comes from phase alignment, anchor frame, or slot handling.
        """
        trace_path = _os.environ.get("ISAACLAB_COMMAND_TRACE")
        if not trace_path:
            return
        try:
            expert = self.get_current_expert_window_term(
                term_name=term_name,
                past_steps=past_steps,
                future_steps=future_steps,
                joint_ids=joint_ids,
                anchor_body_name=anchor_body_name,
                reference_body_names=reference_body_names,
            )
        except Exception:  # pragma: no cover - tracing must never break a run
            expert = None
        record = {
            "term_name": term_name,
            "source": source,
            "episode_length_buf": self.episode_length_buf.detach().cpu().clone(),
            "phase": self._command_hold_phase().detach().cpu().clone(),
            "consumed": consumed.detach().cpu().clone(),
        }
        if raw is not None:
            record["raw"] = raw.detach().cpu().clone()
        if shifted is not None:
            record["shifted"] = shifted.detach().cpu().clone()
        if expert is not None:
            record["expert"] = expert.detach().cpu().clone()
        held = self._held_command_anchor_pose.get(anchor_body_name)
        if held is not None:
            record["held_anchor_pos"] = held[0].detach().cpu().clone()
            record["held_anchor_quat"] = held[1].detach().cpu().clone()
        self._command_trace_records.append(record)
        # Flush incrementally so a crashed or killed run still yields a trace.
        if len(self._command_trace_records) % 50 == 0:
            torch.save(self._command_trace_records, trace_path)

    def flush_command_trace(self) -> str | None:
        """Persist any buffered command-pipeline trace records."""
        trace_path = _os.environ.get("ISAACLAB_COMMAND_TRACE")
        if not trace_path or not self._command_trace_records:
            return None
        torch.save(self._command_trace_records, trace_path)
        return trace_path

    @property
    def policy_command_mode(self) -> str:
        """Return the adapter used for low-level policy command observations."""
        return self._policy_command_mode

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

    def _command_hold_phase(self) -> torch.Tensor:
        """Per-env step offset within the current command hold window."""
        hold_steps = int(self._command_hold_steps)
        return self.episode_length_buf.to(dtype=torch.long) % hold_steps

    @staticmethod
    def _shift_window_by_phase(
        flat: torch.Tensor,
        phase: torch.Tensor,
        *,
        window_steps: int,
    ) -> torch.Tensor:
        """Time-align a held command chunk to the current control step.

        Shifts the frame-major flattened window ``[N, W * D]`` forward by each
        env's hold phase so the leading slot stays time-aligned with the
        current control step, repeating the final frame past the chunk end.
        """
        num_envs, width = flat.shape
        window_steps = int(window_steps)
        if window_steps <= 0 or width % window_steps != 0:
            raise ValueError(
                "Held command window width must be divisible by window steps, "
                f"got width={width}, window_steps={window_steps}."
            )
        per_step_dim = width // window_steps
        view = flat.reshape(num_envs, window_steps, per_step_dim)
        offsets = torch.arange(window_steps, device=flat.device, dtype=torch.long)
        indices = (
            offsets[None, :] + phase.to(device=flat.device, dtype=torch.long)[:, None]
        ).clamp_(max=window_steps - 1)
        shifted = view.gather(
            1, indices[:, :, None].expand(num_envs, window_steps, per_step_dim)
        )
        return shifted.reshape(num_envs, width)

    @staticmethod
    def _clamp_window_to_hold_boundary(
        flat: torch.Tensor,
        phase: torch.Tensor,
        *,
        window_steps: int,
    ) -> torch.Tensor:
        """Limit a fresh command window to the current hold's information.

        The fresh window at phase ``k`` covers frames ``[t, t + W - 1]`` while
        the chunk published at the last renewal only knew frames up to the
        hold boundary at slot ``W - 1 - k``. Slots past the boundary repeat
        the boundary frame (tail padding), so no post-renewal information
        leaks into the command observation.
        """
        num_envs, width = flat.shape
        window_steps = int(window_steps)
        if window_steps <= 0 or width % window_steps != 0:
            raise ValueError(
                "Held command window width must be divisible by window steps, "
                f"got width={width}, window_steps={window_steps}."
            )
        per_step_dim = width // window_steps
        view = flat.reshape(num_envs, window_steps, per_step_dim)
        offsets = torch.arange(window_steps, device=flat.device, dtype=torch.long)
        boundary = (
            window_steps - 1 - phase.to(device=flat.device, dtype=torch.long)
        ).clamp_(min=0)
        indices = torch.minimum(offsets[None, :], boundary[:, None])
        clamped = view.gather(
            1, indices[:, :, None].expand(num_envs, window_steps, per_step_dim)
        )
        return clamped.reshape(num_envs, width)

    def _update_held_command_anchor_pose(
        self, anchor_body_name: str, phase: torch.Tensor
    ) -> None:
        """Track the anchor pose at each env's last command renewal.

        Published chunks are expressed in the anchor frame at publish time;
        re-expressing them each step needs that reference pose.
        """
        anchor_pos_w, anchor_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        anchor_pos_w = anchor_pos_w.reshape(-1, 3)
        anchor_quat_w = anchor_quat_w.reshape(-1, 4)
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if (
            stored is None
            or stored[0].shape != anchor_pos_w.shape
            or stored[0].device != anchor_pos_w.device
        ):
            self._held_command_anchor_pose[anchor_body_name] = (
                anchor_pos_w.clone(),
                anchor_quat_w.clone(),
            )
            return
        if self._external_command_anchor_owner:
            # An external publisher pins this pose at publish time; recapturing
            # it here would discard the frame the packet was expressed in.
            return
        renew_mask = phase == 0
        if bool(renew_mask.any()):
            stored[0][renew_mask] = anchor_pos_w[renew_mask]
            stored[1][renew_mask] = anchor_quat_w[renew_mask]

    def _reexpress_window_in_current_anchor_frame(
        self,
        flat: torch.Tensor,
        *,
        term_name: str,
        anchor_body_name: str,
        window_steps: int,
    ) -> torch.Tensor:
        """Re-express a held chunk from its publish-time anchor frame.

        Standard VLA-WBC middleware refreshes command coordinates with
        odometry each control step; only the chunk *content* is held at the
        planner rate. Position (``*_pos_b``) and rot6d (``*_ori_b``) terms are
        rigidly transformed from the renewal-time anchor frame into the
        current one; joint-space terms are frame-invariant.
        """
        is_position = term_name.endswith("_pos_b")
        is_orientation = term_name.endswith("_ori_b")
        if not is_position and not is_orientation:
            return flat
        stored = self._held_command_anchor_pose.get(anchor_body_name)
        if stored is None:
            return flat
        renewal_pos_w, renewal_quat_w = stored
        current_pos_w, current_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        current_pos_w = current_pos_w.reshape(-1, 3)
        current_quat_w = current_quat_w.reshape(-1, 4)
        # Relative transform from renewal anchor frame to current anchor frame.
        delta_quat = math_utils.quat_mul(
            math_utils.quat_inv(current_quat_w), renewal_quat_w
        )
        delta_pos = math_utils.quat_apply_inverse(
            current_quat_w, renewal_pos_w - current_pos_w
        )
        num_envs, width = flat.shape
        if is_position:
            if width % 3 != 0:
                raise ValueError(
                    f"Position command term {term_name!r} width {width} is not "
                    "divisible by 3."
                )
            vectors = flat.reshape(num_envs, -1, 3)
            num_vectors = vectors.shape[1]
            delta_quat_exp = (
                delta_quat[:, None, :].expand(-1, num_vectors, -1).reshape(-1, 4)
            )
            rotated = math_utils.quat_apply(
                delta_quat_exp, vectors.reshape(-1, 3)
            ).reshape(num_envs, num_vectors, 3)
            return (rotated + delta_pos[:, None, :]).reshape(num_envs, width)
        if width % 6 != 0:
            raise ValueError(
                f"Orientation command term {term_name!r} width {width} is not "
                "divisible by 6."
            )
        delta_mat = math_utils.matrix_from_quat(delta_quat)
        columns = flat.reshape(num_envs, -1, 3, 2)
        rotated = torch.matmul(delta_mat[:, None, :, :], columns)
        return rotated.reshape(num_envs, width)

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
        value = self._clamp_window_to_hold_boundary(
            fresh, self._command_hold_phase(), window_steps=int(future_steps) + 1
        )
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def get_agent_latent_command(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the current agent-published latent command buffer."""
        if env_ids is None:
            return self._agent_latent_command
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return self._agent_latent_command.index_select(0, env_ids)

    def set_agent_latent_command(
        self, latent_command: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> None:
        """Publish the latest agent latent command into the env observation state."""
        latent_command = latent_command.to(device=self.device, dtype=torch.float32)
        if env_ids is None:
            if (
                latent_command.ndim != 2
                or latent_command.shape != self._agent_latent_command.shape
            ):
                raise ValueError(
                    "Latent command shape mismatch. "
                    f"Expected {tuple(self._agent_latent_command.shape)}, got {tuple(latent_command.shape)}."
                )
            self._agent_latent_command.copy_(latent_command)
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if latent_command.ndim != 2 or latent_command.shape != (
            env_ids.shape[0],
            self._agent_latent_dim,
        ):
            raise ValueError(
                "Latent command shape mismatch for indexed update. "
                f"Expected {(env_ids.shape[0], self._agent_latent_dim)}, got {tuple(latent_command.shape)}."
            )
        self._agent_latent_command.index_copy_(0, env_ids, latent_command)

    def reset_agent_latent_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset latent commands for the selected environments to zeros."""
        if env_ids is None:
            self._agent_latent_command.zero_()
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self._agent_latent_command.index_fill_(0, env_ids, 0.0)

    def _resolve_reference_body_visualization_pairs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]] | None:
        """Resolve pairs of (reference body idx, robot body idx) to visualize."""
        if len(self.reference_body_names) == 0:
            return None

        reference_body_pos = None
        reference_body_quat = None
        try:
            reference_body_pos = self.get_expert_trajectory_data("xpos")
            reference_body_quat = self.get_expert_trajectory_data("xquat")
        except KeyError:
            pass
        if reference_body_pos is None or reference_body_quat is None:
            return None
        num_reference_bodies = int(reference_body_pos.shape[1])

        robot_body_names = list(self.robot.body_names)
        robot_name_lookup = {name: idx for idx, name in enumerate(robot_body_names)}
        robot_name_lookup_lower = {
            name.lower(): idx for idx, name in enumerate(robot_body_names)
        }
        robot_normalized_lookup: dict[str, list[int]] = {}
        robot_normalized_names: list[str] = []

        for idx, body_name in enumerate(robot_body_names):
            normalized_name = self._normalize_body_name_for_matching(body_name)
            robot_normalized_names.append(normalized_name)
            robot_normalized_lookup.setdefault(normalized_name, []).append(idx)

        selected_ref_ids: list[int] = []
        selected_robot_ids: list[int] = []
        selected_names: list[str] = []
        used_robot_ids: set[int] = set()

        for ref_id, ref_body_name in enumerate(self.reference_body_names):
            if ref_id >= num_reference_bodies:
                continue
            robot_id: int | None = None

            if ref_body_name in robot_name_lookup:
                robot_id = robot_name_lookup[ref_body_name]
            else:
                ref_body_name_lower = ref_body_name.lower()
                if ref_body_name_lower in robot_name_lookup_lower:
                    robot_id = robot_name_lookup_lower[ref_body_name_lower]
                else:
                    normalized_ref_name = self._normalize_body_name_for_matching(
                        ref_body_name
                    )
                    normalized_matches = robot_normalized_lookup.get(
                        normalized_ref_name, []
                    )
                    if len(normalized_matches) > 0:
                        robot_id = normalized_matches[0]
                    else:
                        prefix_matches = [
                            idx
                            for idx, normalized_robot_name in enumerate(
                                robot_normalized_names
                            )
                            if normalized_robot_name.startswith(normalized_ref_name)
                            or normalized_ref_name.startswith(normalized_robot_name)
                        ]
                        if len(prefix_matches) > 0:
                            robot_id = prefix_matches[0]

            if robot_id is None:
                continue
            if robot_id in used_robot_ids:
                continue

            used_robot_ids.add(robot_id)
            selected_ref_ids.append(ref_id)
            selected_robot_ids.append(robot_id)
            selected_names.append(ref_body_name)

        if len(selected_ref_ids) == 0:
            return None
        return (
            torch.tensor(selected_ref_ids, dtype=torch.long, device=self.device),
            torch.tensor(selected_robot_ids, dtype=torch.long, device=self.device),
            selected_names,
        )

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Unitree-style fixed placement from dataset frame to sim world."""
        env_origins = getattr(self, "_expert_env_origins", None)
        if env_origins is None:
            env_origins = self.scene.env_origins
        if env_ids is None:
            align_pos = env_origins
        else:
            align_pos = env_origins.index_select(
                0, env_ids.to(device=env_origins.device, dtype=torch.long)
            )

        align_quat = align_pos.new_zeros((align_pos.shape[0], 4))
        # Identity quaternion in Isaac Lab 3.0's XYZW layout.
        align_quat[:, 3] = 1.0
        return align_quat, align_pos

    def _transform_reference_pose_to_world(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Place reference pose in simulation world using env origins."""
        align_quat, align_pos = self._get_reference_alignment_transform(env_ids)

        if ref_pos.ndim == 2:
            pos_w = math_utils.quat_apply(align_quat, ref_pos) + align_pos
            if ref_quat is None:
                return pos_w, None
            quat_w = math_utils.quat_mul(align_quat, ref_quat)
            return pos_w, quat_w

        if ref_pos.ndim != 3:
            raise ValueError(
                f"Unsupported ref_pos shape for transform: {tuple(ref_pos.shape)}"
            )

        num_envs, num_items = ref_pos.shape[0], ref_pos.shape[1]
        align_quat_expand = (
            align_quat.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, 4)
        )
        pos_w = math_utils.quat_apply(
            align_quat_expand, ref_pos.reshape(-1, 3)
        ).reshape(num_envs, num_items, 3)
        pos_w = pos_w + align_pos.unsqueeze(1)

        if ref_quat is None:
            return pos_w, None
        quat_w = math_utils.quat_mul(
            align_quat_expand, ref_quat.reshape(-1, 4)
        ).reshape(num_envs, num_items, 4)
        return pos_w, quat_w

    def _transform_reference_body_pose_to_init_alignment(
        self,
        ref_pos: torch.Tensor,
        ref_quat: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map reference body pose into the simulation world frame."""
        return self._transform_reference_pose_to_world(
            ref_pos, ref_quat, env_ids=env_ids
        )

    def _index_copy_reference_rows_(
        self, dst: TensorDict, src: TensorDict, env_ids: torch.Tensor
    ) -> None:
        # Isaac Lab 3.0 hands out int32 env indices; index_copy_ needs int64.
        env_ids = env_ids.to(dtype=torch.long)
        for key in src.keys():
            src_value = src.get(key)
            dst_value = dst.get(key)
            if isinstance(src_value, TensorDict) and isinstance(dst_value, TensorDict):
                self._index_copy_reference_rows_(dst_value, src_value, env_ids)
                continue
            if isinstance(src_value, torch.Tensor) and isinstance(
                dst_value, torch.Tensor
            ):
                dst_value.index_copy_(0, env_ids, src_value)
                continue
            dst.set(key, src_value)

    def _refresh_current_expert_frame(
        self, env_ids: torch.Tensor | None = None, *, advance: bool = False
    ) -> None:
        tm = self.trajectory_manager
        if env_ids is None:
            sampled_env_ids = torch.arange(
                self.num_envs, device=tm._state_device, dtype=torch.long
            )
        else:
            sampled_env_ids = env_ids.to(device=tm._state_device, dtype=torch.long)
        sampled_local_steps = tm.env_step.index_select(0, sampled_env_ids)
        reference = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(env_ids=env_ids, advance=advance)
        )
        if env_ids is None or self.current_expert_frame is None:
            self.current_expert_frame = reference
            self._current_reference_local_step.copy_(
                sampled_local_steps.to(device=self.device, dtype=torch.long)
            )
        else:
            self._index_copy_reference_rows_(
                self.current_expert_frame, reference, env_ids
            )
            self._current_reference_local_step.index_copy_(
                0,
                env_ids.to(device=self.device, dtype=torch.long),
                sampled_local_steps.to(device=self.device, dtype=torch.long),
            )
        self._invalidate_mdp_cache()

    def current_reference_is_final_frame(self) -> torch.Tensor:
        """Return true for envs whose current reward/obs reference is terminal."""
        tm = self.trajectory_manager
        traj_ranks = tm.env_traj_rank.to(device=self.device, dtype=torch.long)
        final_steps = (tm._length.index_select(0, traj_ranks) - 1).to(
            device=self.device, dtype=torch.long
        )
        return self._current_reference_local_step >= final_steps

    @staticmethod
    def _motion_command_cfg_owns_reset(cfg: Any) -> bool:
        """True when the configured ``motion`` command term owns reset sampling.

        Cfg-level twin of :meth:`_motion_command_reset_owner`, needed because
        `_setup_adaptive_failure_reset_sampler` runs before ``super().__init__``
        builds the CommandManager (and hence the term).
        """
        motion_cfg = getattr(getattr(cfg, "commands", None), "motion", None)
        return bool(getattr(motion_cfg, "owns_reset", False))

    def _motion_command_reset_owner(self) -> Any | None:
        """The ``motion`` command term when it owns reset-start sampling (v2).

        Returns None for tasks without the term or without ``owns_reset``
        (v0/v1), which keep the env-inline sampling path byte-identical.
        """
        command_manager = getattr(self, "command_manager", None)
        if command_manager is None:
            return None
        if "motion" not in getattr(command_manager, "active_terms", ()):
            return None
        term = command_manager.get_term("motion")
        if getattr(term.cfg, "owns_reset", False):
            return term
        return None

    def _setup_adaptive_failure_reset_sampler(self, cfg: Any) -> None:
        tm = self.trajectory_manager
        self._adaptive_failure_reset_sampler: SonicAdaptiveResetSampler | None = None
        if self._motion_command_cfg_owns_reset(cfg):
            # v2: the `motion` command term owns the reset-start samplers. It
            # constructs the single instance set in its __init__ (inside
            # `super().__init__`'s load_managers, i.e. after this method) and
            # mirrors the instances back onto these two attributes so the
            # adaptive-failure record/sample hooks keep feeding the same
            # objects. Building them here as well would silently split the
            # SONIC failure statistics across two SonicAdaptiveResetSampler
            # instances.
            self._start_frame_sampler = None
            return
        # The SONIC weight function is needed both by the legacy full-trajectory
        # joint rank+frame path and by reset_start_mode="adaptive".
        if self._random_reset_full_trajectory or self._reset_start_mode == "adaptive":
            self._adaptive_failure_reset_sampler = SonicAdaptiveResetSampler(
                tm._length,
                bin_size=self._adaptive_failure_reset_bin_size,
                sequence_length_agnostic=(
                    self._adaptive_failure_reset_sequence_length_agnostic
                ),
                init_num_failures=self._adaptive_failure_reset_init_num_failures,
                uniform_sampling_rate=self._adaptive_failure_reset_uniform_ratio,
                pre_failure_sample_window=(
                    self._adaptive_failure_reset_pre_failure_window
                ),
                failure_rate_max_over_mean=(
                    self._adaptive_failure_reset_failure_rate_max_over_mean
                ),
            )
        if self._random_reset_full_trajectory:
            # Legacy full-trajectory path: SONIC picks ranks AND frames jointly
            # from the bin distribution; the generic start sampler is unused.
            self._start_frame_sampler = None
            return

        # Resolve the effective starting-frame mode: "auto" keeps the legacy
        # random-range / fixed behavior; explicit modes map 1:1 onto
        # StartFrameSampler. Trajectory selection stays with the manager's
        # reset_schedule in every mode.
        mode = self._reset_start_mode
        if mode == "auto":
            mode = (
                StartFrameSampler.RANDOM
                if self._random_reset_step_max > self._random_reset_step_min
                else StartFrameSampler.FIXED
            )
        if mode == StartFrameSampler.ADAPTIVE:
            weight_fn = self._adaptive_reset_weight_fn
            if weight_fn is None:
                weight_fn = self._adaptive_failure_reset_sampler
            if weight_fn is None:
                raise ValueError(
                    "reset_start_mode='adaptive' requires the SONIC reset "
                    "sampler or a custom `cfg.adaptive_reset_weight_fn`."
                )
            self._start_frame_sampler = StartFrameSampler(
                tm._length,
                mode="adaptive",
                weight_fn=weight_fn,
                device=tm._state_device,
            )
            return
        self._start_frame_sampler = StartFrameSampler(
            tm._length,
            mode=mode,
            fixed_step=self._reference_start_frame,
            random_step_min=self._random_reset_step_min,
            random_step_max=self._random_reset_step_max,
            device=tm._state_device,
        )

    def set_adaptive_reset_weight_fn(self, weight_fn: Any) -> None:
        """Provide a custom adaptive starting-frame weight function.

        The callable must accept ``(trajectory_ranks, frame_steps)`` tensors
        and return one non-negative weight per (rank, step) pair, as expected
        by ``iltools.datasets.reset_sampling.StartFrameSampler``. It replaces
        the SONIC failure-weight function and switches the env to
        ``reset_start_mode='adaptive'`` (trajectory ranks still come from the
        manager's reset schedule).
        """
        if self._random_reset_full_trajectory:
            raise RuntimeError(
                "A custom adaptive weight function is incompatible with "
                "random_reset_full_trajectory (SONIC joint rank+frame sampling)."
            )
        if not callable(weight_fn):
            raise ValueError("weight_fn must be a callable.")
        self._adaptive_reset_weight_fn = weight_fn
        self._reset_start_mode = StartFrameSampler.ADAPTIVE
        self._start_frame_sampler = StartFrameSampler(
            self.trajectory_manager._length,
            mode="adaptive",
            weight_fn=weight_fn,
            device=self.trajectory_manager._state_device,
        )
        # v2 term-owned resets: keep the single-instance guarantee by handing
        # the replacement sampler to the owning `motion` command term too.
        motion_reset_owner = self._motion_command_reset_owner()
        if motion_reset_owner is not None:
            motion_reset_owner._start_frame_sampler = self._start_frame_sampler

    def _reset_tracking_failure_mask(self) -> torch.Tensor:
        failure_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for term_name in self.termination_manager.active_terms:
            term_cfg = self.termination_manager.get_term_cfg(term_name)
            if term_cfg.time_out or term_name == "reference_finished":
                continue
            failure_mask |= self.termination_manager.get_term(term_name)
        return failure_mask

    def _record_adaptive_failure_reset_visits(self) -> None:
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        tm = self.trajectory_manager
        sampler.record_visits(
            tm.env_traj_rank,
            self._current_reference_local_step,
        )

    def _record_adaptive_failure_reset_bins(self, env_ids: torch.Tensor) -> None:
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            return
        tm = self.trajectory_manager
        env_ids_device = env_ids.to(device=self.device, dtype=torch.long)
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        failed_mask = self._reset_tracking_failure_mask().index_select(
            0, env_ids_device
        )
        if not torch.any(failed_mask):
            return
        failed_mask_tm = failed_mask.to(device=tm._state_device)
        sampler.record_failures(
            tm.env_traj_rank.index_select(0, env_ids_tm)[failed_mask_tm],
            self._current_reference_local_step.index_select(0, env_ids_device)[
                failed_mask
            ],
        )

    def _sample_adaptive_failure_resets(
        self, count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampler = self._adaptive_failure_reset_sampler
        if sampler is None:
            raise RuntimeError("Adaptive failure reset sampler is not enabled.")
        return sampler.sample(count)

    def _pinned_joint_ids(self) -> torch.Tensor:
        """Live articulation indices in the action term's pinned joint order.

        Physics backends enumerate the articulation differently, so any joint
        vector that a policy or a recorded dataset consumes must be expressed in
        a fixed order. The action term already pins one via
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

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset the specified environments.

        Notes:
            IsaacLab managers, events, and sensors accept tensor indices and internally move
            them to the appropriate device. We normalize ``env_ids`` to a CUDA long tensor so
            that all internal buffers (which live on ``self.device``) and the trajectory
            manager see consistent indexing.
        """
        # Isaac Lab 3.0 hands out int32 env indices; normalize once here.
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        # Fold the terminal (pre-reset) frame into the ending episode's MPJPE
        # sum before anything below reassigns the tracked trajectory or writes
        # the reset state. This is the last point at which the robot's physical
        # state and the reference it was scored against both still belong to
        # the episode that is ending.
        self._accumulate_terminal_mpjpe_metric(env_ids)

        # Reset trajectory tracking (reassigns trajectories and resets steps).
        tm = self.trajectory_manager
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        motion_reset_owner = self._motion_command_reset_owner()
        if motion_reset_owner is not None:
            # v2: the `motion` command term owns the reset-start samplers and
            # applies them here, at the same point the inline paths below run
            # (before `super()._reset_idx`, whose reset-mode events read the
            # reference at the new cursor). Failure-bin recording stays with
            # the env because it reads termination-manager state; it feeds the
            # same sampler instances through the env-side aliases.
            self._record_adaptive_failure_reset_bins(env_ids)
            motion_reset_owner.resample_reference(env_ids)
        elif self._random_reset_full_trajectory:
            self._record_adaptive_failure_reset_bins(env_ids)
            reset_ranks, reset_steps = self._sample_adaptive_failure_resets(
                int(env_ids_tm.numel())
            )
            tm.reset_envs(
                env_ids_tm,
                ranks=reset_ranks,
                steps=reset_steps,
            )
        else:
            # Record terminal failure bins whenever the SONIC weight function
            # is in use, so adaptive starting frames keep adapting under
            # reset_schedule-driven trajectory selection too. No-op unless an
            # adaptive failure sampler exists.
            self._record_adaptive_failure_reset_bins(env_ids)
            # StartFrameSampler picks the local step (fixed/random/adaptive);
            # the trajectory manager picks the rank via its configured
            # reset_schedule.
            ranks = tm.env_traj_rank.index_select(0, env_ids_tm)
            reset_steps = self._start_frame_sampler.sample_steps(ranks)
            tm.reset_envs(env_ids_tm, steps=reset_steps)
        self.reset_agent_latent_command(env_ids)
        self.reset_agent_trajectory_command(env_ids)

        # Refresh only the resetting rows before reset events consume current_expert_frame.
        self._refresh_current_expert_frame(env_ids, advance=False)

        # Trigger the reset events (curriculum, sensors, managers, etc.) using tensor indices
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
        # After super(), which recreates ``extras["log"]`` from scratch.
        self._emit_mpjpe_episode_metric(env_ids)

        return result

    def _update_video_follow_camera(self) -> None:
        """Point the offscreen recorder camera at one robot before capturing.

        The kit-less Newton GL recorder only supports a static world-frame
        camera, which films empty ground once reference trajectories carry the
        robots away from their env origins (full-trajectory random starts).
        This hook re-aims the recorder at the tracked env's robot root each
        rendered frame; it is a no-op for the interactive/human path and
        whenever the recorder backend does not expose ``update_camera``.
        """
        if self.render_mode != "rgb_array":
            return
        if not bool(getattr(self.cfg, "video_follow_robot", False)):
            return
        capture = getattr(getattr(self, "video_recorder", None), "_capture", None)
        update_camera = getattr(capture, "update_camera", None)
        if update_camera is None:
            return
        try:
            robot = self.scene["robot"]
            env_index = int(getattr(self.cfg, "video_follow_env_index", 0))
            env_index %= self.num_envs
            root_pos = (
                robot.data.root_state_w.torch[env_index, :3]
                .detach()
                .to("cpu", non_blocking=False)
                .tolist()
            )
            eye_offset = tuple(
                float(v)
                for v in getattr(self.cfg, "video_follow_eye_offset", (3.5, 3.5, 2.0))
            )
            lookat_offset = tuple(
                float(v)
                for v in getattr(
                    self.cfg, "video_follow_lookat_offset", (0.0, 0.0, 0.0)
                )
            )
            eye = tuple(root_pos[i] + eye_offset[i] for i in range(3))
            lookat = tuple(root_pos[i] + lookat_offset[i] for i in range(3))
            update_camera(eye, lookat)
            if not getattr(self, "_video_follow_announced", False):
                self._video_follow_announced = True
                logger.info(
                    "Video follow camera active: env=%d eye=%s lookat=%s",
                    env_index,
                    tuple(round(v, 2) for v in eye),
                    tuple(round(v, 2) for v in lookat),
                )
        except Exception:  # never let video framing break training
            logger.warning("Video follow-camera update failed.", exc_info=True)

    def render(self, recompute: bool = False):
        """Render with the recorder camera optionally following a robot."""
        self._update_video_follow_camera()
        return super().render(recompute)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and update reference data."""
        # Standard RL stepping path.
        if not self.replay_only:
            # Get next reference data point (advance=True to move to next step)
            self._refresh_current_expert_frame(advance=True)
            self._record_adaptive_failure_reset_visits()
            super().step(action)
            rollout_state_log = self._compute_rollout_reference_state_log()
            # Accumulated after the physics step and after any reset inside it.
            # Envs that reset this step already had their terminal frame folded
            # into the ending episode by _reset_idx (via
            # _accumulate_terminal_mpjpe_metric, called before the reset write);
            # the state visible here for those envs is the fresh post-reset
            # pose, not something the policy produced, so it must not also be
            # accumulated into the new episode's sum.
            just_reset = (self.reset_terminated | self.reset_time_outs).nonzero(
                as_tuple=True
            )[0]
            mpjpe_log = self._accumulate_mpjpe_metric(exclude_env_ids=just_reset)
            if rollout_state_log or mpjpe_log:
                self.extras.setdefault("log", {}).update(rollout_state_log)
                self.extras.setdefault("log", {}).update(mpjpe_log)
            self._apply_reference_replay_targets()
            # Match IsaacLab command timing: reward/logging use the pre-step
            # reference frame, while returned observations expose the next frame.
            # The pre-step sample already advanced the trajectory cursor, so this
            # refresh must not advance again.
            self._refresh_current_expert_frame(advance=False)
            self._append_causal_planner_history()
            self.obs_buf = self.observation_manager.compute(update_history=True)
            return (
                self.obs_buf,
                self.reward_buf,
                self.reset_terminated,
                self.reset_time_outs,
                self.extras,
            )

        # Replay-only path: ignore physics stepping and evaluate rewards exactly
        # on the replayed reference state.
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        # Sample the current reference frame and advance the internal step by exactly one.
        # `sample(advance=True)` returns frame t and then increments to t+1.
        # This avoids double-advance while keeping reward computation aligned with frame t.
        reference_for_step = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(env_ids=None, advance=True)
        )
        self.current_expert_frame = reference_for_step
        self._record_adaptive_failure_reset_visits()
        self._invalidate_mdp_cache()
        self._replay_reference(reference=reference_for_step)
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
            # Populate Gymnasium-style terminal observation info for vector envs.
            # final_obs/final_info are object arrays with None for non-reset envs.
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

            # if sensors are added to the scene, make sure we render to reflect changes in reset
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
        # Expose post-step reference (frame t+1) for observations/outputs, matching
        # ManagerBasedRLEnv command timing after command_manager.compute().
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

    def get_expert_trajectory_data(
        self, key: str | None = None, joint_indices: Sequence[int] | None = None
    ) -> TensorDict | torch.Tensor:
        """
        Get the current reference data.

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
            if isinstance(data, torch.Tensor):
                return data[..., joint_indices]
            else:
                # Handle TensorDict case - data should be a Tensor
                return data[..., joint_indices]  # type: ignore[return-value]
        else:
            return data  # type: ignore[return-value]

    @staticmethod
    def _normalize_nested_key(key: NestedKey) -> tuple[str, ...]:
        """Normalize a nested key to tuple form."""
        if isinstance(key, tuple):
            return key
        return (key,)

    @staticmethod
    def _denormalize_nested_key(key_parts: tuple[str, ...]) -> NestedKey:
        """Convert tuple-form key back to str when single-token."""
        if len(key_parts) == 1:
            return key_parts[0]
        return key_parts

    @staticmethod
    def _select_last_dim(
        values: torch.Tensor, ids: torch.Tensor | slice
    ) -> torch.Tensor:
        if isinstance(ids, slice):
            return values
        return values.index_select(-1, ids)

    @staticmethod
    def _joint_ids_cache_key(joint_ids: torch.Tensor | Sequence[int] | slice) -> object:
        if isinstance(joint_ids, slice):
            return ("all",)
        if isinstance(joint_ids, torch.Tensor):
            return tuple(int(idx) for idx in joint_ids.tolist())
        return tuple(int(idx) for idx in joint_ids)

    def _sample_expert_trajectory_batch(
        self, batch_size: int
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
        """Sample random expert transitions without advancing env manager state."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        expert_frame, env_ids_tm, global_indices = (
            self.trajectory_manager.sample_random_transitions(batch_size)
        )
        expert_frame = _convert_reference_quats_to_xyzw(expert_frame)
        return (
            expert_frame.to(self.device),
            env_ids_tm.to(self.device),
            global_indices.to(self.device),
        )

    @staticmethod
    def _normalize_expert_macro_split(split: str | None) -> str:
        if split is None:
            return "all"
        normalized = str(split).strip().lower()
        return "all" if normalized == "" else normalized

    def _expert_macro_nonempty_trajectory_ranks(self) -> torch.Tensor:
        tm = self.trajectory_manager
        lengths = getattr(tm, "_length", None)
        if not isinstance(lengths, torch.Tensor):
            raise RuntimeError(
                "Trajectory manager does not expose trajectory lengths for "
                "expert macro sampling."
            )
        nonempty_ranks = torch.nonzero(lengths > 0, as_tuple=False).flatten()
        if int(nonempty_ranks.numel()) == 0:
            raise RuntimeError(
                "Trajectory manager has no nonempty trajectories for expert "
                "macro sampling."
            )
        return nonempty_ranks.to(device=tm._state_device, dtype=torch.long)

    def _expert_macro_split_trajectory_ranks(
        self,
        *,
        split: str | None,
        eval_fraction: float,
        split_seed: int,
    ) -> torch.Tensor:
        normalized = self._normalize_expert_macro_split(split)
        nonempty_ranks = self._expert_macro_nonempty_trajectory_ranks()
        if normalized == "all":
            return nonempty_ranks
        if normalized not in {"train", "eval"}:
            raise ValueError(
                "Expert macro split must be one of 'all', 'train', or 'eval', "
                f"got {split!r}."
            )
        eval_fraction = float(eval_fraction)
        if not 0.0 < eval_fraction < 1.0:
            raise ValueError(
                "eval_fraction must be in (0, 1) when using train/eval "
                f"expert macro splits, got {eval_fraction!r}."
            )

        num_ranks = int(nonempty_ranks.numel())
        if num_ranks < 2:
            logger.warning(
                "Expert macro train/eval split requested with fewer than two "
                "nonempty trajectories; using all trajectories for %s split.",
                normalized,
            )
            return nonempty_ranks

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(split_seed))
        nonempty_cpu = nonempty_ranks.detach().cpu()
        perm = nonempty_cpu.index_select(
            0,
            torch.randperm(num_ranks, generator=generator),
        )
        eval_count = max(1, int(round(num_ranks * eval_fraction)))
        eval_count = min(eval_count, num_ranks - 1)
        selected = perm[:eval_count] if normalized == "eval" else perm[eval_count:]
        selected = selected.sort().values
        return selected.to(
            device=self.trajectory_manager._state_device, dtype=torch.long
        )

    def _sample_expert_window_slice_for_trajectory_ranks(
        self,
        traj_ranks: torch.Tensor,
        local_steps: torch.Tensor,
        *,
        past_steps: int,
        future_steps: int,
    ) -> TensorDict:
        """Sample an expert window from explicit trajectory ranks."""
        if past_steps < 0 or future_steps < 0:
            raise ValueError("Expert window steps must be >= 0.")
        tm = self.trajectory_manager
        traj_ranks_tm = traj_ranks.to(device=tm._state_device, dtype=torch.long)
        local_steps_tm = local_steps.to(device=tm._state_device, dtype=torch.long)
        if tuple(traj_ranks_tm.shape) != tuple(local_steps_tm.shape):
            raise ValueError(
                "traj_ranks and local_steps must have matching shapes for "
                "expert macro sampling."
            )
        window_offsets = torch.arange(
            -past_steps,
            future_steps + 1,
            device=tm._state_device,
            dtype=torch.long,
        )
        lengths = tm._length.index_select(0, traj_ranks_tm).clamp(min=1)
        max_step = lengths - 1
        window_steps = local_steps_tm.unsqueeze(1) + window_offsets.unsqueeze(0)
        window_steps = window_steps.clamp(min=0)
        window_steps = torch.minimum(window_steps, max_step.unsqueeze(1))
        global_indices = (
            tm._start.index_select(0, traj_ranks_tm).unsqueeze(1) + window_steps
        )
        expert_window = tm.rb[global_indices.to(device=tm._storage_device)]
        if getattr(tm, "_device", None) is not None:
            expert_window = expert_window.to(tm._device)
        expert_window = tm._attach_reference_fields(expert_window, use_buffers=False)
        return _convert_reference_quats_to_xyzw(expert_window.to(self.device))

    def _build_reward_input_cache(self, *, device: torch.device) -> None:
        """Pre-materialize expert-side values for the `reward_input` obs group.

        Stores a flat [total_transitions, 2 * num_ref_joints] tensor for the
        expert_motion term (joint_pos + joint_vel concatenated), plus two
        broadcast buffers for the anchor-error terms that are zero / identity
        on the expert side by construction.

        No-op when the active observation config has no `reward_input` group
        (cfg.enable_reward_input_observations=False, the -G1-v2 default): the
        cache buffers stay None and `_reward_input_expert_terms` fails loudly
        if anything still requests expert-side reward_input values.
        """
        if not self._reward_input_group_present:
            self._reward_input_motion_cache = None
            self._reward_input_zero_anchor_pos = None
            self._reward_input_identity_rot6d = None
            return
        tm = self.trajectory_manager
        total = int(tm._end.max().item())
        if total <= 0:
            raise RuntimeError(
                "Trajectory manager has no transitions; cannot build reward_input cache."
            )
        global_indices = torch.arange(
            total, device=tm._storage_device, dtype=torch.int64
        )
        reference = tm.rb[global_indices]
        if tm._device is not None:
            reference = reference.to(tm._device)
        reference = tm._attach_reference_fields(reference, use_buffers=False)
        joint_pos = reference.get("joint_pos")
        joint_vel = reference.get("joint_vel")
        if joint_pos is None or joint_vel is None:
            raise RuntimeError(
                "reward_input cache build failed: trajectory manager did not produce joint_pos/joint_vel."
            )
        self._reward_input_motion_cache = torch.cat([joint_pos, joint_vel], dim=-1).to(
            device=device
        )
        self._reward_input_zero_anchor_pos = torch.zeros(3, device=device)
        identity = torch.zeros(6, device=device)
        identity[0] = 1.0
        identity[4] = 1.0
        self._reward_input_identity_rot6d = identity

    def _reward_input_expert_terms(
        self,
        global_indices: torch.Tensor,
        batch_size: int,
        term_name: str,
    ) -> torch.Tensor | None:
        """Return expert-side reward_input term values from the precomputed cache."""
        motion_cache = self._reward_input_motion_cache
        zero_anchor_pos = self._reward_input_zero_anchor_pos
        identity_rot6d = self._reward_input_identity_rot6d
        if motion_cache is None or zero_anchor_pos is None or identity_rot6d is None:
            raise RuntimeError(
                "Expert-side reward_input values were requested, but this task "
                "has no reward_input observation group "
                "(cfg.enable_reward_input_observations=False). Reward "
                "estimation is opt-in: set "
                "env.enable_reward_input_observations=True and "
                "agent.reward_estimation=true."
            )
        if term_name == "expert_motion":
            idx = global_indices.to(device=motion_cache.device, dtype=torch.int64)
            return motion_cache.index_select(0, idx)
        if term_name == "expert_anchor_pos_b":
            return zero_anchor_pos.expand(batch_size, 3)
        if term_name == "expert_anchor_ori_b":
            return identity_rot6d.expand(batch_size, 6)
        return None

    def _expert_local_steps_from_global_indices(
        self,
        env_ids: torch.Tensor,
        global_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Convert replay-buffer global indices back to local trajectory steps."""
        tm = self.trajectory_manager
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        global_indices_tm = global_indices.to(device=tm._state_device, dtype=torch.long)
        traj_ranks = tm.env_traj_rank[env_ids_tm]
        local_steps = global_indices_tm - tm._start[traj_ranks]
        return local_steps.to(device=self.device, dtype=torch.long)

    def _current_local_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        tm = self.trajectory_manager
        return tm.env_step[env_ids.to(device=tm._state_device, dtype=torch.long)].to(
            device=self.device, dtype=torch.long
        )

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

    def _sample_expert_window_slice(
        self,
        env_ids: torch.Tensor,
        local_steps: torch.Tensor,
        *,
        past_steps: int,
        future_steps: int,
    ) -> TensorDict:
        """Sample an oldest-to-newest expert window around each requested step."""
        if past_steps < 0 or future_steps < 0:
            raise ValueError("Expert window steps must be >= 0.")
        tm = self.trajectory_manager
        env_ids_tm = env_ids.to(device=tm._state_device, dtype=torch.long)
        local_steps_tm = local_steps.to(device=tm._state_device, dtype=torch.long)
        window_offsets = torch.arange(
            -past_steps,
            future_steps + 1,
            device=tm._state_device,
            dtype=torch.long,
        )
        window_steps = local_steps_tm.unsqueeze(1) + window_offsets.unsqueeze(0)
        window_steps = window_steps.clamp(min=0)
        expert_window = tm.sample_slice(
            batch_size=int(window_offsets.shape[0]),
            env_ids=env_ids_tm,
            start_steps=window_steps,
            mode="independent",
        )
        return _convert_reference_quats_to_xyzw(expert_window.to(self.device))

    def _expert_body_pose_fields(
        self, expert_td: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        ref_body_pos_key = (
            self._mdp_reference_body_pos_key
            if hasattr(self, "_mdp_reference_body_pos_key")
            and self._mdp_reference_body_pos_key in expert_td.keys()
            else ("xpos" if "xpos" in expert_td.keys() else "body_pos_w")
        )
        ref_body_quat_key = (
            self._mdp_reference_body_quat_key
            if hasattr(self, "_mdp_reference_body_quat_key")
            and self._mdp_reference_body_quat_key in expert_td.keys()
            else ("xquat" if "xquat" in expert_td.keys() else "body_quat_w")
        )
        body_pos = expert_td.get(ref_body_pos_key)
        body_quat = expert_td.get(ref_body_quat_key)
        if body_pos is None or body_quat is None:
            raise KeyError(
                "Expert batch is missing body pose fields required for expert observations."
            )
        return body_pos, body_quat, body_pos.ndim - 2

    def _raw_expert_state_terms(
        self,
        expert_frame: TensorDict,
        env_ids: torch.Tensor,
        *,
        prefix: tuple[str, ...] = (),
    ) -> dict[str, torch.Tensor]:
        key = (
            (lambda name: name) if len(prefix) == 0 else (lambda name: (*prefix, name))
        )
        root_pos_ref = expert_frame.get(key("root_pos"))
        root_quat_ref = expert_frame.get(key("root_quat"))
        root_lin_vel_ref = expert_frame.get(key("root_lin_vel"))
        root_ang_vel_ref = expert_frame.get(key("root_ang_vel"))
        joint_pos_ref = expert_frame.get(key("joint_pos"))
        joint_vel_ref = expert_frame.get(key("joint_vel"))
        if (
            root_pos_ref is None
            or root_quat_ref is None
            or root_lin_vel_ref is None
            or root_ang_vel_ref is None
            or joint_pos_ref is None
            or joint_vel_ref is None
        ):
            raise KeyError(
                f"Expert batch is missing fields for prefix {prefix or ('current',)}."
            )

        root_pos_w, root_quat_w_opt = self._transform_reference_pose_to_world(
            root_pos_ref, root_quat_ref, env_ids=env_ids
        )
        if root_quat_w_opt is None:
            raise RuntimeError("Failed to transform expert quaternion for sampling.")
        root_quat_w = root_quat_w_opt

        scene = getattr(self, "scene", None)
        if scene is None:
            env_origins = self._expert_env_origins.index_select(0, env_ids)
        else:
            env_origins = scene.env_origins.index_select(0, env_ids)
        root_pos = root_pos_w - env_origins

        align_quat, _ = self._get_reference_alignment_transform(env_ids)
        root_lin_vel = math_utils.quat_apply(align_quat, root_lin_vel_ref)
        root_ang_vel = math_utils.quat_apply(align_quat, root_ang_vel_ref)
        base_lin_vel = math_utils.quat_apply_inverse(root_quat_w, root_lin_vel)
        base_ang_vel = math_utils.quat_apply_inverse(root_quat_w, root_ang_vel)

        default_joint_pos = getattr(self, "_expert_default_joint_pos", None)
        if default_joint_pos is None:
            default_joint_pos = torch.zeros_like(joint_pos_ref)
        else:
            default_joint_pos = default_joint_pos.index_select(0, env_ids).to(
                device=joint_pos_ref.device, dtype=joint_pos_ref.dtype
            )
        default_joint_vel = getattr(self, "_expert_default_joint_vel", None)
        if default_joint_vel is None:
            default_joint_vel = torch.zeros_like(joint_vel_ref)
        else:
            default_joint_vel = default_joint_vel.index_select(0, env_ids).to(
                device=joint_vel_ref.device, dtype=joint_vel_ref.dtype
            )

        action_dim = int(joint_pos_ref.shape[-1])
        action_manager = getattr(self, "action_manager", None)
        if action_manager is not None:
            for attr_name in ("total_action_dim", "action_dim"):
                dim = getattr(action_manager, attr_name, None)
                if dim is not None:
                    action_dim = int(dim)
                    break
        last_action = expert_frame.get(key("last_action"))
        if last_action is None:
            last_action = expert_frame.get(key("action"))
        if last_action is None:
            last_action = torch.zeros(
                (int(joint_pos_ref.shape[0]), action_dim),
                device=joint_pos_ref.device,
                dtype=joint_pos_ref.dtype,
            )

        return {
            "joint_pos": joint_pos_ref,
            "joint_vel": joint_vel_ref,
            "joint_pos_rel": joint_pos_ref - default_joint_pos,
            "joint_vel_rel": joint_vel_ref - default_joint_vel,
            "root_pos": root_pos,
            "root_quat": root_quat_w,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "last_action": last_action,
            "expert_motion": torch.cat([joint_pos_ref, joint_vel_ref], dim=-1),
        }

    def _expert_anchor_terms(
        self,
        expert_frame: TensorDict,
        env_ids: torch.Tensor,
        *,
        context: str,
        anchor_body_name: str = "torso_link",
    ) -> dict[str, torch.Tensor]:
        batch_size = int(env_ids.shape[0])
        if context == "expert":
            zero_anchor_pos = torch.zeros((batch_size, 3), device=self.device)
            identity_rot6d = torch.zeros((batch_size, 6), device=self.device)
            identity_rot6d[:, 0] = 1.0
            identity_rot6d[:, 4] = 1.0
            return {
                "expert_anchor_pos_b": zero_anchor_pos,
                "expert_anchor_ori_b": identity_rot6d,
            }
        if context != "rollout":
            raise ValueError(f"Unsupported expert observation context: {context!r}.")

        compiled = _get_mdp_compiled_module()
        body_pos_source, body_quat_source, body_dim = self._expert_body_pose_fields(
            expert_frame
        )
        anchor_ids = self._get_reference_body_ids_fast((anchor_body_name,))
        expert_anchor_pos = body_pos_source.index_select(body_dim, anchor_ids).squeeze(
            body_dim
        )
        expert_anchor_quat = body_quat_source.index_select(
            body_dim, anchor_ids
        ).squeeze(body_dim)
        expert_anchor_pos_w, expert_anchor_quat_w_opt = (
            self._transform_reference_pose_to_world(
                expert_anchor_pos, expert_anchor_quat, env_ids=env_ids
            )
        )
        if expert_anchor_quat_w_opt is None:
            raise RuntimeError(
                "Failed to transform expert anchor quaternion for rollout observations."
            )
        robot_anchor_pos_w, robot_anchor_quat_w = self._get_robot_anchor_state_w_fast(
            anchor_body_name
        )
        robot_anchor_pos_w = robot_anchor_pos_w.index_select(0, env_ids)
        robot_anchor_quat_w = robot_anchor_quat_w.index_select(0, env_ids)
        anchor_pos_b, anchor_ori_b = compiled.body_pose_in_anchor_frame(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            expert_anchor_pos_w,
            expert_anchor_quat_w_opt,
        )
        return {
            "expert_anchor_pos_b": anchor_pos_b[:, 0, :],
            "expert_anchor_ori_b": compiled.quat_to_rot6d_flat(anchor_ori_b[:, 0, :]),
        }

    def _build_expert_window_terms(
        self,
        expert_window: TensorDict,
        env_ids: torch.Tensor,
        *,
        context: str,
        past_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
    ) -> dict[str, torch.Tensor]:
        compiled = _get_mdp_compiled_module()
        batch_size = int(env_ids.shape[0])
        joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
        joint_pos = self._select_last_dim(expert_window["joint_pos"], joint_ids_t)
        joint_vel = self._select_last_dim(expert_window["joint_vel"], joint_ids_t)
        expert_motion = torch.cat([joint_pos, joint_vel], dim=-1).reshape(
            batch_size, -1
        )

        body_pos_source, body_quat_source, body_dim = self._expert_body_pose_fields(
            expert_window
        )
        anchor_ids = self._get_reference_body_ids_fast((anchor_body_name,))
        anchor_pos = body_pos_source.index_select(body_dim, anchor_ids).squeeze(
            body_dim
        )
        anchor_quat = body_quat_source.index_select(body_dim, anchor_ids).squeeze(
            body_dim
        )
        body_terms_enabled = len(reference_body_names) > 0
        if body_terms_enabled:
            body_ids = self._get_reference_body_ids_fast(tuple(reference_body_names))
            body_pos = body_pos_source.index_select(body_dim, body_ids)
            body_quat = body_quat_source.index_select(body_dim, body_ids)

        if context == "expert":
            center_index = int(past_steps)
            center_anchor_pos = anchor_pos[:, center_index, :]
            center_anchor_quat = anchor_quat[:, center_index, :]
            anchor_pos_b, anchor_ori_b = compiled.body_pose_in_anchor_frame(
                center_anchor_pos,
                center_anchor_quat,
                anchor_pos,
                anchor_quat,
            )
            if body_terms_enabled:
                body_pos_b, body_ori_b = compiled.body_pose_in_anchor_frame(
                    center_anchor_pos,
                    center_anchor_quat,
                    body_pos.reshape(batch_size, -1, 3),
                    body_quat.reshape(batch_size, -1, 4),
                )
        elif context == "rollout":
            window_size = int(anchor_pos.shape[1])
            flat_env_ids = env_ids[:, None].expand(-1, window_size).reshape(-1)
            anchor_pos_w, anchor_quat_w_opt = self._transform_reference_pose_to_world(
                anchor_pos.reshape(-1, 3),
                anchor_quat.reshape(-1, 4),
                env_ids=flat_env_ids,
            )
            if anchor_quat_w_opt is None:
                raise RuntimeError(
                    "Failed to transform expert-window anchor quaternion for rollout observations."
                )
            anchor_pos_w = anchor_pos_w.reshape(batch_size, window_size, 3)
            anchor_quat_w = anchor_quat_w_opt.reshape(batch_size, window_size, 4)
            robot_anchor_pos_w, robot_anchor_quat_w = (
                self._get_robot_anchor_state_w_fast(anchor_body_name)
            )
            robot_anchor_pos_w = robot_anchor_pos_w.index_select(0, env_ids)
            robot_anchor_quat_w = robot_anchor_quat_w.index_select(0, env_ids)
            anchor_pos_b, anchor_ori_b = compiled.body_pose_in_anchor_frame(
                robot_anchor_pos_w,
                robot_anchor_quat_w,
                anchor_pos_w,
                anchor_quat_w,
            )
            if body_terms_enabled:
                body_count = int(body_pos.shape[2])
                flat_body_env_ids = (
                    env_ids[:, None, None]
                    .expand(-1, window_size, body_count)
                    .reshape(-1)
                )
                body_pos_w, body_quat_w_opt = self._transform_reference_pose_to_world(
                    body_pos.reshape(-1, 3),
                    body_quat.reshape(-1, 4),
                    env_ids=flat_body_env_ids,
                )
                if body_quat_w_opt is None:
                    raise RuntimeError(
                        "Failed to transform expert-window body quaternion for rollout observations."
                    )
                body_pos_w = body_pos_w.reshape(batch_size, window_size, body_count, 3)
                body_quat_w = body_quat_w_opt.reshape(
                    batch_size,
                    window_size,
                    body_count,
                    4,
                )
                body_pos_b, body_ori_b = compiled.body_pose_in_anchor_frame(
                    robot_anchor_pos_w,
                    robot_anchor_quat_w,
                    body_pos_w.reshape(batch_size, window_size * body_count, 3),
                    body_quat_w.reshape(batch_size, window_size * body_count, 4),
                )
        else:
            raise ValueError(f"Unsupported expert-window context: {context!r}.")

        terms = {
            "expert_motion": expert_motion,
            "expert_motion_qpos": joint_pos.reshape(batch_size, -1),
            "expert_anchor_pos_b": anchor_pos_b.reshape(batch_size, -1),
            "expert_anchor_ori_b": compiled.quat_to_rot6d_flat(anchor_ori_b).reshape(
                batch_size, -1
            ),
        }
        if body_terms_enabled:
            body_pos_flat = body_pos_b.reshape(batch_size, -1)
            terms["expert_ee_pos_b"] = body_pos_flat
            body_ori_flat = compiled.quat_to_rot6d_flat(body_ori_b).reshape(
                batch_size, -1
            )
            terms["expert_ee_ori_b"] = body_ori_flat
            # Keypoint terms use the same anchor-frame pose calculation under a
            # separately keyed body-set cache. Exposing position and orientation
            # independently lets configs select point targets or complete poses.
            terms["expert_keypoint_pos_b"] = body_pos_flat
            terms["expert_keypoint_ori_b"] = body_ori_flat
        return terms

    def _expert_macro_feature_term_order(self) -> tuple[str, ...]:
        """Expert-window terms that make up one DiffSR macro-state frame.

        Configurable because the skill encoder's input width is defined by this
        selection: the default gives 58+3+6 = 67/frame -> 670 for a 10-frame
        window, byte-identical to the full-body packet. Selecting
        ``expert_motion_qpos`` instead gives 29+3+6 = 38 -> 380, byte-identical
        to the root_qpos packet, which is what a GR00T-style whole-body
        qpos+root latent interface needs. Nothing in the DiffSR trainer has to
        know -- it reads whatever macro state the environment produces.
        """
        configured = getattr(self.cfg, "expert_macro_state_terms", None)
        if configured:
            # A bare string is iterable, so `for name in configured` would
            # silently yield single characters as term names. Hydra delivers
            # `env.expert_macro_state_terms=[a,b,c]` as a string in some
            # invocation forms, so accept and split it rather than producing
            # nonsense terms.
            if isinstance(configured, str):
                configured = [
                    part.strip()
                    for part in configured.strip().strip("[]").split(",")
                    if part.strip()
                ]
            names = tuple(str(name) for name in configured)
            if not names or any(len(name) <= 1 for name in names):
                raise ValueError(
                    "expert_macro_state_terms parsed to "
                    f"{names!r}, which is not a list of term names. Pass it as "
                    "a list, e.g. "
                    "[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]."
                )
            return names
        return (
            "expert_motion",
            "expert_anchor_pos_b",
            "expert_anchor_ori_b",
        )

    def _expert_macro_state_sequence_from_terms(
        self,
        terms: dict[str, torch.Tensor],
        *,
        batch_size: int,
        window_steps: int,
    ) -> torch.Tensor:
        """Convert flattened expert-window terms into [B, T, D] state features."""
        features: list[torch.Tensor] = []
        for term_name in self._expert_macro_feature_term_order():
            value = terms[term_name]
            if value.ndim != 2 or int(value.shape[0]) != int(batch_size):
                raise ValueError(
                    "Expert macro sampler term shape mismatch for "
                    f"{term_name}: expected first dims ({batch_size}, *), "
                    f"got {tuple(value.shape)}."
                )
            if int(value.shape[1]) % int(window_steps) != 0:
                raise ValueError(
                    "Expert macro sampler term width is not divisible by "
                    f"window_steps for {term_name}: shape {tuple(value.shape)}, "
                    f"window_steps={window_steps}."
                )
            features.append(value.reshape(batch_size, window_steps, -1))
        return torch.cat(features, dim=-1)

    def _expert_macro_state_feature_slices_from_terms(
        self,
        terms: dict[str, torch.Tensor],
        *,
        batch_size: int,
        window_steps: int,
    ) -> dict[str, tuple[int, int]]:
        """Return per-timestep [start, end) feature slices for macro states."""
        del batch_size
        cursor = 0
        slices: dict[str, tuple[int, int]] = {}
        for term_name in self._expert_macro_feature_term_order():
            value = terms[term_name]
            if value.ndim != 2:
                raise ValueError(
                    "Expert macro sampler term shape mismatch for "
                    f"{term_name}: expected [B, T*D], got {tuple(value.shape)}."
                )
            if int(value.shape[1]) % int(window_steps) != 0:
                raise ValueError(
                    "Expert macro sampler term width is not divisible by "
                    f"window_steps for {term_name}: shape {tuple(value.shape)}, "
                    f"window_steps={window_steps}."
                )
            width = int(value.shape[1]) // int(window_steps)
            slices[term_name] = (cursor, cursor + width)
            cursor += width
        return slices

    def _build_expert_macro_window_terms(
        self,
        expert_window: TensorDict,
        env_ids: torch.Tensor,
        *,
        context: str,
        past_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
    ) -> dict[str, torch.Tensor]:
        """Build independently selectable joint, EE, and keypoint macro terms."""
        selected = set(self._expert_macro_feature_term_order())
        terms = self._build_expert_window_terms(
            expert_window,
            env_ids,
            context=context,
            past_steps=past_steps,
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
        )
        body_groups = (
            (
                {"expert_ee_pos_b", "expert_ee_ori_b"},
                self._command_ee_body_names,
            ),
            (
                {"expert_keypoint_pos_b", "expert_keypoint_ori_b"},
                self._command_keypoint_body_names,
            ),
        )
        for term_names, reference_body_names in body_groups:
            requested = selected.intersection(term_names)
            if not requested:
                continue
            body_terms = self._build_expert_window_terms(
                expert_window,
                env_ids,
                context=context,
                past_steps=past_steps,
                joint_ids=joint_ids,
                anchor_body_name=anchor_body_name,
                reference_body_names=reference_body_names,
            )
            for term_name in requested:
                terms[term_name] = body_terms[term_name]
        return terms

    def _get_current_expert_window_terms(
        self,
        *,
        past_steps: int,
        future_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
    ) -> dict[str, torch.Tensor]:
        self._ensure_mdp_step_cache()
        joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
        reference_body_names_t = tuple(str(name) for name in reference_body_names)
        cache_key = (
            int(past_steps),
            int(future_steps),
            str(anchor_body_name),
            self._joint_ids_cache_key(joint_ids_t),
            reference_body_names_t,
        )
        cached_terms = self._mdp_expert_window_obs_cache.get(cache_key)
        if cached_terms is not None:
            return cached_terms

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        local_steps = self._current_local_steps(env_ids)
        expert_window = self._sample_expert_window_slice(
            env_ids,
            local_steps,
            past_steps=int(past_steps),
            future_steps=int(future_steps),
        )
        cached_terms = self._build_expert_window_terms(
            expert_window,
            env_ids,
            context="rollout",
            past_steps=int(past_steps),
            joint_ids=joint_ids_t,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names_t,
        )
        self._mdp_expert_window_obs_cache[cache_key] = cached_terms
        return cached_terms

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
        if int(frame_stride) != 1:
            raise NotImplementedError(
                "frame_stride > 1 requires the v2 environment; the legacy "
                "expert-window path samples consecutive frames only."
            )
        value = self._get_current_expert_window_terms(
            past_steps=int(past_steps),
            future_steps=int(future_steps),
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
        )[term_name]
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def _get_current_expert_goal_terms(
        self,
        *,
        goal_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
    ) -> dict[str, torch.Tensor]:
        self._ensure_mdp_step_cache()
        joint_ids_t = self._get_joint_ids_tensor_fast(joint_ids)
        cache_key = (
            int(goal_steps),
            str(anchor_body_name),
            self._joint_ids_cache_key(joint_ids_t),
        )
        cached_terms = self._mdp_expert_goal_obs_cache.get(cache_key)
        if cached_terms is not None:
            return cached_terms

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        local_steps = self._current_local_steps(env_ids) + int(goal_steps)
        expert_goal = self._sample_expert_window_slice(
            env_ids,
            local_steps,
            past_steps=0,
            future_steps=0,
        )
        cached_terms = self._build_expert_window_terms(
            expert_goal,
            env_ids,
            context="rollout",
            past_steps=0,
            joint_ids=joint_ids_t,
            anchor_body_name=anchor_body_name,
        )
        self._mdp_expert_goal_obs_cache[cache_key] = cached_terms
        return cached_terms

    def get_current_expert_goal_term(
        self,
        term_name: str,
        *,
        goal_steps: int,
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self._get_current_expert_goal_terms(
            goal_steps=int(goal_steps),
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
        )[term_name]
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def _map_requested_expert_observations(
        self,
        expert_frame: TensorDict,
        env_ids: torch.Tensor,
        obs_keys: Sequence[NestedKey],
        *,
        context: str,
        prefix: tuple[str, ...] = (),
        local_steps: torch.Tensor | None = None,
        global_indices: torch.Tensor | None = None,
        past_steps: int,
        future_steps: int,
    ) -> dict[NestedKey, torch.Tensor] | None:
        mapped_values: dict[NestedKey, torch.Tensor] = {}
        unknown_terms: list[str] = []
        raw_state_terms = self._raw_expert_state_terms(
            expert_frame, env_ids, prefix=prefix
        )
        anchor_terms_cache: dict[str, dict[str, torch.Tensor]] = {}
        window_terms_cache: dict[tuple[object, ...], dict[str, torch.Tensor]] = {}
        batch_size = int(env_ids.shape[0])

        for obs_key in obs_keys:
            key_tuple = self._normalize_nested_key(obs_key)
            group_name = key_tuple[0] if len(key_tuple) > 1 else "expert_state"
            term_name = key_tuple[-1]

            if group_name == "reward_input":
                if context != "expert" or global_indices is None:
                    unknown_terms.append(term_name)
                    continue
                value = self._reward_input_expert_terms(
                    global_indices, batch_size=batch_size, term_name=term_name
                )
            elif group_name == "expert_window":
                if len(prefix) > 0:
                    unknown_terms.append(term_name)
                    continue
                if local_steps is None:
                    logger.warning(
                        "Expert mapper received expert_window requests without trajectory-local steps."
                    )
                    return None
                if term_name in {"expert_ee_pos_b", "expert_ee_ori_b"}:
                    reference_body_names = self._command_ee_body_names
                elif term_name in {
                    "expert_keypoint_pos_b",
                    "expert_keypoint_ori_b",
                }:
                    reference_body_names = self._command_keypoint_body_names
                else:
                    reference_body_names = ()
                cache_key = (
                    int(past_steps),
                    int(future_steps),
                    self._expert_anchor_body_name,
                    ("all",),
                    reference_body_names,
                )
                if cache_key not in window_terms_cache:
                    expert_window = self._sample_expert_window_slice(
                        env_ids,
                        local_steps,
                        past_steps=int(past_steps),
                        future_steps=int(future_steps),
                    )
                    window_terms_cache[cache_key] = self._build_expert_window_terms(
                        expert_window,
                        env_ids,
                        context=context,
                        past_steps=int(past_steps),
                        joint_ids=slice(None),
                        anchor_body_name=self._expert_anchor_body_name,
                        reference_body_names=reference_body_names,
                    )
                value = window_terms_cache[cache_key].get(term_name)
            elif group_name == "expert_goal":
                if len(prefix) > 0:
                    unknown_terms.append(term_name)
                    continue
                if local_steps is None:
                    logger.warning(
                        "Expert mapper received expert_goal requests without trajectory-local steps."
                    )
                    return None
                goal_steps = int(self._latent_goal_steps)
                cache_key = (
                    goal_steps,
                    0,
                    self._expert_anchor_body_name,
                    ("all",),
                )
                if cache_key not in window_terms_cache:
                    expert_goal = self._sample_expert_window_slice(
                        env_ids,
                        local_steps + goal_steps,
                        past_steps=0,
                        future_steps=0,
                    )
                    window_terms_cache[cache_key] = self._build_expert_window_terms(
                        expert_goal,
                        env_ids,
                        context=context,
                        past_steps=0,
                        joint_ids=slice(None),
                        anchor_body_name=self._expert_anchor_body_name,
                    )
                value = window_terms_cache[cache_key].get(term_name)
            elif group_name in {"expert_state", "", "policy", "critic"}:
                value = raw_state_terms.get(term_name)
                if value is None and term_name in {
                    "expert_anchor_pos_b",
                    "expert_anchor_ori_b",
                }:
                    anchor_terms = anchor_terms_cache.get(self._expert_anchor_body_name)
                    if anchor_terms is None:
                        anchor_terms = self._expert_anchor_terms(
                            expert_frame,
                            env_ids,
                            context=context,
                            anchor_body_name=self._expert_anchor_body_name,
                        )
                        anchor_terms_cache[self._expert_anchor_body_name] = anchor_terms
                    value = anchor_terms.get(term_name)
            else:
                value = None

            if value is None:
                unknown_terms.append(term_name)
                continue
            mapped_values[obs_key] = value

        if len(unknown_terms) > 0:
            for term_name in unknown_terms:
                if term_name in self._expert_sampler_warned_unknown_terms:
                    continue
                logger.warning(
                    "Expert sampler cannot provide term '%s' from trajectory manager.",
                    term_name,
                )
                self._expert_sampler_warned_unknown_terms.add(term_name)
            return None

        return mapped_values

    def _sample_expert_batch_impl(
        self,
        batch_size: int,
        required_keys: Sequence[NestedKey],
        *,
        past_steps: int,
        future_steps: int,
    ) -> TensorDict | None:
        if batch_size <= 0:
            return None
        if len(required_keys) == 0:
            return TensorDict({}, batch_size=[batch_size], device=self.device)

        dedup_required_keys = list(dict.fromkeys(required_keys))
        current_obs_keys: list[NestedKey] = []
        next_obs_keys: list[NestedKey] = []
        needs_action = False

        for key in dedup_required_keys:
            key_tuple = self._normalize_nested_key(key)
            if key_tuple == ("action",):
                needs_action = True
                continue
            if len(key_tuple) > 0 and key_tuple[0] == "next":
                if len(key_tuple) < 2:
                    continue
                next_obs_keys.append(self._denormalize_nested_key(key_tuple[1:]))
                continue
            current_obs_keys.append(self._denormalize_nested_key(key_tuple))

        expert_batch = TensorDict({}, batch_size=[batch_size], device=self.device)
        current_expert_frame: TensorDict | None = None
        current_env_ids: torch.Tensor | None = None
        current_global_indices: torch.Tensor | None = None

        needs_current_transition = (
            len(current_obs_keys) > 0
            or needs_action
            or (len(next_obs_keys) > 0 and self._reference_has_aligned_next)
        )
        if needs_current_transition:
            current_expert_frame, current_env_ids, current_global_indices = (
                self._sample_expert_trajectory_batch(batch_size)
            )

        current_local_steps: torch.Tensor | None = None
        if (
            current_expert_frame is not None
            and current_env_ids is not None
            and current_global_indices is not None
        ):
            current_local_steps = self._expert_local_steps_from_global_indices(
                current_env_ids,
                current_global_indices,
            )

        if len(current_obs_keys) > 0:
            assert (
                current_expert_frame is not None
                and current_env_ids is not None
                and current_local_steps is not None
            )
            mapped_current = self._map_requested_expert_observations(
                current_expert_frame,
                current_env_ids,
                current_obs_keys,
                context="expert",
                local_steps=current_local_steps,
                global_indices=current_global_indices,
                past_steps=int(past_steps),
                future_steps=int(future_steps),
            )
            if mapped_current is None:
                return None
            for key, value in mapped_current.items():
                expert_batch.set(key, value)

        if len(next_obs_keys) > 0:
            next_global_indices: torch.Tensor | None
            if self._reference_has_aligned_next:
                assert current_expert_frame is not None and current_env_ids is not None
                next_expert_frame = current_expert_frame
                next_env_ids = current_env_ids
                next_global_indices = current_global_indices
                next_prefix = ("next",)
            else:
                next_expert_frame, next_env_ids, next_global_indices = (
                    self._sample_expert_trajectory_batch(batch_size)
                )
                next_prefix = ()
            mapped_next = self._map_requested_expert_observations(
                next_expert_frame,
                next_env_ids,
                next_obs_keys,
                context="expert",
                prefix=next_prefix,
                global_indices=next_global_indices,
                past_steps=int(past_steps),
                future_steps=int(future_steps),
            )
            if mapped_next is None:
                return None
            for key, value in mapped_next.items():
                key_tuple = self._normalize_nested_key(key)
                expert_batch.set(("next", *key_tuple), value)

        if needs_action:
            sampled_action = None
            if (
                current_expert_frame is not None
                and "action" in current_expert_frame.keys()
            ):
                sampled_action = current_expert_frame.get("action")
            if sampled_action is None:
                raise RuntimeError(
                    "Expert sampler was asked for action labels, but no recorded "
                    "expert action is available in the expert frame. Provide "
                    "action labels in the expert data."
                )
            sampled_action = sampled_action.to(self.device)
            expert_batch.set("action", sampled_action)

        return expert_batch

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[NestedKey]
    ) -> TensorDict | None:
        """Sample an expert batch for imitation algorithms from trajectory manager."""
        return self._sample_expert_batch_impl(
            batch_size,
            required_keys,
            past_steps=int(self._latent_patch_past_steps),
            future_steps=int(self._latent_patch_future_steps),
        )

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
        """Sample high-level expert macro transitions from the trajectory manager.

        The returned high-level state uses the same feature terms as the
        ``expert_window`` observation group, concatenated per timestep:
        expert_motion, expert_anchor_pos_b, and expert_anchor_ori_b. The sampled
        window is clamped at trajectory boundaries by ``_sample_expert_window_slice``,
        matching existing expert-window behavior. If ``trajectory_ranks`` is
        provided, samples are drawn only from those explicit trajectory ranks and
        ``split`` is ignored.
        """
        batch_size = int(batch_size)
        horizon_steps = int(horizon_steps)
        state_history_steps = int(state_history_steps)
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0.")
        if state_history_steps < 0:
            raise ValueError("state_history_steps must be >= 0.")

        tm = self.trajectory_manager
        if trajectory_ranks is not None:
            selected_ranks = torch.as_tensor(
                trajectory_ranks, device=tm._state_device, dtype=torch.long
            ).reshape(-1)
            if int(selected_ranks.numel()) == 0:
                raise ValueError("trajectory_ranks must select at least one rank.")
            max_rank = int(getattr(tm, "num_trajectories", tm._length.numel()))
            invalid = selected_ranks[
                (selected_ranks < 0) | (selected_ranks >= max_rank)
            ]
            if int(invalid.numel()) > 0:
                bad = sorted({int(item) for item in invalid.detach().cpu().tolist()})
                raise ValueError(
                    f"trajectory_ranks out of range [0, {max_rank - 1}]: {bad}."
                )
            lengths_for_selected = tm._length.index_select(0, selected_ranks)
            empty = selected_ranks[lengths_for_selected <= 0]
            if int(empty.numel()) > 0:
                bad = sorted({int(item) for item in empty.detach().cpu().tolist()})
                raise ValueError(f"trajectory_ranks include empty trajectories: {bad}.")

            choices = torch.randint(
                low=0,
                high=int(selected_ranks.numel()),
                size=(batch_size,),
                device=tm._state_device,
                dtype=torch.long,
            )
            traj_ranks_tm = selected_ranks.index_select(0, choices)
            lengths = tm._length.index_select(0, traj_ranks_tm).clamp(min=1)
            local_steps_tm = torch.floor(
                torch.rand(batch_size, device=tm._state_device)
                * lengths.to(dtype=torch.float32)
            ).to(dtype=torch.long)
            env_ids = torch.arange(batch_size, device=self.device, dtype=torch.long)
            local_steps = local_steps_tm.to(device=self.device, dtype=torch.long)
            traj_rank = traj_ranks_tm.to(device=self.device, dtype=torch.long)
            expert_window = self._sample_expert_window_slice_for_trajectory_ranks(
                traj_ranks_tm,
                local_steps_tm,
                past_steps=state_history_steps,
                future_steps=horizon_steps,
            )
        else:
            self._expert_macro_nonempty_trajectory_ranks()
            normalized_split = self._normalize_expert_macro_split(split)
            if normalized_split == "all":
                _, env_ids, global_indices = self._sample_expert_trajectory_batch(
                    batch_size
                )
                local_steps = self._expert_local_steps_from_global_indices(
                    env_ids,
                    global_indices,
                )
                traj_rank = tm.env_traj_rank.index_select(
                    0, env_ids.to(device=tm._state_device, dtype=torch.long)
                ).to(device=self.device, dtype=torch.long)
                expert_window = self._sample_expert_window_slice(
                    env_ids,
                    local_steps,
                    past_steps=state_history_steps,
                    future_steps=horizon_steps,
                )
            else:
                split_ranks = self._expert_macro_split_trajectory_ranks(
                    split=normalized_split,
                    eval_fraction=float(eval_fraction),
                    split_seed=int(split_seed),
                )
                choices = torch.randint(
                    low=0,
                    high=int(split_ranks.numel()),
                    size=(batch_size,),
                    device=tm._state_device,
                    dtype=torch.long,
                )
                traj_ranks_tm = split_ranks.index_select(0, choices)
                lengths = tm._length.index_select(0, traj_ranks_tm).clamp(min=1)
                local_steps_tm = torch.floor(
                    torch.rand(batch_size, device=tm._state_device)
                    * lengths.to(dtype=torch.float32)
                ).to(dtype=torch.long)
                env_ids = torch.arange(batch_size, device=self.device, dtype=torch.long)
                local_steps = local_steps_tm.to(device=self.device, dtype=torch.long)
                traj_rank = traj_ranks_tm.to(device=self.device, dtype=torch.long)
                expert_window = self._sample_expert_window_slice_for_trajectory_ranks(
                    traj_ranks_tm,
                    local_steps_tm,
                    past_steps=state_history_steps,
                    future_steps=horizon_steps,
                )
        window_terms = self._build_expert_macro_window_terms(
            expert_window,
            env_ids,
            context="expert",
            past_steps=state_history_steps,
            joint_ids=slice(None),
            anchor_body_name=self._expert_anchor_body_name,
        )
        window_steps = state_history_steps + horizon_steps + 1
        current_index = state_history_steps
        sequence = self._expert_macro_state_sequence_from_terms(
            window_terms,
            batch_size=batch_size,
            window_steps=window_steps,
        )
        self._expert_macro_feature_slices = (
            self._expert_macro_state_feature_slices_from_terms(
                window_terms,
                batch_size=batch_size,
                window_steps=window_steps,
            )
        )
        state = sequence[:, current_index, :].contiguous()
        future_window = sequence[:, current_index + 1 :, :].contiguous()
        target = sequence[:, -1, :].contiguous()
        state_history = None
        if state_history_steps > 0:
            state_history = sequence[:, : current_index + 1, :].contiguous()

        state_dim = int(state.shape[-1])
        expected_state = (batch_size, state_dim)
        expected_window = (batch_size, horizon_steps, state_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                "Expert macro sampler produced invalid state shape: "
                f"expected {expected_state}, got {tuple(state.shape)}."
            )
        if tuple(future_window.shape) != expected_window:
            raise ValueError(
                "Expert macro sampler produced invalid future_window shape: "
                f"expected {expected_window}, got {tuple(future_window.shape)}."
            )
        if tuple(target.shape) != expected_state:
            raise ValueError(
                "Expert macro sampler produced invalid target shape: "
                f"expected {expected_state}, got {tuple(target.shape)}."
            )

        hl_payload = {
            "state": state,
            "future_window": future_window,
            "target": target,
            "traj_rank": traj_rank,
            "local_step": local_steps,
        }
        if state_history is not None:
            expected_history = (batch_size, state_history_steps + 1, state_dim)
            if tuple(state_history.shape) != expected_history:
                raise ValueError(
                    "Expert macro sampler produced invalid state_history shape: "
                    f"expected {expected_history}, got {tuple(state_history.shape)}."
                )
            hl_payload["state_history"] = state_history
        hl = TensorDict(
            hl_payload,
            batch_size=[batch_size],
            device=self.device,
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size], device=self.device)

    def current_expert_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        """Return macro transitions aligned to each live environment cursor."""
        horizon_steps = int(horizon_steps)
        state_history_steps = int(state_history_steps)
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0.")
        if state_history_steps < 0:
            raise ValueError("state_history_steps must be >= 0.")
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        batch_size = int(env_ids_t.numel())
        if batch_size <= 0:
            raise ValueError("env_ids must select at least one environment.")

        local_steps = self._current_local_steps(env_ids_t)
        tm = self.trajectory_manager
        traj_rank = tm.env_traj_rank.index_select(
            0, env_ids_t.to(device=tm._state_device, dtype=torch.long)
        ).to(device=self.device, dtype=torch.long)
        trajectory_length = tm._length.index_select(
            0, traj_rank.to(device=tm._state_device, dtype=torch.long)
        ).to(device=self.device, dtype=torch.long)
        expert_window = self._sample_expert_window_slice(
            env_ids_t,
            local_steps,
            past_steps=state_history_steps,
            future_steps=horizon_steps,
        )
        window_terms = self._build_expert_macro_window_terms(
            expert_window,
            env_ids_t,
            context="rollout",
            past_steps=state_history_steps,
            joint_ids=slice(None),
            anchor_body_name=self._expert_anchor_body_name,
        )
        window_steps = state_history_steps + horizon_steps + 1
        current_index = state_history_steps
        sequence = self._expert_macro_state_sequence_from_terms(
            window_terms,
            batch_size=batch_size,
            window_steps=window_steps,
        )
        self._expert_macro_feature_slices = (
            self._expert_macro_state_feature_slices_from_terms(
                window_terms,
                batch_size=batch_size,
                window_steps=window_steps,
            )
        )
        state = sequence[:, current_index, :].contiguous()
        future_window = sequence[:, current_index + 1 :, :].contiguous()
        target = sequence[:, -1, :].contiguous()
        state_history = None
        if state_history_steps > 0:
            state_history = sequence[:, : current_index + 1, :].contiguous()

        state_dim = int(state.shape[-1])
        expected_state = (batch_size, state_dim)
        expected_window = (batch_size, horizon_steps, state_dim)
        if tuple(state.shape) != expected_state:
            raise ValueError(
                "Current expert macro sampler produced invalid state shape: "
                f"expected {expected_state}, got {tuple(state.shape)}."
            )
        if tuple(future_window.shape) != expected_window:
            raise ValueError(
                "Current expert macro sampler produced invalid future_window shape: "
                f"expected {expected_window}, got {tuple(future_window.shape)}."
            )
        if tuple(target.shape) != expected_state:
            raise ValueError(
                "Current expert macro sampler produced invalid target shape: "
                f"expected {expected_state}, got {tuple(target.shape)}."
            )

        hl_payload = {
            "state": state,
            "future_window": future_window,
            "target": target,
            "traj_rank": traj_rank,
            "local_step": local_steps,
            "trajectory_length": trajectory_length,
        }
        if state_history is not None:
            expected_history = (batch_size, state_history_steps + 1, state_dim)
            if tuple(state_history.shape) != expected_history:
                raise ValueError(
                    "Current expert macro sampler produced invalid state_history shape: "
                    f"expected {expected_history}, got {tuple(state_history.shape)}."
                )
            hl_payload["state_history"] = state_history
        hl = TensorDict(
            hl_payload,
            batch_size=[batch_size],
            device=self.device,
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size], device=self.device)

    def expert_trajectory_motion_names(self) -> list[str]:
        """Return motion names indexed by trajectory rank (for language goals)."""
        tm = self.trajectory_manager
        ordered = getattr(tm, "_ordered_traj_list", None)
        if not ordered:
            raise RuntimeError(
                "Trajectory manager does not expose an ordered trajectory list "
                "for language-goal motion-name lookup."
            )
        return [str(item[1]) for item in ordered]

    def current_achieved_macro_transition_batch(
        self,
        horizon_steps: int,
        env_ids: torch.Tensor | Sequence[int] | None = None,
        state_history_steps: int = 0,
    ) -> TensorDict:
        """Return macro transitions with selected command terms achieved by the robot.

        Joint-state, EE, and sparse-keypoint components are replaced independently,
        so the same path supports root+qpos, root+five-keypoint pose, and composed
        ablations. Rollout-context anchor terms remain expert-relative-to-robot and
        the future window/target remain expert-derived.
        """
        horizon_steps = int(horizon_steps)
        batch = self.current_expert_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)

        slices = self.expert_macro_feature_slices(horizon_steps)
        selected = set(slices)
        joint_pos = self.robot.data.joint_pos.torch.index_select(0, env_ids_t).to(
            device=self.device, dtype=torch.float32
        )
        joint_vel = self.robot.data.joint_vel.torch.index_select(0, env_ids_t).to(
            device=self.device, dtype=torch.float32
        )
        achieved_terms: dict[str, torch.Tensor] = {
            "expert_motion": torch.cat([joint_pos, joint_vel], dim=-1),
            "expert_motion_qpos": joint_pos,
        }

        body_groups = (
            (
                "expert_ee_pos_b",
                "expert_ee_ori_b",
                self._command_ee_body_names,
            ),
            (
                "expert_keypoint_pos_b",
                "expert_keypoint_ori_b",
                self._command_keypoint_body_names,
            ),
        )
        for pos_term, ori_term, body_names in body_groups:
            if not selected.intersection({pos_term, ori_term}):
                continue
            if not body_names:
                raise ValueError(
                    f"Macro interface selected {pos_term!r}/{ori_term!r}, but its "
                    "configured robot body set is empty."
                )
            body_ids = self._get_robot_body_ids_by_name_fast(body_names)
            body_pos_b, body_quat_b = self._get_robot_body_state_in_anchor_frame_fast(
                body_ids,
                self._expert_anchor_body_name,
            )
            body_pos_b = body_pos_b.index_select(0, env_ids_t)
            body_quat_b = body_quat_b.index_select(0, env_ids_t)
            achieved_terms[pos_term] = body_pos_b.reshape(len(env_ids_t), -1).to(
                device=self.device, dtype=torch.float32
            )
            achieved_terms[ori_term] = (
                _get_mdp_compiled_module()
                .quat_to_rot6d_flat(body_quat_b)
                .reshape(len(env_ids_t), -1)
                .to(device=self.device, dtype=torch.float32)
            )

        replacements = [name for name in slices if name in achieved_terms]
        if not replacements:
            raise RuntimeError(
                "Configured expert_macro_state_terms contain no achieved robot "
                "component; select joint, EE, or keypoint terms in addition to anchors."
            )
        state = batch.get(("hl", "state")).clone()
        state_history = batch.get(("hl", "state_history"))
        if state_history is not None:
            state_history = state_history.clone()
        for term_name in replacements:
            start_idx, end_idx = slices[term_name]
            achieved = achieved_terms[term_name]
            expected_width = int(end_idx) - int(start_idx)
            if tuple(achieved.shape) != (int(env_ids_t.numel()), expected_width):
                raise ValueError(
                    f"Achieved {term_name} shape mismatch: expected "
                    f"({int(env_ids_t.numel())}, {expected_width}), got "
                    f"{tuple(achieved.shape)}."
                )
            state[:, int(start_idx) : int(end_idx)] = achieved
            if state_history is not None:
                state_history[:, -1, int(start_idx) : int(end_idx)] = achieved
        batch.set(("hl", "state"), state)
        if state_history is not None:
            batch.set(("hl", "state_history"), state_history)
        return batch

    def expert_macro_feature_slices(
        self,
        horizon_steps: int,
    ) -> dict[str, tuple[int, int]]:
        """Return per-timestep macro-state feature slices by source term."""
        horizon_steps = int(horizon_steps)
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0.")
        cached = getattr(self, "_expert_macro_feature_slices", None)
        if cached is None:
            self.sample_expert_macro_transition_batch(
                batch_size=1,
                horizon_steps=horizon_steps,
                split="all",
            )
            cached = getattr(self, "_expert_macro_feature_slices", None)
        if cached is None:
            raise RuntimeError("Expert macro feature slices are unavailable.")
        return {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in cached.items()
        }

    def _replay_reference(
        self, env_ids: torch.Tensor | None = None, reference: TensorDict | None = None
    ):
        """Replay the reference data. If env_ids is provided, only replay the reference data for the given environments.
        If env_ids is not provided, replay the reference data for all environments."""

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
        # Use link/com-specific writers so all articulation data buffers stay coherent.
        # `base_lin_vel` uses root_com_vel_w + root_link_quat_w internally.
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_data_to_sim()
        # Refresh cached kinematics buffers (e.g. root_lin_vel_b) after direct state writes.
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

        # Apply the full per-episode rigid transform (R, t) from reset frame to world frame.
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

    def _setup_reference_velocity_visualizer(self) -> None:
        """Create desired/current frame markers for root and tracked bodies."""
        if not self._reference_vel_vis_enabled:
            return

        # Desired reference body (root) location and current robot root.
        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/Imitation/reference_root_goal"
        goal_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._goal_root_frame_marker = VisualizationMarkers(goal_cfg)
        self._goal_root_frame_marker.set_visibility(True)
        current_cfg = FRAME_MARKER_CFG.copy()
        current_cfg.prim_path = "/Visuals/Imitation/current_root"
        current_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        self._current_root_frame_marker = VisualizationMarkers(current_cfg)
        self._current_root_frame_marker.set_visibility(True)

        body_id_pairs = self._resolve_reference_body_visualization_pairs()
        if body_id_pairs is None:
            return

        self._vis_reference_body_ids, self._vis_robot_body_ids, self._vis_body_names = (
            body_id_pairs
        )
        for body_name in self._vis_body_names:
            current_body_cfg = FRAME_MARKER_CFG.copy()
            current_body_cfg.prim_path = f"/Visuals/Imitation/current_body/{body_name}"
            current_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            current_body_marker = VisualizationMarkers(current_body_cfg)
            current_body_marker.set_visibility(True)
            self._current_body_frame_markers.append(current_body_marker)

            goal_body_cfg = FRAME_MARKER_CFG.copy()
            goal_body_cfg.prim_path = (
                f"/Visuals/Imitation/reference_body_goal/{body_name}"
            )
            goal_body_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            goal_body_marker = VisualizationMarkers(goal_body_cfg)
            goal_body_marker.set_visibility(True)
            self._goal_body_frame_markers.append(goal_body_marker)
