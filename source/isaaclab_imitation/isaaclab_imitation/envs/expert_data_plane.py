"""ExpertDataPlane: the reference-data component of the v2 env fork.

Owned component of :class:`~isaaclab_imitation.envs.imitation_rl_env_v2.ImitationRLEnv`
that replaces the legacy env's inline reference machinery: dataset load,
reference metadata / joint alignment, every MDP fast-path cache and accessor,
frame refresh, alignment transforms, expert window / goal building, expert
batch / macro-transition sampling, the parked reward-input cache, the MPJPE
metric computation, and the offline-dataset mapper params. The legacy env
(``envs/imitation_rl_env_legacy.py``) stays byte-frozen for v0/v1; this plane is
only used by the v2 env.

Construction is two-phase, mirroring the legacy env's lifecycle:

- ``ExpertDataPlane(cfg, env)`` (phase 1, before ``ManagerBasedRLEnv.__init__``
  builds the managers): resolves and builds the Zarr dataset, creates the
  ``ParallelTrajectoryManager``, samples the first reference frame, and
  pre-materializes the reward-input cache.
- ``finalize(scene, robot)`` (phase 2, after the base env constructed the
  scene): retargets the trajectory manager to the live articulation joint
  order, captures env origins / default joint state, finalizes reference body
  names, and initializes the MDP fast paths and the MPJPE metric.

The plane holds a plain back-reference to the env for ``device`` /
``num_envs`` / ``common_step_counter`` / ``scene`` / ``robot`` /
``action_manager`` / ``cfg`` (same idiom as the command terms), so there are
no duplicated counters.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import isaaclab.utils.math as math_utils
import torch
import zarr
from iltools.datasets.lafan1.loader import Lafan1CsvLoader
from iltools.datasets.loaders import load_dataset_loader
from iltools.datasets.manager import ParallelTrajectoryManager, ResetSchedule
from iltools.datasets.utils import make_rb_from
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_imitation.assets.robots import UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab_imitation.envs.imitation_rl_env_v2 import ImitationRLEnv

logger = logging.getLogger(__name__)

_MDP_COMPILED: Any | None = None

_REFERENCE_QUAT_KEYS = (
    "root_quat",
    "xquat",
    "body_quat_w",
    "next_root_quat",
    "next_xquat",
    "next_body_quat_w",
)
_WXYZ_TO_XYZW = [1, 2, 3, 0]

_METRES_TO_MM = 1000.0


def _get_mdp_compiled_module() -> Any:
    global _MDP_COMPILED
    if _MDP_COMPILED is None:
        from isaaclab_imitation.tasks.manager_based.imitation.mdp import _compiled

        _MDP_COMPILED = _compiled
    return _MDP_COMPILED


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


def _normalize_dataset_keys(raw: Any) -> list[str] | None:
    """Normalize `env.dataset_keys` into an explicit list of Zarr array names.

    A Hydra override such as ``env.dataset_keys=[qpos,qvel]`` can arrive as
    the literal string ``"[qpos,qvel]"`` rather than a parsed list. Iterating
    that string yields single characters, which surfaces much later as a
    confusing ``KeyError: "Key '[' not found"``, so normalize it here instead.
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


class ExpertDataPlane:
    """Reference dataset, fast paths, frame refresh, and expert sampling.

    See the module docstring for the ownership split and lifecycle.
    """

    def __init__(self, cfg: Any, env: ImitationRLEnv) -> None:
        """Phase 1: load the dataset and build the trajectory manager.

        Runs in the env's ``__init__`` before ``super().__init__`` (the
        CommandManager is not built yet), exactly where the legacy env loads
        its reference data.
        """
        self._env = env
        device = torch.device(cfg.sim.device)
        num_envs = int(cfg.scene.num_envs)

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
                    "`env.lafan1_manifest_path=/path/to/manifest.json` for "
                    "normal use. If you are configuring the env "
                    "programmatically, provide explicit "
                    "`loader_kwargs.dataset.trajectories.lafan1_csv` entries "
                    "and `dataset_path` before env creation."
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
                        "refresh_zarr_dataset=True requires loader_type + "
                        "loader_kwargs so the zarr dataset can be rebuilt."
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
                "Either dataset_path must be provided, or loader_type + "
                "loader_kwargs must be provided to create a new dataset."
            )

        # Trajectory selection is declared on the reference command channel:
        # the schedule (including the `custom` selector an evaluation driver or
        # per-goal collector supplies) and the start frame come from there, so
        # nothing about which motion an env resets onto is decided here.
        reference_channel = cfg.command_interface.reference
        selection = reference_channel.selection
        reset_schedule = str(selection.schedule)
        custom_reset_fn = selection.custom_fn
        if reset_schedule == ResetSchedule.CUSTOM and custom_reset_fn is None:
            raise ValueError(
                "The reference channel declares schedule='custom' without a "
                "custom_fn(env_ids, num_trajectories) selector."
            )
        wrap_steps = getattr(cfg, "wrap_steps", False)
        # The env parses the reset-start frame once (`_reference_start_frame`);
        # the trajectory manager's initial cursor uses the same value.
        reference_start_frame = env._reference_start_frame
        self._expert_anchor_body_name = str(reference_channel.anchor_body_name).strip()
        if not self._expert_anchor_body_name:
            raise ValueError(
                "The reference command channel's anchor_body_name must be non-empty."
            )

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
                    "reference_joint_names length mismatch with replay buffer "
                    f"qpos. Expected {expected_reference_joint_dim} joints from "
                    f"qpos, got {len(reference_joint_names)} reference names."
                )

        assert len(reference_joint_names) > 0 and len(target_joint_names) > 0, (
            "Reference and target joint names must have the length greater than 0"
        )
        self.reference_joint_names = reference_joint_names
        self.target_joint_names = target_joint_names
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
            custom_reset_fn=custom_reset_fn,
            reset_start_step=reference_start_frame,
            wrap_steps=wrap_steps,
            device=device,
            reference_joint_names=reference_joint_names,
            target_joint_names=target_joint_names,
        )

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

        self.reference_body_names: list[str] = []
        self.reference_site_names: list[str] = []
        self._expert_sampler_warned_unknown_terms: set[str] = set()
        self._expert_macro_feature_slices: dict[str, tuple[int, int]] | None = None

        self._load_reference_metadata(zarr_path)

    # ------------------------------------------------------------------
    # Phase 2: scene-dependent initialization.
    # ------------------------------------------------------------------

    def finalize(self, scene: Any, robot: Any) -> None:
        """Phase 2: bind the plane to the live scene (post-``super().__init__``).

        Mirrors the legacy env's post-super initialization: retarget the
        trajectory manager to the live articulation joint order, capture env
        origins and default joint state, finalize reference body names, build
        the MDP fast-path caches, and resolve the MPJPE metric bodies.
        """
        self._align_reference_target_joints_to_articulation()
        self._expert_env_origins = scene.env_origins.clone()
        self._expert_default_joint_pos = robot.data.default_joint_pos.torch.clone()
        self._expert_default_joint_vel = robot.data.default_joint_vel.torch.clone()
        self._finalize_reference_body_names()
        self._initialize_mdp_fast_paths()
        self._initialize_mpjpe_metric()

    # ------------------------------------------------------------------
    # Reference metadata / joint alignment.
    # ------------------------------------------------------------------

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
        robot = self._robot()
        robot_joint_names = list(robot.joint_names)
        if list(tm.target_joint_names) == robot_joint_names:
            return
        if sorted(tm.target_joint_names) != sorted(robot_joint_names):
            raise RuntimeError(
                "Configured target_joint_names and articulation joint names "
                "are different sets; cannot retarget the reference. "
                f"Difference: {sorted(set(tm.target_joint_names) ^ set(robot_joint_names))}"
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
            tm.state_device,
        )
        tm.target_mask = torch.zeros(
            len(robot_joint_names), dtype=torch.bool, device=tm.state_device
        )
        tm.target_mask[tm.ref_to_target_map] = True
        # Refresh everything that already captured target-ordered joint data.
        self.current_expert_frame = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(advance=False)
        )
        self._build_reward_input_cache(device=torch.device(self._env.device))

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

    def _finalize_reference_body_names(self) -> None:
        """Improve reference body-name mapping for datasets that only provide generic names."""
        ref_body_pos = self.current_expert_frame.get("xpos")
        if ref_body_pos is None:
            ref_body_pos = self.current_expert_frame.get("body_pos_w")
        if ref_body_pos is None or ref_body_pos.ndim < 3:
            return

        num_reference_bodies = int(ref_body_pos.shape[1])
        robot_body_names = list(self._robot().body_names)

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
        cfg_body_names = list(getattr(self._env.cfg, "reference_body_names", []) or [])
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

    def _robot(self) -> Any:
        """The live robot articulation, resolving the scene when needed.

        The CommandManager builds the command terms inside
        ``ManagerBasedRLEnv.load_managers`` (during ``super().__init__``) and
        its manager table print accesses ``term.command`` eagerly -- before
        the env has set its post-super ``robot`` attribute. Mirroring the
        legacy env's guard (``if not hasattr(self, "robot"): self.robot =
        self.scene["robot"]``), every robot read resolves from the scene when
        the env attribute is not yet assigned.
        """
        robot = getattr(self._env, "robot", None)
        if robot is None:
            robot = self._env.scene["robot"]
        return robot

    @staticmethod
    def _normalize_body_name_for_matching(name: str) -> str:
        """Normalize body names for tolerant cross-dataset matching."""
        lowered = name.lower()
        if lowered.endswith("_link"):
            lowered = lowered[:-5]
        return lowered

    # ------------------------------------------------------------------
    # MPJPE metric computation (the `motion` command term delegates here).
    # ------------------------------------------------------------------

    def _initialize_mpjpe_metric(self) -> None:
        """Resolve the bodies used for the root-relative MPJPE training metric.

        This exists because a metric cannot be expressed as a ``RewTerm``
        with ``weight=0.0``: :meth:`RewardManager.compute` skips zero-weight
        terms without calling them, so such a term logs a constant zero. The
        v2 ``motion`` command term owns the ``Metrics/motion/*`` logging and
        delegates the per-step computation to :meth:`_compute_mpjpe_metric`,
        so this resolves only the measurement body set.
        """
        body_names = list(
            self._env.cfg.command_interface.reference.mpjpe_body_names or []
        )
        self._mpjpe_metric_body_names: list[str] = []
        self._mpjpe_metric_body_ids: torch.Tensor | None = None
        if not body_names:
            return

        missing = [
            name for name in body_names if name not in set(self.reference_body_names)
        ]
        if missing:
            raise ValueError(
                "mpjpe_metric_body_names contains bodies absent from the "
                f"reference: {missing}. The metric compares robot bodies "
                "against reference bodies of the same name, so every entry "
                "must exist in both."
            )
        body_ids, resolved = self._robot().find_bodies(body_names, preserve_order=True)
        if list(resolved) != body_names:
            raise RuntimeError(
                "Could not resolve the MPJPE metric bodies in order: "
                f"expected={body_names}, got={list(resolved)}."
            )
        self._mpjpe_metric_body_names = body_names
        self._mpjpe_metric_body_ids = torch.as_tensor(
            body_ids, dtype=torch.long, device=self._env.device
        )

    def _compute_mpjpe_metric(self) -> torch.Tensor | None:
        """Per-environment root-relative MPJPE in metres.

        Mirrors ``mdp.mpjpe_relative_body_pos_m`` and the closed-loop
        evaluators: both sides are expressed relative to their own root, so
        the value measures pose error rather than global drift.

        Kept in metres to match those two, matching ``tracking_mpjpe_m``; the
        conversion to millimetres happens once at the logging boundary.

        Note that only root *position* is subtracted, not root *orientation*,
        so a rotated root rigidly rotates every body within the root-relative
        frame and contributes an error of roughly (distance from root) x
        (rotation) per body. That is why this is non-zero on the first frame
        of an episode: the ``reset_reference_state`` event perturbs the
        initial root orientation by up to 0.1/0.1/0.2 rad, which alone
        measures about 39 mm on the G1's 14-body set, with a further 6 mm
        from the +/-0.1 rad joint noise. Measured with all reset
        randomization disabled the value is exactly 0.00 mm, so there is no
        systematic reference-versus-URDF body-frame offset underneath it.
        """
        if self._mpjpe_metric_body_ids is None:
            return None
        robot_pos_w = self._get_robot_body_pose_w_fast(self._mpjpe_metric_body_ids)[0]
        reference_pos_w = self._get_reference_body_pose_w_fast(
            self._mpjpe_metric_body_names
        )[0]
        robot_root_w = self._robot().data.root_state_w.torch[:, :3]
        reference_root_w = self._get_reference_root_state_w_fast()[0]
        robot_relative = robot_pos_w - robot_root_w[:, None, :]
        reference_relative = reference_pos_w - reference_root_w[:, None, :]
        return torch.linalg.vector_norm(
            robot_relative - reference_relative, dim=-1
        ).mean(dim=-1)

    # ------------------------------------------------------------------
    # MDP fast paths.
    # ------------------------------------------------------------------

    def _initialize_mdp_fast_paths(self) -> None:
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
        robot = self._robot()
        self._mdp_body_name_to_id = {
            name: idx for idx, name in enumerate(robot.body_names)
        }
        self._mdp_body_name_to_id_lower = {
            name.lower(): idx for idx, name in enumerate(robot.body_names)
        }
        self._mdp_body_name_to_id_normalized = {
            self._normalize_body_name_for_matching(name): idx
            for idx, name in enumerate(robot.body_names)
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
        self._mdp_all_body_ids_key = tuple(range(len(robot.body_names)))
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
            self._mdp_cache_step == self._env.common_step_counter
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
        self._mdp_cache_step = self._env.common_step_counter

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
            body_ids_t = torch.tensor(key, dtype=torch.long, device=self._env.device)
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
            joint_ids_t = torch.tensor(key, dtype=torch.long, device=self._env.device)
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

        body_ids = torch.tensor(ref_indices, dtype=torch.long, device=self._env.device)
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
            robot = self._robot()
            anchor_state = (
                robot.data.body_pos_w.torch[:, anchor_body_id],
                robot.data.body_quat_w.torch[:, anchor_body_id],
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
        robot = self._robot()
        if isinstance(body_ids_t, slice):
            body_pose = (
                robot.data.body_pos_w.torch,
                robot.data.body_quat_w.torch,
            )
        else:
            body_pose = (
                robot.data.body_pos_w.torch.index_select(1, body_ids_t),
                robot.data.body_quat_w.torch.index_select(1, body_ids_t),
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
        robot = self._robot()
        if isinstance(body_ids_t, slice):
            body_velocity = (
                robot.data.body_ang_vel_w.torch,
                robot.data.body_lin_vel_w.torch,
            )
        else:
            body_velocity = (
                robot.data.body_ang_vel_w.torch.index_select(1, body_ids_t),
                robot.data.body_lin_vel_w.torch.index_select(1, body_ids_t),
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

    # ------------------------------------------------------------------
    # Expert motion commands.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Alignment transforms.
    # ------------------------------------------------------------------

    def _get_reference_alignment_transform(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Unitree-style fixed placement from dataset frame to sim world."""
        env_origins = getattr(self, "_expert_env_origins", None)
        if env_origins is None:
            env_origins = self._env.scene.env_origins
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

    # ------------------------------------------------------------------
    # Frame refresh.
    # ------------------------------------------------------------------

    def _refresh_current_expert_frame(
        self, env_ids: torch.Tensor | None = None, *, advance: bool = False
    ) -> None:
        tm = self.trajectory_manager
        if env_ids is None:
            sampled_env_ids = torch.arange(
                self._env.num_envs, device=tm.state_device, dtype=torch.long
            )
        else:
            sampled_env_ids = env_ids.to(device=tm.state_device, dtype=torch.long)
        sampled_local_steps = tm.env_step.index_select(0, sampled_env_ids)
        reference = _convert_reference_quats_to_xyzw(
            self.trajectory_manager.sample(env_ids=env_ids, advance=advance)
        )
        if env_ids is None or self.current_expert_frame is None:
            self.current_expert_frame = reference
            self._current_reference_local_step.copy_(
                sampled_local_steps.to(device=self._env.device, dtype=torch.long)
            )
        else:
            self._index_copy_reference_rows_(
                self.current_expert_frame, reference, env_ids
            )
            self._current_reference_local_step.index_copy_(
                0,
                env_ids.to(device=self._env.device, dtype=torch.long),
                sampled_local_steps.to(device=self._env.device, dtype=torch.long),
            )
        self._invalidate_mdp_cache()

    def current_reference_is_final_frame(self) -> torch.Tensor:
        """Return true for envs whose current reward/obs reference is terminal."""
        tm = self.trajectory_manager
        traj_ranks = tm.env_traj_rank.to(device=self._env.device, dtype=torch.long)
        final_steps = (tm.length.index_select(0, traj_ranks) - 1).to(
            device=self._env.device, dtype=torch.long
        )
        return self._current_reference_local_step >= final_steps

    def _current_local_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        tm = self.trajectory_manager
        return tm.env_step[env_ids.to(device=tm.state_device, dtype=torch.long)].to(
            device=self._env.device, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Offline dataset mapper params.
    # ------------------------------------------------------------------

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
        action_term = self._env.action_manager.get_term("joint_pos")
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
        robot = self._robot()
        robot.find_joints(action_joint_names, preserve_order=True)
        action_offset_pool = self._resolve_offline_action_vector_pool(
            action_term._offset,
            name="JointPositionAction offset",
            width=action_width,
            device=self._env.device,
        )
        action_scale = self._resolve_offline_static_action_vector(
            action_term._scale,
            name="JointPositionAction scale",
            width=action_width,
            device=self._env.device,
        )
        if torch.any(action_scale.abs() <= 1.0e-8):
            raise ValueError("JointPositionAction scale must not contain zeros.")
        default_root_height = float(
            robot.data.default_root_state.torch[0, 2].detach().cpu().item()
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

    # ------------------------------------------------------------------
    # Rollout state alignment metrics (diagnostic log).
    # ------------------------------------------------------------------

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

    def _compute_rollout_reference_state_log(self) -> dict[str, float]:
        """Compare the post-step robot state against the aligned reference next state."""
        if self.current_expert_frame is None:
            return {}

        next_joint_pos = self.current_expert_frame.get(("next", "joint_pos"))
        next_joint_vel = self.current_expert_frame.get(("next", "joint_vel"))
        if next_joint_pos is None or next_joint_vel is None:
            return {}

        robot = self._robot()
        metrics = self._compute_rollout_state_alignment_metrics(
            robot.data.joint_pos.torch,
            next_joint_pos.to(device=self._env.device, dtype=torch.float32),
            prefix="rollout_state/joint_pos",
        )
        metrics.update(
            self._compute_rollout_state_alignment_metrics(
                robot.data.joint_vel.torch,
                next_joint_vel.to(device=self._env.device, dtype=torch.float32),
                prefix="rollout_state/joint_vel",
            )
        )
        return metrics

    # ------------------------------------------------------------------
    # Expert window / goal building and batch / macro sampling.
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_nested_key(key: Any) -> tuple[str, ...]:
        """Normalize a nested key to tuple form."""
        if isinstance(key, tuple):
            return key
        return (key,)

    @staticmethod
    def _denormalize_nested_key(key_parts: tuple[str, ...]) -> Any:
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
    def _joint_ids_cache_key(
        joint_ids: torch.Tensor | Sequence[int] | slice,
    ) -> object:
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
            expert_frame.to(self._env.device),
            env_ids_tm.to(self._env.device),
            global_indices.to(self._env.device),
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
        return nonempty_ranks.to(device=tm.state_device, dtype=torch.long)

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
            device=self.trajectory_manager.state_device, dtype=torch.long
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
        traj_ranks_tm = traj_ranks.to(device=tm.state_device, dtype=torch.long)
        local_steps_tm = local_steps.to(device=tm.state_device, dtype=torch.long)
        if tuple(traj_ranks_tm.shape) != tuple(local_steps_tm.shape):
            raise ValueError(
                "traj_ranks and local_steps must have matching shapes for "
                "expert macro sampling."
            )
        window_offsets = torch.arange(
            -past_steps,
            future_steps + 1,
            device=tm.state_device,
            dtype=torch.long,
        )
        lengths = tm.length.index_select(0, traj_ranks_tm).clamp(min=1)
        max_step = lengths - 1
        window_steps = local_steps_tm.unsqueeze(1) + window_offsets.unsqueeze(0)
        window_steps = window_steps.clamp(min=0)
        window_steps = torch.minimum(window_steps, max_step.unsqueeze(1))
        global_indices = (
            tm.start.index_select(0, traj_ranks_tm).unsqueeze(1) + window_steps
        )
        expert_window = tm.rb[global_indices.to(device=tm.storage_device)]
        if getattr(tm, "_device", None) is not None:
            expert_window = expert_window.to(tm.device)
        expert_window = tm.attach_reference_fields(expert_window, use_buffers=False)
        return _convert_reference_quats_to_xyzw(expert_window.to(self._env.device))

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
        total = int(tm.end.max().item())
        if total <= 0:
            raise RuntimeError(
                "Trajectory manager has no transitions; cannot build "
                "reward_input cache."
            )
        global_indices = torch.arange(
            total, device=tm.storage_device, dtype=torch.int64
        )
        reference = tm.rb[global_indices]
        if tm.device is not None:
            reference = reference.to(tm.device)
        reference = tm.attach_reference_fields(reference, use_buffers=False)
        joint_pos = reference.get("joint_pos")
        joint_vel = reference.get("joint_vel")
        if joint_pos is None or joint_vel is None:
            raise RuntimeError(
                "reward_input cache build failed: trajectory manager did not "
                "produce joint_pos/joint_vel."
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
        env_ids_tm = env_ids.to(device=tm.state_device, dtype=torch.long)
        global_indices_tm = global_indices.to(device=tm.state_device, dtype=torch.long)
        traj_ranks = tm.env_traj_rank[env_ids_tm]
        local_steps = global_indices_tm - tm.start[traj_ranks]
        return local_steps.to(device=self._env.device, dtype=torch.long)

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
        env_ids_tm = env_ids.to(device=tm.state_device, dtype=torch.long)
        local_steps_tm = local_steps.to(device=tm.state_device, dtype=torch.long)
        window_offsets = torch.arange(
            -past_steps,
            future_steps + 1,
            device=tm.state_device,
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
        return _convert_reference_quats_to_xyzw(expert_window.to(self._env.device))

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
                "Expert batch is missing body pose fields required for expert "
                "observations."
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

        scene = getattr(self._env, "scene", None)
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
        action_manager = getattr(self._env, "action_manager", None)
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
            zero_anchor_pos = torch.zeros((batch_size, 3), device=self._env.device)
            identity_rot6d = torch.zeros((batch_size, 6), device=self._env.device)
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
                    "Failed to transform expert-window anchor quaternion for "
                    "rollout observations."
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
                        "Failed to transform expert-window body quaternion for "
                        "rollout observations."
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
            # Keypoint terms use the same anchor-frame pose calculation under
            # a separately keyed body-set cache. Exposing position and
            # orientation independently lets configs select point targets or
            # complete poses.
            terms["expert_keypoint_pos_b"] = body_pos_flat
            terms["expert_keypoint_ori_b"] = body_ori_flat
        return terms

    def _expert_macro_feature_term_order(self) -> tuple[str, ...]:
        """Expert-window terms that make up one DiffSR macro-state frame.

        Configurable because the skill encoder's input width is defined by
        this selection: the default gives 58+3+6 = 67/frame -> 670 for a
        10-frame window, byte-identical to the full-body packet. Selecting
        ``expert_motion_qpos`` instead gives 29+3+6 = 38 -> 380, byte-identical
        to the root_qpos packet, which is what a GR00T-style whole-body
        qpos+root latent interface needs. Nothing in the DiffSR trainer has to
        know -- it reads whatever macro state the environment produces.
        """
        configured = getattr(self._env.cfg, "expert_macro_state_terms", None)
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
                self._env._command_ee_body_names,
            ),
            (
                {"expert_keypoint_pos_b", "expert_keypoint_ori_b"},
                self._env._command_keypoint_body_names,
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

        env_ids = torch.arange(
            self._env.num_envs, device=self._env.device, dtype=torch.long
        )
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
        joint_ids: torch.Tensor | Sequence[int] | slice = slice(None),
        anchor_body_name: str = "torso_link",
        reference_body_names: Sequence[str] = (),
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self._get_current_expert_window_terms(
            past_steps=int(past_steps),
            future_steps=int(future_steps),
            joint_ids=joint_ids,
            anchor_body_name=anchor_body_name,
            reference_body_names=reference_body_names,
        )[term_name]
        if env_ids is None:
            return value
        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
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

        env_ids = torch.arange(
            self._env.num_envs, device=self._env.device, dtype=torch.long
        )
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
        env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        return value.index_select(0, env_ids)

    def _map_requested_expert_observations(
        self,
        expert_frame: TensorDict,
        env_ids: torch.Tensor,
        obs_keys: Sequence[Any],
        *,
        context: str,
        prefix: tuple[str, ...] = (),
        local_steps: torch.Tensor | None = None,
        global_indices: torch.Tensor | None = None,
        past_steps: int,
        future_steps: int,
    ) -> dict[Any, torch.Tensor] | None:
        mapped_values: dict[Any, torch.Tensor] = {}
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
                        "Expert mapper received expert_window requests without "
                        "trajectory-local steps."
                    )
                    return None
                if term_name in {"expert_ee_pos_b", "expert_ee_ori_b"}:
                    reference_body_names = self._env._command_ee_body_names
                elif term_name in {
                    "expert_keypoint_pos_b",
                    "expert_keypoint_ori_b",
                }:
                    reference_body_names = self._env._command_keypoint_body_names
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
                        "Expert mapper received expert_goal requests without "
                        "trajectory-local steps."
                    )
                    return None
                goal_steps = int(self._env._latent_goal_steps)
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
                if group_name == "policy" and term_name in {
                    "expert_motion",
                    "expert_motion_qpos",
                    "expert_anchor_pos_b",
                    "expert_anchor_ori_b",
                    "expert_ee_pos_b",
                    "expert_ee_ori_b",
                    "expert_keypoint_pos_b",
                    "expert_keypoint_ori_b",
                }:
                    # Windowed command keys (the encoder families read the
                    # policy command terms): serve the windowed expert
                    # command, mirroring the rollout side. The critic's
                    # command terms stay single-frame (raw state below).
                    if len(prefix) > 0:
                        unknown_terms.append(term_name)
                        continue
                    if local_steps is None:
                        logger.warning(
                            "Expert mapper received policy command-window "
                            "requests without trajectory-local steps."
                        )
                        return None
                    if term_name in {"expert_ee_pos_b", "expert_ee_ori_b"}:
                        reference_body_names = self._env._command_ee_body_names
                    elif term_name in {
                        "expert_keypoint_pos_b",
                        "expert_keypoint_ori_b",
                    }:
                        reference_body_names = self._env._command_keypoint_body_names
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
                else:
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
        required_keys: Sequence[Any],
        *,
        past_steps: int,
        future_steps: int,
    ) -> TensorDict | None:
        if batch_size <= 0:
            return None
        if len(required_keys) == 0:
            return TensorDict({}, batch_size=[batch_size], device=self._env.device)

        dedup_required_keys = list(dict.fromkeys(required_keys))
        current_obs_keys: list[Any] = []
        next_obs_keys: list[Any] = []
        needs_action = False

        for key in dedup_required_keys:
            key_tuple = self._normalize_nested_key(key)
            if key_tuple in (("action",), ("expert_action",)):
                needs_action = True
                continue
            if len(key_tuple) > 0 and key_tuple[0] == "next":
                if len(key_tuple) < 2:
                    continue
                next_obs_keys.append(self._denormalize_nested_key(key_tuple[1:]))
                continue
            current_obs_keys.append(self._denormalize_nested_key(key_tuple))

        expert_batch = TensorDict({}, batch_size=[batch_size], device=self._env.device)
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
            sampled_action = sampled_action.to(self._env.device)
            # The historical `expert_action` name (used by the bilinear
            # offline policy-BC pretrain) is an alias of the recorded
            # `action`; serve whichever key(s) the caller requested.
            requested = {self._normalize_nested_key(key) for key in dedup_required_keys}
            if ("action",) in requested:
                expert_batch.set("action", sampled_action)
            if ("expert_action",) in requested:
                expert_batch.set("expert_action", sampled_action)

        return expert_batch

    def sample_expert_batch(
        self, batch_size: int, required_keys: Sequence[Any]
    ) -> TensorDict | None:
        """Sample an expert batch for imitation algorithms from trajectory manager."""
        return self._sample_expert_batch_impl(
            batch_size,
            required_keys,
            past_steps=int(self._env._latent_patch_past_steps),
            future_steps=int(self._env._latent_patch_future_steps),
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
        expert_motion, expert_anchor_pos_b, and expert_anchor_ori_b. The
        sampled window is clamped at trajectory boundaries by
        ``_sample_expert_window_slice``, matching existing expert-window
        behavior. If ``trajectory_ranks`` is provided, samples are drawn only
        from those explicit trajectory ranks and ``split`` is ignored.
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
                trajectory_ranks, device=tm.state_device, dtype=torch.long
            ).reshape(-1)
            if int(selected_ranks.numel()) == 0:
                raise ValueError("trajectory_ranks must select at least one rank.")
            max_rank = int(getattr(tm, "num_trajectories", tm.length.numel()))
            invalid = selected_ranks[
                (selected_ranks < 0) | (selected_ranks >= max_rank)
            ]
            if int(invalid.numel()) > 0:
                bad = sorted({int(item) for item in invalid.detach().cpu().tolist()})
                raise ValueError(
                    f"trajectory_ranks out of range [0, {max_rank - 1}]: {bad}."
                )
            lengths_for_selected = tm.length.index_select(0, selected_ranks)
            empty = selected_ranks[lengths_for_selected <= 0]
            if int(empty.numel()) > 0:
                bad = sorted({int(item) for item in empty.detach().cpu().tolist()})
                raise ValueError(f"trajectory_ranks include empty trajectories: {bad}.")

            choices = torch.randint(
                low=0,
                high=int(selected_ranks.numel()),
                size=(batch_size,),
                device=tm.state_device,
                dtype=torch.long,
            )
            traj_ranks_tm = selected_ranks.index_select(0, choices)
            lengths = tm.length.index_select(0, traj_ranks_tm).clamp(min=1)
            local_steps_tm = torch.floor(
                torch.rand(batch_size, device=tm.state_device)
                * lengths.to(dtype=torch.float32)
            ).to(dtype=torch.long)
            env_ids = torch.arange(
                batch_size, device=self._env.device, dtype=torch.long
            )
            local_steps = local_steps_tm.to(device=self._env.device, dtype=torch.long)
            traj_rank = traj_ranks_tm.to(device=self._env.device, dtype=torch.long)
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
                    0, env_ids.to(device=tm.state_device, dtype=torch.long)
                ).to(device=self._env.device, dtype=torch.long)
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
                    device=tm.state_device,
                    dtype=torch.long,
                )
                traj_ranks_tm = split_ranks.index_select(0, choices)
                lengths = tm.length.index_select(0, traj_ranks_tm).clamp(min=1)
                local_steps_tm = torch.floor(
                    torch.rand(batch_size, device=tm.state_device)
                    * lengths.to(dtype=torch.float32)
                ).to(dtype=torch.long)
                env_ids = torch.arange(
                    batch_size, device=self._env.device, dtype=torch.long
                )
                local_steps = local_steps_tm.to(
                    device=self._env.device, dtype=torch.long
                )
                traj_rank = traj_ranks_tm.to(device=self._env.device, dtype=torch.long)
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
                    f"expected {expected_history}, got "
                    f"{tuple(state_history.shape)}."
                )
            hl_payload["state_history"] = state_history
        hl = TensorDict(
            hl_payload,
            batch_size=[batch_size],
            device=self._env.device,
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size], device=self._env.device)

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
                self._env.num_envs, device=self._env.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self._env.device, dtype=torch.long
            ).reshape(-1)
        batch_size = int(env_ids_t.numel())
        if batch_size <= 0:
            raise ValueError("env_ids must select at least one environment.")

        local_steps = self._current_local_steps(env_ids_t)
        tm = self.trajectory_manager
        traj_rank = tm.env_traj_rank.index_select(
            0, env_ids_t.to(device=tm.state_device, dtype=torch.long)
        ).to(device=self._env.device, dtype=torch.long)
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
                "Current expert macro sampler produced invalid future_window "
                f"shape: expected {expected_window}, got "
                f"{tuple(future_window.shape)}."
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
        }
        if state_history is not None:
            expected_history = (batch_size, state_history_steps + 1, state_dim)
            if tuple(state_history.shape) != expected_history:
                raise ValueError(
                    "Current expert macro sampler produced invalid "
                    f"state_history shape: expected {expected_history}, got "
                    f"{tuple(state_history.shape)}."
                )
            hl_payload["state_history"] = state_history
        hl = TensorDict(
            hl_payload,
            batch_size=[batch_size],
            device=self._env.device,
        )
        return TensorDict({"hl": hl}, batch_size=[batch_size], device=self._env.device)

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

        Joint-state, EE, and sparse-keypoint components are replaced
        independently, so the same path supports root+qpos, root+five-keypoint
        pose, and composed ablations. Rollout-context anchor terms remain
        expert-relative-to-robot and the future window/target remain
        expert-derived.
        """
        horizon_steps = int(horizon_steps)
        batch = self.current_expert_macro_transition_batch(
            horizon_steps,
            env_ids=env_ids,
            state_history_steps=state_history_steps,
        )
        if env_ids is None:
            env_ids_t = torch.arange(
                self._env.num_envs, device=self._env.device, dtype=torch.long
            )
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self._env.device, dtype=torch.long
            ).reshape(-1)

        slices = self.expert_macro_feature_slices(horizon_steps)
        selected = set(slices)
        robot = self._robot()
        joint_pos = robot.data.joint_pos.torch.index_select(0, env_ids_t).to(
            device=self._env.device, dtype=torch.float32
        )
        joint_vel = robot.data.joint_vel.torch.index_select(0, env_ids_t).to(
            device=self._env.device, dtype=torch.float32
        )
        achieved_terms: dict[str, torch.Tensor] = {
            "expert_motion": torch.cat([joint_pos, joint_vel], dim=-1),
            "expert_motion_qpos": joint_pos,
        }

        body_groups = (
            (
                "expert_ee_pos_b",
                "expert_ee_ori_b",
                self._env._command_ee_body_names,
            ),
            (
                "expert_keypoint_pos_b",
                "expert_keypoint_ori_b",
                self._env._command_keypoint_body_names,
            ),
        )
        for pos_term, ori_term, body_names in body_groups:
            if not selected.intersection({pos_term, ori_term}):
                continue
            if not body_names:
                raise ValueError(
                    f"Macro interface selected {pos_term!r}/{ori_term!r}, but "
                    "its configured robot body set is empty."
                )
            body_ids = self._get_robot_body_ids_by_name_fast(body_names)
            body_pos_b, body_quat_b = self._get_robot_body_state_in_anchor_frame_fast(
                body_ids,
                self._expert_anchor_body_name,
            )
            body_pos_b = body_pos_b.index_select(0, env_ids_t)
            body_quat_b = body_quat_b.index_select(0, env_ids_t)
            achieved_terms[pos_term] = body_pos_b.reshape(len(env_ids_t), -1).to(
                device=self._env.device, dtype=torch.float32
            )
            achieved_terms[ori_term] = (
                _get_mdp_compiled_module()
                .quat_to_rot6d_flat(body_quat_b)
                .reshape(len(env_ids_t), -1)
                .to(device=self._env.device, dtype=torch.float32)
            )

        replacements = [name for name in slices if name in achieved_terms]
        if not replacements:
            raise RuntimeError(
                "Configured expert_macro_state_terms contain no achieved robot "
                "component; select joint, EE, or keypoint terms in addition to "
                "anchors."
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
        cached = self._expert_macro_feature_slices
        if cached is None:
            self.sample_expert_macro_transition_batch(
                batch_size=1,
                horizon_steps=horizon_steps,
                split="all",
            )
            cached = self._expert_macro_feature_slices
        if cached is None:
            raise RuntimeError("Expert macro feature slices are unavailable.")
        return {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in cached.items()
        }
