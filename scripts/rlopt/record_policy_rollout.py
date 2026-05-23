# ruff: noqa: E402
"""Record a deterministic RLOpt policy rollout to video and NPZ."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-Latent-v0")
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=["PPO", "SAC", "FASTSAC", "IPMD", "IPMD_SR", "IPMD_BILINEAR", "GAIL", "AMP", "ASE"],
)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output_npz", type=Path, required=True)
parser.add_argument(
    "--output_manifest",
    type=Path,
    default=None,
    help="Optional manifest path to write for the generated rollout NPZ files.",
)
parser.add_argument(
    "--motion_manifest",
    type=Path,
    default=None,
    help="Optional manifest used to condition the rollout env on a specific motion set.",
)
parser.add_argument(
    "--schema_reference_npz",
    type=Path,
    default=None,
    help=(
        "Optional NPZ whose state-key schema should be mirrored in the output. "
        "Action-label keys are still appended."
    ),
)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help=(
        "Number of vectorized envs/rollouts to collect in parallel. "
        "When >1, output_npz is used as a stem and one original-format NPZ is "
        "written per env with a rollout index suffix."
    ),
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--fps",
    type=float,
    default=0.0,
    help="Output fps. When <= 0, infer it from the env step dt.",
)
parser.add_argument(
    "--refresh_zarr_dataset",
    action="store_true",
    default=False,
    help="Rebuild the manifest-derived Zarr cache before rollout.",
)
parser.add_argument(
    "--keep_after_done",
    action="store_true",
    default=False,
    help="Continue/appended stepped states even if the env reports done or truncated.",
)
parser.add_argument(
    "--preserve_episode_length",
    action="store_true",
    default=False,
    help="Do not extend env.episode_length_s to cover the requested rollout steps.",
)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_folder", type=Path, default=None)
parser.add_argument("--video_length", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp


ALGORITHM_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "FASTSAC": FastSAC,
    "IPMD": IPMD,
    "IPMD_SR": IPMDSR,
    "IPMD_BILINEAR": IPMDBilinear,
    "GAIL": GAIL,
    "AMP": AMP,
    "ASE": ASE,
}

BODY_STATE_KEYS = {
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
}

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ppo_cfg_entry_point": "PPO",
    "rlopt_sac_cfg_entry_point": "SAC",
    "rlopt_fastsac_cfg_entry_point": "FASTSAC",
    "rlopt_ipmd_cfg_entry_point": "IPMD",
    "rlopt_ipmd_sr_cfg_entry_point": "IPMD_SR",
    "rlopt_ipmd_bilinear_cfg_entry_point": "IPMD_BILINEAR",
    "rlopt_gail_cfg_entry_point": "GAIL",
    "rlopt_amp_cfg_entry_point": "AMP",
    "rlopt_ase_cfg_entry_point": "ASE",
}


def resolve_agent_cfg_entry_point(task_name: str | None, algorithm: str) -> str:
    if task_name is None:
        return f"rlopt_{algorithm.lower()}_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    spec = gym.spec(task_id)
    if spec.kwargs.get(algo_entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point
    supported_algorithms = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    msg = (
        "Unsupported task/algo combination: "
        f"task '{task_id}' does not expose an RLOpt config for '{algorithm}'. "
        f"Supported RLOpt algorithms for this task: {supported_algorithms}."
    )
    raise ValueError(msg)


def _unwrap_imitation_env(env) -> ImitationRLEnv:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ImitationRLEnv):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, ImitationRLEnv):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap ImitationRLEnv.")


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _get_optional(td: TensorDictBase, key) -> torch.Tensor | None:
    try:
        value = td.get(key)
    except KeyError:
        return None
    return value if isinstance(value, torch.Tensor) else None


def _copy_env_vector(tensor: torch.Tensor, env_id: int = 0) -> np.ndarray:
    return _to_numpy(tensor[env_id]).astype(np.float32, copy=True)


def _disable_observation_corruption(env_cfg: object) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for group_name in ("policy", "critic", "expert_state", "expert_window", "reward_input"):
        group = getattr(observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _infer_output_fps(base_env: ImitationRLEnv, env_cfg: object) -> float:
    if args_cli.fps > 0.0:
        return float(args_cli.fps)

    step_dt = getattr(base_env, "step_dt", None)
    if step_dt is not None:
        step_dt = float(step_dt)
        if step_dt > 0.0:
            return 1.0 / step_dt

    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return 1.0 / (sim_dt * decimation)

    raise RuntimeError("Could not infer rollout fps. Pass --fps explicitly.")


def _configured_step_dt(env_cfg: object) -> float | None:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = float(getattr(sim_cfg, "dt", 0.0) or 0.0)
    decimation = int(getattr(env_cfg, "decimation", 1) or 1)
    if sim_dt > 0.0 and decimation > 0:
        return sim_dt * decimation
    return None


def _snapshot_robot_state(base_env: ImitationRLEnv, env_id: int = 0) -> dict[str, np.ndarray]:
    robot_data = base_env.robot.data
    return {
        "root_pos": _copy_env_vector(robot_data.root_pos_w, env_id=env_id),
        "root_quat": _copy_env_vector(robot_data.root_quat_w, env_id=env_id),
        "root_lin_vel": _copy_env_vector(robot_data.root_lin_vel_w, env_id=env_id),
        "root_ang_vel": _copy_env_vector(robot_data.root_ang_vel_w, env_id=env_id),
        "joint_pos": _copy_env_vector(robot_data.joint_pos, env_id=env_id),
        "joint_vel": _copy_env_vector(robot_data.joint_vel, env_id=env_id),
        "body_pos_w": _copy_env_vector(robot_data.body_pos_w, env_id=env_id),
        "body_quat_w": _copy_env_vector(robot_data.body_quat_w, env_id=env_id),
        "body_lin_vel_w": _copy_env_vector(robot_data.body_lin_vel_w, env_id=env_id),
        "body_ang_vel_w": _copy_env_vector(robot_data.body_ang_vel_w, env_id=env_id),
    }


def _stack_rollout_frames(frames: list[dict[str, np.ndarray]], fps: float) -> dict[str, np.ndarray]:
    arrays = {
        "fps": np.asarray([float(fps)], dtype=np.float32),
    }
    for key in (
        "root_pos",
        "root_quat",
        "root_lin_vel",
        "root_ang_vel",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        arrays[key] = np.stack([frame[key] for frame in frames], axis=0).astype(np.float32)

    arrays["qpos"] = np.concatenate(
        [arrays["root_pos"], arrays["root_quat"], arrays["joint_pos"]],
        axis=-1,
    ).astype(np.float32)
    arrays["qvel"] = np.concatenate(
        [arrays["root_lin_vel"], arrays["root_ang_vel"], arrays["joint_vel"]],
        axis=-1,
    ).astype(np.float32)
    return arrays


def _adapt_array_to_schema(
    key: str,
    array: np.ndarray,
    schema_array: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    target_shape = tuple(schema_array.shape)
    if array.ndim != schema_array.ndim:
        raise ValueError(
            f"Schema mismatch for '{key}': rollout ndim {array.ndim} does not "
            f"match reference ndim {schema_array.ndim}."
        )

    adapted = array
    note = None
    if key == "fps":
        if adapted.shape != target_shape:
            if adapted.size != int(np.prod(target_shape)):
                raise ValueError(
                    f"Schema mismatch for '{key}': rollout shape {adapted.shape} "
                    f"cannot be reshaped to reference shape {target_shape}."
                )
            adapted = adapted.reshape(target_shape)
    elif adapted.shape[1:] != target_shape[1:]:
        if (
            key in BODY_STATE_KEYS
            and adapted.ndim == 3
            and adapted.shape[1] >= target_shape[1]
            and adapted.shape[2:] == target_shape[2:]
        ):
            # Generic-body reference NPZs are mapped to robot.body_names[:N] by the env.
            adapted = adapted[:, : target_shape[1], :]
            note = (
                f"{key}: truncated rollout bodies from {array.shape[1]} to "
                f"{target_shape[1]} to mirror schema reference ordering"
            )
        else:
            raise ValueError(
                f"Schema mismatch for '{key}': rollout trailing shape "
                f"{adapted.shape[1:]} does not match reference trailing shape "
                f"{target_shape[1:]}."
            )

    if adapted.dtype != schema_array.dtype:
        adapted = adapted.astype(schema_array.dtype, copy=False)
    return adapted, note


def _filter_state_schema(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[str]]:
    schema_npz = args_cli.schema_reference_npz
    if schema_npz is None:
        return dict(arrays), []

    schema_npz = schema_npz.expanduser().resolve()
    if not schema_npz.is_file():
        raise FileNotFoundError(f"Schema reference NPZ not found: {schema_npz}")

    projected: dict[str, np.ndarray] = {}
    projection_notes: list[str] = []
    with np.load(schema_npz, allow_pickle=False) as schema_data:
        schema_keys = set(schema_data.files)
        requested_keys = [key for key in schema_data.files if key in arrays]
        if "fps" in schema_keys and "fps" not in requested_keys:
            requested_keys.insert(0, "fps")
        if not any(key in requested_keys for key in ("qpos", "joint_pos", "body_pos_w")):
            raise ValueError(
                f"Schema reference NPZ {schema_npz} does not share a supported state key "
                "with the rollout arrays."
            )
        for key in requested_keys:
            projected_array, note = _adapt_array_to_schema(key, arrays[key], schema_data[key])
            projected[key] = projected_array
            if note is not None:
                projection_notes.append(note)

    return projected, projection_notes


def _output_path_for_rollout(
    output_npz: Path,
    rollout_index: int,
    total_rollouts: int,
) -> Path:
    if total_rollouts == 1:
        return output_npz

    suffix = output_npz.suffix or ".npz"
    stem = output_npz.stem if output_npz.suffix else output_npz.name
    return output_npz.with_name(f"{stem}_rollout{rollout_index:04d}{suffix}")


def _optional_flat_tensor(
    td: TensorDictBase,
    key,
    *,
    num_envs: int,
    default: float | bool,
) -> torch.Tensor:
    value = _get_optional(td, key)
    if value is None:
        return torch.full((num_envs,), default)

    flat = value.detach().reshape(-1).cpu()
    if flat.numel() == 1 and num_envs > 1:
        flat = flat.expand(num_envs)
    if flat.numel() < num_envs:
        raise RuntimeError(
            f"Expected at least {num_envs} values for tensordict key {key}, got {flat.numel()}."
        )
    return flat[:num_envs]


def _write_manifest(
    *,
    manifest_path: Path,
    output_paths: list[Path],
    output_fps: float,
    metadata: dict[str, object],
) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    manifest_entries = []
    for rollout_index, output_path in enumerate(output_paths):
        entry_path = output_path.expanduser().resolve()
        manifest_entries.append(
            {
                "name": f"{entry_path.stem}",
                "path": os.path.relpath(entry_path, manifest_path.parent),
                "input_fps": float(output_fps),
                "frame_range": [1, _npz_frame_count(entry_path)],
                "rollout_index": int(rollout_index),
            }
        )

    manifest = {
        "dataset_name": manifest_path.stem,
        "dataset": {"trajectories": {"lafan1_csv": manifest_entries}},
        "metadata": {
            **metadata,
            "num_motions": len(manifest_entries),
            "output_fps": float(output_fps),
            "paths_are_relative_to_manifest": True,
            "recorded_action_key": "action",
            "transition_action_key": "transition_action",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote rollout manifest: {manifest_path}")


def _npz_frame_count(npz_path: Path) -> int:
    with np.load(npz_path, allow_pickle=False) as data:
        if "joint_pos" in data.files:
            return int(data["joint_pos"].shape[0])
        if "qpos" in data.files:
            return int(data["qpos"].shape[0])
        if "body_pos_w" in data.files:
            return int(data["body_pos_w"].shape[0])
    raise ValueError(f"Could not infer frame count from rollout NPZ: {npz_path}")


def _write_rollout_npz(
    *,
    output_npz: Path,
    rollout_index: int,
    total_rollouts: int,
    env_id: int,
    frames: list[dict[str, np.ndarray]],
    transition_actions: list[np.ndarray],
    transition_targets: list[np.ndarray],
    transition_rewards: list[float],
    transition_dones: list[bool],
    transition_terminated: list[bool],
    transition_truncated: list[bool],
    transition_local_steps: list[int],
    output_fps: float,
    action_dim: int,
    action_joint_names: list[str],
    base_metadata: dict[str, object],
    stop_reason: str,
) -> Path | None:
    transition_count = len(transition_actions)
    if transition_count <= 0:
        print(f"[WARNING] Skipping rollout {rollout_index}: no valid transitions recorded.")
        return None

    frame_count = transition_count + 1
    state_arrays = _stack_rollout_frames(frames=frames, fps=output_fps)
    arrays, schema_projection_notes = _filter_state_schema(state_arrays)

    transition_action_array = np.stack(transition_actions, axis=0).astype(np.float32)
    transition_target_array = np.stack(transition_targets, axis=0).astype(np.float32)
    action_valid = np.zeros((frame_count,), dtype=np.bool_)
    action_valid[:transition_count] = True
    full_actions = np.zeros((frame_count, action_dim), dtype=np.float32)
    full_targets = np.zeros((frame_count, action_dim), dtype=np.float32)
    full_actions[:transition_count] = transition_action_array
    full_targets[:transition_count] = transition_target_array
    full_actions[transition_count:] = transition_action_array[-1]
    full_targets[transition_count:] = transition_target_array[-1]

    metadata = {
        **base_metadata,
        "rollout_index": int(rollout_index),
        "env_id": int(env_id),
        "transition_count": int(transition_count),
        "frame_count": int(frame_count),
        "saved_state_keys": sorted(arrays),
        "stop_reason": stop_reason,
        "stopped_on_done": stop_reason in {"done", "terminated", "truncated"},
        "schema_projection": schema_projection_notes,
    }
    arrays["action"] = full_actions
    arrays["action_valid"] = action_valid
    arrays["transition_action"] = transition_action_array
    arrays["action_joint_pos_target"] = full_targets
    arrays["transition_action_joint_pos_target"] = transition_target_array
    arrays["action_joint_names"] = np.asarray(action_joint_names)
    arrays["transition_reward"] = np.asarray(transition_rewards, dtype=np.float32)
    arrays["transition_done"] = np.asarray(transition_dones, dtype=np.bool_)
    arrays["transition_terminated"] = np.asarray(transition_terminated, dtype=np.bool_)
    arrays["transition_truncated"] = np.asarray(transition_truncated, dtype=np.bool_)
    arrays["transition_local_step"] = np.asarray(transition_local_steps, dtype=np.int64)
    arrays["action_label_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    arrays["rollout_metadata_json"] = arrays["action_label_metadata_json"]

    output_path = _output_path_for_rollout(output_npz, rollout_index, total_rollouts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[INFO] Wrote rollout NPZ: {output_path}")
    print(
        "[INFO] rollout stats: "
        f"rollout={rollout_index}, frames={frame_count}, "
        f"transitions={transition_count}, fps={output_fps:.3f}, "
        f"action_mean_abs={float(np.mean(np.abs(full_actions[action_valid]))):.4f}, "
        f"stop_reason={stop_reason}"
    )
    return output_path


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive.")
    if args_cli.steps <= 0:
        raise ValueError("--steps must be positive.")
    steps = int(args_cli.steps)
    num_envs = int(args_cli.num_envs)

    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    output_npz = args_cli.output_npz.expanduser().resolve()
    motion_manifest = (
        args_cli.motion_manifest.expanduser().resolve()
        if args_cli.motion_manifest is not None
        else None
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if motion_manifest is not None and not motion_manifest.is_file():
        raise FileNotFoundError(f"Motion manifest not found: {motion_manifest}")

    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", None)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(output_npz.parent)
    if motion_manifest is not None:
        if not hasattr(env_cfg, "lafan1_manifest_path"):
            raise TypeError(f"Task {args_cli.task} does not support --motion_manifest.")
        env_cfg.lafan1_manifest_path = str(motion_manifest)
        if hasattr(env_cfg, "_resolve_manifest_config"):
            env_cfg._resolve_manifest_config()
    if hasattr(env_cfg, "refresh_zarr_dataset"):
        env_cfg.refresh_zarr_dataset = bool(args_cli.refresh_zarr_dataset)
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = 0
    if hasattr(env_cfg, "random_reset_full_trajectory"):
        env_cfg.random_reset_full_trajectory = False
    if hasattr(env_cfg, "random_reset_step_min"):
        env_cfg.random_reset_step_min = 0
    if hasattr(env_cfg, "random_reset_step_max"):
        env_cfg.random_reset_step_max = 0
    if hasattr(env_cfg, "reset_schedule"):
        env_cfg.reset_schedule = "sequential"
    if hasattr(env_cfg, "wrap_steps"):
        env_cfg.wrap_steps = False
    _disable_observation_corruption(env_cfg)
    step_dt = _configured_step_dt(env_cfg)
    if (
        step_dt is not None
        and hasattr(env_cfg, "episode_length_s")
        and not args_cli.preserve_episode_length
    ):
        current_episode_length_s = float(getattr(env_cfg, "episode_length_s"))
        required_episode_length_s = float(steps + 2) * step_dt
        if current_episode_length_s < required_episode_length_s:
            env_cfg.episode_length_s = required_episode_length_s
            print(
                "[INFO] Extended env.episode_length_s for rollout: "
                f"{current_episode_length_s:.3f} -> {required_episode_length_s:.3f}"
            )

    agent_cfg.env.num_envs = num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    if hasattr(agent_cfg, "logger"):
        agent_cfg.logger.backend = ""
        agent_cfg.logger.log_dir = str(output_npz.parent / "agent_logs")
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = env_cfg.sim.device

    render_mode = "rgb_array" if args_cli.video else None
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")

    video_folder = None
    if args_cli.video:
        video_folder = (
            args_cli.video_folder.expanduser().resolve()
            if args_cli.video_folder is not None
            else output_npz.parent / "videos"
        )
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": int(args_cli.video_length or args_cli.steps),
            "disable_logger": True,
        }
        print("[INFO] Recording rollout video.")
        print(video_kwargs)
        raw_env = gym.wrappers.RecordVideo(raw_env, **video_kwargs)

    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(observation_spec=env.observation_spec, backend="gymnasium")
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(args_cli.steps + 2), RewardClipping(-10.0, 5.0)),
    )
    base_env = _unwrap_imitation_env(env)

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))
    collector_policy = agent.collector_policy
    collector_policy.eval()

    action_term = base_env.action_manager.get_term("joint_pos")
    if not isinstance(action_term, JointPositionAction):
        raise TypeError("Expected JointPositionAction term named 'joint_pos'.")
    action_dim = int(action_term.action_dim)
    env_ids = torch.arange(num_envs, device=agent.device, dtype=torch.long)

    output_fps = _infer_output_fps(base_env, env_cfg)
    transition_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    transition_targets: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    transition_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    transition_dones: list[list[bool]] = [[] for _ in range(num_envs)]
    transition_terminated: list[list[bool]] = [[] for _ in range(num_envs)]
    transition_truncated: list[list[bool]] = [[] for _ in range(num_envs)]
    transition_local_steps: list[list[int]] = [[] for _ in range(num_envs)]
    frames: list[list[dict[str, np.ndarray]]] = [[] for _ in range(num_envs)]
    active = [True for _ in range(num_envs)]
    stop_reasons = ["max_steps" for _ in range(num_envs)]

    td = env.reset()
    for env_idx in range(num_envs):
        frames[env_idx].append(_snapshot_robot_state(base_env, env_id=env_idx))
    print(f"[INFO] Recording up to {steps} rollout transitions for {num_envs} env(s)...")
    for _step_idx in range(steps):
        local_steps = base_env._current_local_steps(env_ids).detach().cpu().reshape(-1)

        with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = collector_policy(td)

        action = td.get("action")
        if action is None:
            raise RuntimeError("Policy did not write an 'action' tensor.")
        action_2d = action.reshape(num_envs, action_dim).detach()
        processed_action = base_env._raw_to_processed_action(action_2d, env_ids=env_ids)

        with torch.inference_mode():
            td_step = env.step(td)

        rewards = _optional_flat_tensor(
            td_step, ("next", "reward"), num_envs=num_envs, default=0.0
        )
        dones = _optional_flat_tensor(
            td_step, ("next", "done"), num_envs=num_envs, default=False
        )
        terminateds = _optional_flat_tensor(
            td_step, ("next", "terminated"), num_envs=num_envs, default=False
        )
        truncateds = _optional_flat_tensor(
            td_step, ("next", "truncated"), num_envs=num_envs, default=False
        )

        for env_idx in range(num_envs):
            if not active[env_idx]:
                continue

            done_bool = bool(dones[env_idx].item())
            terminated_bool = bool(terminateds[env_idx].item())
            truncated_bool = bool(truncateds[env_idx].item())
            if (done_bool or terminated_bool or truncated_bool) and not args_cli.keep_after_done:
                if truncated_bool:
                    stop_reasons[env_idx] = "truncated"
                elif terminated_bool:
                    stop_reasons[env_idx] = "terminated"
                else:
                    stop_reasons[env_idx] = "done"
                active[env_idx] = False
                print(
                    "[INFO] Stopping env before appending a done/truncated transition: "
                    f"env={env_idx}, step={_step_idx}, done={done_bool}, "
                    f"terminated={terminated_bool}, truncated={truncated_bool}."
                )
                continue

            transition_actions[env_idx].append(_copy_env_vector(action_2d, env_id=env_idx))
            transition_targets[env_idx].append(_copy_env_vector(processed_action, env_id=env_idx))
            transition_rewards[env_idx].append(float(rewards[env_idx].item()))
            transition_dones[env_idx].append(done_bool)
            transition_terminated[env_idx].append(terminated_bool)
            transition_truncated[env_idx].append(truncated_bool)
            transition_local_steps[env_idx].append(int(local_steps[env_idx].item()))
            frames[env_idx].append(_snapshot_robot_state(base_env, env_id=env_idx))

        if not any(active):
            print("[INFO] All envs reached done/truncated before requested steps.")
            break

        td = step_mdp(td_step, exclude_reward=True, exclude_done=False, exclude_action=True)

    if not any(len(actions) > 0 for actions in transition_actions):
        raise RuntimeError("No valid rollout transitions were recorded.")

    base_metadata = {
        "labeler": "rlopt_policy_rollout",
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "checkpoint": str(checkpoint_path),
        "motion_manifest": str(motion_manifest) if motion_manifest is not None else None,
        "schema_reference_npz": (
            str(args_cli.schema_reference_npz.expanduser().resolve())
            if args_cli.schema_reference_npz is not None
            else None
        ),
        "requested_steps": steps,
        "num_envs": int(num_envs),
        "fps": float(output_fps),
        "video_folder": str(video_folder) if video_folder is not None else None,
        "state_semantics": "simulated robot states from closed-loop policy rollout",
        "action_semantics": "raw JointPositionAction policy action; valid for transition frame t -> t+1",
        "action_joint_pos_target_semantics": "processed joint position target after env action scaling/offset",
        "final_frame_action": "repeats previous action; action_valid marks it invalid",
        "robot_body_names": list(base_env.robot.body_names),
    }
    action_joint_names = list(action_term._joint_names)
    output_paths = []
    for rollout_index in range(num_envs):
        output_path = _write_rollout_npz(
            output_npz=output_npz,
            rollout_index=rollout_index,
            total_rollouts=num_envs,
            env_id=rollout_index,
            frames=frames[rollout_index],
            transition_actions=transition_actions[rollout_index],
            transition_targets=transition_targets[rollout_index],
            transition_rewards=transition_rewards[rollout_index],
            transition_dones=transition_dones[rollout_index],
            transition_terminated=transition_terminated[rollout_index],
            transition_truncated=transition_truncated[rollout_index],
            transition_local_steps=transition_local_steps[rollout_index],
            output_fps=output_fps,
            action_dim=action_dim,
            action_joint_names=action_joint_names,
            base_metadata=base_metadata,
            stop_reason=stop_reasons[rollout_index],
        )
        if output_path is not None:
            output_paths.append(output_path)

    if args_cli.output_manifest is not None:
        _write_manifest(
            manifest_path=args_cli.output_manifest,
            output_paths=output_paths,
            output_fps=output_fps,
            metadata=base_metadata,
        )
    if video_folder is not None:
        print(f"[INFO] Video folder: {video_folder}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
