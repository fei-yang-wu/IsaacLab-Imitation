# ruff: noqa: E402
"""Export action labels from a trained RLOpt policy on a manifest-backed NPZ.

The rollout is teacher-forced through ``ImitationRLEnv.replay_only``: the robot
state is reset to the reference trajectory at each control step, the policy sees
the same observation surface as during play, and the emitted action is stored as
the label for the transition from frame t to t+1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Imitation-G1-Latent-v0",
    help="Registered Isaac Lab task.",
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=["PPO", "SAC", "FASTSAC", "IPMD", "IPMD_SR", "IPMD_BILINEAR", "GAIL", "AMP", "ASE"],
    help="RLOpt algorithm used by the checkpoint.",
)
parser.add_argument("--checkpoint", type=Path, required=True, help="RLOpt checkpoint to load.")
parser.add_argument("--source_npz", type=Path, required=True, help="Source trajectory NPZ to copy and label.")
parser.add_argument("--output_npz", type=Path, required=True, help="Output NPZ with action labels.")
parser.add_argument(
    "--motion_manifest",
    type=Path,
    default=Path("data/unitree/manifests/g1_unitree_dance102_manifest.json"),
    help="Manifest used to build the policy observation/reference stream.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. Currently must be 1.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for env and policy construction.")
parser.add_argument(
    "--max_transitions",
    type=int,
    default=0,
    help="Optional transition limit. Defaults to all source frames minus one.",
)
parser.add_argument(
    "--refresh_zarr_dataset",
    action="store_true",
    help="Rebuild the manifest-derived Zarr cache before labeling.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Action labeling is headless and does not need cameras.
args_cli.headless = True
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
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import AMP, ASE, GAIL, IPMD, IPMDBilinear, IPMDSR, PPO, SAC, FastSAC
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
    try:
        spec = gym.spec(task_id)
    except Exception as exc:
        msg = f"Could not resolve task '{task_id}' from registry."
        raise ValueError(msg) from exc

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


def _disable_observation_corruption(env_cfg: object) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for group_name in ("policy", "critic", "expert_state", "expert_window", "reward_input"):
        group = getattr(observations, group_name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _first_dim(npz_arrays: dict[str, np.ndarray]) -> int:
    for key in ("joint_pos", "qpos", "body_pos_w"):
        value = npz_arrays.get(key)
        if value is not None and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer frame count from source NPZ.")


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs != 1:
        raise ValueError("label_npz_with_policy.py currently supports --num_envs 1 only.")

    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    source_npz = args_cli.source_npz.expanduser().resolve()
    output_npz = args_cli.output_npz.expanduser().resolve()
    motion_manifest = args_cli.motion_manifest.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not source_npz.is_file():
        raise FileNotFoundError(f"Source NPZ not found: {source_npz}")
    if not motion_manifest.is_file():
        raise FileNotFoundError(f"Motion manifest not found: {motion_manifest}")

    with np.load(source_npz, allow_pickle=False) as source_data:
        source_arrays = {key: np.asarray(source_data[key]) for key in source_data.files}
    frame_count = _first_dim(source_arrays)
    transition_count = max(frame_count - 1, 0)
    if args_cli.max_transitions > 0:
        transition_count = min(transition_count, int(args_cli.max_transitions))
    if transition_count <= 0:
        raise ValueError(f"Need at least two frames to export actions, got {frame_count}.")

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(agent_cfg, "seed", None)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.log_dir = str(checkpoint_path.parent)
    env_cfg.lafan1_manifest_path = str(motion_manifest)
    if hasattr(env_cfg, "_resolve_manifest_config"):
        env_cfg._resolve_manifest_config()
    if hasattr(env_cfg, "refresh_zarr_dataset"):
        env_cfg.refresh_zarr_dataset = bool(args_cli.refresh_zarr_dataset)
    if hasattr(env_cfg, "replay_only"):
        env_cfg.replay_only = True
    if hasattr(env_cfg, "replay_reference"):
        env_cfg.replay_reference = True
    if hasattr(env_cfg, "reference_start_frame"):
        env_cfg.reference_start_frame = 0
    if hasattr(env_cfg, "random_reset_full_trajectory"):
        env_cfg.random_reset_full_trajectory = False
    if hasattr(env_cfg, "random_reset_step_min"):
        env_cfg.random_reset_step_min = 0
    if hasattr(env_cfg, "random_reset_step_max"):
        env_cfg.random_reset_step_max = 0
    if hasattr(env_cfg, "wrap_steps"):
        env_cfg.wrap_steps = False
    if hasattr(env_cfg, "reset_schedule"):
        env_cfg.reset_schedule = "sequential"
    _disable_observation_corruption(env_cfg)

    agent_cfg.env.num_envs = 1
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= 1
    agent_cfg.logger.backend = ""
    agent_cfg.logger.log_dir = str(checkpoint_path.parent / "label_export_logs")
    if hasattr(agent_cfg, "device"):
        agent_cfg.device = env_cfg.sim.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported for RLOpt labeling.")
    isaac_env = raw_env.unwrapped

    env = IsaacLabWrapper(raw_env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(observation_spec=env.observation_spec, backend="gymnasium")
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(transition_count + 2), RewardClipping(-10.0, 5.0)),
    )

    agent_class = ALGORITHM_CLASS_MAP[args_cli.algorithm]
    agent = agent_class(env=env, config=agent_cfg)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    agent.load_model(str(checkpoint_path))

    collector_policy = agent.collector_policy
    collector_policy.eval()

    action_term = isaac_env.action_manager.get_term("joint_pos")
    if not isinstance(action_term, JointPositionAction):
        raise TypeError("Expected a JointPositionAction term named 'joint_pos'.")
    action_joint_names = np.asarray(list(action_term._joint_names))
    action_dim = int(action_term.action_dim)

    transition_actions = np.zeros((transition_count, action_dim), dtype=np.float32)
    transition_targets = np.zeros((transition_count, action_dim), dtype=np.float32)
    action_valid = np.zeros((frame_count,), dtype=np.bool_)

    td = env.reset()
    env_ids = torch.zeros((1,), device=agent.device, dtype=torch.long)
    print(f"[INFO] Exporting {transition_count} transition actions...")
    for step_idx in range(transition_count):
        with torch.inference_mode(), set_exploration_type(InteractionType.DETERMINISTIC):
            td = collector_policy(td)
        action = td.get("action")
        if action is None:
            raise RuntimeError("Policy did not write an 'action' tensor.")
        action_2d = action.reshape(1, action_dim).detach()
        processed = isaac_env._raw_to_processed_action(action_2d, env_ids=env_ids)
        transition_actions[step_idx] = action_2d[0].detach().cpu().numpy()
        transition_targets[step_idx] = processed[0].detach().cpu().numpy()
        action_valid[step_idx] = True

        with torch.inference_mode():
            td = env.step(td)
            td = step_mdp(td, exclude_reward=True, exclude_done=False, exclude_action=True)

    full_actions = np.zeros((frame_count, action_dim), dtype=np.float32)
    full_targets = np.zeros((frame_count, action_dim), dtype=np.float32)
    full_actions[:transition_count] = transition_actions
    full_targets[:transition_count] = transition_targets
    if transition_count < frame_count:
        full_actions[transition_count:] = transition_actions[transition_count - 1]
        full_targets[transition_count:] = transition_targets[transition_count - 1]

    metadata = {
        "labeler": "rlopt_policy",
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "checkpoint": str(checkpoint_path),
        "motion_manifest": str(motion_manifest),
        "source_npz": str(source_npz),
        "transition_count": int(transition_count),
        "frame_count": int(frame_count),
        "action_semantics": "raw JointPositionAction policy action; valid for transition frame t -> t+1",
        "final_frame_action": "repeats previous action; action_valid marks it invalid",
    }

    output_arrays = dict(source_arrays)
    output_arrays["action"] = full_actions
    output_arrays["action_valid"] = action_valid
    output_arrays["transition_action"] = transition_actions
    output_arrays["action_joint_pos_target"] = full_targets
    output_arrays["transition_action_joint_pos_target"] = transition_targets
    output_arrays["action_joint_names"] = action_joint_names
    output_arrays["action_label_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **output_arrays)
    print(f"[INFO] Wrote labeled NPZ: {output_npz}")
    print(
        "[INFO] action stats: "
        f"shape={full_actions.shape}, "
        f"min={float(full_actions.min()):.4f}, "
        f"max={float(full_actions.max()):.4f}, "
        f"mean_abs={float(np.mean(np.abs(full_actions[action_valid]))):.4f}"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
