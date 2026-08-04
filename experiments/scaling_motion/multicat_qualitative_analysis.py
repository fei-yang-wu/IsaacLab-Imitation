#!/usr/bin/env python3
# ruff: noqa: E402
"""Qualitative analysis for a grouped Gumbel skill encoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from isaaclab_tasks.utils import add_launcher_args


DEFAULT_MOTION = "jog_ff_loop_180_R_002_A091_M"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("single", "multi", "intervene"))
parser.add_argument("--skill-checkpoint", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--motion", default=DEFAULT_MOTION)
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--motion-count", type=int, default=50)
parser.add_argument("--max-chunks-per-motion", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--policy-checkpoint", type=Path)
parser.add_argument("--group", default="auto")
parser.add_argument("--rollout-steps", type=int, default=10)
parser.add_argument("--random-base-code", action="store_true")
parser.add_argument("--independent-random-codes", action="store_true")
parser.add_argument("--category-count", type=int, default=128)
parser.add_argument("--video", action="store_true")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--smoke", action="store_true")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

import gymnasium as gym
import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import matplotlib
import numpy as np
import torch
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_imitation.envs.imitation_rl_env import ImitationRLEnv
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD
from rlopt.agent.hl_skill_diffsr import HighLevelSkillDiffSRConfig
from rlopt.agent.hl_skill_encoder import build_skill_encoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent / "interface_baselines"))
from paper_protocol_metadata import disable_domain_randomization


TASK = "Isaac-Imitation-G1-Latent-v0"
AGENT_ENTRY_POINT = "rlopt_ipmd_cfg_entry_point"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _mode_dir(name: str) -> Path:
    path = args_cli.output_root.resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not args_cli.overwrite:
        raise FileExistsError(
            f"{path} is non-empty; pass --overwrite to replace outputs."
        )
    return path


def _provenance(output_dir: Path, extra: dict[str, Any]) -> None:
    skill = args_cli.skill_checkpoint.resolve()
    manifest = args_cli.manifest.resolve()
    record = {
        "mode": args_cli.mode,
        "skill_checkpoint": str(skill),
        "skill_checkpoint_sha256": _sha256(skill),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "dataset": str(args_cli.dataset.resolve()),
        "seed": args_cli.seed,
        **extra,
    }
    if args_cli.policy_checkpoint is not None:
        policy = args_cli.policy_checkpoint.resolve()
        record["policy_checkpoint"] = str(policy)
        record["policy_checkpoint_sha256"] = _sha256(policy)
    _json(output_dir / "provenance.json", record)


def _manifest_entries() -> list[dict[str, Any]]:
    data = json.loads(args_cli.manifest.resolve().read_text())
    return list(data["dataset"]["trajectories"]["lafan1_csv"])


def _entry_path(entry: dict[str, Any]) -> Path:
    return (args_cli.manifest.resolve().parent / str(entry["path"])).resolve()


def _trajectory_length(entry: dict[str, Any]) -> int:
    with np.load(_entry_path(entry), mmap_mode="r") as data:
        return int(data["joint_pos"].shape[0])


def _select_motion_names() -> list[str]:
    entries = _manifest_entries()
    by_name = {str(entry["name"]): entry for entry in entries}
    if args_cli.motion not in by_name:
        raise ValueError(f"Motion {args_cli.motion!r} is absent from the manifest.")
    count = 5 if args_cli.smoke else int(args_cli.motion_count)
    ranked = sorted(
        (entry for entry in entries if str(entry["name"]) != args_cli.motion),
        key=lambda entry: hashlib.sha256(
            f"{args_cli.seed}\0{entry['name']}".encode()
        ).digest(),
    )
    selected = [args_cli.motion]
    for entry in ranked:
        if _trajectory_length(entry) >= 4 * args_cli.stride:
            selected.append(str(entry["name"]))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Only found {len(selected)} eligible motions; requested {count}."
        )
    return selected


def _disable_corruption(env_cfg: Any) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for name in ("policy", "critic", "expert_state", "expert_window", "reward_input"):
        group = getattr(observations, name, None)
        if group is not None and hasattr(group, "enable_corruption"):
            group.enable_corruption = False


def _disable_terminations(env_cfg: Any) -> list[str]:
    disabled: list[str] = []
    terms = getattr(env_cfg, "terminations", None)
    if terms is None:
        return disabled
    for name in dir(terms):
        if name.startswith("_") or getattr(terms, name, None) is None:
            continue
        value = getattr(terms, name)
        if hasattr(value, "func"):
            setattr(terms, name, None)
            disabled.append(name)
    return sorted(disabled)


def _configure_env(env_cfg: Any, motions: list[str], num_envs: int) -> None:
    env_cfg.lafan1_manifest_path = str(args_cli.manifest.resolve())
    env_cfg.dataset_path = str(args_cli.dataset.resolve())
    env_cfg.motions = motions
    env_cfg._resolve_manifest_config(
        dataset_path_explicit=True,
        motions_explicit=True,
        timing_explicit=False,
    )
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = int(args_cli.seed)
    env_cfg.refresh_zarr_dataset = False
    env_cfg.reference_start_frame = 0
    env_cfg.random_reset_full_trajectory = False
    env_cfg.random_reset_step_min = 0
    env_cfg.random_reset_step_max = 0
    env_cfg.reset_schedule = "sequential"
    env_cfg.wrap_steps = False
    _disable_corruption(env_cfg)


def _unwrap(env: Any) -> ImitationRLEnv:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
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


def _build_raw_env(env_cfg: Any) -> tuple[Any, ImitationRLEnv]:
    raw = gym.make(TASK, cfg=env_cfg)
    return raw, _unwrap(raw)


def _load_encoder(state_dim: int, device: torch.device) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(
        args_cli.skill_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
    if config.latent_mode != "gumbel_multicat":
        raise ValueError(f"Expected gumbel_multicat, got {config.latent_mode!r}.")
    window_steps = config.horizon_steps - 1
    encoder = build_skill_encoder(
        state_dim=state_dim,
        window_steps=window_steps,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
    ).to(device)
    encoder.load_state_dict(checkpoint["skill_encoder_state_dict"])
    encoder.eval().requires_grad_(False)
    return encoder, config.to_dict()


def _macro_sequence(
    env: ImitationRLEnv,
    ranks: torch.Tensor,
    starts: torch.Tensor,
    window_steps: int = 10,
) -> torch.Tensor:
    if ranks.shape != starts.shape or ranks.ndim != 1:
        raise ValueError("ranks and starts must be matching vectors.")
    tm = env.trajectory_manager
    ranks_tm = ranks.to(device=tm._state_device, dtype=torch.long)
    starts_tm = starts.to(device=tm._state_device, dtype=torch.long)
    expert_window = env._sample_expert_window_slice_for_trajectory_ranks(
        ranks_tm,
        starts_tm,
        past_steps=0,
        future_steps=window_steps - 1,
    )
    env_ids = torch.arange(ranks.numel(), device=env.device, dtype=torch.long)
    terms = env._build_expert_window_terms(
        expert_window,
        env_ids,
        context="expert",
        past_steps=0,
        joint_ids=slice(None),
        anchor_body_name=env._expert_anchor_body_name,
    )
    sequence = env._expert_macro_state_sequence_from_terms(
        terms, batch_size=ranks.numel(), window_steps=window_steps
    )
    if sequence.shape[:2] != (ranks.numel(), window_steps):
        raise RuntimeError(f"Unexpected macro sequence shape: {tuple(sequence.shape)}.")
    return sequence


@torch.inference_mode()
def _encode_rows(
    env: ImitationRLEnv,
    ranks: np.ndarray,
    starts: np.ndarray,
) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {
        "latent": [],
        "category": [],
        "probability": [],
        "margin": [],
        "entropy": [],
        "state": [],
    }
    encoder = None
    for begin in range(0, len(ranks), args_cli.batch_size):
        end = min(begin + args_cli.batch_size, len(ranks))
        sequence = _macro_sequence(
            env,
            torch.as_tensor(ranks[begin:end]),
            torch.as_tensor(starts[begin:end]),
        )
        if encoder is None:
            encoder, _ = _load_encoder(int(sequence.shape[-1]), sequence.device)
        raw = encoder._raw(sequence[:, 0], sequence[:, 1:])
        logits = raw.reshape(raw.shape[0], encoder.groups, encoder.categories)
        probabilities = logits.softmax(dim=-1)
        top2 = probabilities.topk(2, dim=-1).values
        categories = logits.argmax(dim=-1)
        z = encoder(sequence[:, 0], sequence[:, 1:])
        reconstructed = encoder.codebook[
            torch.arange(encoder.groups, device=z.device)[None, :], categories
        ].reshape_as(z)
        torch.testing.assert_close(z, reconstructed)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        for key, value in (
            ("latent", z),
            ("category", categories),
            ("probability", top2[..., 0]),
            ("margin", top2[..., 0] - top2[..., 1]),
            ("entropy", entropy),
            ("state", sequence[:, 0]),
        ):
            outputs[key].append(value.detach().cpu().numpy())
    return {key: np.concatenate(values) for key, values in outputs.items()}


def _starts(length: int) -> np.ndarray:
    count = max(0, (length - 10) // args_cli.stride + 1)
    starts = np.arange(count, dtype=np.int64) * args_cli.stride
    return starts[:2] if args_cli.smoke else starts


def _group_statistics(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    categories = data["category"]
    rows: list[dict[str, Any]] = []
    for group in range(categories.shape[1]):
        counts = np.bincount(categories[:, group], minlength=128)
        probs = counts[counts > 0] / max(counts.sum(), 1)
        entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
        max_entropy = math.log(max(2, min(128, len(categories))))
        rows.append(
            {
                "group": group,
                "unique_categories": int((counts > 0).sum()),
                "normalized_entropy": entropy / max_entropy,
                "switch_rate": float(np.mean(np.diff(categories[:, group]) != 0))
                if len(categories) > 1
                else 0.0,
                "mean_probability": float(data["probability"][:, group].mean()),
                "median_probability": float(np.median(data["probability"][:, group])),
                "mean_margin": float(data["margin"][:, group].mean()),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_single(output: Path, starts: np.ndarray, data: dict[str, np.ndarray]) -> None:
    categories = data["category"]
    distances = (categories[1:] != categories[:-1]).mean(axis=1)
    coords = PCA(n_components=min(2, len(data["latent"]))).fit_transform(data["latent"])
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    image = axes[0].imshow(categories.T, aspect="auto", interpolation="nearest")
    axes[0].set(title="Group-local category IDs", ylabel="group")
    axes[0].set_xticks(range(len(starts)), starts)
    fig.colorbar(image, ax=axes[0], label="nominal category ID")
    axes[1].plot(starts[1:], distances, marker="o")
    axes[1].set(title="Adjacent code Hamming distance", ylabel="fraction changed")
    axes[2].plot(coords[:, 0], coords[:, 1], marker="o")
    for index, step in enumerate(starts):
        axes[2].annotate(str(step), coords[index])
    axes[2].set(title="PCA latent trajectory", xlabel="PC1", ylabel="PC2")
    fig.savefig(output / "latent_trajectory.png", dpi=180)
    plt.close(fig)


def _run_single(env_cfg: Any) -> None:
    output = _mode_dir("single_motion")
    entry = next(
        (entry for entry in _manifest_entries() if entry["name"] == args_cli.motion),
        None,
    )
    if entry is None:
        raise ValueError(f"Unknown motion: {args_cli.motion}")
    starts = _starts(_trajectory_length(entry))
    if len(starts) < 2:
        raise ValueError("The focus motion has fewer than two complete windows.")
    _configure_env(env_cfg, [args_cli.motion], min(args_cli.batch_size, len(starts)))
    raw, env = _build_raw_env(env_cfg)
    try:
        names = env.expert_trajectory_motion_names()
        rank = names.index(args_cli.motion)
        ranks = np.full(len(starts), rank, dtype=np.int64)
        data = _encode_rows(env, ranks, starts)
    finally:
        raw.close()
    np.savez_compressed(
        output / "chunks.npz",
        motion=np.asarray([args_cli.motion]),
        local_step=starts,
        **data,
    )
    stats = _group_statistics(data)
    _write_csv(output / "group_statistics.csv", stats)
    _plot_single(output, starts, data)
    _provenance(
        output,
        {
            "motion": args_cli.motion,
            "stride": args_cli.stride,
            "window_steps": 10,
            "chunk_count": len(starts),
        },
    )
    _json(
        output / "summary.json",
        {
            "motion": args_cli.motion,
            "chunks": len(starts),
            "artifacts": sorted(str(path.resolve()) for path in output.iterdir()),
        },
    )
    print(f"[INFO] Single-motion outputs: {output.resolve()}")


def _topk_accuracy(
    distances: np.ndarray, train_labels: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    order = distances.argsort(axis=1)
    ranked_labels = train_labels[order]
    top1 = ranked_labels[:, 0]
    top1_acc = float(np.mean(top1 == labels))
    top5_acc = float(np.mean(np.any(ranked_labels[:, :5] == labels[:, None], axis=1)))
    reciprocal = []
    for row, label in zip(ranked_labels, labels, strict=True):
        hits = np.flatnonzero(row == label)
        reciprocal.append(1.0 / (int(hits[0]) + 1) if len(hits) else 0.0)
    return top1_acc, top5_acc, float(np.mean(reciprocal)), top1


def _plot_multi(
    output: Path,
    latent: np.ndarray,
    labels: np.ndarray,
    test_labels: np.ndarray,
    predicted: np.ndarray,
    nmi: np.ndarray,
) -> None:
    pca = PCA(n_components=2, random_state=args_cli.seed).fit_transform(latent)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=args_cli.seed,
        perplexity=min(30, max(5, len(latent) // 20)),
    ).fit_transform(latent)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(pca[:, 0], pca[:, 1], c=labels, s=8, cmap="nipy_spectral")
    axes[0].set(title="PCA by motion", xlabel="PC1", ylabel="PC2")
    axes[1].scatter(tsne[:, 0], tsne[:, 1], c=labels, s=8, cmap="nipy_spectral")
    axes[1].set(title="t-SNE by motion", xlabel="dim 1", ylabel="dim 2")
    fig.savefig(output / "motion_embeddings.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(confusion_matrix(test_labels, predicted), interpolation="nearest")
    axes[0].set(
        title="Continuous latent 1-NN confusion", xlabel="predicted", ylabel="true"
    )
    axes[1].bar(np.arange(len(nmi)), nmi)
    axes[1].set(title="Category/motion normalized mutual information", xlabel="group")
    fig.savefig(output / "motion_identification.png", dpi=180)
    plt.close(fig)


def _run_multi(env_cfg: Any) -> None:
    output = _mode_dir("multi_motion")
    names = _select_motion_names()
    rows_per_motion = 2 if args_cli.smoke else args_cli.max_chunks_per_motion
    _configure_env(
        env_cfg, names, min(args_cli.batch_size, len(names) * rows_per_motion)
    )
    raw, env = _build_raw_env(env_cfg)
    try:
        loaded = env.expert_trajectory_motion_names()
        name_to_rank = {name: rank for rank, name in enumerate(loaded)}
        ranks: list[int] = []
        starts: list[int] = []
        labels: list[int] = []
        cap = 4 if args_cli.smoke else args_cli.max_chunks_per_motion
        for label, name in enumerate(names):
            rank = name_to_rank[name]
            length = int(env.trajectory_manager._length[rank].item())
            motion_starts = _starts(length)[:cap]
            if len(motion_starts) < 2:
                raise RuntimeError(f"Motion {name!r} has fewer than two chunks.")
            ranks.extend([rank] * len(motion_starts))
            starts.extend(motion_starts.tolist())
            labels.extend([label] * len(motion_starts))
        rank_array = np.asarray(ranks, dtype=np.int64)
        start_array = np.asarray(starts, dtype=np.int64)
        label_array = np.asarray(labels, dtype=np.int64)
        data = _encode_rows(env, rank_array, start_array)
    finally:
        raw.close()

    train = np.zeros(len(labels), dtype=bool)
    for label in np.unique(label_array):
        indices = np.flatnonzero(label_array == label)
        train[indices[::2]] = True
    test = ~train
    z_train, z_test = data["latent"][train], data["latent"][test]
    code_train, code_test = data["category"][train], data["category"][test]
    train_labels, test_labels = label_array[train], label_array[test]
    z_dist = np.linalg.norm(z_test[:, None] - z_train[None, :], axis=-1)
    code_dist = np.mean(code_test[:, None] != code_train[None, :], axis=-1)
    z_top1, z_top5, z_mrr, predicted = _topk_accuracy(z_dist, train_labels, test_labels)
    code_top1, code_top5, code_mrr, _ = _topk_accuracy(
        code_dist, train_labels, test_labels
    )
    centroids = np.stack(
        [z_train[train_labels == label].mean(0) for label in sorted(set(train_labels))]
    )
    centroid_pred = np.linalg.norm(
        z_test[:, None] - centroids[None, :], axis=-1
    ).argmin(1)
    nmi = np.asarray(
        [
            normalized_mutual_info_score(label_array, data["category"][:, group])
            for group in range(data["category"].shape[1])
        ]
    )
    metrics = {
        "motion_count": len(names),
        "chunk_count": len(labels),
        "chance_top1": 1.0 / len(names),
        "continuous_1nn": {"top1": z_top1, "top5": z_top5, "mrr": z_mrr},
        "categorical_1nn": {
            "top1": code_top1,
            "top5": code_top5,
            "mrr": code_mrr,
        },
        "continuous_centroid_top1": float(np.mean(centroid_pred == test_labels)),
        "own_centroid_distance_mean": float(
            np.linalg.norm(z_test - centroids[test_labels], axis=-1).mean()
        ),
    }
    np.savez_compressed(
        output / "chunks.npz",
        motion_names=np.asarray(names),
        trajectory_rank=rank_array,
        motion_index=label_array,
        local_step=start_array,
        **data,
    )
    _json(output / "metrics.json", metrics)
    _write_csv(
        output / "group_motion_nmi.csv",
        [
            {"group": index, "motion_nmi": float(value)}
            for index, value in enumerate(nmi)
        ],
    )
    _plot_multi(
        output,
        data["latent"],
        label_array,
        test_labels,
        predicted,
        nmi,
    )
    _provenance(
        output,
        {
            "motion_names": names,
            "stride": args_cli.stride,
            "max_chunks_per_motion": args_cli.max_chunks_per_motion,
        },
    )
    print(json.dumps(metrics, indent=2))
    print(f"[INFO] Multi-motion outputs: {output.resolve()}")


def _as_tensor(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _snapshot(env: ImitationRLEnv) -> dict[str, np.ndarray]:
    robot = env.scene["robot"]
    origins = env.scene.env_origins[:, None, :]
    return {
        "root_pos": (_as_tensor(robot.data.root_pos_w) - origins[:, 0])
        .detach()
        .cpu()
        .numpy(),
        "root_quat": _as_tensor(robot.data.root_quat_w).detach().cpu().numpy(),
        "joint_pos": _as_tensor(robot.data.joint_pos).detach().cpu().numpy(),
        "joint_vel": _as_tensor(robot.data.joint_vel).detach().cpu().numpy(),
        "body_pos": (_as_tensor(robot.data.body_pos_w) - origins)
        .detach()
        .cpu()
        .numpy(),
    }


def _refresh(td: TensorDictBase, env: ImitationRLEnv) -> TensorDictBase:
    observations = env.observation_manager.compute(update_history=False)
    for group_name, group_obs in observations.items():
        if isinstance(group_obs, dict):
            for term_name, value in group_obs.items():
                td.set((group_name, term_name), value)
        else:
            td.set(group_name, group_obs)
    return td


def _focus_group(single: dict[str, np.ndarray]) -> int:
    if args_cli.group != "auto":
        group = int(args_cli.group)
        if not 0 <= group < 64:
            raise ValueError("--group must be auto or an integer in [0, 63].")
        return group
    stats = _group_statistics(single)
    eligible = [
        row
        for row in stats
        if row["unique_categories"] >= 2 and row["median_probability"] >= 0.5
    ]
    candidates = eligible or [row for row in stats if row["unique_categories"] >= 2]
    if not candidates:
        raise RuntimeError("No group uses at least two categories on the focus motion.")
    best = max(
        candidates,
        key=lambda row: (
            row["normalized_entropy"],
            row["switch_rate"],
            row["mean_margin"],
            -row["group"],
        ),
    )
    return int(best["group"])


def _configure_intervention_env(env_cfg: Any, num_envs: int) -> dict[str, Any]:
    _configure_env(env_cfg, [args_cli.motion], num_envs)
    randomization = disable_domain_randomization(env_cfg)
    disabled_terminations = _disable_terminations(env_cfg)
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), 1.0)
    solver = getattr(getattr(env_cfg.sim, "physics", None), "solver_cfg", None)
    return {
        "domain_randomization": randomization,
        "disabled_terminations": disabled_terminations,
        "physics_solver": {
            "njmax": int(getattr(solver, "njmax", 0)),
            "nconmax": int(getattr(solver, "nconmax", 0)),
        },
    }


def _wrap_for_policy(raw: Any, steps: int) -> tuple[Any, ImitationRLEnv]:
    wrapped = IsaacLabWrapper(raw)
    wrapped = wrapped.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=wrapped.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=wrapped,
        transform=Compose(
            RewardSum(), StepCounter(steps + 2), RewardClipping(-10.0, 5.0)
        ),
    )
    return env, _unwrap(env)


def _representatives(effects: np.ndarray, baseline: int, count: int = 8) -> list[int]:
    chosen = [baseline]
    candidates = set(range(len(effects))) - {baseline}
    while candidates and len(chosen) < count:
        candidate = max(
            candidates,
            key=lambda index: min(
                np.linalg.norm(effects[index] - effects[item]) for item in chosen
            ),
        )
        chosen.append(candidate)
        candidates.remove(candidate)
    return chosen


def _plot_intervention(
    output: Path,
    metrics: np.ndarray,
    metric_names: list[str],
    body_pos: np.ndarray,
    categories: np.ndarray,
    baseline_index: int,
    label_name: str = "category",
) -> None:
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    normalized = (metrics - metrics.mean(0)) / metrics.std(0).clip(min=1e-8)
    image = ax.imshow(
        normalized.T, aspect="auto", interpolation="nearest", cmap="coolwarm"
    )
    ax.set(
        title=f"Standardized {label_name} effects",
        xlabel=label_name,
        ylabel="effect metric",
    )
    ax.set_yticks(range(len(metric_names)), metric_names)
    fig.colorbar(image, ax=ax)
    fig.savefig(output / "category_effects.png", dpi=180)
    plt.close(fig)

    coords = PCA(n_components=2).fit_transform(normalized)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.scatter(coords[:, 0], coords[:, 1], c=categories, cmap="viridis", s=25)
    for index in _representatives(normalized, baseline_index):
        ax.annotate(str(categories[index]), coords[index])
    ax.set(
        title=f"{label_name.title()} effects in PCA space", xlabel="PC1", ylabel="PC2"
    )
    fig.savefig(output / "category_effect_pca.png", dpi=180)
    plt.close(fig)

    representatives = _representatives(normalized, baseline_index)
    selected = body_pos[representatives]
    center = selected[:, 0].mean(axis=(0, 1))
    span = max(0.8, float(np.ptp(selected[..., [0, 2]], axis=(0, 1, 2)).max()))
    fig, axes = plt.subplots(2, 4, figsize=(12, 7), constrained_layout=True)
    points = []
    for ax, index, trajectory in zip(axes.flat, representatives, selected):
        point = ax.scatter(trajectory[0, :, 0], trajectory[0, :, 2], s=12)
        points.append((point, trajectory))
        ax.set(
            title=f"{label_name} {categories[index]}",
            xlim=(center[0] - span, center[0] + span),
            ylim=(center[2] - span, center[2] + span),
            aspect="equal",
        )
    for ax in axes.flat[len(representatives) :]:
        ax.axis("off")

    def update(frame: int):
        for point, trajectory in points:
            point.set_offsets(trajectory[frame, :, [0, 2]])
        return [point for point, _ in points]

    movie = animation.FuncAnimation(
        fig, update, frames=selected.shape[1], interval=180, blit=False
    )
    movie.save(output / "representative_categories.gif", writer="pillow", fps=5)
    plt.close(fig)


def _run_intervene(env_cfg: Any, agent_cfg: Any) -> None:
    if args_cli.policy_checkpoint is None:
        raise ValueError("--policy-checkpoint is required for intervention mode.")
    if args_cli.independent_random_codes:
        count = 3 if args_cli.smoke else int(args_cli.category_count)
        if count < 2:
            raise ValueError("--category-count must be at least two.")
        rng = np.random.default_rng(args_cli.seed)
        code_categories = rng.integers(0, 128, size=(count, 64), dtype=np.int64)
        if np.unique(code_categories, axis=0).shape[0] != count:
            raise RuntimeError("Random full-code sampling produced a duplicate.")
        base_code_categories = code_categories[0]
        group = -1
        baseline_category = -1
        category_ids = np.arange(count, dtype=np.int64)
    elif args_cli.random_base_code:
        count = 3 if args_cli.smoke else int(args_cli.category_count)
        if not 2 <= count <= 128:
            raise ValueError("--category-count must be in [2, 128].")
        rng = np.random.default_rng(args_cli.seed)
        base_code_categories = rng.integers(0, 128, size=64, dtype=np.int64)
        group = (
            int(rng.integers(0, 64))
            if args_cli.group == "auto"
            else int(args_cli.group)
        )
        if not 0 <= group < 64:
            raise ValueError("--group must be auto or an integer in [0, 63].")
        baseline_category = int(base_code_categories[group])
        alternatives = np.delete(np.arange(128, dtype=np.int64), baseline_category)
        category_ids = np.concatenate(
            [
                np.asarray([baseline_category], dtype=np.int64),
                rng.choice(alternatives, size=count - 1, replace=False),
            ]
        )
    else:
        single_path = args_cli.output_root.resolve() / "single_motion" / "chunks.npz"
        if not single_path.is_file():
            raise FileNotFoundError(
                f"Missing {single_path}; run the single-motion mode first."
            )
        single = dict(np.load(single_path))
        group = _focus_group(single)
        base_code_categories = np.asarray(single["category"][0], dtype=np.int64)
        baseline_category = int(base_code_categories[group])
        if args_cli.smoke:
            alternatives = [
                item for item in (0, 127, 1, 126) if item != baseline_category
            ]
            category_ids = np.asarray(
                [baseline_category, *alternatives[:2]], dtype=np.int64
            )
        else:
            category_ids = np.arange(128, dtype=np.int64)
    if not args_cli.independent_random_codes:
        code_categories = np.repeat(
            base_code_categories[None, :], len(category_ids), axis=0
        )
        code_categories[:, group] = category_ids
    output = _mode_dir(
        "random_code_rollout"
        if args_cli.independent_random_codes
        else "category_intervention"
    )
    protocol = _configure_intervention_env(env_cfg, len(category_ids))
    num_envs = env_cfg.scene.num_envs
    if args_cli.video:
        camera_distance = 2.5 * math.ceil(math.sqrt(num_envs))
        env_cfg.viewer.eye = (
            camera_distance,
            0.9 * camera_distance,
            0.9 * camera_distance,
        )
        env_cfg.viewer.lookat = (0.0, 0.0, 0.8)
        env_cfg.viewer.resolution = (1920, 1080)
        env_cfg.video_follow_robot = False
    agent_cfg.env.num_envs = num_envs
    agent_cfg.env.env_name = TASK
    agent_cfg.seed = args_cli.seed
    agent_cfg.logger.backend = ""
    agent_cfg.logger.log_dir = str(output / "agent_logs")
    agent_cfg.ipmd.hl_skill_checkpoint_path = str(args_cli.skill_checkpoint.resolve())
    agent_cfg.ipmd.hl_skill_horizon_steps = 10
    agent_cfg.ipmd.latent_steps_min = 10
    agent_cfg.ipmd.latent_steps_max = 10
    agent_cfg.ipmd.latent_learning.code_period = 10
    agent_cfg.ipmd.latent_learning.command_phase_mode = "sin_cos"
    agent_cfg.ipmd.latent_learning.code_latent_dim = 256
    agent_cfg.sync_input_keys()

    raw = gym.make(
        TASK,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    video_dir: Path | None = None
    if args_cli.video:
        video_dir = output / "videos"
        video_prefix = (
            f"random{num_envs}-codes"
            if args_cli.independent_random_codes
            else f"group{group:02d}-random{num_envs}-intervention"
        )
        raw = gym.wrappers.RecordVideo(
            raw,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.rollout_steps,
            name_prefix=video_prefix,
            disable_logger=True,
        )
    env, base = _wrap_for_policy(raw, args_cli.rollout_steps)
    try:
        agent = IPMD(env=env, config=agent_cfg)
        agent.load_model(str(args_cli.policy_checkpoint.resolve()))
        policy = agent.actor_critic.get_policy_operator()
        policy.eval()
        encoder, encoder_config = _load_encoder(67, torch.device(agent.device))
        baseline_codes = torch.as_tensor(
            base_code_categories, device=agent.device, dtype=torch.long
        )
        if (encoder.groups, encoder.categories) != (64, 128):
            raise ValueError(
                "The intervention expects a 64-group, 128-category encoder."
            )
        group_ids = torch.arange(encoder.groups, device=agent.device)
        baseline_z = encoder.codebook[group_ids, baseline_codes].reshape(-1)
        codes = torch.as_tensor(code_categories, device=agent.device, dtype=torch.long)
        variant_z = encoder.codebook[group_ids.unsqueeze(0), codes].reshape(
            num_envs, -1
        )
        code_dim = encoder.codebook.shape[-1]
        if not args_cli.independent_random_codes:
            unchanged = variant_z.clone()
            unchanged[:, group * code_dim : (group + 1) * code_dim] = baseline_z[
                group * code_dim : (group + 1) * code_dim
            ]
            torch.testing.assert_close(unchanged, baseline_z.repeat(num_envs, 1))

        td = env.reset()
        initial = _snapshot(base)
        env_origins = base.scene.env_origins.detach().cpu().numpy()
        for key in ("joint_pos", "joint_vel", "root_pos", "root_quat"):
            np.testing.assert_allclose(
                initial[key],
                np.repeat(initial[key][:1], num_envs, axis=0),
                atol=1e-5,
                rtol=0,
            )
        if args_cli.random_base_code or args_cli.independent_random_codes:
            phase_offset = 0.0
            native_baseline_equivalence: bool | None = None
        else:
            native = agent._hl_skill_command_sampler.sample_for_step(
                td, device=torch.device(agent.device), dtype=torch.float32
            )
            torch.testing.assert_close(
                native[:, : baseline_z.numel()],
                baseline_z.repeat(num_envs, 1),
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                native[:, -2:], native[:1, -2:].repeat(num_envs, 1)
            )
            phase_offset = math.atan2(
                float(native[0, -2].item()), float(native[0, -1].item())
            )
            manual0 = torch.cat(
                [
                    baseline_z.repeat(num_envs, 1),
                    native[:1, -2:].repeat(num_envs, 1),
                ],
                dim=-1,
            )
            torch.testing.assert_close(native, manual0, atol=1e-5, rtol=1e-5)
            native_td = td.clone()
            manual_td = td.clone()
            native_td.set(("policy", "latent_command"), native)
            manual_td.set(("policy", "latent_command"), manual0)
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                native_action = policy(native_td).get("action")
                manual_action = policy(manual_td).get("action")
            torch.testing.assert_close(
                native_action, manual_action, atol=1e-5, rtol=1e-5
            )
            native_baseline_equivalence = True

        snapshots = [initial]
        actions: list[np.ndarray] = []
        for step in range(args_cli.rollout_steps):
            angle = phase_offset + 2.0 * math.pi * (step % 10) / 10.0
            phase = torch.tensor(
                [math.sin(angle), math.cos(angle)],
                device=agent.device,
                dtype=variant_z.dtype,
            ).repeat(num_envs, 1)
            command = torch.cat([variant_z, phase], dim=-1)
            base.set_agent_latent_command(command)
            td = _refresh(td, base)
            td.set(("policy", "latent_command"), command)
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = policy(td)
                actions.append(td.get("action").detach().cpu().numpy())
                stepped = env.step(td)
                td = step_mdp(
                    stepped,
                    exclude_reward=True,
                    exclude_done=False,
                    exclude_action=True,
                )
            snapshots.append(_snapshot(base))
    finally:
        env.close()

    video_paths = sorted(video_dir.glob("*.mp4")) if video_dir is not None else []
    if args_cli.video and not video_paths:
        raise RuntimeError(f"No video was written to {video_dir}.")
    stacked = {
        key: np.stack([snapshot[key] for snapshot in snapshots], axis=1)
        for key in snapshots[0]
    }
    action_array = np.stack(actions, axis=1)
    baseline_index = (
        0
        if args_cli.independent_random_codes
        else int(np.flatnonzero(category_ids == baseline_category)[0])
    )
    final_root = stacked["root_pos"][:, -1] - stacked["root_pos"][:, 0]
    final_joint = stacked["joint_pos"][:, -1] - stacked["joint_pos"][:, 0]
    final_body = stacked["body_pos"][:, -1] - stacked["body_pos"][:, 0]
    metrics = np.column_stack(
        [
            final_root,
            np.linalg.norm(final_joint, axis=1),
            np.linalg.norm(final_body, axis=-1).mean(axis=1),
            np.linalg.norm(action_array, axis=-1).mean(axis=1),
            stacked["root_pos"][:, :, 2].min(axis=1),
        ]
    )
    metric_names = [
        "root_dx",
        "root_dy",
        "root_dz",
        "joint_delta_l2",
        "body_delta_mean",
        "action_l2_mean",
        "min_root_height",
    ]
    renewal_steps = np.arange(0, args_cli.rollout_steps, 10, dtype=np.int64)
    label_name = "code_index" if args_cli.independent_random_codes else "category"
    rows = [
        {
            label_name: int(category),
            "is_baseline": int(index == baseline_index),
            **{
                name: float(value)
                for name, value in zip(metric_names, row, strict=True)
            },
        }
        for index, (category, row) in enumerate(zip(category_ids, metrics, strict=True))
    ]
    np.savez_compressed(
        output / "rollouts.npz",
        category=category_ids,
        group=np.asarray([group]),
        baseline_category=np.asarray([baseline_category]),
        base_code_category=base_code_categories,
        sampled_category=category_ids,
        sampled_code_category=code_categories,
        code_renewal_step=renewal_steps,
        latent=variant_z.detach().cpu().numpy(),
        action=action_array,
        env_origin=env_origins,
        **stacked,
    )
    _write_csv(output / "effects.csv", rows)
    _write_csv(
        output / "category_layout.csv",
        [
            {
                "env_index": index,
                label_name: int(category),
                "is_baseline": int(index == baseline_index),
                "code_categories": " ".join(
                    str(value) for value in code_categories[index]
                ),
                "env_origin_x": float(env_origins[index, 0]),
                "env_origin_y": float(env_origins[index, 1]),
                "env_origin_z": float(env_origins[index, 2]),
            }
            for index, category in enumerate(category_ids)
        ],
    )
    _plot_intervention(
        output,
        metrics,
        metric_names,
        stacked["body_pos"],
        category_ids,
        baseline_index,
        label_name,
    )
    _provenance(
        output,
        {
            "motion": args_cli.motion,
            "start_frame": 0,
            "group": None if args_cli.independent_random_codes else group,
            "baseline_category": (
                None if args_cli.independent_random_codes else baseline_category
            ),
            "categories": int(num_envs),
            "rollout_steps": args_cli.rollout_steps,
            "random_base_code": bool(args_cli.random_base_code),
            "independent_random_codes": bool(args_cli.independent_random_codes),
            "base_code_categories": base_code_categories.tolist(),
            "sampled_categories": (
                None if args_cli.independent_random_codes else category_ids.tolist()
            ),
            "sampled_code_categories": code_categories.tolist(),
            "code_is_constant": True,
            "code_renewal_period_steps": 10,
            "code_renewal_steps": renewal_steps.tolist(),
            "encoder_config": encoder_config,
            "protocol": protocol,
            "phase_offset_radians": phase_offset,
            "native_baseline_equivalence": native_baseline_equivalence,
            "video_paths": [str(path.resolve()) for path in video_paths],
        },
    )
    _json(
        output / "summary.json",
        {
            "group": None if args_cli.independent_random_codes else group,
            "baseline_category": (
                None if args_cli.independent_random_codes else baseline_category
            ),
            "categories": int(num_envs),
            "random_base_code": bool(args_cli.random_base_code),
            "independent_random_codes": bool(args_cli.independent_random_codes),
            "native_baseline_equivalence": native_baseline_equivalence,
            "video_paths": [str(path.resolve()) for path in video_paths],
            "artifacts": sorted(str(path.resolve()) for path in output.iterdir()),
        },
    )
    print(f"[INFO] Intervention outputs: {output.resolve()}")
    for path in video_paths:
        print(f"[INFO] Retained video: {path.resolve()}")


@hydra_task_config(TASK, AGENT_ENTRY_POINT)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: Any,
) -> None:
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if args_cli.stride <= 0 or args_cli.batch_size <= 0:
        raise ValueError("--stride and --batch-size must be positive.")
    if args_cli.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive.")
    if args_cli.random_base_code and args_cli.independent_random_codes:
        raise ValueError(
            "--random-base-code and --independent-random-codes are mutually exclusive."
        )
    for path in (args_cli.skill_checkpoint, args_cli.manifest, args_cli.dataset):
        if not path.resolve().exists():
            raise FileNotFoundError(path.resolve())
    with launch_simulation(env_cfg, args_cli):
        if args_cli.mode == "single":
            _run_single(env_cfg)
        elif args_cli.mode == "multi":
            _run_multi(env_cfg)
        else:
            _run_intervene(env_cfg, agent_cfg)


if __name__ == "__main__":
    main()
