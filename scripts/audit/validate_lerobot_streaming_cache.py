#!/usr/bin/env python3

"""Probe the LeRobot streaming cache across one or more Unitree WBT repos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _append_workspace_sources() -> None:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]
    candidate_paths = [
        repo_root / "ImitationLearningTools",
        repo_root / "ImitationLearningTools" / "iltools",
    ]
    for candidate in candidate_paths:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


_append_workspace_sources()

from iltools.datasets.lerobot_stream import (  # noqa: E402
    LeRobotStreamingCacheConfig,
    StreamingTensorDictReplayCache,
    UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES,
    UnitreeG1WBT29DofMapper,
    UnitreeG1WBT29DofMapperConfig,
)


def _load_repo_ids(path: str | None) -> list[str]:
    if path is None:
        return []
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(repo_id) for repo_id in payload]
    if isinstance(payload, dict) and isinstance(payload.get("repo_ids"), list):
        return [str(repo_id) for repo_id in payload["repo_ids"]]
    raise ValueError("--repo_ids_file must contain a JSON list or {'repo_ids': [...]}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that Unitree LeRobot rows can stream into the TorchRL "
            "offline cache across one or more repos."
        )
    )
    parser.add_argument(
        "--repo_ids",
        nargs="*",
        default=None,
        help="One or more Hugging Face dataset repo ids.",
    )
    parser.add_argument(
        "--repo_ids_file",
        type=str,
        default="data/unitree/g1_wbt_lerobot_repos.json",
        help="JSON list or object with repo_ids. Used when --repo_ids is empty.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/iltools_lerobot_stream_probe",
    )
    parser.add_argument("--min_ready_transitions", type=int, default=64)
    parser.add_argument("--max_cache_transitions", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="Total episode cap. Use 0 for no total cap.",
    )
    parser.add_argument(
        "--max_episodes_per_repo",
        type=int,
        default=1,
        help="Per-repo episode cap for bounded probes. Use 0 for no per-repo cap.",
    )
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=300.0,
        help="Readiness timeout in seconds.",
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        default=False,
        help="After readiness, wait for the bounded stream to finish.",
    )
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        default=False,
        help="Keep the local memmap cache after the probe.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_ids = list(args.repo_ids or []) or _load_repo_ids(args.repo_ids_file)
    if len(repo_ids) == 0:
        raise ValueError("At least one repo id is required.")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")
    if args.min_ready_transitions <= 0:
        raise ValueError("--min_ready_transitions must be positive.")
    if args.max_cache_transitions < args.min_ready_transitions:
        raise ValueError("--max_cache_transitions must be >= --min_ready_transitions.")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    if cache_dir.exists() and not args.keep_cache:
        shutil.rmtree(cache_dir)

    mapper_cfg = UnitreeG1WBT29DofMapperConfig(
        dt=1.0 / float(args.fps),
        default_joint_pos=[0.0] * 29,
        action_scale=[1.0] * 29,
        dataset_joint_names=UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES,
        target_joint_names=UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES,
    )
    mapper = UnitreeG1WBT29DofMapper(mapper_cfg)
    cache_cfg = LeRobotStreamingCacheConfig(
        repo_id=repo_ids[0],
        repo_ids=tuple(repo_ids),
        split=str(args.split),
        cache_dir=cache_dir,
        min_ready_transitions=int(args.min_ready_transitions),
        max_cache_transitions=int(args.max_cache_transitions),
        low_watermark=min(
            int(args.min_ready_transitions),
            int(args.max_cache_transitions),
        ),
        starvation_timeout_s=float(args.timeout_s),
        batch_size=int(args.batch_size),
        max_episodes=None if int(args.max_episodes) <= 0 else int(args.max_episodes),
        max_episodes_per_repo=None
        if int(args.max_episodes_per_repo) <= 0
        else int(args.max_episodes_per_repo),
        mapper=mapper_cfg,
    )
    cache = StreamingTensorDictReplayCache(cache_cfg, mapper=mapper)
    try:
        cache.start()
        cache.wait_until_ready(timeout_s=float(args.timeout_s))
        if args.drain:
            thread = cache._thread
            if thread is not None:
                thread.join(timeout=float(args.timeout_s))
                if thread.is_alive():
                    raise TimeoutError("Timed out waiting for LeRobot stream drain.")
        sample = cache.sample(int(args.batch_size))
        summary = {
            "repo_count": len(repo_ids),
            "repos": repo_ids,
            "repos_completed": cache._repos_completed,
            "ready_transitions": cache.ready_transitions,
            "episodes_written": cache._episodes_written,
            "sample_batch_size": sample.batch_size.numel(),
            "sample_keys": sorted(str(key) for key in sample.keys(True)),
            "policy_joint_pos_shape": list(sample.get(("policy", "joint_pos")).shape),
            "expert_action_shape": list(sample["expert_action"].shape),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        cache.stop()
        if cache_dir.exists() and not args.keep_cache:
            shutil.rmtree(cache_dir)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
