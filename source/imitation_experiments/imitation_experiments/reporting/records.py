"""Reduce an evaluation ``summary.json`` into one auditable report record.

The headline tracking number on this project is the **episode-mean** MPJPE:
the unweighted mean of ``per_environment[i].tracking_metrics.tracking_mpjpe_mm``
over the episodes of a run. A balanced run (equal episodes per motion) makes
that identical to the motion-averaged value, which is how the campaign tables
are read.

Two other MPJPE reductions also live in the same file and must not be confused
with it:

* ``metric_means.tracking_mpjpe_mm`` weights every valid transition, so long
  episodes dominate. On the 2026-08-13 best planner arm it reads 53.62 mm
  against the 46.95 mm headline.
* ``successful_trajectory_metrics.tracking_mpjpe_mm`` drops failed episodes,
  which flatters an arm exactly where it is weakest (48.27 mm on that arm).

All three are carried on the record so the page can show the spread instead of
silently picking one.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class MotionScore:
    """Per-motion reduction over the episodes that ran that motion."""

    motion_name: str
    episode_count: int
    mpjpe_mm: float
    fall_free_rate: float


@dataclass(frozen=True)
class EpisodeScore:
    """One scored episode with the identity used for exact board matching."""

    trajectory_rank: int | None
    motion_name: str
    start_frame: int | None
    env_seed: int | None
    repeat_index: int | None
    mpjpe_mm: float | None
    fell: bool

    @property
    def identity(self) -> tuple[int, int, int, int] | None:
        if None in (
            self.trajectory_rank,
            self.start_frame,
            self.env_seed,
            self.repeat_index,
        ):
            return None
        return (
            int(self.trajectory_rank),
            int(self.start_frame),
            int(self.env_seed),
            int(self.repeat_index),
        )


@dataclass(frozen=True)
class EvalRecord:
    """One evaluation run, reduced to the values the results page reports."""

    label: str
    run_dir: str
    summary_path: str
    completed_at: str

    contract_hash: str | None
    protocol_id: str
    protocol_hash: str | None
    protocol_backend: str | None
    board_id: str | None
    board_hash: str | None
    profile_id: str | None
    profile_hash: str | None

    task: str | None
    algorithm: str | None
    interface: str | None
    kind: str  # "oracle" | "planner"

    tracker_checkpoint: str | None
    tracker_sha256: str | None
    tracker_frozen: bool | None
    tracker_parameters: int | None

    planner_checkpoint: str | None
    planner_update: int | None
    planner_quantizer: str | None
    planner_latent_dim: int | None
    planner_action_horizon: int | None
    planner_consumption: str | None
    planner_type: str | None
    planner_parameters: int | None
    planner_latency_p50_ms: float | None
    published_vs_oracle_z_cosine: float | None

    # Inference knobs, recorded in the artifact from 2026-08-16 onward. Older
    # summaries carry None here and the spec must declare them by hand.
    planner_temporal_ensemble: str | None
    planner_temporal_ensemble_decay: float | None
    planner_inference_steps: int | None
    planner_samples_per_publication: int | None
    planner_consume_slots: int | None

    episode_count: int
    motion_count: int
    max_steps: int | None
    episode_length_s: float | None
    seed: int | None
    push_disabled: bool | None
    fall_height_m: float | None
    termination_profile: str | None
    done_rate: float | None
    valid_transition_count: int | None

    mpjpe_mm: float | None
    mpjpe_mm_transition_weighted: float | None
    mpjpe_mm_successful_only: float | None
    fall_free_rate: float | None
    tracking_success_rate: float | None
    threshold_tracking_success_rate: float | None
    survival_steps_mean: float | None

    per_motion: tuple[MotionScore, ...] = field(default=())
    episodes: tuple[EpisodeScore, ...] = field(default=())

    @property
    def is_balanced(self) -> bool:
        """True when every motion received the same number of episodes."""
        if not self.per_motion:
            return False
        counts = {score.episode_count for score in self.per_motion}
        return len(counts) == 1

    @property
    def episodes_per_motion(self) -> int | None:
        return self.per_motion[0].episode_count if self.is_balanced else None

    @property
    def protocol_pinned(self) -> bool:
        return self.protocol_id != "unpinned" and self.protocol_hash is not None


def _get(mapping: Any, *path: str) -> Any:
    """Return a nested value, or ``None`` if any hop is missing or not a dict."""
    node = mapping
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _mpjpe_mm(metrics: dict[str, Any]) -> float | None:
    """Read a millimetre MPJPE from a metric block, converting metres if needed."""
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("tracking_mpjpe_mm")
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        metres = metrics.get("tracking_mpjpe_m")
        if isinstance(metres, dict):
            metres = metres.get("mean")
        value = None if metres is None else float(metres) * 1000.0
    return None if value is None else float(value)


def _reduce_per_motion(
    environments: list[Any],
    *,
    board_cases: list[Any],
    default_seed: int | None,
    default_start_frame: int | None,
) -> tuple[tuple[MotionScore, ...], tuple[EpisodeScore, ...], float | None, int, int]:
    """Group episode rows by motion and return scores plus the episode mean."""
    by_motion: dict[str, list[tuple[float | None, bool]]] = defaultdict(list)
    episode_values: list[float] = []
    episode_scores: list[EpisodeScore] = []

    for index, environment in enumerate(environments):
        if not isinstance(environment, dict):
            continue
        board_case = board_cases[index] if index < len(board_cases) else {}
        board_case = board_case if isinstance(board_case, dict) else {}
        motion = str(environment.get("motion_name") or "unnamed")
        value = _mpjpe_mm(environment.get("tracking_metrics") or {})
        fell = bool(environment.get("fell"))
        by_motion[motion].append((value, fell))
        episode_scores.append(
            EpisodeScore(
                trajectory_rank=_optional_int(
                    environment.get("trajectory_rank", board_case.get("trajectory_rank"))
                ),
                motion_name=motion,
                start_frame=_optional_int(
                    environment.get(
                        "start_frame",
                        board_case.get("start_frame", default_start_frame),
                    )
                ),
                env_seed=_optional_int(
                    environment.get("env_seed", board_case.get("env_seed", default_seed))
                ),
                repeat_index=_optional_int(
                    environment.get("repeat_index", board_case.get("repeat_index", 0))
                ),
                mpjpe_mm=value,
                fell=fell,
            )
        )
        if value is not None:
            episode_values.append(value)

    scores: list[MotionScore] = []
    for motion, rows in sorted(by_motion.items()):
        values = [value for value, _ in rows if value is not None]
        scores.append(
            MotionScore(
                motion_name=motion,
                episode_count=len(rows),
                mpjpe_mm=mean(values) if values else float("nan"),
                fall_free_rate=1.0 - (sum(fell for _, fell in rows) / len(rows)),
            )
        )

    episode_mean = mean(episode_values) if episode_values else None
    return (
        tuple(scores),
        tuple(episode_scores),
        episode_mean,
        len(episode_values),
        len(by_motion),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def load_summary(summary_path: Path, repo_root: Path) -> EvalRecord:
    """Load one ``summary.json`` into an :class:`EvalRecord`.

    Args:
        summary_path: Path to the evaluation summary file.
        repo_root: Repository root, used to store repo-relative provenance paths.
    """
    summary_path = Path(summary_path)
    with summary_path.open() as handle:
        summary = json.load(handle)

    metadata = summary.get("metadata") or {}
    aggregate = summary.get("aggregate") or {}
    contract = summary.get("contract") or {}
    protocol = summary.get("protocol") or {}
    board = summary.get("board") or {}
    profile = summary.get("profile") or {}
    planner = metadata.get("gr00t_planner")
    planner = planner if isinstance(planner, dict) else None

    run_dir = summary_path.parent
    try:
        relative_dir = str(run_dir.relative_to(repo_root))
    except ValueError:
        relative_dir = str(run_dir)

    environments = summary.get("per_environment") or []
    board_cases = board.get("cases") if isinstance(board, dict) else []
    board_cases = board_cases if isinstance(board_cases, list) else []
    per_motion, episodes, episode_mean, episode_count, motion_count = _reduce_per_motion(
        environments,
        board_cases=board_cases,
        default_seed=_optional_int(metadata.get("seed")),
        default_start_frame=_optional_int(metadata.get("reference_start_frame")),
    )

    declared_motions = metadata.get("motion_names") or summary.get("motion_names") or []
    if motion_count == 0 and isinstance(declared_motions, list):
        motion_count = len(declared_motions)

    return EvalRecord(
        label=str(metadata.get("label") or run_dir.name),
        run_dir=relative_dir,
        summary_path=str(summary_path.relative_to(repo_root))
        if summary_path.is_relative_to(repo_root)
        else str(summary_path),
        completed_at=_iso_day(summary_path),
        contract_hash=contract.get("content_hash"),
        protocol_id=str(protocol.get("protocol_id") or "unpinned"),
        protocol_hash=protocol.get("content_hash"),
        protocol_backend=protocol.get("backend"),
        board_id=board.get("board_id"),
        board_hash=board.get("content_hash"),
        profile_id=profile.get("profile_id"),
        profile_hash=profile.get("content_hash"),
        task=summary.get("task") or metadata.get("task"),
        algorithm=summary.get("algorithm") or metadata.get("algorithm"),
        interface=metadata.get("interface"),
        kind="planner" if planner and planner.get("checkpoint") else "oracle",
        tracker_checkpoint=_get(metadata, "low_level_tracker", "checkpoint_path")
        or summary.get("low_level_checkpoint"),
        tracker_sha256=_get(metadata, "low_level_tracker", "checkpoint_sha256"),
        tracker_frozen=_get(metadata, "low_level_tracker", "policy_frozen"),
        tracker_parameters=_get(
            metadata, "low_level_tracker", "policy_parameter_count"
        ),
        planner_checkpoint=(planner or {}).get("checkpoint"),
        planner_update=(planner or {}).get("update"),
        planner_quantizer=(planner or {}).get("quantizer"),
        planner_latent_dim=(planner or {}).get("action_dim"),
        planner_action_horizon=(planner or {}).get("action_horizon"),
        planner_consumption=(planner or {}).get("consumption"),
        planner_type=_get(metadata, "planner_metadata", "planner_type"),
        planner_parameters=_get(metadata, "planner_metadata", "parameter_count"),
        planner_latency_p50_ms=_get(planner or {}, "planner_latency_ms", "p50"),
        published_vs_oracle_z_cosine=(planner or {}).get(
            "published_vs_oracle_z_cosine_mean"
        ),
        planner_temporal_ensemble=(planner or {}).get("temporal_ensemble"),
        planner_temporal_ensemble_decay=(planner or {}).get("temporal_ensemble_decay"),
        planner_inference_steps=(planner or {}).get("num_inference_timesteps"),
        planner_samples_per_publication=(planner or {}).get("samples_per_publication"),
        planner_consume_slots=(planner or {}).get("consume_slots"),
        episode_count=episode_count or int(metadata.get("num_envs") or 0),
        motion_count=motion_count,
        max_steps=summary.get("max_steps"),
        episode_length_s=metadata.get("episode_length_s"),
        seed=metadata.get("seed"),
        push_disabled=summary.get("disable_push_event"),
        fall_height_m=metadata.get("fall_height_m"),
        termination_profile=summary.get("sonic_termination_profile"),
        done_rate=aggregate.get("done_rate"),
        valid_transition_count=aggregate.get("valid_transition_count"),
        mpjpe_mm=episode_mean,
        mpjpe_mm_transition_weighted=_mpjpe_mm(summary.get("metric_means") or {}),
        mpjpe_mm_successful_only=_mpjpe_mm(
            summary.get("successful_trajectory_metrics")
            or summary.get("successful_metrics")
            or {}
        ),
        fall_free_rate=aggregate.get("fall_free_rate"),
        tracking_success_rate=aggregate.get("tracking_success_rate"),
        threshold_tracking_success_rate=aggregate.get(
            "threshold_tracking_success_rate"
        ),
        survival_steps_mean=aggregate.get("survival_steps_mean"),
        per_motion=per_motion,
        episodes=episodes,
    )


def _iso_day(path: Path) -> str:
    """Return the summary file's modification day, used to order the history."""
    from datetime import datetime, timezone

    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.date().isoformat()
