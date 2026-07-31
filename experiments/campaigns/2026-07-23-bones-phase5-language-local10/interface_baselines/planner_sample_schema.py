"""Versioned sample records shared by all causal interface planners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


PLANNER_SAMPLE_FORMAT = "causal_interface_planner_sample"
PLANNER_SAMPLE_VERSION = 2
PAIRED_TARGET_CONTRACT = {
    "causal": {"state": "causal_state_history", "target": "causal_target"},
    "demonstration": {
        "state": "demonstration_state_history",
        "target": "demonstration_target",
    },
}


def _sample_row_count(sample: Mapping[str, Any]) -> int:
    target = sample.get("causal_target")
    if not isinstance(target, torch.Tensor) or target.ndim == 0:
        raise ValueError("Planner sample requires a row-shaped causal_target tensor.")
    return int(target.shape[0])


def concatenate_planner_samples(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join validated sample batches without changing their row order."""
    if not samples:
        raise ValueError("At least one planner sample is required.")
    row_counts = [_sample_row_count(sample) for sample in samples]
    expected_keys = set(samples[0])
    for sample in samples[1:]:
        if set(sample) != expected_keys:
            raise ValueError("Planner sample batches have different keys.")

    result: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        first = values[0]
        if key == "step":
            result[key] = min(int(value) for value in values)
        elif isinstance(first, torch.Tensor) and all(
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
            for value, rows in zip(values, row_counts, strict=True)
        ):
            result[key] = torch.cat(values, dim=0).contiguous()
        elif isinstance(first, list) and all(
            isinstance(value, list) and len(value) == rows
            for value, rows in zip(values, row_counts, strict=True)
        ):
            result[key] = [item for value in values for item in value]
        else:
            if any(value != first for value in values[1:]):
                raise ValueError(
                    f"Planner sample field {key!r} differs across batches."
                )
            result[key] = first
    return result


class PlannerSampleWriter:
    """Write planner rows in bounded chunks instead of one tiny file per step."""

    def __init__(self, output_dir: Path, *, rows_per_file: int = 1) -> None:
        if int(rows_per_file) <= 0:
            raise ValueError("rows_per_file must be positive.")
        self.output_dir = output_dir.expanduser().resolve()
        self.rows_per_file = int(rows_per_file)
        self._pending: list[Mapping[str, Any]] = []
        self._pending_rows = 0
        self.file_count = 0
        self.row_count = 0

    def add(self, sample: Mapping[str, Any]) -> None:
        rows = _sample_row_count(sample)
        self._pending.append(sample)
        self._pending_rows += rows
        self.row_count += rows
        if self._pending_rows >= self.rows_per_file:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample = concatenate_planner_samples(self._pending)
        destination = self.output_dir / f"sample_step_{self.file_count:06d}.pt"
        temporary = destination.with_suffix(".pt.partial")
        torch.save(sample, temporary)
        temporary.replace(destination)
        self.file_count += 1
        self._pending.clear()
        self._pending_rows = 0


def _select_sample_row(
    sample: Mapping[str, Any], *, index: int, rows: int
) -> dict[str, Any]:
    """Select one row while preserving batch-invariant sample metadata."""
    result: dict[str, Any] = {}
    for key, value in sample.items():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == rows
        ):
            result[key] = value[index : index + 1]
        elif isinstance(value, list) and len(value) == rows:
            result[key] = [value[index]]
        else:
            result[key] = value
    return result


class CompletedTrajectorySampleWriter:
    """Commit only complete, variable-length trajectory segments.

    Samples arrive at planner-publication time, before the environment reports
    whether the current control step ended an episode. Rows are therefore held
    by ``(env_id, episode_id)`` until :meth:`complete` is called. This prevents
    a global collection cutoff from leaking partially observed trajectories
    into the training set.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        motion_names: Sequence[str],
        trajectories_per_motion: int,
        rows_per_file: int = 1,
    ) -> None:
        names = tuple(str(name).strip() for name in motion_names)
        if not names or any(not name for name in names):
            raise ValueError("motion_names must contain non-empty names.")
        if len(names) != len(set(names)):
            raise ValueError("motion_names must be unique.")
        if int(trajectories_per_motion) <= 0:
            raise ValueError("trajectories_per_motion must be positive.")
        self.motion_names = names
        self.trajectories_per_motion = int(trajectories_per_motion)
        self._writer = PlannerSampleWriter(output_dir, rows_per_file=int(rows_per_file))
        self._buffers: dict[tuple[int, int], list[dict[str, Any]]] = {}
        self._motion_by_key: dict[tuple[int, int], str] = {}
        self._closed_keys: set[tuple[int, int]] = set()
        self._counts = {name: 0 for name in names}
        self._records: list[dict[str, Any]] = []

    @property
    def output_dir(self) -> Path:
        return self._writer.output_dir

    @property
    def file_count(self) -> int:
        return self._writer.file_count

    @property
    def row_count(self) -> int:
        return self._writer.row_count

    @property
    def complete_budget(self) -> bool:
        return all(
            count == self.trajectories_per_motion for count in self._counts.values()
        )

    @property
    def completed_trajectory_count(self) -> int:
        return sum(self._counts.values())

    @property
    def buffered_trajectory_count(self) -> int:
        return len(self._buffers)

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def missing(self) -> dict[str, int]:
        return {
            name: self.trajectories_per_motion - count
            for name, count in self._counts.items()
            if count < self.trajectories_per_motion
        }

    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def add(self, sample: Mapping[str, Any]) -> None:
        rows = _sample_row_count(sample)
        env_ids = sample.get("env_id")
        episode_ids = sample.get("episode_id")
        motion_names = sample.get("motion_name")
        if (
            not isinstance(env_ids, torch.Tensor)
            or not isinstance(episode_ids, torch.Tensor)
            or not isinstance(motion_names, list)
            or len(motion_names) != rows
        ):
            raise ValueError(
                "Completed-trajectory collection requires row-aligned env_id, "
                "episode_id, and motion_name fields."
            )
        env_values = env_ids.detach().cpu().reshape(-1).tolist()
        episode_values = episode_ids.detach().cpu().reshape(-1).tolist()
        if len(env_values) != rows or len(episode_values) != rows:
            raise ValueError("Trajectory identity tensors are not row-aligned.")
        for index, (env_id, episode_id, raw_motion) in enumerate(
            zip(env_values, episode_values, motion_names, strict=True)
        ):
            key = (int(env_id), int(episode_id))
            motion = str(raw_motion)
            if key in self._closed_keys:
                raise ValueError(f"Received a row for already closed trajectory {key}.")
            previous_motion = self._motion_by_key.setdefault(key, motion)
            if previous_motion != motion:
                raise ValueError(
                    f"Trajectory {key} changed motion from {previous_motion!r} "
                    f"to {motion!r}."
                )
            self._buffers.setdefault(key, []).append(
                _select_sample_row(sample, index=index, rows=rows)
            )

    def complete(
        self,
        *,
        env_ids: Sequence[int],
        episode_ids: Sequence[int],
        motion_names: Sequence[str],
    ) -> None:
        """Close segments and commit only those still inside the exact budget."""
        if not (len(env_ids) == len(episode_ids) == len(motion_names)):
            raise ValueError("Completed trajectory fields must have equal lengths.")
        for raw_env, raw_episode, raw_motion in zip(
            env_ids, episode_ids, motion_names, strict=True
        ):
            key = (int(raw_env), int(raw_episode))
            motion = str(raw_motion)
            if key in self._closed_keys:
                raise ValueError(f"Trajectory {key} was completed more than once.")
            rows = self._buffers.pop(key, [])
            buffered_motion = self._motion_by_key.pop(key, motion)
            if buffered_motion != motion:
                raise ValueError(
                    f"Completed trajectory {key} reports motion {motion!r}, "
                    f"but buffered rows use {buffered_motion!r}."
                )
            self._closed_keys.add(key)
            count = self._counts.get(motion)
            if count is None or count >= self.trajectories_per_motion:
                continue
            if not rows:
                raise ValueError(f"Completed trajectory {key} has no planner rows.")
            for sample in rows:
                self._writer.add(sample)
            self._counts[motion] = count + 1
            self._records.append(
                {
                    "env_id": key[0],
                    "episode_id": key[1],
                    "motion_name": motion,
                    "rows": len(rows),
                }
            )

    def flush(self) -> None:
        """Flush accepted trajectories; incomplete buffers remain discarded."""
        self._writer.flush()


def add_sample_format_metadata(
    metadata: Mapping[str, Any],
    *,
    collection_stage: str,
    planner_interval_steps: int,
    control_rate_hz: float = 50.0,
) -> dict[str, Any]:
    """Return metadata with the common data and timing contract attached."""
    interval = int(planner_interval_steps)
    if interval <= 0:
        raise ValueError("planner_interval_steps must be positive.")
    rate = float(control_rate_hz)
    if rate <= 0.0:
        raise ValueError("control_rate_hz must be positive.")
    result = dict(metadata)
    result["sample_format"] = {
        "name": PLANNER_SAMPLE_FORMAT,
        "version": PLANNER_SAMPLE_VERSION,
    }
    result["paired_target_contract"] = PAIRED_TARGET_CONTRACT
    result["collection_stage"] = str(collection_stage)
    result["control_rate_hz"] = rate
    result["planner_interval_steps"] = interval
    result["planner_rate_hz"] = rate / float(interval)
    return result


def _history(
    value: torch.Tensor,
    *,
    name: str,
    history_frames: int,
    frame_dim: int,
) -> torch.Tensor:
    if value.ndim == 2:
        expected_flat = int(history_frames) * int(frame_dim)
        if int(value.shape[-1]) != expected_flat:
            raise ValueError(
                f"{name} flat width {value.shape[-1]} does not match {expected_flat}."
            )
        value = value.reshape(value.shape[0], history_frames, frame_dim)
    expected_tail = (int(history_frames), int(frame_dim))
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape [N, {history_frames}, {frame_dim}], "
            f"got {tuple(value.shape)}."
        )
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _row_vector(
    value: torch.Tensor | int,
    *,
    name: str,
    rows: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long).detach().cpu().reshape(-1)
    if tensor.numel() == 1 and rows != 1:
        tensor = tensor.expand(rows).clone()
    if tuple(tensor.shape) != (int(rows),):
        raise ValueError(f"{name} must have {rows} rows, got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _target(
    value: torch.Tensor,
    *,
    name: str,
    rows: int,
    target_encoding: Mapping[str, Any],
) -> torch.Tensor:
    target_kind = str(target_encoding.get("kind", "continuous"))
    if target_kind == "continuous":
        target = value.detach().to(device="cpu", dtype=torch.float32)
    elif target_kind == "categorical_sequence":
        raw_target = value.detach().to(device="cpu")
        if raw_target.is_floating_point() and not torch.equal(
            raw_target, raw_target.round()
        ):
            raise ValueError(f"Categorical {name} contains non-integer values.")
        target = raw_target.to(dtype=torch.long)
        horizon = int(target_encoding.get("horizon", -1))
        codebook_size = int(target_encoding.get("codebook_size", -1))
        if horizon <= 0 or codebook_size <= 1:
            raise ValueError(
                "Categorical target encoding requires positive horizon and "
                "codebook_size greater than one."
            )
        if target.ndim == 2 and int(target.shape[-1]) != horizon:
            raise ValueError(
                f"Categorical {name} width {target.shape[-1]} != horizon {horizon}."
            )
        if target.numel() > 0 and (
            int(target.min().item()) < 0 or int(target.max().item()) >= codebook_size
        ):
            raise ValueError(f"Categorical {name} has an out-of-range token ID.")
    else:
        raise ValueError(f"Unsupported metadata.target_encoding kind {target_kind!r}.")
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.ndim != 2 or int(target.shape[0]) != rows:
        raise ValueError(f"{name} must be rank-2 with {rows} rows.")
    return target.contiguous()


def build_planner_sample(
    *,
    causal_state_history: torch.Tensor,
    demonstration_state_history: torch.Tensor,
    causal_target: torch.Tensor,
    demonstration_target: torch.Tensor,
    trajectory_rank: torch.Tensor,
    episode_id: torch.Tensor | int,
    control_step: torch.Tensor | int,
    planner_step: torch.Tensor | int,
    env_id: torch.Tensor | int | None = None,
    motion_names: Sequence[str],
    metadata: Mapping[str, Any],
    language_embedding: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Build one validated, serializable paired planner sample batch.

    State aliases are retained because they are unambiguous. There is no generic
    target alias: each target must be selected with its paired state.
    """
    observation_spec = metadata.get("planner_observation_spec")
    if not isinstance(observation_spec, Mapping):
        raise ValueError("metadata must contain planner_observation_spec.")
    sample_format = metadata.get("sample_format")
    expected_format = {"name": PLANNER_SAMPLE_FORMAT, "version": PLANNER_SAMPLE_VERSION}
    if sample_format != expected_format:
        raise ValueError(f"metadata sample_format must equal {expected_format}.")
    if metadata.get("paired_target_contract") != PAIRED_TARGET_CONTRACT:
        raise ValueError(
            "metadata paired_target_contract does not match the versioned "
            "causal/demonstration pairing contract."
        )
    history_frames = int(observation_spec.get("history_frames", -1))
    frame_dim = int(observation_spec.get("frame_dim", -1))
    if history_frames <= 0 or frame_dim <= 0:
        raise ValueError("planner_observation_spec has invalid history dimensions.")

    causal = _history(
        causal_state_history,
        name="causal_state_history",
        history_frames=history_frames,
        frame_dim=frame_dim,
    )
    demonstration = _history(
        demonstration_state_history,
        name="demonstration_state_history",
        history_frames=history_frames,
        frame_dim=frame_dim,
    )
    rows = int(causal.shape[0])
    if int(demonstration.shape[0]) != rows:
        raise ValueError(
            "Causal and demonstration histories have different row counts."
        )
    target_encoding = metadata.get("target_encoding", {"kind": "continuous"})
    if not isinstance(target_encoding, Mapping):
        raise ValueError("metadata.target_encoding must be a mapping when provided.")
    causal_target_cpu = _target(
        causal_target,
        name="causal_target",
        rows=rows,
        target_encoding=target_encoding,
    )
    demonstration_target_cpu = _target(
        demonstration_target,
        name="demonstration_target",
        rows=rows,
        target_encoding=target_encoding,
    )
    if causal_target_cpu.shape != demonstration_target_cpu.shape:
        raise ValueError(
            "Causal and demonstration targets must have identical shapes, got "
            f"{tuple(causal_target_cpu.shape)} and "
            f"{tuple(demonstration_target_cpu.shape)}."
        )
    names = [str(name) for name in motion_names]
    if len(names) != rows:
        raise ValueError(f"motion_names must have {rows} entries, got {len(names)}.")

    traj_rank = _row_vector(trajectory_rank, name="trajectory_rank", rows=rows)
    episode = _row_vector(episode_id, name="episode_id", rows=rows)
    control = _row_vector(control_step, name="control_step", rows=rows)
    planner = _row_vector(planner_step, name="planner_step", rows=rows)
    # `episode_id` counts resets per environment, so it is NOT a trajectory
    # identifier on its own: env 0 episode 1 and env 3 episode 1 are different
    # rollouts that happen to share a number. A trajectory-wise train/val split
    # needs the (env_id, episode_id) pair. Both are stored; consumers combine
    # them. No single uid is derived here because the shards are written
    # independently and any per-shard encoding would collide across files.
    # -1 marks a writer that predates this field.
    env = _row_vector(
        torch.full((rows,), -1, dtype=torch.long) if env_id is None else env_id,
        name="env_id",
        rows=rows,
    )
    record: dict[str, Any] = {
        "causal_state_history": causal,
        "demonstration_state_history": demonstration,
        "causal_target": causal_target_cpu,
        "demonstration_target": demonstration_target_cpu,
        "trajectory_rank": traj_rank,
        "episode_id": episode,
        "env_id": env,
        "control_step": control,
        "planner_step": planner,
        "motion_name": names,
        "metadata": dict(metadata),
        "planner_state": causal.flatten(1),
        "expert_planner_state": demonstration.flatten(1),
        "traj_rank": traj_rank,
        "step": int(planner.min().item()) if rows else 0,
    }
    if language_embedding is not None:
        lang = language_embedding.detach().to(device="cpu", dtype=torch.float32)
        if lang.ndim != 2 or int(lang.shape[0]) != rows:
            raise ValueError(f"language_embedding must be rank-2 with {rows} rows.")
        if int(lang.shape[1]) <= 0:
            raise ValueError(
                "language_embedding must have positive width when provided; "
                "omit it for a state-only planner sample."
            )
        record["language_embedding"] = lang.contiguous()
        record["lang"] = lang.contiguous()
    return record
