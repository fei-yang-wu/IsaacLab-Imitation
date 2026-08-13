"""GR00T action head as an Isaac latent-command source.

`Gr00tSkillCommandSampler` slots into the same place as
`FrozenSkillCommanderSampler`: the frozen low-level policy asks the sampler
for a latent command whenever an environment's hold expires, and the base
class owns hold length, phase channels, and per-environment renewal. Only
the production of `z` is replaced — here it comes from the verbatim GR00T
N1.7 action head reading the causal robot history and a cached language
goal.

Two consumption modes, matching the Embodied-Control rehearsal harness:

- `open_loop` (default, the "basic" arm): one head call per `slots`
  publications. Slot 0 is published immediately, slots 1..N-1 are cached per
  environment and consumed on the next renewals. This is what the head was
  trained to produce — consecutive published latents.
- `fresh`: one head call per publication, always publishing slot 0.

FSQ heads regress the PRE-quantization bounded vector (SONIC convention),
so this sampler snaps onto the lattice before publishing — the same
consume-time snap the deployment runtime performs.

The head runs in the `isaaclab` environment through the `gr00t.model` stub
(see `gr00t_head.ensure_gr00t_importable`); no GR00T dataset stack is
imported.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imitation_experiments.planner.gr00t_head import (
    PLANNER_STATE_HISTORY,
    PLANNER_STATE_WIDTH,
    build_batch,
    build_g1_head_config,
    denormalize_minmax,
    ensure_gr00t_importable,
    import_head_classes,
    normalize_minmax,
)


class Gr00tSkillCommandSampler:
    """Mixin providing GR00T-head `z` production for the frozen sampler.

    Applied to `FrozenSkillCommanderSampler` by `build_gr00t_sampler`, which
    keeps that class's renewal/phase machinery untouched.
    """

    def configure_gr00t(
        self,
        *,
        checkpoint_path: str | Path,
        goal_features_path: str | Path,
        goal_name: str | Sequence[str],
        num_envs: int,
        consumption: str = "open_loop",
        fsq_half_levels: Tensor | None = None,
        device: torch.device | str = "cuda",
        expected_target_mode: str = "latent",
        consume_slots: int | None = None,
        num_inference_timesteps: int = 4,
        samples_per_publication: int = 1,
        temporal_ensemble: str = "none",
        temporal_ensemble_decay: float = 0.5,
    ) -> dict[str, Any]:
        if consumption not in {"open_loop", "fresh"}:
            msg = f"consumption must be open_loop|fresh, got {consumption!r}."
            raise ValueError(msg)
        device = torch.device(device)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        goal_table = torch.load(
            Path(goal_features_path), map_location="cpu", weights_only=False
        )
        goal_names = list(goal_table["goal_names"])
        # One goal for every environment, or an explicit per-environment
        # assignment. The assignment is fixed for the whole run and is never
        # re-derived from the reference cursor or a trajectory reassignment.
        assigned = (
            [str(goal_name)] * int(num_envs)
            if isinstance(goal_name, str)
            else [str(name) for name in goal_name]
        )
        if len(assigned) != int(num_envs):
            msg = (
                f"per-environment goal list has {len(assigned)} entries for "
                f"{int(num_envs)} environments."
            )
            raise ValueError(msg)
        missing = sorted({name for name in assigned if name not in goal_names})
        if missing:
            msg = f"goals {missing} are not in the feature cache {goal_names}."
            raise KeyError(msg)
        self._gr00t_goal_names = assigned
        self._gr00t_goal_index = torch.tensor(
            [goal_names.index(name) for name in assigned],
            dtype=torch.long,
            device=device,
        )

        self._gr00t_horizon = int(checkpoint["action_horizon"])
        self._gr00t_action_dim = int(checkpoint.get("action_dim", 38))
        self._gr00t_target_mode = str(checkpoint.get("target_mode") or "latent")
        if self._gr00t_target_mode != expected_target_mode:
            msg = (
                f"expected a {expected_target_mode!r}-target head, got "
                f"target_mode={self._gr00t_target_mode!r}. A latent head "
                "publishes through the latent command channel; a chunk head "
                "publishes explicit packets through the chunk actor term."
            )
            raise ValueError(msg)

        # Install the `gr00t.model` stub BEFORE any GR00T import: the real
        # package __init__ drags the dataset stack (pandas/lmdb/msgpack), which
        # the isaaclab environment deliberately does not carry.
        ensure_gr00t_importable(stub_model_package=True)
        config = build_g1_head_config(
            trunk=checkpoint["trunk_config"],
            action_horizon=self._gr00t_horizon,
            max_action_dim=self._gr00t_action_dim,
            state_dropout_prob=0.0,
            num_inference_timesteps=int(num_inference_timesteps),
        )
        _, head_cls = import_head_classes(stub_model_package=True)
        head = head_cls(config)
        head.load_state_dict(checkpoint["head_state_dict"], strict=True)
        head.to(device).eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        self._gr00t_head = head

        norm = checkpoint["normalization"]
        self._gr00t_state_q01 = norm["state_q01"].to(device)
        self._gr00t_state_q99 = norm["state_q99"].to(device)
        self._gr00t_action_q01 = norm["action_q01"].to(device)
        self._gr00t_action_q99 = norm["action_q99"].to(device)

        # Whole table resident; rows are gathered per environment at publish.
        self._gr00t_features = goal_table["features"].to(device)
        self._gr00t_feature_mask = goal_table["attention_mask"].to(device).bool()
        self._gr00t_device = device
        self._gr00t_consumption = consumption
        # Slots consumed before re-planning. Below the horizon this is a
        # receding-horizon schedule: a hold-1 SONIC-style head predicts 30
        # latents but republishes every 10 control steps, so the last 20 slots
        # exist only to give the prediction a longer context and are dropped.
        self._gr00t_slots = (
            self._gr00t_horizon if consume_slots is None else int(consume_slots)
        )
        if not 1 <= self._gr00t_slots <= self._gr00t_horizon:
            msg = (
                f"consume_slots must be in 1..{self._gr00t_horizon}, got "
                f"{self._gr00t_slots}."
            )
            raise ValueError(msg)
        self._gr00t_fsq_half = (
            None if fsq_half_levels is None else fsq_half_levels.to(device)
        )
        # Per-environment cache of not-yet-published slots.
        self._gr00t_cache = torch.zeros(
            (int(num_envs), self._gr00t_horizon, self._gr00t_action_dim),
            device=device,
            dtype=torch.float32,
        )
        self._gr00t_cursor = torch.full(
            (int(num_envs),), self._gr00t_horizon, dtype=torch.long, device=device
        )
        # Flow matching starts from a fresh noise draw, so two calls on the
        # same state give different latents. Averaging several draws before
        # the FSQ snap cuts that sampling variance at inference cost only, with
        # no retraining. Averaging is done on the CONTINUOUS pre-snap value:
        # averaging snapped lattice points would drift off the lattice.
        self._gr00t_samples = int(samples_per_publication)
        if self._gr00t_samples < 1:
            msg = (
                f"samples_per_publication must be >= 1, got {samples_per_publication}."
            )
            raise ValueError(msg)
        # Temporal ensembling (ACT-style). At publication k the head predicts
        # slots covering publications k..k+H-1, so publication k is also
        # covered by the predictions made at k-1, k-2, ... Blending those
        # overlapping estimates averages independent flow draws taken from
        # DIFFERENT states, which is a different operation from averaging
        # redraws at one state: it smooths the published sequence rather than
        # collapsing one prediction toward its conditional mean.
        if temporal_ensemble not in {"none", "exponential"}:
            msg = f"temporal_ensemble must be none|exponential, got {temporal_ensemble!r}."
            raise ValueError(msg)
        self._gr00t_ensemble = temporal_ensemble
        self._gr00t_ensemble_decay = float(temporal_ensemble_decay)
        if temporal_ensemble != "none":
            # ring[n, age, slot, :] = the prediction made `age` publications
            # ago. Publication k reads slot `age` of the entry aged `age`.
            self._gr00t_ring = torch.zeros(
                (
                    int(num_envs),
                    self._gr00t_horizon,
                    self._gr00t_horizon,
                    self._gr00t_action_dim,
                ),
                device=device,
                dtype=torch.float32,
            )
            self._gr00t_ring_age = torch.zeros(
                (int(num_envs),), dtype=torch.long, device=device
            )
        self._gr00t_calls = 0
        self._gr00t_latency_ms: list[float] = []
        return {
            "checkpoint": str(checkpoint_path),
            "goal_name": goal_name if isinstance(goal_name, str) else "per_env",
            "goal_assignment": sorted(set(assigned)),
            "goals_per_env": len(set(assigned)),
            "goal_features": str(goal_features_path),
            "action_horizon": self._gr00t_horizon,
            "action_dim": self._gr00t_action_dim,
            "consumption": consumption,
            "quantizer": "fsq" if fsq_half_levels is not None else "none",
            "update": int(checkpoint.get("update", -1)),
        }

    def _gr00t_predict(
        self, planner_state: Tensor, goal_index: Tensor | None = None
    ) -> Tensor:
        """Run the head on `[B, 930]` causal history; return `[B, H, D]`.

        `goal_index` selects one cached language goal per row. Passing None
        means every row in this batch shares environment 0's goal, which is
        only correct for a single-goal run.
        """
        state = planner_state.reshape(
            -1, PLANNER_STATE_HISTORY, PLANNER_STATE_WIDTH
        ).to(self._gr00t_device, torch.float32)
        state = normalize_minmax(state, self._gr00t_state_q01, self._gr00t_state_q99)
        rows = int(state.shape[0])
        if goal_index is None:
            goal_index = self._gr00t_goal_index[:1].expand(rows)
        goal_index = goal_index.to(self._gr00t_device).reshape(-1)
        if int(goal_index.numel()) != rows:
            msg = (
                f"goal index has {int(goal_index.numel())} entries for {rows} "
                "planner rows."
            )
            raise ValueError(msg)
        backbone_output, action_input = build_batch(
            state=state,
            action=None,
            action_mask=None,
            language_features=self._gr00t_features.index_select(0, goal_index),
            language_attention_mask=self._gr00t_feature_mask.index_select(
                0, goal_index
            ),
        )
        # Latency is measured around the head's root forward call only, CUDA
        # synchronized, with the warmup call excluded by the caller.
        if self._gr00t_device.type == "cuda":
            torch.cuda.synchronize(self._gr00t_device)
        start = (
            torch.cuda.Event(enable_timing=True)
            if self._gr00t_device.type == "cuda"
            else None
        )
        end = torch.cuda.Event(enable_timing=True) if start is not None else None
        if start is not None:
            start.record()
        with torch.inference_mode():
            result = self._gr00t_head.get_action(backbone_output, action_input, None)
        if start is not None:
            end.record()
            torch.cuda.synchronize(self._gr00t_device)
            self._gr00t_latency_ms.append(float(start.elapsed_time(end)))
        self._gr00t_calls += 1
        prediction = result["action_pred"].float()
        prediction = denormalize_minmax(
            prediction, self._gr00t_action_q01, self._gr00t_action_q99
        )
        if getattr(self, "_gr00t_samples", 1) > 1:
            for _ in range(self._gr00t_samples - 1):
                with torch.inference_mode():
                    extra = self._gr00t_head.get_action(
                        backbone_output, action_input, None
                    )
                prediction = prediction + denormalize_minmax(
                    extra["action_pred"].float(),
                    self._gr00t_action_q01,
                    self._gr00t_action_q99,
                )
                self._gr00t_calls += 1
            prediction = prediction / float(self._gr00t_samples)
        if not torch.isfinite(prediction).all():
            msg = "GR00T action head produced a non-finite prediction."
            raise RuntimeError(msg)
        return prediction

    def gr00t_z(self, planner_state: Tensor, env_ids: Tensor) -> Tensor:
        """Latent to publish for each environment in `env_ids`."""
        env_ids = env_ids.to(self._gr00t_device).reshape(-1)
        if getattr(self, "_gr00t_ensemble", "none") == "exponential":
            horizon = self._gr00t_horizon
            prediction = self._gr00t_predict(
                planner_state, self._gr00t_goal_index[env_ids]
            )
            ring = self._gr00t_ring[env_ids]
            ring = torch.roll(ring, shifts=1, dims=1)
            ring[:, 0] = prediction
            self._gr00t_ring[env_ids] = ring
            age = torch.clamp(self._gr00t_ring_age[env_ids] + 1, max=horizon)
            self._gr00t_ring_age[env_ids] = age
            # Slot `i` of the entry aged `i` is this publication's estimate
            # from `i` publications back. Entries older than the number of
            # publications since reset are stale and must not contribute.
            slots = torch.arange(horizon, device=self._gr00t_device)
            candidates = ring[:, slots, slots, :]
            weight = torch.pow(
                torch.tensor(self._gr00t_ensemble_decay, device=self._gr00t_device),
                slots.float(),
            ).unsqueeze(0)
            weight = weight * (slots.unsqueeze(0) < age.unsqueeze(1)).float()
            weight = weight / weight.sum(dim=1, keepdim=True).clamp(min=1.0e-8)
            z = (candidates * weight.unsqueeze(-1)).sum(dim=1)
            if self._gr00t_fsq_half is not None:
                half = self._gr00t_fsq_half
                z = torch.clamp(torch.round(z * half), -half, half - 1.0) / half
            return z.to(dtype=torch.float32)
        if self._gr00t_consumption == "fresh":
            z = self._gr00t_predict(planner_state, self._gr00t_goal_index[env_ids])[
                :, 0
            ]
        else:
            needs = self._gr00t_cursor[env_ids] >= self._gr00t_slots
            if bool(needs.any()):
                rows = needs.nonzero(as_tuple=False).reshape(-1)
                prediction = self._gr00t_predict(
                    planner_state[rows], self._gr00t_goal_index[env_ids[rows]]
                )
                self._gr00t_cache[env_ids[rows]] = prediction
                self._gr00t_cursor[env_ids[rows]] = 0
            cursor = self._gr00t_cursor[env_ids]
            z = self._gr00t_cache[env_ids, cursor]
            self._gr00t_cursor[env_ids] = cursor + 1
        if self._gr00t_fsq_half is not None:
            half = self._gr00t_fsq_half
            z = torch.clamp(torch.round(z * half), -half, half - 1.0) / half
        return z.to(dtype=torch.float32)

    def gr00t_reset(self, env_ids: Tensor | None = None) -> None:
        """Drop cached slots so a reset environment re-plans from its own state."""
        if env_ids is None:
            self._gr00t_cursor.fill_(self._gr00t_slots)
            # Predictions made before a reset describe the pre-reset pose, so
            # they must not be blended into the new episode.
            if getattr(self, "_gr00t_ensemble", "none") != "none":
                self._gr00t_ring_age.zero_()
            return
        selected = env_ids.to(self._gr00t_device).reshape(-1)
        self._gr00t_cursor[selected] = self._gr00t_slots
        if getattr(self, "_gr00t_ensemble", "none") != "none":
            self._gr00t_ring_age[selected] = 0

    def gr00t_assert_goal_matches(
        self, env_ids: Tensor, motion_names: Sequence[str]
    ) -> None:
        """Fail immediately if an environment's motion is not its goal.

        The goal assignment is fixed at setup; the reference channel may
        reassign a trajectory on reset. A silent divergence would evaluate one
        motion while conditioning on another's language, so it is an error.
        """
        env_ids = env_ids.reshape(-1).tolist()
        if len(motion_names) != len(env_ids):
            msg = f"{len(motion_names)} motion names for {len(env_ids)} environments."
            raise ValueError(msg)
        wrong = [
            (env, str(name), self._gr00t_goal_names[env])
            for env, name in zip(env_ids, motion_names, strict=True)
            if str(name) != self._gr00t_goal_names[env]
        ]
        if wrong:
            msg = (
                "goal/reference mismatch (env, live motion, assigned goal): "
                f"{wrong[:5]}"
            )
            raise ValueError(msg)

    def gr00t_stats(self) -> dict[str, Any]:
        latency = self._gr00t_latency_ms[1:]  # exclude the warmup call
        record: dict[str, Any] = {"head_calls": int(self._gr00t_calls)}
        if latency:
            values = torch.tensor(latency)
            record["planner_latency_ms"] = {
                "count": len(latency),
                "p50": float(values.quantile(0.5)),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        else:
            record["planner_latency_ms"] = None
        return record


__all__ = ["Gr00tSkillCommandSampler"]
