"""Attribute terminations to the interval push during evaluation.

The push is an explicit interval event, so attribution does not need a
counterfactual: the tracker wraps the ``push_robot`` event function, records
the last push step per environment, and reports the control-step distance
between that push and any later termination. A termination with no push since
the environment's last reset carries distance ``None`` and lands in the
``no_push_seen`` bucket.

Torch-only on purpose: the wrapping entry point receives the live event
manager, but every accounting path is testable without Isaac.
"""

from __future__ import annotations

from typing import Any

import torch

# Fixed histogram edges in control steps (50 Hz -> 0.5 s per 25 steps). The
# open tail bucket catches everything past the last edge.
HISTOGRAM_EDGES: tuple[int, ...] = (5, 10, 25, 50, 100, 250)

# Terms that end an episode without being a failure. They stay visible in the
# per-term breakdown but are excluded from the overall distances, histogram,
# and ``frac_within``: attributing a completed reference to a push is noise.
NON_FAILURE_TERMS: frozenset[str] = frozenset({"reference_finished", "time_out"})


class PushAttributionTracker:
    """Track push-to-termination distances for one evaluation run."""

    def __init__(self, num_envs: int) -> None:
        self._num_envs = int(num_envs)
        self._last_push_step = torch.full((self._num_envs,), -1, dtype=torch.long)
        self.push_events = 0
        # distances in control steps, one entry per attributed termination
        self.distances: list[int] = []
        self.per_term_distances: dict[str, list[int]] = {}
        self.no_push_seen = 0

    # -- recording ---------------------------------------------------------
    def record_push(self, env_ids: Any, step: int) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long).reshape(-1).cpu()
        self._last_push_step[ids] = int(step)
        self.push_events += int(ids.numel())

    def on_terminal(
        self,
        terminated_mask: torch.Tensor,
        done_mask: torch.Tensor,
        step: int,
        term_masks: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Account terminated envs, then clear pushes for every done env.

        ``done_mask`` must include truncations: after any reset the previous
        push belongs to a finished episode and must not attribute a later
        termination.
        """
        terminated = terminated_mask.reshape(-1).cpu().bool()
        term_ids = torch.nonzero(terminated, as_tuple=False).reshape(-1)
        for env_id in term_ids.tolist():
            last = int(self._last_push_step[env_id].item())
            if last < 0:
                self.no_push_seen += 1
                continue
            distance = int(step) - last
            hit_terms = [
                name
                for name, mask in (term_masks or {}).items()
                if bool(mask.reshape(-1).cpu()[env_id])
            ]
            is_failure = not term_masks or any(
                name not in NON_FAILURE_TERMS for name in hit_terms
            )
            if is_failure:
                self.distances.append(distance)
            for name in hit_terms:
                self.per_term_distances.setdefault(name, []).append(distance)
        done_ids = torch.nonzero(
            done_mask.reshape(-1).cpu().bool(), as_tuple=False
        ).reshape(-1)
        self._last_push_step[done_ids] = -1

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "push_events": int(self.push_events),
            "terminations_with_push": len(self.distances),
            "terminations_no_push_seen": int(self.no_push_seen),
            "steps_since_push_histogram": histogram_since_push(self.distances),
            "frac_within": {
                str(edge): _frac_within(self.distances, edge) for edge in (10, 25, 50)
            },
            "per_term": {
                name: {
                    "count": len(values),
                    "median_steps": float(torch.tensor(values).float().median()),
                    "histogram": histogram_since_push(values),
                }
                for name, values in sorted(self.per_term_distances.items())
            },
        }


def histogram_since_push(
    distances: list[int], edges: tuple[int, ...] = HISTOGRAM_EDGES
) -> dict[str, int]:
    """Bucket distances into ``<=edge`` bins plus an open tail bucket."""
    counts = {f"<={edge}": 0 for edge in edges}
    counts[f">{edges[-1]}"] = 0
    for value in distances:
        for edge in edges:
            if value <= edge:
                counts[f"<={edge}"] += 1
                break
        else:
            counts[f">{edges[-1]}"] += 1
    return counts


def _frac_within(distances: list[int], edge: int) -> float:
    if not distances:
        return float("nan")
    return sum(1 for value in distances if value <= edge) / len(distances)


def attach_push_tracker(
    event_manager: Any, num_envs: int
) -> PushAttributionTracker | None:
    """Wrap the ``push_robot`` event term so pushes are recorded.

    Returns ``None`` when the manager carries no live ``push_robot`` term
    (every profile except ``all``). The wrapper preserves the original
    function and signature; the event fires exactly as before.
    """
    try:
        term_cfg = event_manager.get_term_cfg("push_robot")
    except (AttributeError, KeyError, ValueError):
        return None
    if term_cfg is None or term_cfg.func is None:
        return None
    tracker = PushAttributionTracker(num_envs)
    original = term_cfg.func

    def _recording_push(env: Any, env_ids: Any, **kwargs: Any):
        if env_ids is None:
            ids: Any = torch.arange(num_envs)
        else:
            ids = env_ids
        tracker.record_push(ids, int(getattr(env, "common_step_counter", 0)))
        return original(env, env_ids, **kwargs)

    term_cfg.func = _recording_push
    return tracker
