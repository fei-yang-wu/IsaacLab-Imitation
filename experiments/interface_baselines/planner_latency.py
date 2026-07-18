"""Measure root planner forward latency during closed-loop publication."""

from __future__ import annotations

from contextlib import contextmanager
import math
from statistics import fmean, pstdev
import time
from typing import Iterator

import torch


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class PlannerForwardTimer:
    """Time only forwards through one root planner module."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self._enabled = False
        self._start: float | None = None
        self._latencies_ms: list[float] = []
        self._batch_sizes: list[int] = []
        parameter = next(module.parameters(), None)
        self._uses_cuda = parameter is not None and parameter.device.type == "cuda"
        self._pre_handle = module.register_forward_pre_hook(self._pre_forward)
        self._post_handle = module.register_forward_hook(self._post_forward)

    def _synchronize(self) -> None:
        parameter = next(self.module.parameters(), None)
        if parameter is not None and parameter.device.type == "cuda":
            torch.cuda.synchronize(parameter.device)

    def _pre_forward(
        self, _module: torch.nn.Module, inputs: tuple[object, ...]
    ) -> None:
        if not self._enabled:
            return
        self._synchronize()
        self._start = time.perf_counter()
        batch_size = -1
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                batch_size = int(value.shape[0])
                break
        self._batch_sizes.append(batch_size)

    def _post_forward(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        _output: object,
    ) -> None:
        if not self._enabled or self._start is None:
            return
        self._synchronize()
        self._latencies_ms.append((time.perf_counter() - self._start) * 1000.0)
        self._start = None

    @contextmanager
    def enabled(self) -> Iterator[None]:
        previous = self._enabled
        self._enabled = True
        try:
            yield
        finally:
            self._enabled = previous
            self._start = None

    def summary(self, *, warmup_calls: int = 1) -> dict[str, object]:
        skipped = min(max(int(warmup_calls), 0), len(self._latencies_ms))
        measured = self._latencies_ms[skipped:]
        return {
            "unit": "ms",
            "scope": "high_level_planner_forward_only",
            "synchronized_cuda": self._uses_cuda,
            "total_call_count": len(self._latencies_ms),
            "warmup_calls_excluded": skipped,
            "measured_call_count": len(measured),
            "batch_sizes": sorted(set(self._batch_sizes)),
            "mean": fmean(measured) if measured else math.nan,
            "std": pstdev(measured) if measured else math.nan,
            "p50": _percentile(measured, 0.5) if measured else math.nan,
            "p95": _percentile(measured, 0.95) if measured else math.nan,
        }

    def close(self) -> None:
        self._pre_handle.remove()
        self._post_handle.remove()
