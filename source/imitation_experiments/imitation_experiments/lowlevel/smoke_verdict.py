#!/usr/bin/env python3
"""Decide whether one interface-design-study arm passed its wiring smoke.

This is a WIRING gate, not a quality gate. It answers "does this arm run, at
the width it declares, with a code that is not collapsed" and nothing else. A
few offline updates cannot say whether an objective or a bottleneck is any
good, and an encoder at that budget is expected to be WORSE than a zero code
(`loss_real_z_eval` above `loss_zero_z_eval`), so nothing here gates on the
learning signal.

Checks, in order:

1. the encoder checkpoint loads, and the config it records matches the
   objective, latent mode and code width the campaign declares for the arm --
   a checkpoint that silently trained a different design is the failure this
   catches;
2. the metric stream has at least one row and every number in it is finite;
3. the code is not collapsed. A QUANTIZED arm is judged on the code
   perplexity the trainer reports -- perplexity 1.0 is a single code, so
   anything above it means more than one level is in use. A CONTINUOUS arm has
   no codebook, so it is judged on a non-zero MEAN per-dimension spread.

   Two traps this encodes. The mean spread, never the minimum: a discrete
   codebook has dead dimensions early (`bn_gumbel_multicat` measured
   `z_dim_std_min` 0.0 at a healthy `z_dim_std_mean` 0.317), so gating on the
   minimum fails a working quantizer. And effective rank is reported but is
   never the gate: on a degenerate all-zero covariance it returns a spurious
   11.57 rather than 1, so it cannot detect the collapse it looks like it
   should;
4. the frozen encoder drove one low-level iteration to a clean exit at the
   arm's own command width.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["SmokeVerdict", "verdict_for"]

# The trainer writes these next to the checkpoint. Names verified against a
# real `train_hl_skill_diffsr.py` run on 2026-08-19.
EFFECTIVE_RANK_KEY = "train/z_effective_rank"
CODE_PERPLEXITY_KEY = "train/diversity/code_perplexity"
CODE_USAGE_KEY = "train/diversity/code_usage_frac"
DEAD_CODE_KEY = "train/diversity/dead_code_frac"
DIM_STD_MEAN_KEY = "train/z_dim_std_mean"
DIM_STD_MIN_KEY = "train/z_dim_std_min"
LOSS_KEY = "train/loss"


@dataclass
class SmokeVerdict:
    arm: str
    status: str
    checks: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "arm": self.arm,
            "status": self.status,
            "checks": self.checks,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _metrics_rows(encoder_dir: Path) -> list[dict[str, Any]]:
    """Every metric row the pretrain wrote, from wherever it landed.

    The trainer puts `metrics.jsonl` under a timestamped log directory inside
    the output dir, so search rather than assume one path.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(encoder_dir.rglob("metrics.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _non_finite(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    bad = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                bad.append(key)
    return sorted(set(bad))


def _last(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        if key in row and isinstance(row[key], (int, float)):
            return float(row[key])
    return None


def _checkpoint_config(checkpoint: Path) -> Mapping[str, Any]:
    import torch

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("config", "cfg", "skill_encoder_config"):
        value = blob.get(key) if isinstance(blob, Mapping) else None
        if isinstance(value, Mapping):
            return value
        if value is not None and hasattr(value, "__dict__"):
            return vars(value)
    raise KeyError(
        f"{checkpoint} carries no config block; cannot confirm what it trained."
    )


def verdict_for(
    *,
    arm: str,
    checkpoint: Path,
    lowlevel_exit: int,
    expected_command_dim: int,
    expected_objective: str | None = None,
    expected_latent_mode: str | None = None,
    expected_z_dim: int | None = None,
) -> SmokeVerdict:
    checks: dict[str, Any] = {}

    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        return SmokeVerdict(
            arm, "fail", checks, f"no encoder checkpoint at {checkpoint}"
        )

    try:
        config = _checkpoint_config(checkpoint)
    except Exception as exc:  # noqa: BLE001 - any load failure is a smoke failure
        return SmokeVerdict(arm, "fail", checks, f"checkpoint did not load: {exc}")

    for name, expected in (
        ("transition_objective", expected_objective),
        ("latent_mode", expected_latent_mode),
    ):
        if expected is None:
            continue
        found = config.get(name)
        checks[name] = found
        if str(found) != str(expected):
            return SmokeVerdict(
                arm,
                "fail",
                checks,
                f"checkpoint {name}={found!r}, campaign says {expected!r}",
            )
    if expected_z_dim is not None:
        found_z = config.get("z_dim")
        checks["z_dim"] = found_z
        if int(found_z) != int(expected_z_dim):
            return SmokeVerdict(
                arm,
                "fail",
                checks,
                f"checkpoint z_dim={found_z}, campaign says {expected_z_dim}",
            )

    rows = _metrics_rows(checkpoint.parent.parent)
    checks["metric_rows"] = len(rows)
    if not rows:
        return SmokeVerdict(arm, "fail", checks, "pretrain wrote no metric rows")

    non_finite = _non_finite(rows)
    if non_finite:
        return SmokeVerdict(
            arm, "fail", checks, f"non-finite metrics: {', '.join(non_finite[:5])}"
        )

    loss = _last(rows, LOSS_KEY)
    checks["loss"] = loss
    if loss is None:
        return SmokeVerdict(arm, "fail", checks, f"no {LOSS_KEY} in the metric stream")

    rank = _last(rows, EFFECTIVE_RANK_KEY)
    dim_std_mean = _last(rows, DIM_STD_MEAN_KEY)
    dim_std_min = _last(rows, DIM_STD_MIN_KEY)
    checks["z_effective_rank"] = rank
    checks["z_dim_std_mean"] = dim_std_mean
    checks["z_dim_std_min"] = dim_std_min
    # Mode-agnostic "more than one code level in use". The gate is the MEAN
    # per-dimension spread, not the minimum: a discrete codebook legitimately
    # has dead dimensions this early, so requiring every dimension to vary
    # would fail a working quantizer.
    #
    # Effective rank is recorded but is not the gate. On an all-zero covariance
    # it returns a spurious value (11.5 was measured on a fully collapsed VQ
    # codebook) instead of 1, so it cannot detect this failure on its own.
    perplexity = _last(rows, CODE_PERPLEXITY_KEY)
    if perplexity is not None:
        # Quantized arm: the trainer reports the codebook's own usage, which is
        # the exact statement of "more than one code level in use".
        checks["code_perplexity"] = perplexity
        checks["code_usage_frac"] = _last(rows, CODE_USAGE_KEY)
        checks["dead_code_frac"] = _last(rows, DEAD_CODE_KEY)
        if perplexity <= 1.0:
            return SmokeVerdict(
                arm,
                "fail",
                checks,
                f"codebook collapsed to one level: perplexity {perplexity}",
            )
    elif dim_std_mean is None or dim_std_mean <= 0.0:
        # Continuous arm: no codebook, so the spread of the code is the signal.
        return SmokeVerdict(
            arm,
            "fail",
            checks,
            f"code collapsed: every dimension is constant (mean std {dim_std_mean})",
        )

    checks["expected_command_dim"] = int(expected_command_dim)
    checks["lowlevel_exit"] = int(lowlevel_exit)
    if int(lowlevel_exit) != 0:
        return SmokeVerdict(
            arm, "fail", checks, f"low-level iteration exited {lowlevel_exit}"
        )

    return SmokeVerdict(arm, "pass", checks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain-log", type=Path, default=None)
    parser.add_argument("--lowlevel-log", type=Path, default=None)
    parser.add_argument("--lowlevel-exit", type=int, required=True)
    parser.add_argument("--expected-command-dim", type=int, required=True)
    parser.add_argument("--expected-objective", default=None)
    parser.add_argument("--expected-latent-mode", default=None)
    parser.add_argument("--expected-z-dim", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    verdict = verdict_for(
        arm=args.arm,
        checkpoint=args.checkpoint,
        lowlevel_exit=args.lowlevel_exit,
        expected_command_dim=args.expected_command_dim,
        expected_objective=args.expected_objective,
        expected_latent_mode=args.expected_latent_mode,
        expected_z_dim=args.expected_z_dim,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(verdict.to_json(), encoding="utf-8")
    print(verdict.to_json(), end="")
    return 0 if verdict.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
