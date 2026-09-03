"""Per-episode metrics for the skill-composition probes.

Everything here reads the traces that :class:`~imitation_experiments.evaluation.latent_blend.LatentBlendSampler`
writes into the evaluator summary JSON (``metadata.latent_blend.traces``), so
it runs offline on any machine. Joint indices follow the G1 29-DoF Isaac Lab
order (``left_hip_pitch`` 0, ``right_hip_pitch`` 1, ``left_shoulder_pitch``
11, ``right_shoulder_pitch`` 12).

    python -m imitation_experiments.evaluation.composition_metrics \\
        --results logs/composition_probe --out logs/composition_probe/table.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

CONTROL_DT = 0.02
HIP_PITCH = (0, 1)
SHOULDER_PITCH = (11, 12)
UPRIGHT_FALLEN = 0.5


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def window_mean(values: Sequence[float], lo: int, hi: int) -> float:
    picked = _finite(values[lo:hi])
    return float(picked.mean()) if picked.size else float("nan")


def fallen_steps(upright: Sequence[float], threshold: float = UPRIGHT_FALLEN) -> int:
    return int((_finite(upright) < threshold).sum())


def stride_frequency_hz(
    joint_pos: Sequence[Sequence[float]],
    lo: int,
    hi: int,
    *,
    dt: float = CONTROL_DT,
    indices: tuple[int, int] = HIP_PITCH,
    band: tuple[float, float] = (0.3, 4.0),
) -> tuple[float, float]:
    """Dominant hip-pitch frequency in ``band`` over steps ``[lo, hi)``.

    Returns ``(frequency_hz, band_power_fraction)``; the fraction is the share
    of the (detrended) signal power inside ``band``, so a standing robot
    reads a low fraction and its frequency should be ignored.
    """
    rows = [r for r in joint_pos[lo:hi] if len(r) > max(indices)]
    if len(rows) < 32:
        return float("nan"), float("nan")
    signal = np.asarray([[r[i] for i in indices] for r in rows], dtype=np.float64)
    # Left and right hip pitch are anti-phase in gait; their difference doubles
    # the stride signal and cancels the common posture drift.
    x = signal[:, 0] - signal[:, 1]
    x = x - x.mean()
    x = x - np.polyval(np.polyfit(np.arange(x.size), x, 1), np.arange(x.size))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, d=dt)
    total = float(spectrum[1:].sum())
    if total <= 0.0:
        return float("nan"), 0.0
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float("nan"), 0.0
    band_power = float(spectrum[in_band].sum())
    peak = int(np.argmax(np.where(in_band, spectrum, -1.0)))
    return float(freqs[peak]), band_power / total


def arm_swing_amplitude(
    joint_pos: Sequence[Sequence[float]],
    lo: int,
    hi: int,
    *,
    indices: tuple[int, int] = SHOULDER_PITCH,
) -> float:
    """Mean over both shoulders of the 5th-95th percentile range (rad)."""
    rows = [r for r in joint_pos[lo:hi] if len(r) > max(indices)]
    if len(rows) < 8:
        return float("nan")
    arr = np.asarray([[r[i] for i in indices] for r in rows], dtype=np.float64)
    ranges = np.percentile(arr, 95, axis=0) - np.percentile(arr, 5, axis=0)
    return float(ranges.mean())


def settling_time_steps(
    target_speed: Sequence[float],
    source_speed_ref: float,
    start: int,
    *,
    tolerance: float = 0.15,
    hold_steps: int = 25,
) -> int | None:
    """First step at or after ``start`` from which the target's speed stays
    within ``tolerance`` (relative, floored at 0.1 m/s) of the source's
    reference speed for ``hold_steps`` consecutive steps; ``None`` if never."""
    if not math.isfinite(source_speed_ref):
        return None
    tol = max(0.1, tolerance * abs(source_speed_ref))
    arr = np.asarray(target_speed, dtype=np.float64)
    ok = np.isfinite(arr) & (np.abs(arr - source_speed_ref) <= tol)
    run = 0
    for step in range(start, arr.size):
        run = run + 1 if ok[step] else 0
        if run >= hold_steps:
            return step - hold_steps + 1 - start
    return None


def peak_action_delta(
    action_delta: Sequence[float], start: int, window: int = 50
) -> float:
    picked = _finite(action_delta[start : start + window])
    return float(picked.max()) if picked.size else float("nan")


def joint_gait_distance(
    target_joints: Sequence[Sequence[float]],
    source_joints: Sequence[Sequence[float]],
    lo: int,
    hi: int,
) -> float:
    """Mean absolute joint difference (rad) between target and source robots
    over the same steps -- a same-time-index gait match."""
    pairs = [
        (t, s)
        for t, s in zip(target_joints[lo:hi], source_joints[lo:hi])
        if len(t) and len(s) and len(t) == len(s)
    ]
    if not pairs:
        return float("nan")
    t = np.asarray([p[0] for p in pairs], dtype=np.float64)
    s = np.asarray([p[1] for p in pairs], dtype=np.float64)
    return float(np.abs(t - s).mean())


def episode_metrics(
    target: dict[str, Any],
    source: dict[str, Any],
    *,
    start_step: int,
    ramp_steps: int,
    steps: int | None = None,
) -> dict[str, Any]:
    """All per-episode numbers for one (target, source) trace pair.

    Windows: ``pre`` is the 100 steps before ``start_step`` (or from 0),
    ``ramp`` is the ramp, ``post`` runs from the ramp end to the last step.
    A held mix (``start_step=0``, ``ramp_steps=0``) has only ``post``.
    """
    n = steps if steps is not None else len(target["root_speed"])
    pre_lo, pre_hi = max(0, start_step - 100), start_step
    ramp_lo, ramp_hi = start_step, start_step + ramp_steps
    post_lo, post_hi = start_step + ramp_steps, n
    src_post_speed = window_mean(source["root_speed"], post_lo, post_hi)
    tgt_freq, tgt_band = stride_frequency_hz(target["joint_pos"], post_lo, post_hi)
    src_freq, src_band = stride_frequency_hz(source["joint_pos"], post_lo, post_hi)
    return {
        "fallen_steps": fallen_steps(target["upright"]),
        "fall_free": fallen_steps(target["upright"]) == 0,
        "source_fallen_steps": fallen_steps(source["upright"]),
        "upright_min": float(_finite(target["upright"]).min())
        if _finite(target["upright"]).size
        else float("nan"),
        "speed_pre": window_mean(target["root_speed"], pre_lo, pre_hi),
        "speed_ramp": window_mean(target["root_speed"], ramp_lo, ramp_hi),
        "speed_post": window_mean(target["root_speed"], post_lo, post_hi),
        "source_speed_post": src_post_speed,
        "stride_hz_post": tgt_freq,
        "stride_band_fraction_post": tgt_band,
        "source_stride_hz_post": src_freq,
        "source_stride_band_fraction_post": src_band,
        "arm_swing_post": arm_swing_amplitude(target["joint_pos"], post_lo, post_hi),
        "source_arm_swing_post": arm_swing_amplitude(
            source["joint_pos"], post_lo, post_hi
        ),
        "action_delta_pre": window_mean(target["action_delta"], pre_lo, pre_hi),
        "action_delta_ramp": window_mean(target["action_delta"], ramp_lo, ramp_hi),
        "action_delta_post": window_mean(target["action_delta"], post_lo, post_hi),
        "peak_action_delta_after_switch": peak_action_delta(
            target["action_delta"], start_step
        ),
        "settling_steps": settling_time_steps(
            target["root_speed"], src_post_speed, start_step
        ),
        "joint_gait_distance_post": joint_gait_distance(
            target["joint_pos"], source["joint_pos"], post_lo, post_hi
        ),
        "code_distance_pre": window_mean(target["code_distance"], pre_lo, pre_hi),
        "code_distance_post": window_mean(target["code_distance"], post_lo, post_hi),
    }


def monotone_fraction(values_by_alpha: Sequence[tuple[float, float]]) -> float | None:
    """Share of consecutive alpha steps along which ``value`` moves toward
    the ``alpha=max`` end, i.e. ``sign(v[i+1] - v[i]) == sign(v_last - v_first)``.
    ``None`` when fewer than three finite points or no net change."""
    pts = sorted((a, v) for a, v in values_by_alpha if math.isfinite(v))
    if len(pts) < 3:
        return None
    net = pts[-1][1] - pts[0][1]
    if abs(net) < 1e-9:
        return None
    direction = 1.0 if net > 0 else -1.0
    steps = [(b[1] - a[1]) * direction >= 0.0 for a, b in zip(pts, pts[1:])]
    return sum(steps) / len(steps)


def summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    """Per-target rows from one evaluator summary JSON with a blend block."""
    payload = json.loads(Path(summary_path).read_text())
    meta = payload.get("metadata") or {}
    blend = meta.get("latent_blend")
    if not blend or "traces" not in blend:
        return []
    schedule = blend["schedule"]
    traces = blend["traces"]
    ranks = {
        int(e["env_id"]): int(e["trajectory_rank"])
        for e in payload.get("per_environment", [])
    }
    names = {
        int(e["env_id"]): e.get("motion_name")
        for e in payload.get("per_environment", [])
    }
    rows = []
    for spec in blend["specs"]:
        t, s = str(spec["target"]), str(spec["source"])
        if t not in traces or s not in traces:
            continue
        metrics = episode_metrics(
            traces[t],
            traces[s],
            start_step=int(schedule["start_step"]),
            ramp_steps=int(schedule["ramp_steps"]),
        )
        rows.append(
            {
                "summary": str(summary_path),
                "label": meta.get("label"),
                "target_env": spec["target"],
                "source_env": spec["source"],
                "minus_env": spec.get("minus"),
                "target_rank": ranks.get(spec["target"]),
                "source_rank": ranks.get(spec["source"]),
                "target_name": names.get(spec["target"]),
                "source_name": names.get(spec["source"]),
                "start_step": schedule["start_step"],
                "ramp_steps": schedule["ramp_steps"],
                "final_alpha": schedule["final_alpha"],
                **metrics,
            }
        )
    return rows


def aggregate(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Group rows by ``keys`` and report count, fall-free rate, and the means
    of the numeric metrics."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    numeric = [
        "speed_pre",
        "speed_ramp",
        "speed_post",
        "source_speed_post",
        "stride_hz_post",
        "source_stride_hz_post",
        "arm_swing_post",
        "source_arm_swing_post",
        "action_delta_pre",
        "action_delta_ramp",
        "action_delta_post",
        "peak_action_delta_after_switch",
        "joint_gait_distance_post",
        "code_distance_post",
    ]
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        agg: dict[str, Any] = {k: v for k, v in zip(keys, key)}
        agg["n"] = len(members)
        agg["fall_free_rate"] = sum(1 for m in members if m["fall_free"]) / len(members)
        settled = [
            m["settling_steps"] for m in members if m["settling_steps"] is not None
        ]
        agg["settled_rate"] = len(settled) / len(members)
        agg["settling_steps_median"] = float(np.median(settled)) if settled else None
        for metric in numeric:
            vals = _finite(m[metric] for m in members)
            agg[metric] = float(vals.mean()) if vals.size else None
        out.append(agg)
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


def markdown_table(aggregates: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| " + " | ".join(_fmt(a.get(c)) for c in columns) + " |\n" for a in aggregates
    )
    return head + body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, required=True, help="directory tree of summary JSONs"
    )
    parser.add_argument("--out", type=Path, default=None, help="markdown table path")
    parser.add_argument(
        "--rows-out", type=Path, default=None, help="per-episode rows as JSON"
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["label", "final_alpha", "ramp_steps", "start_step"],
    )
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.results).rglob("*.json")):
        try:
            rows.extend(summary_rows(path))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    aggregates = aggregate(rows, args.group_by)
    columns = list(args.group_by) + [
        "n",
        "fall_free_rate",
        "settled_rate",
        "settling_steps_median",
        "speed_pre",
        "speed_post",
        "source_speed_post",
        "stride_hz_post",
        "source_stride_hz_post",
        "arm_swing_post",
        "action_delta_post",
        "peak_action_delta_after_switch",
        "joint_gait_distance_post",
        "code_distance_post",
    ]
    table = markdown_table(aggregates, columns)
    print(table)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)
    if args.rows_out is not None:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        args.rows_out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
