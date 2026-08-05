#!/usr/bin/env python3
"""Aggregate a local RLOpt hyperparameter screen into a comparison table.

This is a screening aggregator, not a paper-result aggregator. A screen arm is a
short fixed-frame block (50M by default) that is far too early to report as a
result: the reference 12288x12 run needs roughly 1200 iterations to reach an
episode length of 400, and a 50M block at 12288x24 is 170. What a screen can
resolve is the shape of the *early* learning curve and whether the optimizer is
behaving -- which is exactly where the adaptive-KL learning-rate rule and the
entropy bonus act.

Every arm is required to have run the same frame budget and geometry. That is
the whole point of a screen, and silently ranking arms that saw different
amounts of data is the failure mode most likely to produce a confident wrong
answer, so a mismatch is an error rather than a warning.

Input is the screen directory the launcher builds: one subdirectory per arm,
each holding an ``arm.json`` written by the launcher and the ``scalars/`` tree
written by RLOpt's CSV metrics backend. Nothing here contacts W&B, so a screen
can be re-aggregated offline from the run directory alone.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


# Reported per arm, in the order the table shows them. The learning columns come
# first because they answer "did this arm learn faster"; the optimizer-health
# columns follow because they answer "and was the optimizer sane while it did".
PROGRESS_METRICS = (
    "episode/length",
    "episode/return",
    "train/step_reward_mean",
    "mpjpe_mm",
)

# The v2 command interface reports MPJPE under the reference command term; the
# v1/strict surface reported it flat. Accept either so a screen can compare an
# arm across surfaces, and so a surface rename does not silently blank the
# column. First match wins.
METRIC_ALIASES = {
    # `mpjpe_l_mm` is the current name; `mpjpe_mm` is the pre-2026-08-04 alias,
    # kept so older runs still resolve. Note runs before that date logged the
    # error at the terminal step rather than an episode mean, so a curve that
    # spans the change is not comparable.
    "mpjpe_mm": (
        "Metrics/reference/mpjpe_l_mm",
        "Metrics/reference/mpjpe_mm",
        "Metrics/mpjpe_mm",
    ),
}
OPTIMIZER_METRICS = (
    "train/lr",
    "train/kl_approx",
    "train/entropy",
    "train/clip_fraction",
    "train/explained_variance",
    "train/grad_norm",
)

# Fields every arm must agree on for the comparison to mean anything: the data
# budget. Deliberately NOT rollout_steps -- an arm that batches the same 50M
# frames into 170 iterations of 294,912 instead of 340 of 147,456 saw exactly
# the same amount of data, and comparing those two is a question worth asking
# rather than a mistake to block. It is reported as a column instead.
GEOMETRY_KEYS = ("num_envs", "total_frames")

# Rows pulled per W&B run. A screen arm logs one row per iteration (340 at the
# default geometry), so this returns the curve in full with a wide margin.
_WANDB_HISTORY_SAMPLES = 4000


class ScreenError(RuntimeError):
    """A screen could not be aggregated as specified."""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--screen_root",
        type=Path,
        help="Directory holding one subdirectory per arm, each with summary.json. "
        "Use this for a local screen run with the CSV metrics backend.",
    )
    source.add_argument(
        "--wandb_group",
        help="W&B group holding one run per arm. Use this for a cluster screen, "
        "where arms run concurrently on separate nodes and log to W&B rather "
        "than to a shared filesystem.",
    )
    parser.add_argument(
        "--wandb_project",
        default="g1-lafan1",
        help="W&B project searched for --wandb_group.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B entity. Defaults to the API's default entity.",
    )
    parser.add_argument(
        "--wandb_arm_prefix",
        default="",
        help="Strip this prefix from a run's exp_name to recover the arm name. "
        "The cluster launcher names runs '<screen_tag>_<arm>'.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Markdown table to write. Refuses to overwrite an existing file.",
    )
    parser.add_argument(
        "--baseline",
        default="b0_baseline",
        help="Arm name that other arms are reported as a delta against.",
    )
    parser.add_argument(
        "--tail_fraction",
        type=float,
        default=0.2,
        help="Trailing fraction of each curve averaged into the reported score. "
        "Averaging a tail rather than reading the last point keeps the ranking "
        "from turning on single-iteration noise.",
    )
    return parser.parse_args(argv)


def read_scalar_csv(path: Path) -> list[tuple[int, float]]:
    """Read one ``step,value`` CSV written by RLOpt's CSV metrics backend.

    Malformed lines are skipped rather than fatal: the file is appended to and
    flushed per scalar during training, so a run killed mid-write can leave a
    torn final line, and that is not a reason to discard the whole arm.
    """
    points: list[tuple[int, float]] = []
    for line in path.read_text().splitlines():
        step_text, _, value_text = line.partition(",")
        try:
            value = float(value_text)
        except ValueError:
            continue
        try:
            step = int(float(step_text))
        except ValueError:
            continue
        if math.isfinite(value):
            points.append((step, value))
    return points


def load_scalar_history(arm_dir: Path) -> list[dict[str, Any]]:
    """Merge an arm's per-metric CSVs into one step-indexed history.

    The CSV backend writes a separate file per metric, all keyed by the same
    global step, so the merge is a join on step rather than an interleave.
    """
    scalar_roots = sorted(arm_dir.glob("**/scalars"))
    if not scalar_roots:
        raise ScreenError(f"{arm_dir}: no scalars/ directory; did the arm run?")
    by_step: dict[int, dict[str, Any]] = {}
    for scalar_root in scalar_roots:
        for csv_path in sorted(scalar_root.glob("**/*.csv")):
            metric = csv_path.relative_to(scalar_root).with_suffix("").as_posix()
            for step, value in read_scalar_csv(csv_path):
                by_step.setdefault(step, {"step": step})[metric] = value
    if not by_step:
        raise ScreenError(f"{arm_dir}: scalars/ held no readable points")
    return [by_step[step] for step in sorted(by_step)]


def load_arm(arm_dir: Path) -> dict[str, Any]:
    """Read one arm's metadata and metric history, failing loudly if malformed."""
    metadata_path = arm_dir / "arm.json"
    if not metadata_path.is_file():
        raise ScreenError(f"{metadata_path}: missing; the launcher writes this")
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise ScreenError(f"{metadata_path}: not valid JSON ({exc})") from exc
    for required in ("arm", "geometry"):
        if required not in payload:
            raise ScreenError(f"{metadata_path}: missing '{required}'")
    payload["history"] = load_scalar_history(arm_dir)
    return payload


def discover_arms(screen_root: Path) -> list[dict[str, Any]]:
    """Load every arm under ``screen_root``, sorted by arm name."""
    if not screen_root.is_dir():
        raise ScreenError(f"{screen_root}: not a directory")
    arm_dirs = sorted(p.parent for p in screen_root.glob("*/arm.json"))
    if not arm_dirs:
        raise ScreenError(f"{screen_root}: no */arm.json found")
    arms = [load_arm(path) for path in arm_dirs]
    arms.sort(key=lambda arm: str(arm["arm"]))
    return arms


def arm_record_from_wandb_run(run: Any, arm_prefix: str = "") -> dict[str, Any]:
    """Build the same arm record ``load_arm`` produces, from a W&B run.

    Kept deliberately close to the on-disk shape so everything downstream --
    geometry matching, tail scoring, the table -- is shared between a local and
    a cluster screen rather than reimplemented per source.
    """
    name = str(run.config.get("logger", {}).get("exp_name") or run.name)
    if arm_prefix and name.startswith(arm_prefix):
        name = name[len(arm_prefix) :]

    config = run.config
    collector = config.get("collector", {}) or {}
    env_cfg = config.get("env", {}) or {}
    num_envs = int(env_cfg.get("num_envs", 0))
    frames_per_batch = int(collector.get("frames_per_batch", 0))
    # frames_per_batch is the whole-iteration count; the rollout length is what
    # the launcher varied, and it is that ratio the table reports.
    rollout_steps = frames_per_batch // num_envs if num_envs else 0

    # `history(samples=...)`, not `scan_history()`: scan streams every logged row
    # over the network and a whole screen's worth of runs does not finish in a
    # usable time. A screen arm logs one row per iteration -- 340 at the default
    # geometry -- so a 4000-sample ceiling returns the curve in full and only
    # subsamples a run far longer than a screen, where the tail means the scoring
    # uses are unaffected anyway.
    history = [
        {
            ("step" if key == "_step" else key): value
            for key, value in row.items()
            if value is not None
        }
        for row in run.history(samples=_WANDB_HISTORY_SAMPLES, pandas=False)
    ]
    if not history:
        raise ScreenError(f"{run.name}: W&B run has no logged history")

    # total_frames is what the arm actually saw, not what it was asked to run:
    # a screen arm that died early must not be silently compared as if it had
    # completed. require_matched_geometry then rejects it.
    total_frames = max(int(row.get("step", 0)) for row in history)

    return {
        "arm": name,
        "description": "",
        "overrides": [],
        "geometry": {
            "num_envs": num_envs,
            "rollout_steps": rollout_steps,
            "frames_per_batch": frames_per_batch,
            "total_frames": total_frames,
            "seed": int(config.get("seed", 0)),
        },
        "wandb_run": run.id,
        "wandb_state": run.state,
        "history": history,
    }


def discover_arms_from_wandb(
    project: str,
    group: str,
    entity: str | None = None,
    arm_prefix: str = "",
) -> list[dict[str, Any]]:
    """Load every arm in a W&B group, sorted by arm name."""
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ScreenError(
            "reading a screen from W&B needs the wandb package installed"
        ) from exc

    api = wandb.Api()
    entity = entity or api.default_entity
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}))
    if not runs:
        raise ScreenError(f"{entity}/{project}: no runs in group '{group}'")
    arms = [arm_record_from_wandb_run(run, arm_prefix) for run in runs]
    arms.sort(key=lambda arm: str(arm["arm"]))
    return arms


def require_matched_geometry(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject a screen whose arms did not see the same amount of data."""
    reference = arms[0]
    ref_geom = {key: reference["geometry"].get(key) for key in GEOMETRY_KEYS}
    for arm in arms[1:]:
        geom = {key: arm["geometry"].get(key) for key in GEOMETRY_KEYS}
        if geom != ref_geom:
            raise ScreenError(
                "Arms did not run a matched screen and cannot be ranked: "
                f"{reference['arm']} has {ref_geom}, {arm['arm']} has {geom}."
            )
    return ref_geom


def tail_mean(history: list[dict[str, Any]], key: str, tail_fraction: float) -> float:
    """Mean of ``key`` over the trailing fraction of the curve.

    Points where the metric is absent or non-finite are skipped; a metric absent
    from the whole tail reports NaN rather than raising, because arms
    legitimately differ in which optional metrics they log.
    """
    if not 0.0 < tail_fraction <= 1.0:
        raise ScreenError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    count = max(1, round(len(history) * tail_fraction))
    candidates = METRIC_ALIASES.get(key, (key,))
    values = []
    for point in history[-count:]:
        for candidate in candidates:
            value = point.get(candidate)
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
                break
    if not values:
        return math.nan
    return statistics.fmean(values)


def wall_clock_scores(
    history: list[dict[str, Any]], tail_fraction: float
) -> dict[str, float]:
    """Progress per minute of training wall-clock.

    The screen's objective is quality per unit time, not per sample: a knob that
    buys nothing but costs gradient work is a loss even when its per-frame curve
    looks flat, and reward-per-step is a density that hides both.

    Time is ``time/collecting + time/training`` summed over iterations, which is
    the training loop only -- it excludes the ~7 min Isaac startup that every arm
    pays identically and would otherwise dilute every rate by a constant.

    Note these rates are not a pure speed measurement. Collection cost falls as
    episodes lengthen (fewer resets), so an arm that learns better also runs
    faster and the time axis compounds the two. That is the intended reading:
    it is what a GPU-hour actually buys.
    """
    per_iteration = [
        float(point["time/collecting"]) + float(point["time/training"])
        for point in history
        if isinstance(point.get("time/collecting"), (int, float))
        and isinstance(point.get("time/training"), (int, float))
        and math.isfinite(point.get("time/collecting", math.nan))
        and math.isfinite(point.get("time/training", math.nan))
    ]
    minutes = sum(per_iteration) / 60.0
    if not per_iteration or minutes <= 0.0:
        return {
            "train_minutes": math.nan,
            "return_per_min": math.nan,
            "ep_len_per_min": math.nan,
        }
    # Tail means, not final points: per-iteration return is noisy enough at this
    # scale that an endpoint misranks arms (b0's peak was 10.94 against a final
    # 8.85 in the 2026-08-02 screen -- a 24% swing on one sample).
    return {
        "train_minutes": minutes,
        "return_per_min": tail_mean(history, "episode/return", tail_fraction)
        / minutes,
        "ep_len_per_min": tail_mean(history, "episode/length", tail_fraction)
        / minutes,
    }


def score_arms(
    arms: list[dict[str, Any]], tail_fraction: float
) -> list[dict[str, Any]]:
    """Reduce each arm's curve to the reported scalar scores."""
    scored = []
    for arm in arms:
        history = arm["history"]
        scores = {
            key: tail_mean(history, key, tail_fraction)
            for key in (*PROGRESS_METRICS, *OPTIMIZER_METRICS)
        }
        # The LR is the diagnostic that motivated this screen, so also report how
        # much it moved. A healthy controller settles; the per-minibatch rule
        # produced a serially uncorrelated log-LR over a 3.5B-frame run.
        lrs = [
            float(point["train/lr"])
            for point in history
            if isinstance(point.get("train/lr"), (int, float))
            and math.isfinite(point.get("train/lr", math.nan))
            and point.get("train/lr", 0.0) > 0.0
        ]
        scores["lr_geomean"] = (
            math.exp(statistics.fmean([math.log(v) for v in lrs])) if lrs else math.nan
        )
        scores["lr_spread"] = (max(lrs) / min(lrs)) if lrs else math.nan
        scores.update(wall_clock_scores(history, tail_fraction))
        scored.append(
            {
                "arm": arm["arm"],
                "description": arm.get("description", ""),
                "overrides": arm.get("overrides", []),
                "wall_time_s": arm.get("wall_time_s"),
                "rollout_steps": arm["geometry"].get("rollout_steps"),
                "updates_per_m_frames": updates_per_m_frames(arm),
                "scores": scores,
            }
        )
    return scored


def updates_per_m_frames(arm: dict[str, Any]) -> float:
    """Optimizer steps per million frames, from the arm's overrides.

    This is ``epochs / mini_batch_size`` scaled to a million frames -- note it
    does not involve ``frames_per_batch`` at all, which is exactly why the
    r12-vs-r24 cluster comparison confounded two axes: doubling the rollout
    while doubling the minibatch halves this number.

    Returns NaN when an arm's overrides do not pin both values, rather than
    guessing from config defaults that the launcher may not have used.
    """
    settings: dict[str, float] = {}
    for override in arm.get("overrides", []):
        key, _, value = str(override).partition("=")
        for name, suffix in (("epochs", "loss.epochs"), ("mb", "loss.mini_batch_size")):
            if key.endswith(suffix):
                try:
                    settings[name] = float(value)
                except ValueError:
                    return math.nan
    epochs = settings.get("epochs", 5.0)  # launcher default; overridden by a7
    mini_batch = settings.get("mb")
    if not mini_batch:
        return math.nan
    return epochs / mini_batch * 1_000_000.0


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-3):
        return f"{value:.3g}"
    return f"{value:.{digits}g}"


def warn_rate_gain_with_tracking_loss(
    scored: list[dict[str, Any]], baseline: str
) -> list[str]:
    """Flag arms that win on rate while tracking quality gets worse.

    Both goal metrics are per-minute rates over quantities that grow with
    episode length, and episode length is set by the termination thresholds. So
    anything that loosens a termination raises return per minute and episode
    length per minute mechanically, without the policy tracking any better --
    the 2026-08-02 termination-curriculum arm gained 1.80x on return per minute
    while its MPJPE went from 71.6 mm to 99.2 mm.

    MPJPE is per-frame and length-independent, so it is the check. An arm that
    gains rate while losing MPJPE has relaxed the test rather than improved on
    it, and the table says so rather than leaving it to be noticed.
    """
    base = next((a for a in scored if a["arm"] == baseline), None)
    if base is None:
        return []
    base_rate = base["scores"].get("return_per_min", math.nan)
    base_mpjpe = base["scores"].get("mpjpe_mm", math.nan)
    if not (math.isfinite(base_rate) and math.isfinite(base_mpjpe)):
        return []

    flagged = []
    for arm in scored:
        if arm["arm"] == baseline:
            continue
        rate = arm["scores"].get("return_per_min", math.nan)
        mpjpe = arm["scores"].get("mpjpe_mm", math.nan)
        if not (math.isfinite(rate) and math.isfinite(mpjpe)):
            continue
        # Strictly better rate, strictly worse tracking.
        if rate > base_rate and mpjpe > base_mpjpe:
            flagged.append(
                f"- `{arm['arm']}`: return/min {rate:.3f} vs {base_rate:.3f} "
                f"(+{100 * (rate / base_rate - 1):.0f}%), but MPJPE {mpjpe:.2f} mm "
                f"vs {base_mpjpe:.2f} (+{100 * (mpjpe / base_mpjpe - 1):.0f}% worse)"
            )
    if not flagged:
        return []
    return [
        "",
        "## Rate gained, tracking lost",
        "",
        "These arms beat the baseline on return per minute while tracking "
        "*worse*. Per-minute rates grow with episode length, and episode length "
        "is set by the termination thresholds, so loosening a termination "
        "raises the rate without improving the policy. Read these as a relaxed "
        "test, not a result.",
        "",
        *flagged,
    ]


def render_markdown(
    scored: list[dict[str, Any]],
    geometry: dict[str, Any],
    baseline: str,
    tail_fraction: float,
) -> str:
    """Render the screen as a Markdown report."""
    baseline_arm = next((a for a in scored if a["arm"] == baseline), None)
    lines = [
        "# RLOpt hyperparameter screen",
        "",
        f"Geometry: {geometry.get('num_envs')} envs, "
        f"{geometry.get('total_frames')} frames per arm. Rollout length and "
        "update density vary by arm and are reported per row.",
        f"Scores are the mean over the trailing {tail_fraction:.0%} of each curve.",
        "",
        "**This is a screen, not a result.** At this budget the arms are still in "
        "the earliest phase of learning; it ranks early progress and optimizer "
        "health, and cannot speak to late-training behaviour.",
        "",
    ]
    if baseline_arm is None:
        lines.append(
            f"> Baseline arm `{baseline}` is absent, so no deltas are reported.\n"
        )

    # ret/min leads the scoring columns: the objective is quality per unit
    # wall-clock, not per sample. r_step is retained as a diagnostic only -- it
    # is a reward density, so a short-episode arm can post a healthy r_step
    # while making no progress at all.
    header = [
        "arm",
        "steps",
        "upd/Mf",
        "ret/min",
        "eplen/min",
        "min",
        "ep_len",
        "return",
        "r_step",
        "mpjpe_mm",
        "lr_geomean",
        "lr_spread",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for arm in scored:
        s = arm["scores"]
        row = [
            f"`{arm['arm']}`",
            str(arm.get("rollout_steps") or "?"),
            _fmt(arm.get("updates_per_m_frames"), 3),
            _fmt(s["return_per_min"], 3),
            _fmt(s["ep_len_per_min"], 3),
            _fmt(s["train_minutes"], 3),
            _fmt(s["episode/length"]),
            _fmt(s["episode/return"]),
            _fmt(s["train/step_reward_mean"]),
            _fmt(s["mpjpe_mm"]),
            _fmt(s["lr_geomean"]),
            _fmt(s["lr_spread"], 3),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines += warn_rate_gain_with_tracking_loss(scored, baseline)
    lines += ["", "## Optimizer health", ""]
    header2 = ["arm", "kl_approx", "entropy", "clip_frac", "expl_var", "grad_norm"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "|".join(["---"] * len(header2)) + "|")
    for arm in scored:
        s = arm["scores"]
        row = [
            f"`{arm['arm']}`",
            _fmt(s["train/kl_approx"]),
            _fmt(s["train/entropy"]),
            _fmt(s["train/clip_fraction"]),
            _fmt(s["train/explained_variance"]),
            _fmt(s["train/grad_norm"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    if baseline_arm is not None:
        lines += ["", f"## Change vs `{baseline}`", ""]
        lines.append("| arm | ep_len | return | r_step | mpjpe_mm |")
        lines.append("|---|---|---|---|---|")
        base = baseline_arm["scores"]
        for arm in scored:
            if arm["arm"] == baseline:
                continue
            s = arm["scores"]
            cells = []
            for key in PROGRESS_METRICS:
                ref, val = base[key], s[key]
                if not (math.isfinite(ref) and math.isfinite(val)) or ref == 0:
                    cells.append("n/a")
                else:
                    cells.append(f"{(val - ref) / abs(ref):+.1%}")
            lines.append(f"| `{arm['arm']}` | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "Lower MPJPE is better; the other three are higher-is-better. A "
            "positive MPJPE delta is therefore a regression."
        )

    lines += ["", "## Arms", ""]
    for arm in scored:
        wall = arm["wall_time_s"]
        wall_s = f" ({wall / 60:.0f} min)" if isinstance(wall, (int, float)) else ""
        lines.append(f"- `{arm['arm']}`{wall_s} — {arm['description']}")
        for override in arm["overrides"]:
            lines.append(f"  - `{override}`")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.out.exists():
        raise ScreenError(f"{args.out}: refusing to overwrite an existing report")
    if args.screen_root is not None:
        arms = discover_arms(args.screen_root)
    else:
        arms = discover_arms_from_wandb(
            project=args.wandb_project,
            group=args.wandb_group,
            entity=args.wandb_entity,
            arm_prefix=args.wandb_arm_prefix,
        )
    geometry = require_matched_geometry(arms)
    scored = score_arms(arms, args.tail_fraction)
    report = render_markdown(scored, geometry, args.baseline, args.tail_fraction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"[INFO] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
