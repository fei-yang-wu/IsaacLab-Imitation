"""Audit local mounted PPO scalar logs and export training curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_scalars(run_dir: Path) -> dict[str, np.ndarray]:
    directories = list(run_dir.glob("*/scalars"))
    if len(directories) != 1:
        raise ValueError("Expected exactly one scalar-log directory in the run.")
    result = {}
    for path in sorted(directories[0].rglob("*.csv")):
        rows = [[float(x) for x in row] for row in csv.reader(path.open()) if row]
        values = np.asarray(rows, dtype=np.float64)
        if values.size == 0:
            continue
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError(f"Invalid scalar series: {path}")
        if (np.diff(values[:, 0]) <= 0).any():
            raise ValueError(f"Non-increasing scalar steps: {path}")
        result[str(path.relative_to(directories[0]).with_suffix(""))] = values
    return result


def summarize(
    scalars, *, num_envs: int, expected_iterations: int, window: int = 100
) -> dict:
    batch = 24 * num_envs
    required = ["train/loss_objective", "train/loss_critic", "train/grad_norm"]
    last = min(int(scalars[name][-1, 0]) for name in required)
    complete = last == expected_iterations * batch
    if last > expected_iterations * batch or last % batch:
        raise ValueError("The observed frame count disagrees with the declared budget.")
    for name in required:
        steps = scalars[name][scalars[name][:, 0] <= last, 0]
        if not np.array_equal(steps, np.arange(batch, last + 1, batch)):
            raise ValueError(f"Missing optimization records: {name}")
    summaries = {}
    for name, values in scalars.items():
        values = values[values[:, 0] <= last]
        if not len(values):
            continue
        summaries[name] = {
            "records": len(values),
            "first_window_mean": float(values[:window, 1].mean()),
            "last_window_mean": float(values[-window:, 1].mean()),
            "minimum": float(values[:, 1].min()),
            "maximum": float(values[:, 1].max()),
        }
    assistance = scalars.get("Curriculum/fixed_timestep")
    changes = []
    if assistance is not None:
        for i in np.flatnonzero(np.diff(assistance[:, 1]) != 0) + 1:
            if assistance[i, 0] <= last:
                changes.append(
                    {
                        "iteration": float(assistance[i, 0] / batch),
                        "scale": float(assistance[i, 1]),
                    }
                )
    return {
        "qualification": "one-seed local training curve; fixed-start evaluation is separate",
        "complete": complete,
        "frames": last,
        "iterations": last // batch,
        "expected_iterations": expected_iterations,
        "num_envs": num_envs,
        "rollout_steps": 24,
        "all_logged_scalars_finite": True,
        "all_optimization_rows_present": True,
        "window_iterations": window,
        "assistance_changes": changes,
        "metrics": summaries,
    }


def plot_curves(scalars, summary, output: Path, *, window: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("Reward per control step", ["train/step_reward_mean"], "reward"),
        ("Episode return", ["episode/return"], "return"),
        ("Episode duration", ["episode/length"], "control steps"),
        (
            "Hand–object contact reward",
            ["Episode_Reward/contact_wrench_support_reward"],
            "logged reward",
        ),
        (
            "Reference-end share among reset episodes",
            ["Episode_Termination/time_out"],
            "fraction",
        ),
        (
            "Object / object-relative wrist errors (reset summaries)",
            [
                "Metrics/sharpa_reference/object_position_error",
                "Metrics/sharpa_reference/left_wrist_position_error",
                "Metrics/sharpa_reference/right_wrist_position_error",
            ],
            "metres",
        ),
        ("Critic loss", ["train/loss_critic"], "loss"),
        ("Approximate policy KL", ["train/kl_approx"], "KL"),
    ]
    colors = ["#126782", "#dc7b1c", "#7f4f9c"]
    fig, axes = plt.subplots(
        4, 2, figsize=(14, 13), sharex=True, constrained_layout=True
    )
    for ax, (title, keys, ylabel) in zip(axes.flat, panels, strict=True):
        for i, name in enumerate(keys):
            if name not in scalars:
                continue
            values = scalars[name]
            values = values[values[:, 0] <= summary["frames"]]
            if len(values) == 0:
                continue
            x = values[:, 0] / (24 * summary["num_envs"])
            label = (
                name.rsplit("/", 1)[-1].replace("_position_error", "").replace("_", " ")
            )
            if "wrist_position_error" in name:
                label += " relative to object"
            ax.plot(x, values[:, 1], color=colors[i], alpha=0.15, linewidth=0.7)
            w = min(window, len(values))
            ax.plot(
                x[w - 1 :],
                np.convolve(values[:, 1], np.ones(w) / w, mode="valid"),
                color=colors[i],
                linewidth=1.7,
                label=label,
            )
        for change in summary["assistance_changes"]:
            ax.axvline(
                change["iteration"], color="#a13d3d", linestyle="--", linewidth=1
            )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(ylabel)
        if keys == ["train/kl_approx"]:
            kl = scalars[keys[0]][:, 1]
            if kl.min() > 0:
                ax.set_yscale("log")
                ax.set_ylim(float(kl.min()) * 0.75, float(kl.max()) * 1.25)
            else:
                ax.set_yscale("symlog", linthresh=1e-4)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        if len(keys) > 1:
            ax.legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("PPO rollout iteration")
    status = "complete" if summary["complete"] else "partial"
    fig.suptitle(
        f"Vega U + dual Sharpa PPO — one seed, {status}\n"
        f"{summary['iterations']:,}/{summary['expected_iterations']:,} iterations · {summary['frames'] / 1e6:.3f}M frames · "
        f"{summary['num_envs']} environments\n"
        f"Faint lines: raw logs; solid lines: {window}-iteration means; dashed lines: assistance changes",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--expected-iterations", type=int, required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.expected_iterations, args.window) < 1:
        parser.error("Counts and smoothing window must be positive.")
    scalars = load_scalars(args.run_dir)
    summary = summarize(
        scalars,
        num_envs=args.num_envs,
        expected_iterations=args.expected_iterations,
        window=args.window,
    )
    if args.require_complete and not summary["complete"]:
        raise ValueError("The training run is still incomplete.")
    output = args.output.expanduser().resolve()
    plot_curves(scalars, summary, output, window=args.window)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training curves: {output}")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("complete", "iterations", "frames", "assistance_changes")
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
