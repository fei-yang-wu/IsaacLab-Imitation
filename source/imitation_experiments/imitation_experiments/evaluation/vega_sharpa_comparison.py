"""Combine complete, matched box-endpoint evaluations and draw a scorecard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def collect_comparison(reports):
    """Reject partial or mismatched evaluations before comparing policies."""
    rows = {}
    labels = []
    sources = []
    identity = None
    checkpoint_stem = "model_step_3072000"
    for label, path in reports:
        path = Path(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        if (
            not report.get("complete")
            or len(report["results"]) != report["expected_result_count"]
        ):
            raise ValueError(f"Incomplete evaluation: {path}")
        protocol = report["protocol"]
        current = {
            "reference_manifest_sha256": report["reference_manifest_sha256"],
            "reference_sha256": report["reference_sha256"],
            "seed": report["seed"],
            "num_envs": report["num_envs"],
            "protocol": protocol["name"],
            "position_tolerance_m": protocol["position_tolerance_m"],
            "orientation_tolerance_rad_secondary": protocol[
                "orientation_tolerance_rad_secondary"
            ],
            "observation_corruption": protocol["observation_corruption"],
        }
        if current["protocol"] != "object-task":
            raise ValueError(
                "Comparison requires fixed-horizon object-task evaluations."
            )
        if current["position_tolerance_m"] != 0.05:
            raise ValueError("This comparison requires the frozen 5 cm criterion.")
        if identity is not None and current != identity:
            raise ValueError("Evaluation protocols or Reference identities differ.")
        identity = current
        sources.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        if label not in labels:
            labels.append(label)
        selected = [
            r
            for r in report["results"]
            if r["policy"] == checkpoint_stem and r["start_mode"] == "first"
        ]
        if not selected:
            raise ValueError(f"No selected checkpoint/first-start row: {path}")
        for result in selected:
            endpoint = result["object_endpoint"]
            if not result["policy_and_normalizer_unchanged"] or not all(
                endpoint["reached_reference_end"]
            ):
                raise ValueError(
                    "Every trial must reach the deadline with a frozen policy."
                )
            if endpoint["trial_count"] != current["num_envs"]:
                raise ValueError(
                    "Evaluation trial count differs from the declared world count."
                )
            key = (label, float(result["assistance_scale"]))
            if key in rows:
                raise ValueError(f"Duplicate policy/assistance row: {key}")
            rows[key] = {
                "label": label,
                "assistance": key[1],
                "checkpoint": result["checkpoint"],
                "start_frames": result["start_frames"],
                "control_steps": result["episode_control_steps"],
                "position_error_m": endpoint["position_error_m"],
                "orientation_error_rad": endpoint["orientation_error_rad"],
                "success_count": endpoint["position_success_count"],
                "trial_count": endpoint["trial_count"],
            }
    if not rows:
        raise ValueError("No comparison rows.")
    expected = {(label, scale) for label in labels for scale in (1.0, 0.75, 0.0)}
    if set(rows) != expected:
        raise ValueError("Every policy requires assistance 1, 0.75 and 0.")
    first = next(iter(rows.values()))
    if any(
        r["start_frames"] != first["start_frames"]
        or r["control_steps"] != first["control_steps"]
        for r in rows.values()
    ):
        raise ValueError("Evaluation starts or fixed horizons differ.")
    return {
        "qualification": "One training seed; simulated worlds are not independent training seeds.",
        "identity": identity,
        "sources": sources,
        "labels": labels,
        "rows": [
            rows[(label, scale)] for scale in (1.0, 0.75, 0.0) for label in labels
        ],
    }


def write_artifacts(comparison, output):
    """Write tabular data and a static figure with all per-trial errors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(comparison, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "policy",
                "assistance",
                "mean_final_error_cm",
                "success_5cm",
                "trials",
                "mean_orientation_error_deg",
            ]
        )
        for row in comparison["rows"]:
            writer.writerow(
                [
                    row["label"],
                    row["assistance"],
                    100 * row["position_error_m"]["mean"],
                    row["success_count"],
                    row["trial_count"],
                    np.degrees(row["orientation_error_rad"]["mean"]),
                ]
            )
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    colors = ["#64748b", "#2563eb", "#d97706", "#7c3aed", "#059669"]
    max_error = max(
        max(r["position_error_m"]["per_trial"]) * 100 for r in comparison["rows"]
    )
    jitter = np.random.default_rng(0)
    for ax, scale in zip(axes, (1.0, 0.75, 0.0), strict=True):
        selected = [r for r in comparison["rows"] if r["assistance"] == scale]
        tick_labels = []
        for index, row in enumerate(selected):
            errors = np.asarray(row["position_error_m"]["per_trial"]) * 100
            color = colors[index % len(colors)]
            ax.scatter(
                index + jitter.uniform(-0.13, 0.13, len(errors)),
                errors,
                s=13,
                alpha=0.55,
                color=color,
            )
            ax.scatter(
                index,
                100 * row["position_error_m"]["mean"],
                marker="D",
                s=43,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            tick_labels.append(
                f"{row['label']}\n{row['success_count']}/{row['trial_count']}"
            )
        ax.axhline(5, linestyle="--", linewidth=1, color="#b91c1c")
        ax.set_yscale("log")
        ax.set_ylim(0.5, max(150, max_error * 1.3))
        ax.set_xticks(range(len(selected)), tick_labels, fontsize=9)
        ax.set_title("Unassisted" if scale == 0 else f"Assistance {scale:g}")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Final box-center error (cm, log scale)")
    fig.suptitle("Vega–Sharpa: fixed-horizon box placement", fontsize=16)
    fig.text(
        0.5,
        0.89,
        f"1,000 PPO iterations · seed {comparison['identity']['seed']} · "
        f"{comparison['identity']['num_envs']} first-start trials per condition",
        ha="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "Dots: trials. Diamonds: means. Counts: success within 5 cm (dashed line).",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.88))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="LABEL=JSON; repeat a label to combine assisted/unassisted reports.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [value.split("=", 1) for value in args.report]
    write_artifacts(collect_comparison(reports), args.output)


if __name__ == "__main__":
    main()
