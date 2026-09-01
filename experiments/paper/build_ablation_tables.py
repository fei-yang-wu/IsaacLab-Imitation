#!/usr/bin/env python3
"""Emit the three star-v2 ablation tables as LaTeX.

Numbers are read from the scored evaluation JSONs, never transcribed, so a
table cannot drift from the rows it claims to report. Every table is pinned to
one checkpoint: arms train at different speeds, and taking each arm's deepest
row would put a 2.6B checkpoint beside a 2.0B one.

A row whose arm has no scored row at that checkpoint prints `--` and is listed
in a trailing comment, so a gap stays visible in the draft instead of silently
vanishing. A table with no populated row at all exits non-zero, because that
means the checkpoint or directory is wrong.

    python experiments/paper/build_ablation_tables.py --table target
    python experiments/paper/build_ablation_tables.py --table all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from imitation_experiments.paths import REPO_ROOT
from imitation_experiments.reporting.ablation_sections import scored_rows

# 4,070 iterations x 20,480 envs x 24 rollout steps lands the 2B screen here.
SCREEN_FRAMES = 2000486400
EVAL_DIR = REPO_ROOT / "logs" / "latent_star_v2_eval"

BUDGET_NOTE = (
    "All arms share one interface and one training budget "
    "({budget:.1f}B environment frames) and are scored on the same "
    "4{{,}}096-clip board with domain randomization off. Success rate uses "
    "SONIC's termination definition; both MPJPE columns are success-only. "
    "One seed per arm."
)


@dataclass(frozen=True)
class Row:
    """One printed row. `arm=None` marks a cell this campaign does not have."""

    arm: str | None
    cells: tuple[str, ...]
    highlight: bool = False
    note: str = ""


@dataclass(frozen=True)
class Table:
    label: str
    caption: str
    colspec: str
    header: tuple[str, ...]
    tabcolsep: str
    arraystretch: str
    body: list[Row | None] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)


def _t(label, caption, colspec, header, tabcolsep, arraystretch, body):
    return Table(label, caption, colspec, header, tabcolsep, arraystretch, body)


TARGET = _t(
    "tab:predictive_target",
    "Predictive-target ablation.",
    "@{}lrrr@{}",
    ("Prediction target",),
    "3.0pt",
    "1.07",
    [
        Row("hub", (r"\textbf{Ours}",), highlight=True),
        Row("g2_twohead", ("Split end-point $+$ chunk",)),
        Row("g2_endpoint", ("End-point",)),
        Row("g2_mlp", ("End-point-det",)),
        Row("g2_token", ("Next-latent",)),
        Row(None, ("Next-anchor",), note="chunkra was dropped by user decision"),
        Row(None, ("Next-pair",), note="diff_pair was never added to the campaign"),
        None,
        Row("g2_delta", ("End-point delta",)),
        Row("g2_state_occupancy", ("Successor occupancy",)),
        Row("g2_semimarkov", ("Semi-Markov",)),
        Row("g2_trip", ("Triplet context",)),
    ],
)

DESIGN = _t(
    "tab:design_ablation",
    "Additional interface design choices.",
    "@{}llrrr@{}",
    ("Design axis", "Variant"),
    "2.7pt",
    "1.05",
    [
        Row("hub", (r"\multicolumn{2}{l}{\textbf{Ours}}",), highlight=True),
        None,
        Row("g3_cont128", (r"\multirow{2}{*}{Latent width}", "Cont. 128-D")),
        Row("g3_cont256", ("", "Cont. 256-D")),
        None,
        Row("g3_fsq64", (r"FSQ", r"$64\times32$")),
        None,
        Row(
            "g3_multicat_gumbel", (r"\multirow{3}{*}{Codebook}", r"Gumbel $64\times32$")
        ),
        Row("g3_multicat", ("", r"Cat. $64\times32$")),
        Row("g3_vq64", ("", "VQ-EMA")),
        None,
        Row("g4_fullbody670", (r"\multirow{5}{*}{Enc. input}", r"$+$ joint velocity")),
        Row("g4_stride5", ("", "Stride $5$")),
        Row("g4_window_full", ("", "Full window")),
        Row("g4_h5", ("", "Horizon $5$")),
        Row("g4_h20", ("", "Horizon $20$")),
        None,
        Row("g4_anchor_robot", (r"\multirow{2}{*}{Anchoring}", "Robot frame")),
        Row("g4_anchor_expert", ("", "Expert heading")),
        None,
        Row("g5_hold5", (r"\multirow{2}{*}{Cadence}", "Hold $5$")),
        Row("g5_hold10", ("", "Hold $10$")),
        None,
        Row("g5_phase_none_h10", ("Phase", "No phase (hold $10$)")),
    ],
)

REPR = _t(
    "tab:repr_ablation",
    "Representation-learning ablation.",
    "@{}llrrr@{}",
    ("Representation", "Training"),
    "2.5pt",
    "1.06",
    [
        Row("hub", (r"\textbf{Ours}", "offline, cont. 64"), highlight=True),
        Row("g3_cont256", ("Ours", "offline, cont. 256")),
        Row("g3_fsq64", ("Ours", "offline, FSQ")),
        Row("g3_vq64", ("Ours", "offline, VQ")),
        Row("g3_multicat", ("Ours", "offline, categorical")),
        None,
        Row("g1_recon_ae", ("Reconstruction", "offline, cont.")),
        Row("g1_recon_fsq", ("Reconstruction", "offline, FSQ")),
        Row("g1_recon_vq", ("Reconstruction", "offline, VQ")),
        None,
        Row("g1_post_ae", ("Recon.", "joint, cont.")),
        Row("g1_post_pg_ae", ("PG", "joint, cont.")),
        Row("g1_post_pg_fsq", ("PG", "joint, FSQ")),
        Row("g1_post_pg_vq", ("PG", "joint, VQ")),
        Row("g1_post_pgrecon_ae", ("Recon.+PG", "joint, cont.")),
        Row("g1_post_pgrecon_fsq", ("Recon.+PG", "joint, FSQ")),
        Row("g1_post_pgrecon_vq", ("Recon.+PG", "joint, VQ")),
    ],
)

TABLES = {"repr": REPR, "target": TARGET, "design": DESIGN}


def build(table: Table, at_frames: int, eval_dir: Path) -> tuple[str, int]:
    scores = scored_rows(eval_dir, row="clean", at_frames=at_frames)
    header = " & ".join(table.header)
    lines = [
        r"\begin{table}[t]",
        r"\caption{",
        f"{table.caption}",
        BUDGET_NOTE.format(budget=at_frames / 1e9),
        r"}",
        rf"\label{{{table.label}}}",
        r"\centering",
        r"\small",
        rf"\setlength{{\tabcolsep}}{{{table.tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{table.arraystretch}}}",
        rf"\begin{{tabular}}{{{table.colspec}}}",
        r"\toprule",
        rf"{header}",
        r"& SR $\uparrow$",
        r"& MPJPE-L $\downarrow$",
        r"& MPJPE-G $\downarrow$ \\",
        r"\midrule",
        "",
    ]
    missing: list[str] = []
    populated = 0
    for row in table.body:
        if row is None:
            lines += [r"\midrule", ""]
            continue
        scored = scores.get(row.arm) if row.arm else None
        if scored is None or scored.get("success_rate") is None:
            missing.append(row.note or (row.arm or "?"))
            sr = local = glob = "--"
        else:
            populated += 1
            sr = f"{100 * scored['success_rate']:.2f}"
            local = _fmt(scored["mpjpe_local_mm"])
            glob = _fmt(scored["mpjpe_global_mm"])
        if row.highlight:
            lines.append(r"\rowcolor{ctrlblue}")
        cells = list(row.cells)
        lines.append(cells[0])
        for cell in cells[1:]:
            lines.append(f"& {cell}")
        lines += [f"& {sr} & {local} & {glob} \\\\", ""]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    if missing:
        text += "\n\n% NOT AVAILABLE at this checkpoint: " + "; ".join(missing)
    return text, populated


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=[*TABLES, "all"], default="all")
    parser.add_argument("--at-frames", type=int, default=SCREEN_FRAMES)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args(argv)
    if not args.eval_dir.is_dir():
        parser.error(f"eval dir does not exist: {args.eval_dir}")

    wanted = list(TABLES) if args.table == "all" else [args.table]
    total = 0
    for index, name in enumerate(wanted):
        text, populated = build(TABLES[name], args.at_frames, args.eval_dir)
        total += populated
        if index:
            print()
        print(text)
    if total == 0:
        parser.error(
            f"no arm has a scored row at {args.at_frames} frames in {args.eval_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
