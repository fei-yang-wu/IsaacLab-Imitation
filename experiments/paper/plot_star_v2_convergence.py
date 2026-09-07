#!/usr/bin/env python3
"""Convergence figures for the star-v2 command-interface ablation.

Reads the scored curve points (`logs/report/star_v2_curves/star_v2_curve_points.csv`,
built by `logs/report/star_v2_curves/export_curve_points.py`), joins the dense
early-phase rerun (checkpoints every 9.83M to 0.2B) onto the 200M-grid rows of
the original 5B runs, drops the arms that diverged, and draws one figure per
ablation group: success rate, MPJPE-L, MPJPE-G against environment frames, with
the hub ("Ours") as the reference line in every panel.

Encoding. Arms are RANKED inside each group by success rate at the screen
budget and colored along one blue ramp, deep blue (best) through light blue to
gray-blue (worst); the hub is the navy reference line. The legend lists arms in
rank order with the value that ranked them. Line style still names the
variant inside a family (continuous / FSQ / VQ, EMA / stop-grad / online
target, hold 5 / hold 10), so two neighbours on the ramp remain separable.
`--color-by family` restores the six-hue categorical palette (validated for
colorblind separation in legend order). Font: IBM Plex Sans, loaded from the
user font directory when present.

The two runs are one training recipe but two training runs: a faint rule at
0.2B marks the join. Over 43 arms the median success-rate gap at the join is
0.009 (90th percentile 0.022); the arms whose runs disagree or that collapsed
are excluded by EXCLUDE_ARMS and printed on every run.

Outputs (`logs/report/star_v2_curves/paper/`):
    star_v2_convergence_<group>.{pdf,png}   one figure per ablation group
    star_v2_convergence_overview.{pdf,png}  success rate only, all groups
    star_v2_convergence_figure_data.csv     exactly the points that were drawn
    star_v2_convergence_meta.json           palette, exclusions, join stats

    python experiments/paper/plot_star_v2_convergence.py
    python experiments/paper/plot_star_v2_convergence.py --xscale linear
    python experiments/paper/plot_star_v2_convergence.py --groups g1 g3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import pandas as pd  # noqa: E402

from imitation_experiments.paths import REPO_ROOT  # noqa: E402

# ----------------------------------------------------------------------------
# Inputs and outputs
# ----------------------------------------------------------------------------
POINTS_CSV = REPO_ROOT / "logs/report/star_v2_curves/star_v2_curve_points.csv"
OUT_DIR = REPO_ROOT / "logs/report/star_v2_curves/paper"
SCREEN_B = 2.0  # the ablation tables report 2B; curves stop there too
JOIN_B = 0.2  # the dense rerun ends and the 200M grid begins

# Arms left out of every figure. Each entry names why, and the list is printed
# on every run so an omission never passes silently.
EXCLUDE_ARMS: dict[str, str] = {
    "g3_vq64": "collapsed: success rate 0.00 throughout (frozen VQ codebook)",
    "g1_recon_vq": "collapsed in the original run (0.05 at 2B) and the rerun "
    "disagrees at the join (0.345 vs 0.016)",
    "g5_phase_none_h10": "diverged: success rate flat at 0.37, MPJPE-L flat at 67 mm",
}
# A safety net for arms not listed above: any arm whose original 2B success
# rate falls under this is excluded too, and reported.
MIN_FINAL_SUCCESS = 0.40

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
FONT_DIRS = (
    Path.home() / ".local/share/fonts/IBMPlexSans",
    Path.home() / ".local/share/fonts",
    Path.home() / ".fonts",
)
FONT_FAMILY = "IBM Plex Sans"

INK = "#0B2A5B"  # hub / reference line and axis ink: deep navy
INK_SOFT = "#4A5B78"  # secondary text
GRID = "#D9E1EC"
SURFACE = "#FFFFFF"
# Categorical hues in legend order. Validated: lightness band, chroma floor,
# adjacent-pair colorblind separation, contrast >= 3:1 on white. Adjacent
# magenta/teal sits in the 6-8 floor band, which is legal because line style
# is the second encoding.
HUES = {
    "blue": "#2B6CE0",
    "amber": "#BF7318",
    "teal": "#12A0A0",
    "magenta": "#C8407A",
    "violet": "#7B4FE0",
    "green": "#4F9A32",
}
# Rank ramp, best -> worst: deep blue, light blue, gray-blue. Sampled evenly
# for however many arms a group has; the hub stays INK above the top of it.
RAMP = ("#1746B8", "#3D7BE0", "#7FB3EE", "#9DB0C6")
STYLES = {"solid": "-", "dashed": (0, (4.5, 2.2)), "dotted": (0, (1.4, 1.8))}

LW_HUB, LW_ARM = 1.6, 0.9
MS_HUB, MS_ARM = 1.9, 1.4
# The dense rerun puts a point every 9.83M frames; between THIN_LO and THIN_HI
# (B) only every THIN_EVERY-th point is drawn, so evaluation noise between
# neighbours does not read as structure. Ranking uses the last point, which
# is outside the window and never thinned. `--thin-every 1` draws everything.
THIN_LO, THIN_HI, THIN_EVERY = 0.03, 0.3, 2


@dataclass(frozen=True)
class ArmStyle:
    hue: str
    style: str = "solid"


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    # arm -> style; the dict order is the legend order
    arms: dict[str, ArmStyle]
    # legend entries for the style axis, when a group uses one
    style_legend: dict[str, str] | None = None


# Hue = family, line style = variant. One paper label per arm comes from the
# campaign file through the points CSV, so the figure and the table agree.
GROUPS: tuple[Group, ...] = (
    Group(
        "g1",
        "Reconstruction and posterior routes",
        {
            "g1_recon_ae": ArmStyle("blue"),
            "g1_recon_fsq": ArmStyle("blue", "dashed"),
            "g1_recon_vq": ArmStyle("blue", "dotted"),
            "g1_post_ae": ArmStyle("magenta"),
            "g1_post_pg_ae": ArmStyle("teal"),
            "g1_post_pg_fsq": ArmStyle("teal", "dashed"),
            "g1_post_pg_vq": ArmStyle("teal", "dotted"),
            "g1_post_pgrecon_ae": ArmStyle("amber"),
            "g1_post_pgrecon_fsq": ArmStyle("amber", "dashed"),
            "g1_post_pgrecon_vq": ArmStyle("amber", "dotted"),
        },
        {"solid": "continuous", "dashed": "FSQ", "dotted": "VQ"},
    ),
    Group(
        "g2a",
        "Factorization target",
        {
            "g2_twohead": ArmStyle("blue"),
            "g2_endpoint": ArmStyle("teal"),
            "g2_delta": ArmStyle("amber"),
            "g2_state_occupancy": ArmStyle("magenta"),
            "g2_semimarkov": ArmStyle("violet"),
            "g2_phi_bilinear": ArmStyle("green"),
        },
    ),
    Group(
        "g2b",
        "Predictor form",
        {
            "g2_sg": ArmStyle("blue", "dashed"),
            "g2_online": ArmStyle("blue", "dotted"),
            "g2_lejepa_ema": ArmStyle("violet"),
            "g2_lejepa_sg": ArmStyle("violet", "dashed"),
            "g2_lejepa_online": ArmStyle("violet", "dotted"),
            "g2_mlp": ArmStyle("amber"),
            "g2_dsrsig": ArmStyle("teal"),
            "g2_nosig": ArmStyle("magenta"),
            "g2_trip": ArmStyle("green"),
            "g2_token": ArmStyle("green", "dashed"),
        },
        {
            "solid": "EMA target",
            "dashed": "stop-grad target",
            "dotted": "online target",
        },
    ),
    Group(
        "g3",
        "Encoder space",
        {
            "g3_cont256": ArmStyle("blue"),
            "g3_cont128": ArmStyle("blue", "dashed"),
            "g3_fsq64": ArmStyle("teal"),
            "g3_vq64": ArmStyle("teal", "dotted"),
            "g3_multicat": ArmStyle("amber"),
            "g3_multicat_gumbel": ArmStyle("amber", "dashed"),
        },
    ),
    Group(
        "g4",
        "Encoder input and window",
        {
            "g4_anchor_robot": ArmStyle("blue"),
            "g4_anchor_expert": ArmStyle("blue", "dashed"),
            "g4_h5": ArmStyle("violet", "dashed"),
            "g4_h20": ArmStyle("violet", "dotted"),
            "g4_fullbody670": ArmStyle("amber"),
            "g4_stride5": ArmStyle("teal"),
            "g4_window_full": ArmStyle("magenta"),
        },
    ),
    Group(
        "g56",
        "Command cadence and encoder adaptation",
        {
            "g5_hold5": ArmStyle("blue", "dashed"),
            "g5_hold10": ArmStyle("blue", "dotted"),
            "g5_phase_none_h10": ArmStyle("magenta", "dotted"),
            "g6_dyn": ArmStyle("amber"),
        },
    ),
)
HUB = "hub"

X_TICKS_LOG = (0.01, 0.03, 0.1, 0.3, 1, 2)
# Whole-number ticks inside each metric's observed range (25-140 mm local,
# 100-1200 mm global), so a log axis still reads in millimetres.
Y_TICKS_LOG = {
    "mpjpe_local_mm": (25, 50, 100, 150),
    "mpjpe_global_mm": (100, 200, 400, 800, 1200),
}

PANELS = (
    ("success_rate", "Success rate", "linear"),
    ("mpjpe_local_mm", "MPJPE-L (mm)", "log"),
    ("mpjpe_global_mm", "MPJPE-G (mm)", "log"),
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_points(path: Path) -> pd.DataFrame:
    if not path.is_file():
        sys.exit(
            f"[FATAL] points CSV missing: {path}\n  build it with "
            "logs/report/star_v2_curves/export_curve_points.py"
        )
    df = pd.read_csv(path)
    needed = {
        "run",
        "arm",
        "paper_label",
        "frames_b",
        "success_rate",
        "mpjpe_local_mm",
        "mpjpe_global_mm",
        "num_successful_envs",
    }
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"[FATAL] points CSV lacks columns {sorted(missing)}")
    if "early" not in set(df["run"]):
        sys.exit("[FATAL] no `early` rows: the dense rerun has not been exported")
    return df


def joined_series(df: pd.DataFrame, max_b: float) -> pd.DataFrame:
    """One ordered series per arm: dense rerun below JOIN_B, then the 200M grid.

    The two corpora overlap only at the join (0.197B rerun against 0.200B
    original); both points are kept so the join is visible, not smoothed.
    """
    early = df[(df["run"] == "early") & (df["frames_b"] < JOIN_B)]
    grid = df[(df["run"] == "original") & (df["frames_b"] >= JOIN_B - 1e-6)]
    out = pd.concat([early, grid], ignore_index=True)
    out = out[out["frames_b"] <= max_b + 1e-6]
    return out.sort_values(["arm", "frames_b"]).reset_index(drop=True)


def thin_series(series: pd.DataFrame, lo: float, hi: float, every: int) -> pd.DataFrame:
    """Keep every `every`-th point inside [lo, hi] B, plus the window's last
    point, so the drawn curve leaves the window where the data does."""
    if every <= 1:
        return series
    fb = series["frames_b"].to_numpy()
    inside = (fb >= lo - 1e-9) & (fb <= hi + 1e-9)
    idx = inside.nonzero()[0]
    keep = set(idx[::every].tolist())
    if len(idx):
        keep.add(int(idx[-1]))
    mask = ~inside
    mask[list(keep)] = True
    return series[mask].reset_index(drop=True)


def join_stats(df: pd.DataFrame) -> dict[str, float]:
    e = df[(df["run"] == "early") & (df["frames_b"] > 0.19)].set_index("arm")
    o = df[(df["run"] == "original") & df["frames_b"].between(0.19, 0.21)].set_index(
        "arm"
    )
    j = (
        e[["success_rate"]]
        .join(o[["success_rate"]], lsuffix="_e", rsuffix="_o")
        .dropna()
    )
    gap = (j["success_rate_e"] - j["success_rate_o"]).abs()
    return {
        "arms": int(len(j)),
        "median_abs_sr_gap": float(gap.median()),
        "p90_abs_sr_gap": float(gap.quantile(0.9)),
        "max_abs_sr_gap": float(gap.max()),
        "max_gap_arm": str(gap.idxmax()),
    }


def excluded_arms(df: pd.DataFrame) -> dict[str, str]:
    out = dict(EXCLUDE_ARMS)
    final = df[
        (df["run"] == "original")
        & df["frames_b"].between(SCREEN_B - 0.01, SCREEN_B + 0.01)
    ]
    for arm, sr in zip(final["arm"], final["success_rate"]):
        if arm not in out and arm != "untrained" and sr < MIN_FINAL_SUCCESS:
            out[arm] = f"diverged: success rate {sr:.3f} at {SCREEN_B:g}B"
    return out


# ----------------------------------------------------------------------------
# Style helpers
# ----------------------------------------------------------------------------
def install_font() -> str:
    found = False
    for d in FONT_DIRS:
        if d.is_dir():
            for f in list(d.glob("IBMPlexSans*.ttf")) + list(
                d.glob("IBMPlexSans*.otf")
            ):
                font_manager.fontManager.addfont(str(f))
                found = True
    names = {f.name for f in font_manager.fontManager.ttflist}
    if found and FONT_FAMILY in names:
        return FONT_FAMILY
    print(
        f"[WARN] {FONT_FAMILY} not found under {[str(d) for d in FONT_DIRS]}; "
        "falling back to DejaVu Sans",
        file=sys.stderr,
    )
    return "DejaVu Sans"


def apply_style(font: str) -> None:
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.edgecolor": INK_SOFT,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "text.color": INK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "pdf.fonttype": 42,  # embed TrueType so the PDF keeps the font
            "ps.fonttype": 42,
        }
    )


def ramp_colors(n: int) -> list[str]:
    """`n` colors evenly along RAMP, deep blue first."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    if n <= 0:
        return []
    if n == 1:
        return [RAMP[0]]
    cmap = LinearSegmentedColormap.from_list("rank_blue", RAMP)
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def rank_arms(
    group: Group, series: dict[str, pd.DataFrame], key: str = "success_rate"
) -> list[tuple[str, float]]:
    """Arms present in `series`, best first by the last point of `key`."""
    scored = []
    for arm in group.arms:
        if arm in series and len(series[arm]):
            scored.append((arm, float(series[arm][key].iloc[-1])))
    return sorted(scored, key=lambda t: -t[1])


def arm_colors(
    group: Group, series: dict[str, pd.DataFrame], color_by: str
) -> tuple[list[str], dict[str, str], dict[str, float]]:
    """Legend order, arm -> color, arm -> ranking value."""
    if color_by == "rank":
        ranked = rank_arms(group, series)
        order = [a for a, _ in ranked]
        colors = dict(zip(order, ramp_colors(len(order))))
        values = dict(ranked)
        return order, colors, values
    order = [a for a in group.arms if a in series]
    return order, {a: HUES[group.arms[a].hue] for a in order}, {}


def line_kwargs(style: ArmStyle, color: str) -> dict:
    return {"color": color, "linestyle": STYLES[style.style]}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_panel(
    ax,
    key: str,
    ylabel: str,
    yscale: str,
    series: dict[str, pd.DataFrame],
    group: Group,
    labels: dict[str, str],
    xscale: str,
    order: list[str],
    colors: dict[str, str],
    values: dict[str, float],
) -> None:
    hub = series[HUB]
    ax.plot(
        hub["frames_b"],
        hub[key],
        color=INK,
        lw=LW_HUB,
        marker="o",
        ms=MS_HUB,
        mew=0,
        zorder=6,
        label=labels[HUB],
        solid_capstyle="round",
    )
    # Best first, so a later (worse) line never paints over a better one.
    for depth, arm in enumerate(reversed(order)):
        g = series[arm]
        text = labels[arm]
        if arm in values:
            text = f"{text} ({values[arm]:.2f})"
        ax.plot(
            g["frames_b"],
            g[key],
            lw=LW_ARM,
            marker="o",
            ms=MS_ARM,
            mew=0,
            zorder=3 + depth * 0.01,
            label=text,
            solid_capstyle="round",
            **line_kwargs(group.arms[arm], colors[arm]),
        )
    ax.axvline(JOIN_B, color=INK_SOFT, lw=0.6, ls=(0, (2, 3)), alpha=0.7, zorder=1)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == "log":
        ax.set_xlim(0.0085, SCREEN_B * 1.08)
        ax.set_xticks(X_TICKS_LOG)
        ax.set_xticklabels([f"{t:g}" for t in X_TICKS_LOG])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xlim(-0.03, SCREEN_B * 1.02)
    if key == "success_rate":
        ax.set_ylim(0, 1.0)
    ax.grid(True, which="major", alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("environment frames (B)")
    if yscale == "log":
        ticks = Y_TICKS_LOG[key]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(ticks[0] * 0.9, ticks[-1] * 1.15)


def style_legend(ax, group: Group) -> None:
    if not group.style_legend:
        return
    used = {st.style for st in group.arms.values()}
    handles = [
        Line2D([], [], color=INK_SOFT, lw=1.3, linestyle=STYLES[s])
        for s in group.style_legend
        if s in used
    ]
    names = [n for s, n in group.style_legend.items() if s in used]
    leg = ax.legend(
        handles,
        names,
        loc="lower right",
        frameon=False,
        title=None,
        handlelength=2.6,
        borderaxespad=0.4,
    )
    ax.add_artist(leg)


def figure_for_group(
    group: Group,
    series: dict[str, pd.DataFrame],
    labels: dict[str, str],
    xscale: str,
    out_dir: Path,
    join: dict,
    color_by: str,
    thin_every: int,
) -> list[Path]:
    # Constrained layout places the outside legend and the title without a
    # hand-tuned gap, whatever the legend's row count.
    fig = plt.figure(figsize=(7.16, 2.35), layout="constrained")  # IEEE double column
    axes = fig.subplots(1, 3)
    order, colors, values = arm_colors(group, series, color_by)
    for ax, (key, ylabel, yscale) in zip(axes, PANELS):
        draw_panel(
            ax,
            key,
            ylabel,
            yscale,
            series,
            group,
            labels,
            xscale,
            order,
            colors,
            values,
        )
    style_legend(axes[0], group)
    handles, names = axes[0].get_legend_handles_labels()
    # Hub first, then rank order (the axis was painted worst-first).
    handles, names = [handles[0]] + handles[1:][::-1], [names[0]] + names[1:][::-1]
    # Long ranked labels: four columns keep the legend inside the 7.16 in width.
    ncol = 4 if len(names) > 6 else min(6, len(names))
    fig.legend(
        handles,
        names,
        loc="outside upper center",
        ncol=ncol,
        # The group title rides on the legend, so the two can never collide.
        title=group.title,
        title_fontproperties={"weight": "semibold", "size": 9.5},
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.text(
        0.995,
        -0.02,
        f"bones_testbed4096_v1, DR off, seed 0. Dense rerun left of the rule at "
        f"{JOIN_B:g}B (median SR gap at the join {join['median_abs_sr_gap']:.3f})."
        + (" Legend: success rate at 2B, best first." if color_by == "rank" else "")
        + (
            f" Points thinned {thin_every}x in {THIN_LO:g}-{THIN_HI:g}B."
            if thin_every > 1
            else ""
        ),
        ha="right",
        va="top",
        fontsize=6.4,
        color=INK_SOFT,
    )
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"star_v2_convergence_{group.key}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def overview_figure(
    series: dict[str, pd.DataFrame],
    labels: dict[str, str],
    xscale: str,
    out_dir: Path,
    color_by: str,
) -> list[Path]:
    """Success rate only, one small panel per group, hub repeated everywhere."""
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.3), sharex=True, sharey=True)
    for ax, group in zip(axes.flat, GROUPS):
        hub = series[HUB]
        ax.plot(hub["frames_b"], hub["success_rate"], color=INK, lw=LW_HUB, zorder=6)
        order, colors, _ = arm_colors(group, series, color_by)
        for depth, arm in enumerate(reversed(order)):
            g = series[arm]
            ax.plot(
                g["frames_b"],
                g["success_rate"],
                lw=LW_ARM,
                zorder=3 + depth * 0.01,
                **line_kwargs(group.arms[arm], colors[arm]),
            )
        ax.axvline(JOIN_B, color=INK_SOFT, lw=0.6, ls=(0, (2, 3)), alpha=0.7, zorder=1)
        ax.set_xscale(xscale)
        if xscale == "log":
            ax.set_xlim(0.0085, SCREEN_B * 1.08)
            ax.set_xticks(X_TICKS_LOG)
            ax.set_xticklabels([f"{t:g}" for t in X_TICKS_LOG])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.9)
        ax.set_title(group.title, fontsize=8.2, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("environment frames (B)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Success rate")
    fig.text(
        0.005,
        -0.01,
        f"{labels[HUB]} in navy in every panel; the group's arms run deep blue "
        "(best at 2B) to gray-blue (worst). Labels in the per-group figures."
        if color_by == "rank"
        else f"{labels[HUB]} in navy in every panel; colored lines are the "
        "group's arms (see the per-group figures for labels).",
        fontsize=6.6,
        color=INK_SOFT,
    )
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"star_v2_convergence_overview.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--points", type=Path, default=POINTS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--xscale",
        choices=("log", "linear"),
        default="log",
        help="log spreads the dense early phase; linear matches the tables' axis",
    )
    ap.add_argument("--max-b", type=float, default=SCREEN_B)
    ap.add_argument(
        "--thin-every",
        type=int,
        default=THIN_EVERY,
        help=f"draw every N-th point between {THIN_LO:g}B and {THIN_HI:g}B; 1 draws all",
    )
    ap.add_argument(
        "--color-by",
        choices=("rank", "family"),
        default="rank",
        help="rank: one blue ramp ordered by success rate at the screen budget; "
        "family: six categorical hues, one per arm family",
    )
    ap.add_argument(
        "--groups",
        nargs="*",
        default=None,
        help="subset of group keys: " + " ".join(g.key for g in GROUPS),
    )
    args = ap.parse_args(argv)

    df = load_points(args.points)
    font = install_font()
    apply_style(font)

    excluded = excluded_arms(df)
    join = join_stats(df)
    labels = dict(zip(df["arm"], df["paper_label"]))
    labels.setdefault(HUB, "Ours")

    kept = df[~df["arm"].isin(set(excluded) | {"untrained"})]
    series_df = joined_series(kept, args.max_b)
    series = {
        arm: thin_series(g.reset_index(drop=True), THIN_LO, THIN_HI, args.thin_every)
        for arm, g in series_df.groupby("arm")
    }
    series_df = pd.concat(series.values(), ignore_index=True)
    if HUB not in series:
        sys.exit("[FATAL] hub has no rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [g for g in GROUPS if not args.groups or g.key in set(args.groups)]
    if not wanted:
        sys.exit(f"[FATAL] no group matches {args.groups}")

    written: list[Path] = []
    for group in wanted:
        present = [a for a in group.arms if a in series]
        if not present:
            print(f"[skip] {group.key}: every arm excluded")
            continue
        written += figure_for_group(
            group,
            series,
            labels,
            args.xscale,
            args.out_dir,
            join,
            args.color_by,
            args.thin_every,
        )
    written += overview_figure(series, labels, args.xscale, args.out_dir, args.color_by)

    # Exactly the drawn points, with the group and style keys, so a reader can
    # rebuild any panel without this script.
    arm_group = {a: g.key for g in GROUPS for a in g.arms}
    arm_style = {a: st for g in GROUPS for a, st in g.arms.items()}
    data = series_df.copy()
    data["group"] = data["arm"].map(arm_group).fillna("hub")
    rank_of: dict[str, int] = {}
    color_of: dict[str, str] = {HUB: INK}
    for group in GROUPS:
        order, colors, _ = arm_colors(group, series, args.color_by)
        color_of.update(colors)
        rank_of.update({a: i + 1 for i, a in enumerate(order)})
    data["rank_in_group"] = data["arm"].map(rank_of).astype("Int64")
    data["hue"] = data["arm"].map(color_of)
    data["linestyle"] = data["arm"].map(
        lambda a: "solid" if a == HUB else arm_style[a].style
    )
    cols = [
        "group",
        "arm",
        "paper_label",
        "run",
        "frames",
        "frames_b",
        "success_rate",
        "mpjpe_local_mm",
        "mpjpe_global_mm",
        "num_successful_envs",
        "rank_in_group",
        "hue",
        "linestyle",
    ]
    data_path = args.out_dir / "star_v2_convergence_figure_data.csv"
    data[[c for c in cols if c in data.columns]].to_csv(data_path, index=False)
    meta_path = args.out_dir / "star_v2_convergence_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "points_csv": str(args.points),
                "xscale": args.xscale,
                "max_b": args.max_b,
                "join_b": JOIN_B,
                "font": font,
                "ink": INK,
                "hues": HUES,
                "color_by": args.color_by,
                "thin": {"lo_b": THIN_LO, "hi_b": THIN_HI, "every": args.thin_every},
                "ramp_best_to_worst": RAMP,
                "rank_in_group": rank_of,
                "excluded_arms": excluded,
                "join_stats": join,
                "board": "bones_testbed4096_v1",
                "randomization": "none",
                "seed": 0,
            },
            indent=2,
        )
    )

    print(f"font: {font}")
    print("excluded arms:")
    for arm, why in excluded.items():
        print(f"  {arm}: {why}")
    print(
        f"join: {join['arms']} arms, median |dSR| {join['median_abs_sr_gap']:.4f}, "
        f"p90 {join['p90_abs_sr_gap']:.4f}, max {join['max_abs_sr_gap']:.3f} ({join['max_gap_arm']})"
    )
    print(f"wrote {data_path} ({len(data)} rows)")
    print(f"wrote {meta_path}")
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
