"""Render the resolved records and spec into one self-contained HTML page.

Design rules this file follows, in priority order:

1. **A number never appears without its provenance.** Every row carries a
   disclosure with the summary path, the tracker digest, the episode and motion
   counts, and the protocol flags.
2. **A difference smaller than the evaluation noise is not a result.** Deltas
   inside the configured band render as "unresolved", not as a win.
3. **Nothing is fetched.** Charts are server-rendered SVG, equations are
   MathML, the only script is table sorting and detail toggling. The page opens
   from a file:// URL with no network.
"""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from math import isfinite

from imitation_experiments.reporting.math_cards import MethodCard, cards_for
from imitation_experiments.reporting.records import EvalRecord
from imitation_experiments.reporting.spec import Ablation, ReportSpec, RunRef


class ComparisonMismatch(ValueError):
    """A report table attempted a comparison across incompatible evidence."""


_STYLE = """
:root {
  color-scheme: light dark;
  --paper: #fbfaf8;
  --panel: #ffffff;
  --ink: #14181d;
  --ink-soft: #565f6b;
  --line: #e2e0da;
  --line-hard: #c9c6bd;
  --accent: #2f6f8f;
  --good: #2c6e49;
  --warn: #8a5a00;
  --bad: #9b3227;
  --unresolved: #6b6b6b;
  --chip: #f0eee8;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #12151a;
    --panel: #181c22;
    --ink: #e9ecef;
    --ink-soft: #9aa4b0;
    --line: #262c34;
    --line-hard: #39414b;
    --accent: #7fc0e0;
    --good: #74c69d;
    --warn: #e0b166;
    --bad: #ec8c80;
    --unresolved: #97a0aa;
    --chip: #202630;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px 96px; }
header.masthead { border-bottom: 1px solid var(--line); padding: 40px 0 24px; margin-bottom: 8px; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.02em; }
.sub { color: var(--ink-soft); margin: 0 0 14px; max-width: 62ch; }
.stamp { font-family: var(--mono); font-size: 11.5px; color: var(--ink-soft); }
nav.toc {
  position: sticky; top: 0; z-index: 20;
  display: flex; gap: 6px; flex-wrap: wrap;
  padding: 10px 0; margin-bottom: 12px;
  background: color-mix(in srgb, var(--paper) 92%, transparent);
  backdrop-filter: blur(6px);
  border-bottom: 1px solid var(--line);
}
nav.toc a {
  font-size: 12.5px; text-decoration: none; color: var(--ink-soft);
  padding: 5px 10px; border-radius: 999px; border: 1px solid transparent;
}
nav.toc a:hover { color: var(--ink); border-color: var(--line-hard); }
section { padding-top: 34px; }
h2 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
h3 { font-size: 15px; margin: 26px 0 8px; }
.section-note { color: var(--ink-soft); font-size: 13.5px; margin: 0 0 18px; max-width: 74ch; }
.headline {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 1px; background: var(--line); border: 1px solid var(--line);
  border-radius: 10px; overflow: hidden; margin: 16px 0 8px;
}
.headline > div { background: var(--panel); padding: 16px 18px; }
.headline .k { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-soft); }
.headline .v { font-size: 27px; font-variant-numeric: tabular-nums; margin-top: 4px; letter-spacing: -0.02em; }
.headline .v small { font-size: 14px; color: var(--ink-soft); font-weight: 400; }
.headline .n { font-size: 12px; color: var(--ink-soft); margin-top: 3px; }
table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
th, td { padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
thead th {
  font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-soft); font-weight: 600; cursor: pointer; user-select: none;
  position: sticky; top: 0; background: var(--panel); z-index: 1;
}
thead th::after { content: " \\2195"; opacity: 0.25; }
thead th.nosort { cursor: default; }
thead th.nosort::after { content: ""; }
tbody tr:last-child td { border-bottom: none; }
tbody tr.best td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.name { white-space: normal; min-width: 190px; }
.chip {
  display: inline-block; font-size: 11px; padding: 2px 7px; border-radius: 999px;
  background: var(--chip); color: var(--ink-soft); margin-right: 4px; white-space: nowrap;
}
.chip.good { color: var(--good); }
.chip.warn { color: var(--warn); }
.chip.bad { color: var(--bad); }
.delta.good { color: var(--good); }
.delta.bad { color: var(--bad); }
.delta.flat { color: var(--unresolved); }
details.prov { margin: 0; }
details.prov > summary {
  cursor: pointer; font-size: 11.5px; color: var(--ink-soft); list-style: none;
}
details.prov > summary::-webkit-details-marker { display: none; }
details.prov > summary::before { content: "\\25B8 "; }
details.prov[open] > summary::before { content: "\\25BE "; }
.provbody {
  font-family: var(--mono); font-size: 11px; line-height: 1.7;
  color: var(--ink-soft); padding: 8px 0 2px; white-space: normal;
}
.provbody b { color: var(--ink); font-weight: 600; }
.callout {
  border-left: 3px solid var(--line-hard); padding: 10px 0 10px 14px;
  margin: 14px 0; color: var(--ink-soft); font-size: 13.5px;
}
.callout.verdict { border-left-color: var(--accent); }
.callout.caveat { border-left-color: var(--warn); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
.card {
  border: 1px solid var(--line); border-radius: 10px; background: var(--panel);
  padding: 18px 20px 16px;
}
.card .kicker { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--accent); }
.card h4 { margin: 4px 0 8px; font-size: 16.5px; letter-spacing: -0.01em; }
.card p { margin: 0 0 12px; font-size: 13.5px; color: var(--ink-soft); }
.eq { margin: 12px 0; overflow-x: auto; }
.eq math { font-size: 17px; }
.eq .cap { font-size: 12px; color: var(--ink-soft); margin-top: 2px; }
.where { font-size: 12px; color: var(--ink-soft); margin: 6px 0 0; padding-left: 0; list-style: none; }
.where li { margin: 2px 0; }
.where code { font-family: var(--mono); color: var(--ink); }
.shapes { width: 100%; font-size: 12.5px; margin-top: 10px; }
.shapes td { padding: 4px 0; border: none; white-space: normal; }
.shapes td:nth-child(2) { font-family: var(--mono); text-align: right; color: var(--ink); }
.shapes td:nth-child(3) { color: var(--ink-soft); padding-left: 12px; }
.src { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); margin-top: 12px; word-break: break-all; }
figure { margin: 18px 0; }
figcaption { font-size: 12px; color: var(--ink-soft); margin-top: 6px; }
svg { max-width: 100%; height: auto; display: block; }
footer { margin-top: 60px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 12px; color: var(--ink-soft); }
"""

_SCRIPT = """
document.querySelectorAll('table[data-sortable] thead th:not(.nosort)').forEach((th, index) => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const body = table.tBodies[0];
    const groups = [];
    let current = null;
    for (const row of Array.from(body.rows)) {
      if (row.classList.contains('detail')) {
        if (current) current.push(row);
      } else {
        current = [row];
        groups.push(current);
      }
    }
    const ascending = th.dataset.dir !== 'asc';
    th.dataset.dir = ascending ? 'asc' : 'desc';
    groups.sort((a, b) => {
      const cellA = a[0].cells[index], cellB = b[0].cells[index];
      const keyA = cellA ? cellA.dataset.sort ?? cellA.textContent.trim() : '';
      const keyB = cellB ? cellB.dataset.sort ?? cellB.textContent.trim() : '';
      const numA = parseFloat(keyA), numB = parseFloat(keyB);
      const bothNumeric = !Number.isNaN(numA) && !Number.isNaN(numB);
      const order = bothNumeric ? numA - numB : keyA.localeCompare(keyB);
      return ascending ? order : -order;
    });
    groups.flat().forEach(row => body.appendChild(row));
  });
});
"""


def _e(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None or not isfinite(float(value)):
        return '<span class="chip">n/a</span>'
    return f"{float(value):.{digits}f}{suffix}"


def _sortkey(value: float | None) -> str:
    if value is None or not isfinite(float(value)):
        return ' data-sort="999999"'
    return f' data-sort="{float(value):.6f}"'


def _delta(value: float | None, base: float | None, band_pct: float) -> str:
    """Render a relative difference, marking anything inside the noise band."""
    if value is None or base is None or not base:
        return '<span class="chip">n/a</span>'
    relative = 100.0 * (value - base) / base
    if abs(relative) < band_pct:
        return (
            f'<span class="delta flat" title="Inside the ~{band_pct:g}% evaluation '
            f'noise band; directional only">{relative:+.1f}%</span>'
            ' <span class="chip">unresolved</span>'
        )
    tone = "good" if relative < 0 else "bad"
    return f'<span class="delta {tone}">{relative:+.1f}%</span>'


def _provenance(record: EvalRecord, extra: dict[str, str] | None = None) -> str:
    rows: list[tuple[str, str]] = [
        ("summary", record.summary_path),
        ("task", record.task or "-"),
        ("episodes", f"{record.episode_count} over {record.motion_count} motions"),
    ]
    if record.episodes_per_motion:
        rows.append(("balance", f"{record.episodes_per_motion} episodes per motion"))
    else:
        rows.append(
            ("balance", "UNBALANCED - motion-average differs from episode-mean")
        )
    rows += [
        ("tracker", record.tracker_checkpoint or "-"),
        ("tracker sha256", record.tracker_sha256 or "-"),
        ("tracker frozen", str(record.tracker_frozen)),
    ]
    if record.planner_checkpoint:
        rows += [
            ("planner", record.planner_checkpoint),
            ("planner update", str(record.planner_update)),
            ("latent dim", str(record.planner_latent_dim)),
            ("action horizon", str(record.planner_action_horizon)),
            ("consumption", str(record.planner_consumption)),
            (
                "head p50 latency",
                f"{record.planner_latency_p50_ms:.0f} ms"
                if record.planner_latency_p50_ms
                else "-",
            ),
            ("published vs oracle cos", _num(record.published_vs_oracle_z_cosine, 3)),
        ]
        if record.planner_temporal_ensemble is not None:
            rows += [
                ("temporal ensemble", record.planner_temporal_ensemble),
                (
                    "ensemble decay",
                    "n/a"
                    if record.planner_temporal_ensemble_decay is None
                    else f"{record.planner_temporal_ensemble_decay:g}",
                ),
                ("ODE steps", str(record.planner_inference_steps)),
                (
                    "samples per publication",
                    str(record.planner_samples_per_publication),
                ),
                ("consume slots", str(record.planner_consume_slots)),
            ]
        else:
            rows.append(
                (
                    "inference knobs",
                    "NOT RECORDED - evaluated before 2026-08-16; ensembling, "
                    "sample count, and ODE steps come from the report spec",
                )
            )
    rows += [
        (
            "protocol",
            f"max_steps={record.max_steps}, episode_length_s="
            f"{record.episode_length_s}, fall_height_m={record.fall_height_m}",
        ),
        ("termination", str(record.termination_profile)),
        ("push disabled", str(record.push_disabled)),
        ("seed", str(record.seed)),
        ("done_rate", _num(record.done_rate, 3)),
        (
            "valid transitions",
            f"{record.valid_transition_count:,}"
            if record.valid_transition_count
            else "-",
        ),
        ("MPJPE, episode mean", _num(record.mpjpe_mm, 2, " mm")),
        (
            "MPJPE, transition weighted",
            _num(record.mpjpe_mm_transition_weighted, 2, " mm"),
        ),
        ("MPJPE, successful only", _num(record.mpjpe_mm_successful_only, 2, " mm")),
    ]
    for key, value in (extra or {}).items():
        rows.append((key, value))
    body = "<br>".join(f"<b>{_e(k)}</b> {_e(v)}" for k, v in rows)
    return (
        '<details class="prov"><summary>provenance</summary>'
        f'<div class="provbody">{body}</div></details>'
    )


def _svg_history(points: list[tuple[str, str, float, float]]) -> str:
    """Step chart of the best-so-far tracker MPJPE against its date."""
    if len(points) < 2:
        return ""
    width, height = 860, 240
    left, right, top, bottom = 56, 18, 20, 46
    values = [value for _, _, value, _ in points]
    low, high = min(values), max(values)
    span = max(high - low, 1e-6)
    low -= span * 0.18
    high += span * 0.12
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_at(index: int) -> float:
        if len(points) == 1:
            return left + plot_w / 2
        return left + plot_w * index / (len(points) - 1)

    def y_at(value: float) -> float:
        return top + plot_h * (1.0 - (value - low) / (high - low))

    grid, ticks = [], 4
    for step in range(ticks + 1):
        value = low + (high - low) * step / ticks
        y = y_at(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="var(--line)" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
            f'fill="var(--ink-soft)">{value:.0f}</text>'
        )

    path: list[str] = []
    for index, (_, _, value, _) in enumerate(points):
        x, y = x_at(index), y_at(value)
        path.append(("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}")
        if index + 1 < len(points):
            path.append(f"L{x_at(index + 1):.1f},{y:.1f}")

    marks = []
    for index, (date, name, value, survival) in enumerate(points):
        x, y = x_at(index), y_at(value)
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="var(--accent)"/>'
            f"<title>{_e(name)} - {value:.2f} mm, fall-free {survival:.3f} ({_e(date)})</title>"
            f'<text x="{x:.1f}" y="{y - 11:.1f}" text-anchor="middle" font-size="10.5" '
            f'font-weight="600" fill="var(--ink)">{value:.1f}</text>'
            f'<text x="{x:.1f}" y="{height - 26:.1f}" text-anchor="middle" font-size="10" '
            f'fill="var(--ink-soft)">{_e(date[5:])}</text>'
            f'<text x="{x:.1f}" y="{height - 13:.1f}" text-anchor="middle" font-size="9.5" '
            f'fill="var(--ink-soft)">{_e(name[:22])}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Best low-level tracker MPJPE over time">'
        f"{''.join(grid)}"
        f'<path d="{" ".join(path)}" fill="none" stroke="var(--accent)" '
        'stroke-width="2" stroke-linejoin="round"/>'
        f"{''.join(marks)}"
        f'<text x="{left - 8}" y="{top - 6}" text-anchor="end" font-size="10" '
        'fill="var(--ink-soft)">mm</text>'
        "</svg>"
    )


def _svg_motion_bars(record: EvalRecord, ceiling: EvalRecord | None) -> str:
    """Per-motion MPJPE, sorted worst first, with the oracle ceiling behind it."""
    scores = [score for score in record.per_motion if isfinite(score.mpjpe_mm)]
    if not scores:
        return ""
    scores.sort(key=lambda score: score.mpjpe_mm, reverse=True)
    ceiling_by_motion = {
        score.motion_name: score.mpjpe_mm
        for score in (ceiling.per_motion if ceiling else ())
        if isfinite(score.mpjpe_mm)
    }
    row_h, label_w, width = 17, 250, 900
    height = row_h * len(scores) + 26
    bar_w = width - label_w - 70
    top_value = max(
        max(score.mpjpe_mm for score in scores),
        max(ceiling_by_motion.values(), default=0.0),
    )
    rows = []
    for index, score in enumerate(scores):
        y = 20 + index * row_h
        length = bar_w * score.mpjpe_mm / top_value
        parts = [
            f'<text x="{label_w - 8}" y="{y + 9}" text-anchor="end" font-size="10" '
            f'fill="var(--ink-soft)">{_e(score.motion_name[:38])}</text>'
        ]
        ceiling_value = ceiling_by_motion.get(score.motion_name)
        if ceiling_value is not None:
            parts.append(
                f'<rect x="{label_w}" y="{y + 2}" '
                f'width="{bar_w * ceiling_value / top_value:.1f}" height="{row_h - 5}" '
                'fill="var(--ink-soft)" opacity="0.22"/>'
            )
        tone = "var(--bad)" if score.fall_free_rate < 1.0 else "var(--accent)"
        parts.append(
            f'<rect x="{label_w}" y="{y + 4}" width="{length:.1f}" height="{row_h - 9}" '
            f'fill="{tone}" opacity="0.85"><title>{_e(score.motion_name)}: '
            f"{score.mpjpe_mm:.1f} mm, fall-free {score.fall_free_rate:.2f}, "
            f"n={score.episode_count}</title></rect>"
        )
        parts.append(
            f'<text x="{label_w + length + 6:.1f}" y="{y + 9}" font-size="10" '
            f'fill="var(--ink)">{score.mpjpe_mm:.1f}</text>'
        )
        rows.append("".join(parts))
    legend = (
        f'<rect x="{label_w}" y="4" width="10" height="8" fill="var(--accent)" opacity="0.85"/>'
        f'<text x="{label_w + 15}" y="12" font-size="10" fill="var(--ink-soft)">arm</text>'
    )
    if ceiling_by_motion:
        legend += (
            f'<rect x="{label_w + 52}" y="4" width="10" height="8" fill="var(--ink-soft)" '
            'opacity="0.22"/>'
            f'<text x="{label_w + 67}" y="12" font-size="10" fill="var(--ink-soft)">'
            "oracle ceiling</text>"
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Per-motion MPJPE for {_e(record.label)}">'
        f"{legend}{''.join(rows)}</svg>"
    )


def restricted_mpjpe(ceiling: EvalRecord, arm: EvalRecord) -> tuple[float | None, int]:
    """Score an oracle ceiling on the arm's exact ordered episode identities.

    An oracle run usually covers more motions than the arm it bounds. The
    2026-08-13 campaign compared a 28-motion planner against a 30-motion oracle
    and, separately, an episode-mean planner number against a
    transition-weighted oracle number. Both make the reported gap wrong in a
    direction nobody can see from the table. Restricting the ceiling to the
    arm's own motion set, under the arm's own reduction, removes both.
    """
    if not ceiling.protocol_pinned or not arm.protocol_pinned:
        return None, 0
    if ceiling.protocol_hash != arm.protocol_hash:
        raise ComparisonMismatch(
            "Restricted oracle comparison requires one protocol hash: "
            f"{ceiling.protocol_hash} != {arm.protocol_hash}."
        )
    ceiling_by_episode = {
        episode.identity: episode
        for episode in ceiling.episodes
        if episode.identity is not None
    }
    arm_identities = [episode.identity for episode in arm.episodes]
    if not arm_identities or any(identity is None for identity in arm_identities):
        raise ComparisonMismatch(
            "Restricted oracle comparison requires complete episode identities."
        )
    missing = [identity for identity in arm_identities if identity not in ceiling_by_episode]
    if missing:
        raise ComparisonMismatch(
            "Arm board is not an exact subset of oracle board; missing episode keys "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}."
        )
    matched = [
        ceiling_by_episode[identity].mpjpe_mm
        for identity in arm_identities
        if identity is not None
    ]
    if any(value is None or not isfinite(value) for value in matched):
        raise ComparisonMismatch("Oracle subset contains an episode without MPJPE.")
    motion_count = len(
        {
            ceiling_by_episode[identity].motion_name
            for identity in arm_identities
            if identity is not None
        }
    )
    return sum(float(value) for value in matched if value is not None) / len(matched), motion_count


def _status_chip(record: EvalRecord, reference: EvalRecord | None) -> str:
    """Flag the conditions that keep a row preliminary or unmatched.

    A protocol difference against the table's control is the one thing a reader
    cannot see from the numbers, so it is marked on the row rather than left to
    the provenance disclosure.
    """
    flags: list[str] = []
    if not record.protocol_pinned:
        flags.append('<span class="chip warn">protocol unpinned</span>')
    if not record.is_balanced:
        flags.append('<span class="chip warn">unbalanced</span>')
    if record.done_rate is not None and record.done_rate < 0.999:
        flags.append(f'<span class="chip warn">done {record.done_rate:.3f}</span>')
    if reference is not None and record is not reference:
        if (
            record.protocol_hash is not None
            and reference.protocol_hash is not None
            and record.protocol_hash != reference.protocol_hash
        ):
            flags.append('<span class="chip warn">protocol mismatch</span>')
        if (
            record.board_hash is not None
            and reference.board_hash is not None
            and record.board_hash != reference.board_hash
        ):
            flags.append('<span class="chip warn">board mismatch</span>')
        if record.max_steps != reference.max_steps:
            flags.append(
                f'<span class="chip warn">step cap {record.max_steps} vs '
                f"{reference.max_steps}</span>"
            )
        if record.task != reference.task:
            flags.append(f'<span class="chip warn">task {_e(record.task)}</span>')
        if record.motion_count != reference.motion_count:
            flags.append(
                f'<span class="chip warn">{record.motion_count} motions vs '
                f"{reference.motion_count}</span>"
            )
    flags.append('<span class="chip">1 seed</span>')
    return "".join(flags)


def arm_attributes(ref: RunRef, record: EvalRecord) -> dict[str, str]:
    """Merge the spec's declared attributes with what the artifact now records.

    The evaluation began writing its inference knobs into
    `metadata.gr00t_planner` on 2026-08-16. Where the artifact has them, they
    win: a hand-written YAML attribute cannot contradict the run it describes.
    Older summaries carry nothing, so the spec's declaration still stands.
    """
    merged = dict(ref.attributes)
    ensemble = record.planner_temporal_ensemble
    if ensemble is not None:
        decay = record.planner_temporal_ensemble_decay
        merged["ensemble"] = (
            "none" if ensemble == "none" or decay is None else f"{ensemble} {decay:g}"
        )
    if record.planner_samples_per_publication not in (None, 1):
        merged["samples"] = str(record.planner_samples_per_publication)
    if record.planner_inference_steps not in (None, -1):
        merged["ODE steps"] = str(record.planner_inference_steps)
    return merged


def _metric_row(
    ref: RunRef,
    record: EvalRecord,
    *,
    baseline: EvalRecord | None,
    ceiling: EvalRecord | None,
    band_pct: float,
    best: bool,
    show_ceiling: bool,
) -> str:
    attributes = "".join(
        f'<span class="chip">{_e(key)}: {_e(value)}</span>'
        for key, value in arm_attributes(ref, record).items()
    )
    ceiling_value, ceiling_motions = (
        restricted_mpjpe(ceiling, record) if ceiling else (None, 0)
    )
    gap = (
        record.mpjpe_mm - ceiling_value
        if ceiling_value is not None and record.mpjpe_mm is not None
        else None
    )
    extra: dict[str, str] = {}
    if ceiling is not None:
        extra["ceiling run"] = ceiling.summary_path
        extra["ceiling, matched motions"] = (
            f"{ceiling_value:.2f} mm over {ceiling_motions} motions"
            if ceiling_value is not None
            else "no shared motions"
        )
        extra["ceiling, own full set"] = (
            f"{ceiling.mpjpe_mm:.2f} mm over {ceiling.motion_count} motions"
            if ceiling.mpjpe_mm is not None
            else "n/a"
        )
    cells = [
        f'<td class="name" data-sort="{_e(ref.name)}"><strong>{_e(ref.name)}</strong>'
        f"<br>{attributes}{_status_chip(record, baseline)}</td>",
        f'<td class="num"{_sortkey(record.mpjpe_mm)}>{_num(record.mpjpe_mm, 2)}</td>',
        f'<td class="num"{_sortkey(record.fall_free_rate)}>'
        f"{_num(record.fall_free_rate, 3)}</td>",
    ]
    if show_ceiling:
        cells += [
            f'<td class="num"{_sortkey(ceiling_value)}>{_num(ceiling_value, 2)}</td>',
            f'<td class="num"{_sortkey(gap)}>{_num(gap, 2)}</td>',
        ]
    is_control = record is baseline
    cells += [
        '<td class="num">'
        + (
            '<span class="chip">control</span>'
            if is_control
            else (
                _delta(record.mpjpe_mm, baseline.mpjpe_mm, band_pct)
                if baseline is not None
                and record.protocol_pinned
                and baseline.protocol_pinned
                and record.protocol_hash == baseline.protocol_hash
                and record.board_hash == baseline.board_hash
                else '<span class="chip warn">no comparable delta</span>'
            )
        )
        + "</td>",
        f"<td>{_provenance(record, extra)}</td>",
    ]
    classes = ' class="best"' if best else ""
    main = f"<tr{classes}>{''.join(cells)}</tr>"
    chart = _svg_motion_bars(record, ceiling)
    if not chart:
        return main
    detail = (
        f'<tr class="detail"><td colspan="{7 if show_ceiling else 5}">'
        f'<details class="prov"><summary>per-motion breakdown '
        f"({len(record.per_motion)} motions)</summary>"
        f"<figure>{chart}<figcaption>Sorted worst first. A red bar marks a motion "
        "with at least one fall. The grey bar behind it is the oracle ceiling for "
        "the same motion on the same tracker, so the visible gap is what the "
        "planner costs.</figcaption></figure></details></td></tr>"
    )
    return main + detail


def _metric_table(
    rows: list[tuple[RunRef, EvalRecord, EvalRecord | None]],
    *,
    baseline: EvalRecord | None,
    band_pct: float,
) -> str:
    pinned = [record for _, record, _ in rows if record.protocol_pinned]
    protocol_hashes = {record.protocol_hash for record in pinned}
    if len(protocol_hashes) > 1:
        protocols = {
            record.protocol_id: record.protocol_hash for record in pinned
        }
        raise ComparisonMismatch(
            f"One comparison table contains multiple protocol hashes: {protocols}."
        )
    board_hashes = {record.board_hash for record in pinned}
    if len(board_hashes) > 1:
        boards = {record.board_id: record.board_hash for record in pinned}
        raise ComparisonMismatch(
            f"One comparison table contains multiple board hashes: {boards}."
        )
    scored = [record.mpjpe_mm for _, record, _ in rows if record.mpjpe_mm is not None]
    best_value = min(scored) if scored else None
    # The tracker scoreboard rows ARE ceilings, so the two ceiling columns would
    # be empty there. Show them only when at least one row declares one.
    show_ceiling = any(ceiling is not None for _, _, ceiling in rows)
    body = "".join(
        _metric_row(
            ref,
            record,
            baseline=baseline,
            ceiling=ceiling,
            band_pct=band_pct,
            best=record.mpjpe_mm == best_value,
            show_ceiling=show_ceiling,
        )
        for ref, record, ceiling in rows
    )
    ceiling_headers = (
        "<th title=\"Same tracker under oracle commands, scored on this arm's own "
        'motions and reduced the same way">Ceiling, matched</th>'
        "<th>Gap over ceiling</th>"
        if show_ceiling
        else ""
    )
    return (
        '<div class="scroll"><table data-sortable><thead><tr>'
        "<th>Arm</th><th>MPJPE (mm)</th><th>Fall-free</th>"
        f'{ceiling_headers}<th>vs control</th><th class="nosort">Audit</th>'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _method_card(card: MethodCard) -> str:
    equations = ""
    for equation in card.equations:
        where = ""
        if equation.where:
            where = (
                '<ul class="where">'
                + "".join(
                    f"<li><code>{_e(symbol)}</code> &mdash; {_e(text)}</li>"
                    for symbol, text in equation.where
                )
                + "</ul>"
            )
        caption = (
            f'<div class="cap">{_e(equation.caption)}</div>' if equation.caption else ""
        )
        equations += (
            f'<div class="eq" role="math" aria-label="{_e(equation.plain)}">'
            f"{equation.mathml}{caption}{where}</div>"
        )
    shapes = ""
    if card.shapes:
        shapes = (
            '<table class="shapes">'
            + "".join(
                f"<tr><td>{_e(name)}</td><td>{_e(value)}</td><td>{_e(note)}</td></tr>"
                for name, value, note in card.shapes
            )
            + "</table>"
        )
    caveat = (
        f'<div class="callout caveat">{_e(card.caveat)}</div>' if card.caveat else ""
    )
    return (
        f'<article class="card" id="card-{_e(card.id)}">'
        f'<div class="kicker">{_e(card.kicker)}</div>'
        f"<h4>{_e(card.title)}</h4><p>{_e(card.blurb)}</p>"
        f"{equations}{shapes}{caveat}"
        f'<div class="src">{_e(card.source)}</div></article>'
    )


def _ablation_block(
    ablation: Ablation,
    resolved: dict[str, tuple[RunRef, EvalRecord, EvalRecord | None]],
    band_pct: float,
) -> str:
    rows = [resolved[row.id] for row in ablation.rows]
    baseline = resolved[ablation.baseline][1]
    verdict = (
        f'<div class="callout verdict"><strong>Reading.</strong> {_e(ablation.verdict)}</div>'
        if ablation.verdict
        else ""
    )
    tone = {"verified": "good", "preliminary": "warn", "refuted": "bad"}.get(
        ablation.status, "warn"
    )
    return (
        f'<h3 id="ab-{_e(ablation.id)}">{_e(ablation.title)} '
        f'<span class="chip {tone}">{_e(ablation.status)}</span></h3>'
        f'<p class="section-note">{_e(ablation.question)}<br>'
        f"<em>Variable isolated:</em> {_e(ablation.variable)}. "
        f"<em>Control:</em> {_e(resolved[ablation.baseline][0].name)}.</p>"
        f"{_metric_table(rows, baseline=baseline, band_pct=band_pct)}"
        f"{verdict}"
    )


def render_html(
    spec: ReportSpec,
    records: dict[str, EvalRecord],
    *,
    git_describe: str = "",
) -> str:
    """Return the complete self-contained results page."""

    def resolve(ref: RunRef) -> tuple[RunRef, EvalRecord, EvalRecord | None]:
        return ref, records[ref.run], records.get(ref.ceiling) if ref.ceiling else None

    headline_ref, headline, headline_ceiling = resolve(spec.low_level_headline)

    history_points: list[tuple[str, str, float, float]] = []
    best_so_far = float("inf")
    for point in spec.low_level_history:
        record = records[point.run]
        if record.mpjpe_mm is None:
            continue
        best_so_far = min(best_so_far, record.mpjpe_mm)
        history_points.append(
            (point.date, point.name, best_so_far, record.fall_free_rate or float("nan"))
        )

    low_level_rows = [resolve(ref) for ref in spec.low_level_rows]
    planner_rows = [resolve(ref) for ref in spec.planner_rows]
    planner_baseline = next(
        (record for ref, record, _ in planner_rows if ref.id == spec.planner_baseline),
        None,
    )

    ablation_blocks = []
    for ablation in spec.ablations:
        resolved = {row.id: resolve(row) for row in ablation.rows}
        ablation_blocks.append(_ablation_block(ablation, resolved, spec.noise_band_pct))

    cards = "".join(_method_card(card) for card in cards_for(spec.method_cards))

    caveats = "".join(f"<li>{_e(text)}</li>" for text in spec.caveats)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    headline_cards = (
        f'<div><div class="k">Best tracker MPJPE</div>'
        f'<div class="v">{_num(headline.mpjpe_mm, 2)} <small>mm</small></div>'
        f'<div class="n">{_e(headline_ref.name)}</div></div>'
        f'<div><div class="k">Fall-free</div>'
        f'<div class="v">{_num(headline.fall_free_rate, 3)}</div>'
        f'<div class="n">fall-only termination, no push</div></div>'
        f'<div><div class="k">Protocol</div>'
        f'<div class="v">{headline.motion_count} <small>motions</small></div>'
        f'<div class="n">{headline.episode_count} episodes, '
        f"{headline.episodes_per_motion or '?'} per motion</div></div>"
        f'<div><div class="k">Tracker digest</div>'
        f'<div class="v" style="font-size:14px;font-family:var(--mono);word-break:break-all">'
        f"{_e((headline.tracker_sha256 or '-')[:16])}&hellip;</div>"
        f'<div class="n">strict restore, frozen</div></div>'
    )

    planner_best = min(
        (
            record.mpjpe_mm
            for _, record, _ in planner_rows
            if record.mpjpe_mm is not None
        ),
        default=None,
    )
    planner_headline = ""
    if planner_best is not None:
        best_ref, best_record, best_ceiling = min(
            (row for row in planner_rows if row[1].mpjpe_mm is not None),
            key=lambda row: row[1].mpjpe_mm,
        )
        ceiling_value = (
            restricted_mpjpe(best_ceiling, best_record)[0] if best_ceiling else None
        )
        gap = None if ceiling_value is None else best_record.mpjpe_mm - ceiling_value
        planner_headline = (
            f'<div class="headline">'
            f'<div><div class="k">Best planner MPJPE</div>'
            f'<div class="v">{_num(best_record.mpjpe_mm, 2)} <small>mm</small></div>'
            f'<div class="n">{_e(best_ref.name)}</div></div>'
            f'<div><div class="k">Fall-free</div>'
            f'<div class="v">{_num(best_record.fall_free_rate, 3)}</div>'
            f'<div class="n">{best_record.motion_count} motions, '
            f"{best_record.episode_count} episodes</div></div>"
            f'<div><div class="k">Cost over its own oracle</div>'
            f'<div class="v">{_num(gap, 2)} <small>mm</small></div>'
            f'<div class="n">same tracker, same protocol</div></div>'
            f'<div><div class="k">Head latency p50</div>'
            f'<div class="v">{_num(best_record.planner_latency_p50_ms, 0)} <small>ms</small></div>'
            f'<div class="n">{_e(best_record.planner_consumption or "-")} consumption</div></div>'
            f"</div>"
        )

    return f"""<title>{_e(spec.title)}</title>
<style>{_STYLE}</style>
<div class="wrap">
<header class="masthead">
  <h1>{_e(spec.title)}</h1>
  <p class="sub">{_e(spec.subtitle)}</p>
  <div class="stamp">generated {generated} &middot; spec updated {_e(spec.updated)}
  &middot; {_e(git_describe or "no git description")}
  &middot; noise band &plusmn;{spec.noise_band_pct:g}% relative</div>
</header>

<nav class="toc">
  <a href="#low-level">1. Low level</a>
  <a href="#planner">2. Planner + low level</a>
  <a href="#ablations">3. Ablations</a>
  <a href="#methods">4. Methods</a>
  <a href="#reading">Reading rules</a>
</nav>

<section id="low-level">
  <h2>1. Low-level policy &mdash; moving best</h2>
  <p class="section-note">The tracker ceiling under oracle commands. Every planner
  row later on this page is bounded below by one of these numbers, so this is the
  reference the interface work has to move first.</p>
  <div class="headline">{headline_cards}</div>
  {_provenance(headline)}
  <figure>{_svg_history(history_points)}
  <figcaption>Best tracker MPJPE achieved as of each date, monotone by
  construction. A flat segment means that date's run did not beat the standing
  best.</figcaption></figure>
  <h3>Tracker scoreboard</h3>
  {_metric_table(low_level_rows, baseline=headline, band_pct=spec.noise_band_pct)}
</section>

<section id="planner">
  <h2>2. Planner driving the low level</h2>
  <p class="section-note">Closed loop: a causal planner publishes commands into a
  frozen tracker. "Gap over ceiling" is the same tracker's oracle score subtracted
  from the arm, so it separates what the planner costs from what the tracker
  cannot do.</p>
  {planner_headline}
  {_metric_table(planner_rows, baseline=planner_baseline, band_pct=spec.noise_band_pct)}
</section>

<section id="ablations">
  <h2>3. Ablations</h2>
  <p class="section-note">One variable per block, against a named control. A
  difference inside the &plusmn;{spec.noise_band_pct:g}% band is marked unresolved
  rather than reported as a win.</p>
  {"".join(ablation_blocks)}
</section>

<section id="methods">
  <h2>4. Methods</h2>
  <p class="section-note">What each approach optimizes, what it publishes across
  the command boundary, and which module implements it.</p>
  <div class="cards">{cards}</div>
</section>

<section id="reading">
  <h2>Reading rules</h2>
  <ul class="section-note">{caveats}</ul>
</section>

<footer>
  Built by <code>python -m imitation_experiments.reporting</code> from evaluation
  <code>summary.json</code> artifacts. No number on this page was transcribed by
  hand; open any row's <em>provenance</em> disclosure for the file that backs it.
</footer>
</div>
<script>{_SCRIPT}</script>
"""


def render_data(spec: ReportSpec, records: dict[str, EvalRecord]) -> str:
    """Return the reduced records as JSON, written beside the page for reuse."""
    from dataclasses import asdict

    payload = {
        "title": spec.title,
        "updated": spec.updated,
        "noise_band_pct": spec.noise_band_pct,
        "records": {run: asdict(record) for run, record in sorted(records.items())},
    }
    return json.dumps(payload, indent=2, sort_keys=False)
