# Copyright (c) 2026, IsaacLab-Imitation Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tile stroboscopic sequence composites into one paper figure.

Takes the per-motion images `render_paper_policy_video.py --shot sequence`
writes and lays them out on a grid, for a figure that spans both columns of an
IEEE two-column manuscript (`figure*`, 7.16 in). Two rows of two panels is the
default; the grid follows `--columns`.

Each panel is cropped to the ink it actually carries -- the robot and its
shadow, found by differencing against the panel's own backdrop colour -- so a
composite that left half its frame empty does not waste half a column. Every
panel is then padded back to one common aspect, so the panels stay the same
size and the robots stay the same scale relative to each other. Scaling panels
independently would make a robot's apparent size an artefact of how far it
walked.

Writes PNG and, when Pillow is available, a PDF at the requested figure width
so LaTeX places it at 1:1 without resampling. The LaTeX snippet to include it
is printed at the end.

Example:

.. code-block:: bash

    pixi run python scripts/viz/assemble_sequence_figure.py \\
        --panels logs/.../sequence/rank-000029-*.png \\
                 logs/.../sequence/rank-000017-*.png \\
                 logs/.../sequence/rank-000013-*.png \\
                 logs/.../sequence/rank-000002-*.png \\
        --labels "walk forward" "lift a crate and walk" \\
                 "open the door and walk out" "lift low to high" \\
        --output outputs/paper/figures/qualitative_sequences.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# A pixel is ink when it differs from its own ROW's background by more than
# this many grey levels. Comparing against one colour sampled at the corners
# does not work here: the cyclorama carries a vertical gradient, so most of an
# empty frame reads as ink and the crop keeps the whole panel -- which is
# exactly how a four-panel figure ends up over four inches tall.
_INK_THRESHOLD = 10
_ROBUST_BACKGROUND_PERCENTILE = 50  # row-fit refit keeps this fraction as background
# Fraction of the tight ink box kept as breathing room on each side. Vertical
# is generous by default: a tight vertical crop reads as the figure clipping
# the robots' heads and feet, which is a worse look than a few extra pixels of
# backdrop. Horizontal needs less -- the pose spread already leaves margin at
# both ends of the row.
_MARGIN_FRACTION_Y = 0.12
_MARGIN_FRACTION_X = 0.04
# A row or column sets the crop only when its ink is a real share of the
# busiest line's. An absolute count cannot do this job: the top rows of a panel
# carry a dozen pixels of renderer noise, which is enough to hold the crop at
# full height and leave a figure twice as tall as the robots in it.
_MIN_INK_SHARE = 0.02
_SPAN_GAP = 24  # blank lines allowed inside one band before it counts as two
_FAINT_INK_SHARE = 0.002  # ink share that still counts when growing a band


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels", type=str, nargs="+", required=True, help="Panel images, in order."
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Caption text per panel; printed in the LaTeX snippet, not drawn.",
    )
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument(
        "--gutter",
        type=int,
        default=18,
        help="Pixels between panels, at the assembled resolution.",
    )
    parser.add_argument(
        "--figure_width_in",
        type=float,
        default=7.16,
        help="Target width in inches; 7.16 is IEEE two-column full width.",
    )
    parser.add_argument(
        "--margin_y",
        type=float,
        default=_MARGIN_FRACTION_Y,
        help="Vertical breathing room, as a fraction of each panel's tight ink box.",
    )
    parser.add_argument(
        "--margin_x",
        type=float,
        default=_MARGIN_FRACTION_X,
        help="Horizontal breathing room, as a fraction of each panel's tight ink box.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _read(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _row_background(panel: np.ndarray) -> np.ndarray:
    """A smooth backdrop estimate: one straight line fitted across each row.

    The cyclorama is lit from one side, so a row is not flat and its median is
    not its background -- comparing against the median leaves the whole lit
    side reading as ink, which keeps the crop at full height. A line per row
    tracks both gradients.

    The fit is robust (two passes), not a plain least squares over every pixel
    in the row. A single wide row -- an 11-pose teaser spans nearly the whole
    frame width -- gives the naive fit enough dark robot and shadow pixels to
    drag the line toward them, which then reads as ink almost everywhere and
    a crop that finds no edge to trim. The first pass fits the whole row, the
    second refits using only the pixels the first pass called background.
    """
    height, width, _ = panel.shape
    x = np.linspace(-1.0, 1.0, width)
    values = panel.astype(np.float32)
    mean, slope = _fit_row_line(values, x, weight=None)
    residual = np.abs(values - (mean + slope * x[None, :, None])).max(axis=2)
    # Keep the more-background half of each row. The panels this guards
    # against carry robots and shadow over well under half of any row, so this
    # margin holds even there; a plain single robot loses nothing by it.
    cutoff = np.percentile(
        residual, _ROBUST_BACKGROUND_PERCENTILE, axis=1, keepdims=True
    )
    weight = (residual <= cutoff).astype(np.float32)
    mean2, slope2 = _fit_row_line(values, x, weight=weight)
    # A row with too little kept background (e.g. one edge-to-edge blur) falls
    # back to the first pass rather than dividing by a near-zero determinant.
    enough = weight.sum(axis=1, keepdims=True) > 0.1 * width
    mean = np.where(enough[..., None], mean2, mean)
    slope = np.where(enough[..., None], slope2, slope)
    return mean + slope * x[None, :, None]


def _fit_row_line(
    values: np.ndarray, x: np.ndarray, weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-row, per-channel weighted least-squares fit of `value = mean + slope*x`.

    Closed form over the 2x2 normal equations, vectorised over every row and
    channel at once. `weight` is `None` for the ordinary (unweighted) fit and
    an (H, W) 0/1 mask for the robust refit.
    """
    height, width, channels = values.shape
    w = np.ones((height, width), dtype=np.float32) if weight is None else weight
    sum_w = w.sum(axis=1)
    sum_wx = (w * x[None, :]).sum(axis=1)
    sum_wxx = (w * x[None, :] * x[None, :]).sum(axis=1)
    det = sum_w * sum_wxx - sum_wx * sum_wx
    det = np.where(np.abs(det) < 1.0e-6, 1.0, det)
    mean = np.empty((height, 1, channels), dtype=np.float32)
    slope = np.empty((height, 1, channels), dtype=np.float32)
    for channel in range(channels):
        sum_wy = (w * values[..., channel]).sum(axis=1)
        sum_wxy = (w * x[None, :] * values[..., channel]).sum(axis=1)
        mean[:, 0, channel] = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
        slope[:, 0, channel] = (sum_w * sum_wxy - sum_wx * sum_wy) / det
    return mean, slope


def _ink_mask(panel: np.ndarray, plate: np.ndarray | None = None) -> np.ndarray:
    """Pixels that carry robot or shadow rather than backdrop.

    `plate` is the empty-scene render the sequence renderer writes next to
    every composite (`<stem>_plate.png`): the exact per-pixel background, same
    camera and lighting, robot hidden. When it is available, ink is a straight
    diff against it. Without one, `_row_background` fits a straight line per
    row as an approximation -- and it is only an approximation: the cyclorama
    is not perfectly linear across a full-width row, so on a row where several
    robots mask out much of the width (an 11-pose teaser at torso height), the
    line's slope can be dragged just enough to misjudge a far column by ten-odd
    grey levels, which reads as ink at the very edge the crop is trying to find.
    """
    background = (
        plate.astype(np.float32) if plate is not None else _row_background(panel)
    )
    residual = np.abs(panel.astype(np.float32) - background)
    return residual.max(axis=2) > _INK_THRESHOLD


def _dominant_span(counts: np.ndarray) -> tuple[int, int]:
    """The band of lines that actually holds the subject.

    First-to-last is not it. A panel can carry a few rows of renderer residue
    against its top edge, and taking the outermost qualifying line then keeps
    everything between that residue and the robots -- half a panel of empty
    backdrop, in a figure whose whole point is to be short. Grouping the
    qualifying lines into runs and keeping the heaviest run drops the residue
    and nothing else.
    """
    indices = np.flatnonzero(counts > _MIN_INK_SHARE * counts.max())
    if indices.size == 0:
        return 0, 0
    breaks = np.flatnonzero(np.diff(indices) > _SPAN_GAP) + 1
    best = max(np.split(indices, breaks), key=lambda run: counts[run].sum())
    # The band found above is the body. Grow it over anything that still
    # carries ink, because a raised hand is a few pixels wide and would fail
    # the share test that located the body -- cropping to the band alone cuts
    # the hands off the top of a lifting motion.
    floor = max(2.0, _FAINT_INK_SHARE * counts.max())
    start = _grow_span(counts, int(best[0]), -1, floor)
    end = _grow_span(counts, int(best[-1]), 1, floor)
    return start, end + 1


def _grow_span(counts: np.ndarray, index: int, step: int, floor: float) -> int:
    """Walk outward from a band while ink keeps appearing.

    Blank lines are tolerated up to `_SPAN_GAP` in a row, which bridges the gap
    between a hand and a head without reaching the detached strip of renderer
    residue that sits against a panel's edge.
    """
    edge, blanks, cursor = index, 0, index
    while 0 <= cursor < counts.size:
        if counts[cursor] > floor:
            edge, blanks = cursor, 0
        else:
            blanks += 1
            if blanks > _SPAN_GAP:
                break
        cursor += step
    return edge


def _ink_bbox(
    panel: np.ndarray, plate: np.ndarray | None = None
) -> tuple[int, int, int, int]:
    """Rows and columns of the panel that carry robot or shadow."""
    ink = _ink_mask(panel, plate)
    top, bottom = _dominant_span(ink.sum(axis=1))
    left, right = _dominant_span(ink.sum(axis=0))
    if bottom <= top or right <= left:
        return 0, panel.shape[0], 0, panel.shape[1]
    return top, bottom, left, right


def _body_column_center(ink: np.ndarray, top: int, bottom: int) -> float:
    """Horizontal centre of the ROBOTS, not of the robots-plus-shadow box.

    The key light is fixed, so every pose's shadow trails the same direction
    on screen and stretches the ink bbox further on that one side. Padding the
    full bbox symmetrically then still reads as unbalanced -- the gap looks
    equal in pixels but unequal from a viewer's eye, which measures from the
    robot, not from the shadow it happens to be standing next to. The upper
    body band (see the pose-spread measurement in the renderer, the same
    trick) is shadow-free, since a shadow is cast on the floor.
    """
    band = ink[top : top + int(round(0.55 * (bottom - top)))]
    columns = np.flatnonzero(band.any(axis=0))
    if columns.size == 0:
        return 0.5 * ink.shape[1]
    return 0.5 * (float(columns[0]) + float(columns[-1]))


def _crop_column_window(
    panel: np.ndarray, plate: np.ndarray | None, margin_x: float
) -> tuple[int, int]:
    """Left/right crop bounds, centred on the robots so both margins match.

    Padding the raw ink bbox symmetrically gives EQUAL PIXEL padding, not an
    equal-looking gap: if the bbox itself is off-centre (a shadow reaching
    further right than the body reaches left), the robot ends up nearer one
    edge than the other. Centring on the body instead and sizing the window to
    the larger of its two reaches to the true ink edge keeps the shadow fully
    inside the crop while making the robot-to-edge gap the same on both sides.
    """
    ink = _ink_mask(panel, plate)
    top, bottom = _dominant_span(ink.sum(axis=1))
    left, right = _dominant_span(ink.sum(axis=0))
    if bottom <= top or right <= left:
        return 0, panel.shape[1]
    center = _body_column_center(ink, top, bottom)
    half = max(center - left, right - center) + margin_x * (right - left)
    return (
        max(0, int(round(center - half))),
        min(panel.shape[1], int(round(center + half))),
    )


def _crop_box(
    panel: np.ndarray,
    margin_y: float,
    margin_x: float,
    plate: np.ndarray | None,
) -> tuple[int, int, int, int]:
    top, bottom, _, _ = _ink_bbox(panel, plate)
    pad_y = int(margin_y * (bottom - top))
    left, right = _crop_column_window(panel, plate, margin_x)
    return max(0, top - pad_y), min(panel.shape[0], bottom + pad_y), left, right


def _crop_to_ink(
    panel: np.ndarray,
    margin_y: float = _MARGIN_FRACTION_Y,
    margin_x: float = _MARGIN_FRACTION_X,
    plate: np.ndarray | None = None,
) -> np.ndarray:
    top, bottom, left, right = _crop_box(panel, margin_y, margin_x, plate)
    return panel[top:bottom, left:right]


def _pad_to(
    panel: np.ndarray, height: int, width: int, plate: np.ndarray | None = None
) -> np.ndarray:
    """Centre the panel in a common box, filled with its own backdrop colour.

    Padding rather than stretching: the robots must stay comparable in size
    across panels, and a per-panel scale would break that.
    """
    top = (height - panel.shape[0]) // 2
    left = (width - panel.shape[1]) // 2
    # Pad with each row's own backdrop colour, extended above and below, so a
    # panel that needed less height than its neighbours still reads as one
    # continuous backdrop instead of gaining a visible box. Fit against the
    # cropped PANEL when there is no plate for it: the crop trims right where
    # the ink ends, so its edge rows still carry faint robot or shadow
    # residue, and repeating that residue across the whole padded margin reads
    # as a smeared ghost of the pose. The plate has none of it -- it is the
    # same scene with the robot hidden -- so ghosting is possible only for a
    # panel this function had to fall back on the row-fit for in the first
    # place.
    source = plate if plate is not None else panel
    background = np.clip(_row_background(source), 0, 255).astype(np.uint8)
    if width > panel.shape[1]:
        edge = background[:, -1:, :]
        background = np.concatenate(
            [background, np.repeat(edge, width - panel.shape[1], axis=1)], axis=1
        )
    out = np.concatenate(
        [
            np.repeat(background[:1], top, axis=0),
            background[:, :width],
            np.repeat(background[-1:], height - top - panel.shape[0], axis=0),
        ]
    )
    out[top : top + panel.shape[0], left : left + panel.shape[1]] = panel
    return out


def assemble(
    panels: list[np.ndarray],
    columns: int,
    gutter: int,
    margin_y: float = _MARGIN_FRACTION_Y,
    margin_x: float = _MARGIN_FRACTION_X,
    plates: list[np.ndarray | None] | None = None,
) -> np.ndarray:
    plates = plates or [None] * len(panels)
    boxes = [
        _crop_box(panel, margin_y, margin_x, plate)
        for panel, plate in zip(panels, plates)
    ]
    cropped = [panel[t:b, lo:hi] for panel, (t, b, lo, hi) in zip(panels, boxes)]
    # Crop the plate with the SAME box as its panel, so the padding fill (see
    # _pad_to) lines up row for row and column for column with what it pads.
    cropped_plates = [
        plate[t:b, lo:hi] if plate is not None else None
        for plate, (t, b, lo, hi) in zip(plates, boxes)
    ]
    cell_h = max(panel.shape[0] for panel in cropped)
    cell_w = max(panel.shape[1] for panel in cropped)
    boxed = [
        _pad_to(panel, cell_h, cell_w, plate)
        for panel, plate in zip(cropped, cropped_plates)
    ]
    rows = (len(boxed) + columns - 1) // columns
    canvas = np.full(
        (
            rows * cell_h + (rows - 1) * gutter,
            columns * cell_w + (columns - 1) * gutter,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for index, panel in enumerate(boxed):
        row, column = divmod(index, columns)
        y = row * (cell_h + gutter)
        x = column * (cell_w + gutter)
        canvas[y : y + cell_h, x : x + cell_w] = panel
    return canvas


def main() -> None:
    args = _parse_args()
    paths = [Path(p) for p in args.panels]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing panel images: {missing}")
    plate_paths = [p.with_name(f"{p.stem}_plate{p.suffix}") for p in paths]
    plates = [_read(pp) if pp.is_file() else None for pp in plate_paths]
    print(
        f"[FIGURE] exact plate found for {sum(p is not None for p in plates)}/"
        f"{len(plates)} panels; the rest use the row-fit approximation"
    )
    figure = assemble(
        [_read(p) for p in paths],
        int(args.columns),
        int(args.gutter),
        float(args.margin_y),
        float(args.margin_x),
        plates,
    )

    from PIL import Image

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(figure)
    # The stored DPI makes the figure land at the requested width without
    # LaTeX resampling it.
    dpi = figure.shape[1] / float(args.figure_width_in)
    image.save(output, dpi=(dpi, dpi))
    pdf = output.with_suffix(".pdf")
    image.save(pdf, "PDF", resolution=dpi)
    print(
        f"[FIGURE] {figure.shape[1]}x{figure.shape[0]} px, {dpi:.0f} dpi at "
        f"{args.figure_width_in:.2f} in"
    )
    print(f"[FIGURE] wrote {output}")
    print(f"[FIGURE] wrote {pdf}")

    labels = args.labels or [p.stem for p in paths]
    print("\n% IEEE two-column full-width figure")
    print("\\begin{figure*}[t]")
    print("  \\centering")
    print(f"  \\includegraphics[width=\\textwidth]{{{pdf.name}}}")
    print("  \\caption{" + "; ".join(labels) + ".}")
    print("  \\label{fig:qualitative-sequences}")
    print("\\end{figure*}")


if __name__ == "__main__":
    main()
