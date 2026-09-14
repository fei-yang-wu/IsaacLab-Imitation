#!/usr/bin/env python3
"""Cluster window latents and choose what to show for each cluster.

Stage 2 of the latent-semantics analysis, and the Isaac-free half. It reads the
``latents.npz`` written by ``qualitative_latent_semantics.py``, clusters the
rows, and writes a manifest naming the motions stage 3 should render for each
cluster.

This stage deliberately does NOT try to tell you what a cluster means. Word
statistics over the annotations can say which words are frequent, which is not
the same as a meaning you can defend. The conclusion comes from watching stage
3's video: eight robots, each replaying a different member window of one
cluster, at the same time. If the cluster is real they visibly do the same
thing, and you name it from what you see. The text here is a hint to check that
reading against, not the answer.

Two choices make the videos worth watching:

* **One member per motion.** Eight windows of a single clip would look
  identical and prove nothing; the point is whether DIFFERENT motions land
  together.
* **Members are the rows nearest the cluster centroid** -- what the cluster is
  most typical of. ``--member_selection farthest`` instead spreads the picks
  across the cluster by farthest-point sampling, which shows its extent
  including the boundary and therefore reads as far less coherent.

Run it in the default Pixi environment; nothing here needs Isaac::

    pixi run python experiments/qualitative/src/qualitative_latent_semantics_cluster.py \\
        --run_dir outputs/.../latent_semantics_encode \\
        --output_dir outputs/.../latent_semantics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# OpenBLAS is built for at most 128 threads, and KMeans opens one BLAS pool per
# worker. On a machine with more cores than that (this one reports 240) the
# pools exhaust OpenBLAS's memory regions and the process dies with SIGSEGV
# before the first fit finishes -- not an out-of-memory error, a hard crash. Cap
# the pools unless the caller has already chosen a number. This must run before
# numpy is imported, because the BLAS backend reads these at load time.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "8")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qualitative_common as qc  # noqa: E402

#: Per-motion free-text goals for the full BONES-SEED set, relative to the repo.
DEFAULT_LANGUAGE_JSON = (
    "data/bones_seed_sonic_129k_50hz/g1_bones_seed_sonic_full_language.json"
)


#: Motion NPZ files for the full BONES-SEED set, relative to the repo.
DEFAULT_NPZ_DIR = "data/bones_seed_sonic_129k_50hz/npz/g1"


def window_dynamics(
    motions: np.ndarray,
    local_steps: np.ndarray,
    *,
    npz_dir: Path,
    window_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two per-window speeds (m/s) from the dataset NPZ files.

    ``root_speed``
        Window mean of ``|root_lin_vel|`` -- is the robot going anywhere.

    ``limb_speed``
        Window mean of the mean over the 5 fastest bodies of
        ``|body_lin_vel_w - root_lin_vel|`` -- how hard the limbs move
        RELATIVE to the root. A mean over all 30 bodies would drown a big
        arm swing under 25 near-still bodies and pass every slow walk on
        root translation alone, biasing the kept set toward locomotion.
        Root-relative top-5 instead scores an in-place dance or kick high
        and a robot standing near-still low, whatever its clip is named.
    """
    manifest = next(npz_dir.parent.parent.glob("*_manifest.json"))
    entries = json.loads(manifest.read_text())["dataset"]["trajectories"]["lafan1_csv"]
    name_to_file = {str(e["name"]): Path(str(e["path"])).name for e in entries}

    per_motion: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    root_speeds = np.zeros(motions.shape[0], dtype=np.float32)
    limb_speeds = np.zeros(motions.shape[0], dtype=np.float32)
    for index, (motion, start) in enumerate(zip(motions, local_steps)):
        name = str(motion)
        if name not in per_motion:
            frames = np.load(npz_dir / name_to_file[name])
            body_vel = frames["body_lin_vel_w"]
            root_vel = frames["root_lin_vel"]
            relative = np.linalg.norm(body_vel - root_vel[:, None, :], axis=-1)
            per_motion[name] = (
                np.linalg.norm(root_vel, axis=-1).astype(np.float32),
                np.sort(relative, axis=1)[:, -5:].mean(axis=1).astype(np.float32),
            )
        root, limb = per_motion[name]
        start = int(start)
        stop = min(start + int(window_frames), root.shape[0])
        root_speeds[index] = float(root[start:stop].mean())
        limb_speeds[index] = float(limb[start:stop].mean())
    return root_speeds, limb_speeds


@dataclass(frozen=True)
class GoalTable:
    """``motion -> language_goal``, plus the mirror flag."""

    goal: dict[str, str]
    is_mirror: dict[str, bool]

    def __len__(self) -> int:
        return len(self.goal)


def load_goal_table(path: str | Path) -> GoalTable:
    """Read the BONES-SEED language sidecar."""
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    goal: dict[str, str] = {}
    is_mirror: dict[str, bool] = {}
    for entry in payload["motions"]:
        name = str(entry["name"])
        text = str(entry.get("language_goal") or "").strip()
        if not text:
            msg = f"Motion {name!r} has an empty language_goal in {path}."
            raise ValueError(msg)
        goal[name] = text
        is_mirror[name] = bool(entry.get("is_mirror", False))
    return GoalTable(goal=goal, is_mirror=is_mirror)


def join_goals(motions: np.ndarray, table: GoalTable) -> np.ndarray:
    """Goal text per row. Refuses to drop a row it cannot label."""
    unknown = sorted({str(name) for name in motions if str(name) not in table.goal})
    if unknown:
        shown = ", ".join(unknown[:5])
        msg = (
            f"{len(unknown)} motions have no language_goal entry (e.g. {shown}). "
            "The latents and the language table describe different datasets."
        )
        raise KeyError(msg)
    return np.asarray([table.goal[str(name)] for name in motions])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def corpus_document_frequency(goals: list[str]) -> tuple[dict[str, int], int]:
    """How many goal strings contain each term."""
    frequency: dict[str, int] = {}
    for text in goals:
        for token in set(_tokenize(text)):
            frequency[token] = frequency.get(token, 0) + 1
    return frequency, len(goals)


def distinctive_terms(
    goals_in_cluster: list[str],
    document_frequency: dict[str, int],
    total_documents: int,
    *,
    top_k: int = 6,
) -> list[str]:
    """Terms common INSIDE a cluster and rare outside it -- a hint, not a name.

    Plain tf-idf. Without the idf half the top terms of every cluster would be
    the corpus's most frequent words, which distinguishes nothing.
    """
    if not goals_in_cluster:
        return []
    counts: dict[str, int] = {}
    for text in goals_in_cluster:
        for token in set(_tokenize(text)):
            counts[token] = counts.get(token, 0) + 1
    scored: list[tuple[str, float]] = []
    for token, count in counts.items():
        term_frequency = count / len(goals_in_cluster)
        idf = np.log((total_documents + 1.0) / (document_frequency.get(token, 0) + 1.0))
        scored.append((token, float(term_frequency * idf)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _ in scored[:top_k]]


def _one_per_motion_order(
    member_rows: np.ndarray, ranks: np.ndarray, order: np.ndarray
) -> list[int]:
    """Walk ``order`` keeping the first row seen for each motion."""
    chosen: list[int] = []
    seen: set[int] = set()
    for position in order:
        row = int(member_rows[position])
        rank = int(ranks[row])
        if rank in seen:
            continue
        seen.add(rank)
        chosen.append(row)
    return chosen


def pick_cluster_members(
    latents: np.ndarray,
    centroid: np.ndarray,
    member_rows: np.ndarray,
    ranks: np.ndarray,
    *,
    count: int,
    strategy: str = "centroid",
) -> list[int]:
    """Choose the rows to render for one cluster, at most one per motion.

    One per motion is the point of the exercise: several windows of a single
    clip would look alike on screen whatever the code space does, so they would
    make any cluster look coherent. Distinct motions landing together is the
    claim being tested.

    ``strategy`` decides WHICH distinct motions:

    ``farthest`` (default)
        Farthest-point sampling. Start from the row nearest the centroid, then
        repeatedly add the candidate whose nearest already-chosen member is as
        far away as possible. The clip then spans the cluster instead of
        sampling one dense corner of it, so near-duplicate motions do not take
        up eight slots. This is the STRICTER test: if members picked to be as
        unlike each other as the cluster allows still look like the same
        action, the cluster has a topic. A coherent cluster will look less tidy
        under this than under ``centroid``, and that is the intended cost.

    ``centroid``
        The ``count`` rows nearest the centroid -- what the cluster is most
        typical of. Flattering by construction, and useful when the question is
        "what is the core of this cluster" rather than "does all of it agree".
    """
    if member_rows.size == 0:
        return []
    points = latents[member_rows]
    to_centroid = np.linalg.norm(points - centroid[None, :], axis=1)

    if strategy == "centroid":
        return _one_per_motion_order(member_rows, ranks, np.argsort(to_centroid))[
            : int(count)
        ]
    if strategy != "farthest":
        raise ValueError(
            f"Unknown member selection strategy: {strategy!r}. "
            "Use 'farthest' or 'centroid'."
        )

    # Deduplicate by motion first, so the greedy walk cannot spend a pick on a
    # second window of a motion it already showed.
    candidates = np.asarray(
        _one_per_motion_order(member_rows, ranks, np.argsort(to_centroid)),
        dtype=np.int64,
    )
    if candidates.size <= int(count):
        return [int(row) for row in candidates]

    candidate_points = latents[candidates]
    # Seed on the most typical member, so every clip still contains one row
    # that represents the cluster's core rather than only its boundary.
    chosen_positions = [0]
    nearest = np.linalg.norm(candidate_points - candidate_points[0][None, :], axis=1)
    while len(chosen_positions) < int(count):
        nearest[chosen_positions] = -1.0
        pick = int(np.argmax(nearest))
        if nearest[pick] <= 0.0:
            break
        chosen_positions.append(pick)
        nearest = np.minimum(
            nearest,
            np.linalg.norm(candidate_points - candidate_points[pick][None, :], axis=1),
        )
    return [int(candidates[position]) for position in chosen_positions]


def _plot_clusters(
    output_dir: Path,
    latents: np.ndarray,
    labels: np.ndarray,
    *,
    rendered_rows: list[int],
    tsne_rows: int,
    seed: int,
) -> None:
    """Scatters of the clustering, with the rendered members marked.

    Two figures because they answer different questions. PCA
    (``clusters_scatter.png``) is a faithful linear shadow: distances mean
    something, but 64 dimensions squashed into 2 overlap heavily, so it
    mostly shows the gross shape. t-SNE (``tsne_scatter.png``, its own
    compact figure) separates neighbourhoods so cluster structure is
    legible, at the price of distances between blobs being meaningless --
    do not read "these two clusters are close" off it.

    The rows that became videos are ringed in black. That is the link between
    these pictures and the gallery: you can see whether a clip sampled the
    middle of its cluster or its edge.
    """
    plt = qc._matplotlib()
    from sklearn.decomposition import PCA

    rendered = np.asarray(rendered_rows, dtype=np.int64)

    figure, axis = plt.subplots(figsize=(10.0, 9.0))
    projected = PCA(n_components=2, random_state=seed).fit_transform(latents)
    _scatter_panel(
        axis,
        projected,
        labels,
        np.isin(np.arange(latents.shape[0]), rendered),
        "PCA of window latents (distances meaningful, heavy overlap)",
    )
    figure.tight_layout()
    path = qc.save_figure(figure, output_dir / "clusters_scatter.png")
    plt.close(figure)
    print(f"[INFO] Wrote {path} (and .pdf sibling)")

    if tsne_rows > 0:
        from sklearn.manifold import TSNE

        rng = np.random.default_rng(seed)
        rows = latents.shape[0]
        if rows > tsne_rows:
            pool = np.setdiff1d(np.arange(rows), rendered)
            extra = rng.choice(
                pool, size=max(0, tsne_rows - rendered.size), replace=False
            )
            # Always carry the rendered rows so they can be marked here too.
            pick = np.sort(np.concatenate([rendered, extra]))
        else:
            pick = np.arange(rows)
        embedded = TSNE(
            n_components=2, random_state=seed, init="pca", perplexity=30
        ).fit_transform(latents[pick])
        figure, axis = plt.subplots(figsize=(4.6, 4.1))
        _scatter_panel(
            axis,
            embedded,
            labels[pick],
            np.isin(pick, rendered),
            "t-SNE visualization of the commands",
            dot_size=9.0,
            dot_alpha=0.5,
            title_fontsize=qc.TITLE_FONTSIZE + 4,
            show_rendered=False,
        )
        figure.tight_layout(pad=0.4)
        path = qc.save_figure(figure, output_dir / "tsne_scatter.png")
        plt.close(figure)
        print(f"[INFO] Wrote {path} (and .pdf sibling)")


def _densest_point(pts: np.ndarray, *, k: int = 10, sample: int = 2000) -> np.ndarray:
    """The cluster member with the highest local density.

    The label anchor. A cluster's MEAN lands in empty space whenever t-SNE (or
    PCA) splits the cluster into several blobs or stretches it, so the number
    floats far from where the points actually sit. The member whose k-th
    nearest same-cluster neighbour is closest sits inside the densest blob by
    construction.
    """
    if len(pts) < 3:
        return pts.mean(axis=0)
    if len(pts) > sample:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), size=sample, replace=False)]
    k = min(k, len(pts) - 1)
    squared = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    kth = np.partition(squared, k, axis=1)[:, k]
    return pts[int(np.argmin(kth))]


def _scatter_panel(
    axis,
    points,
    labels,
    is_rendered,
    title,
    *,
    dot_size: float = 4.0,
    dot_alpha: float = 0.55,
    title_fontsize: float = qc.TITLE_FONTSIZE,
    show_rendered: bool = True,
) -> None:
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap=qc.CLUSTER_CMAP,
        s=dot_size,
        alpha=dot_alpha,
        linewidths=0,
    )
    if show_rendered and is_rendered.any():
        axis.scatter(
            points[is_rendered, 0],
            points[is_rendered, 1],
            facecolors="none",
            edgecolors="black",
            s=42,
            linewidths=0.9,
            label="rendered in the gallery",
        )
        axis.legend(loc="upper right", fontsize=qc.LEGEND_FONTSIZE, frameon=False)
    for cluster in np.unique(labels):
        centre = _densest_point(points[labels == cluster])
        axis.annotate(
            str(int(cluster)),
            centre,
            fontsize=qc.TITLE_FONTSIZE,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "white", "alpha": 0.8, "lw": 0},
        )
    axis.set_title(title, fontsize=title_fontsize)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_linewidth(0.8)
        axis.spines[side].set_color("0.7")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--language_json", type=str, default=None)
    parser.add_argument(
        "--k",
        type=int,
        default=24,
        help=(
            "Number of clusters. Fixed on purpose: silhouette rises "
            "monotonically with k on this space, so picking k by it just "
            "returns the largest value offered. Choose a k you are willing to "
            "sit through one video for."
        ),
    )
    parser.add_argument(
        "--members",
        type=int,
        default=8,
        help="Motions rendered per cluster, one per motion.",
    )
    parser.add_argument(
        "--member_selection",
        type=str,
        default="centroid",
        choices=["farthest", "centroid"],
        help=(
            "Which members to render. centroid (default): the rows nearest the "
            "centroid, what the cluster is most typical of. farthest: spread "
            "across the cluster by farthest-point sampling, which shows its "
            "extent including the boundary and reads as much less coherent."
        ),
    )
    parser.add_argument(
        "--tsne_rows",
        type=int,
        default=6000,
        help="Rows subsampled for the t-SNE panel of the scatter. 0 disables it.",
    )
    parser.add_argument(
        "--min_local_step",
        type=int,
        default=0,
        help=(
            "Drop windows whose reference start frame is below this. BONES-SEED "
            "clips open in a neutral standing pose, so early windows from "
            "different motions are near-identical and pull every cluster "
            "toward one shared standing blob; 50 skips the first second at "
            "50 Hz. 0 keeps every window."
        ),
    )
    parser.add_argument(
        "--min_root_speed",
        type=float,
        default=0.0,
        help=(
            "With --min_limb_speed, the static-window gate: a window is KEPT "
            "when its mean |root_lin_vel| reaches this many m/s OR its limb "
            "speed reaches --min_limb_speed; a window below BOTH is dropped. "
            "The OR matters: thresholding one whole-body mean instead would "
            "pass slow walking on root translation alone while dropping "
            "in-place dances and kicks, biasing every cluster toward "
            "locomotion. 0 disables this half of the gate."
        ),
    )
    parser.add_argument(
        "--min_limb_speed",
        type=float,
        default=0.0,
        help=(
            "Other half of the static-window gate: the window mean, over the "
            "5 fastest bodies, of |body_lin_vel_w - root_lin_vel| in m/s -- "
            "limb motion relative to the root, so an in-place dance scores "
            "high and near-still standing scores near zero. On this dataset "
            "root 0.4 OR limb 0.6 keeps roughly the most dynamic 40%. "
            "0 disables this half of the gate."
        ),
    )
    parser.add_argument(
        "--exclude_motion_regex",
        type=str,
        default="",
        help=(
            "Drop every window whose motion NAME matches this regex "
            "(case-insensitive), e.g. 'idle' to exclude all idle-family clips "
            "entirely, including their occasional moving windows. Empty "
            "disables the gate."
        ),
    )
    parser.add_argument(
        "--window_frames",
        type=int,
        default=10,
        help="Frames per encoded window, for the speed-gate measurements.",
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        default=None,
        help="Motion NPZ directory for the speed gates (default: BONES-SEED).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    latents_path = run_dir / "latents.npz"
    if not latents_path.is_file():
        msg = (
            f"No latents.npz under {run_dir}. Run the encode stage "
            "(qualitative_latent_semantics.py) first."
        )
        raise FileNotFoundError(msg)

    data = np.load(latents_path, allow_pickle=False)
    latents = np.asarray(data["latent"], dtype=np.float32)
    motions = np.asarray(data["motion"])
    ranks = np.asarray(data["rank"], dtype=np.int64)
    local_steps = np.asarray(data["local_step"], dtype=np.int64)
    motion_lengths = np.asarray(data["motion_length"], dtype=np.int64)
    print(f"[INFO] {latents.shape[0]} rows x {latents.shape[1]} latent values.")

    min_local_step = int(args.min_local_step)
    rows_before = int(latents.shape[0])
    motions_before = int(np.unique(motions).size)
    if min_local_step > 0:
        keep = local_steps >= min_local_step
        if not bool(keep.any()):
            msg = (
                f"--min_local_step {min_local_step} drops all {rows_before} "
                "rows. Lower it, or re-run the encode stage with more windows "
                "per motion."
            )
            raise SystemExit(msg)
        latents = latents[keep]
        motions = motions[keep]
        ranks = ranks[keep]
        local_steps = local_steps[keep]
        motion_lengths = motion_lengths[keep]
        motions_after = int(np.unique(motions).size)
        print(
            f"[INFO] min_local_step={min_local_step}: kept "
            f"{latents.shape[0]}/{rows_before} rows; "
            f"{motions_before - motions_after} of {motions_before} motions "
            "lost every window (all their windows start earlier)."
        )

    exclude_motion_regex = str(args.exclude_motion_regex or "")
    if exclude_motion_regex:
        pattern = re.compile(exclude_motion_regex, re.IGNORECASE)
        keep = np.asarray(
            [pattern.search(str(name)) is None for name in motions], dtype=bool
        )
        if not bool(keep.any()):
            raise SystemExit(
                f"--exclude_motion_regex {exclude_motion_regex!r} drops every "
                "remaining row. Loosen it."
            )
        dropped_motions = int(np.unique(motions[~keep]).size)
        latents, motions, ranks = latents[keep], motions[keep], ranks[keep]
        local_steps, motion_lengths = local_steps[keep], motion_lengths[keep]
        print(
            f"[INFO] exclude_motion_regex={exclude_motion_regex!r}: kept "
            f"{latents.shape[0]} rows; removed {dropped_motions} matching "
            "motions entirely."
        )

    min_root_speed = float(args.min_root_speed)
    min_limb_speed = float(args.min_limb_speed)
    if min_root_speed > 0.0 or min_limb_speed > 0.0:
        npz_dir = Path(args.npz_dir or (qc.repo_root() / DEFAULT_NPZ_DIR)).expanduser()
        root_speeds, limb_speeds = window_dynamics(
            motions,
            local_steps,
            npz_dir=npz_dir.resolve(),
            window_frames=int(args.window_frames),
        )
        # Keep a window that clears EITHER active gate; drop only windows
        # that are static on both counts (going nowhere AND limbs near
        # still). A disabled gate (0) keeps nothing by itself.
        keep = np.zeros(latents.shape[0], dtype=bool)
        if min_root_speed > 0.0:
            keep |= root_speeds >= min_root_speed
        if min_limb_speed > 0.0:
            keep |= limb_speeds >= min_limb_speed
        if not bool(keep.any()):
            raise SystemExit(
                f"--min_root_speed {min_root_speed} / --min_limb_speed "
                f"{min_limb_speed} drop every remaining row (max root "
                f"{float(root_speeds.max()):.3f}, max limb "
                f"{float(limb_speeds.max()):.3f} m/s). Lower them."
            )
        in_place = keep & (
            root_speeds < (min_root_speed if min_root_speed > 0.0 else np.inf)
        )
        latents, motions, ranks = latents[keep], motions[keep], ranks[keep]
        local_steps, motion_lengths = local_steps[keep], motion_lengths[keep]
        motions_after = int(np.unique(motions).size)
        print(
            f"[INFO] speed gate (root>={min_root_speed} OR "
            f"limb>={min_limb_speed} m/s): kept {latents.shape[0]} rows over "
            f"{motions_after} motions; {int(in_place.sum())} of the kept rows "
            "are in-place dynamic (limb-only)."
        )

    language_json = (
        Path(args.language_json or (qc.repo_root() / DEFAULT_LANGUAGE_JSON))
        .expanduser()
        .resolve()
    )
    table = load_goal_table(language_json)
    goals = join_goals(motions, table)
    print(f"[PASS] joined {len(table)} language entries.")

    output_dir = qc.prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    model = KMeans(n_clusters=int(args.k), n_init=10, random_state=int(args.seed))
    labels = model.fit_predict(latents)
    rng = np.random.default_rng(int(args.seed))
    sample = (
        rng.choice(latents.shape[0], size=5000, replace=False)
        if latents.shape[0] > 5000
        else np.arange(latents.shape[0])
    )
    silhouette = float(silhouette_score(latents[sample], labels[sample]))
    # Reported, never used to choose k: it climbs with k on this space, so as a
    # selector it would always return the largest option.
    print(f"[INFO] k={args.k}, silhouette={silhouette:+.4f} (diagnostic only).")

    frequency, total_documents = corpus_document_frequency(sorted(set(goals.tolist())))

    clusters: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    all_rendered_rows: list[int] = []
    for cluster in range(int(args.k)):
        member_rows = np.flatnonzero(labels == cluster)
        if member_rows.size == 0:
            print(f"[WARN] cluster {cluster} is empty; skipping.")
            continue
        chosen = pick_cluster_members(
            latents,
            model.cluster_centers_[cluster],
            member_rows,
            ranks,
            count=int(args.members),
            strategy=str(args.member_selection),
        )
        all_rendered_rows.extend(int(row) for row in chosen)
        cluster_goals = goals[member_rows].tolist()
        terms = distinctive_terms(cluster_goals, frequency, total_documents)
        counts: dict[str, int] = {}
        for text in cluster_goals:
            counts[text] = counts.get(text, 0) + 1
        top_goals = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:5]

        clusters.append(
            {
                "cluster": int(cluster),
                "size": int(member_rows.size),
                "motions": int(len(set(ranks[member_rows].tolist()))),
                "term_hint": terms,
                "members": [
                    {
                        "rank": int(ranks[row]),
                        "motion": str(motions[row]),
                        "start_frame": int(local_steps[row]),
                        "motion_length": int(motion_lengths[row]),
                        "language_goal": str(goals[row]),
                    }
                    for row in chosen
                ],
            }
        )
        summary.append(
            {
                "cluster": int(cluster),
                "size": int(member_rows.size),
                "motions": int(len(set(ranks[member_rows].tolist()))),
                "rendered": len(chosen),
                "term_hint": " ".join(terms),
                "top_goals": " | ".join(f"{t} x{c}" for t, c in top_goals),
            }
        )

    (output_dir / "clusters.json").write_text(
        json.dumps(
            {
                "encode_run_dir": str(run_dir),
                "k": int(args.k),
                "members_per_cluster": int(args.members),
                "member_selection": str(args.member_selection),
                "seed": int(args.seed),
                "silhouette": silhouette,
                "clusters": clusters,
            },
            indent=2,
        )
    )
    with (output_dir / "cluster_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "clusters_window.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "rank", "motion", "local_step", "language_goal"])
        for index in range(latents.shape[0]):
            writer.writerow(
                [
                    int(labels[index]),
                    int(ranks[index]),
                    str(motions[index]),
                    int(local_steps[index]),
                    str(goals[index]),
                ]
            )

    qc.write_provenance(
        output_dir,
        mode="latent_semantics_cluster",
        encode_run_dir=str(run_dir),
        latents_sha256=qc.sha256(latents_path),
        language_json=str(language_json),
        language_json_sha256=qc.sha256(language_json),
        k=int(args.k),
        members_per_cluster=int(args.members),
        member_selection=str(args.member_selection),
        seed=int(args.seed),
        silhouette_diagnostic=silhouette,
        min_local_step=min_local_step,
        exclude_motion_regex=exclude_motion_regex,
        min_root_speed=min_root_speed,
        min_limb_speed=min_limb_speed,
        window_frames=int(args.window_frames),
        rows_before_step_filter=rows_before,
        rows=int(latents.shape[0]),
        motions=int(len(set(ranks.tolist()))),
    )

    _plot_clusters(
        output_dir,
        latents,
        labels,
        rendered_rows=sorted(all_rendered_rows),
        tsne_rows=int(args.tsne_rows),
        seed=int(args.seed),
    )

    print(f"\n[INFO] {len(clusters)} clusters, {int(args.members)} motions each:")
    for row in summary:
        print(
            f"  cluster {row['cluster']:3d}  n={row['size']:6d}  "
            f"motions={row['motions']:5d}  hint: {row['term_hint']}"
        )
    print(f"\n[INFO] Output root: {output_dir}")
    print("[INFO] Next: render one video per cluster with the gallery stage.")


if __name__ == "__main__":
    main()
