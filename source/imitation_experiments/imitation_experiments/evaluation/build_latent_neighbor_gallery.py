#!/usr/bin/env python3
"""Build auditable video and contact-sheet galleries of latent neighbors."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont


DEFAULT_TARGETS = (
    "activity",
    "locomoting",
    "moving_forward",
    "slow_locomotion",
    "turning",
    "jumping",
    "manipulating",
    "object_loaded",
    "torso_lowered",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis_dir", type=Path, required=True)
    parser.add_argument("--phase_clip_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--exclude_motion_names", nargs="*", default=())
    return parser.parse_args()


def phase_for_publication(
    phases_by_motion: dict[str, list[dict[str, Any]]],
    motion_name: str,
    reference_step: int,
) -> dict[str, Any]:
    """Return the end-exclusive semantic phase containing a publication step."""
    phases = phases_by_motion.get(str(motion_name), [])
    matches = [
        phase
        for phase in phases
        if int(phase["start_step"]) <= reference_step < int(phase["end_step"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one phase for {motion_name}@{reference_step}, got {len(matches)}."
        )
    return matches[0]


def select_median_queries(
    publications: Sequence[dict[str, str]],
    neighbors_by_query: dict[int, list[dict[str, str]]],
    *,
    targets: Sequence[str],
    neighbor_count: int,
    excluded_motion_names: set[str],
) -> list[dict[str, Any]]:
    """Choose a deterministic median-performance positive query per target.

    Median selection avoids presenting only the most flattering neighborhood.
    For ``activity``, a match means the exact multiclass activity agrees.  For
    every other target, only positive queries are eligible and a match means
    that binary trait is also positive in the neighbor.
    """
    selected: list[dict[str, Any]] = []
    used_motions: set[str] = set()
    for target in targets:
        candidates: list[dict[str, Any]] = []
        for publication in publications:
            motion = publication["motion_name"]
            if motion in excluded_motion_names:
                continue
            if target != "activity" and int(publication.get(target, "0")) != 1:
                continue
            query_index = int(publication["publication_index"])
            neighbor_rows: list[dict[str, str]] = []
            neighbor_motions: set[str] = set()
            for row in neighbors_by_query.get(query_index, []):
                if row["neighbor_motion"] in neighbor_motions:
                    continue
                neighbor_rows.append(row)
                neighbor_motions.add(row["neighbor_motion"])
                if len(neighbor_rows) == neighbor_count:
                    break
            if len(neighbor_rows) != neighbor_count:
                continue
            if target == "activity":
                matches = [int(row["activity_match"]) for row in neighbor_rows]
                query_value = publication["semantic_activity"]
            else:
                matches = [int(row[f"{target}_neighbor"]) for row in neighbor_rows]
                query_value = "true"
            candidates.append(
                {
                    "target": target,
                    "query_value": query_value,
                    "publication": publication,
                    "neighbors": neighbor_rows,
                    "match_rate": sum(matches) / len(matches),
                }
            )
        if not candidates:
            raise ValueError(f"No positive gallery query is available for {target!r}.")
        rates = sorted(candidate["match_rate"] for candidate in candidates)
        median = rates[len(rates) // 2]
        candidates.sort(
            key=lambda candidate: (
                abs(candidate["match_rate"] - median),
                candidate["publication"]["motion_name"] in used_motions,
                candidate["publication"]["motion_name"],
                int(candidate["publication"]["reference_step"]),
            )
        )
        chosen = candidates[0]
        used_motions.add(chosen["publication"]["motion_name"])
        selected.append(chosen)
    return selected


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _thumbnail(video: Path, output: Path, *, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(duration * 0.5, 0.0):.6f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=360:-2",
            str(output),
        ],
        check=True,
    )


def _contact_sheet(
    path: Path,
    images: Sequence[Path],
    headings: Sequence[str],
    subtitles: Sequence[str],
) -> None:
    thumbnails = [Image.open(image).convert("RGB") for image in images]
    width = max(image.width for image in thumbnails)
    image_height = max(image.height for image in thumbnails)
    header_height = 76
    canvas = Image.new(
        "RGB", (width * len(images), image_height + header_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=11)
    for index, image in enumerate(thumbnails):
        x = index * width + (width - image.width) // 2
        canvas.paste(image, (x, header_height))
        draw.text((index * width + 8, 8), headings[index][:44], fill="black", font=font)
        draw.multiline_text(
            (index * width + 8, 30),
            subtitles[index][:100],
            fill="#444444",
            font=small,
            spacing=2,
        )
    canvas.save(path, quality=92)


def main() -> None:
    args = _parse_args()
    if args.neighbors < 1:
        raise ValueError("--neighbors must be positive.")
    analysis_dir = args.analysis_dir.expanduser().resolve()
    manifest_path = args.phase_clip_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(parents=True)

    publications = _read_csv(analysis_dir / "unique_publications.csv")
    neighbor_rows = _read_csv(analysis_dir / "cross_motion_neighbors.csv")
    neighbors_by_query: dict[int, list[dict[str, str]]] = {}
    for row in neighbor_rows:
        neighbors_by_query.setdefault(int(row["query_index"]), []).append(row)
    for rows in neighbors_by_query.values():
        rows.sort(key=lambda row: int(row["neighbor_rank"]))

    phase_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases_by_motion: dict[str, list[dict[str, Any]]] = {}
    for phase in phase_manifest["clips"]:
        phases_by_motion.setdefault(str(phase["motion_name"]), []).append(phase)
    chosen = select_median_queries(
        publications,
        neighbors_by_query,
        targets=args.targets,
        neighbor_count=args.neighbors,
        excluded_motion_names=set(args.exclude_motion_names),
    )

    gallery_rows: list[dict[str, Any]] = []
    for gallery_index, item in enumerate(chosen):
        publication = item["publication"]
        records = [
            {
                "role": "query",
                "motion_name": publication["motion_name"],
                "reference_step": int(publication["reference_step"]),
                "latent_distance": None,
                "kinematic_distance": None,
            }
        ]
        records.extend(
            {
                "role": f"neighbor {row['neighbor_rank']}",
                "motion_name": row["neighbor_motion"],
                "reference_step": int(row["neighbor_reference_step"]),
                "latent_distance": float(row["latent_distance"]),
                "kinematic_distance": float(row["kinematic_distance"]),
            }
            for row in item["neighbors"]
        )
        image_paths: list[Path] = []
        headings: list[str] = []
        subtitles: list[str] = []
        for column, record in enumerate(records):
            phase = phase_for_publication(
                phases_by_motion,
                record["motion_name"],
                record["reference_step"],
            )
            video = Path(phase["output_video"]).resolve()
            thumbnail = thumbnail_dir / f"row{gallery_index:02d}_col{column:02d}.jpg"
            _thumbnail(video, thumbnail, duration=float(phase["probe"]["duration_s"]))
            image_paths.append(thumbnail)
            headings.append(f"{record['role']}: {record['motion_name']}")
            subtitles.append(
                f"{phase['label']}\nactivity={phase['activity']} step={record['reference_step']}"
            )
            record.update(
                {
                    "phase_label": phase["label"],
                    "activity": phase["activity"],
                    "video": str(video),
                    "thumbnail": str(thumbnail),
                }
            )
        sheet = output_dir / f"{gallery_index:02d}_{item['target']}.jpg"
        _contact_sheet(sheet, image_paths, headings, subtitles)
        gallery_rows.append(
            {
                "target": item["target"],
                "query_value": item["query_value"],
                "selection": (
                    "median semantic match among the k nearest distinct neighbor "
                    "motions for eligible positive queries"
                ),
                "top_k_match_rate": item["match_rate"],
                "contact_sheet": str(sheet),
                "records": records,
            }
        )

    cards: list[str] = []
    for row in gallery_rows:
        videos = "".join(
            f"<figure><video controls muted loop preload='metadata' src='{html.escape(Path(record['video']).as_uri())}'></video>"
            f"<figcaption><b>{html.escape(record['role'])}</b><br>{html.escape(record['phase_label'])}"
            f"<br><small>{html.escape(record['motion_name'])}</small></figcaption></figure>"
            for record in row["records"]
        )
        cards.append(
            f"<section><h2>{html.escape(row['target'])}: median top-{args.neighbors} match "
            f"{row['top_k_match_rate']:.0%}</h2><div class='videos'>{videos}</div></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Cross-motion latent neighbor gallery</title><style>
body{{font:15px system-ui;margin:24px;background:#10141c;color:#eef2f7}} h1{{margin-bottom:4px}}
p{{color:#aeb9c8}} section{{border-top:1px solid #344054;margin-top:28px;padding-top:14px}}
.videos{{display:grid;grid-template-columns:repeat(6,minmax(180px,1fr));gap:12px}}
figure{{margin:0;background:#1b2432;padding:8px;border-radius:8px}} video{{width:100%;height:auto}}
figcaption{{margin-top:6px;line-height:1.3}} small{{color:#aeb9c8;overflow-wrap:anywhere}}
</style></head><body><h1>Cross-motion latent neighbors</h1>
<p>Each row uses a median-performance positive query, not the best case. The query motion is excluded from all neighbors.</p>
{"".join(cards)}</body></html>"""
    (output_dir / "gallery.html").write_text(document, encoding="utf-8")
    (output_dir / "gallery_manifest.json").write_text(
        json.dumps(
            {
                "schema": "latent_neighbor_gallery_v1",
                "analysis_dir": str(analysis_dir),
                "phase_clip_manifest": str(manifest_path),
                "query_selection": (
                    "median semantic match among the k nearest distinct neighbor "
                    "motions for eligible positive queries; the query motion is excluded"
                ),
                "rows": gallery_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[PASS] {len(gallery_rows)} gallery rows -> {output_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
