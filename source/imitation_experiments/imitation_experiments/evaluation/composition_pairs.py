"""Pick clip pairs for the skill-composition probes.

Clips come from the reference-array manifest (names and lengths) and are
restricted to a survivor list (clips the tracker completes on the board), so
a fall under a composed code is attributable to the composition, not to a
clip the tracker cannot track alone. Categories are name regexes; pairs are
drawn per category pair with a fixed seed.

    python -m imitation_experiments.evaluation.composition_pairs \\
        --manifest <ref_arrays>/reference_arrays_manifest.json \\
        --survivors logs/latent64_probe_mirror/eval/*.json \\
        --min-frames 320 --pairs-per-kind 10 --out pairs.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

CATEGORIES: dict[str, tuple[str, str]] = {
    # name: (include regex, exclude regex)
    "walk": (
        r"^walk_forward|^walk_fast_forward|^walking_forward",
        r"backward|turn|side|stair|stop|start|carry|crate|door|box|pick|jump|circle|zigzag|zombie|drunk|limp|injur|sneak|tip",
    ),
    "jog": (r"^(jog|run)_", r"stop|start|backward|side|jump|zigzag|dance"),
    "turn": (r"^(walk|jog)_.*turn|^turn_", r"jump|dance|stop|start"),
    "stand": (r"^(idle|stand)", r"dance|sit|jump"),
    "wave": (r"wave", r"dance"),
    "crouch": (r"crouch|squat", r"jump|dance"),
}

DEFAULT_KINDS: tuple[tuple[str, str], ...] = (
    ("walk", "jog"),
    ("walk", "turn"),
    ("walk", "stand"),
    ("stand", "wave"),
    ("crouch", "walk"),
    ("jog", "turn"),
)


def load_clips(manifest: Path) -> list[dict[str, Any]]:
    info = json.loads(Path(manifest).read_text())["traj_info"]
    names = [t[1] for t in info["ordered_traj_list"]]
    starts, ends = info["start_index"], info["end_index"]
    return [
        {"rank": i, "name": names[i], "frames": int(ends[i] - starts[i])}
        for i in range(len(names))
    ]


def load_survivors(paths: Sequence[Path]) -> set[int] | None:
    """Ranks that succeeded in EVERY given evaluator summary; ``None`` if no
    summaries were given."""
    survivors: set[int] | None = None
    for path in paths:
        payload = json.loads(Path(path).read_text())
        ok = {
            int(e["trajectory_rank"])
            for e in payload.get("per_environment", [])
            if e.get("tracking_success")
        }
        survivors = ok if survivors is None else survivors & ok
    return survivors


def categorize(
    clips: Sequence[dict[str, Any]], min_frames: int, survivors: set[int] | None
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in CATEGORIES}
    for clip in clips:
        if clip["frames"] < min_frames:
            continue
        if survivors is not None and clip["rank"] not in survivors:
            continue
        for kind, (inc, exc) in CATEGORIES.items():
            if re.search(inc, clip["name"], re.I) and not re.search(
                exc, clip["name"], re.I
            ):
                out[kind].append(clip)
                break
    return out


def draw_pairs(
    by_kind: dict[str, list[dict[str, Any]]],
    kinds: Sequence[tuple[str, str]] = DEFAULT_KINDS,
    *,
    pairs_per_kind: int = 10,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs: list[dict[str, Any]] = []
    for a_kind, b_kind in kinds:
        a_pool, b_pool = by_kind.get(a_kind, []), by_kind.get(b_kind, [])
        if not a_pool or not b_pool:
            continue
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(seen) < pairs_per_kind and attempts < 50 * pairs_per_kind:
            attempts += 1
            a, b = rng.choice(a_pool), rng.choice(b_pool)
            if a["rank"] == b["rank"] or (a["rank"], b["rank"]) in seen:
                continue
            seen.add((a["rank"], b["rank"]))
            pairs.append(
                {
                    "kind": f"{a_kind}->{b_kind}",
                    "a": a["rank"],
                    "b": b["rank"],
                    "a_name": a["name"],
                    "b_name": b["name"],
                    "a_frames": a["frames"],
                    "b_frames": b["frames"],
                }
            )
    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--survivors", type=Path, nargs="*", default=[])
    parser.add_argument("--min-frames", type=int, default=320)
    parser.add_argument("--pairs-per-kind", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    clips = load_clips(args.manifest)
    survivors = load_survivors(args.survivors)
    by_kind = categorize(clips, args.min_frames, survivors)
    pairs = draw_pairs(by_kind, pairs_per_kind=args.pairs_per_kind, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "survivors": [str(p) for p in args.survivors],
                "min_frames": args.min_frames,
                "seed": args.seed,
                "pool_sizes": {k: len(v) for k, v in by_kind.items()},
                "pairs": pairs,
            },
            indent=1,
        )
    )
    print(json.dumps({k: len(v) for k, v in by_kind.items()}), "pairs:", len(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
