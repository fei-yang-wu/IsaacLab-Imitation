"""Build a language-goal embedding table for the System-1 skill commander.

This reads a LAFAN1 manifest, collects the unique motion names
(e.g. ``dance1_subject1``), turns each name into a short natural-language phrase
(e.g. ``dance``), and embeds every phrase into a fixed-length vector. The result
is saved as a torch table mapping ``motion_name -> embedding`` that the
downstream commander trainer and rollout sampler load directly, so no text model
is needed at train or rollout time.

Two backends are supported:

* ``dummy`` (default): deterministic pseudo-random unit vectors seeded by the
  phrase text. Needs no external model, so the whole commander pipeline can be
  built and tested before a real text encoder is wired up. Names that clean to
  the same phrase share the same vector, mirroring how a real text encoder would
  group ``dance1`` and ``dance2`` together.
* ``sentence-transformer``: real sentence-transformer embeddings (lazy import;
  only required when this backend is selected).

The table is keyed by the *raw motion name* so the environment's per-trajectory
name lookup always resolves exactly, while the embedding *value* reflects the
cleaned phrase.

Example:
    pixi run python scripts/rlopt/build_language_goal_embeddings.py \
        --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
        --backend dummy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "lafan1" / "manifests" / "g1_lafan1_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "lafan1" / "language" / "g1_lafan1_name_embeddings.pt"
)
# Matches sentence-transformers/all-MiniLM-L6-v2 so the table width is stable
# whether or not the dummy backend is used.
DEFAULT_EMBED_DIM = 384


def humanize_motion_name(name: str) -> str:
    """Convert a motion name into a short phrase.

    ``dance1_subject1`` -> ``dance``; ``fallAndGetUp1_subject4`` -> ``fall and
    get up``; ``fightAndSports1_subject1`` -> ``fight and sports``.
    """
    base = re.sub(r"_subject\d+$", "", str(name))  # drop subject suffix
    base = re.sub(r"\d+$", "", base)  # drop trailing motion index
    base = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", base)  # split camelCase
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip().lower()
    return base or str(name).strip().lower()


def _extract_manifest_entries(data: Any) -> list[dict[str, Any]]:
    """Mirror ``load_lafan1_manifest`` key lookups to find the trajectory list."""
    if isinstance(data, dict):
        entries = data.get("dataset", {}).get("trajectories", {}).get("lafan1_csv")
        if entries is None:
            entries = data.get("lafan1_csv", data.get("motions"))
        if entries is None:
            entries = data
    else:
        entries = data
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            "Manifest must define a non-empty 'dataset.trajectories.lafan1_csv' list."
        )
    return entries


def load_motion_names(manifest_path: Path) -> list[str]:
    """Return the ordered, de-duplicated motion names declared in a manifest."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = _extract_manifest_entries(data)
    names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{index} must be a mapping.")
        name = entry.get("name")
        if not name:
            path_value = entry.get("path") or entry.get("file")
            if path_value is None:
                raise ValueError(f"Manifest entry #{index} needs a 'name' or 'path'.")
            name = Path(str(path_value)).stem
        names.append(str(name))
    # De-duplicate while preserving manifest order.
    return list(dict.fromkeys(names))


def _dummy_embedding(phrase: str, dim: int, seed: int) -> torch.Tensor:
    """Deterministic standard-normal vector seeded by the phrase text."""
    digest = hashlib.sha256(f"{seed}:{phrase}".encode("utf-8")).digest()
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest[:8], "little"))
    return torch.randn(dim, generator=generator, dtype=torch.float32)


def embed_phrases(
    phrases: list[str],
    *,
    backend: str,
    embed_dim: int,
    model_name: str,
    seed: int,
) -> tuple[torch.Tensor, int, str | None]:
    """Embed phrases; returns (matrix[N, D], D, resolved_model_name_or_None)."""
    if backend == "dummy":
        matrix = torch.stack(
            [_dummy_embedding(phrase, embed_dim, seed) for phrase in phrases]
        )
        return matrix, embed_dim, None
    if backend == "sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "backend='sentence-transformer' requires the "
                "'sentence-transformers' package. Install it in the default "
                "Pixi env, or use --backend dummy."
            ) from exc
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            phrases, convert_to_numpy=True, normalize_embeddings=False
        )
        matrix = torch.as_tensor(vectors, dtype=torch.float32)
        return matrix, int(matrix.shape[-1]), model_name
    raise ValueError(f"Unknown backend: {backend!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a language-goal embedding table (M0) for the skill commander."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(DEFAULT_MANIFEST),
        help="LAFAN1 manifest JSON to read motion names from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output torch table path (.pt).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=("dummy", "sentence-transformer"),
        help="Embedding backend. 'dummy' needs no external model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model (sentence-transformer backend only).",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help="Embedding dimension for the dummy backend.",
    )
    parser.add_argument(
        "--raw_names",
        action="store_true",
        default=False,
        help="Embed the literal motion name instead of a cleaned phrase.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the deterministic dummy backend.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    names = load_motion_names(manifest_path)
    phrases = [n if args.raw_names else humanize_motion_name(n) for n in names]

    # Embed each unique phrase once, then expand to one row per motion name so
    # the table can always be looked up by the exact name the env emits.
    unique_phrases = list(dict.fromkeys(phrases))
    phrase_matrix, embed_dim, model_name = embed_phrases(
        unique_phrases,
        backend=args.backend,
        embed_dim=args.embed_dim,
        model_name=args.model,
        seed=args.seed,
    )
    phrase_matrix = torch.nn.functional.normalize(phrase_matrix, dim=-1)
    phrase_to_row = {phrase: row for row, phrase in enumerate(unique_phrases)}
    embeddings = torch.stack([phrase_matrix[phrase_to_row[p]] for p in phrases])

    table = {
        "names": names,
        "phrases": phrases,
        "name_to_index": {name: index for index, name in enumerate(names)},
        "embeddings": embeddings.contiguous(),
        "embed_dim": int(embed_dim),
        "backend": args.backend,
        "model": model_name,
        "raw_names": bool(args.raw_names),
        "manifest": str(manifest_path),
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, output_path)

    model_suffix = f" ({model_name})" if model_name else ""
    print(f"[M0] manifest:       {manifest_path}")
    print(f"[M0] backend:        {args.backend}{model_suffix}")
    print(f"[M0] motion names:   {len(names)}")
    print(f"[M0] unique phrases: {len(unique_phrases)}")
    print(f"[M0] embedding dim:  {embed_dim}")
    print(f"[M0] saved table ->  {output_path}")
    preview = ", ".join(
        f"{name}->'{phrase}'" for name, phrase in list(zip(names, phrases))[:6]
    )
    print(f"[M0] sample mapping: {preview}{' ...' if len(names) > 6 else ''}")


if __name__ == "__main__":
    main()
