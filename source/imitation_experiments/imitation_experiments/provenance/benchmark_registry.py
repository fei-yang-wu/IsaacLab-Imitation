#!/usr/bin/env python3
"""Living benchmark registry: one append-only record per evaluated checkpoint.

The registry answers, for any number that reaches a slide or a paper: which
checkpoint produced it, which encoder it was bound to, what protocol ran, and
where the evidence lives. It is deliberately append-only and content-addressed
-- a stored row is never edited, because the point is to be able to trust an
old number without re-deriving it.

Layout, both locally and mirrored to the evidence repo:

    benchmark/
      index.jsonl                       one JSON object per eval, append-only
      runs/<record_id>/summary.json     the evaluator's own output, verbatim
      runs/<record_id>/record.json      the registry row, standalone
      runs/<record_id>/binding.json     encoder <-> tracker binding proof

``record_id`` is ``<campaign>__<arm>__<checkpoint_tag>__<protocol>``, which is
stable across re-registration of the same evaluation and sorts usefully.

Weights ARE stored, content-addressed under ``blobs/<sha256>.pt``. The cluster
copy is not a backup: ICE scratch is a 300 GB quota that gets pruned, and a
TIMEOUT can wipe node-local output. Content addressing means an encoder shared
by several checkpoints of one arm is uploaded once, and a re-registered
identical file costs nothing.

``blobs/`` is git-ignored -- it is a model store, not source. The durable copy
is the private HF mirror; ``restore`` pulls a record's weights back by sha.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from imitation_experiments.paths import REPO_ROOT

DEFAULT_REGISTRY = REPO_ROOT / "experiments" / "benchmark"
DEFAULT_HF_REPO = "GeorgiaTech/g1-imitation-benchmark"
INDEX_NAME = "index.jsonl"
BLOBS_DIRNAME = "blobs"
_GITIGNORE = "# Model store: weights live in the HF mirror, never in git.\n*\n"


def _store_blob(registry: Path, source: Path) -> dict[str, Any]:
    """Copy a checkpoint into the content-addressed store; dedupe by sha256."""
    digest = _sha256(source)
    blobs = registry / BLOBS_DIRNAME
    blobs.mkdir(parents=True, exist_ok=True)
    (blobs / ".gitignore").write_text(_GITIGNORE)
    target = blobs / f"{digest}.pt"
    if not target.exists():
        # Copy to a temp name then rename: a half-copied blob under its final
        # content-addressed name would look complete and corrupt the store.
        staging = blobs / f".{digest}.partial"
        shutil.copy2(source, staging)
        staging.rename(target)
    return {
        "sha256": digest,
        "blob": f"{BLOBS_DIRNAME}/{digest}.pt",
        "bytes": target.stat().st_size,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class BenchmarkRecord:
    """One evaluated checkpoint under one protocol."""

    record_id: str
    campaign: str
    arm: str
    seed: int
    # Interface under test, the axis this project keeps getting wrong when it
    # is left implicit: command width, hold, phase, anchor frame, bottleneck.
    interface: dict[str, Any]
    # Checkpoint identity. `frames_global` is None when the checkpoint predates
    # cumulative_env_frames and its step is segment-local -- recorded as None
    # rather than guessed, so nobody compares it as if it were matched.
    checkpoint: dict[str, Any]
    encoder: dict[str, Any]
    # Which data the arm TRAINED on and which set it was SCORED on. Two
    # different corpora (129k BONES-SEED vs the 30-motion compositionality
    # set), and a number is meaningless without both.
    dataset: dict[str, Any]
    protocol: dict[str, Any]
    metrics: dict[str, float]
    evidence: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)


def _load_index(registry: Path) -> list[dict[str, Any]]:
    index = registry / INDEX_NAME
    if not index.is_file():
        return []
    rows = []
    for line in index.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def register(
    *,
    registry: Path,
    eval_dir: Path,
    campaign: str,
    arm: str,
    seed: int,
    interface: dict[str, Any],
    checkpoint_path: str,
    checkpoint_tag: str,
    frames_global: int | None,
    encoder_path: str,
    binding_json: Path | None,
    train_dataset: str,
    protocol: str,
    notes: str,
    local_checkpoint: Path | None = None,
    local_encoder: Path | None = None,
    store_weights: bool = True,
) -> BenchmarkRecord:
    """Append one evaluation to the registry; refuse to silently overwrite."""
    summary_path = eval_dir / "summary.json"
    if not summary_path.is_file():
        msg = f"no summary.json under {eval_dir}"
        raise FileNotFoundError(msg)
    summary = json.loads(summary_path.read_text())

    means = summary.get("metric_means", {})
    aggregate = summary.get("aggregate", {})
    metrics = {
        "mpjpe_l_mm": means.get("tracking_mpjpe_mm"),
        "mpjpe_g_mm": means.get("tracking_mpjpe_g_mm"),
        "fall_free_rate": aggregate.get("fall_free_rate"),
        "survival_steps_mean": aggregate.get("survival_steps_mean"),
        "num_metric_rows": summary.get("num_metric_rows"),
    }
    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        # A partial metric set means the eval did not run the protocol it
        # claims; recording it would put an unverifiable row in the index.
        msg = f"summary.json is missing {missing}; not registering {eval_dir}"
        raise ValueError(msg)

    record_id = f"{campaign}__{arm}__{checkpoint_tag}__{protocol}"
    run_dir = registry / "runs" / record_id
    if run_dir.exists():
        msg = (
            f"record {record_id} already exists at {run_dir}. The registry is "
            "append-only: delete it deliberately, or use a distinct protocol tag."
        )
        raise FileExistsError(msg)

    tracker_blob: dict[str, Any] | None = None
    encoder_blob: dict[str, Any] | None = None
    if store_weights:
        if local_checkpoint is None or local_encoder is None:
            msg = (
                "storing weights needs --local-checkpoint and --local-encoder; "
                "pass --no-store-weights to register metrics only."
            )
            raise ValueError(msg)
        tracker_blob = _store_blob(registry, Path(local_checkpoint))
        encoder_blob = _store_blob(registry, Path(local_encoder))

    # The eval's own command line is the authority on what it scored: the
    # launcher can be edited later, the frozen command cannot.
    eval_dataset = next(
        (
            token.split("=", 1)[1]
            for token in str(summary.get("command", "")).split()
            if token.startswith("env.data.persist_id=")
        ),
        "unknown",
    )

    record = BenchmarkRecord(
        record_id=record_id,
        campaign=campaign,
        arm=arm,
        seed=seed,
        interface=interface,
        checkpoint={
            "tag": checkpoint_tag,
            "cluster_path": checkpoint_path,
            "sha256": tracker_blob["sha256"] if tracker_blob else None,
            "blob": tracker_blob["blob"] if tracker_blob else None,
            "bytes": tracker_blob["bytes"] if tracker_blob else None,
            "frames_global": frames_global,
            "frames_note": (
                "segment-local step; global frame count unknown"
                if frames_global is None
                else "global env frames"
            ),
        },
        encoder={
            "cluster_path": encoder_path,
            "sha256": encoder_blob["sha256"] if encoder_blob else None,
            "blob": encoder_blob["blob"] if encoder_blob else None,
            "bytes": encoder_blob["bytes"] if encoder_blob else None,
        },
        dataset={"train": train_dataset, "eval": eval_dataset},
        protocol={
            "name": protocol,
            "eval_label": summary.get("metadata", {}).get("label"),
            "num_envs": summary.get("num_envs"),
            "max_steps": summary.get("max_steps"),
            "seed": summary.get("seed"),
            "survival_definition": summary.get("survival_definition"),
            "tracking_terminations_enabled": summary.get(
                "tracking_terminations_enabled"
            ),
            "stop_reason": summary.get("stop_reason"),
            "steps_run": summary.get("steps_run"),
        },
        metrics=metrics,
        evidence={
            "summary_json": f"runs/{record_id}/summary.json",
            "binding_json": (
                f"runs/{record_id}/binding.json" if binding_json else None
            ),
            "local_eval_dir": str(eval_dir),
        },
        provenance={
            "git_commit": _git_commit(),
            "notes": notes,
            "registered_by": os.environ.get("USER", "unknown"),
        },
    )

    run_dir.mkdir(parents=True)
    shutil.copy2(summary_path, run_dir / "summary.json")
    if binding_json and Path(binding_json).is_file():
        shutil.copy2(binding_json, run_dir / "binding.json")
    (run_dir / "record.json").write_text(json.dumps(asdict(record), indent=2) + "\n")
    with (registry / INDEX_NAME).open("a") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return record


def render_table(registry: Path) -> str:
    """Markdown leaderboard, best MPJPE-L first, with the fairness columns."""
    rows = _load_index(registry)
    if not rows:
        return "_No records yet._\n"
    rows.sort(key=lambda r: r["metrics"]["mpjpe_l_mm"])
    out = [
        "| arm | interface | frames | MPJPE-L | MPJPE-G | fall-free | record |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        iface = r["interface"]
        width = iface.get("command_dim", "?")
        hold = iface.get("hold", "?")
        frames = r["checkpoint"]["frames_global"]
        frames_txt = f"{frames / 1e9:.2f}B" if frames else "seg-local"
        m = r["metrics"]
        out.append(
            f"| `{r['arm']}` | {width}-D h{hold} | {frames_txt} "
            f"| {m['mpjpe_l_mm']:.2f} | {m['mpjpe_g_mm']:.1f} "
            f"| {m['fall_free_rate']:.3f} | `{r['record_id']}` |"
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="register one evaluation")
    add.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    add.add_argument("--eval-dir", type=Path, required=True)
    add.add_argument("--campaign", required=True)
    add.add_argument("--arm", required=True)
    add.add_argument("--seed", type=int, default=0)
    add.add_argument("--command-dim", type=int, required=True)
    add.add_argument("--z-dim", type=int, required=True)
    add.add_argument("--hold", type=int, required=True)
    add.add_argument("--phase-mode", default="sin_cos")
    add.add_argument("--anchor-mode", default="robot_heading")
    add.add_argument("--bottleneck", required=True)
    add.add_argument("--objective", default="endpoint")
    add.add_argument("--checkpoint-path", required=True, help="cluster path")
    add.add_argument("--checkpoint-tag", required=True)
    add.add_argument(
        "--frames-global",
        type=int,
        default=None,
        help="global env frames; omit when the checkpoint step is segment-local",
    )
    add.add_argument("--encoder-path", required=True)
    add.add_argument("--binding-json", type=Path, default=None)
    add.add_argument("--local-checkpoint", type=Path, default=None)
    add.add_argument("--local-encoder", type=Path, default=None)
    add.add_argument(
        "--no-store-weights",
        action="store_true",
        help="register metrics only; the weights are then NOT backed up",
    )
    add.add_argument(
        "--train-dataset",
        default="bones_seed_sonic_full_129785@e714bbff",
        help="persist_id of the corpus the arm trained on",
    )
    add.add_argument("--protocol", default="oracle30_fallonly_newton")
    add.add_argument("--notes", default="")

    table = sub.add_parser("table", help="print the markdown leaderboard")
    table.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)

    push = sub.add_parser("push", help="mirror the registry to a private HF repo")
    push.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    push.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    push.add_argument(
        "--backend",
        choices=("hf", "wandb"),
        default="wandb",
        help=(
            "wandb: one versioned artifact per record, linked to the run that "
            "produced it (lineage). hf: a private dataset repo (bulk offsite)."
        ),
    )
    push.add_argument("--wandb-project", default="g1-benchmark")
    push.add_argument("--wandb-entity", default=None)
    push.add_argument("--message", default="update benchmark registry")
    push.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip blobs/ (metrics + evidence only)",
    )

    restore = sub.add_parser(
        "restore", help="fetch one record's weights back from the HF mirror"
    )
    restore.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    restore.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    restore.add_argument("--record-id", required=True)
    restore.add_argument("--dest", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "add":
        args.registry.mkdir(parents=True, exist_ok=True)
        record = register(
            registry=args.registry,
            eval_dir=args.eval_dir,
            campaign=args.campaign,
            arm=args.arm,
            seed=args.seed,
            interface={
                "command_dim": args.command_dim,
                "z_dim": args.z_dim,
                "hold": args.hold,
                "phase_mode": args.phase_mode,
                "anchor_mode": args.anchor_mode,
                "bottleneck": args.bottleneck,
                "objective": args.objective,
            },
            checkpoint_path=args.checkpoint_path,
            checkpoint_tag=args.checkpoint_tag,
            frames_global=args.frames_global,
            encoder_path=args.encoder_path,
            binding_json=args.binding_json,
            train_dataset=args.train_dataset,
            protocol=args.protocol,
            notes=args.notes,
            local_checkpoint=args.local_checkpoint,
            local_encoder=args.local_encoder,
            store_weights=not args.no_store_weights,
        )
        print(f"[REGISTRY] added {record.record_id}")
        print(
            f"[REGISTRY]   MPJPE-L {record.metrics['mpjpe_l_mm']:.2f} mm  "
            f"MPJPE-G {record.metrics['mpjpe_g_mm']:.1f} mm  "
            f"fall-free {record.metrics['fall_free_rate']:.3f}"
        )
        return 0

    if args.cmd == "table":
        print(render_table(args.registry), end="")
        return 0

    if args.cmd == "restore":
        from huggingface_hub import hf_hub_download

        rows = {r["record_id"]: r for r in _load_index(args.registry)}
        row = rows.get(args.record_id)
        if row is None:
            msg = f"unknown record_id {args.record_id!r}"
            raise SystemExit(msg)
        args.dest.mkdir(parents=True, exist_ok=True)
        for kind in ("checkpoint", "encoder"):
            blob = row[kind].get("blob")
            if not blob:
                print(f"[REGISTRY] {kind}: no blob stored for this record")
                continue
            local = hf_hub_download(
                repo_id=args.repo_id, repo_type="dataset", filename=blob
            )
            out = args.dest / f"{kind}.pt"
            shutil.copy2(local, out)
            # Content addressing is only a guarantee if it is checked on the
            # way back: a silently corrupt restore is worse than a missing one.
            got = _sha256(out)
            if got != row[kind]["sha256"]:
                msg = f"{kind} sha mismatch: expected {row[kind]['sha256']}, got {got}"
                raise SystemExit(msg)
            print(f"[REGISTRY] restored {kind} -> {out} (sha verified)")
        return 0

    if args.cmd == "push" and args.backend == "wandb":
        import wandb

        rows = _load_index(args.registry)
        if not rows:
            print("[REGISTRY] nothing to push")
            return 0
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="benchmark-registry",
            name="benchmark-registry-sync",
        )
        for row in rows:
            art = wandb.Artifact(
                name=row["record_id"].replace("__", "-"),
                type="tracker-checkpoint",
                metadata={
                    "arm": row["arm"],
                    "campaign": row["campaign"],
                    **row["interface"],
                    **row["metrics"],
                    "frames_global": row["checkpoint"]["frames_global"],
                    "train_dataset": row["dataset"]["train"],
                    "eval_dataset": row["dataset"]["eval"],
                    "git_commit": row["provenance"].get("git_commit"),
                },
            )
            # Tags are how the registry is browsed: dataset source first, so
            # "which checkpoints came from BONES-SEED 129k" is one click.
            artifact_tags = [
                f"train:{row['dataset']['train'].split('@')[0]}",
                f"eval:{row['dataset']['eval'].split('@')[0]}",
                f"campaign:{row['campaign']}",
                f"arm:{row['arm']}",
                f"bottleneck:{row['interface'].get('bottleneck', 'unknown')}",
                f"hold:{row['interface'].get('hold', '?')}",
            ]
            try:
                art.tags = artifact_tags  # wandb >= 0.18
            except Exception:  # pragma: no cover - older clients ignore tags
                pass
            run_dir = args.registry / "runs" / row["record_id"]
            for name in ("summary.json", "record.json", "binding.json"):
                f = run_dir / name
                if f.is_file():
                    art.add_file(str(f), name=name)
            # W&B dedupes by content hash too, so a shared encoder across an
            # arm's records uploads once and later artifacts just reference it.
            for kind, fname in (("checkpoint", "tracker.pt"), ("encoder", "encoder.pt")):
                blob = row[kind].get("blob")
                if blob and (args.registry / blob).is_file():
                    art.add_file(str(args.registry / blob), name=fname)
            run.log_artifact(art, aliases=["latest", row["dataset"]["train"].split("@")[0]])
        run.finish()
        print(f"[REGISTRY] pushed {len(rows)} artifacts to {args.wandb_project}")
        return 0

    # push (hf)
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id, repo_type="dataset", private=True, exist_ok=True
    )
    api.upload_folder(
        folder_path=str(args.registry),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message,
        ignore_patterns=["blobs/*"] if args.metadata_only else None,
    )
    scope = "metadata only" if args.metadata_only else "including weights"
    print(f"[REGISTRY] mirrored {args.registry} -> {args.repo_id} (private, {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
