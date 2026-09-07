"""Derive a short, densely checkpointed rerun from a training campaign.

The convergence curves of a campaign start at its first saved checkpoint. When
that interval is 200M frames, every curve jumps from the untrained origin to a
value already near its plateau, and the early phase is invisible. This tool
writes a sibling campaign that trains every arm again for a short budget with
a fine checkpoint interval, so the early phase can be scored.

What it keeps, verbatim: every arm's interface, rewards, reset selection ramp,
termination curriculum, environment count, optimizer, and network sizes. The
frame-keyed schedules (the reset ramp, the termination curriculum) depend on
the step counter, not on the budget, so the first N frames of the rerun follow
the same schedule the original run did.

What it changes:

* `frame_cap` and `save_interval`, and therefore `max_iterations`;
* the output root, so the rerun never writes into the original trees;
* the encoder binding: the rerun reuses the ORIGINAL run's pretrained encoder
  from the archive instead of pretraining again, so the encoder is the same
  tensor the original tracker trained against;
* the stages: only the first tracker segment survives, with its
  `depends_on` removed. The pretrain and resume segments are dropped;
* the W&B group and every run id prefix, so the rerun cannot resume or
  overwrite an original W&B run.

The rerun is a fresh training run. Its checkpoints are not the original run's
early checkpoints, and its curve joins the original curve at the first common
frame count only up to run-to-run noise.

    python -m imitation_experiments.pipeline.derive_early_phase_campaign \
        --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
        --name latent-star-v2-early --wandb-group latent-star-v2-early \
        --frame-cap 200000000 --save-interval 9830400 \
        --output-root /storage/ice-shared/vip-vwt/scratch-fwu91/latent_star_v2_early \
        --archive-root /storage/ice-shared/vip-vwt/scratch-fwu91/archived_data/latent_star_v2_checkpoints \
        --out experiments/campaigns/2026-09-05-star-v2-early-dense/campaign.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

# RLOpt tags a run `logdir:<19-char timestamp>_wandb-<run id>` and W&B caps a
# tag at 64 characters, so a run id must stay at or below 31.
MAX_RUN_ID_LEN = 31
KEPT_STAGE = "lowlevel1"
EXTRA_TAG = "early-dense"


def _retag(value: str, old_group: str, new_group: str) -> str:
    tags = [t for t in value.split(",") if t]
    tags = [new_group if t == old_group else t for t in tags]
    if EXTRA_TAG not in tags:
        tags.append(EXTRA_TAG)
    return ",".join(tags)


def _replace_group_tokens(node: Any, old_group: str, new_group: str) -> Any:
    """Rewrite W&B group tokens inside argument lists; leave paths alone."""
    if isinstance(node, list):
        out: list[Any] = []
        follow = False
        for item in node:
            if follow:
                out.append(new_group if item == old_group else item)
                follow = False
                continue
            if isinstance(item, str) and item == f"agent.logger.group_name={old_group}":
                out.append(f"agent.logger.group_name={new_group}")
            elif item == "--wandb_group":
                out.append(item)
                follow = True
            else:
                out.append(_replace_group_tokens(item, old_group, new_group))
        return out
    if isinstance(node, dict):
        return {
            k: _replace_group_tokens(v, old_group, new_group) for k, v in node.items()
        }
    return node


def derive(
    campaign_path: Path,
    *,
    name: str,
    wandb_group: str,
    frame_cap: int,
    save_interval: int,
    output_root: str,
    archive_root: str,
    time_limit: str = "03:00:00",
    id_prefix_old: str = "lsv2-",
    id_prefix_new: str = "lsv2e-",
) -> str:
    # `json` round-trip breaks YAML anchor sharing, so editing one arm's stage
    # cannot silently edit every arm that aliased the same block.
    doc = json.loads(json.dumps(yaml.safe_load(campaign_path.read_text())))
    old_group = str(doc["wandb_group"])
    if frame_cap <= 0 or save_interval <= 0:
        raise ValueError("frame_cap and save_interval must be positive")
    if save_interval > frame_cap:
        raise ValueError("save_interval must not exceed frame_cap")

    doc["name"] = name
    doc["wandb_group"] = wandb_group
    base = doc["vars"]
    for key in (
        "frame_cap",
        "save_interval",
        "output_root",
        "encoder_ckpt",
        "hub_encoder",
    ):
        if key not in base:
            raise ValueError(f"training campaign has no var `{key}`")
    base["frame_cap"] = int(frame_cap)
    base["save_interval"] = int(save_interval)
    base["archive_root"] = archive_root.rstrip("/")
    base["output_root"] = f"{output_root.rstrip('/')}/${{vars.arm}}_seed${{vars.seed}}"
    base["encoder_ckpt"] = (
        "${vars.archive_root}/${vars.arm}_seed${vars.seed}/encoder/checkpoints/latest.pt"
    )
    base["hub_encoder"] = "${vars.archive_root}/hub_seed0/encoder/checkpoints/latest.pt"
    doc["vars"] = _replace_group_tokens(base, old_group, wandb_group)

    preflight = doc.setdefault("preflight", {})
    required = list(preflight.get("require_container_paths") or [])
    if base["archive_root"] not in required:
        required.append(base["archive_root"])
    preflight["require_container_paths"] = required
    preflight["output_container_path"] = base["output_root"]

    for arm, spec in doc["arms"].items():
        arm_vars = spec.setdefault("vars", {})
        run_id = str(arm_vars.get("wandb_id", base.get("wandb_id", "")))
        if not run_id.startswith(id_prefix_old):
            raise ValueError(
                f"{arm}: wandb_id {run_id!r} lacks prefix {id_prefix_old!r}"
            )
        run_id = id_prefix_new + run_id[len(id_prefix_old) :]
        # `-s<seed>` is appended by the stage env; budget for one digit.
        if len(run_id) + 3 > MAX_RUN_ID_LEN:
            raise ValueError(f"{arm}: run id {run_id!r}-s0 exceeds {MAX_RUN_ID_LEN}")
        arm_vars["wandb_id"] = run_id
        if (
            "encoder_ckpt" in arm_vars
            and arm_vars["encoder_ckpt"] != "${vars.hub_encoder}"
        ):
            raise ValueError(
                f"{arm}: per-arm encoder_ckpt {arm_vars['encoder_ckpt']!r} is not "
                "the hub indirection; map it by hand"
            )
        kept = [s for s in spec["stages"] if s.get("name") == KEPT_STAGE]
        if len(kept) != 1:
            raise ValueError(f"{arm}: expected exactly one `{KEPT_STAGE}` stage")
        stage = kept[0]
        stage.pop("depends_on", None)
        stage.pop("dependency_kind", None)
        stage["time_limit"] = time_limit
        stage["args"] = _replace_group_tokens(stage.get("args"), old_group, wandb_group)
        env = stage.get("env") or {}
        if "CLUSTER_WANDB_TAGS" in env:
            env["CLUSTER_WANDB_TAGS"] = _retag(
                str(env["CLUSTER_WANDB_TAGS"]), old_group, wandb_group
            )
        spec["stages"] = [stage]

    header = (
        "# GENERATED by imitation_experiments.pipeline.derive_early_phase_campaign\n"
        f"# from {campaign_path.as_posix()}.\n"
        "# Do not hand-edit: every arm's interface is copied from the training\n"
        "# campaign. Re-run the generator instead.\n"
        "#\n"
        f"# Short, densely checkpointed rerun: frame_cap {frame_cap}, a checkpoint\n"
        f"# every {save_interval} frames, ONE tracker segment per arm, the ORIGINAL\n"
        "# run's pretrained encoder read from the archive (no pretrain stage), a new\n"
        "# W&B group and run-id prefix. A fresh training run, not the original\n"
        "# run's early checkpoints.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--frame-cap", type=int, required=True)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--time-limit", default="03:00:00")
    parser.add_argument("--id-prefix-old", default="lsv2-")
    parser.add_argument("--id-prefix-new", default="lsv2e-")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    text = derive(
        args.campaign,
        name=args.name,
        wandb_group=args.wandb_group,
        frame_cap=args.frame_cap,
        save_interval=args.save_interval,
        output_root=args.output_root,
        archive_root=args.archive_root,
        time_limit=args.time_limit,
        id_prefix_old=args.id_prefix_old,
        id_prefix_new=args.id_prefix_new,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
