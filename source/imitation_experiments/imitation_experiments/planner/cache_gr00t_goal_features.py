"""Cache per-goal GR00T backbone text features and the warm-start bundle.

The goal set is finite, so the Cosmos-Reason2-2B (Qwen3-VL) backbone runs
exactly once per goal here — offline, in the ``gr00t`` Pixi environment — and
the action head cross-attends over the cached features at train and eval time.
This keeps the pretrained cross-attention keys in-distribution while removing
the VLM from every runtime path.

Also exports the action-head warm-start bundle (trunk state dict + the
checkpoint's head config) so training never needs the full checkpoint again.

Run from the repository root:

    pixi run -e gr00t python -m imitation_experiments.planner.cache_gr00t_goal_features \
        --goals_json experiments/campaigns/<...>/goals.json \
        --output_dir outputs/gr00t_goal_features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from imitation_experiments.planner.gr00t_head import (
    PRETRAINED_KEEP_PREFIXES,
    ensure_gr00t_importable,
    gr00t_submodule_commit,
    save_provenance,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, default="nvidia/GR00T-N1.7-3B")
    parser.add_argument(
        "--goals_json",
        type=Path,
        default=None,
        help=(
            "JSON file: either a list of goal strings or an object "
            "{goal_name: goal_text}. Required unless --bundle_only or "
            "--language_sidecar is given."
        ),
    )
    parser.add_argument(
        "--language_sidecar",
        type=Path,
        default=None,
        help=(
            "Motion-manifest language sidecar "
            "(g1_..._manifest_language.json). Goals become "
            "{motions[].name: motions[].language_goal}."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--formalize_language",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Lowercase + strip punctuation, matching the GR00T processor default.",
    )
    parser.add_argument(
        "--export_head_bundle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export the action-head trunk state dict for the warm start.",
    )
    parser.add_argument(
        "--bundle_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export only the warm-start bundle, reading the checkpoint "
            "safetensors directly. Requires no VLM backbone instantiation, so "
            "it works without access to the gated Cosmos-Reason2-2B repo. "
            "--goals_json is ignored in this mode."
        ),
    )
    return parser.parse_args()


def _load_goals(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        goals = {str(item): str(item) for item in raw}
    elif isinstance(raw, dict):
        goals = {str(key): str(value) for key, value in raw.items()}
    else:
        msg = f"{path} must hold a list of strings or an object of strings."
        raise ValueError(msg)
    if not goals:
        msg = f"{path} contains no goals."
        raise ValueError(msg)
    return goals


def _load_goals_from_sidecar(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text())
    motions = raw.get("motions")
    if not isinstance(motions, list) or not motions:
        msg = f"{path} has no 'motions' list."
        raise ValueError(msg)
    goals: dict[str, str] = {}
    for entry in motions:
        name = entry.get("name")
        text = entry.get("language_goal")
        if not name or not text:
            msg = f"{path}: motion entry lacks name/language_goal: {entry!r:.120}"
            raise ValueError(msg)
        goals[str(name)] = str(text)
    return goals


def _formalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def _export_bundle_from_shards(args: argparse.Namespace) -> None:
    """Warm-start bundle straight from the checkpoint shards (no VLM load)."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    snapshot = Path(
        snapshot_download(args.model_path, allow_patterns=["*.safetensors*", "config.json"])
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    trunk_keys = [
        key
        for key in weight_map
        if key.startswith("action_head.")
        and key.removeprefix("action_head.").startswith(PRETRAINED_KEEP_PREFIXES)
    ]
    if not trunk_keys:
        msg = f"{args.model_path}: no action-head trunk keys found in the index."
        raise RuntimeError(msg)
    shards = {weight_map[key] for key in trunk_keys}
    trunk: dict[str, torch.Tensor] = {}
    for shard in sorted(shards):
        tensors = load_file(str(snapshot / shard))
        for key in trunk_keys:
            if weight_map[key] == shard:
                trunk[key.removeprefix("action_head.")] = tensors[key]
    source_config = json.loads((snapshot / "config.json").read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "action_head_trunk.pt"
    torch.save(
        {
            "trunk_state_dict": trunk,
            "source_model_path": str(args.model_path),
            "source_snapshot": str(snapshot),
            "source_config": source_config,
            "gr00t_submodule_commit": gr00t_submodule_commit(),
        },
        bundle_path,
    )
    save_provenance(
        args.output_dir / "action_head_trunk_provenance.json",
        {
            "source_model_path": str(args.model_path),
            "source_snapshot": str(snapshot),
            "num_trunk_tensors": len(trunk),
            "num_trunk_params": int(sum(v.numel() for v in trunk.values())),
            "trunk_sha256": {
                key: hashlib.sha256(
                    value.cpu().contiguous().float().numpy().tobytes()
                ).hexdigest()
                for key, value in sorted(trunk.items())
            },
        },
    )
    print(
        f"[PASS] exported warm-start bundle ({len(trunk)} tensors, "
        f"{sum(v.numel() for v in trunk.values())/1e6:.1f}M params) -> {bundle_path}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    if args.bundle_only:
        _export_bundle_from_shards(args)
        return
    if (args.goals_json is None) == (args.language_sidecar is None):
        msg = "Provide exactly one of --goals_json or --language_sidecar."
        raise SystemExit(msg)
    ensure_gr00t_importable()
    from transformers import AutoProcessor  # noqa: PLC0415
    from transformers.feature_extraction_utils import BatchFeature  # noqa: PLC0415

    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7  # noqa: PLC0415

    if args.language_sidecar is not None:
        goals = _load_goals_from_sidecar(args.language_sidecar)
    else:
        goals = _load_goals(args.goals_json)
    device = torch.device(args.device)

    print(f"[cache] loading {args.model_path} ...", flush=True)
    model = Gr00tN1d7.from_pretrained(args.model_path)
    model.eval()
    # bf16 matches GR00T's deployment path (load_bf16) and is required by the
    # checkpoint's flash-attention setting; features are stored as float32.
    backbone = model.backbone.to(device=device, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        getattr(model.config, "model_name", "nvidia/Cosmos-Reason2-2B")
    )

    texts = []
    names = list(goals.keys())
    for name in names:
        text = goals[name]
        if args.formalize_language:
            text = _formalize(text)
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": text}]}
        ]
        texts.append(
            processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
        )
    tokenized = processor(text=texts, images=None, return_tensors="pt", padding=True)

    with torch.no_grad():
        vl_input = BatchFeature(
            data={
                "input_ids": tokenized["input_ids"].to(device),
                "attention_mask": tokenized["attention_mask"].to(device),
                "pixel_values": None,
                "image_grid_thw": None,
            }
        )
        backbone_output = backbone(vl_input)

    features = backbone_output["backbone_features"].float().cpu()
    attention_mask = backbone_output["backbone_attention_mask"].cpu()
    if bool(backbone_output["image_mask"].any()):
        msg = "Text-only forward produced image tokens; refusing to cache."
        raise RuntimeError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "goal_names": names,
        "goal_texts": {name: goals[name] for name in names},
        "features": features,
        "attention_mask": attention_mask,
        "input_ids": tokenized["input_ids"].cpu(),
    }
    features_path = args.output_dir / "goal_features.pt"
    torch.save(table, features_path)

    def _sha(t: torch.Tensor) -> str:
        return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()

    provenance = {
        "model_path": str(args.model_path),
        "gr00t_submodule_commit": gr00t_submodule_commit(),
        "formalize_language": bool(args.formalize_language),
        "num_goals": len(names),
        "feature_shape": list(features.shape),
        "features_sha256": _sha(features),
        "attention_mask_sha256": _sha(attention_mask.long()),
        "backbone_select_layer": int(getattr(model.config, "select_layer", -1)),
    }
    save_provenance(args.output_dir / "goal_features_provenance.json", provenance)
    print(f"[PASS] cached {len(names)} goals -> {features_path}", flush=True)

    if args.export_head_bundle:
        head_state = model.action_head.state_dict()
        trunk = {
            key: value
            for key, value in head_state.items()
            if key.startswith(PRETRAINED_KEEP_PREFIXES)
        }
        if not trunk:
            msg = "Checkpoint action head has no trunk keys; wrong checkpoint?"
            raise RuntimeError(msg)
        config_record = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
        bundle_path = args.output_dir / "action_head_trunk.pt"
        torch.save(
            {
                "trunk_state_dict": trunk,
                "source_model_path": str(args.model_path),
                "source_config": config_record,
                "gr00t_submodule_commit": gr00t_submodule_commit(),
            },
            bundle_path,
        )
        save_provenance(
            args.output_dir / "action_head_trunk_provenance.json",
            {
                "source_model_path": str(args.model_path),
                "num_trunk_tensors": len(trunk),
                "num_trunk_params": int(sum(v.numel() for v in trunk.values())),
                "trunk_sha256": {
                    key: hashlib.sha256(
                        value.cpu().contiguous().float().numpy().tobytes()
                    ).hexdigest()
                    for key, value in sorted(trunk.items())
                },
            },
        )
        print(f"[PASS] exported warm-start bundle -> {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
