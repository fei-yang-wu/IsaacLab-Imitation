#!/usr/bin/env python3
"""Preflight checks for fair interface-baseline comparison launches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SUPPORTED_MODEL_SIZES = {"tiny", "small", "medium", "large"}
HAND_DESIGNED_INTERFACES = {"ee_trajectory", "full_body_trajectory"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train_manifest",
        type=Path,
        default=Path(
            _env(
                "TRAIN_MANIFEST",
                _env(
                    "MANIFEST",
                    "data/unitree/manifests/g1_unitree_dance102_manifest.json",
                ),
            )
        ),
    )
    parser.add_argument(
        "--eval_manifest",
        type=Path,
        default=Path(
            _env(
                "EVAL_MANIFEST",
                _env(
                    "MANIFEST",
                    "data/unitree/manifests/g1_unitree_dance102_manifest.json",
                ),
            )
        ),
    )
    parser.add_argument(
        "--interfaces",
        nargs="*",
        default=_split_words(_env("INTERFACES", "full_body_trajectory ee_trajectory")),
        help="Hand-designed interfaces that will be launched.",
    )
    parser.add_argument(
        "--run_latent",
        action=argparse.BooleanOptionalAction,
        default=_env("RUN_LATENT_BASELINE", _env("RUN_LATENT", "0")) == "1",
    )
    parser.add_argument(
        "--full_body_checkpoint",
        type=Path,
        default=_path_env(
            "FULL_BODY_TRAJECTORY_CHECKPOINT", _env("LOW_LEVEL_CHECKPOINT")
        ),
    )
    parser.add_argument(
        "--ee_checkpoint",
        type=Path,
        default=_path_env("EE_TRAJECTORY_CHECKPOINT", _env("LOW_LEVEL_CHECKPOINT")),
    )
    parser.add_argument(
        "--latent_low_level_checkpoint",
        type=Path,
        default=_path_env("LATENT_LOW_LEVEL_CHECKPOINT"),
    )
    parser.add_argument(
        "--latent_skill_checkpoint",
        type=Path,
        default=_path_env("LATENT_SKILL_CHECKPOINT"),
    )
    parser.add_argument(
        "--latent_planner_checkpoint",
        type=Path,
        default=_path_env("LATENT_PLANNER_CHECKPOINT"),
    )
    parser.add_argument(
        "--latent_dataset_path",
        type=Path,
        default=_path_env("LATENT_DATASET_PATH"),
        help="Latent skill dataset path used by latent held-out runs.",
    )
    parser.add_argument(
        "--require_latent_dataset_path",
        action="store_true",
        default=False,
        help="Require --latent_dataset_path when --run_latent is enabled.",
    )
    parser.add_argument(
        "--allow_missing_latent_dataset",
        action="store_true",
        default=False,
        help="Allow --latent_dataset_path to be absent, for planning-only dry runs.",
    )
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=_split_words(_env("MODEL_SIZES", _env("MODEL_SIZE", "medium"))),
    )
    parser.add_argument(
        "--sample_budgets",
        nargs="*",
        default=_split_words(_env("SAMPLE_BUDGETS", "10000")),
    )
    parser.add_argument(
        "--output_roots",
        nargs="*",
        type=Path,
        default=[],
        help="Planned output roots. Existing roots are warnings unless --fail_if_output_exists is set.",
    )
    parser.add_argument("--fail_if_output_exists", action="store_true", default=False)
    parser.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        default=False,
        help="Only for dry-runs on machines without checkpoints.",
    )
    parser.add_argument(
        "--allow_missing_motion_files",
        action="store_true",
        default=False,
        help="Allow manifest-referenced motion files to be absent.",
    )
    return parser.parse_args()


def _split_words(value: str) -> list[str]:
    return [part for part in str(value).split() if part]


def _path_env(name: str, default: str = "") -> Path | None:
    value = _env(name, default)
    return Path(value) if value else None


def _is_existing_file(path: Path | None) -> bool:
    return path is not None and path.expanduser().is_file()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.expanduser().is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest is not a JSON object: {path}")
    return payload


def _manifest_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        return []
    trajectories = dataset.get("trajectories")
    if not isinstance(trajectories, dict):
        return []
    entries: list[dict[str, Any]] = []
    for value in trajectories.values():
        if isinstance(value, list):
            entries.extend(entry for entry in value if isinstance(entry, dict))
    return entries


def _check_manifest(path: Path, *, allow_missing_motion_files: bool) -> list[str]:
    warnings: list[str] = []
    payload = _load_manifest(path)
    entries = _manifest_entries(payload)
    if not entries:
        raise ValueError(f"Manifest has no trajectory entries: {path}")
    manifest_dir = path.expanduser().resolve().parent
    relative_to_manifest = bool(
        payload.get("metadata", {}).get("paths_are_relative_to_manifest", True)
        if isinstance(payload.get("metadata"), dict)
        else True
    )
    missing_motion_files: list[str] = []
    for entry in entries:
        raw_path = entry.get("path")
        if raw_path in (None, ""):
            missing_motion_files.append(f"<missing path for {entry.get('name', '?')}>")
            continue
        motion_path = Path(str(raw_path)).expanduser()
        if not motion_path.is_absolute() and relative_to_manifest:
            motion_path = manifest_dir / motion_path
        if not motion_path.is_file():
            missing_motion_files.append(str(motion_path))
    if missing_motion_files and not allow_missing_motion_files:
        sample = "\n  ".join(missing_motion_files[:5])
        raise FileNotFoundError(
            f"Manifest {path} references missing motion files:\n  {sample}"
        )
    if missing_motion_files:
        warnings.append(
            f"Manifest {path} has {len(missing_motion_files)} missing motion files."
        )
    return warnings


def _check_checkpoint(
    label: str,
    path: Path | None,
    *,
    allow_missing_checkpoints: bool,
) -> list[str]:
    if _is_existing_file(path):
        return []
    if allow_missing_checkpoints:
        return [f"Missing checkpoint allowed for {label}: {path}"]
    raise FileNotFoundError(f"Missing checkpoint for {label}: {path}")


def _check_existing_path(
    label: str,
    path: Path | None,
    *,
    required: bool,
    allow_missing: bool,
) -> list[str]:
    if path is None:
        if required and not allow_missing:
            raise FileNotFoundError(f"Missing required path for {label}: {path}")
        if required:
            return [f"Missing required path allowed for {label}: {path}"]
        return []
    if path.expanduser().exists():
        return []
    if allow_missing:
        return [f"Missing path allowed for {label}: {path}"]
    raise FileNotFoundError(f"Missing path for {label}: {path}")


def _check_model_sizes(model_sizes: list[str]) -> None:
    missing = [size for size in model_sizes if size not in SUPPORTED_MODEL_SIZES]
    if missing:
        raise ValueError(
            f"Unsupported MODEL_SIZES={missing}; expected {sorted(SUPPORTED_MODEL_SIZES)}"
        )


def _check_sample_budgets(sample_budgets: list[str]) -> None:
    for budget in sample_budgets:
        if budget in {"all", "0"}:
            continue
        try:
            value = int(budget)
        except ValueError as exc:
            raise ValueError(f"Invalid sample budget: {budget}") from exc
        if value <= 0:
            raise ValueError(f"Sample budget must be positive, all, or 0: {budget}")


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    warnings: list[str] = []
    warnings.extend(
        _check_manifest(
            args.train_manifest,
            allow_missing_motion_files=bool(args.allow_missing_motion_files),
        )
    )
    if (
        args.eval_manifest.expanduser().resolve()
        != args.train_manifest.expanduser().resolve()
    ):
        warnings.extend(
            _check_manifest(
                args.eval_manifest,
                allow_missing_motion_files=bool(args.allow_missing_motion_files),
            )
        )
    _check_model_sizes([str(size) for size in args.model_sizes])
    _check_sample_budgets([str(budget) for budget in args.sample_budgets])

    interfaces = set(str(interface) for interface in args.interfaces)
    unknown = interfaces - HAND_DESIGNED_INTERFACES
    if unknown:
        raise ValueError(
            f"Unsupported hand-designed interfaces: {sorted(unknown)}; "
            f"expected {sorted(HAND_DESIGNED_INTERFACES)}"
        )
    if "full_body_trajectory" in interfaces:
        warnings.extend(
            _check_checkpoint(
                "full_body_trajectory",
                args.full_body_checkpoint,
                allow_missing_checkpoints=bool(args.allow_missing_checkpoints),
            )
        )
    if "ee_trajectory" in interfaces:
        warnings.extend(
            _check_checkpoint(
                "ee_trajectory",
                args.ee_checkpoint,
                allow_missing_checkpoints=bool(args.allow_missing_checkpoints),
            )
        )
    if bool(args.run_latent):
        for label, checkpoint in (
            ("latent_low_level", args.latent_low_level_checkpoint),
            ("latent_skill", args.latent_skill_checkpoint),
            ("latent_planner", args.latent_planner_checkpoint),
        ):
            warnings.extend(
                _check_checkpoint(
                    label,
                    checkpoint,
                    allow_missing_checkpoints=bool(args.allow_missing_checkpoints),
                )
            )
        warnings.extend(
            _check_existing_path(
                "latent_dataset",
                args.latent_dataset_path,
                required=bool(args.require_latent_dataset_path),
                allow_missing=bool(args.allow_missing_latent_dataset),
            )
        )
    existing_roots = [
        str(path)
        for path in args.output_roots
        if path is not None and path.expanduser().exists()
    ]
    if existing_roots and args.fail_if_output_exists:
        raise FileExistsError(f"Output roots already exist: {existing_roots}")
    if existing_roots:
        warnings.append(
            f"Output roots already exist and may be reused: {existing_roots}"
        )

    return {
        "status": "pass",
        "interfaces": sorted(interfaces),
        "run_latent": bool(args.run_latent),
        "latent_dataset_path": ""
        if args.latent_dataset_path is None
        else str(args.latent_dataset_path),
        "model_sizes": list(args.model_sizes),
        "sample_budgets": list(args.sample_budgets),
        "warnings": warnings,
    }


def main() -> None:
    report = run_preflight(_parse_args())
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")
    print(
        "[PASS] Preflight ok: interfaces={interfaces} latent={latent} "
        "model_sizes={model_sizes} sample_budgets={sample_budgets}".format(
            interfaces=",".join(report["interfaces"]),
            latent=report["run_latent"],
            model_sizes=",".join(report["model_sizes"]),
            sample_budgets=",".join(report["sample_budgets"]),
        )
    )


if __name__ == "__main__":
    main()
