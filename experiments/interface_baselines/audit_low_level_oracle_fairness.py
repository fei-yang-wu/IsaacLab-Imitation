#!/usr/bin/env python3
"""Audit oracle low-level controller parity across command representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check low-level oracle eval summaries before planner comparison."
    )
    parser.add_argument("summary_jsons", nargs="+", type=Path)
    parser.add_argument(
        "--survival_metric",
        default="aggregate.survival_steps_mean",
        help="Dot path for the survival/completion metric.",
    )
    parser.add_argument(
        "--return_metric",
        default="aggregate.return_sum_mean",
        help="Dot path for the return/tracking metric.",
    )
    parser.add_argument(
        "--survival_rel_tol",
        type=float,
        default=0.05,
        help="Allowed relative spread for survival metric.",
    )
    parser.add_argument(
        "--return_rel_tol",
        type=float,
        default=0.05,
        help="Allowed relative spread for return metric.",
    )
    parser.add_argument("--json_out", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return payload


def _get_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Metric path {path!r} not found.")
        current = current[part]
    return current


def _as_float(value: Any, *, metric: str, path: Path) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} metric {metric!r} is not numeric: {value!r}") from exc


def _label(payload: dict[str, Any], path: Path) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in ("interface", "label", "planner_type"):
            value = metadata.get(key)
            if value not in (None, ""):
                return str(value)
    return path.parent.name or path.stem


def _spread_check(rows: list[dict[str, Any]], metric: str, rel_tol: float) -> dict[str, Any]:
    values = [float(row[metric]) for row in rows]
    min_value = min(values)
    max_value = max(values)
    spread = max_value - min_value
    scale = max(1.0, abs(sum(values) / len(values)))
    allowed = float(rel_tol) * scale
    return {
        "metric": metric,
        "min": min_value,
        "max": max_value,
        "spread": spread,
        "relative_tolerance": float(rel_tol),
        "allowed_spread": allowed,
        "pass": bool(spread <= allowed),
    }


def main() -> None:
    args = _parse_args()
    if float(args.survival_rel_tol) < 0.0 or float(args.return_rel_tol) < 0.0:
        raise ValueError("Relative tolerances must be non-negative.")
    rows: list[dict[str, Any]] = []
    for path in args.summary_jsons:
        resolved = path.expanduser().resolve()
        payload = _load_json(resolved)
        rows.append(
            {
                "path": str(resolved),
                "label": _label(payload, resolved),
                "survival": _as_float(
                    _get_path(payload, str(args.survival_metric)),
                    metric=str(args.survival_metric),
                    path=resolved,
                ),
                "return": _as_float(
                    _get_path(payload, str(args.return_metric)),
                    metric=str(args.return_metric),
                    path=resolved,
                ),
            }
        )
    checks = [
        _spread_check(rows, "survival", float(args.survival_rel_tol)),
        _spread_check(rows, "return", float(args.return_rel_tol)),
    ]
    payload = {
        "pass": bool(all(check["pass"] for check in checks)),
        "checks": checks,
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        output = args.json_out.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not payload["pass"]:
        raise SystemExit("Low-level oracle fairness audit failed.")


if __name__ == "__main__":
    main()
