#!/usr/bin/env python
"""CPU pre-flight for the grouped-VQ capacity grid.

Builds every (groups, categories) point of the ablation with the exact
``gumbel_multicat`` spec the launchers use, then checks that each one:

* constructs at z_dim=256 (groups must divide z_dim),
* produces a finite z of the right shape plus a finite regularizer in both the
  sampled and deterministic paths,
* selects codes inside ``[0, categories)`` with one index per group,
* survives a state-dict round-trip with tensor-identical deterministic output.

This costs seconds and catches a broken grid point before any Isaac Lab or
cluster time is spent. It is a wiring check, not a quality check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rlopt.agent.hl_skill_encoder import SkillLatentSpec, build_skill_encoder

# Mirrors groupvq_grid.sh; keep the two tables in sync.
GRID: tuple[tuple[str, int, int], ...] = (
    ("g16_c128", 16, 128),
    ("g32_c128", 32, 128),
    ("g64_c128", 64, 128),
    ("g128_c128", 128, 128),
    ("g64_c16", 64, 16),
    ("g64_c64", 64, 64),
    ("g64_c512", 64, 512),
)

Z_DIM = 256
STATE_DIM = 93
WINDOW_STEPS = 10
HIDDEN_DIMS = (1024, 512, 512)


def check_arm(name: str, groups: int, categories: int, *, batch: int) -> dict:
    torch.manual_seed(0)
    spec = SkillLatentSpec(
        latent_mode="gumbel_multicat",
        categorical_groups=groups,
        categorical_categories=categories,
        gumbel_hard=True,
    )
    encoder = build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=WINDOW_STEPS,
        z_dim=Z_DIM,
        hidden_dims=HIDDEN_DIMS,
        spec=spec,
    )
    state = torch.randn(batch, STATE_DIM)
    window = torch.randn(batch, WINDOW_STEPS, STATE_DIM)

    z_sampled, reg, _ = encoder.encode(state, window, deterministic=False, step=0)
    if tuple(z_sampled.shape) != (batch, Z_DIM):
        msg = f"{name}: sampled z shape {tuple(z_sampled.shape)} != {(batch, Z_DIM)}"
        raise AssertionError(msg)
    if not torch.isfinite(z_sampled).all() or not torch.isfinite(reg):
        msg = f"{name}: non-finite sampled z or regularizer"
        raise AssertionError(msg)

    z_det = encoder(state, window)
    if not torch.isfinite(z_det).all():
        msg = f"{name}: non-finite deterministic z"
        raise AssertionError(msg)

    raw = encoder._raw(state, window)  # noqa: SLF001 - grid wiring check
    _, code, _, _ = encoder._quantize(  # noqa: SLF001
        encoder._pre_quantize(raw),  # noqa: SLF001
        deterministic=True,
        step=None,
    )
    if tuple(code.shape) != (batch, groups):
        msg = f"{name}: code shape {tuple(code.shape)} != {(batch, groups)}"
        raise AssertionError(msg)
    if int(code.min()) < 0 or int(code.max()) >= categories:
        msg = f"{name}: code index outside [0, {categories})"
        raise AssertionError(msg)

    clone = build_skill_encoder(
        state_dim=STATE_DIM,
        window_steps=WINDOW_STEPS,
        z_dim=Z_DIM,
        hidden_dims=HIDDEN_DIMS,
        spec=spec,
    )
    clone.load_state_dict(encoder.state_dict())
    if not torch.equal(clone(state, window), z_det):
        msg = f"{name}: state-dict round-trip changed the deterministic latent"
        raise AssertionError(msg)

    metrics = encoder.diversity_metrics(state, window)
    return {
        "arm": name,
        "groups": groups,
        "categories": categories,
        "code_dim": Z_DIM // groups,
        "nominal_bits": int(groups * (categories.bit_length() - 1)),
        "encoder_parameters": int(sum(p.numel() for p in encoder.parameters())),
        "codebook_parameters": int(encoder.codebook.numel()),
        "code_perplexity": float(metrics["code_perplexity"]),
        "code_usage_frac": float(metrics["code_usage_frac"]),
        "effective_rank": float(metrics["effective_rank"]),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the per-arm records.",
    )
    args = parser.parse_args()

    records: list[dict] = []
    status = 0
    for name, groups, categories in GRID:
        try:
            record = check_arm(name, groups, categories, batch=args.batch)
        except Exception as exc:  # noqa: BLE001 - report every failing grid point
            record = {
                "arm": name,
                "groups": groups,
                "categories": categories,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = 1
        records.append(record)
        flag = "PASS" if record["passed"] else "FAIL"
        print(f"[{flag}] {json.dumps(record)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"grid": records}, indent=2) + "\n")
        print(f"[INFO] Wrote {args.output}")
    return status


if __name__ == "__main__":
    sys.exit(main())
