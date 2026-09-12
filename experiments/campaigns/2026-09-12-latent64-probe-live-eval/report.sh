#!/usr/bin/env bash
# Mirror the scored rows of the three arms and print a matched-frame table.
#   ./report.sh
set -euo pipefail
CAMPAIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${CAMPAIGN_DIR}" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
REMOTE="${REMOTE:-ice}"
EVAL_DIR="${EVAL_DIR:-scratch/Research/IsaacLab/data/eval/latest_eval}"
LOCAL="${LOCAL:-${REPO_ROOT}/logs/latent64_probe_live_eval}"
mkdir -p "${LOCAL}"
rsync -aq --include='z64_merged_seed0_clean_f*.json' --include='z64_merged_noreg_seed0_clean_f*.json' --include='z64_poe_seed0_clean_f*.json' --exclude='*' "${REMOTE}:${EVAL_DIR}/" "${LOCAL}/"
pixi run python - "${LOCAL}" <<'PY'
import json, re, sys
from pathlib import Path
rows = {}
for p in Path(sys.argv[1]).glob("*_seed0_clean_f*.json"):
    m = re.match(r"(.+)_seed0_clean_f(\d+)\.json", p.name)
    d = json.loads(p.read_text()); ok = d.get("successful_metrics") or {}
    rows[(m.group(1), int(m.group(2)))] = (
        d["aggregate"]["tracking_success_rate"],
        (ok.get("tracking_mpjpe_mm") or {}).get("mean"),
        (ok.get("tracking_mpjpe_g_mm") or {}).get("mean"),
        (ok.get("body_jerk_mps3") or {}).get("mean"),
    )
arms = ["z64_merged", "z64_merged_noreg", "z64_poe"]
frames = sorted({f for _, f in rows})
print("bones_testbed4096_v1 clean, seed 0, one pass per checkpoint. cells: SR / MPJPE-L / MPJPE-G / jerk")
print(f"{'frames':>8} | " + " | ".join(f"{a:^34}" for a in arms))
for f in frames:
    cells = []
    for a in arms:
        r = rows.get((a, f))
        cells.append("-" if r is None else f"{r[0]:.4f} / {r[1]:.2f} / {r[2]:.2f} / {r[3]:.0f}" if r[1] is not None else f"{r[0]:.4f}")
    print(f"{f/1e9:>7.2f}B | " + " | ".join(f"{c:^34}" for c in cells))
PY
