"""Cluster entrypoint for the skill-composition probe driver.

A cluster stage runs one script path inside the container; this forwards to
`imitation_experiments.evaluation.composition_probe.main`, which spawns one
evaluator process per setting through `run_evaluator.py`. The name matches
`run_singularity.sh`'s `scripts/rlopt/eval*.py` branch, which exports the
CU130 torch site-packages, the NVIDIA library path and the NCCL preload
before `/isaac-sim/python.sh`; the children inherit that environment. Under
any other name the stage falls to the bare `/isaac-sim/python.sh` branch and
every child dies on `libtorch_cuda.so: undefined symbol: ncclDevCommDestroy`
(jobs 5627467-5627475, 2026-09-02).

    python scripts/rlopt/eval_composition_probe.py --arm lstm_affine \\
        --checkpoint newest:/data/lstm_hub64_10b/lstm_affine_seed0/tracker \\
        --encoder /data/past_chunk_affine_64d/p5_affine_seed0/encoder/checkpoints/latest.pt \\
        --override agent.ppo.rnn_hidden_size=256 --pairs pairs.json \\
        --test held_alpha --out /data/eval/composition_probe/lstm_affine \\
        --reference-arrays /storage/ice-shared/.../ref_arrays/...
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source" / "imitation_experiments"))

from imitation_experiments.evaluation.composition_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
