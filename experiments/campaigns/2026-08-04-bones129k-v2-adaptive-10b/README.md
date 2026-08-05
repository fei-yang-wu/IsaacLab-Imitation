# BONES-SEED 129k adaptive-reset 10B run

This is the low-level-from-scratch follow-up to the completed 1B BONES-SEED v2
run. It reuses the accepted deterministic root+qpos encoder and changes only
the low-level reset distribution to the existing `sonic` preset:

```text
env.command_interface.reference.selection=sonic
```

That preset jointly samples trajectory/frame bins over the complete motion,
uses sequence-length-agnostic weighting so long clips do not dominate merely
by having more bins, assigns 90% of probability from observed failure rates,
and retains 10% uniform exploration. Evaluation remains pinned to named
motions and frame 0; adaptive reset sampling is training-only.

The scaling geometry is 32,768 environments × 6 steps on the local 96 GiB RTX
PRO 6000. The prior 1B run established this as the largest geometry with a
reasonable VRAM reserve. The 10B cap is 50,863 iterations and 10,000,072,704
actual environment frames. The low-level policy starts from scratch; encoder
pretraining is not repeated.

## Reference data

The full dataset must pass a content gate at 129,785 motions and 47,491,234
transitions either way. `DATA_SOURCE` picks how it is read:

`DATA_SOURCE=arrays` (default) memory-maps prebuilt training-shaped arrays from
NVMe and opens neither the Zarr nor the replay:

```text
/mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1
```

`DATA_SOURCE=replay` keeps the original path through the versioned replay cache,
retained so the two can be compared directly:

```text
/mnt/storage/fwu91/bones_seed_full/rb/g1_bones_seed_sonic_full_129785_e714bbff_v2
```

Why the default moved: derived from the replay, the macro and runtime caches
read about 133 GB to keep about 55 GB, and `body_pos_w` plus `body_quat_w` are
read twice, once by each. `/mnt/storage` is a 7200-rpm disk, so that is 12-20
minutes on every process start. The arrays are already in both caches' layout,
so a launch reads about 50 GB sequentially from NVMe instead. Build them with
`python -m imitation_experiments.data.build_reference_arrays`; see
`.agents/skills/bones-seed-dataset/references/full-129k-cache.md` for the exact
command and the measured equivalence against the replay.

Print and validate the command:

```bash
STAGE=plan \
  experiments/campaigns/2026-08-04-bones129k-v2-adaptive-10b/run.sh
```

After the cache passes and the W&B group `bones129k-v2-adaptive-reset` is
confirmed, launch locally:

```bash
STAGE=lowlevel \
CONFIRM_RUN=bones129k-v2-adaptive-10b \
  experiments/campaigns/2026-08-04-bones129k-v2-adaptive-10b/run.sh
```

Expected output root:

```text
/mnt/storage/fwu91/bones_seed_full/runs/bones129k_root_qpos_v2_adaptive_e32768_r6_10b_seed0
```

Monitor the reference-end fraction together with reset-start phase, failure
causes, local MPJPE, and global/root tracking. The old `[0,200]` sampler clamps
out-of-range starts to the final frame on short trajectories, so its
`reference_finished` fraction is not directly comparable to this run.
