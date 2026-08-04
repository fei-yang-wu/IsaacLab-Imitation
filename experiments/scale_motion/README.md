# BONES-SEED Motion Scaling

Select a deterministic BONES-SEED subset, then train the current G1 v2 IPMD
policy with DiffSR latent control. Run every command from the repository root.

## Install

```bash
pixi install -e isaaclab
```

## Select 5,000 motions

This creates a manifest, matching language sidecar, and required-shard list.
The source and selection hashes make the ordered motion set identical across
machines.

```bash
pixi run python experiments/scale_motion/select_bones_seed_motion.py \
  --source-manifest data/bones_seed_sonic_129k_50hz/g1_bones_seed_sonic_full_manifest.json \
  --npz-root data/bones_seed_sonic_129k_50hz/npz/g1 \
  --language-sidecar data/bones_seed_sonic_129k_50hz/g1_bones_seed_sonic_full_language.json \
  --output-manifest data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected.json \
  --output-language-sidecar data/bones_seed_sonic_129k_50hz/language/bones-seed-sonic-5000-selected_language.json \
  --shard-index data/bones_seed_sonic_129k_50hz/shard_index.json \
  --required-shards-output data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected_shards.txt \
  --count 5000 \
  --shuffle-seed 0 \
  --require-files \
  --expected-source-manifest-sha256 eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688 \
  --expected-selected-names-sha256 729a39e8cd8da75d946cc50294ac19bcaaf399551b9c8383f221f90374bd739b
```

Expected result:

```text
selected motions: 5000
required shards: 103
selected names SHA-256: 729a39e8cd8da75d946cc50294ac19bcaaf399551b9c8383f221f90374bd739b
```

Do not add `--available-files-only` for a cross-machine selection. It changes
the candidate pool based on files present on each machine.

## Choose the training manifest

Set `MANIFEST_PATH` to the selected manifest. Give each distinct manifest a
separate `DATASET_PATH`; caches from another selection must not be reused.

Preview the command:

```bash
MANIFEST_PATH=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected.json \
DATASET_PATH=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000_selected \
DRY_RUN=1 \
LOGGER_BACKEND=none \
bash scripts/rlopt/run_local_v2_pipeline.sh
```

Build the new cache and run a 10M-frame local check:

```bash
MANIFEST_PATH=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected.json \
DATASET_PATH=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000_selected \
CACHE_REFRESH=true \
TOTAL_FRAMES=10000000 \
LOGGER_BACKEND=none \
bash scripts/rlopt/run_local_v2_pipeline.sh
```

Use `CACHE_REFRESH=true` only for the first cache build. Later runs use the
same manifest and cache with `CACHE_REFRESH=false`.

To train from the existing validated 5,000-motion artifacts instead:

```bash
MANIFEST_PATH=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000.json \
DATASET_PATH=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000 \
CACHE_REFRESH=false \
TOTAL_FRAMES=10000000 \
LOGGER_BACKEND=none \
bash scripts/rlopt/run_local_v2_pipeline.sh
```

## Small smoke run

```bash
MANIFEST_PATH=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected.json \
DATASET_PATH=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000_selected \
CACHE_REFRESH=true \
PRETRAIN_UPDATES=1 \
NUM_ENVS=16 \
TOTAL_FRAMES=384 \
LOGGER_BACKEND=none \
bash scripts/rlopt/run_local_v2_pipeline.sh
```

## Reuse a skill encoder

```bash
MANIFEST_PATH=data/bones_seed_sonic_129k_50hz/manifests/bones-seed-sonic-5000-selected.json \
DATASET_PATH=data/bones_seed_sonic_129k_50hz/g1_hl_diffsr_5000_selected \
SKIP_PRETRAIN=1 \
SKILL_CKPT=/absolute/path/to/checkpoints/latest.pt \
TOTAL_FRAMES=10000000 \
LOGGER_BACKEND=none \
bash scripts/rlopt/run_local_v2_pipeline.sh
```

The encoder must use horizon 10 and `z_dim=256`. For a matched run, use an
encoder trained on the same selected manifest.

## Active configuration

```text
task: Isaac-Imitation-G1-v2
algorithm: IPMD
agent: rlopt_ipmd_tuned_cfg_entry_point
skill source: hl_skill
encoder: deterministic DiffSR/det-SR
horizon: 10 steps
latent command: 256 code + 2 phase values = 258
physics: newton_mjwarp
```

Do not use `run_local_pretrain_lowlevel.sh`; it targets the frozen pre-v2 task.
Use about 10M local frames for debugging and at most 50M for a serious check.
