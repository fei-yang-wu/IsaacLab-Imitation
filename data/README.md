# Local Data Layout

This directory is for local and generated motion assets. Most of it remains git-ignored on purpose.

Tracked manifests and templates live under `source/isaaclab_imitation/isaaclab_imitation/manifests/`:

- `g1_lafan1_manifest.template.json`: tracked template for a full local G1 LAFAN1 manifest

Expected local paths:

- `data/lafan1/raw/g1/`: downloaded or source CSV motions
- `data/lafan1/npz/g1/`: converted G1 NPZ motions
- `data/lafan1/manifests/g1_lafan1_manifest.json`: full local manifest
- `data/lafan1/manifests/g1_debug_manifest.json`: smaller local subset manifest

Common flows:

1. Download the Hugging Face G1 dataset and prepare NPZ plus a full manifest:

```bash
./scripts/data/download_g1_lafan1_data.sh
```

Equivalent lower-level Python command:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
    --prepare-npz --headless
```

For the G1 CSV set, you can auto-trim the common arms-up alignment pose while
building HF-ready NPZ files with:

```bash
pixi run -e isaaclab python scripts/data/setup_lafan1_dataset.py \
    --prepare-npz --headless \
    --auto_trim_mode g1_shoulder_roll
```

2. If NPZ files already exist, regenerate the full manifest:

```bash
pixi run python scripts/data/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

3. If you want to hand-edit a manifest instead of generating one, copy the tracked template:

```bash
mkdir -p data/lafan1/manifests
cp source/isaaclab_imitation/isaaclab_imitation/manifests/g1_lafan1_manifest.template.json \
   data/lafan1/manifests/g1_lafan1_manifest.json
```

After copying, replace the placeholder motion names and paths with your local NPZ files.

If you already have raw CSV files plus existing NPZ files and only want to add
per-motion trim ranges to the manifest, use:

```bash
pixi run -e isaaclab python scripts/data/prepare_lafan1_from_csv.py \
    --csv_dir data/lafan1/raw/g1 \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json \
    --recursive \
    --assume_npz_exists \
    --auto_trim_mode g1_shoulder_roll
```

For the generic `Isaac-Imitation-G1-LafanTrack-v0` task, pass your manifest explicitly with:

```bash
env.lafan1_manifest_path=./data/lafan1/manifests/g1_lafan1_manifest.json
```
