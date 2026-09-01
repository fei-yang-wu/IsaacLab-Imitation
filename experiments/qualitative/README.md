# Qualitative latent-interface ablation

Train an FSQ (`fsq64`) and an unquantized (`deter64`) encoder + tracker pair,
then analyze them qualitatively. All launchers are `.sh` files in this
directory; all Python lives in `src/`. Shared defaults, checkpoint resolution,
and data gates live in `qualitative_env.sh` (sourced by every launcher).

Common to every launcher:

- Run from the repository root.
- `DRY_RUN=1` validates checkpoints and data, prints the exact command, and
  launches nothing. Use it first.
- `SMOKE=1` runs a tiny variant into a separate `-smoke` output root.
- `OVERWRITE=1` replaces an existing output directory.
- Video needs **two render-capable GPUs visible** (GPUs 0-3 on this box), e.g.
  `CUDA_VISIBLE_DEVICES=1,3`. With one visible GPU, Kit segfaults. `VIDEO=0`
  disables video.
- `LATENT_ARM=fsq64|deter64` picks the arm (default `fsq64`).
  `TRACKER_ARM=tuned|sonic` picks the tracker capacity (default `tuned`).
  `qualitative_ncoord_intervention.sh` and
  `qualitative_skill_composability.sh` refuse `deter64` (no discrete code to
  edit or sample).
- Training output: `logs/ablate_latent/{encoder,lowlevel}/<run tag>`.
  Analysis output:
  `outputs/qualitative_analysis/bones129k-<arm>/<TRACKER_ARM>/<mode>/`, each
  with a `provenance.json`.

## Training scripts

| script | what it does |
|---|---|
| `pretrain_skill_encoder_fsq64.sh` | stage 1: pretrain the DiffSR skill encoder with the `sonic_fsq` bottleneck (64 coordinates x 32 levels) |
| `pretrain_skill_encoder_deter64.sh` | same, `deterministic` bottleneck (64 continuous values, no code) |
| `train_lowlevel_fsq64.sh` | stage 2: train the 50 Hz tracker on the frozen fsq64 encoder (5B frames — Skynet scale) |
| `train_lowlevel_deter64.sh` | same, on the frozen deter64 encoder |

```bash
DRY_RUN=1 bash experiments/qualitative/pretrain_skill_encoder_fsq64.sh
bash experiments/qualitative/pretrain_skill_encoder_fsq64.sh
TRACKER_ARM=sonic bash experiments/qualitative/train_lowlevel_fsq64.sh
```

Useful variables: `FRAME_CAP` and `TRAIN_NUM_ENVS` shrink stage 2 for a local
check; `ENCODER_CKPT` points stage 2 at a different accepted encoder;
`WANDB_GROUP` overrides the W&B group. Accept an encoder only after its
held-out `loss_real_z_eval` is flat and clearly below the zero-z and shuffled-z
controls.

## Analysis launchers

| launcher | what it does | steps a policy |
|---|---|---|
| `qualitative_reference_rollout.sh` | reference vs. tracked robot side by side, one MP4 per motion | yes |
| `qualitative_ncoord_intervention.sh` | resample N shared code coordinates mid-rollout across 32 robots | yes |
| `qualitative_latent_semantics.sh` | encode windows, KMeans-cluster them, render one video per cluster | encoder only |
| `qualitative_motion_switch_grid.sh` | 8 robots on 8 motions all switch to one shared motion, no reset | yes |
| `qualitative_skill_composability.sh` | 8 robots each chain fresh uniformly random codes every 2 s | yes |

### 1. Reference vs. rollout

```bash
CUDA_VISIBLE_DEVICES=1,3 NUM_MOTIONS=4 \
  bash experiments/qualitative/qualitative_reference_rollout.sh
```

| variable | meaning |
|---|---|
| `NUM_MOTIONS=4` | clips rendered, one motion each |
| `RANKS=` / `MOTIONS=` | pin explicit trajectories instead of the seeded draw |
| `MAX_STEPS=400` | cap each rollout |
| `FALL_HEIGHT=0.4` | base height counted as a fall |
| `SEED=1` | change the draw |

### 2. N-coordinate intervention (`fsq64` only)

Encodes a base code after a warmup, resamples `N_GROUPS` coordinates
(uniform over the 32 levels), and holds the edited code across 32 robots.

```bash
for n in 1 2 4 8 16 32; do
  CUDA_VISIBLE_DEVICES=1,3 N_GROUPS=$n \
    bash experiments/qualitative/qualitative_ncoord_intervention.sh
done
```

| variable | meaning |
|---|---|
| `N_GROUPS=4` | coordinates resampled, shared by all 32 robots |
| `WARMUP_SECONDS=2.0` | normal control before the edit |
| `ROLLOUT_STEPS=250` | steps the edited code is held |
| `SEED=1` | different base motion and replacement levels |

### 3. Latent semantics

Three stages: **encode** (Isaac; `NUM_MOTIONS` x `WINDOWS_PER_MOTION` windows
-> `latents.npz`), **cluster** (no Isaac; filter static/idle windows, KMeans,
PCA/t-SNE scatters, pick members per cluster), **gallery** (Isaac; one video
per cluster with its member motions replayed side by side, plus per-member
filmstrip PNGs). The robots replay reference poses; nothing tracks.

```bash
CUDA_VISIBLE_DEVICES=1,3 bash experiments/qualitative/qualitative_latent_semantics.sh

# Re-cluster and re-render without re-encoding.
SKIP_ENCODE=1 K_CLUSTERS=16 OVERWRITE=1 \
  bash experiments/qualitative/qualitative_latent_semantics.sh

# Re-render an existing clustering only.
SKIP_ENCODE=1 SKIP_CLUSTER=1 CLUSTERS=0,3,7 OVERWRITE=1 \
  bash experiments/qualitative/qualitative_latent_semantics.sh
```

| variable | meaning |
|---|---|
| `NUM_MOTIONS=4000` | motions drawn from the 129,785 |
| `WINDOWS_PER_MOTION=8` | windows per motion, spread over the whole clip |
| `K_CLUSTERS=24` | clusters, and therefore videos |
| `MEMBERS_PER_CLUSTER=8` | motions shown per cluster video |
| `MEMBER_SELECTION=farthest` | spread members across the cluster instead of the most typical (`centroid`) |
| `MIN_LOCAL_STEP=50` | drop windows starting in the first second (near-identical standing starts); 0 keeps all |
| `MIN_ROOT_SPEED=0.4` | with `MIN_LIMB_SPEED`, the static-window gate: keep a window when root speed >= this m/s OR root-relative top-5 body speed >= `MIN_LIMB_SPEED`; drop windows static on both counts. `0` disables a half |
| `MIN_LIMB_SPEED=0.6` | limb half of the gate (m/s); keeps in-place dances/kicks that root speed alone would drop |
| `EXCLUDE_MOTION_REGEX=idle` | drop every window of motions whose name matches (case-insensitive); empty keeps them |
| `LANGUAGE_JSON=` | language sidecar for the per-cluster `term_hint` |
| `LOOPS=2 SLOWDOWN=2` | playback: replays per clip, slowdown factor |
| `CONTEXT_FRAMES=25` | reference frames shown around the 10-frame window; 0 shows only the encoded window |
| `FILMSTRIP_MEMBERS=3` | members per cluster that get a still filmstrip; 0 disables |
| `FILMSTRIP_FRAMES=6` | frames per filmstrip |
| `FILMSTRIP_PX=300` | filmstrip tile height in px |
| `TSNE_ROWS=6000` | rows subsampled for the t-SNE scatter; 0 skips it |
| `CLUSTERS=0,3,7` | render only these clusters |
| `SKIP_ENCODE=1` / `SKIP_CLUSTER=1` | reuse an earlier stage |
| `SEED=1` | change the motion draw and clustering seed |

Note: clustering runs in the 64-D latent space; t-SNE/PCA only draw the result.

`src/qualitative_filmstrip_scene.py` re-renders the gallery filmstrips as
publication-style strips: one MuJoCo scene per member, one G1 copy per sampled
timestep, gray floor and background with shadows. No Isaac needed:

```bash
MUJOCO_GL=egl pixi run python experiments/qualitative/src/qualitative_filmstrip_scene.py \
  --gallery_dir outputs/qualitative_analysis/bones129k-sonic-fsq64/tuned/latent_semantics_gallery \
  --num_frames 6
```

Key flags: `--num_frames` (0 = the gallery's own frames), `--spacing`,
`--face_yaw_deg`, `--width`/`--height`, `--cam_distance`/`--cam_elevation`,
`--only cluster_003_member_1`. Output: `<gallery>/filmstrips_scene/` plus
`scene_strips.json` with each strip's frames, `term_hint`, and
`language_goal`.

### 4. Motion switch

```bash
CUDA_VISIBLE_DEVICES=1,3 bash experiments/qualitative/qualitative_motion_switch_grid.sh
```

| variable | meaning |
|---|---|
| `NUM_ROBOTS=8` | robots, and motions drawn |
| `SWITCH_AT_STEP=200` | steps on the first motion (200 = 4 s) |
| `AFTER_STEPS=150` | steps after the switch |
| `SWITCH_MOTION=` / `SWITCH_RANK=` | what everyone switches to |
| `SWITCH_COMMAND_FRAME=robot` | per-robot deployment command after the switch (default `reference`: identical command for all) |
| `SWITCH_ALIGN=none` | keep the dataset placement instead of aligning to each robot's xy |
| `MOTIONS=a,b,c` | pin the starting motions |
| `ENV_SPACING=4.0` | meters between robots on screen |
| `SEED=1` | change the draw |

### 5. Skill composability (`fsq64` only)

```bash
CUDA_VISIBLE_DEVICES=1,3 bash experiments/qualitative/qualitative_skill_composability.sh
```

| variable | meaning |
|---|---|
| `NUM_ROBOTS=8` | robots, each with its own code sequence |
| `SEGMENT_STEPS=100` | steps one code is held (must be a multiple of `HORIZON_STEPS`) |
| `NUM_SEGMENTS=10` | codes per robot |
| `WARMUP_SECONDS=1.0` | encoder-driven prefix before the first random code |
| `FALL_HEIGHT=0.4` | base height counted as a fall |
| `RESET_FALLEN=0` | leave a fallen robot down (default resets it at segment boundaries) |
| `MOTION=` / `RANK=` | pin the warmup motion |
| `ENV_SPACING=4.0` | meters between robots on screen |
| `SEED=1` | a different set of code sequences |
