# BONES-SEED 129k cache workflow

Use this reference for the complete 129,785-motion BONES-SEED export, for the
large local replay cache, or when making a small evaluation manifest.  Run all
commands from the repository root.

## Frozen full-dataset identity

The local full snapshot used by the 2026-08-04 v2 scale campaign is:

```text
data root          /mnt/storage/fwu91/bones_seed_full
full manifest      /mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json
manifest sha256    eb0ad052afe72fb6228f4be9d52132c9cb9a52ac9c561751e49e6ca31346e688
Zarr root          /mnt/storage/fwu91/bones_seed_full/zarr/g1_bones_seed_sonic_full
motion count       129785
transition count   47491234
persist id         bones_seed_sonic_full_129785@e714bbff
```

The replay schema is exactly:

```text
qpos qvel root_pos root_quat root_lin_vel root_ang_vel
joint_pos joint_vel body_pos_w body_quat_w body_lin_vel_w body_ang_vel_w
```

Do not infer validity from directory size.  The `iltools_rb_manifest.json`
sidecar is the authority and must match the full content identity, null
dataset/motion/trajectory selections, exact keys, 129,785 ordered trajectories,
and 47,491,234 written transitions.

## The cache layers

Keep these distinct when explaining memory or startup time:

| Layer | Lifetime | Full-set size | Purpose |
| --- | --- | ---: | --- |
| Zarr motion store | on disk, reusable | about 196 GiB | Hierarchical trajectory source built from the NPZ manifest. About 41 files per trajectory, so roughly 5.3M files. |
| ILTools persisted replay | on disk, reusable | about 95 GiB | CPU memmaps for the complete replay schema. `qpos.memmap` alone is about 6.4 GiB; it is not the whole schema. |
| **reference arrays** | **on disk, reusable** | **49.4 GB** | **The two derived caches' contents, written once straight from the NPZs in the layout the environment consumes. Replaces both layers above for training.** |
| `root_qpos` macro cache | one process | about 6.4 GiB / 6.7 GB | Dense `joint_pos` plus anchor position/quaternion, normally materialized in VRAM for encoder-window sampling. |
| low-level runtime cache | one process | about 44.8 GiB | Dense `qpos`, internal `qvel`, and 14 selected-body states for fast live gathers. |

The 6.x GiB number can therefore mean either `qpos.memmap` or the compact
root+qpos device cache.  Neither is the full 95 GiB trajectory schema.

### Why the reference arrays exist

Derived from the replay, the two runtime caches read about **133 GB to keep
about 55 GB**, and `body_pos_w` plus `body_quat_w` are read twice, once by each:

| cache | keeps | reads from the replay |
| --- | ---: | ---: |
| `root_qpos` macro | 6.84 GB | `joint_pos` 5.5 + `body_pos_w` 17.1 + `body_quat_w` 22.8 = **45.4 GB** |
| runtime, 14 bodies | 48.07 GB | `qpos` 6.8 + `qvel` 6.6 + all four `body_*` 74.1 = **87.5 GB** |

`/mnt/storage` is a 7200-rpm spinning disk, so that is 12-20 minutes on every
process start, and 133 GB does not fit the page cache, so nothing amortizes
between the two builds.  The reference arrays are already in both caches'
layout: the runtime cache is memory-mapped in place and the macro cache is one
contiguous read, so a launch reads about 50 GB sequentially from NVMe instead.

Equivalence with the Zarr path was measured against the packed replay: over 360
rows sampled from 60 random trajectories, root position, all 29 joint positions,
all 35 `qvel` components, and all 30 bodies' positions and quaternions are
bit-identical.  The only difference is the root quaternion (`qpos[3:7]` and the
`root_quat` field) at most **1.79e-07**, one to three float32 ULP, because the
Zarr export re-normalized it before rounding.

### Build the reference arrays

The builder reads the NPZ tree directly.  Those files are `STORED` zip members
and there is one per trajectory, so it is 129,785 file opens instead of the
Zarr's ~5.3M, and no decompression.  Passing `--traj_info` keeps the trajectory
order and row offsets byte-compatible with the existing replay, which matters
because planner goal indices are trajectory ranks.

```bash
pixi run python -m imitation_experiments.data.build_reference_arrays \
  --manifest /mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json \
  --traj_info /mnt/storage/fwu91/bones_seed_full/rb_packed/g1_bones_seed_sonic_full/iltools_rb_manifest.json \
  --output_dir /mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1 \
  --persist_id bones_seed_sonic_full_129785@e714bbff \
  --anchor_body pelvis \
  --body_names pelvis left_hip_roll_link left_knee_link left_ankle_roll_link \
    right_hip_roll_link right_knee_link right_ankle_roll_link torso_link \
    left_shoulder_roll_link left_elbow_link left_wrist_yaw_link \
    right_shoulder_roll_link right_elbow_link right_wrist_yaw_link \
  --workers 4 --expected_motions 129785 --expected_transitions 47491234 \
  --verify_load
```

Put the output on `/mnt/hsstorage` (NVMe), not `/mnt/storage`.  Measured on the
idle spinning tree: 107 files/s at 1 worker, 113 at 2, 120 at 4, and 120 at 8,
so 4 is the knee and the whole build is about 18 minutes.  Validate an existing
directory without rebuilding by adding `--validate_only`.

Measure on an idle disk.  A `find` or `rg` over the 5.3M-file Zarr tree pins
that drive at 100% utilization for hours, and every throughput number taken
while one is running is wrong by roughly an order of magnitude.  Check with
`iostat -x -d sda 3 2` before trusting a timing.

Eight arrays are written, and the quaternion conventions differ by consumer:
`body_quat_w` keeps the dataset's WXYZ order, `anchor_quat_w` is pre-swizzled to
XYZW.  `joint_pos`/`joint_vel` are not written -- they are `qpos[:, 7:]` and
`qvel[:, 6:]`.

Consume them with:

```text
env.data.manifest=null
env.data.reference_arrays_dir=/absolute/path/to/validated_reference_arrays
env.data.reference_arrays_warm_workers=8
env.data.runtime_cache_device=cpu
env.data.macro_cache_device=cuda:0
env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]
env.data.persist_id=bones_seed_sonic_full_129785@e714bbff
```

No `cache_dir`, no `persist_dir`, no `keys`: neither the Zarr nor the replay is
opened.  The directory is keyed by its **body list** — order matters, these are
column positions — and loading refuses one built for a different set rather than
reading the wrong columns.
`experiments/campaigns/2026-08-04-bones129k-v2-adaptive-10b/run.sh` selects
between the two sources with `DATA_SOURCE=arrays` (default) or
`DATA_SOURCE=replay`, and preflights whichever is chosen.

The **anchor body is the one part of the identity that need not match**.  Its
pose is also in `body_pos_w`/`body_quat_w`, so any anchor inside the retained
body set is derived at load time; only an anchor outside the set fails.  Both
anchors in use — `pelvis` for the v2 reference channel and `torso_link` for two
termination terms — are inside the tracked 14, so one artifact serves every
current v2 config.  Deriving reads the full body block once, so prefer a
directory built with the matching `--anchor_body` for repeated runs.

Beware: `Isaac-Imitation-G1-v2` anchors on **`pelvis`**
(`config/g1/imitation_g1_env_v2.py:175`), and `stamp_anchor_body` at line 321
overwrites the anchor in every observation and reward term with it.  Grepping
`anchor_body_name` is misleading — `common/rewards.py` shows `torso_link`
literals that do not survive the stamp, `common/terminations.py` is not stamped
and keeps its own, and `G1_OBS_ANCHOR_BODY_NAME` is a superseded default.

Nothing in the builder is specific to this dataset or robot: joint count, body
count, `qpos`/`qvel` widths, and both name lists come from the data, so it works
for any NPZ tree this repo's CSV converters produce.

### Publish and fetch

The artifact is 49.4 GB, which matters on ICE: the 300 GB per-user quota cannot
hold the source form (about 103 GB of NPZ plus a 136-157 GB Zarr, peaking near
260 GB with both co-resident), and the arrays skip the 4.3 h NPZ->Zarr and 3.1 h
Zarr->replay builds entirely.

```bash
# From the workstation, after --verify_load has passed.
pixi run python -m imitation_experiments.data.publish_reference_arrays push \
  --source_dir /mnt/hsstorage/fwu91/bones_seed_ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1 \
  --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays \
  --persist_id bones_seed_sonic_full_129785@e714bbff \
  --anchor_body pelvis --body_names <the 14> \
  --expected_motions 129785 --expected_transitions 47491234 \
  --source_repo GeorgiaTech/g1_bones_seed_sonic_129k_50hz --dry_run

# On a compute node.
pixi run python -m imitation_experiments.data.publish_reference_arrays fetch \
  --repo_id GeorgiaTech/g1_bones_seed_sonic_129k_50hz_refarrays \
  --dest_dir "${SLURM_TMPDIR:-/tmp}/g1_bones_seed_refarrays" \
  --persist_id bones_seed_sonic_full_129785@e714bbff \
  --expected_motions 129785 --expected_transitions 47491234
```

Both directions run the same validation the environment applies at load time, so
a directory that would be refused by training is refused before 49 GB moves.
`HF_HUB_DISABLE_XET=1` is set automatically — the Xet backend has stalled
partway through large uploads on this account.  The 10.87 MB
`reference_arrays_manifest.json` must travel with the arrays: it carries the
identity *and* the 129,785-entry trajectory table, and without it the arrays are
unloadable.

The repo is **public**, so no token is needed on a compute node.  That was not
the first choice: a private upload of this size exceeded the `GeorgiaTech` org's
private-storage limit and returned
`403 Forbidden: Private repository storage limit reached` on **every private
repo in the org**, including other people's.  Private storage is metered and
org-wide; public storage is not.  Do not push 49 GB privately here.  The 103 GB
source NPZ tree is already public in the same org and this artifact is a strict
subset of it, so publishing exposes nothing new.

Fetch validates sizes and identity, not contents against the source NPZs.  Do
that once at build time with `--verify_load`.

## Hard safety rules

- One `persist_dir` belongs to one source identity, one selection, and one key
  list.  Never point a subset manifest, `env.data.clips=[...]`, or one-motion
  visualization at the full replay directory.
- Never rebuild a large replay buffer in place.  Build into a fresh, versioned
  directory, validate it, and only then update a launcher to use the new path.
- A changed manifest, replaced NPZ, changed key list, or changed selection needs
  a new `persist_id` and a new directory.
- `env.data.cache_refresh=true` requires the manifest that can rebuild that
  Zarr.  Never use it with `env.data.manifest=null`.
- Consumer launchers must validate `iltools_rb_manifest.json` before Isaac or
  training starts.  A directory containing large old memmaps can still have a
  subset sidecar and is then invalid for the full run.
- The ILTools persisted identity is metadata-based, not a bytewise hash of all
  95 GiB.  Preserve the manifest/NPZ hashes as provenance and quarantine any
  directory left by an interrupted build.

## Build or validate the full replay cache

The skill-owned wrapper builds directly from an existing Zarr store without
starting Isaac Sim.  It refuses a nonempty mismatched output directory.

Choose a fresh path for a build:

```bash
pixi run python \
  .agents/skills/bones-seed-dataset/scripts/build_replay_cache.py \
  --zarr /mnt/storage/fwu91/bones_seed_full/zarr/g1_bones_seed_sonic_full \
  --persist-dir /mnt/storage/fwu91/bones_seed_full/rb/g1_bones_seed_sonic_full_129785_e714bbff_v2 \
  --persist-id bones_seed_sonic_full_129785@e714bbff \
  --expected-motions 129785 \
  --expected-transitions 47491234
```

This is an hours-scale sequential build and needs about 95 GiB of disk space.
The process writes the sidecar only after filling the replay buffer.  If it is
interrupted, leave the partial directory quarantined and rerun with another
fresh path; do not resume or overwrite it.

Validate a candidate without rebuilding:

```bash
pixi run python \
  .agents/skills/bones-seed-dataset/scripts/build_replay_cache.py \
  --zarr /mnt/storage/fwu91/bones_seed_full/zarr/g1_bones_seed_sonic_full \
  --persist-dir /absolute/path/to/candidate_replay_cache \
  --persist-id bones_seed_sonic_full_129785@e714bbff \
  --expected-motions 129785 \
  --expected-transitions 47491234 \
  --validate-only \
  --verify-load
```

`--verify-load` exercises the actual ILTools reopen path, reconciles every one
of the 129,785 trajectory identities and transition spans with the source Zarr
metadata, and compares all replay fields at the beginning, middle, and end of
five distributed trajectories against the source arrays. Use it before a
training launch; sidecar-only validation is not sufficient after an interrupted
or accidentally reused memmap directory.

For full-dataset pretraining and low-level training, reuse it with:

```text
env.data.manifest=null
env.data.cache_dir=/mnt/storage/fwu91/bones_seed_full/zarr/g1_bones_seed_sonic_full
env.data.cache_refresh=false
env.data.storage_device=cpu
env.data.persist_dir=/absolute/path/to/validated_replay_cache
env.data.persist_id=bones_seed_sonic_full_129785@e714bbff
env.data.keys=[qpos,qvel,root_pos,root_quat,root_lin_vel,root_ang_vel,joint_pos,joint_vel,body_pos_w,body_quat_w,body_lin_vel_w,body_ang_vel_w]
```

Add the compact root+qpos encoder path only when needed:

```text
env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
env.data.macro_cache_device=cuda:0
env.data.macro_cache_chunk_size=262144
```

For low-level live tracking, additionally use:

```text
env.data.runtime_cache_device=cpu
env.data.runtime_cache_chunk_size=262144
env.data.runtime_cache_body_names=[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]
```

The runtime cache keeps `qvel` for resets, velocity rewards, and privileged
observations.  It does not add qvel to the root+qpos encoder or actor command.

## Small evaluation manifests

Do not use `env.data.clips` against the full Zarr/replay pair for local oracle
evaluation.  Write a content-specific manifest and give it separate Zarr and
replay paths:

```bash
SUBSET_ROOT=/mnt/storage/fwu91/bones_seed_full/eval/subset6_v1

pixi run python -m imitation_experiments.data.write_motion_subset_manifest \
  --manifest /mnt/storage/fwu91/bones_seed_full/manifests/g1_bones_seed_sonic_full_manifest.json \
  --motion_names \
    neutral_walk_180_R_002_A116_M \
    jog_ff_loop_180_R_002_A143 \
    jump_ff_180_R_002_A143 \
    dance_basic_double_slide_270_R_002_A309 \
    clap_enthusiastic_002_A122_M \
    looking_around_on_ground_002_A053 \
  --output "${SUBSET_ROOT}/manifest.json"
```

On the first evaluation launch, build an isolated small Zarr:

```text
env.data.manifest=/mnt/storage/fwu91/bones_seed_full/eval/subset6_v1/manifest.json
env.data.cache_dir=/mnt/storage/fwu91/bones_seed_full/eval/subset6_v1/zarr
env.data.cache_refresh=true
env.data.storage_device=cuda:0
env.data.persist_dir=null
env.data.persist_id=null
env.data.macro_cache_device=null
env.data.runtime_cache_device=null
```

On later launches, change only `env.data.cache_refresh=false`.  Six motions fit
comfortably in the ordinary GPU replay path, so no persisted replay, macro
cache, or 44.8 GiB runtime cache is needed.  Keep strict qualification and the
non-terminating full-horizon pass pinned to explicit named motions and frame 0;
adaptive training resets do not belong in evaluation.

If a persisted subset cache is ever useful, give both `persist_dir` and
`persist_id` subset-specific values derived from the subset-manifest hash.  It
must never share the full-dataset directory or full-dataset `persist_id`.
