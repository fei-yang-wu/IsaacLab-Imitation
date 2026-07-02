# Language-Conditioned Skill Commander (System 2)

## Context

We already have a working **skill encoder** (`HighLevelSkillEncoder`) and a
skill-conditioned **low-level policy** (System 1) that consumes a continuous
skill command `z`. Today `z` is produced by an **oracle**: during low-level RL
the `FrozenHighLevelSkillCommandSampler` peeks at the expert's *future motion
window* and runs `z = encoder(state, future_window)`. That requires the
reference trajectory's future at inference time.

This page describes the **System-2 skill commander**: a policy that maps
`(current_state, language_goal) -> z`, removing the dependence on the future
window. It is the "high-level manager" anticipated in
[Hierarchical Planning over Spectral Skills](hierarchical-spectral-planning.md);
the low-level policy is unchanged and keeps consuming `z`. See also
[IPMD Representation Learning](ipmd-representation-learning.md).

"Start slow": use each LAFAN1 trajectory's **name** (e.g. `dance1_subject1`) as
the language instruction, embed it once into a `{motion_name: vector}` table,
and **distill** the frozen encoder's `z` into the commander.

## Approved approach and decisions (2026-06-15)

- **Train signal: supervised distillation.** Regress `commander(state, lang) -> z_target`,
  where `z_target` comes from the frozen `HighLevelSkillEncoder`. No sim in the
  loop for v0. (RL fine-tune through the frozen low-level is a later option.)
- **Language goal: trajectory names, embedded offline into a table.** RL/runtime
  only loads a tensor table — no text model at train or rollout time. Started
  with a **dummy** deterministic-embedding backend (no external dependency); a
  real `sentence-transformer` backend is wired but optional until needed.
- **State input: staged.** v0 uses the encoder-consistent **expert macro state**
  (matches the distillation target). A later milestone swaps in the robot's
  **achieved** state for true closed-loop control.
- **`z` target, not the full command.** Distill the encoder latent `z`; rebuild
  the command (`z` / `phi` / `z_phi` + optional phase) at rollout with the frozen
  diffsr via the existing `_command_code_from_state_z`, so the commander is a
  drop-in for *only* the "encode the future window" step.

## Constraints for future agents

- The commander must distill to the **same `z`** the target low-level policy was
  trained against: inherit `horizon_steps`, `encoder_window_mode`, `z_dim`, and
  `command_mode` from the loaded skill checkpoint — do not re-pick them.
- Do not retrain the low-level policy for v0. Validate by swapping the IPMD
  `command_source` from `hl_skill` to `skill_commander` at play time on an
  existing frozen low-level checkpoint; `hl_skill` (oracle) is the upper bound.
- Keep the table keyed by the **raw motion name** (the env emits motion names);
  the cleaned phrase only decides which text is embedded.
- Do not hardcode dimensions — read them from the env / skill checkpoint at
  runtime (`state_dim` ≈ 67 today, `z_dim` = 256 today).

## Submodule boundary (per AGENTS.md)

This feature genuinely requires **RLOpt** edits — the skill encoder, command
samplers, and IPMD `command_source` all live there. Bump the top-level submodule
pointer when RLOpt changes land.

- **RLOpt (submodule):** new `skill_commander.py` (network + config +
  trainer + sampler); new `command_source="skill_commander"` in IPMD.
- **This repo:** env plumbing (return `traj_rank`, add a motion-name accessor),
  the offline embedding script, the training entrypoint. The text-model
  dependency stays in the offline script here.

## Grounded code reference map

Paths relative to repo root. Line numbers are hints as of RLOpt pointer
`a8dcc6c` and the current in-repo env; class/function names are the stable
anchors.

### Skill encoder / sampler / trainer — `RLOpt/rlopt/agent/hl_skill_diffsr.py`

| Symbol | ≈Line | Notes |
|---|---|---|
| `HighLevelSkillDiffSRConfig` | 93 | dataclass: `horizon_steps=25`, `z_dim=256`, `diffsr_feature_dim=128`, `diffsr_embed_dim=512`, `encoder_window_mode="full"\|"intermediate"`, `encoder_hidden_dims=(1024,512,512)`. `from_dict`/`to_dict`/`validate`. |
| `HighLevelSkillEncoder` | 220 | `forward(state[B,D], future_window[B,W,D]) -> z[B,z_dim]`. Input `D*(W+1)`; MLP `Linear→LayerNorm→Mish`, final `Linear→z_dim`. |
| `FrozenHighLevelSkillCommandSampler` | 279 | online latent-command source (the thing System 2 replaces). |
| `._command_code_from_state_z(state, z)` | 660 | `z` / `phi=diffsr.forward_phi(state,z)` / `z_phi`. **Reuse surface.** |
| `._append_command_phase(codes, phase)` | 671 | optional `sin_cos` phase (`phase_dim=2`). |
| `._encode_current_macro_batch(env_ids)` | 680 | calls `current_expert_macro_transition_batch`, then `z = skill_encoder(state, _encoder_input_window(future_window))`. **Override point.** |
| `.sample_for_step(td, device, dtype)` | 879 | renewal scheduling + returns `[B, latent_dim]`. Reuse as-is. |
| `HighLevelSkillDiffSRTrainer` | 976 | `train_step` 1487, `evaluate` 1522, `train` 1587, `save_checkpoint` 1634, `load_checkpoint` 1639. **Mirror for the commander trainer.** Uses `sample_expert_macro_transition_batch`. |

Checkpoint dict keys: `skill_encoder_state_dict`, `diffsr_state_dict`, `config`,
optimizer state, feature normalization.

### Env macro samplers — `source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py`

| Symbol | ≈Line | Notes |
|---|---|---|
| `sample_expert_macro_transition_batch(batch_size, horizon_steps, split=None, eval_fraction=0.1, split_seed=0)` | 3062 | returns `{"hl": {state[B,D], future_window[B,W,D], target[B,D]}}`. **Does NOT yet return trajectory identity.** Split path has per-sample `traj_ranks_tm` (≈3115); `all` path uses `_sample_expert_trajectory_batch`. |
| `current_expert_macro_transition_batch(horizon_steps, env_ids=None)` | 3183 | same `hl` structure aligned to live env cursors. |
| `_expert_macro_feature_term_order()` | 2595 | `(expert_motion, expert_anchor_pos_b, expert_anchor_ori_b)`. |
| `_sample_expert_trajectory_batch(batch_size)` | 2025 | → `trajectory_manager.sample_random_transitions` → `(frame, env_ids_tm, global_indices)`. |
| `_expert_macro_split_trajectory_ranks(...)` | 2064 | deterministic train/eval split over nonempty trajectory ranks. |

`state_dim ≈ 67` = `expert_motion` 58 (29 DoF × 2) + `expert_anchor_pos_b` 3 +
`expert_anchor_ori_b` 6 (rot6d). TorchRL wrapper exposes both samplers at
`source/isaaclab_imitation/isaaclab_imitation/envs/rlopt.py:105,121`.

### Trajectory identity — `ImitationLearningTools/iltools/datasets/manager.py`

| Symbol | ≈Line | Notes |
|---|---|---|
| `env_traj_rank` | 198 | per-env current trajectory rank tensor (**rollout-time goal lookup**). |
| `get_env_traj_info(env_id) -> (dataset, motion, trajectory)` | 277 | rank → motion name. |
| `_ordered_traj_list` / `get_traj_rank(...)` | 189 / 282 | rank ↔ name mapping. |

Also `get_ith_traj_info(rank, ordered_traj_list)` in
`ImitationLearningTools/iltools/datasets/utils.py:291`.

### Integration / config

- IPMD command source: `RLOpt/rlopt/agent/ipmd/ipmd.py` — `command_source` ∈
  `{random, posterior, hl_skill, skill_commander}`. `skill_commander` builds
  `FrozenSkillCommanderSampler` into the same `_hl_skill_command_sampler`
  slot (config: `skill_commander_checkpoint_path`, `skill_commander_embeddings_path`;
  reuses `hl_skill_command_mode` / `hl_skill_horizon_steps` / `latent_dim`).
- Gym envs registered in
  `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/__init__.py`
  (`Isaac-Imitation-G1-Latent-v0` etc.). Latent input keys:
  `.../config/g1/agents/rlopt_ipmd_cfg.py` (`LATENT_POLICY_INPUT_KEYS`).
- Manifest format: `dataset.trajectories.lafan1_csv[].name`; loader
  `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/lafan1_manifest.py::load_lafan1_manifest`.
  Real 40-motion manifest: `data/lafan1/manifests/g1_lafan1_manifest.json`.
- Templates to mirror: `scripts/rlopt/train_hl_skill_diffsr.py`,
  `scripts/rlopt/train_hl_skill_pipeline.py`; RLOpt tests
  `RLOpt/tests/test_hl_skill_diffsr.py`.

## Milestones

### M0 — Language goal embedding table — DONE

`scripts/rlopt/build_language_goal_embeddings.py` (this repo).

- Reads a manifest -> unique motion names -> resolves each to an offline prompt.
  If manifest entries contain `language.<prompt_tier>`, those captions are used
  directly; otherwise it falls back to built-in LaFAN1 templates
  (`humanize_motion_name`: `dance1_subject1` -> "dance").
- Prompt tiers: `raw_name`, `category`, `robot_instruction`,
  `kinematic_description`, `event_level`, `attribute_text`. Default is
  `category`, preserving the old cleaned-name behavior.
- `--prompt_json` can still override manifest or built-in prompts by raw
  motion name or category for curated ablations. Explicit prompt JSON takes
  precedence over manifest-embedded captions.
- Embeds unique prompts, L2-normalizes, expands to one row **per motion name**.
- Backends: `--backend dummy` (default; deterministic, seeded by prompt text,
  no dep) and `--backend sentence-transformer` (lazy import; default model
  `all-MiniLM-L6-v2`, 384-d).
- Output table (`data/lafan1/language/g1_lafan1_name_embeddings.pt`, **not
  committed**): keys `names`, `phrases` (back-compatible prompt alias),
  `prompt_texts`, `categories`, `prompt_tier`, `name_to_index`,
  `embeddings[N,D]`, `embed_dim`, `backend`, `model`, `embedding_model`,
  `normalized`, `raw_names`, `manifest`, `manifest_sha256`, `prompt_json`,
  `manifest_prompt_count`, `prompt_json_prompt_count`.
- Validated on the 40-motion manifest → **8 phrases**
  (`dance, fall and get up, fight, fight and sports, jumps, run, sprint, walk`),
  table `[40, 384]`, unit-normalized, deterministic, phrase-grouping confirmed
  (two `dance*` names share a vector; dance vs walk ≈ orthogonal). `ruff` clean.

Run the old category baseline:

```bash
pixi run python scripts/rlopt/build_language_goal_embeddings.py \
  --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
  --backend dummy \
  --prompt_tier category
```

Run the captioned-manifest path used for current language-conditioning tests:

```bash
pixi run python scripts/rlopt/build_language_goal_embeddings.py \
  --manifest data/lafan1/language/g1_lafan1_manifest.with_codex_storyboard_language_v1.json \
  --backend sentence-transformer \
  --model all-MiniLM-L6-v2 \
  --prompt_tier attribute_text \
  --output data/lafan1/language/g1_lafan1_minilm_attribute_codex_storyboard_v1.pt
```

### Language prompt and embedding audits - OFFLINE INFRA READY

- `scripts/rlopt/language_prompts.py` owns deterministic prompt tiers for the
  current LaFAN1 categories: `dance`, `fall and get up`, `fight`,
  `fight and sports`, `jumps`, `run`, `sprint`, `walk`.
- `scripts/rlopt/audit_language_embeddings.py` audits saved embedding tables
  without Isaac: cosine statistics, nearest neighbors, category clustering,
  intra/inter-category separation, and expected semantic checks such as
  `walk` near `run`/`sprint` and `run` near `sprint`.
- `scripts/rlopt/audit_language_motion_alignment.py` optionally compares a
  saved embedding table against precomputed oracle-`z` centroids or samples;
  if no z file is supplied, it exits after language-space diagnostics and does
  not launch Isaac.
- Recommended primary ablation model: `Qwen/Qwen3-Embedding-8B` through the
  `sentence-transformer` backend when local memory allows. Fallbacks:
  `Qwen/Qwen3-Embedding-4B` or `voyageai/voyage-4-nano`; keep MiniLM, dummy,
  and no-language as baselines.
- Dance102 has only one motion in the current manifest, so it is useful for
  closed-loop smoke/debug but is **not** a language-discrimination benchmark by
  itself.

### LaFAN1 motion-description drafting - OFFLINE INFRA READY

`scripts/rlopt/draft_lafan1_motion_descriptions.py` creates a reviewable
language sidecar from the G1 LaFAN1 NPZ files. It computes root speed, root
travel, height change, turning, arm/leg activity, and simple rhythm statistics,
then writes deterministic draft fields keyed by raw motion name:
`short_caption`, `robot_instruction`, `kinematic_description`, `event_level`,
`attribute_text`, `distinguishing_features`, `stats_summary`, and
`review_status="draft"`.

The sidecar is directly usable as a prompt override. Once captions are baked
into a manifest via `--manifest_output`, pass that captioned manifest directly
to `build_language_goal_embeddings.py` and omit `--prompt_json` unless making
an explicit override ablation:

```bash
pixi run python scripts/rlopt/draft_lafan1_motion_descriptions.py \
  --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
  --output /tmp/g1_lafan1_motion_descriptions.draft.json \
  --markdown_output /tmp/g1_lafan1_motion_descriptions.draft.md

pixi run python scripts/rlopt/build_language_goal_embeddings.py \
  --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
  --backend sentence-transformer \
  --model Qwen/Qwen3-Embedding-8B \
  --prompt_tier kinematic_description \
  --prompt_json /tmp/g1_lafan1_motion_descriptions.draft.json \
  --output /tmp/g1_lafan1_qwen8b_kinematic_descriptions.pt
```

For visual review, first render/reference-record videos with the existing Isaac
tools, then build compact contact sheets without launching Isaac:

```bash
pixi run python scripts/rlopt/make_motion_storyboards.py \
  --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
  --video_dir /path/to/rendered/lafan1/videos \
  --output_dir /tmp/lafan1_storyboards \
  --sample_count 16 \
  --columns 4

pixi run python scripts/rlopt/draft_lafan1_motion_descriptions.py \
  --manifest data/lafan1/manifests/g1_lafan1_manifest.json \
  --storyboard_dir /tmp/lafan1_storyboards \
  --output /tmp/g1_lafan1_motion_descriptions.with_storyboards.json \
  --markdown_output /tmp/g1_lafan1_motion_descriptions.with_storyboards.md
```

Review workflow: inspect each storyboard/video plus its stats, ask Codex or a
vision-language model for JSON edits only to semantically wrong or insufficiently
distinct language fields, then rebuild embedding tables and run
`audit_language_embeddings.py`. If we want captions embedded in the manifest for
future loaders, pass `--manifest_output /tmp/g1_lafan1_manifest.with_language.json`;
the source manifest is not modified by default.

### M1 — Commander network + distillation trainer (offline) — DONE (unit-tested)

- **Edit (this repo)** `envs/imitation_rl_env.py`: add `hl["traj_rank"]`
  (LongTensor `[B]`) to both macro samplers (split path = `traj_ranks_tm`;
  `all`/current path = `tm.env_traj_rank[env_ids]`); additive/back-compatible.
  Add accessor `expert_trajectory_motion_names() -> list[str]` (rank-indexed).
- **New (RLOpt submodule)** `RLOpt/rlopt/agent/skill_commander.py`:
  - `SkillCommanderConfig` (mirror `HighLevelSkillDiffSRConfig`):
    `skill_checkpoint_path`, `language_embeddings_path`, `lang_embed_dim=384`,
    `hidden_dims`, optimizer/iter/eval/split params, `cosine_loss_coeff`,
    `z_norm_coeff`. `z_dim`/`horizon_steps`/`encoder_window_mode`/`command_mode`
    are loaded from the skill checkpoint.
  - `SkillCommander(nn.Module)`: `forward(state, lang_emb) -> z` (MLP on
    `concat[state, lang_emb]`, same `Linear→LayerNorm→Mish` block style).
  - `SkillCommanderTrainer` (mirror `HighLevelSkillDiffSRTrainer`):
    freeze encoder from checkpoint; build `rank → embedding` lookup from the
    table + name accessor; per step sample `split="train"` macro batch →
    `z_target = frozen_encoder(state, encoder_input_window(future_window))`,
    `z_hat = commander(state, lookup[traj_rank])`,
    `loss = MSE + cosine_loss_coeff·(1−cos) + z_norm_coeff·‖z_hat‖²`; eval on
    `split="eval"` (**held-out trajectory names**) reporting z-MSE, cosine, and
    optional reconstruction via the frozen diffsr.
- **New (this repo)** `scripts/rlopt/train_skill_commander.py` — mirror
  `train_hl_skill_diffsr.py` (Hydra/argparse + env build + trainer + save). Args:
  `--task Isaac-Imitation-G1-Latent-v0`, `--skill-checkpoint`,
  `--language-embeddings`, dims/iters.
- **Tests (RLOpt)** `RLOpt/tests/test_skill_commander.py`: shapes;
  distillation loss drops on overfit batch; eval runs; sampler returns
  `[B, latent_dim]`. Wire into the `test-rlopt` pixi task.

### M2 — Rollout integration — DONE (code; pending Isaac-Sim rollout validation)

- **New (RLOpt)** `FrozenSkillCommanderSampler(FrozenHighLevelSkillCommandSampler)`:
  override only the `z` production in `_encode_current_macro_batch` — keep
  pulling `state` from `current_expert_macro_transition_batch`, read per-env
  `tm.env_traj_rank → name → lookup` for `lang`, compute `z = commander(state, lang)`.
  Reuse `_command_code_from_state_z`, `_append_command_phase`, `_sample_steps`,
  `_done_mask`, `sample_for_step` unchanged.
- **Edit (RLOpt)** `ipmd/ipmd.py`: add `command_source="skill_commander"` branch
  (config: `skill_commander_checkpoint_path`, `language_embeddings_path`; read
  `command_mode`/`horizon_steps` from the commander/skill checkpoint).
- **Validate** with `play.py` on a frozen low-level policy (trained with
  `command_source=hl_skill`) by switching to `skill_commander`; compare against
  `hl_skill` (oracle upper bound).

### M3 — Achieved-state robustness — v0 DONE; full closed-loop = follow-up

- **v0 (done):** `state_noise_std` augments the commander's state input with
  per-dim-scaled Gaussian noise during distillation (target z stays from the
  clean expert state), nudging the commander to rely on the language goal and
  tolerate non-expert states. Validated knob: the real-vs-shuffled language
  cosine gap grows from ~0.004 (no noise) to ~0.021 (std=5.0).
- **Finding:** with the expert-reference macro state as input, language is
  largely **redundant** — the state alone nearly determines the next skill.
  Noise only weakly induces language reliance, confirming the real fix is
  achieved-state conditioning.
- **Follow-up (full M3):** feed the robot's **achieved** macro state (needs a new
  env accessor) and distill on rollout-collected achieved states, or **RL
  fine-tune** the commander through the frozen low-level policy on env reward.

## Environment / setup notes

- The skill commander's RLOpt code lives in the submodule; before M1 in a fresh
  worktree: `git submodule update --init --recursive`, then
  `pixi install --locked` (and `-e isaaclab`), and `pixi reinstall rlopt` after
  RLOpt edits.
- M0 needs neither submodules nor Isaac Sim (pure stdlib + torch, no repo
  imports) — it runs in the **default** Pixi env. If a worktree env is not yet
  installed, M0 can be run from the main checkout's `default` env.
- The embedding table and `data/` are generated artifacts — do not commit.

## Validation summary

1. `pixi run test-rlopt` (commander unit tests, default env).
2. Offline smoke: `train_skill_commander.py` with tiny iters; train
   z-MSE falls and eval cosine rises on held-out names.
3. Rollout smoke: `pixi run -e isaaclab` play with
   `command_source=skill_commander`; compare to `hl_skill`.
4. `pixi run lint`, `pixi run format-check`, `pixi run typecheck`.

## Local validation results (2026-06-15, RTX PRO 6000, Pixi isaaclab)

- **M0**: dummy + `sentence-transformer` (MiniLM, 384-d) tables built; 40 names ->
  8 phrases; MiniLM cosine structure sensible (walk·run 0.46 > walk·fight 0.34).
- **M1**: `train_skill_commander.py` (Isaac, headless, 300 updates) —
  train z_mse 0.34->0.06, held-out eval z_cosine 0.25->0.77. Distillation works
  and generalizes to held-out trajectory names.
- **M2**: `command_source=skill_commander` drove the low-level policy in a real
  Isaac rollout (`train.py`, fresh policy, ~44 fps) with `--video` clips recorded;
  no crashes. (Old 2026-06-08 checkpoints don't load under the current config due
  to network-shape version skew — unrelated to this feature.)
- **M3 v0**: `state_noise_std` sweep (0 / 1 / 5) controls language reliance as
  above.
- `pixi run test-rlopt`: 74 passed.

Run-env notes: Isaac needs `OMNI_KIT_ACCEPT_EULA=YES` for non-interactive runs;
pass absolute paths for the manifest/table/checkpoints when the worktree lacks a
local `data/lafan1`; `--video --video_interval 1` writes one clip per step (use a
larger interval for a single clip).
