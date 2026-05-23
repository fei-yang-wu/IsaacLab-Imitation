# Hierarchical Planning over Spectral Skills (GR00T-style HL + Bilinear LL)

## Context

The repo already trains a bilinear/spectral world model and a skill-conditioned
low-level policy via RLOpt's IPMD-Bilinear pipeline (`Isaac-Imitation-G1-Latent-v0`,
checkpointed at 500m / 1b frames). What's missing is a **high-level manager**
that picks skill sequences in a planning loop.

This plan describes a GR00T-style hierarchical planner stacked on top of the
existing low-level. The high-level is vision-language enriched; the low-level
is proprioceptive-only and unchanged. The "skill code" `z` is the bottleneck
between the two — it's a direction in the spectral coefficient space the
low-level was trained against, and the high-level learns to navigate that space.

## Two-Timescale Planning and Control

The central decomposition is a two-timescale hierarchy. The high-level planner
operates at a coarse temporal scale and chooses a skill command or subgoal every
`K` environment steps, while the low-level controller runs at the native control
rate and stabilizes the robot toward the currently active command.

Let `t` denote the low-level control timestep and let `m` denote the high-level
decision index, with `t = mK`. The high-level observation may include proprio,
vision, and task context:

```math
o_m^{HL} = \left(x_{mK}, I_{mK}^{ego}, I_{mK}^{tp}, c\right),
\qquad
h_m = E_\theta(o_m^{HL}).
```

Here `x_t` is the robot proprioceptive state, `I_t^{ego}` and `I_t^{tp}` are the
ego and third-person images, `c` is optional task or language context, and
`h_m` is the high-level latent state used for planning. The high-level action is
not a torque or joint target. It is a motor-actionable latent command:

```math
z_m \in \mathcal{Z},
```

where `\mathcal{Z}` is the skill/subgoal space understood by the low-level
policy. Depending on the experiment, `z_m` can be interpreted either as a skill
command sampled from a learned command manifold, or as an encoded subgoal state
that the low-level should reach over the next `K` steps.

The low-level policy is goal-conditioned control:

```math
a_t \sim \pi_\phi(a_t \mid x_t, z_m),
\qquad
t \in \{mK, \ldots, (m+1)K - 1\}.
```

Thus the low-level remains blind: it only consumes proprioception and the active
latent command. All camera and task-level reasoning is pushed into the
high-level planner. The shared interface between the two layers is the command
space `\mathcal{Z}`, not the raw observation space.

The high-level model predicts the coarse effect of executing a low-level command
for one command period:

```math
\hat{h}_{m+1} = T_\psi(h_m, z_m).
```

At planning time, the manager searches over a horizon of high-level commands
`z_{m:m+H-1}` and rolls the learned coarse dynamics forward:

```math
\hat{h}_{m+i+1} = T_\psi(\hat{h}_{m+i}, z_{m+i}),
\qquad
\hat{h}_m = h_m.
```

Given a sequence of goal embeddings `h^g_{m+1:m+H}`, planning solves

```math
z^\star_{m:m+H-1}
= \arg\min_{z_{m:m+H-1}}
\sum_{i=1}^{H}
d\!\left(\hat{h}_{m+i}, h^g_{m+i}\right)
+ \lambda \sum_{i=0}^{H-1} \Omega(z_{m+i}),
```

where `d` measures goal mismatch in the high-level latent space and
`\Omega(z)` keeps candidate commands on the valid skill manifold. Only the first
planned command `z_m^\star` is executed; after `K` low-level steps, the system
re-encodes the current observation and replans.

This receding-horizon structure gives the desired division of labor:

- the **high level** performs long-horizon semantic and visual reasoning by
  selecting skill commands or subgoals;
- the **low level** performs fast whole-body stabilization and tracking
  conditioned on the active command;
- the **shared latent command space** makes the two layers compositional, because
  the high-level planner only proposes commands that the low-level controller was
  trained to interpret.

User-confirmed design choices (this design doc is the converged form after
iterative discussion):

| Concern | Choice |
|---|---|
| HL fusion of proprio + vision | late concat → MLP (GR00T-style) |
| HL state inputs | proprio + ego camera + third-person camera |
| VL backbone | frozen DINOv2 + small trainable adapter |
| `T_k` target | single fused `h_{t+k}` tensor |
| `T_k` parameterisation | plain MLP first; bilinear `h + W(h)·G(z)` as Phase-2 swap |
| Goal source | reference clip frames → proprio + rendered ego + rendered third-person |
| Rendering stage | live during rollout collection |
| Planner | CEM over `z_{1:H}` |
| Low-level | unchanged `BilinearPolicyHead(s_proprio, z)` |
| MVP success criterion | planned-z beats random-z on dance102 tracking |

## Architecture

```
Inputs at time t (HIGH-LEVEL):
  s_proprio_t  (existing observation)
  image_ego_t  (NEW: head-mounted camera)
  image_tp_t   (NEW: fixed third-person camera)

High-level state encoder:
  F_t = pool over embed_dim of  F(s_proprio_t)               # frozen bilinear F
  v_t = adapter( concat[ DINOv2(image_ego_t),
                         DINOv2(image_tp_t) ] )              # DINOv2 frozen; adapter trained
  h_t = MLP_fuse([F_t, v_t])                                  # late concat → MLP

Goal encoding (clip frame at offset i·k from current time):
  F_g^i = pool(F(s_proprio_g^i))                              # from LAFAN1 reference (free)
  v_g^i = adapter( concat[ DINOv2(rendered_ego_at_g^i),
                           DINOv2(rendered_tp_at_g^i) ] )
  h_g^i = MLP_fuse([F_g^i, v_g^i])                            # same fusion MLP

High-level world model:
  T_k(h, z) → h_{t+k}                                         # plain MLP for MVP
  # Phase 2: h_{t+k} = h + W(h)·G(z)  (bilinear / spectral)

Planner (re-invoked every k env steps):
  CEM over z_{1:H} sequences:
    unroll  ĥ_{i+1} = T_k(ĥ_i, z_i)
    cost    = Σ_i ‖ĥ_i − h_g^i‖²
  emit z* = first skill of best elite
  call env.set_agent_latent_command(z*)

Low-level (unchanged):
  BilinearPolicyHead(s_proprio, z*) for k env steps
```

Parameter ownership:

| Component | Trainable? | Trained when |
|---|---|---|
| Bilinear `F`, `g`, low-level policy | frozen | already done (IPMD checkpoint) |
| DINOv2 backbone | frozen | n/a |
| VL adapter | trainable | jointly with `T_k` in one offline stage |
| `MLP_fuse` | trainable | jointly with `T_k` |
| `T_k` | trainable | one offline stage on labeled rollouts |
| CEM planner | parameter-free | online at eval |

## The spectral hook

The "transition learned with spectral decomposition" framing shows up in two
places:

1. **At the low level (already exists).** `BilinearPolicyHead` consumes `z` via
   `F(s)z`; skills are directions in the spectral coefficient space of the
   bilinear factorisation `φ(s,a) = g(a)ᵀ F(s)` (`RLOpt/.../ipmd/module.py:138`).
2. **At the high level (Phase 2 swap).** `T_k` can be re-parameterised as
   `h_{t+k} = h + W(h)·G(z)` — the coarse-grained, h-space analog of the
   low-level bilinear factorisation. The plain-MLP `T_k` in the MVP is just the
   warm start; the bilinear swap is one-line because the external signature is
   the same.

## MVP scope ("beat random-z on dance102")

The smallest end-to-end experiment that would justify continued investment.

Pipeline:

1. **One-time data collection** — re-roll out the existing scratch_1b bilinear
   checkpoint on dance102 with both cameras enabled and posterior-inferred `z`
   saved per window. ~30k frames/clip × 2 cameras × 224² ≈ 5–15 GB per clip.
   Expect 2–4× slowdown vs. cameraless rollout.
2. **One-time training** — jointly fit VL adapter + `MLP_fuse` + `T_k` on
   `(h_t, z, h_{t+k})` triples. Small offline run on a single GPU.
3. **Two evals on held-out frames** — identical env/checkpoint/cameras, only
   the latent-command controller differs:
   - baseline: existing `RandomLatentCommandSampler`
   - this work: `HierarchicalSkillController` + `CEMSkillPlanner`
4. **Metric**: per-step tracking error, averaged over ≥5 seeds. Pass = planned
   beats random with statistical significance (paired t-test, p < 0.05).

Explicitly out of scope for MVP:

- Posterior-z baseline (queued as follow-up #1)
- Cross-clip generalisation
- Bilinear `T_k` swap (queued as follow-up #2)
- Language goals
- Learned manager network (BC / RL on top of CEM)

## Near-Term Low-Level Experiment: Future-Goal AE Commands

Before adding cameras or a learned high-level manager, test whether the
low-level bilinear policy can use a slower, continuous goal embedding as its
action:

1. Register `Isaac-Imitation-G1-Latent-Goal-v0`, which exposes a single
   future reference goal state through the `expert_goal` observation group.
2. Encode `expert_goal.{expert_motion, expert_anchor_pos_b, expert_anchor_ori_b}`
   with the continuous `patch_autoencoder` posterior.
3. Hold the inferred latent command for `k` env steps via
   `ipmd.latent_learning.posterior_command_period=k`.
4. Use a 128D AE latent command and a 64D bilinear feature basis, forcing the
   `BilinearPolicyHead` to learn a command projector from goal embedding to
   spectral direction.

First sweep:

| Run | Goal offset | Command period | Latent dim | SR dim | Notes |
|---|---:|---:|---:|---:|---|
| goal25_period25 | ~0.5s at 50 Hz | 25 | 128 | 64 | First debug / baseline |
| goal50_period25 | ~1.0s | 25 | 128 | 64 | Longer lookahead, same control cadence |
| goal100_period50 | ~2.0s | 50 | 128 | 64 | Slower high-level action |

The expected failure mode is command-space mismatch: if reconstruction-only
latents are useful but not aligned with the bilinear basis, the learned
projector should help. If even this fails, the low-level may need to train with
explicitly sampled manager actions rather than posterior-inferred references.

## Stages

### Stage 0 — Camera env additions (read-only env config change)

Modify `imitation_g1_latent_env_cfg.py` to add ego-centric and third-person
camera terms (IsaacLab `TiledCameraCfg` with the existing `scene` parameters).
Cameras emit RGB at 224² (resize later if needed).

### Stage 1 — Data collection with cameras + z labels

Extend `scripts/rlopt/record_policy_rollout.py`:
- write camera buffers to NPZ alongside existing `qpos`/`qvel`/etc.
- additionally run `_latent_learner.infer_batch_latents` (or VQ posterior) per
  window and save the inferred `z`
- preserve the existing schema so non-VL consumers don't break

Run once on the dance102 clip(s).

### Stage 2 — High-level encoder + transition

New module `RLOpt/rlopt/agent/imitation/hl_encoder.py`:
- `VLAdapter(nn.Module)` — concat DINOv2 features → trainable MLP → `v_t`
- `HighLevelEncoder(nn.Module)` — owns `MLP_fuse([pool(F(s_p)), v_t]) → h_t`
- caches `F` reference (frozen) and DINOv2 reference (frozen)

New module `RLOpt/rlopt/agent/imitation/latent_transition.py`:
- `LatentTransition(nn.Module)` — plain MLP `(h_t, z) → h_{t+k}` for MVP
- abstract signature so bilinear variant can drop in without touching callers

New script `scripts/rlopt/train_hl_world_model.py`:
- load NPZs from Stage 1, build `(h_t, z, h_{t+k})` triples
- joint MSE loss on `T_k(h_t, z) − h_{t+k}`
- `F`, DINOv2, low-level head all frozen
- logs to `logs/hl_world_model/`

### Stage 3 — Skill space abstraction

New module `RLOpt/rlopt/agent/imitation/skill_space.py`:
- `SkillSpace` ABC with `sample(n)` and `to_latent_command(z)`
- `DiscreteVQSkillSpace` — wraps an existing VQ codebook (or a quick k-means
  on Stage-1 inferred `z` as a stand-in)
- `ContinuousSphericalSkillSpace` — `F.normalize(randn(n, D), dim=-1)` (for
  future continuous mode)

MVP uses the discrete variant.

### Stage 4 — CEM planner

New module `RLOpt/rlopt/agent/imitation/skill_planner.py`:
- `CEMSkillPlanner(skill_space, latent_transition, hl_encoder, plan_horizon, n_samples, n_elites, n_iters)`
- `plan(h_t, h_goals) → z*`
- vectorised over `n_samples × num_envs`

### Stage 5 — Hierarchical controller

New module `RLOpt/rlopt/agent/imitation/hierarchical_commands.py`:
- `HierarchicalSkillController` mirrors `LatentCommandController` public surface
  (drop-in compatible with `LatentCommandCollectorPolicy`)
- per-env countdown; re-plan at boundary or on `done`
- calls existing `env.set_agent_latent_command(z*)`

### Stage 6 — Eval script

New script `scripts/rlopt/play_hierarchical.py`:
- args: `--checkpoint`, `--hl_ckpt`, `--task Isaac-Imitation-G1-Latent-v0`,
  `--motion_manifest`, `--baseline {planned,random}`
- runs each baseline over N seeds × full dance102 held-out window
- logs per-step tracking error and overall reward
- emits a small CSV for the paired t-test

## Critical files

**New (in this work):**
- `RLOpt/rlopt/agent/imitation/hl_encoder.py` — VL adapter + fusion encoder
- `RLOpt/rlopt/agent/imitation/latent_transition.py` — `T_k` MLP
- `RLOpt/rlopt/agent/imitation/skill_space.py` — pluggable skill abstraction
- `RLOpt/rlopt/agent/imitation/skill_planner.py` — CEM
- `RLOpt/rlopt/agent/imitation/hierarchical_commands.py` — manager loop
- `scripts/rlopt/train_hl_world_model.py` — offline training
- `scripts/rlopt/play_hierarchical.py` — evaluation

**Modified:**
- `source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/imitation_g1_latent_env_cfg.py`
  — add ego + third-person `TiledCameraCfg` terms
- `scripts/rlopt/record_policy_rollout.py` — emit camera frames + inferred `z`

**Unchanged (reuse map):**
- `BilinearSR.encode_state` (`RLOpt/.../ipmd/module.py:132`) — proprio side of `h`
- `BilinearPolicyHead.forward` (`RLOpt/.../ipmd_bilinear.py:262`) — low-level
- `env.set_agent_latent_command` (`imitation_rl_env.py:1437`) — env hook
- `LatentCommandController.publish_latents_to_env` (`latent_commands.py:213`)
- `PatchVQVAELatentLearner.infer_batch_latents` — for discrete skill labelling

## Verification

1. **Shape smoke**: load checkpoint + untrained HL stack; confirm
   `h_t = MLP_fuse([pool(F(s)), adapter([DINOv2(ego), DINOv2(tp)])])` has the
   expected shape end-to-end on a single GPU batch.
2. **Stage-1 sanity**: re-roll one dance102 clip; spot-check that camera NPZ
   arrays render correctly (eyeball saved PNG samples) and `inferred_z` has
   non-trivial variance across frames.
3. **Stage-2 fit**: training loss for `T_k` drops below an "identity" baseline
   (`h_{t+k} = h_t`) on held-out windows. If it doesn't, the VL adapter or
   pooling choice needs revisiting.
4. **Stage-4 toy**: synthetic test where `h_goals` are pulled from a recorded
   rollout a few `k`-steps ahead; CEM's final cost should be strictly lower
   than random skill sequences (≥ 2× margin).
5. **End-to-end (MVP exit criterion)**: `play_hierarchical.py` on held-out
   dance102 frames, planned beats random in mean tracking error over ≥ 5 seeds
   with paired t-test p < 0.05.
6. **Regression**: existing random-z rollout path still works unchanged
   (`set_agent_latent_command` API and observation spec preserved).

Smoke command (illustrative):

```bash
conda run -n "${CONDA_ENV:-SL}" python scripts/rlopt/play_hierarchical.py \
    --task Isaac-Imitation-G1-Latent-v0 \
    --checkpoint <bilinear_scratch_1b.pt> \
    --hl_ckpt logs/hl_world_model/latest/ckpt.pt \
    --baseline planned \
    env.lafan1_manifest_path=./data/unitree/manifests/g1_unitree_dance102_manifest.json
```

## Open questions (resolve as part of Stage 0/1)

1. **Camera resolution**: 224² (DINOv2-native) vs 84² (cheap). Start at 224
   and downsample if render cost dominates.
2. **Goal pose rendering** in IsaacLab: cheapest path is a second invisible
   "ghost" robot articulation set to the reference pose, re-rendered through
   the same cameras. Worth checking whether IsaacLab supports two robots in
   the same scene without physics interference (it does via
   `replicate_physics=False` or visualisation-only assets).
3. **Pooling of `F(s)`**: `F(s).mean(dim=1) ∈ ℝ^D` is the default. If Stage-3
   fit is poor, try flattened `F(s) ∈ ℝ^{E·D}`.
4. **k (skill duration)**: start at `k=5` env steps (matches existing
   `latent_steps_min/max` defaults in `RandomLatentCommandSampler`).
5. **Plan horizon H**: start at 6–8 high-level steps (~30–40 env steps).

## Follow-ups (post-MVP, in priority order)

1. Beat the existing posterior-z baseline (`_inject_posterior_latent_command`)
   — proves planning helps over greedy posterior inference.
2. Swap `T_k` to bilinear `h + W(h)·G(z)` — closes the "spectral decomposition
   all the way through" loop.
3. Cross-clip generalisation: train on dance102, evaluate on another LAFAN1 clip.
4. Language-conditioned goals via the VL backbone's text tower.
5. Manager network distilled from CEM (faster inference; optional RL
   fine-tune).
