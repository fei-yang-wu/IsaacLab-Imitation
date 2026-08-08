---
name: policy-eval-video
description: Render policy-vs-reference evaluation videos for a low-level tracking checkpoint in this IsaacLab-Imitation repo, one video per motion at the motion's own length, with tracking terminations disabled. Use when the user asks for eval videos, rollout videos, per-motion videos, visual inspection of a checkpoint, "does it look right", side-by-side reference comparison, or when an oracle/M3 evaluation needs its mandated non-terminating diagnostic render.
---

# Policy evaluation videos

Render a trained low-level checkpoint against the expert reference and keep a
video per motion.

## Use the wrapper

```bash
pixi run python .agents/skills/policy-eval-video/scripts/render_policy_videos.py \
  --checkpoint /path/to/model_step_N.pt \
  --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  --output_root logs/videos/<run_tag> \
  --reference_arrays /path/to/reference_arrays --persist_id <id> \
  -- <hydra overrides the recipe needs>
```

Everything after `--` is forwarded verbatim as Hydra overrides. The wrapper
discovers the motions from the reference-array sidecar, loops them, and prints
each retained video's **absolute** path at the end.

For a language-conditioned M3 planner, add all three of
`--planner_checkpoint`, `--skill_checkpoint`, and `--language_embeddings`.
The wrapper then binds each rendered reference-array rank to its own sidecar
motion name as the explicit SkillCommander goal. It refuses partial planner
arguments or a source without rank-to-motion names, preventing a single goal
from being silently reused across different reference videos.

For an ordered H30 latent planner, also pass
`--latent_temporal_ensemble exponential` or
`--latent_temporal_ensemble clipped_gated`. The latter means gate, then clip,
then use the same exponential age weights; it is not a non-exponential mode.

It calls `scripts/viz/compare_policy_reference.py`, which is the right
underlying tool: it draws the reference alongside the policy, disables tracking
terminations by default, and runs until the selected trajectory ends.

## Three traps that produce a wrong answer quietly

**1. The agent entry point must match training.** A checkpoint trained under
`rlopt_ipmd_tuned_cfg_entry_point` uses `[1024,1024,512]`, silu, and input
normalization on both nets. Build the default architecture instead and you
either fail to load or load something that is not the trained policy.

- `scripts/rlopt/record_policy_rollout.py` has **no** agent-config flag. Do not
  use it for a tuned checkpoint.
- `scripts/rlopt/play.py` needs `--agent`, added in commit `2e595bf`. Without
  it, it hardcodes `rlopt_{algo}_cfg_entry_point`.
- `compare_policy_reference.py` has `--agent_entry_point`. Use this one.

The wrapper makes the flag required so it cannot be forgotten.

**2. `--reference_visualization` takes a value**, not a bare flag:
`body_markers` | `robot` | `both`. Passed as a flag it swallows the next
argument and argparse dies. `both` draws the marker tensors training uses *and*
the qpos articulation replay.

**3. Do not pass `--video_length` for per-motion renders.** It caps every motion
at the same frame count and silently truncates the long ones — a 774-frame clip
under a 500-step cap loses its last 5.5 s. Omit it and each motion runs to its
own end. Only set `--video_seconds` when you deliberately want a uniform cap.

## The protocol this repo mandates

Every low-level oracle and M3 evaluation needs a **full-horizon diagnostic pass
with all early terminations disabled, including `base_too_low`**, and a video
retained from that same pass. A fall must stay in frame instead of resetting,
or the video shows a truncated rollout rather than the failure.

`compare_policy_reference.py` disables tracking terminations unless
`--keep_terminations` is given, so the default is already the diagnostic pass.
The wrapper keeps that default and says so in its startup banner.

When the evaluation protocol retains startup/reset randomization but removes
pushes, pass `--randomized_no_push`. This forwards the visualizer's paired
`--keep_domain_randomization --disable_push_event` flags: reset-state, mass,
material, and COM randomization stay active, while only `push_robot` is
removed. Policy action selection remains deterministic; environment
randomization and stochastic policy sampling are separate choices.

**Always print the video's absolute path to stdout.** Remote sessions cannot
pass video files back, so the path is the deliverable. The wrapper's summary
does this.

## Data sources

Reference arrays (preferred; see the `bones-seed-dataset` skill):

```
--reference_arrays /path/to/arrays --persist_id <id>
```

The wrapper reads `reference_arrays_manifest.json` to enumerate motions and
resolve their names, and refuses a directory without one — an interrupted build
writes arrays but no sidecar.

Manifest plus Zarr:

```
--motion_manifest /path/manifest.json --dataset_path /path/zarr
```

The two are different sources and cannot be combined; the environment refuses
both together and so does the wrapper.

## Guards

Startup errors rather than a wasted render or a wrong video:

- `--agent_entry_point` is required
- checkpoint must exist
- reference-array directory must carry its sidecar
- the two data sources are mutually exclusive
- `--reference_visualization` is constrained to its three choices
- a source with more than 32 trajectories needs explicit `--ranks` or
  `--max_motions`, since renders are minutes each and the 129k set would never
  finish

## Cost, and where to run

Roughly 2–3 minutes of Isaac startup per motion plus the rollout, one process
per motion. Prefer the **local workstation**: a fresh Isaac Lab container is
expensive to initialize per cluster job, and video is a local-inspection task.
Render on a cluster only when the video is produced inside an already-running
training job.

Rendering needs Kit, so do not pass `--assert-kitless`.

## Worked example

Six BONES-SEED motions against a 4B checkpoint, each at its own length:

```bash
pixi run python .agents/skills/policy-eval-video/scripts/render_policy_videos.py \
  --checkpoint .../model_step_4025155584.pt \
  --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  --output_root logs/videos/bones129k_4b_per_motion \
  --reference_arrays /mnt/hsstorage/fwu91/bones_seed_ref_arrays/eval_subset6/arrays \
  --persist_id bones_seed_subset6@e714bbff \
  -- physics=newton_mjwarp \
  env.sim.physics.solver_cfg.njmax=289 env.sim.physics.solver_cfg.nconmax=200 \
  env.data.runtime_cache_device=cpu env.data.macro_cache_device=cuda:0 \
  env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b] \
  env.command_interface.actor.dim=258 \
  agent.ipmd.latent_dim=258 agent.ipmd.command_source=hl_skill \
  agent.ipmd.hl_skill_checkpoint_path=.../encoder/checkpoints/latest.pt \
  agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
  agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
  agent.ipmd.latent_learning.code_period=10 \
  agent.ipmd.latent_learning.command_phase_mode=sin_cos \
  agent.ipmd.latent_learning.code_latent_dim=256 \
  agent.ipmd.hl_skill_finetune_enabled=false
```

Each video came out one frame short of its reference length — the transition
count — confirming no cap was applied:

| motion | frames | duration |
|---|---:|---:|
| `neutral_walk_180_R_002_A116_M` | 520 / 521 | 10.4 s |
| `jog_ff_loop_180_R_002_A143` | 110 / 111 | 2.2 s |
| `jump_ff_180_R_002_A143` | 131 / 132 | 2.6 s |
| `dance_basic_double_slide_270_R_002_A309` | 398 / 399 | 8.0 s |
| `clap_enthusiastic_002_A122_M` | 773 / 774 | 15.5 s |
| `looking_around_on_ground_002_A053` | 596 / 597 | 11.9 s |

## Related

- `bones-seed-dataset` — building and fetching the reference arrays
- `g1-encoder-interface` — invoke before pairing an encoder with a checkpoint
