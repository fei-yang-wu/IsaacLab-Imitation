---
name: g1-encoder-interface
description: Select the G1 skill-encoder input interface (full_body qpos+qvel+root vs root_qpos qpos+root) for training, pretraining and evaluation. Use when a run must change what the DiffSR encoder compresses, when pairing a policy with an encoder, or when a job fails with "hl/state shape mismatch".
---

# G1 skill-encoder input interface

The DiffSR skill encoder compresses a 10-frame window of *reference* state into
the latent command the actor consumes. **Which reference components go into that
window is configuration**, and it is the only thing this skill is about.

Two interfaces are in use:

| interface | per-frame components | width | 10-frame encoder input |
|---|---|---|---|
| `full_body` (default) | 29 joint qpos + 29 joint qvel + 3 root pos + 6 root ori | 67 | **670** |
| `root_qpos` | 29 joint qpos + 3 root pos + 6 root ori | 38 | **380** |

`root_qpos` is `full_body` minus joint velocity.

**The actor's command is unchanged either way** — 258 (`z_dim` 256 + `sin_cos`
phase). Only what was compressed into `z` differs. Do not change
`env.command_interface.actor.dim` when switching interfaces.

## The knob

One env field, on both the training and the pretrain entrypoints:

```bash
# full_body -- the default; omit the flag entirely
# root_qpos:
env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
```

`expert_motion` is the 58-wide qpos+qvel term; `expert_motion_qpos` is the
29-wide qpos-only term. The anchor terms are the same in both.

Resolution lives in `ImitationRLEnv._effective_expert_macro_state_terms()`
(`config/g1/common/tracking_env.py`), which the data plane reads via
`_expert_macro_feature_term_order()` (`envs/expert_data_plane.py`). `None` means
`full_body`. The field also accepts the raw Hydra string form `"[a,b,c]"` and
parses it, so the CLI form above works as written.

Do **not** try to select the interface through
`env.command_interface.encoder`. That preset chooses the *window*
(`causal9`, `future10`, ...), not the component set.

## Each interface needs its own encoder — this is the trap

An encoder's first layer is built for one input width. A 380-D encoder cannot
consume a 670-D macro state. So **an interface change requires a matching
encoder pretrain**; you cannot reuse the default one.

Encoders are also dataset-specific. A LAFAN1 `root_qpos` encoder cannot drive a
BONES-SEED policy.

Pretrain one with the same override:

```bash
python scripts/rlopt/train_hl_skill_diffsr.py \
  --task Isaac-Imitation-G1-v2 --headless --assert-kitless \
  --latent_mode deterministic --z_dim 256 \
  --horizon_steps 10 --encoder_window_mode intermediate \
  --output_dir <store>/lafan1_v2_root_qpos_det_sr_h10_z256_seed0 \
  --num_updates 50000 --batch_size 8192 --seed 0 \
  env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
```

Flags are underscore-separated (`--z_dim`, not `--z-dim`); the only hyphenated
one is `--assert-kitless`. `--horizon_steps` defaults to 25, so pass 10
explicitly to match the current recipe.

Then point training at it with `agent.ipmd.hl_skill_checkpoint_path=<...>/latest.pt`
**and** repeat the `expert_macro_state_terms` override. Both, every time.

## Verify the width before spending a slot

Nothing downstream validates an encoder's input space against the interface it
is paired with, and the record will happily claim the wrong interface. Read the
first layer:

```bash
python -c "
import torch
d = torch.load('<encoder>/latest.pt', map_location='cpu', weights_only=False)
k, v = next(iter(d['skill_encoder_state_dict'].items()))
n = int(v.shape[1])
print(k, tuple(v.shape), {380: 'root_qpos', 670: 'full_body'}.get(n, f'UNKNOWN {n}'))
"
```

A launcher that runs often should pin the encoder by sha256 rather than by name
— see `experiments/campaigns/2026-08-04-eval-tracking-screen/submit_root_qpos_ice.sh`.

## Failure modes

**Mismatch fails loudly, at the first forward:**

```
ValueError: hl/state shape mismatch: expected (N, 38), got (N, 67)
```

`expected` is the encoder's width, `got` is what the env published. Fix whichever
is wrong — usually a missing `expert_macro_state_terms` override on a run that
loaded a `root_qpos` encoder.

**Evaluation must replay the override.** `evaluate_checkpoint` rebuilds the env
and the actor from the task id plus whatever overrides you pass. Omit
`expert_macro_state_terms` and you rebuild the wrong interface. Pass the same
override and the same encoder as training:

```bash
python -m imitation_experiments.lowlevel.evaluate_checkpoint \
  --task Isaac-Imitation-G1-v2 --algo IPMD --checkpoint <ckpt> \
  --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  ... \
  agent.ipmd.hl_skill_checkpoint_path=<root_qpos encoder> \
  env.expert_macro_state_terms=[expert_motion_qpos,expert_anchor_pos_b,expert_anchor_ori_b]
```

**Brackets survive the cluster path.** `[a,b,c]` reaches the compute node intact
through sbatch and singularity; the backslash-escaped form printed in dry-run
output is display-only.

## Contract test

`test_macro_state_terms_select_the_encoder_input_width` in
`source/isaaclab_imitation/tests/test_g1_metrics_contract.py` pins 670 vs 380.
Run `pixi run -e isaaclab test-isaaclab` after touching any of this.

## Why root_qpos exists

Measured at 500M on LAFAN1 with matched rewards (2026-08-04): strict MPJPE-G
lands inside the full_body arm's three-seed range, so dropping joint velocity
costs nothing in precision, while survival was the highest measured (445.3) and
full-horizon MPJPE-G the best (0.1503 against full_body's 0.1740-0.2303). Single
seed on a pass with ~28% training-seed spread, so treat it as the leading
candidate rather than a settled result. See
`experiments/campaigns/2026-08-04-eval-tracking-screen/README.md`.
