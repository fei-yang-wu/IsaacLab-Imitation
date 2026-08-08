# Discrete FSQ64 language planner on the selected ten BONES-SEED goals

This campaign repeats the 2026-08-05 selected-ten oracle-trajectory planner
protocol without changing it, driven by the **discrete** command interface
instead of the continuous one.

**"Discrete version"** here means the scaled SONIC-FSQ64 stack from campaign
`2026-08-06-bones129k-sonic-fsq-scale`: the shared DiffSR successor-representation
encoder emits 64 finite-scalar-quantization coordinates at 32 levels each
(about 320 bits), and the SONIC-sized tracker
(`[2048, 2048, 1024, 1024, 512, 512]`, SiLU) consumes that code plus two
sin/cos phase channels, so the actor command is **66 values** wide. The
continuous baseline of 2026-08-05 uses a 256-wide real-valued code, a
258-value command, and a `[1024, 1024, 512]` tracker. Both encoders read the
same 380-value `root_qpos` macro state, hold one command for ten 50 Hz control
steps, and were trained on the full 129,785-motion BONES-SEED reference set.

Bound artifacts:

| role | path | SHA-256 |
|---|---|---|
| tracker, 4.5B frames (ICE `5570936`, `sonic_tracker_h200_retry1`) | `logs/bones129k_sonic_fsq_scale_eval/4500357120/fsq64_sonic/model_step_4500357120.pt` | `1e8555a5…9653` |
| frozen scaled FSQ64 encoder (ICE `5570673`) | `logs/bones129k_sonic_fsq_scale_eval/encoders/fsq64_scaled.pt` | `6a4a7248…da14` |
| selected-ten manifest | `data/bones_seed_language10_v1/manifests/…_manifest.json` | `60a5b7a5…e7d1` |
| MiniLM goal table | `data/bones_seed_language10_v1/language/…_minilm_goal_embeddings.pt` | `04624a22…e0cd` |
| selected-ten reference arrays | `data/bones_seed_language10_v1/reference_arrays/root_qpos_v1` | `e8996c26…90ee` |

`validate_latent_skill_checkpoint_binding` passes: the tracker embeds the same
13 encoder tensors as the selected skill checkpoint. The launcher re-runs that
binding check on every invocation and refuses a hash mismatch on any of the
five artifacts above.

Two contract values differ from the 2026-08-05 launcher, both forced by the
tracker itself:

- `njmax` is 320, not 289. The matched low-level evaluation of this exact
  checkpoint overflowed the Newton constraint buffer at 289
  (`nefc overflow - please increase njmax to 296`), and its valid results
  used 320.
- The actor and critic hidden widths are passed explicitly, because a strict
  restore of the SONIC-sized tracker fails against the entry-point default
  geometry.

## Protocol

Unchanged from 2026-08-05: one 1,000-environment collection process (ten
motions x 100 environments), frame-0 starts, deterministic policy actions,
SONIC tracking terminations with foot XYZ and base height disabled, no push
event, all other randomization retained; one complete oracle trajectory per
environment; the medium flow Transformer trained on oracle rows only for
10,000 updates with a trajectory-wise 80/20 split; closed-loop evaluation of
every 2,000-update checkpoint against all ten explicit language goals at 100
environments per goal.

The planner regresses the encoder's published code directly, exactly as in the
continuous baseline. It is **not** projected back onto the FSQ lattice at
publication; keeping the planner path byte-identical to the baseline is what
makes the two interfaces comparable. Lattice snapping remains an available
follow-up diagnostic, not part of this run.

## Collection result (seed 0, 2026-08-07)

999 of 1,000 environments finished their reference without a SONIC tracking
failure (0.999). The single failure was one `ee_body_pos` termination on
`cellphone_typing_sequence_one_hand_idle_R_001_A423`; every other goal was
100/100. The run recorded 513,058 valid control transitions across seven
sample shards. The continuous baseline collected 1,000/1,000 and 513,700
transitions on the same ten goals, so the demonstration sets are matched in
size and quality.

## Commands

From the repository root:

```bash
MODE=print experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
MODE=run experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
MODE=video experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
```

Single stages and optimizer-preserving extensions work as in the 2026-08-05
campaign:

```bash
MODE=run RESUME=1 STAGE=collect experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
MODE=run RESUME=1 STAGE=train NUM_UPDATES=20000 experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
MODE=run RESUME=1 STAGE=eval NUM_UPDATES=20000 experiments/campaigns/2026-08-07-bones-language10-fsq64-planner/run.sh
```

Output root: `logs/bones_language10_fsq64_planner_seed0`. `milestone_curve.md`
carries the closed-loop budget curve; `MODE=video` renders the mandatory
non-terminating randomized/no-push comparison videos and prints their absolute
paths.
