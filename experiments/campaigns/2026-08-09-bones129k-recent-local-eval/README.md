# Recent ICE low-level tracker local evaluation

## Question

Which of the seven finished BONES-129k low-level tracker jobs from 2026-08-08
and 2026-08-09 has the best SONIC-compatible tracking result?

## Protocol

The launcher evaluates each latest finished checkpoint twice:

- `selected10`: the canonical ten language-development motions, with one
  environment per motion;
- `scoreboard4096`: one environment per trajectory for the frozen rank block
  12288 through 16383.

Both suites use seed 0, frame-0 starts, deterministic actions, normal startup
and reset randomization, no push, and the released SONIC tracking thresholds.
The 4,096 suite uses the same rank block as the 2026-08-08 scoreboard.

The seven rows keep their training-time encoder input, anchor frame, latent
size, actor/value network size, critic channels, and encoder fine-tune mode.
The running 10B L2T job is not included because it has no final checkpoint.
The 4,096 suite maps the reference arrays and builds its compact macro cache in
host memory. This keeps the 20 GiB local GPU available for the Newton scene.

## Run

From the repository root:

```bash
experiments/campaigns/2026-08-09-bones129k-recent-local-eval/run.sh
```

To print only the completed result table:

```bash
experiments/campaigns/2026-08-09-bones129k-recent-local-eval/run.sh --report
```

The public NVIDIA SONIC v1.1 checkpoint uses its native 640-value encoder,
FSQ tokenizer, 930-value proprioception history, and 29-action decoder. It has
a separate launcher because it is not an IPMD policy:

```bash
MODE=print experiments/campaigns/2026-08-09-bones129k-recent-local-eval/run_sonic_v1_1.sh
MODE=smoke experiments/campaigns/2026-08-09-bones129k-recent-local-eval/run_sonic_v1_1.sh
experiments/campaigns/2026-08-09-bones129k-recent-local-eval/run_sonic_v1_1.sh
```

The launcher pins Hugging Face `nvidia/GEAR-SONIC` commit `9c0ff22b4ffe` and
checkpoint SHA-256 `af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43`.
It requires the files under
`logs/downloaded_checkpoints/nvidia_GEAR_SONIC_9c0ff22/sonic_v1_1/`.

The launcher is resume-safe. It validates an existing JSON file before it
skips that run. Results are under `logs/bones129k_recent_ice_local_eval/`.

## Results

Completed locally on 2026-08-10. All 15 comparison JSON artifacts passed the
protocol checks. The 4,096 rows all use rank SHA-256
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`.
No run timed out and no solver constraint-buffer overflow was present.

| Tracker | Selected 10 success | Selected 10 successful MPJPE-L (mm) | 4,096 success | 4,096 successful MPJPE-L (mm) |
| --- | ---: | ---: | ---: | ---: |
| `expert_heading` | 2/10 | 103.74 | 3,758/4,096 (0.9175) | 32.27 |
| `critic_no_latent` | 1/10 | 75.10 | 3,731/4,096 (0.9109) | 33.26 |
| `fsq64_heading_critic_no_latent` | 0/10 | - | 3,681/4,096 (0.8987) | 35.02 |
| `z256_scaled` | 0/10 | - | 3,746/4,096 (0.9146) | **23.27** |
| `ee_reward` | **5/10** | 94.19 | 3,748/4,096 (0.9150) | 32.56 |
| `encoder_finetune` | 1/10 | 98.13 | 3,732/4,096 (0.9111) | 34.55 |
| `fullbody_encoder` | 0/10 | - | 0/4,096 (0.0000) | - |
| NVIDIA `sonic_v1_1` | not run here | - | **4,073/4,096 (0.9944)** | 26.96 |

The selected ten motions are a difficult and high-variance development set.
The 4,096 block gives the stable ranking. NVIDIA SONIC v1.1 has the highest
success rate overall. `expert_heading` has the highest local-policy success
rate, while `z256_scaled` has nearly the same local success rate and much lower
successful tracking error. The `fullbody_encoder` checkpoint fails in two to
six steps, mainly through `anchor_ori`; it is not a usable tracker.

Success-only error is biased when success sets differ. On the 3,743 motions
completed by both SONIC v1.1 and `z256_scaled`, SONIC v1.1 has 25.87 mm MPJPE-L
and `z256_scaled` has 23.22 mm. Thus v1.1 completes 327 more motions, while the
local scaled tracker is 2.65 mm tighter on their common success set.

## Status

Complete as of 2026-08-10. The ICE 10B L2T job is still separate and was not
included because it did not have a final checkpoint when this comparison ran.

## Additional FSQ evaluation

Five earlier BONES-129k FSQ trackers were added to the same local protocol on
2026-08-10. They use the same selected-ten manifest and the same frozen
4,096-motion rank block. All large passes completed with zero timeouts, the
released SONIC termination terms only, and rank SHA-256
`786ef6775930c34179b774cb215e233c3f7b2bb32ef46bb6fc660206324e8285`.

| Tracker | Frames | Command contract | Selected 10 | 4,096 success | Successful MPJPE-L |
| --- | ---: | --- | ---: | ---: | ---: |
| `online_fsq32_future10` | 3.05B | online FSQ64, consecutive future window, hold 1 | 3/10 | 1,412/4,096 (0.3447) | 64.76 mm |
| `online_fsq32_v2_keypoint_stride5` | 3.50B | online FSQ64, 14-keypoint plus root-orientation window at stride 5, hold 1 | 2/10 | 1,410/4,096 (0.3442) | 63.08 mm |
| `fsq64_tuned_stride1` | 5.00B | frozen root-qpos FSQ64, tuned tracker, stride 1, hold 10 | 2/10 | 3,578/4,096 (0.8735) | 27.19 mm |
| `fsq64_scaled_stride1` | 5.00B | frozen root-qpos FSQ64, scaled tracker, stride 1, hold 10 | 0/10 | 3,702/4,096 (0.9038) | 25.44 mm |
| `fsq64_scaled_stride5` | 5.00B | frozen root-qpos FSQ64, scaled tracker, stride 5, hold 10 | 2/10 | 2,790/4,096 (0.6812) | 37.88 mm |

The frozen stride-1 pair is the useful FSQ result. Scaling the tracker adds 124
completed motions and lowers successful MPJPE-L by 1.75 mm. Changing that
scaled recipe from stride 1 to stride 5 removes 912 completed motions and
raises successful MPJPE-L by 12.43 mm. The online hold-1 rows fail on about
two-thirds of the block under this controller recipe. The release-aligned
keypoint input does not recover success relative to the consecutive-window
online row.

The main failure remains `ee_body_pos`. Its event count is 314 for scaled
stride 1, 1,165 for scaled stride 5, 2,084 for online consecutive FSQ, and
2,078 for online keypoint FSQ. These are ankle-or-wrist Z-error events under
the released 0.25 m threshold.

Downloaded checkpoints and generated result JSON files are under
`logs/downloaded_checkpoints/bones129k_fsq_variants/` and
`logs/bones129k_recent_ice_local_eval/{selected10,scoreboard4096}/`.
