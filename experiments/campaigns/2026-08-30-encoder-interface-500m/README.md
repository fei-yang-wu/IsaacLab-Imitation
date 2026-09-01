# Encoder interface at 500M (2026-08-30)

Which encoder makes the best tracker? Nine encoders, one tracker recipe,
500M environment frames each, hold 1.

This is Tier C of the endpoint-collapse investigation in
[`../2026-08-29-endpoint-collapse-probe/`](../2026-08-29-endpoint-collapse-probe/README.md).
That campaign asked what the pretrain objective demands of the encoder window
and found it demands very little — at horizon 10, an encoder seeing two window
slots matched or beat one seeing nine on both pretrain eval losses. Pretrain
loss is not what we ship, so this campaign asks the deployable form of the
question: bind each encoder to a tracker, train, and compare tracking.

## Arms

Every arm matches the `smooth-ablation-5b` base recipe except the encoder it
binds, the horizon that encoder was pretrained at, and the parallel
environment count. Frozen encoder, 256-D code, 258-wide command, hold 1,
12,288 envs x 24 rollout steps.

The environment count moved from the base recipe's 20,480 to 12,288 on
2026-08-30, when the campaign moved from H200 to H100 nodes to reach free
GPUs sooner. The smaller count keeps the run inside H100 memory. It changes
`frames_per_batch` to 294,912 and the minibatch to 221,184, and it holds the
500M frame budget. Every arm moved together, so the encoder ordering inside
this campaign is still a single-variable comparison. The `prod` arm is no
longer an exact recipe match for the 5B and 50B history, so treat it as this
campaign's noise-floor and reference row rather than as a bridge row.

| arm | encoder | pretrain horizon | encoder sees |
|---|---|---:|---|
| `prod` | `diffntp_chunk_h1_ee_wide` (round-4) | 10 | slots 1-9 |
| `suffix1` | this investigation | 10 | slot 9 only |
| `suffix2` | " | 10 | slots 8-9 |
| `suffix5` | " | 10 | slots 5-9 |
| `suffix9` | " | 10 | slots 1-9 |
| `h1` | " | 1 | slot 1 (**is** the endpoint) |
| `h2` | " | 2 | slot 1 |
| `h5` | " | 5 | slots 1-4 |
| `h10` | " | 10 | slots 1-9 |

`agent.ipmd.hl_skill_horizon_steps` is set per arm to match the checkpoint.
`FrozenHighLevelSkillCommandSampler` raises on a mismatch rather than feeding
the encoder a wrong-width window, so a misconfigured arm dies at startup
instead of training on garbage.

## Two things to read before reading the numbers

**The noise floor is built in.** `prod`, `suffix9` and `h10` are the same
encoder configuration pretrained three times independently. Their spread is
this campaign's encoder-initialization noise. Any difference between other
arms smaller than that spread is not a result.

**`h1` is a ceiling reference, not a curve point.** Horizon 1 has no
intermediate window, so its encoder had to use `window_mode=full`, and its one
visible frame IS the endpoint target. Its code can carry the answer outright.

## Budget caveat

500M frames is a screen, not a verdict. The production regime is 5B and this
recipe's milestone curve flattens only past roughly 3-4B. The 2026-08-20
interface design study already showed a short budget inventing a "64-D hold-1
dead zone" that a longer budget erased. Treat any ordering here as a
hypothesis about the 5B ordering and do not promote an arm on it alone.

Two further protocol notes. The failure-share ramp runs 0.8 to 0.2 over the
first 1B, so at 500M every arm stops mid-ramp at roughly 0.5 — identical
across arms, but not the converged selection regime. The termination
curriculum (5M to 30M) does complete well inside the budget.

## Submission

Submitted 2026-08-30 on ICE, H100, seed 0, one segment per arm, W&B group
`encoder-interface-500m`. The suffix family runs in the `ice-gpu` partition
under the `coe-ice` QoS; the horizon family runs in `coe-gpu` under
`coe-grade`, because every `ice-gpu` H100 was allocated while `coe-gpu` held
free ones. `coe-gpu` uses `PreemptMode=REQUEUE`, so a horizon arm can be
requeued and restart from its last checkpoint.

| arm | job | partition |
|---|---:|---|
| `prod` | 5599258 | ice-gpu |
| `suffix1` | 5599260 | ice-gpu |
| `suffix2` | 5599261 | ice-gpu |
| `suffix5` | 5599262 | ice-gpu |
| `suffix9` | 5599263 | ice-gpu |
| `h1` | 5599366 | coe-gpu |
| `h2` | 5599367 | coe-gpu |
| `h5` | 5599369 | coe-gpu |
| `h10` | 5599358 | coe-gpu |

Superseded submissions, none of which produced frames:

- 5599236-5599245, H200, cancelled while pending, because H100 nodes were free.
- 5599264-5599267, H100 in `ice-gpu`, cancelled while pending, to move the
  horizon family to `coe-gpu`.
- 5599355 (`h1`), 5599356 (`h2`), 5599357 (`h5`), which died within 45 seconds
  with `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy
  or unavailable` on nodes `atl1-1-03-011-8-0` and `atl1-1-03-011-13-0`. The
  GPUs were unusable, not the recipe. Both nodes are now in the stage's
  `exclude` list.

## Running

```bash
./experiments/campaigns/2026-08-30-encoder-interface-500m/submit_all.sh
```

The script waits for the pretrain jobs of the Tier B campaign to leave the
queue, verifies every encoder checkpoint exists on the cluster, and refuses to
submit if any is missing. Scoring uses the standard checkpoint-tree evaluation
on `paper_testbed4096_v1`; per AGENTS.md every result table also carries the
`sonic_v1_1` row for that same board.

## Results (2026-08-30, seed 0, `bones_testbed4096_v1`)

Scored with `eval.sh` on the final checkpoint, `f500170752` (0.50B frames).
`clean` is `--randomization none`; `robust` is `no_push`. The `sonic_v1_1`
row is the released SONIC checkpoint on this same board, taken from Table A of
`experiments/campaigns/2026-08-17-paper-metric-canon/README.md`; it is matched
over 3,932 of the 4,096 clips, so it is a reference point rather than a row
from this sweep, and it carries no jerk column.

| arm | row | SR | MPJPE-L | MPJPE-G | jerk | acc |
|---|---|---:|---:|---:|---:|---:|
| public `sonic_v1_1` | clean | 0.9888 | 26.25 mm | 177.41 mm | - | 3.34 |
| `srccur10` | clean | 0.8789 | 28.42 mm | 164.04 mm | 214.0 | 5.11 |
| `h10` | clean | 0.8765 | 29.60 mm | 155.81 mm | 225.4 | 5.33 |
| `suffix2` | clean | 0.8723 | 30.09 mm | 181.21 mm | 207.4 | 5.04 |
| `prod` | clean | 0.8713 | 30.06 mm | 145.02 mm | 210.8 | 5.13 |
| `suffix9` | clean | 0.8701 | 29.84 mm | 153.26 mm | 212.2 | 5.14 |
| `suffix5` | clean | 0.8694 | 29.14 mm | 157.96 mm | 212.3 | 5.10 |
| `h5` | clean | 0.8677 | 30.38 mm | 132.93 mm | 251.9 | 5.55 |
| `suffix1` | clean | 0.8635 | 31.87 mm | 169.83 mm | 213.2 | 5.29 |
| `h2` | clean | 0.7891 | 34.65 mm | 157.26 mm | 265.5 | 5.88 |
| `h1` | clean | 0.7878 | 37.82 mm | 151.07 mm | 244.8 | 5.67 |
| `srccur10` | robust | 0.8518 | 31.65 mm | 301.41 mm | 260.4 | 5.97 |
| `suffix9` | robust | 0.8494 | 33.30 mm | 247.23 mm | 247.6 | 5.81 |
| `h10` | robust | 0.8469 | 32.62 mm | 259.66 mm | 258.9 | 5.95 |
| `suffix2` | robust | 0.8450 | 33.56 mm | 302.49 mm | 239.8 | 5.67 |
| `prod` | robust | 0.8374 | 33.51 mm | 236.40 mm | 242.8 | 5.71 |
| `suffix5` | robust | 0.8369 | 32.41 mm | 278.57 mm | 257.3 | 5.91 |
| `suffix1` | robust | 0.8333 | 35.17 mm | 257.70 mm | 244.8 | 5.85 |
| `h5` | robust | 0.8318 | 34.43 mm | 212.72 mm | 286.7 | 6.24 |
| `h2` | robust | 0.7078 | 41.93 mm | 313.77 mm | 288.8 | 6.54 |
| `h1` | robust | 0.7041 | 45.09 mm | 297.31 mm | 267.3 | 6.26 |

### The noise floor this campaign measured

`prod`, `suffix9` and `h10` are the same recipe pretrained three times. Their
clean spread is 0.0064 SR, 0.46 mm MPJPE-L and 10.8 mm MPJPE-G; their robust
MPJPE-G spread is 23.3 mm. Checkpoint-to-checkpoint movement inside one arm is
larger still: `h10` clean SR moves 0.0135 and its MPJPE-G moves 16.4 mm
between `f400195584` and `f500170752`. Read both numbers before calling any
gap real.

### Window content: no resolved ordering

Every suffix arm and both wide-horizon arms sit inside the replicate band on
SR. `suffix1` is the weakest of them at 0.8635 clean, which is 0.0066 below
the band and not resolved. The two-visible-slot pretrain advantage found in
Tier B does not reappear here: `suffix2` and `suffix9` are level.

### The horizon cliff: the one resolved finding

`h1` (0.7878) and `h2` (0.7891) lose about 0.08 SR against `h5` (0.8677) and
`h10` (0.8765) on the clean row, and about 0.13 SR on the robust row. The gap
is more than ten times the replicate spread, and its mechanism is visible in
the termination counts: `ee_body_pos` terminations roughly double, from
399-468 at horizon 10 to 743-748 at horizons 1 and 2. A horizon-1 or
horizon-2 encoder cannot express enough of the future for the tracker to keep
the wrists on the reference. `h1` was already a ceiling reference by
construction, so `h2` carries this result.

### phi past-chunk conditioning: a null

`srccur10` pretrained phi on the past chunk `s[t-10..t]` anchored at `s_t`,
against its matched control `h10`. Both checkpoints scored:

| checkpoint | row | arm | SR | MPJPE-L | MPJPE-G | jerk |
|---|---|---|---:|---:|---:|---:|
| f500170752 | clean | `srccur10` | 0.8789 | 28.42 mm | 164.04 mm | 214.0 |
| f500170752 | clean | `h10` | 0.8765 | 29.60 mm | 155.81 mm | 225.4 |
| f400195584 | clean | `srccur10` | 0.8662 | 29.51 mm | 182.76 mm | 220.8 |
| f400195584 | clean | `h10` | 0.8630 | 30.50 mm | 172.21 mm | 220.9 |
| f500170752 | robust | `srccur10` | 0.8518 | 31.65 mm | 301.41 mm | 260.4 |
| f500170752 | robust | `h10` | 0.8469 | 32.62 mm | 259.66 mm | 258.9 |
| f400195584 | robust | `srccur10` | 0.8313 | 32.90 mm | 327.27 mm | 270.3 |
| f400195584 | robust | `h10` | 0.8303 | 33.69 mm | 277.30 mm | 255.8 |

SR moves by 0.001 to 0.005 in `srccur10`'s favor, inside the replicate band.
MPJPE-L is about 1 mm better at both checkpoints, a consistent sign but only
3%, well inside the unresolved band. MPJPE-G is worse at both checkpoints:
about 10 mm on the clean row, which is inside checkpoint variance, and 42 to
50 mm on the robust row, which is about twice the replicate spread and keeps
its sign. So the past chunk buys a small local-pose gain and pays for it in
global root drift under randomization, and the +5.1% pretrain endpoint penalty
neither survived nor reversed. Treat the tracking effect as a null at 500M on
one seed.

`srcpast10` has no tracker arm and cannot get one: its encoder window is
anchored on `s[t-10]`, and the frozen command sampler anchors on the live
robot heading only.

### Scoring notes

The first pass of `h1`, `h2`, `h5`, `h10` and the `suffix9` robust row died
with `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.43 GiB`,
caused by an unrelated process holding 54 GiB on the same GPU. They were
re-scored on an idle GPU; the failed attempts wrote no rows.
