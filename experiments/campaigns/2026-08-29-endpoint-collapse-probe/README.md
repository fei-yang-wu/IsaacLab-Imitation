# Endpoint-collapse probe (2026-08-29)

Question: does the `diffntp_chunk` skill code z summarize the intermediate
frames of its window, or does it collapse to a function of the last visible
frames?

Definitions used on this page:

- **diffntp_chunk**: `hl_skill_diffsr` pretraining with
  `transition_objective=jepa_ntp`, `jepa_loss=sigreg_ebm`,
  `jepa_ntp_head=diff_chunk`. The loss is the endpoint DiffSR term
  `p(s[t+10] | s_t, z)` plus the chunk diffusion term
  `p(s[t+11..t+20] | s_t, z)`.
- **Visible window**: the encoder input. The production recipe uses
  `encoder_window_mode=intermediate`, so the encoder sees `s_t` plus
  `s[t+1..t+9]` and never the endpoint `s[t+10]`.
- **Endpoint collapse**: the hypothesis that z is in effect a function of the
  last visible frames `s[t+8], s[t+9]` only. Because the macro state
  (`root_qpos`, 380-wide input) carries no velocities, two frames suffice for
  a velocity estimate, so the collapse target is the last frame pair, not the
  single endpoint.
- **suffixN**: new `encoder_window_mode` value. The encoder sees only the last
  N slots of the intermediate window (`s[t+10-N..t+9]`). `suffix9` equals
  `intermediate` at `horizon_steps=10`.

## Tier A: offline probes on the frozen round-4 checkpoint

Checkpoint under test:
`logs/pareto_stack_mirror/diffntp_chunk_h1_ee_wide_seed0/encoder/checkpoints/latest.pt`
(the encoder every smooth-ablation-5b arm binds).

```bash
./experiments/campaigns/2026-08-29-endpoint-collapse-probe/run_tier_a.sh
```

The probe module is
`imitation_experiments.capacity.probe_skill_window_usage`. It rebuilds
expert windows with the exact sampler math (slot-0 heading anchor, interleaved
rot6d; validated against the compiled MDP primitives to 1e-6) and reports:

1. **Frame-sufficiency R2**: held-out R2 of z regressed from frame subsets
   (`last1`, `last2`, `state_last`, `endpoint_unseen`, `visible_all`),
   ridge and small-MLP probes, split by motion.
2. **Per-offset probes**: R2 of every window frame from z, from the visible
   boundary pair, and from boundary+z. The `z_increment_over_boundary` column
   for mid slots is the linear information z carries about mid frames beyond
   what boundary interpolation already implies.
3. **Sensitivity**: normalized latent response to on-manifold mid-frame
   replacement (`mid_swap`), last-frame replacement (`last_swap`),
   interpolated mids (`mid_interp`), and RMSE-matched last-frame noise.
4. **Integrated gradients**: per-slot attribution of
   `||z(x) - z(baseline)||^2` against a batch-permuted counterfactual window.

Collapse signature: `last2` R2 near `visible_all` R2, mid-slot increments
near zero, `mid_swap` response far below `last_swap`, attribution mass on the
state and last slots. These probes are correlational; the retrain arms below
are the falsifier.

## Tier B: suffix-k pretrain arms (dose-response falsifier)

Fixed: the full round-4 diffntp_chunk recipe (horizon 10, z 256, batch 8192,
`Isaac-Imitation-G1-v2`, BONES-SEED full reference arrays, seed 0). The one
moved variable is `encoder_window_mode`: `suffix1`, `suffix2`, `suffix5`,
`suffix9` (= production `intermediate`, retrained for logging parity).

```bash
./experiments/campaigns/2026-08-29-endpoint-collapse-probe/pretrain_suffix_arm.sh suffix1
./experiments/campaigns/2026-08-29-endpoint-collapse-probe/pretrain_suffix_arm.sh suffix2
./experiments/campaigns/2026-08-29-endpoint-collapse-probe/pretrain_suffix_arm.sh suffix5
./experiments/campaigns/2026-08-29-endpoint-collapse-probe/pretrain_suffix_arm.sh suffix9
```

Comparison metrics (now logged per arm in `metrics.jsonl`; the trainer split
was added for this campaign): `train/jepa_endpoint_loss_eval` (dynamics /
endpoint term) and `train/jepa_ntp_loss_eval` (next-chunk prediction term),
both on the frozen 10% trajectory eval split.

Decision rule:

- Both eval losses flat in k: the objective never demanded more than the
  boundary frames. The summarization claim dies for this recipe.
- Losses improve with k: intermediate frames carry information the objective
  uses; the curve shape says how much.
- Flat after k=2: z is boundary state plus velocity, still not a
  summarization.

Local runs are qualification screens. Any paper claim re-runs the winning
comparison on the cluster with repeated seeds per the frozen protocol.

### Run record

Local (workstation, seed 0, 2026-08-29): suffix1, suffix2, suffix5 completed
(50k updates each, ~48 min/arm); suffix9 was killed by the host OOM killer at
its fourth consecutive Isaac boot and then cancelled in favor of the cluster.
Two concurrent local arms do not fit: each holds about 47 GiB of GPU memory
for the macro cache.

ICE (2026-08-30, `campaign.yaml` in this directory, working tree packed with
drift=true): jobs 5598329 (suffix1 s1), 5598330 (suffix1 s2), 5598331
(suffix2 s1), 5598332 (suffix2 s2), 5598333 (suffix5 s1), 5598335
(suffix5 s2), 5598338 (suffix9 s1), 5598339 (suffix9 s2), 5598340
(suffix9 s0). Outputs land in
`/data/endpoint_collapse_probe/<arm>_seed<seed>/encoder/`.

Seed 0 was resubmitted to ICE for suffix1/2/5 (jobs 5598582, 5598580,
5598581) so every arm carries three seeds from one machine. The local seed-0
runs stay as a cross-machine replication, not as table rows.

### Result (ICE seeds 1-2 per arm, seed 0 for suffix9; tail-averaged)

Score with `./pull_and_score.sh`. Tail mean of the last 5,000 updates per
run, then mean over seeds; the bracket is the min and max of the per-seed
tail means, not a standard deviation.

| arm | visible slots | endpoint eval | next-chunk eval |
|---|---:|---|---|
| suffix1 | 1 | 0.2767 [0.2751, 0.2783] | 7.7333 [7.7014, 7.7652] |
| suffix2 | 2 | **0.1877** [0.1835, 0.1918] | **7.0827** [7.0623, 7.1031] |
| suffix5 | 5 | 0.1947 [0.1888, 0.2006] | 7.0852 [7.0829, 7.0875] |
| suffix9 (production) | 9 | 0.2129 [0.2097, 0.2183] | 7.1714 [7.1545, 7.1830] |

Cross-machine replication, local seed 0: suffix1 0.2927 / 7.7200,
suffix2 0.1872 / 7.0858, suffix5 0.1973 / 7.3141. Every local value sits in
or beside its ICE seed range, so the machine is not driving the ordering.

**Reading.** The whole gain is in the step from one visible frame to two:
endpoint -32%, next-chunk -8.4%, far outside seed spread. From two frames to
nine, both losses stop improving and drift slightly the wrong way
(suffix2 beats production suffix9 by 11.9% on endpoint and 1.2% on
next-chunk, with disjoint seed ranges on both). Frames 3 through 9 of the
window add nothing this objective can use. This is the "flat after k=2"
branch of the decision rule above.

**What a suffix arm actually hides, and what it does not.** Every window slot
is expressed in slot 0's heading frame, and a suffix slice does NOT
re-anchor. Measured on 900 reference windows, the anchored planar
displacement grows linearly across the window (slot 0 exactly 0 by
construction, slot 9 at 3.7 cm on average), so the last visible frame carries
the NET nine-step displacement from the window start. A suffix2 encoder
therefore still sees a window-scale quantity: where the motion ended up
relative to where it began, plus the rate at which it arrives. What suffix2
loses relative to suffix9 is only the PATH SHAPE between those points.

So the correct statement of this result is narrower than "z is a local
boundary code": the objective does not reward encoding path shape, but the
information z does keep is integrated over the window, not read off a single
instant. Calling the two visible frames "the boundary" invites the wrong
reading.

**Scope.** These are pretraining eval losses. They say the objective does not
require path shape; they do not yet say the tracker is indifferent to it.
Tier C answers that.

### Open follow-ups this result implies

1. **Matched-width coverage (the clean path-shape test).** suffix1/5/9 differ
   in encoder input width as well as coverage, so dimensionality is a
   confound. Compare `suffix3` (slots 7,8,9) against a strided arm seeing
   slots 1,5,9: identical input width and parameter count, opposite temporal
   coverage. Flat means path shape is genuinely inert; a strided win means
   coverage matters and the suffix ladder was confounded by width.
2. **Displacement removal.** Re-anchor the visible suffix onto its own first
   visible slot, which deletes the net-displacement channel and leaves local
   pose and velocity. If the losses degrade sharply, the integrated
   displacement was carrying the result, and the window-summary claim
   survives in its weaker form.

Neither arm exists yet; both are one `encoder_window_mode` value plus a slice.

## Not covered here

- Tier C (tracker command swap on the 4096 board) waits for the Tier B
  verdict.
- The deployment bundle (`external/Embodied-Control` `bundle.py`) does not
  know `suffixN` window modes; suffix arms must not be exported as policy
  bundles.
