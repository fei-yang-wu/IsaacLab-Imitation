# Running Experiments Locally

The current default G1 pipeline, on a workstation. Status: 2026-08-03.

One script runs both stages:

```bash
bash scripts/rlopt/run_local_v2_pipeline.sh
```

It pretrains a **deterministic continuous (det-SR)** skill encoder, then trains
the low-level tracker on the **tuned recipe** conditioned on that frozen
encoder. Same task, agent contract, encoder and data as the ICE 5B runs — only
the scale differs. `DRY_RUN=1` prints both commands and runs nothing.

## What "the default" currently means

| | value | why it is this |
|---|---|---|
| task | `Isaac-Imitation-G1-v2` | highest-numbered `-G1-vN`; the older ids stay registered for frozen reproductions only |
| agent | `rlopt_ipmd_tuned_cfg_entry_point` | the tuned/scaled recipe from the 2026-08-02 screen: 4.6x return/min at matched 100M |
| encoder | `--latent-mode deterministic`, horizon 10, `z_dim` 256 | det-SR; discrete variants (fsq/vq/categorical) exist but are not the default |
| command width | **258** = `z_dim` + 2 | the `sin_cos` phase. Dropping it is catastrophic — episode length 21 against 144 |
| terminations | **instantaneous** | the registered protocol every oracle-qualification number is stated against |
| physics | `newton_mjwarp`, njmax 288 / nconmax 200 | the mjwarp-aligned solver settings the cluster trains on |

The tuned recipe is selected **by entry point**, never by copying its fields.
It is a separate registered class, so earlier runs keep resolving what they
resolved. Reconstructing it from a copied override list is what shipped two 5B
runs at the wrong rollout on 2026-08-03.

The environment half is not on the agent config and is passed explicitly:
`action_rate_l2=0`, `tracking_reward_points=4.0`, termination curriculum 5M→30M.
The script does this for you.

**`tracking_reward_points.weight=4.0` is currently stale.** It was tuned when
that term tracked 3 points without the feet; it now tracks SONIC's 5. Carried
forward unvalidated, so do not treat it as tuned.

## The termination window is off by default

`--termination_window N` makes a tracking term hold for N consecutive control
steps before ending the episode, so a contact spike or push transient can
resolve instead of killing the episode. Thresholds are untouched — only the
boundary moves.

```bash
TERMINATION_WINDOW=10 bash scripts/rlopt/run_local_v2_pipeline.sh
```

Two things to know before using it:

- **3 is under-powered.** The probe on a 1B checkpoint measured that a window of
  3 rescues only 6.7–13.4% of violation onsets, 10 rescues 29.6–49.7%, and a run
  that resolves takes 9.5–13.4 steps. The transients are ~10 steps.
- **It inflates the rate metrics mechanically.** Locally, window 1 → 25 took
  episode length 5.73 → 30.95 while per-step reward fell 0.072 → 0.021. Compare
  on **MPJPE**, which is per-frame and length-independent, and evaluate the
  resulting checkpoint under the *instantaneous* protocol.

Measure rather than guess with `--termination_window_probe`, which disables
tracking terminations and logs `Termination_Window/<term>/recovered_below_<k>_frac`.
There is no upstream value to copy: SONIC ships the counter shape but no
released configuration enables it.

## Budget

From `AGENTS.md`: about **10M frames** for routine debugging, at most about
**50M** for a serious local check, and **do not run 100M locally**. Local runs
qualify code; Skynet/ICE produce convergence and paper numbers.

```bash
TOTAL_FRAMES=10000000 bash scripts/rlopt/run_local_v2_pipeline.sh
```

The script derives iterations from `TOTAL_FRAMES / (NUM_ENVS x 24)`. Default
`NUM_ENVS=4096`; the cluster uses 12288, which measured 55.6 GB of VRAM, so
scale to your card.

## Data

The script needs the LAFAN1 manifest and a zarr cache:

```
data/lafan1/manifests/g1_lafan1_manifest.json
data/lafan1/zarr/g1_hl_diffsr
```

See README "Data preparation" to build them. The **first** run against a fresh
dataset must build the cache (`CACHE_REFRESH=true`); afterwards leave it false,
because a refresh rebuilds the cache underneath any other job reading it.

Audit a tree before trusting it — `data/` has been overwritten by concurrent
work before:

```bash
pixi run -e default python scripts/audit/audit_g1_lafan1_body_frames.py --manifest data/lafan1/manifests/g1_lafan1_manifest.json
```

## Evaluating a checkpoint

Two passes, both required by `AGENTS.md`. Strict is the protocol number;
full-horizon is the honest tracking number, because the strict pass scores MPJPE
only over frames a surviving episode reached.

There is also a separate, optional **SONIC-compatible reporting pass**. It uses
SONIC's released evaluation thresholds, disables `foot_pos_xyz` and the
interval push while retaining startup and reset randomization, evaluates mode
actions deterministically, requires each motion to reach its reference
endpoint, and reports MPJPE-L only over successful motions. It does not replace
either pass below. See
[SONIC-Compatible Success Evaluation](sonic-success-evaluation.md).

```bash
# strict: every termination active -- this is the qualification pass
pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_checkpoint \
  --task Isaac-Imitation-G1-v2 --algo IPMD \
  --checkpoint <ckpt> --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \
  --num_envs 10 --steps 500 --seed 0 --reference_start_frame 0 \
  --motion_manifest ./data/lafan1/manifests/g1_lafan1_manifest.json \
  --dataset_path ./data/lafan1/zarr/g1_hl_diffsr \
  --output_json logs/eval/strict.json --label strict --headless \
  physics=newton_mjwarp env.command_interface.actor.dim=258 \
  agent.ipmd.latent_dim=258 agent.ipmd.command_source=hl_skill \
  agent.ipmd.hl_skill_checkpoint_path=<encoder.pt> \
  agent.ipmd.hl_skill_horizon_steps=10 agent.ipmd.hl_skill_command_mode=z \
  agent.ipmd.latent_steps_min=10 agent.ipmd.latent_steps_max=10 \
  agent.ipmd.latent_learning.code_period=10 \
  agent.ipmd.latent_learning.command_phase_mode=sin_cos \
  agent.ipmd.latent_learning.code_latent_dim=256 \
  agent.ipmd.hl_skill_finetune_enabled=false

# full-horizon diagnostic: add these two, keep everything else identical
  --disable_early_terminations --keep_after_done
```

`--agent_entry_point` is **required** for a tuned checkpoint. Without it the
tool builds the default network and dies on a state-dict mismatch.

Note how far apart the two passes are — on the 2026-08-03 LAFAN1 checkpoint,
25.22 mm strict against 68.08 mm over the full 10 s. Quote the full-horizon
number as tracking quality.

## Reading the metrics

- **`mpjpe_l_mm` (MPJPE-L)** — root-relative, each side against its own root
  position. Pose error with global drift removed. Logged as `mpjpe_mm` too, for
  continuity with older runs.
- **`mpjpe_g_mm` (MPJPE-G)** — world frame, no alignment. Counts drift, always
  the larger. This is the number comparable to published PHC/SONIC-lineage work.

Report both. MPJPE-L alone flatters a policy that holds its pose while walking
away from the reference: the LAFAN1 checkpoint read 25 mm local with 152 mm of
root error underneath it.

## Diagnosing a plateaued reward

If a tracking metric stops improving, check whether the reward term that owns it
still has gradient before changing anything else. IsaacLab logs

```
Episode_Reward/<term> = weight · mean(kernel) · ep_len / max_episode_length_steps
```

so the kernel value is recoverable from any live run:

```python
kernel = episode_reward / (weight * ep_len / 500)      # 500-step horizon
err    = std * sqrt(-log(kernel))                       # implied error
grad   = weight * (-2 * err / std**2) * kernel          # d(w·r)/d(err)
```

A kernel near 1.0 means the exp term is saturated — the policy is paid almost
nothing for further precision there.

Measured on the G1 low level in August 2026, `motion_body_pos` — the term whose
error *is* MPJPE — sat at kernel 0.970 and supplied ~23× less gradient than
`tracking_reward_points`. Narrowing its `std` from 0.30 to 0.05 cut eval MPJPE
by 18.8% with no new reward term. See
`experiments/campaigns/2026-08-04-eval-tracking-screen/`.

**Do this at the TRAINING operating point, not the evaluation one.** Training
runs with domain randomization and exploration noise, so its errors are much
larger and a term can be saturated at eval while having plenty of gradient
during training. Computing it from eval numbers produced a confidently wrong
answer about which term to change.

**And check what the metric is actually made of first.** For the G1, per-step
MPJPE was flat at ~11 mm for 300 steps and then diverged, so the horizon-averaged
number was mostly failures rather than precision — which meant the reward work
moved the strict pass and barely touched the full-horizon one. Decompose before
optimising.

## Gotchas that have cost real time

- **The env is not run-to-run deterministic on Newton at a fixed seed.** Three
  identical runs gave episode length 6.03 / 5.45 / 5.94. Do not read a single
  short run as a result; the 2026-08-02 screen measured ~2% seed spread and
  larger node-to-node variation.
- **Per-minute rates are gameable.** Anything that lengthens episodes — a looser
  threshold, a termination window, a curriculum that has not finished annealing
  — raises return/min and length/min with no better policy. MPJPE is the check.
- **Scoring at a mark some arms have not reached** clamps them to their final
  value and flatters them. Only compare at a mark every arm reached.
- **Isaac entrypoints re-assign `env_cfg` after Hydra.** Trust the run's
  `summary.json` / dumped `params/env.yaml`, not what you passed on the command
  line.

## W&B

Log to the existing shared project (`g1-lafan1`) rather than creating a new one
per experiment. Use tags for environment, primary change, and main features, and
a concise functional group name (`local-v2`, `planner-ablation`) — not a
timestamp. **Confirm the group name with the team before launching.**

## Cluster

Local runs qualify code. For convergence and paper numbers use ICE/Skynet —
see `experiments/campaigns/` for the dated launchers, each of which dry-runs by
default. Related: [Experiment Workflow](experiment-workflow.md),
[LAFAN1 local training](lafan1-local-training.md) (the frozen pre-v2 recipe).
