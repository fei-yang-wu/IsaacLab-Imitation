# Fair Interface Baselines

This workflow compares learned skill latents against hand-designed planner
interfaces for the same frozen low-level policy. The first target is Dance102,
no language, and matched achieved-state finetuning.

The paper-facing baseline is a strong internal VLA-style comparison, not a
direct OpenPI/GR00T reproduction. The goal is to make the hand-designed
interfaces competitive inside this repo's data, robot, simulator, planner, and
finetune stack, then ask whether a learned skill latent is a better high-low
interface under those controlled conditions. External VLA integrations remain
future work unless the project scope expands to their tokenizer/action head,
embodiment adapter, data conversion, and evaluation infrastructure.

## Interfaces

- `latent_skill`: existing SkillCommander route from the
  `nostalgic-stonebraker-6e0bbc` branch.
- `ee_trajectory`: planner predicts future wrist/foot end-effector pose
  trajectory commands.
- `full_body_trajectory`: planner predicts future whole-body trajectory
  commands.

The hand-designed planners use the same flow-matching family and budget as the
latent commander comparison: oracle-drive sample collection, expert-state
pretrain, achieved-state finetune, and closed-loop eval.

## Dance102 Command

Set the frozen low-level checkpoints explicitly. `LOW_LEVEL_CHECKPOINT` is a
convenience fallback for the hand-designed interfaces, but the intended fair run
uses the interface-specific checkpoints because the EE and full-body low-level
policies are trained separately.

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_100073472.pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_100073472.pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/.../models/model_step_....pt \
LATENT_SKILL_CHECKPOINT=logs/hl_skill_diffsr/.../checkpoints/latest.pt \
LATENT_PLANNER_CHECKPOINT=logs/skill_commander/.../checkpoints/latest.pt \
NUM_ENVS=1 \
STEPS=1000 \
STATE_HISTORY_STEPS=0 \
COMMAND_FUTURE_STEPS=25 \
FINETUNE_UPDATES=2000 \
FINETUNE_WEIGHT_DECAY=1.0e-4 \
experiments/interface_baselines/run_dance102_fair_interface_comparison.sh
```

The default run includes `latent_skill`, `full_body_trajectory`, and
`ee_trajectory`. Set `RUN_LATENT=0` for a hand-designed-only smoke/debug pass.
If the latent planner is language-conditioned, also set
`LATENT_LANGUAGE_EMBEDDINGS`; no-language checkpoints should leave it unset.
For train/eval splits, set `TRAIN_MANIFEST` and `EVAL_MANIFEST`; the legacy
`MANIFEST` variable remains the single-manifest default for Dance102.

The script writes:

- per-result-root `interface_comparison_run_provenance.json`
- oracle low-level evals
- oracle-drive achieved-state samples
- expert-state planner pretrain checkpoints for hand-designed interfaces
- offline expert-state planner target evals
- pretrained closed-loop evals
- achieved-state finetune checkpoints
- offline achieved-state planner target evals
- finetuned closed-loop evals
- long-form `interface_comparison.csv` / `interface_comparison.md`
- paper-facing wide `interface_comparison_wide.csv` /
  `interface_comparison_wide.md`

under `logs/interface_baselines/dance102_fair_interface_comparison` by default.

## Strong Internal Planner Baseline

For reviewer-facing comparisons against VLA-style command spaces, use the
chunked Transformer flow planner. The small MLP planner from the basic fair
runner is only a diagnostic. The strong planner keeps the same data protocol and
closed-loop command-buffer API, but
uses a stronger action-chunk model:

- command vectors are split into fixed-width tokens;
- state is encoded as conditioning tokens;
- command-term embeddings distinguish full-body, anchor, and EE blocks;
- state and target dimensions are normalized before flow training;
- offline target-error summaries are written before each closed-loop eval;
- checkpoints still publish original-unit command tensors during eval.

Run a hand-designed-interface strong baseline:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
MODEL_SIZES="small medium large" \
SAMPLE_BUDGETS="1000 10000 50000" \
PRETRAIN_UPDATES=2000 \
FINETUNE_UPDATES=2000 \
NUM_ENVS=1 \
EVAL_STEPS=1000 \
COLLECT_STEPS=10000 \
experiments/interface_baselines/run_dance102_strong_interface_comparison.sh
```

`STEPS` remains a convenience default for both evaluation and collection.
Set `EVAL_STEPS` and `COLLECT_STEPS` separately for high-sample sweeps so a
10k oracle-drive collection does not also turn every closed-loop eval into a
10k-step eval. During achieved-state finetuning,
`USE_CHECKPOINT_NORMALIZATION=1` keeps the pretrain checkpoint's state/target
normalization buffers instead of recomputing them on rollout samples. `SEED` is
passed through oracle evaluation, sample collection, planner training, and
closed-loop evaluation, so separate result roots can be used as actual repeat
seeds rather than eval-only seeds. For held-out splits, set `TRAIN_MANIFEST`
and `EVAL_MANIFEST`; the legacy `MANIFEST` value is used for both only when
those are unset.

The strong runner uses `BATCH_SIZE=256` as the effective optimizer batch and
`MICRO_BATCH_SIZE=32` by default for gradient accumulation. This keeps the
comparison budget fixed while fitting the medium full-body trajectory planner
on A40-class GPUs. It also uses `TRAIN_ENDPOINT_STEPS=4` for the auxiliary
training endpoint loss by default while keeping `FLOW_STEPS=16` for offline
evaluation, closed-loop planner inference, and reported target metrics.

Use `MODEL_SIZE=medium` for a single-size run, or `MODEL_SIZES="small medium
large"` for an in-script model-size sweep. Use `MODEL_SIZE=tiny` only for smoke
tests. Reviewer-facing sweeps should report at least `small`, `medium`, and one
large or high-sample setting if compute allows. The intended interpretation is
not that EE/full-body should be forced through a tiny planner, but whether they
need substantially more model/data than the latent interface to remain stable
closed-loop.

The strong script writes nested result variants such as:

```text
logs/interface_baselines/dance102_strong_interface_comparison/
  interface_comparison_run_provenance.json
  ee_trajectory/chunked_transformer_medium_1000/
  full_body_trajectory/chunked_transformer_medium_1000/
```

The summarizer preserves the planner variant in both the long and wide tables.
If several sample budgets are run under one result root, the wide table reports
each variant's trainer `num_samples`, not merely the shared oracle-drive
collection size. Sample counts are tensor rows, not `sample_step_*.pt` file
counts, so multi-env oracle-drive collection is accounted for correctly.
New planner checkpoints also record source sample count, selected sample count,
held-out target-eval sample count, model preset, parameter count, effective
batch size, microbatch size, gradient-accumulation steps, reported inference
steps, auxiliary training endpoint-loss steps, and separate expert-state
pretrain vs achieved-state finetune update counts. These fields are lifted
into the wide, aggregate, and sweep-summary tables when available, so
capacity/data sweeps can be reported without reverse-engineering metadata from
directory names.
The full fair runner also writes latent `eval_pretrained_expert_state` and
`eval_finetuned_achieved_state` summaries from saved latent rollout samples, so
offline target error is available for all three interfaces.

## LAFAN1 Motion Tracking Evaluation

Use this runner when you want to run the LAFAN1 motions one by one. It can reuse
existing latent checkpoints or train the base stack first, then runs oracle
tracking, planner eval, per-motion planner finetuning, and finetuned planner
tracking.

For the full reproduction recipe and current result table, see
[`lafan1-motion-tracking-evaluation.md`](lafan1-motion-tracking-evaluation.md).

Run a smoke pass on one trajectory:

```bash
DRY_RUN=1 \
RANKS=0 \
LIMIT=1 \
RUN_BASE_PIPELINE=0 \
SKILL_CHECKPOINT=/path/to/skill_encoder.pt \
PLANNER_CHECKPOINT=/path/to/base_planner.pt \
LOW_LEVEL_CHECKPOINT=/path/to/ipmd_low_level.pt \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

Baselines are off by default. To run a command-chunk baseline, turn them on,
pick one interface, and pass the matching checkpoint. It is usually cleaner to
run `ee_trajectory` and `full_body_trajectory` as separate jobs.

```bash
RUN_HAND_DESIGNED_BASELINES=1 \
BASELINE_INTERFACES=ee_trajectory \
EE_TRAJECTORY_CHECKPOINT=/path/to/ee_ipmd.pt \
experiments/interface_baselines/run_lafan1_motion_tracking_evaluation.sh
```

The chunk length defaults to the same horizon as the latent policy.

For repeat seeds, prefer the multiseed wrapper. It runs the strong
hand-designed baseline once per seed, aggregates the per-seed roots, audits the
selected variant when the run has a single model/sample setting, and writes the
sweep summary:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/.../models/model_step_....pt \
LATENT_SKILL_CHECKPOINT=logs/hl_skill_diffsr/.../checkpoints/latest.pt \
LATENT_PLANNER_CHECKPOINT=logs/skill_commander/.../checkpoints/latest.pt \
RUN_LATENT_BASELINE=1 \
SEEDS="0 1 2" \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
SELECTED_SAMPLE_COUNT=10000 \
OUTPUT_PREFIX=logs/interface_baselines/dance102_strong_interface_comparison_10k \
LATENT_OUTPUT_PREFIX=logs/interface_baselines/dance102_latent_h10_10k_wd1e-4 \
AGGREGATE_OUTPUT_DIR=logs/interface_baselines/dance102_strong_internal_interface_comparison_10k_wd1e-4_seed0_seed1_seed2 \
MIN_ORACLE_SURVIVAL=800 \
MIN_ORACLE_SUCCESS_RATE=0.8 \
experiments/interface_baselines/run_dance102_strong_interface_multiseed.sh
```

For model/data sweeps with multiple variants, the wrapper aggregates, writes
`interface_sweep_summary.md`, writes `interface_sweep_selected.csv`, and audits
the variants chosen by that selected table. Set
`AUDIT_PLANNER_VARIANTS="ee_trajectory=... full_body_trajectory=..."` only when
you intentionally want to override the selected-table variants.
Before launching seed jobs, the wrapper runs
`preflight_interface_comparison.py` by default to check manifests, referenced
motion files, checkpoint paths, model presets, sample budgets, and planned
output roots; set `RUN_PREFLIGHT=0` only after those checks are known-good.
It also runs capacity-metadata backfill before aggregation by default, so older
latent roots can satisfy the parameter-count audit; set
`RUN_CAPACITY_BACKFILL=0` only when all roots already have current metadata.
`AUDIT_EXPECTED_SEEDS` defaults to `SEEDS` and should only be overridden when
aggregating intentionally different seed sets.
If latent runs already exist, leave `RUN_LATENT_BASELINE=0` and include them via
`EXTRA_AGGREGATE_GLOBS` or `EXTRA_AGGREGATE_ROOTS` instead.
For latent-only wrapper debugging, set `INTERFACES=` and
`RUN_LATENT_BASELINE=1`; the wrapper will skip hand-designed roots and audit
only `latent_skill`.

Alternatively, aggregate repeat seeds manually after each per-seed root has
been summarized:

```bash
pixi run python experiments/interface_baselines/aggregate_interface_comparison_seeds.py \
    --glob "logs/interface_baselines/dance102_strong_interface_comparison_10k_seed[0-9]" \
    --output_dir logs/interface_baselines/dance102_strong_interface_comparison_10k_multiseed \
    --refresh
```

This writes per-seed rows, a full mean/std CSV, and a paper-facing markdown
table under the chosen `--output_dir`.
The multiseed wrappers also write `interface_comparison_audit.json` and
`interface_comparison_audit.md` under `AGGREGATE_OUTPUT_DIR` whenever
`RUN_AUDIT=1`.
Single-seed result roots and aggregate roots also get
`interface_comparison_run_provenance.json` before launching the expensive work.
That file records key environment variables, planned result roots, git
commit/branch, dirty status, and submodule revisions.

If a result root was produced before offline target-error summaries were added,
backfill those summaries without rerunning Isaac:

```bash
pixi run python experiments/interface_baselines/backfill_offline_target_evals.py \
    --glob "logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed[0-9]_full" \
    --glob "logs/interface_baselines/dance102_strong_interface_comparison_10k_seed[0-9]"
```

Then rerun the per-root summarizer or pass `--refresh` to the aggregate script.
The paper-facing aggregate includes `pretrained_expert_target_rmse` and
`finetuned_achieved_target_rmse` when those summaries are present.

If a result root was produced before capacity metadata was added, backfill
parameter counts and variant metadata from existing checkpoints:

```bash
pixi run python experiments/interface_baselines/backfill_planner_capacity_metadata.py \
    --glob "logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed[0-9]_full" \
    --glob "logs/interface_baselines/dance102_strong_interface_comparison_10k_seed[0-9]"
```

Then rerun the summarizer or aggregate script with `--refresh`.

Before using an aggregate table in a paper draft, audit it for missing rows,
seed mismatches, sample-count mismatches, and absent offline target-error
summaries:

```bash
pixi run python experiments/interface_baselines/audit_interface_comparison.py \
    --aggregate_dir logs/interface_baselines/dance102_strong_internal_interface_comparison_10k_wd1e-4_seed0_seed1_seed2 \
    --expected_seeds 0 1 2 \
    --expected_sample_count 10000 \
    --expected_planner_num_updates 2000 \
    --expected_planner_finetune_num_updates 2000 \
    --expected_planner_batch_size 256 \
    --expected_planner_lr 1.0e-4 \
    --expected_planner_weight_decay 1.0e-4 \
    --expected_planner_flow_num_inference_steps 16 \
    --expected_planner_flow_inference_noise_std 0.0 \
    --expected_hand_designed_planner_state_history_steps 0 \
    --expected_hand_designed_planner_command_past_steps 0 \
    --expected_hand_designed_planner_command_future_steps 25 \
    --require_selected \
    --require_provenance
```

Use `--require_selected` for any paper-facing aggregate. It fails the audit if
`interface_sweep_selected.csv` is missing or if the selected rows are missing
required survival, return, tracking, planner-target, parameter-count,
sample-count, or offline target-error metrics. Detailed capacity knobs such as
microbatch size and endpoint-loss steps are reported when checkpoints contain
them, but are not required for older latent planner checkpoints.
Use `--use_selected_variants` with `--require_selected` for model/data sweeps;
the multiseed wrappers do this by default, so the final audit follows the
declared sweep-selection rule instead of requiring manual variant names.
The wrappers also forward `MIN_ORACLE_SURVIVAL` and
`MIN_ORACLE_SUCCESS_RATE` into the final audit when set.
They pass the matched optimizer/inference budget into the final audit as well:
`FINETUNE_UPDATES`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, `FLOW_STEPS`, and
`FLOW_NOISE_STD`. For hand-designed interfaces, they also audit the planner
input/window contract via `STATE_HISTORY_STEPS`, `COMMAND_PAST_STEPS`, and
`COMMAND_FUTURE_STEPS`; latent rows are skipped for that specific gate because
older latent summaries may not store the history count explicitly.
Set `AUDIT_EXPECTED_PRETRAIN_UPDATES` only for runs where every selected
interface should have the same expert-state pretrain count.
They persist the audit report as `interface_comparison_audit.json` and
`interface_comparison_audit.md` inside the aggregate directory.
The matching `interface_comparison_run_provenance.json` captures the run setup
needed to interpret those audit results later. The multiseed wrappers pass
`--require_provenance` by default, so the final audit also verifies the
aggregate provenance JSON and every selected per-seed result root provenance
JSON before the table is considered paper-facing.
For held-out or cross-motion tables, also gate the comparison on oracle
low-level competence. For example, add `--min_oracle_survival 800` and
`--min_oracle_success_rate 0.8` when a row should only be considered
paper-facing if the corresponding low-level policy can already execute the
motion under perfect commands. This prevents a high finetuned/oracle ratio from
making an interface look competitive when the oracle ceiling itself is weak.

For model/data sweeps, generate a best-variant summary after aggregation. The
default selection metric is `finetuned_survival_oracle_ratio`, with survival,
return, and tracking-error tie-breakers:

```bash
pixi run python experiments/interface_baselines/analyze_interface_sweep.py \
    --aggregate_dir logs/interface_baselines/dance102_strong_internal_interface_comparison_10k_wd1e-4_seed0_seed1_seed2
```

This writes `interface_sweep_summary.csv` and `interface_sweep_summary.md`.
Use it when reporting multiple `MODEL_SIZES` or `SAMPLE_BUDGETS` settings so
the paper table selects the strongest hand-designed internal baseline by a
declared rule rather than by manual inspection. The analyzer also writes
`interface_sweep_selected.csv` and `interface_sweep_selected.md`; those selected
files contain the latent reference plus the top-ranked hand-designed variant for
each interface and are the intended source for paper-facing best-baseline rows.
They include oracle, pretrained, and finetuned done rates,
success/reference-end survival rates, survival steps, return, tracking,
action-smoothness, and planner-target metrics, plus offline target-error,
capacity, sample-count, and pretrain/finetune update-count metrics.
Set `SELECTED_SAMPLE_COUNT=10000` in the multiseed wrappers, or pass
`--selected_sample_count 10000` directly to the analyzer, when the paper table
must be restricted to the matched 10k finetune budget. Leave it unset when the
intended diagnostic is the strongest available hand-designed baseline across a
data/model sweep. The multiseed wrappers also pass
`--expected_selected_interfaces`, so the analyzer fails rather than writing an
incomplete selected table if the requested sample-count filter drops an
interface or if a selected row is missing required paper metrics. Selected rows
are ordered as `latent_skill`, `ee_trajectory`, then `full_body_trajectory` to
keep the learned-interface reference visually separate from the hand-designed
baselines. The multiseed wrappers run sweep analysis before the final audit and
pass `--require_selected`, so the selected table is also checked by
`audit_interface_comparison.py`.

## Current Dance102 Results

The current reviewer-facing comparison is a strong internal VLA-style baseline,
not an external OpenPI/GR00T integration. It uses the same simulator, dataset,
frozen low-level policy protocol, achieved-state inputs, 10k oracle-drive
samples, 2000 achieved-state finetune updates, batch size 256, and 1000-step
closed-loop evals across all interfaces. The hand-designed interfaces use
`MODEL_SIZE=medium` chunked Transformer planners with 2000 expert-state
pretrain updates before the matched achieved-state finetune. For the medium
full-body planner, this batch size is usually implemented with microbatch
gradient accumulation to avoid changing the effective update budget. Training
may use fewer endpoint-loss integration steps than evaluation; the reported
planner still uses the configured 16-step flow inference path.
The wrappers also pass `FINETUNE_WEIGHT_DECAY=${WEIGHT_DECAY}` into latent
rollout finetuning, so the learned-latent and hand-designed interfaces share
the same optimizer regularization in new runs.

The audited h10 10k per-seed runs currently live under:

```text
logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed0_full/
logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed1_full/
logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed2_full/
logs/interface_baselines/dance102_strong_interface_comparison_h10_10k_seed0/
logs/interface_baselines/dance102_strong_interface_comparison_h10_10k_seed1/
logs/interface_baselines/dance102_strong_interface_comparison_h10_10k_seed2/
```

The combined three-seed aggregate is:

```text
logs/interface_baselines/dance102_strong_internal_interface_comparison_h10_10k_wd1e-4_seed0_seed1_seed2/
```

The latent oracle and pretrained closed-loop summaries are unchanged from the
earlier 10k latent roots; the matched aggregate above reruns the latent
achieved-state finetune and finetuned closed-loop eval with
`weight_decay=1.0e-4` so the optimizer budget matches the hand-designed
baselines.
Those reused latent roots record 5000 expert-state planner pretrain updates,
while the current hand-designed strong baseline uses 2000 expert-state
pretrain updates. The controlled budget claim is therefore the matched 2000
achieved-state finetune budget, not a fully matched pretrain budget. If a paper
table needs pretrain parity, rerun either the hand-designed planners with
`PRETRAIN_UPDATES=5000` or the latent planner with 2000 pretrain updates and
report that table separately.

History-matched h10 three-seed aggregate (`STATE_HISTORY_STEPS=10`,
`COMMAND_PAST_STEPS=0`, `COMMAND_FUTURE_STEPS=25`; audit passed 213 checks,
0 failed):

| Interface | Input dim | Output dim | Planner params | Samples | Oracle survival | Pretrained survival | Finetuned survival | Finetuned/oracle survival | Offline expert RMSE | Offline achieved RMSE | Closed-loop target RMSE | Finetuned return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `latent_skill` | 670 | 256 | 1.94M | 10000 | 1000 | 1000 | 1000 | 1.000 | 0.6507 | 0.1219 | 0.1341 | 73.37 |
| `ee_trajectory` | 737 | 936 | 22.59M | 10000 | 1000 | 109.67 +/- 41.02 | 1000 | 1.000 | 0.0218 | 0.0146 | 0.0158 | 86.15 |
| `full_body_trajectory` | 737 | 1742 | 22.60M | 10000 | 1000 | 77.00 +/- 55.49 | 968.00 +/- 55.43 | 0.968 +/- 0.055 | 0.2364 | 0.1200 | 0.1258 | 80.97 |

The h10 result changes the strongest story. EE and full-body trajectory
interfaces are expressive under oracle commands: all oracle rows reach
1000-step survival. They are also not intrinsically unworkable planner
interfaces: after matched achieved-state finetune, EE trajectory reaches full
1000-step survival across all three seeds, and full-body trajectory reaches
968 average survival with two of three seeds succeeding to the end.

The remaining planner-burden signal is distribution shift and interface
complexity. Before achieved-state finetune, the hand-designed planners collapse
closed-loop despite low offline target RMSE (`ee_trajectory` averages 109.7
survival, `full_body_trajectory` averages 77.0). The latent planner survives
1000 steps even before this rollout finetune, but its return is low
(`16.96`) and its closed-loop target RMSE is high (`1.70`), so survival alone
should not be reported as "good pretrained latent tracking." After finetune,
the hand-designed planners can catch up or nearly catch up, but they require
larger output spaces and roughly 22.6M-parameter Transformer planners versus
the 256-dimensional, 1.94M-parameter latent planner.

No-history three-seed aggregate (`STATE_HISTORY_STEPS=0`; diagnostic, not the
primary paper table):

| Interface | Input dim | Output dim | Planner params | Samples | Oracle survival | Pretrained survival | Finetuned survival | Finetuned/oracle survival | Offline expert RMSE | Offline achieved RMSE | Closed-loop target RMSE | Finetuned return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ee_trajectory` | 67 | 936 | 21.21M | 10000 | 1000 | 70.00 +/- 8.89 | 390.33 +/- 97.33 | 0.3903 +/- 0.0973 | 0.0241 +/- 0.0015 | 0.0177 +/- 0.0008 | 0.1626 +/- 0.0614 | 28.58 +/- 6.74 |
| `full_body_trajectory` | 67 | 1742 | 21.23M | 10000 | 1000 | 84.00 +/- 19.52 | 362.67 +/- 102.71 | 0.3627 +/- 0.1027 | 0.2741 +/- 0.0134 | 0.1600 +/- 0.0069 | 0.9448 +/- 0.1070 | 21.95 +/- 5.56 |
| `latent_skill` | 670 | 256 | 1.94M | 10000 | 1000 | 1000 | 1000 | 1.0000 +/- 0.0000 | 0.6507 +/- 0.0739 | 0.1219 +/- 0.0034 | 0.1341 +/- 0.0067 | 73.37 +/- 2.12 |

The no-history repeat strengthens the interface-burden story but should not be
used as the main fair table. The hand-designed
interfaces are expressive under oracle commands: both have 1000-step oracle
survival. The failure is planner-facing. Even with a stronger chunked
Transformer planner with about 21M parameters and the same achieved-state
finetune budget, EE and full-body commands recover only about 39% and 36% of
oracle survival, respectively. The latent planner is about 1.94M parameters
and recovers full survival. The offline oracle-drive target errors are useful
diagnostics but not sufficient: EE can fit oracle-drive samples with low RMSE
and still fail under planner-driven closed-loop distribution shift. The latent
interface has the smallest command dimension and recovers full 1000-step
survival in all three seeds.

The no-history aggregate above uses the current achieved macro state
(`STATE_HISTORY_STEPS=0`, input dim 67), while the latent SkillCommander
checkpoint uses a history-conditioned input (state dim 670). The summary and
audit scripts now surface planner input dimension and history/window metadata
so this knob is visible in paper tables. The live history-matched EE/full-body
rerun below is the clean next stress test for removing this as a reviewer
objection.

The same caveat applies to pretrain steps: the primary fair knob currently
matched across interfaces is achieved-state finetune, while expert-state
pretrain count remains visible as a reported metric.

Reproduce the history-matched aggregate with a distinct output prefix so the
no-history diagnostic above remains reproducible:

```bash
RUN_LATENT_BASELINE=0 \
EXTRA_AGGREGATE_GLOBS="logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed[0-9]_full" \
SEEDS="0 1 2" \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
SELECTED_SAMPLE_COUNT=10000 \
STATE_HISTORY_STEPS=10 \
COMMAND_PAST_STEPS=0 \
COMMAND_FUTURE_STEPS=25 \
OUTPUT_PREFIX=logs/interface_baselines/dance102_strong_interface_comparison_h10_10k \
AGGREGATE_OUTPUT_DIR=logs/interface_baselines/dance102_strong_internal_interface_comparison_h10_10k_wd1e-4_seed0_seed1_seed2 \
MIN_ORACLE_SURVIVAL=800 \
MIN_ORACLE_SUCCESS_RATE=0.8 \
experiments/interface_baselines/run_dance102_strong_interface_multiseed.sh
```

The multiseed wrapper forwards `STATE_HISTORY_STEPS`,
`COMMAND_PAST_STEPS`, and `COMMAND_FUTURE_STEPS` into each per-seed
hand-designed run, and the selected/audit tables report the resulting planner
input metadata.

This does not claim that an external VLA stack could not do better with more
engineering, larger models, or different data. The intended claim is narrower
and cleaner: under a controlled internal stack, optimizing the high-low
interface as a learned skill latent produces a much easier planner target than
hand-designed full-state or end-effector trajectory interfaces for expressive
Dance102 motion.

## Paper Framing

Use these results as a controlled interface-ablation claim, not as an external
foundation-model comparison. A defensible wording is:

> Given the same low-level-control stack, achieved-state planner inputs,
> oracle-drive finetune data budget, and closed-loop evaluation protocol, a
> learned skill latent provides a compact high-level action interface that is
> less brittle before achieved-state finetune, while strong hand-designed
> trajectory planners can recover after matched achieved-state finetuning at
> substantially larger output/model scale.

Important limitations to keep visible:

- This is a no-language, single-motion Dance102 diagnostic.
- The hand-designed baselines use stronger chunked Transformer planners, while
  the latent row uses the existing SkillCommander route; this is conservative
  for the latent claim but not an architecture-matched VLA reproduction.
- The current reused latent rows have 5000 expert-state pretrain updates,
  whereas the hand-designed rows use 2000 unless `PRETRAIN_UPDATES=5000` is set.
  The achieved-state finetune budget is matched.
- The oracle low-level ceilings show that EE and full-body commands are
  expressive enough; the hard part is high-level generation under closed-loop
  achieved-state inputs, especially before achieved-state finetune. The h10
  aggregate shows that strong achieved-state finetune can recover full survival
  for EE trajectory and near-full survival for full-body trajectory, so avoid
  saying they are intrinsically unworkable.
- External OpenPI/GR00T comparisons should be described as future work unless
  the project takes on their data format, tokenizer/action head, robot adapter,
  and evaluation engineering.

### Reviewer-Facing Baseline Bar

For the paper, treat OpenPI/GR00T-style systems as motivation for the command
space, not as the baseline implementation. The strong internal baseline is
acceptable only if it gives the hand-designed interfaces a serious chance
inside this repo's controlled stack:

- Use the history-matched setting (`STATE_HISTORY_STEPS=10`) for the primary
  Dance102 table, because the latent planner is history conditioned.
- Use chunked Transformer planners for `ee_trajectory` and
  `full_body_trajectory`; the small MLP planner is diagnostic only.
- Require oracle competence before interpreting planner results
  (`MIN_ORACLE_SURVIVAL` and `MIN_ORACLE_SUCCESS_RATE`).
- Match the achieved-state finetune budget, optimizer settings, flow inference
  settings, sample count, and closed-loop evaluation horizon across interfaces.
- Report planner input dimension, output dimension, parameter count, pretrain
  updates, achieved-state finetune updates, sample count, and target errors so
  the interface burden is visible instead of hidden in a return number.
- Include at least one capacity or data-scaling point for the hand-designed
  interfaces if compute allows. If a larger/high-sample hand-designed planner
  closes the survival gap, report that honestly and shift the claim toward
  compactness, sample efficiency, and reduced achieved-state finetune burden.

Next experiments, in priority order:

1. Run the multi-motion held-out split now that the Dance102 machinery is
   stable.
2. Run at least one larger or higher-sample hand-designed planner setting if
   compute allows, to check that the gap is not a medium-model artifact.
3. Add latent robustness diagnostics only if cheap: latent update interval and
   latent command noise.

## Held-Out Split

For the full local G1 LAFAN1 set, use the wrapper. It creates train/eval
manifests from `data/lafan1/manifests/g1_lafan1_manifest.json` when
`TRAIN_MANIFEST` and `EVAL_MANIFEST` are not already set, then launches the
strong internal hand-designed baseline:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
HELDOUT_FRACTION=0.2 \
SEED=0 \
MODEL_SIZES="small medium large" \
SAMPLE_BUDGETS=10000 \
PRETRAIN_UPDATES=2000 \
FINETUNE_UPDATES=2000 \
NUM_ENVS=1 \
EVAL_STEPS=1000 \
COLLECT_STEPS=10000 \
experiments/interface_baselines/run_lafan1_heldout_strong_interface_comparison.sh
```

If `TRAIN_MANIFEST` and `EVAL_MANIFEST` are both already set, the wrapper uses
those split manifests directly and does not require `FULL_MANIFEST`.

For repeat held-out seeds, use the LAFAN1 multiseed wrapper. It creates one
split and result root per seed, runs the same strong hand-designed baseline,
then backfills capacity metadata, aggregates, audits, and writes a sweep
summary. Set `RUN_LATENT_BASELINE=1` only when the latent low-level, skill, and
planner checkpoints were trained for the same train split or a strictly
comparable multi-motion training set. The wrapper passes `LATENT_MOTION_NAME=`
by default for held-out runs, so latent eval uses the split manifest instead of
forcing the Dance102 motion filter:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
LATENT_LOW_LEVEL_CHECKPOINT=logs/rlopt/ipmd_bilinear/Isaac-Imitation-G1-Latent-v0/.../models/model_step_....pt \
LATENT_SKILL_CHECKPOINT=logs/hl_skill_diffsr/.../checkpoints/latest.pt \
LATENT_PLANNER_CHECKPOINT=logs/skill_commander_lafan1_train/.../checkpoints/latest.pt \
LATENT_DATASET_PATH=data/lafan1/g1_lafan1_hl_diffsr_train \
RUN_LATENT_BASELINE=1 \
SEEDS="0 1 2" \
HELDOUT_FRACTION=0.2 \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
SELECTED_SAMPLE_COUNT=10000 \
PRETRAIN_UPDATES=2000 \
FINETUNE_UPDATES=2000 \
NUM_ENVS=1 \
EVAL_STEPS=1000 \
COLLECT_STEPS=10000 \
MIN_ORACLE_SURVIVAL=800 \
MIN_ORACLE_SUCCESS_RATE=0.8 \
experiments/interface_baselines/run_lafan1_heldout_strong_interface_multiseed.sh
```

`LATENT_DATASET_PATH` is required in latent held-out mode to avoid accidentally
using the Dance102 latent dataset default. With `RUN_LATENT_BASELINE=0`, the
wrapper remains hand-designed-only:
`ee_trajectory` and `full_body_trajectory`. With `RUN_LATENT_BASELINE=1`, it
adds `latent_skill` to the same per-seed roots and aggregate audit. Do not use
Dance102-only latent checkpoints for the held-out table; that would test
cross-dataset checkpoint mismatch rather than the interface.
Set `MIN_ORACLE_SURVIVAL` and `MIN_ORACLE_SUCCESS_RATE` for paper-facing
held-out runs so weak oracle low-level ceilings fail the aggregate audit.
For non-dry-run launches, the single-seed held-out wrapper runs
`preflight_interface_comparison.py` after creating the split manifests. It
checks train/eval manifests, referenced motion files, checkpoints, model/sample
settings, and the required latent dataset path before launching Isaac.

### Cluster Submission

Use the cluster submit wrapper when the full LAFAN1 data should be prepared on
the cluster instead of locally:

Use `MODE=dance102-strong-multiseed` for the full Dance102 multiseed wrapper,
`MODE=lafan1-heldout-multiseed` for split LAFAN1 held-out sweeps, and
`MODE=multimotion-heldout` when `TRAIN_MANIFEST` and `EVAL_MANIFEST` are
already prepared.

For the Dance102 history-matched reviewer check, use the same local command
recipe as above but submit it through the cluster wrapper:

```bash
MODE=dance102-strong-multiseed \
RUN_LATENT_BASELINE=0 \
EXTRA_AGGREGATE_GLOBS="logs/interface_baselines/dance102_latent_h10_10k_wd1e-4_seed[0-9]_full" \
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-06-23_09-37-39/models/model_step_600047616.pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/2026-06-23_09-39-48/models/model_step_650084352.pt \
SEEDS="0 1 2" \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
SELECTED_SAMPLE_COUNT=10000 \
STATE_HISTORY_STEPS=10 \
COMMAND_PAST_STEPS=0 \
COMMAND_FUTURE_STEPS=25 \
OUTPUT_PREFIX=logs/interface_baselines/dance102_strong_interface_comparison_h10_10k \
AGGREGATE_OUTPUT_DIR=logs/interface_baselines/dance102_strong_internal_interface_comparison_h10_10k_wd1e-4_seed0_seed1_seed2 \
MIN_ORACLE_SURVIVAL=800 \
MIN_ORACLE_SUCCESS_RATE=0.8 \
experiments/interface_baselines/submit_cluster_interface_baselines.sh
```

Use `DRY_RUN=1` first. The submitter dry run should print the cluster
invocation with `--env STATE_HISTORY_STEPS=10`, `--env COMMAND_PAST_STEPS=0`,
`--env COMMAND_FUTURE_STEPS=25`, and any auto-generated
`CLUSTER_EXTRA_RSYNC_SPECS`. To inspect the actual per-seed child commands
without submitting a cluster job, run
`experiments/interface_baselines/run_dance102_strong_interface_multiseed.sh`
locally with the same env vars and `DRY_RUN=1`; that dry run should print the
per-seed `STATE_HISTORY_STEPS=10` child commands and the final audit flags
`--expected_hand_designed_planner_state_history_steps 10`,
`--expected_hand_designed_planner_command_past_steps 0`, and
`--expected_hand_designed_planner_command_future_steps 25`.
During a live cluster job, Slurm stdout is in the submitted snapshot directory
such as `.../isaaclab_YYYYMMDD_HHMMSS/output_<jobid>.log`, but generated
result roots are written through the container bind mount to the shared
`${CLUSTER_ISAACLAB_DIR}/logs` tree. With the default cluster config, that means
monitor result tables under
`$HOME/scratch/Research/IsaacLab/isaaclab/logs/interface_baselines/...`, not
only under the submitted snapshot directory.
The submitter forwards `EXTRA_AGGREGATE_GLOBS` and `EXTRA_AGGREGATE_ROOTS`.
If matching local result directories exist, it also adds them to
`CLUSTER_EXTRA_RSYNC_SPECS` and syncs those specific artifacts even though the
normal repo sync excludes generated `logs/`. If a path has no local match, the
submitter leaves it unchanged so the job can still use a cluster-side result
root. Set `AUTO_SYNC_EXTRA_AGGREGATE_ROOTS=0` to disable this artifact sync, or
set `RUN_LATENT_BASELINE=1` with the latent checkpoint/dataset env vars when
the job should recreate the latent rows before aggregation.

```bash
MODE=lafan1-heldout-multiseed \
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
RUN_LATENT_BASELINE=0 \
SEEDS="0 1 2" \
HELDOUT_FRACTION=0.2 \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
SELECTED_SAMPLE_COUNT=10000 \
COLLECT_STEPS=10000 \
EVAL_STEPS=1000 \
MIN_ORACLE_SURVIVAL=800 \
MIN_ORACLE_SUCCESS_RATE=0.8 \
experiments/interface_baselines/submit_cluster_interface_baselines.sh
```

For `MODE=lafan1-*`, the submitter disables the default job-argument manifest
append and clears `CLUSTER_G1_MANIFEST_PATH` before submission. The cluster
runtime then uses `${CLUSTER_G1_DATA_ROOT:-${CLUSTER_DATA_DIR}/lafan1}` and
`${CLUSTER_G1_DATA_ROOT}/manifests/g1_lafan1_manifest.json`, downloading or
refreshing the full G1 LAFAN1 NPZ tree according to
`CLUSTER_G1_MANIFEST_REFRESH_POLICY=auto`. Inside the container, the LAFAN1
wrappers default `FULL_MANIFEST` to
`${CLUSTER_DATA_DIR}/lafan1/manifests/g1_lafan1_manifest.json`, so the
preflight-generated cluster manifest is used without copying it into the repo
workspace. Held-out split manifests are written beside that full manifest by
default, under `${CLUSTER_DATA_DIR}/lafan1/manifests/interface_baselines`, so
their relative NPZ paths still resolve to the cluster dataset tree. Use
`DRY_RUN=1` to print the
submission command without contacting the cluster. The interface-baseline
submitter defaults to `CLUSTER_GIT_SYNC_FIRST=0`, so source sync uses the
guarded rsync path instead of a git clone plus local-patch replay; set
`CLUSTER_GIT_SYNC_FIRST=1` only when you specifically want git-first sync. It
also excludes `IsaacLab/`, `RLOpt/`, and `ImitationLearningTools/` from the
top-level rsync by default via `CLUSTER_EXTRA_RSYNC_EXCLUDES`; RLOpt and
ImitationLearningTools are synced separately from the current worktree
submodules. IsaacLab is hardlinked from the previous cluster snapshot by
default via `CLUSTER_LINK_ISAACLAB_FROM_PREVIOUS=1`, avoiding a fresh dependency
tree copy on every dirty-worktree submission. Disable that flag only if the
previous snapshot lacks a valid IsaacLab tree or you explicitly sync IsaacLab
through `CLUSTER_ISAACLAB_LOCAL_PATH`. The submitter also defaults to
`CLUSTER_SKIP_CACHE_COPY=1`, which keeps short interface eval/collection jobs
from spending startup time copying the full Isaac Sim cache into per-job
temporary storage. `CLUSTER_OVERLAY_SIZE_MB` defaults to `8192` for these jobs;
raise it only if container writes exceed that overlay budget. The submitter also
sets `CLUSTER_USE_SHARED_SIF=1` by default, so after the first extraction the
cluster can reuse a shared container image instead of untarring the SIF into
each job's temporary directory. The cluster launcher also sets
`INTERFACE_BASELINE_PYTHON_CMD=/isaac-sim/python.sh` inside the container; local
shell workflows still default to `pixi run python`. It also redirects generated
LAFAN1 Zarr datasets to
`${CLUSTER_DATA_DIR}/lafan1/zarr_cache` through
`ISAACLAB_IMITATION_LAFAN1_ZARR_CACHE_ROOT`, and generated Unitree URDF-to-USD
artifacts to a per-job
`${CLUSTER_DATA_DIR}/isaaclab_imitation/unitree_usd_cache/<job-id>` directory
through `ISAACLAB_IMITATION_UNITREE_USD_CACHE_ROOT`. These redirects keep
loader caches and IsaacLab asset-converter outputs off the small container
overlay.
The submitter also auto-syncs parent directories for local checkpoint files
passed through checkpoint env vars such as `FULL_BODY_TRAJECTORY_CHECKPOINT`,
`EE_TRAJECTORY_CHECKPOINT`, and `LATENT_PLANNER_CHECKPOINT`. These directories
are synced as rsync-only artifacts through `CLUSTER_EXTRA_RSYNC_SPECS`, not as
extra git repositories. Repo-local absolute paths are rewritten to repo-relative
paths on the cluster; external local paths are copied under
`external/interface_baseline_checkpoints/<ENV_VAR>`.
Set `AUTO_SYNC_LOCAL_CHECKPOINTS=0` if you are passing paths that already exist
on the cluster.
It also points `CLUSTER_RLOPT_LOCAL_PATH` and
`CLUSTER_IMITATION_TOOLS_LOCAL_PATH` at the current worktree submodules by
default, avoiding mixed-branch jobs where top-level experiment scripts come
from this worktree but dependency overlays come from the main checkout.

The full G1 LAFAN1 manifest is local data and is not shipped in git. If the
wrapper reports that the manifest is missing, prepare it with:

```bash
./scripts/download_g1_lafan1_data.sh
```

or, if NPZ files already exist:

```bash
pixi run python scripts/write_lafan1_npz_manifest.py \
    --npz_dir data/lafan1/npz/g1 \
    --manifest_path data/lafan1/manifests/g1_lafan1_manifest.json
```

To create train/eval manifests manually from any multi-motion LAFAN1-style
manifest:

```bash
pixi run python experiments/interface_baselines/split_lafan1_manifest.py \
    --manifest data/unitree/manifests/g1_multi_motion_manifest.json \
    --heldout_fraction 0.2 \
    --seed 0 \
    --output_dir data/unitree/manifests \
    --prefix g1_multi_motion_seed0
```

Use `--heldout_names` for named held-out motions, or `--heldout_patterns`
for groups such as `walk*`. The splitter writes
`g1_multi_motion_seed0_train.json` and
`g1_multi_motion_seed0_heldout.json`.

The strong hand-designed baseline supports separate train and eval manifests.
Use `TRAIN_MANIFEST` for oracle-drive sample collection and planner
pretrain/finetune, and `EVAL_MANIFEST` for oracle, pretrained, and finetuned
closed-loop evaluation:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
EE_TRAJECTORY_CHECKPOINT=logs/rlopt/ipmd/Isaac-Imitation-G1-v0/.../models/model_step_....pt \
TRAIN_MANIFEST=data/unitree/manifests/g1_train_motions_manifest.json \
EVAL_MANIFEST=data/unitree/manifests/g1_heldout_motions_manifest.json \
MODEL_SIZES="small medium large" \
SAMPLE_BUDGETS=10000 \
PRETRAIN_UPDATES=2000 \
FINETUNE_UPDATES=2000 \
NUM_ENVS=1 \
EVAL_STEPS=1000 \
COLLECT_STEPS=10000 \
experiments/interface_baselines/run_multimotion_heldout_interface_comparison.sh
```

This wrapper intentionally covers the strong hand-designed interfaces first,
because they already consume arbitrary train/eval manifests. The latent route
still uses the existing SkillCommander single-motion command path in this
worktree; add a multi-motion latent commander route before treating held-out
latent numbers as paper-comparable.

## Fast Checks

Pure Python:

```bash
pixi run python experiments/interface_baselines/smoke_test_interface_planner.py
```

Dry-run the orchestration:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=/tmp/full_body.pt \
EE_TRAJECTORY_CHECKPOINT=/tmp/ee.pt \
LATENT_LOW_LEVEL_CHECKPOINT=/tmp/latent_low.pt \
LATENT_SKILL_CHECKPOINT=/tmp/skill.pt \
LATENT_PLANNER_CHECKPOINT=/tmp/planner.pt \
DRY_RUN=1 \
experiments/interface_baselines/run_dance102_fair_interface_comparison.sh
```

Dry-run the strong planner orchestration:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=/tmp/full_body.pt \
EE_TRAJECTORY_CHECKPOINT=/tmp/ee.pt \
DRY_RUN=1 \
MODEL_SIZE=tiny \
SAMPLE_BUDGETS=1000 \
experiments/interface_baselines/run_dance102_strong_interface_comparison.sh
```

Dry-run the multiseed wrapper:

```bash
FULL_BODY_TRAJECTORY_CHECKPOINT=/tmp/full_body.pt \
EE_TRAJECTORY_CHECKPOINT=/tmp/ee.pt \
LATENT_LOW_LEVEL_CHECKPOINT=/tmp/latent_low.pt \
LATENT_SKILL_CHECKPOINT=/tmp/skill.pt \
LATENT_PLANNER_CHECKPOINT=/tmp/planner.pt \
DRY_RUN=1 \
RUN_LATENT_BASELINE=1 \
SEEDS="0 1" \
MODEL_SIZE=medium \
SAMPLE_BUDGETS=10000 \
OUTPUT_PREFIX=/tmp/dance102_strong_10k \
LATENT_OUTPUT_PREFIX=/tmp/dance102_latent_10k \
AGGREGATE_OUTPUT_DIR=/tmp/dance102_strong_10k_multiseed \
experiments/interface_baselines/run_dance102_strong_interface_multiseed.sh
```

Isaac smoke commands should use tiny settings first:

```bash
RUN_LATENT=0 \
FULL_BODY_TRAJECTORY_CHECKPOINT=/path/to/full_body_checkpoint.pt \
EE_TRAJECTORY_CHECKPOINT=/path/to/ee_checkpoint.pt \
NUM_ENVS=1 \
STEPS=100 \
FINETUNE_UPDATES=10 \
experiments/interface_baselines/run_dance102_fair_interface_comparison.sh
```

For the strong planner, reuse existing collected samples when possible:

```bash
RUN_ORACLE=0 \
FORCE_COLLECT=0 \
MODEL_SIZE=tiny \
SAMPLE_BUDGETS=1000 \
PRETRAIN_UPDATES=2 \
FINETUNE_UPDATES=2 \
EVAL_STEPS=100 \
COLLECT_STEPS=100 \
experiments/interface_baselines/run_dance102_strong_interface_comparison.sh
```

Do not commit generated logs, checkpoints, videos, or sample tensors.
