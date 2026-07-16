# BONES-SEED Phase 5 Data Preparation

Last checked: 2026-07-16.

This page records the data gate for the shared language-conditioned Phase 5
experiment. Both the current 10-motion and 100-motion sets pass the body-frame,
body-name, language-goal, and temporal-event checks. A fresh, provenance-complete
100-motion paper tree was generated and copied to persistent Skynet storage on
2026-07-14. The vision/task benchmark and shared planner protocol remain in
`wiki/causal-interface-paper-plan.md`.

## Fresh 100-Motion Paper Tree

The final fresh export completed on 2026-07-14 using:

```bash
pixi run -e isaaclab python scripts/prepare_bones_seed_phase5.py \
    --csv-dir data/bones_seed/raw/g1_100 \
    --language-sidecar data/bones_seed_100/language/g1_bones_seed_100_language.json \
    --output-root data/bones_seed_phase5_corrected/bones_seed_100 \
    --dataset-name bones_seed_100_phase5 \
    --device cuda:0 \
    --require-temporal-events
```

The preparation record is `complete`, and the strict preflight passes 100/100
motions with body names, language goals, and temporal events. The existing
MiniLM table matches the fresh manifest in exact motion order and has shape
`100 x 384`.

The fresh tree was also compared with the previously corrected 100-motion
tree using `scripts/compare_bones_seed_exports.py`. All 100 NPZs are
byte-identical, the name order is identical, and both name-keyed aggregate
hashes are
`2f373adfd14797c6a3708a4b430344118a3d6f9b0c90e6bbc0fb0d80b4075e74`.
This independently recovers the missing NPZ provenance while preserving the
previously validated motion data.

Important identifiers:

| Artifact | SHA-256 |
| --- | --- |
| Fresh manifest | `fd285a287d98a8478574da211b7dbf1cf8fbfca974ecf9ba62c200e4a3b87b97` |
| Preparation record | `53dfcb3718f758edbf81b817066f4573548aa2a214ed17642162c29b6169bd37` |
| MiniLM goal table | `3a50746d575d3c8d36c2c4e460acf4834a22a74e663a27d9f04ac8a6137c7975` |

Local paths are under
`data/bones_seed_phase5_corrected/bones_seed_100/`. The checksum-identical
persistent Skynet copy is mounted inside jobs at:

```text
/data/bones_seed_phase5/bones_seed_100
```

Use these cluster paths for final runs:

```text
MANIFEST=/data/bones_seed_phase5/bones_seed_100/manifests/g1_bones_seed_100_phase5_manifest.json
PREPARATION_RECORD=/data/bones_seed_phase5/bones_seed_100/preparation/preparation.json
LANGUAGE_EMBEDDINGS=/data/bones_seed_phase5/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt
LATENT_DATASET_PATH=/data/bones_seed_phase5/bones_seed_100/zarr/latent_seed0
VANILLA_DATASET_PATH=/data/bones_seed_phase5/bones_seed_100/zarr/vanilla_seed0
```

An `rsync --checksum --delete --dry-run` comparison produced no differences
after transfer. The host-side persistent path is
`/nethome/fwu91/scratch/Research/IsaacLab/data/bones_seed_phase5/bones_seed_100`.

## Language Planner Code Gates

As of 2026-07-14, the current corrected 10-motion set passes a real simulator
smoke for both main planner outputs. The shared planner receives a ten-frame
causal robot history (`10 x 93` values) and one 384-D MiniLM goal embedding.
Only its output changes: 256 values for DiffSR or 670 values for the exact
ten-frame vanilla packet. Each row collected two language-labeled samples,
took one tiny-model update, ran offline inference, and completed a twenty-step
closed-loop call for `Neutral_kick_trash_001_A057`.

The code gate is:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=/path/to/skill_encoder.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla.pt \
experiments/interface_baselines/run_bones_seed_language_smoke.sh
```

The runner performs the body-name/event preflight, writes a separate
one-motion evaluation manifest, collects both interfaces, trains the same tiny
backbone for one update, evaluates both with the explicit goal, and runs
`audit_bones_seed_language_interface.py`. Goal selection never reads the
expert cursor. This is only a code check and does not replace the fresh export
or the final shared multi-goal Skynet study.

The next integration gate passed on two goals with the reusable
`run_bones_seed_multigoal_language_comparison.py` runner. It used one shared
planner per interface across both goals, balanced demonstration and
planner-rollout rows per goal, and explicit matching goal/motion selection for
every closed-loop call. The final
`audit_bones_seed_multigoal_language_comparison.py` report passed. This smoke
used one row per goal, one update per stage, and ten evaluation steps, so its
tracking scores are not research results.

The paper rollout stage keeps one goal per array task but collects that goal's
rows with ten parallel environments. Each environment receives the same
explicit language goal and the same one-motion reference restriction. Exact
row selection prevents a partial final batch from exceeding the 1,000-row
budget, and any goal/reference mismatch is a hard failure. Closed-loop results
before and after planner-driven training still use one environment per goal.

For the final workflow, use the shell wrapper with the freshly prepared
manifest and matching language table:

```bash
LATENT_LOW_LEVEL_CHECKPOINT=/path/to/latent.pt \
LATENT_SKILL_CHECKPOINT=/path/to/skill_encoder.pt \
VANILLA_TRACKER_CHECKPOINT=/path/to/vanilla.pt \
MANIFEST=/path/to/fresh/bones_seed_manifest.json \
LANGUAGE_EMBEDDINGS=/path/to/matching_goal_embeddings.pt \
LATENT_DATASET_PATH=/path/to/new/latent_cache \
VANILLA_DATASET_PATH=/path/to/new/vanilla_cache \
PREPARATION_RECORD=/path/to/fresh/preparation/preparation.json \
VANILLA_QUALIFICATION_AUDIT=/path/to/vanilla_qualification.json \
LATENT_QUALIFICATION_AUDIT=/path/to/latent_qualification.json \
STREAMED_EQUIVALENCE_CERTIFICATE=/path/to/equivalence.json \
OUTPUT_ROOT=logs/interface_baselines/bones_seed_multigoal_seed0 \
SEED=0 \
experiments/interface_baselines/run_bones_seed_multigoal_language_comparison.sh
```

Use Skynet for the complete motion set and paper seeds. Local work should only
prove that the code follows the protocol; it should not be scaled into a
second long training campaign.

For the final three planner seeds, use
`submit_bones_seed_multiseed_pipeline_skynet.sh` rather than invoking the raw
runner manually. It fixes seeds `0 1 2`, validates all three before the first
submission, refuses stale output roots, and preserves the workspace archive
hash and five-stage Slurm job chain in each run's `cluster_submission.json`.

The cluster wrapper treats those three low-level reports and the preparation
record as hard gates. It verifies their file and checkpoint hashes, requires
both oracle success rates to be at least 0.8, and rejects an equivalence
certificate that misses a packet phase, asynchronous renewal, or tracker-state
immutability. A dry run with the current diagnostic checkpoints is expected to
fail these gates; that is the intended block before final Skynet planner jobs.

On 2026-07-15 the exact final handoff was rendered with the guaranteed
`model_step_1000046592.pt` latent and vanilla paths and the `latest.pt` skill
encoder used by latent training. It produced exactly three seed launchers,
three `bones_pipeline` dependency chains, distinct seed 0/1/2 output roots,
and `0-99%4` goal arrays. No planner job was submitted. Low-level jobs
`3501873` and `3501960` were still running, and authoritative qualification
job `3503363` remained dependency-blocked; the real three-seed submission must
wait for that qualification to finish and pass.

Both 1B-frame low-level jobs completed successfully on 2026-07-15. The first
qualification run failed in its data-preflight command before either oracle
evaluation; it incorrectly supplied a cache-path argument to the raw-data
audit. That routing is corrected and replacement job `3512041` was submitted
on 2026-07-16 with fresh output root
`logs/interface_baselines/bones_seed_100_low_level_qualification_seed0_retry_20260716`.
It completed in 11m10s. Direct vanilla passed with `0.90` strict success and
DiffSR latent passed with `0.84`, both above the fixed `0.80` gate. The skill
binding, 100-motion data preflight, controller-specific cache audits, and
streamed equivalence certificate all pass. The equivalence check covers all
ten packet phases and asynchronous renewal, with maximum command and action
differences `3.02e-7` and `1.31e-6`.

After this gate passed, the fixed seed 0/1/2 paper pipelines were submitted.
Their complete five-stage job chains are recorded in
`wiki/causal-interface-paper-plan.md` and in each persistent
`cluster_submission.json`. At first runtime inspection, prepare jobs `3512092`,
`3512097`, and `3512113` had each passed this fresh-data preflight and written
their seed-specific run configuration. The dependent rollout, fine-tune,
final-evaluation, and summarize jobs must still finish and pass their audits
before paper aggregation.

## The Local Data Changed During This Audit

The first read-only audit on 2026-07-14, before 20:05 EDT, found the scene-grid
defect in the local 10-motion and 100-motion sets. Their root and body positions
used different coordinate frames, and every NPZ omitted `body_names`:

| Earlier set | Language coverage | Body-frame result | Largest offset |
| --- | --- | --- | --- |
| 10 motions | Exact, 10/10 goals | 0/10 passed | 3 m |
| 100 motions | Exact, 100/100 goals | 1/100 passed | 9 m |

At about 20:05, while this work was still running, another process replaced
the NPZs under the same paths. At 20:08 it also changed both manifests and
wrote in-tree preflight reports. This task did not perform or authorize those
writes. The exact command, source hashes, temporary output tree, and responsible
process are unknown. The earlier `/tmp` audit reports were later overwritten by
the required second audit, so hashes for the failing reports are unavailable.

The files currently at those paths pass the body-frame and language-coverage
audit:

| Current set | Body-frame result | Largest offset | `body_names` | Temporal events |
| --- | --- | --- | --- | --- |
| 10 motions | 10/10 passed | 3.58e-7 m | 10/10 | 10/10 motions |
| 100 motions | 100/100 passed | 1.36e-6 m | 100/100 | 100/100 motions (after the 2026-07-14 late-evening sidecar enrichment below; 0/100 at NPZ replacement time) |

Current-state identifiers are:

| Set | NPZ replacement time | Manifest SHA-256 | Aggregate NPZ SHA-256 |
| --- | --- | --- | --- |
| 10 motions | 2026-07-14 20:05:06 EDT | `1301a4e4e25e3b99e39396e2f51d7d7fe530cc4c970e7b635b1b20c191cb1fee` | `56cd5a9710718c85d5e77ea79807f34bfb4ccfa38a3d1e928a6baf405803535b` |
| 100 motions | 2026-07-14 20:05:27-28 EDT | `891c883ce192565d5f02eec175d5f2cc40dd39245565b6f1c3c8004dc29af2b5` | `a9ec2d4947e829e867b45274faabd507fcd2e0e787ea320a2629f1246c661983` |

The aggregate NPZ hash is SHA-256 over sorted records of
`filename + NUL + file_sha256 + newline`. The current manifests say they were
regenerated from raw CSVs, but they contain no exact preparation command or
per-input hashes. Passing the numerical audit does not recover that missing
provenance.

Any cache or policy produced before 20:05 used the defective files and remains
diagnostic only. The current in-place replacements pass the data checks but do
not satisfy the reproducibility gate for final paper runs. Use the fresh-tree
wrapper below to create the paper dataset with complete provenance.

Reproduce the audits from the repository root:

```bash
pixi run python scripts/audit_bones_seed_phase5.py \
    --manifest data/bones_seed/manifests/g1_bones_seed_10_manifest.json \
    --report /tmp/bones_seed_10_phase5_preflight.json

pixi run python scripts/audit_bones_seed_phase5.py \
    --manifest data/bones_seed_100/manifests/g1_bones_seed_100_manifest.json \
    --report /tmp/bones_seed_100_phase5_preflight.json
```

Both commands should currently exit zero, including with
`--require-body-names --require-temporal-events` on the 100-motion manifest.
If the files change again, compare their manifest and aggregate hashes with
the table above and treat the result as a new dataset state.

## 2026-07-14 Late-Evening Sidecar Enrichment and HF Sync

After the NPZ replacement above, the 100-motion language sidecar and its
derived artifacts were updated in place. The NPZs and both manifests were not
touched (the manifest SHA-256 values in the table above still apply).

1. **Temporal events merged.** Per-motion `events` and `num_events` were
   copied verbatim from
   `data/bones_seed/raw/seed_timeline_annotations/timelines.jsonl` (142,220
   rows, keyed by `filename`) into
   `data/bones_seed_100/language/g1_bones_seed_100_language.json` for all
   100 motions (291 events total). This is the same provenance path as the
   10-motion sidecar: its existing events were verified to be byte-identical
   copies of rows from that file. `propagated_from_filename` was synced from
   the timeline rows on 72 records for schema consistency with the 10-motion
   sidecar. Events were not invented; the earlier blocker note referred to
   the absent gated `metadata/seed_metadata_v002_temporal_labels.jsonl`, but
   the local `timelines.jsonl` provides full coverage for the curated 100.
   Sidecar SHA-256 changed from
   `7f6c07b4526bdc65f4db43ab4ba2575e0d50211f0cfc696b5fb5753e51f6c275` to
   `65c496406bf1afbb4185d188fa8089503a68ca80c23d64f2c0b7335f9bcb3023`.
2. **Goal-embedding table added.**
   `data/bones_seed_100/language/g1_bones_seed_100_minilm_goal_embeddings.pt`
   was built with the sidecar-aware `build_language_goal_embeddings.py`
   (commit `22a9dcd` on `feat/dataset-w-goals`; the copy on the eval branch
   cannot read sidecars) using backend `sentence-transformer`,
   model `all-MiniLM-L6-v2`, dim 384, 100/100 sidecar hits. Every embedded
   phrase is the sidecar `language_goal`, which for all 100 motions is the
   upstream `content_short_description`.
3. **Preflight regenerated.**
   `data/bones_seed_100/manifests/g1_bones_seed_100_preflight.json` now pins
   the enriched sidecar hash and passes
   `--require-body-names --require-temporal-events`.
4. **Stale duplicate removed.** The older
   `data/bones_seed/language/g1_bones_seed_100_language.json` (a previous
   selection differing by one motion, referenced by nothing) was deleted so
   no job can resolve the wrong sidecar by path confusion.
5. **HF re-synced.** The three changed files were pushed to
   `GeorgiaTech/g1_bones_seed_100_50hz` in commit
   `11a9e2eabd391f60ba343c554a4ba787d0d48bd8` after checksum-verifying that
   the other 105 remote files were identical to local; the pushed files were
   checksum-verified after upload.

Embedding-model note: on this goal set, MiniLM's nearest-neighbor structure
was measured directly. The 100 embeddings are full rank with worst-pair L2
gap 0.153, exact-lookup retrieval is 100/100, and nearest-prototype
classification stays above 99.8% under Gaussian noise up to the vector norm.
Larger encoders (bge-large, Qwen3-Embedding-0.6B) do not separate the
directional near-pairs (for example "low to high" vs "high to low" pickups)
any better, so the closed-set Phase 5 protocol keeps MiniLM. Revisit only if
the evaluation adds novel-paraphrase goals; the targeted fix there is a
hard-negative contrastive fine-tune, not a larger off-the-shelf encoder.

## Fresh Corrected Export

`scripts/prepare_bones_seed_phase5.py` is the supported wrapper. It:

1. Requires exact CSV/language-sidecar coverage and a language goal for every
   motion.
2. Refuses an existing output directory and refuses an output directory that
   contains, or is contained by, the raw CSV directory.
3. Calls the corrected `scripts/batch_csv_to_npz.py` once for the selected
   motions. That exporter removes each Isaac scene origin before saving
   `body_pos_w` and records `body_names`.
4. Writes relative manifest paths, a normalized language sidecar, source and
   artifact hashes, the exact exporter command, and a preparation record.
5. Runs `scripts/audit_bones_seed_phase5.py --require-body-names`. The
   preparation record is marked complete only when this gate passes.

Use `--dry-run` first. It validates and hashes the inputs without writing any
files:

```bash
pixi run python scripts/prepare_bones_seed_phase5.py \
    --csv-dir data/bones_seed/raw/g1 \
    --language-sidecar data/bones_seed/language/g1_bones_seed_10_language.json \
    --output-root data/bones_seed_phase5_corrected/bones_seed_10 \
    --dataset-name bones_seed_10_phase5 \
    --dry-run
```

Run the real export through the Isaac Lab environment and choose a new output
root:

```bash
pixi run -e isaaclab python scripts/prepare_bones_seed_phase5.py \
    --csv-dir data/bones_seed/raw/g1 \
    --language-sidecar data/bones_seed/language/g1_bones_seed_10_language.json \
    --output-root data/bones_seed_phase5_corrected/bones_seed_10 \
    --dataset-name bones_seed_10_phase5 \
    --device cuda:0 \
    --require-temporal-events
```

For the current curated 100-motion subset, use its matching sidecar at
`data/bones_seed_100/language/g1_bones_seed_100_language.json`. The older
copy under `data/bones_seed/language/` belonged to a previous selection that
differed by one motion; it was deleted on 2026-07-14 (see the enrichment
section above).

```bash
pixi run -e isaaclab python scripts/prepare_bones_seed_phase5.py \
    --csv-dir data/bones_seed/raw/g1_100 \
    --language-sidecar data/bones_seed_100/language/g1_bones_seed_100_language.json \
    --output-root data/bones_seed_phase5_corrected/bones_seed_100 \
    --dataset-name bones_seed_100_phase5 \
    --device cuda:0
```

The wrapper never overwrites or resumes a partial tree. If export fails, keep
the partial tree as a failure record or remove it manually after inspection,
then rerun with a different output root.

## Annotation Blocker (Resolved 2026-07-14)

An earlier version of this page recorded that the 100-motion sidecar had
language goals but no temporal events, so `--require-temporal-events` failed
for all 100 motions. This was resolved by the late-evening sidecar enrichment
documented above: events now cover 100/100 motions, sourced verbatim from the
local `data/bones_seed/raw/seed_timeline_annotations/timelines.jsonl` rather
than the absent gated `metadata/seed_metadata_v002_temporal_labels.jsonl`.
Fresh exports through `prepare_bones_seed_phase5.py` can therefore require
the event gate for the 100-motion set as well. The rule stands: never invent
events from motion filenames; only propagate them from upstream annotation
files.

## Gate Before Training

For final experiments, use the fresh 100-motion tree above. Its preparation
record says `complete`, its preflight report says `passed: true`, and its NPZs
were independently reproduced byte-for-byte. Build separate fresh latent and
vanilla Zarr caches from that manifest. Low-level oracle audits used to open
the planner gate must reference this exact manifest hash and the matching
per-interface cache path; older LAFAN1 or diagnostic BONES-SEED audits do not
satisfy the final Phase-5 gate.
