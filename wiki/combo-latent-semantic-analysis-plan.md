# Combo encoder: semantic structure of motion windows

Status: reference-only analysis phase completed for the locally available
labels, 2026-09-07. User selected the encoder used by the combo 50B campaign.
The broad geometry, repeated 2,000-family checks, and full temporal-sidecar
expansion are complete; human-curated semantic validation remains.

## First results (2026-09-07)

The local full reference-array cache passed an identity audit: 129,785
trajectories and 47,491,234 transitions, with the expected qpos, anchor pose,
and body-state arrays. The inspected combo encoder is the frozen checkpoint
`logs/combo_50b_local_20260907/encoder/latest.pt` (SHA-256
`da83c3dd00da6b389a6801552a7503b94c143a4737972bd0dfe5a557b84e5810`). These
are reference-only representation results; they do not establish a completed
50B tracker result or closed-loop semantic behavior.

The initial 500-family scale run sampled 2,500 windows (five per family) and compared
the exact raw 380-D combo input with its 64-D latent. Both representations were
standardized and reduced to PCA-50 for metric calculations. The latent had
19 components for 90% variance versus 26 for raw, but raw aligned more closely
with the kinematic descriptor (cross-motion Spearman rho 0.788 versus 0.726;
k=10 retrieved-to-random kinematic distance ratio 0.547 versus 0.545, with
overlapping family-bootstrap intervals). HDBSCAN labeled every point noise in
both spaces. At K=10, latent silhouette was 0.044 and motion AMI 0.129; raw
silhouette was 0.169 and motion AMI 0.118. These are continuous-geometry
sanity checks, not a semantic success criterion.

The separate 30-motion human phase set supplied 1,000 complete H10 windows in
six activities: locomotion, manipulation, combined locomotion/manipulation,
gesture, stationary, and transition. On three valid grouped family-held-out
splits, latent balanced accuracy was 0.383 versus raw 0.345, and latent
macro-F1 was 0.302 versus raw 0.296. Local k=10 activity agreement favored raw
(0.363 versus 0.324; matched random 0.167). Six-way K-means AMI was 0.235 latent
versus 0.250 raw, with silhouettes 0.128 versus 0.186. The current evidence is
therefore mixed: the combo latent modestly improves a global linear semantic
readout, while the raw window retains stronger local activity geometry. The
next primary measurements should quantify smoothness and cross-motion reuse of
semantic attributes along trajectories, rather than try to produce discrete
clusters.

The trajectory continuity check supports this framing. For the 30 annotated
motions, adjacent-window distance divided by a five-window offset distance was
`0.628` in latent PCA-50 and `0.603` in raw PCA-50. Activity-boundary distance
was larger than within-activity distance in both spaces, with ratios `1.203`
latent and `1.340` raw. These are descriptive, correlated-window statistics,
not independent significance tests, but they show smooth temporal paths and a
weak boundary effect without requiring discrete regions.

The expanded scale run sampled 10,000 windows from 2,000 normalized action
families at each of three sampling seeds. Across seeds, latent-to-kinematic
cross-motion Spearman rho was `0.745 +/- 0.009`, while raw-to-kinematic rho was
`0.802 +/- 0.005`. The k=10 retrieved-to-random kinematic distance ratio was
`0.495 +/- 0.003` for latent and `0.500 +/- 0.003` for raw; the family-bootstrap
intervals overlap in every seed. Thus the larger sample confirms strong,
continuous cross-motion locality in both spaces, with raw input geometry
slightly more aligned to the kinematic descriptor. It does not turn the
question into a clustering problem.

Artifacts:

- `logs/combo_semantic_main_20260907/analysis.json` and
  `reference_scale_summary.png`: 500-family paired raw/latent scale run.
- `logs/combo_semantic_scale2k_20260907/analysis.json`,
  `logs/combo_semantic_scale2k_s1_20260907/analysis.json`, and
  `logs/combo_semantic_scale2k_s2_20260907/analysis.json`: 2,000-family,
  10,000-window runs at seeds 0/1/2.
- `logs/combo_semantic_annotations_v2_20260907/analysis.json` and
  `semantic_tsne.png`: grouped semantic probe on the 30-motion annotation set.
- `logs/combo_semantic_annotations_v3_20260907/analysis.json`: same semantic
  probe with temporal smoothness and activity-boundary diagnostics.
- `logs/combo_semantic_scale2k_clip_categories_20260907.json`: weak clip-level
  category probe over the 2,000-family expansion.
- `logs/combo_semantic_annotations_v5_20260907/analysis.json`: per-axis
  grouped probes, retrieval, and temporal smoothness on the 30-motion set.
- `logs/combo_semantic_annotations_v7_20260907/analysis.json`: per-axis
  probes with the architecture-matched random-encoder control.
- `logs/combo_semantic_temporal20k_v2_20260907/analysis.json`: full temporal
  sidecar expansion over 3,861 normalized action families with raw event text,
  keyword-derived event classes/axes, and random-encoder controls.
- `logs/combo_semantic_temporal20k_s1_20260907/analysis.json` and
  `logs/combo_semantic_temporal20k_s2_20260907/analysis.json`: repeated event
  sampling at seeds 1/2 over the same family population.

The temporal sidecar was synchronized from `mel07876d` and verified
byte-for-byte locally (142,220 rows, SHA-256
`379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8`). It
joins all 129,785 language-sidecar motions and provides 327,930 event records.
One event-centered H10 window was sampled for each of the 3,861 eligible
normalized action families. Because the sidecar has free-text event
descriptions rather than a fixed semantic ontology, the scaled probe retains
the original text and uses explicit keyword-derived event classes and axes.
On three grouped family-held-out splits, the event-class probe gave latent
balanced accuracy `0.366` / macro-F1 `0.313`, versus raw `0.349` / `0.292` and
random `0.361` / `0.293` in seed 0. Across event-sampling seeds 0/1/2, the
event-class probe averaged latent `0.363 +/- 0.010` / `0.311 +/- 0.007`, raw
`0.353 +/- 0.011` / `0.295 +/- 0.006`, and random `0.351 +/- 0.014` /
`0.289 +/- 0.009`. Event-class k=10 retrieval favored raw (`0.351 +/- 0.003`)
over latent (`0.327 +/- 0.003`) and random (`0.343 +/- 0.002`). A clip-category
probe with rare categories merged only for the linear split averaged latent
`0.528 +/- 0.005` / `0.423 +/- 0.003`, raw `0.527 +/- 0.009` / `0.417 +/- 0.004`,
and random `0.508 +/- 0.008` / `0.397 +/- 0.008`; category retrieval again
favored raw (`0.744`) over latent (`0.675`). These labels are useful scale
diagnostics, not equivalent to manual temporal annotations.

The next step is to replace the keyword-derived labels with a small human-audited
event ontology, repeat the same extraction at two additional sampling seeds,
and add phase-trajectory inspection for representative cross-family neighbors.
Until that audit is done, the scaled temporal numbers establish promising
accessibility and continuous organization, but not a definitive claim about
human semantic meaning. The t-SNE maps remain display diagnostics only.

As a weak broader check, the full 129,785-motion language sidecar does provide
one clip-level category per motion. On the 2,000-family, 10,000-window sample,
there are 20 categories. Grouped category classification gives balanced
accuracy `0.296` latent versus `0.277` raw; k=10 category agreement is `0.492`
latent versus `0.534` raw, with a matched random rate of `0.155`. These labels
are repeated across every window of a clip and therefore cannot establish
within-clip temporal semantics. They support the same tentative pattern as the
30-motion temporal probe but remain secondary evidence.

The 14-axis probe on the annotated 30-motion set is more informative than the
single activity label. On valid grouped splits, latent balanced accuracy is
higher than raw for locomoting (`0.815` vs `0.779`), manipulating (`0.567` vs
`0.531`), torso lowered (`0.745` vs `0.636`), turning (`0.599` vs `0.549`),
sideways motion (`0.593` vs `0.515`), slow locomotion (`0.750` vs `0.722`),
and jumping (`0.763` vs `0.736`). Raw is higher for object-loaded and backward
motion. Retrieval is mixed by attribute: latent is stronger for hand activity
and slow/forward locomotion, while raw is stronger for manipulation and
object-loaded windows. This supports attribute-specific semantic locality,
not a single globally ordered semantic axis.

The architecture-matched random encoder control scores `0.297` balanced
accuracy and `0.238` macro-F1 on the six-way activity probe, below learned
latent (`0.383` / `0.302`) and raw (`0.345` / `0.296`). The learned latent also
beats the random encoder on locomoting, manipulating, torso-lowered, turning,
and object-loaded axes, while random remains competitive for some hand-action
attributes. This is evidence for learned reorganization in part of the
semantic space, not a uniform improvement over the input.

## Question and scope

Does the learned latent organize short motion windows by reusable movement
meaning across different source motions? Compare the same windows in raw
motion space and latent space. Distinguish three possible findings:

1. Motion geometry is preserved.
2. Semantic labels are more accessible through simple distances or linear
   readouts in the latent than in raw motion.
3. Semantic attributes vary smoothly and are reusable across motions.

These are separate claims. A deterministic encoder cannot add information
absent from its input; the useful question is whether learning reorganizes
that information. Since motion and labels are continuous or multi-label,
discrete clustering is only a secondary diagnostic for accidental modes,
sampling artifacts, or motion-identity leakage.

This is a reference-only representation study initially. It does not measure
tracker success, execution robustness, or skill composition.

## Target and input contract

Use the frozen `p5_affine` encoder bound to
`experiments/campaigns/2026-09-03-combo-50b/`. The local inspected release is
the 46,000,373,760-frame tracker snapshot within that campaign, not proof of
50B completion. Its bundle records tensor-identical frozen encoder binding:

- Encoder checkpoint: `logs/combo_50b_local_20260907/encoder/latest.pt`.
- Recorded checkpoint SHA-256:
  `da83c3dd00da6b389a6801552a7503b94c143a4737972bd0dfe5a557b84e5810`.
- Export manifest:
  `logs/combo_50b_local_20260907/hf_release/controller/combo_46b/manifest.json`.
- Input: 10 x 38 root_qpos values, stride 1 at 50 Hz, horizon 10,
  intermediate-window mode; output: continuous 64-D z.
- Runtime anchoring: robot heading, XY origin removal, preserved height and
  tilt. Hold 1 publishes each control tick. Exclude appended phase coordinates
  from latent analysis. The past-5 affine training head is not the z command.

Before extraction, hash the actual checkpoint and confirm tensor binding to
the intended combo snapshot. If the final 50B tracker is available, bind to
it; if its encoder is unchanged, reference embeddings are unchanged. Do not
attribute encoder semantics to 50B of tracker training.

Use the native window builder and verify actual frame offsets, order,
orientation representation, and anchoring against the runtime/export path on
real windows. In reference-only extraction the reference anchor supplies the
canonical robot anchor; operational robot/reference mismatch is a later test.
Ten frames span 0.18 seconds between sample endpoints, approximately a 0.2 s
window. Do not silently lengthen the encoder input to fit an activity label.

## Dataset and sampling

Use the broad BONES-SEED G1 export and its temporal annotations, not the
selected-ten development set or a subset selected for tracker success. The
dataset skill documents a 129,785-motion export; audit the actual manifest,
arrays, units, joint/body order, frame conventions, and annotation coverage
before fixing the eligible population. Record all exclusions and hashes.

Proposed staged budget:

| Stage | Independent sources | Windows | Purpose |
| --- | --- | --- | --- |
| Pilot | 100 action families, up to 2 distinct clips each | about 2,000 | Packing, labels, coverage, first plots |
| Main | 500 action families, up to 2 distinct clips each | about 10,000 | Paired raw/latent analysis |
| Expansion | 2,000 families | 10,000 | Completed at seeds 0/1/2; five windows per family |

These are targets, not verified inventory counts. Sample ten eligible windows
per clip where duration permits; never duplicate or pad windows to meet quota.
Use three sampling seeds for the main study and report repeated windows across
resamples. Cap t-SNE at a fixed stratified 10,000-point subset per draw.

Stratify using metadata before viewing embeddings: locomotion and direction,
turning, jumping, lowering/rising, reaching/manipulation, gestures, dance,
exercise, stationary periods, and transitions. Include different speeds,
actors, styles, and multi-phase clips. Avoid walking dominating by duration.
Report the balanced sample's composition; do not present it as the natural
dataset prevalence.

One macro sample is one native encoder window, not one frame, whole clip, or
rollout repeat. Primary statistics use non-overlapping windows spread across
clips and events. Record family, source, actor, take, mirror identity, start
frame, frame offsets, phase overlap, and split. Keep mirrored variants and
near-duplicate takes together. Hold out whole normalized action families for
probes, with broad semantic categories represented across folds; use actor
holdout as a secondary check where supported. Exclude the query family in
retrieval. Probe holdout does not imply holdout from encoder pretraining.

Separately retain dense consecutive windows from 20-30 multi-phase clips for
trajectory plots. Do not treat those correlated points as independent samples.

## Semantic labels

Freeze labels before inspecting embeddings. Map temporal events to window
intervals with explicit timestamp/FPS conversion. Use multi-label primitive
attributes plus coarse phase classes; keep task/object-context labels separate.
Mark mixed, ambiguous, and unlabelled windows explicitly. Primary pure-phase
tests require the full input window inside an annotated phase; boundaries get
a separate analysis. Do not copy whole-clip descriptions onto every window.

Audit about 200 stratified windows using short videos, with longer context
available for annotation verification. Have two raters inspect an overlapping
subset without seeing embeddings and record agreement and corrections. Retain
an unknown class when the short window cannot establish a task meaning.
Text-embedding similarity is optional supporting evidence, not ground truth.

## Paired representations

For every sample retain:

| Representation | Role |
| --- | --- |
| Exact packed 380-D input | Direct raw-window baseline with identical information |
| Learned 64-D z | Representation being tested |
| Raw PCA-64 | Linear compression baseline at matched width |
| Single-frame pose and simple velocity/path descriptors | Diagnose static pose versus temporal movement |
| Random encoder with matching architecture, several seeds | Diagnose effects of learned weights |

Optional body/velocity descriptors derived from the same time support provide
an interpretable kinematic comparison. Report their extra derived features
explicitly. No longer-horizon data belongs in the primary raw/latent comparison.

Fit scaling and PCA only on probe-training folds. Use feature standardization
with a low-variance floor as the primary geometry; report latent unscaled and
raw feature-block-balanced sensitivity checks. LayerNorm is part of the frozen
model, not a substitute for documenting analysis scaling. Do not whiten by
default. Check finite values, latent variance, effective rank, and outliers.

## Analyses, in order

1. **PCA:** raw and latent scree/cumulative-variance curves, PCA-2 and PCA-3,
   and raw PCA-64 baseline. Color the same samples by semantics, source family,
   actor, speed, root height, and phase. Inspect raw loadings for nuisance
   dominance. Separate PCA axes are not directly aligned between spaces.
2. **t-SNE:** paired maps with identical sample IDs and color keys;
   perplexities 15/30/50 and three seeds. Use up to 50 PCs as preprocessing,
   recording retained variance in each representation. Report neighborhood
   trustworthiness. Treat apparent gaps, island sizes, and inter-island
   distances as visualization, not quantitative semantic separation.
3. **Continuity sanity checks:** retain K-means and a small predefined HDBSCAN
   parameter grid in higher-dimensional feature/PCA spaces, never t-SNE, only
   to detect motion-identity leakage, sampling artifacts, or accidental modes.
   Compare matched PCA-50 raw/latent spaces and native-space sensitivity, but
   do not treat silhouette, AMI, or cluster count as a primary semantic score.
4. **Semantic accessibility:** cross-family nearest-neighbor agreement and
   retrieval at k=5/10/20; regularized linear probes with grouped validation,
   balanced accuracy/macro-F1 and per-label precision/recall. Tune using
   training/validation groups, then evaluate untouched groups. Use the same
   splits and tuning budget across representations. Compare with class-prior,
   raw, PCA-64, random-encoder, and group-aware shuffled-label baselines.
5. **Representational similarity analysis:** compare raw, latent, kinematic,
   and semantic dissimilarity matrices on a fixed balanced subset of about
   2,000 windows. Test associations using family-aware resampling/permutations,
   not independent pairwise-distance tests. Check semantics within matched
   pose/speed bands to diagnose obvious nuisance explanations; this does not
   establish semantics independent of all kinematics.
6. **Temporal inspection:** overlay dense phase trajectories and inspect
   cross-family neighbor video galleries, including errors. Compare single
   pose versus full-window performance. Time reversal/shuffling can serve as
   secondary order-sensitivity diagnostics, explicitly qualified as possibly
   out-of-distribution inputs. Longer 1-2 s latent-sequence summaries are a
   separate follow-up if short-window task semantics remain ambiguous.

Bootstrap whole action families for paired raw/latent differences. Preserve
within-family temporal dependence in null construction. Show intervals and
per-class sample counts. Sampling/projection seeds are not encoder-training
replicates; conclusions initially concern this frozen encoder only.

## Interpretation and deliverables

Promising evidence is reproducible semantic retrieval or linear-readout
improvement over raw/PCA/random baselines on held-out families, with meaningful
neighbor videos and effects not explained solely by clip identity or speed.
If raw and latent are similar, report semantic preservation/compression without
claiming improved organization. If locality or held-out readouts improve,
report continuous semantic organization and trajectory smoothness. If only
t-SNE looks convincing, the evidence is inconclusive. Freeze primary metrics
before examining the main set:
paired cross-family k=10 semantic agreement and grouped linear-probe macro-F1.

Deliver an immutable sampling/config manifest, paired feature arrays, label
audit, PCA/t-SNE figures, clustering and probe tables with intervals, and a
linked neighbor/phase video gallery. Keep generated artifacts under `logs/`;
keep shared implementation and focused tests in `source/imitation_experiments/`.
No simulator rollout is needed for the main extraction; budget the pilot on
local CPU/GPU, measure cost, and use the cluster control plane if expansion
requires large compute. Encoder retraining is outside this plan.

## Reuse and references

The older `2026-08-06-bones-latent-compositionality` campaign reports locality
with weak discrete clustering for a different encoder. Its reported results
are background, not current combo measurements; underlying results were not
recomputed for this plan. Reuse its annotation, retrieval, bootstrap, gallery,
and trajectory infrastructure after adapting contracts.

In particular, `analyze_reference_latent_scale.py` currently uses full
frame-zero orientation removal and translation subtraction. It must not be
used unchanged for combo's heading-only, height-preserving inputs.

- [t-SNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html): initialization and perplexity sensitivity.
- [Representational similarity analysis](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full): comparing representation geometry with behavioral and model structure.
