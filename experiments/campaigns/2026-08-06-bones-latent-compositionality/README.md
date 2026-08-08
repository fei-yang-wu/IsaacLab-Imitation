# BONES latent-command compositionality

Status (2026-08-06): complete for the frozen gamma-0.97, rollout-24
root-qpos DiffSR encoder. The result supports **continuous cross-motion
kinematic locality**, not clean discrete semantic clusters.

The hypothesis is: if two short windows move similarly, their 256-D latent
commands should be near each other even when they come from different source
motions. t-SNE does not test this and does not perform clustering; it only lays
high-dimensional points out in two dimensions for inspection. This campaign
therefore uses t-SNE only as a visualization and tests the hypothesis in the
standardized PCA-50 latent and kinematic spaces.

## Definitions

- A **unique publication** is one `(motion, reference step)` command. The 100
  randomized rollouts of that command are averaged, and their latent RMS spread
  is retained. This prevents 100 copies of one command from masquerading as 100
  independent motion examples.
- **Cross-motion retrieval** finds the latent-nearest publications after
  excluding every publication from the query motion.
- The **kinematic distance ratio** is the retrieved neighbors' mean kinematic
  distance divided by the mean distance to all eligible cross-motion windows.
  Random retrieval is 1.0; lower is better.
- **Neighbor recall** is the overlap between latent-nearest and
  kinematic-nearest cross-motion sets. Its lift is measured against random
  expected overlap.
- **Adjusted mutual information (AMI)** measures agreement between cluster
  assignments and a label while correcting for chance. Comparing motion AMI
  with activity AMI reveals whether clusters mostly identify clips or shared
  semantics.
- **HDBSCAN** is a density-based clustering algorithm that may label points as
  noise. It is run on latent PCA coordinates, never on t-SNE.
- A **motion-bootstrap confidence interval** resamples whole source motions,
  not individual publications. This keeps many steps from one clip from
  creating artificially narrow error bars.
- A **semantic region** is one of ten coarse movement classes shared across
  source motions. The ordered rules in `semantic_region_taxonomy.json` are
  frozen before looking at the embedding: jumping; loaded, turning,
  backward/sideways, or forward locomotion; lowered or upright manipulation;
  gesture; stationary; and transition. Detailed human phase descriptions are
  retained and are not replaced by this vocabulary.
- A **phase centroid** is the mean PCA-50 latent feature of all unique
  publications inside one annotated phase. It is a summary of a segment, not a
  new training sample.
- **Held-out-motion phase classification** builds each semantic-region
  centroid using other source motions only, then assigns a phase from the
  omitted motion to its nearest centroid.
- The **return ratio** applies to a non-adjacent `A -> other -> A` semantic
  sequence. It divides the distance between the two `A` phase centroids by the
  mean outward and inward excursion-leg distance. Below 1 means the trajectory
  returns closer to its earlier `A` region than the size of its excursion.

The 30-step kinematic descriptor removes root translation and initial yaw, then
uses eight temporal samples of joint pose/velocity, pelvis path, and 14-body
relative position plus linear/angular velocity. It measures physical movement,
not language-text similarity.

## Protocol

The operational study collected 100 full oracle-policy trajectories for each
of 30 diverse clips in one 3,000-environment Newton rollout. Policy actions were
deterministic. Domain randomization remained enabled, pushes were disabled,
and foot-position XYZ plus base-height termination were disabled. All other
SONIC tracking terminations remained active. The collection produced 137,801
planner publications and 3,000 completed episodes; aggregate SONIC success was
0.9003.

Three clips were failure-heavy: panic 0/100, big-dog walking 7/100, and rock-out
20/100. The robust control removes all three before retrieval. Of the other 27,
the lowest success is exercise at 89/100 and 23 motions are 100/100.

The semantic phases are 85 human-described BONES temporal events converted to
exact zero-based, end-exclusive 50 Hz intervals. Boolean traits are manually
curated in `semantic_traits30.json`. They include locomotion, direction, speed,
jumping, turning, torso lowering, manipulation, object load, and hand activity.

## Results

| set | motions | unique windows | cross-motion distance rho | k=10 distance ratio (95% CI) | k=10 neighbor lift |
|---|---:|---:|---:|---:|---:|
| original selected 10 | 10 | 441 | 0.697 | 0.709 (0.666–0.777) | 13.2x |
| all rollout 30 | 30 | 1,364 | 0.792 | 0.605 (0.562–0.685) | 27.9x |
| robust rollout 27 | 27 | 1,239 | 0.769 | 0.609 (0.565–0.691) | 25.7x |
| reference-only scale | 500 action families | 2,500 | 0.827 | 0.542 (0.530–0.553) | 69.5x |

Here **distance rho** is Spearman rank correlation between every eligible
cross-motion latent distance and its kinematic distance. The 500-family row is
an intrinsic encoder test: it encodes canonical expert H10 root-qpos windows
without policy rollout. One non-mirrored actor/take is sampled from each
normalized action family, so duplicate actors and takes cannot trivialize it.
It complements rather than replaces the operational rollout rows.

On all 30 rollout motions, k=10 activity agreement improves by 13.1 percentage
points over its matched random baseline (95% CI 7.8–21.3). Locomotion improves
by 21.1 points, forward motion by 14.2, and slow locomotion by 9.9. In contrast,
the manipulation and object-loaded confidence intervals cross zero; for
positive queries their precision is actually below the matched baseline. This
matches the visual neighbor gallery: walking style transfers, while an arm
movement for carrying or lifting often retrieves a gesture with similar joint
kinematics but a different task meaning.

Clustering gives the same caution. On the 30 rollout motions, HDBSCAN's motion
AMI is 0.981 versus activity AMI 0.502, 64% of publications are noise, and only
one third of clusters contain multiple motions. On 500 action families,
HDBSCAN labels every point as noise and K-means silhouettes are weak. The
latent is therefore best described as a **continuous movement manifold with
local shared geometry**, not a clean bank of semantic skill clusters.

### Semantic phases and trajectory traversal

The primary traversal analysis uses all 72 annotated phases from the 27
reliably tracked motions and all 1,317 unique publications, including the tail
that the earlier 30-step kinematic-window test had to drop. Every distance,
classifier, and cluster below uses standardized latent PCA-50. PCA-2 and t-SNE
are display coordinates only.

| set | publications / phases | publication k=10 agreement vs random | phase-centroid k=3 agreement vs random | held-out balanced accuracy | K-means semantic AMI | `A -> other -> A` returns |
|---|---:|---:|---:|---:|---:|---:|
| robust 27 | 1,317 / 72 | 25.0% vs 13.5% | 27.3% vs 13.3% | 43.4% | 0.232 | 3/4 |
| all 30 | 1,448 / 85 | 23.8% vs 13.9% | 26.7% vs 13.7% | 31.6% | 0.194 | 4/7 |

For the robust set, the k=10 improvement is +11.5 percentage points with a
motion-bootstrap 95% interval of +5.6 to +19.4; phase-centroid k=3 improves by
+14.0 points with a +4.7 to +21.0 interval. The held-out classifier's 43.4%
balanced accuracy is above its ten-class 10% balanced-chance level and the
20.8% raw majority-class accuracy. This supports shared local information
that generalizes across motions.

It does **not** establish ten clean clusters. Only 36.1% of phase centroids are
closer to a same-region phase than to every different-region phase after
excluding their source motion. Ten-cluster K-means has 0.232 semantic AMI,
0.486 purity, 0.148 silhouette, and 0.428 seed-stability ARI; HDBSCAN labels
every phase centroid as noise. Forward locomotion and stationary phases give
the strongest cross-motion publication improvements (+21.4 and +31.0 points).
Loaded locomotion and upright manipulation are slightly below their matched
random rates (-3.6 and -3.8), so those broad labels do not define one local
region in this encoder.

The temporal result is more specific. `Neutral_stoop_down_001_A057` follows
forward locomotion -> loaded locomotion -> forward locomotion with return ratio
0.633. The drinking sequence returns from its manipulation excursion to its
initial stationary region at 0.250, and the cellphone sequence returns from
stationary to upright manipulation at 0.319. Its second stationary return
fails at 1.839, so the analysis retains a counterexample instead of presenting
only successful paths. Same-type phase-transition directions improve mean
cosine by only 0.018 over other transition types, which is too weak to claim a
shared transition algebra.

The result is therefore: **some semantic phase classes are locally reusable,
and several multi-phase trajectories visibly leave and revisit those regions;
the whole latent space is still a continuous, motion-entangled manifold rather
than a semantic atlas of discrete skills.**

## Reproduce

Run from the repository root. Existing complete artifacts are skipped.

```bash
STAGES=prepare,collect,annotate,analyze,trajectory,scale,gallery \
  experiments/campaigns/2026-08-06-bones-latent-compositionality/run.sh
```

The default reruns only the lightweight analyses and gallery checks:

```bash
experiments/campaigns/2026-08-06-bones-latent-compositionality/run.sh
```

Key outputs:

- `logs/bones_language30_compositionality_oracle_seed0/cross_motion_analysis_all30/analysis.json`
- `logs/bones_language30_compositionality_oracle_seed0/cross_motion_analysis_robust27/analysis.json`
- `logs/bones_language30_compositionality_oracle_seed0/reference_scale_500families_seed0/analysis.json`
- `logs/bones_language30_compositionality_oracle_seed0/cross_motion_neighbor_gallery_robust27_distinct/gallery.html`
- `logs/bones_language30_compositionality_oracle_seed0/semantic_trajectory_analysis_robust27/analysis.json`
- `logs/bones_language30_compositionality_oracle_seed0/semantic_trajectory_analysis_robust27/semantic_trajectory_map.html`
- `logs/bones_language30_compositionality_oracle_seed0/semantic_trajectory_analysis_robust27/latent_map_by_motion.png`

The interactive map defaults to source-motion coloring. Each t-SNE group is
directly labeled `M01`–`M27`, and the legend maps those codes to compact motion
names. Switch **Background legend** to **Semantic regions** to restore the
shared-region coloring while retaining the selected trajectory overlay.

Every gallery row chooses a median-performance positive query, not a best case,
and shows the five nearest distinct source motions. This deliberately exposes
both successful locality and semantic failure cases.
