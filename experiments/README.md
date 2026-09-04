# Experiments

Start here before running anything under `experiments/`.

The complete live-file classification is in
[`SCRIPT_INVENTORY.md`](SCRIPT_INVENTORY.md). The 64 paths removed in the
2026-07-23 cleanup and their recovery instructions are in
[`PRUNED_SCRIPTS.md`](PRUNED_SCRIPTS.md).

The experiment surface has three layers:

- **Library** — `source/imitation_experiments/` is the installable, tested
  implementation (planner training, command publication, low-level tracking,
  evaluation, audits, provenance). Launchers invoke it with
  `python -m imitation_experiments.<subpackage>.<module>`. All shared Python
  lives here with a test; none lives under `experiments/`.
- **Campaigns** — `campaigns/YYYY-MM-DD-short-name/` records when a concrete
  experiment protocol was frozen and gives collaborators one front door for
  that work. A campaign directory is thin: a `README.md`, configuration, and
  the frozen shell launchers. Frozen campaigns are append-only (status and
  provenance updates); their launchers are never silently rewritten.
- **Release** — `paper/` is the small, stable set of release-facing
  entrypoints that must not move as campaigns come and go.

## Current work

| Priority | Campaign | State |
| --- | --- | --- |
| Primary (submission authorized) | [`2026-09-04-direct-affine-phi`](campaigns/2026-09-04-direct-affine-phi/README.md) | One arm pretrains a 64-D affine feature, then trains a tracker for 10B frames with 20,480 environments. The command is 64-D phi plus two phase values. z64, weight decay, linear critic schedule, LayerNorm past-5 encoder, and LSTM; no explicit actor history. W&B project `g1-bs-pareto`, group `direct-affine-phi`. Earlier 256-D-feature plans are obsolete. |
| PARKED (complete 2026-08-25) | [`2026-08-22-reward-estimation`](campaigns/2026-08-22-reward-estimation/README.md) | IPMD direct reward estimation (IRL) across explicit and latent-headline trackers, eight arms at 10B, four estimator configurations (vanilla / grad penalty 0.5 / grad penalty 5.0 + logit reg / goal-conditioned paired input). **Two results, both one seed:** the stack is a null on tracking — every row inside noise of its no-reward-estimation counterpart (explicit 0.956-0.960 SR / 17.5-17.9 mm, latent 0.937 / 23.6-24.2 vs headline 0.9368 / 23.61) — and the estimator itself saturates in every configuration, because the diff objective has no interior optimum. Parked as future work: reviving it needs a different objective, harder negatives, and a calibration target. Machinery stays in-tree, off by default. |
| Primary (smoked, not submitted) | [`2026-08-20-posterior-interface`](campaigns/2026-08-20-posterior-interface/README.md) | The counterpart to the frozen star: the code is learned THROUGH the policy (`command_source=posterior`) instead of pretrained offline and frozen. A 3 x 3 grid: learning signal (reconstruction / policy gradient / both) x latent space (AE / FSQ / EMA-VQ), at the star's own hold 10 and 2B budget, with no pretrain stage. All nine smoke on an input view ALIGNED with the star's control (`root_qpos`, no qvel), plans resolve, 43 contract tests including one that pins the two halves of the input declaration together. The posterior quantizer is a different implementation from the offline one, so the star's bottleneck findings are hypotheses here — `post_*_vq` directly tests whether `bn_vq_ema`'s collapse was the quantizer or the path. |
| Primary (COMPLETE 2026-08-20) | [`2026-08-19-interface-design-study`](campaigns/2026-08-19-interface-design-study/README.md) | The paper's low-level section as a design study of learned command interfaces. A star of 34 arms (29 active) over four axes -- what trains the encoder, how the code is constrained, what the encoder reads, and how the code reaches the tracker -- each changing exactly ONE field from the `ctrl` hub, plus one width-x-hold interaction probe. Every arm trains to the same 2B frames and **no arm is promoted**: budget is an axis, read off checkpoint curves on the new `paper_milestone_testbed256_v1` profile. **29 arms are active and every one passed its wiring smoke**; the three co-trained arms and `use_phi`/`use_z_phi` are deferred to tier 4 by user decision on 2026-08-19 and are never planned. W&B group `interface-design-study` is confirmed. One seed per arm. **All 29 arms trained to 2B and scored**: 29 clean + 29 robust + 232 milestone rows, zero eval failures. The control configuration wins; discrete costs ~0.03 SR against continuous at this budget; `bn_vq_ema` collapses outright; nine arms had not converged at 2B. Results in `wiki/current-status.md`. |
| Primary (prepared, not submitted) | [`2026-08-18-bones129k-latent-design-ablation`](campaigns/2026-08-18-bones129k-latent-design-ablation/README.md) | Paper method ablation: endpoint DiffSR versus exact-window reconstruction, crossed with continuous 256-D versus SONIC FSQ 64 x 32. All arms fix `root_qpos`, hold 10, offline pretrain then freeze, and 10B tracker frames. Both reconstruction arms passed one-update pretrain and 128-frame frozen-tracker smokes; all four seed-0 plans resolve offline. W&B group `latent-design-ablation` is proposed and needs confirmation before submit. |
| Supporting (local, complete) | [`2026-08-09-bones129k-recent-local-eval`](campaigns/2026-08-09-bones129k-recent-local-eval/README.md) | Seven recent finished ICE trackers plus NVIDIA SONIC v1.1 evaluated on the frozen 4,096-motion block. v1.1 leads success at 0.9944. `expert_heading` leads local-policy success at 0.9175; `z256_scaled` reaches 0.9146 and leads local successful MPJPE-L at 23.27 mm. The full-body encoder row fails all motions. The running 10B L2T job remains excluded until it has a final checkpoint. |
| Primary (ICE, incomplete) | [`2026-08-09-bones129k-l2t`](campaigns/2026-08-09-bones129k-l2t/README.md) | Privileged explicit-command teacher plus deployable DiffSR latent-command student on all 129,785 motions. H200 job `5573723` completed normally but was incorrectly capped at 1B instead of the required 10B frames. It is an incomplete qualification and was not resubmitted. The local launcher now defaults to 10B. W&B project/group: `g1-bones-seed` / `l2t`, run `2znme7lg`. |
| Primary (ICE, running) | [`2026-08-06-bones129k-skill-encoding`](campaigns/2026-08-06-bones129k-skill-encoding/README.md) | Three root-qpos DiffSR factorizations—checkpoint occupancy, fixed-skill semi-Markov chain, and endpoint delta—passed source-bound smokes and were submitted as matched 5B-frame H200 runs. Encoder jobs `5570344`, `5570351`, and `5570358` are running; dependent controllers `5570359`, `5570368`, and `5570370` wait on arm-specific `afterok` gates. W&B group: `skill-encoding-ablation`. |
| Supporting (local, complete) | [`2026-08-06-bones-latent-compositionality`](campaigns/2026-08-06-bones-latent-compositionality/README.md) | Cross-motion analysis over 30 randomized oracle-policy motions plus a 500-action-family reference control. Shared phase labels improve held-out local retrieval, and 3/4 robust `A -> other -> A` trajectories return in latent space, but unsupervised clusters remain weak and motion-entangled. |
| Primary (local, complete) | [`2026-08-06-bones-language10-latent-receding`](campaigns/2026-08-06-bones-language10-latent-receding/README.md) | The 70-rollout seven-row grid passed its strict aggregate. Future-publication H3 fresh-only wins the SR-first ranking at 0.401 versus 0.329 for H1; future clipped/gated is the quality Pareto point at 0.396 SR and 39.72 mm successful MPJPE-L. Current/stale-frame overlap is rejected. |
| Primary (local) | [`2026-08-05-bones-language10-oracle-pretrain`](campaigns/2026-08-05-bones-language10-oracle-pretrain/README.md) | Selected-ten language baseline: one 1,000-env complete-trajectory collection, oracle-only medium planner pretraining for 10k updates, deterministic SONIC evaluation every 2k, plus stored root-qpos and 30-frame lookahead for later interface/architecture ablations. The real end-to-end smoke passes. |
| Primary (local) | [`2026-08-04-bones129k-v2-adaptive-10b`](campaigns/2026-08-04-bones129k-v2-adaptive-10b/README.md) | Full 129,785-motion low-level-from-scratch follow-up: accepted root+qpos encoder, tuned v2 global-tracking rewards, SONIC full-trajectory adaptive resets, guarded fresh replay cache, and 32,768 × 6 local geometry under a 10B-frame cap. |
| Primary | [`2026-07-29-sonic-official-fsq`](campaigns/2026-07-29-sonic-official-fsq/README.md) | Official-window SONIC low-level reproduction with the sample-efficient `[0, 200]` reset sampler: one 64-D normalized FSQ command (32 levels per coordinate) recomputed from current + nine future frames every 50 Hz step. ICE H200 job `5549500` submitted 2026-07-29; first 1,259,962,368-frame segment is pending resources under a 5B cumulative cap. |
| Primary | [`2026-07-29-latent-holdout-horizon`](campaigns/2026-07-29-latent-holdout-horizon/README.md) | Command-interface ablation at fixed latent space: one shared frozen h10 encoder, hold in {5, 1} against the hold=10 control curve. Submitted to ICE 2026-07-29 as `5548369` (hold=5) and `5548370` (hold=1), one 16h segment each, no resume chain. |
| Previous (paper-protocol diagnostic) | [`2026-07-23-bones-phase5-language-local10`](campaigns/2026-07-23-bones-phase5-language-local10/README.md) | Historical row-budgeted local10 workflow retained for frozen Phase-5 provenance. The active selected-ten local experiment is the 2026-08-05 trajectory-first campaign above. |
| Previous (paper-protocol diagnostic) | [`2026-07-23-bones-phase5-language-h200`](campaigns/2026-07-23-bones-phase5-language-h200/README.md) | Historical guarded H200 row-budget pilot. It is not the current selected-ten collection front door. |
| Primary | [`2026-07-22-latent-learning-ablation`](campaigns/2026-07-22-latent-learning-ablation/README.md) | All twelve local 10M qualification arms passed; the H200 submission remains deliberately gated and was not submitted as of the recorded 2026-07-22 status. |
| Supporting | [`2026-07-22-bones-h10-scale`](campaigns/2026-07-22-bones-h10-scale/README.md) | The wall-clock screen selected the H200 16,384 x 12 profile used by the latent-ablation campaign. |
| Previous | [`2026-07-21-interface-scale`](campaigns/2026-07-21-interface-scale/README.md) | Historical tracker-scale campaign. Read its result and data-loss notes; do not treat its launchers as the current submission surface. |

“Current” is an explicit decision in this file, not whichever directory sorts
last. Update this table when the project changes focus.

Changing scheduler state is recorded in
[`wiki/current-status.md`](../wiki/current-status.md). Query the scheduler
before treating any dated job state as live.

## Paper-facing surface

[`paper/`](paper/README.md) is reserved for the stable, public reproduction
entrypoint. It is intentionally marked as staging while Phase 4 and Phase 5
remain incomplete. Exploratory launchers, one-off recovery scripts, and
diagnostics do not belong there.

The authoritative two-row paper protocol remains
[`wiki/causal-interface-paper-plan.md`](../wiki/causal-interface-paper-plan.md).

## Directory map

| Path | Purpose |
| --- | --- |
| `../source/imitation_experiments/` | The shared experiment library and its tests (`pixi run test-experiments`). |
| `campaigns/` | Dated experiment decisions, status, and the frozen shell launchers a collaborator invokes. |
| `paper/` | Stable release-facing entrypoints only; no implementation, diagnostics, or tests. |

Shared implementation has exactly one home — the package — so a launcher's
path states which protocol froze it while the code it drives stays current,
importable, and tested. Older group directories (`interface_baselines/`,
`command_space_ablation/`) now hold only their campaign's shell launchers;
their former Python contents live in the package subpackages
`data`, `planner`, `lowlevel`, `evaluation`, `audit`, `provenance`,
`pipeline`, and `capacity`.

Campaign-local Python that is genuinely single-campaign (for example
`latent_ablation/` in `2026-07-22-latent-learning-ablation`) may stay in that
campaign, but anything imported or invoked by another campaign, `paper/`, or
`scripts/` must move into the package.

`paper/` holds the guarded entrypoints and nothing else: `run.sh`, the Phase-4
and Phase-5 submit plus aggregate scripts, and the release-bundle builder. The
tests that span this boundary live in `source/imitation_experiments/tests/`
and reach the paper entrypoints through that directory's `conftest.py`.

Every retained script must have a row in `SCRIPT_INVENTORY.md`. Completed
one-off launchers do not remain beside live code merely as an archive: their
paths and reasons are recorded in `PRUNED_SCRIPTS.md`, while their source stays
recoverable from Git history.

Never submit a script solely because it exists. Begin with a dated campaign or
the paper front door, inspect its recorded status, and dry-run any guarded
launcher before allowing scheduler mutation.

## Adding a campaign

Use an ISO date plus a descriptive slug:

```text
experiments/campaigns/2026-07-23-short-purpose/
```

The date is the protocol-freeze or first-submission date, not every rerun date.
Each campaign must contain a `README.md` that records:

- research question and scope;
- current status and last verified date;
- canonical launcher paths;
- frozen data, checkpoint, and protocol identities;
- dry-run or local qualification command;
- submission command and safety gate;
- result/job pointers;
- superseding campaign, when applicable.

Keep the campaign directory itself thin: a `README.md`, configuration, and
the shell wrappers a collaborator invokes. Put reusable Python in
`source/imitation_experiments/` with a test and call it from the wrapper with
`python -m`; do not add importable modules, `sys.path` mutation, or copies of
another campaign's code to a campaign directory. After a real submission,
append status and provenance; do not silently rewrite the frozen protocol.
