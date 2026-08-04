# 2026-07-30 — deterministic-latent end-to-end chain

**Question.** Full pipeline behavior of the deterministic DiffSR latent
interface: offline encoder pretrain → latent-command tracker → oracle rollout
collection → chunked-transformer planner → offline and closed-loop evaluation.

**Status.** Scaffolded 2026-07-30. Recipe refreshed 2026-07-31 to the
qualified enc380 route-study parameters (the strongest chain to date); the
continuous deterministic bottleneck was selected as the default encoder for
its performance. Dry-run verified; no real training executed from this
campaign yet.

**Front door.** `./run.sh` renders the complete plan (dry run is the frozen
default). Execute with `./run.sh dry_run=false`, or run a stage subset, e.g.
`./run.sh stages='[pretrain,low_level]' dry_run=false`.

**Implementation.** The campaign owns no code: the conductor is
`imitation_experiments.pipeline.run_latent_e2e` (base config
`imitation_experiments/pipeline/conf/latent_e2e.yaml`, which carries the
enc380 recipe), driving the paper stages
`experiments/paper/pipeline/pretrain_latent_encoder.py` and
`train_low_level.py`, the skill-commander oracle evaluator for collection and
closed-loop evaluation, and the package materialize/planner modules. This
campaign pins exactly one variant in
[`conf/det_latent_e2e.yaml`](conf/det_latent_e2e.yaml).

**Frozen identities (the enc380 recipe).**

- Encoder: continuous **deterministic** bottleneck, z_dim 256, horizon 10,
  hidden 1024/512/512, macro window
  `[expert_motion_qpos, expert_anchor_pos_b, expert_anchor_ori_b]`
  (38/frame × 10 = 380), 50k updates at batch 8192.
- Tracker: `latent_skill` on `Isaac-Imitation-G1-Latent-Strict-v0`, h10 hold,
  258-wide command (z256 + sin/cos phase).
- Route: **root_qpos** — the planner predicts the 380-value packet, which runs
  through the frozen encoder at publication.
- Collection: 100 envs, 15,000 control steps, 100 balanced trajectories on
  `walk1_subject1`, saved planner training samples (1000 rows/file).
- Planner: medium flow transformer (`--model_size medium`, d_model 512),
  30k updates, batch 1024 (micro 256), lr 1e-4, wd 1e-4, 16 flow steps.
- Evaluation: 10 envs × 500 steps survival pass; the full-horizon diagnostic
  pass is a documented override in the base config.
- Seed 0. Additional seeds are `./run.sh seed=N`, not edits.

**Gates.** The conductor refuses a non-empty output root without
`resume=true`, refuses a resumed stage whose recorded input hashes changed,
and stops at the first failing stage. Strict oracle qualification for paper
use remains the separate enc380 qualification gate; this campaign's `binding`
stage is necessary but not sufficient for paper-facing claims.

**Results.** None yet; append run roots and job pointers here after real runs.
