# 2026-07-30 — deterministic-latent end-to-end chain

**Question.** Full pipeline behavior of the deterministic DiffSR latent
interface: offline encoder pretrain → latent-command tracker → rollout
collection → chunked-transformer planner → offline and closed-loop evaluation.

**Status.** Scaffolded 2026-07-30; dry-run verified, no real training executed
yet.

**Front door.** `./run.sh` renders the complete plan (dry run is the frozen
default). Execute with `./run.sh dry_run=false`, or run a stage subset, e.g.
`./run.sh stages='[pretrain,low_level]' dry_run=false`.

**Implementation.** The campaign owns no code: the conductor is
`imitation_experiments.pipeline.run_latent_e2e` (base config
`imitation_experiments/pipeline/conf/latent_e2e.yaml`), which drives the paper
stages `experiments/paper/pipeline/pretrain_latent_encoder.py` and
`train_low_level.py` plus the package collect/merge/train/eval modules. This
campaign pins exactly one variant in [`conf/det_latent_e2e.yaml`](conf/det_latent_e2e.yaml).

**Frozen identities.**

- Latent strategy: `deterministic` (offline lineage), z_dim 256.
- Interface: `latent_skill`; tracker binding validated by the tensor-identity
  gate (`binding` stage) before any collection.
- Data: corrected LAFAN1 manifest + content-bound cache as configured in the
  base pipeline config; replace manifest and cache together only.
- Seed 0. Additional seeds are `./run.sh seed=N output_root=...`, not edits.

**Gates.** The conductor refuses a non-empty output root without
`resume=true`, refuses a resumed stage whose recorded input hashes changed,
and stops at the first failing stage. Strict oracle qualification for paper
use remains the separate campaign gate
(`2026-07-23-bones-phase5-language-local10/interface_baselines/run_lafan1_low_level_qualification.sh`);
this campaign's `binding` stage is necessary but not sufficient for
paper-facing claims.

**Results.** None yet; append run roots and job pointers here after real runs.
