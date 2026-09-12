# PoE with a command-independent base field (2026-09-12)

User-authorized matched arm `z64_poe_base`, using the existing `z64_poe`
seed-0 run in [the control campaign](../2026-09-11-poe-z64-10b/README.md).
The control is not resubmitted. No research results yet.

The question is whether giving common denoising its own field improves skill
learning without changing the 64-D command interface:

```
control:   eps = sum_k z_k V_k(s, noisy_future, diffusion_time)
treatment: eps = c(s, noisy_future, diffusion_time) + sum_k z_k V_k(...)
```

The shared pair-conditioned mu network predicts 65 fields and phi is `[1; z]`.
The first field is c and sees no z. The only model changes are
`--diffsr_phi_parameterization identity_bias` and `--diffsr_feature_dim 65`,
replacing `identity` / `64`. The extra output field adds parameters; this is
part of the treatment, not a parameter-count-matched comparison. Convex code
mixtures preserve exact denoiser affinity. Exact clean PoE sampling is not
claimed, and the noisy-score restriction remains.

## Matched protocol

- BONES-SEED 129,785 references, persist ID
  `bones_seed_sonic_full_129785@e714bbff`; same frozen source arrays.
- Deterministic 64-D z, encoder hidden LayerNorm, intermediate window,
  horizon 10, source history five plus current, current source anchor.
- Merged raw `diff_chunk` / `boundary_next`; endpoint coefficient zero.
- SIGReg and latent L2 both zero. Same noise schedule and denoising objective.
- 50,000 pretraining updates, batch 8,192, seed 0.
- Frozen encoder, hold 1, sin/cos phase, 66-wide actor command.
- Same full-batch IPMD tracker: 20,480 environments x 24 steps, 10B frames,
  checkpoint every 500M. Each segment carries max_iterations=20,346.
- Same reward, reset, curriculum, actor/critic, and reference-prefetch settings.
- W&B `g1-bs-pareto` / `latent64-probe-10b`; tracker run ID
  `l64p-z64poe-base-s0`, separate from control `l64p-z64poe-s0`.
- ICE, one H200, 16 CPUs, 160G memory per stage, 15:59 walltime. Pretrain to
  tracker is afterok; tracker continuation is afterany with the full frame goal.

Persistent output:
`/storage/ice-shared/vip-vwt/scratch-fwu91/latent64_probe_10b/z64_poe_base_seed0/`.

Control submission: jobs `5764381 -> 5764382 -> 5764383`, source archive
`78026cb53278c886bb60095162a9d96e47c0f972228e48f58b7bd92151d42ae6`.
Live checked during preparation: pretrain completed; first tracker running;
second tracker pending. Do not compare at unmatched frames or infer a repeat
seed result from this single-seed pair. Fresh head/encoder optimization can
introduce initialization noise even with matching seed.

## Validation

- Fifteen focused RLOpt tests passed: original affine/identity behavior, base
  field, convex and general offset algebra, encoder gradients, checkpoint
  roundtrip, sampling, and dimension guards.
- The campaign parity test passed. All resolved stages match the control
  except the two coupled head settings and run/output identities.
- A source comparison against the control archive checked 328 Python files
  across RLOpt, entrypoints, environments, and experiment code. Only the
  three intended files differ: the SR module, config docstring, and CLI choices.
- PyTorch's import-time `torch.jit.script_method` deprecation is escalated by
  existing test warning settings. The focused tests pass when
  `torch.utils.mkldnn` is imported before pytest; no warning policy was changed.
- Ruff reports 20 existing findings in `hl_skill_diffsr.py`, whose only change
  here is the new parameterization's docstring. The changed SR module, CLI,
  and campaign parity test pass lint.
- Local Isaac/Newton pretraining completed five updates at batch 64, using
  campaign-resolved arguments with local arrays, CPU caches, and logging off.
  The checkpoint retains z64, 65 fields, identity_bias, pair conditioning,
  past-five source, and both regularizers off. This is wiring evidence only.
- Tracker smoke passed one full iteration (768 frames, 32 environments),
  loading the new encoder and retaining the strict kit-less invariant.
- All eight ICE preflight checks passed. Parent and RLOpt diff checks passed.

Local evidence: `logs/poe_base_z64_smoke/` contains resolved smoke commands,
pretraining/tracker logs, checkpoint, and the control source comparison.

Focused checks from the repository root:

```bash
pixi run python - <<'PY'
import torch.utils.mkldnn
import pytest
raise SystemExit(pytest.main([
    '-q', 'RLOpt/tests/test_ipmd_components.py',
    '-k', 'poe or identity_phi or pair_mu or mu_conditioning or affine or unknown_phi',
    '--tb=short', '-o', 'addopts=',
]))
PY
pixi run pytest -q source/imitation_experiments/tests/test_poe_base_campaign.py
pixi run python -m imitation_experiments.pipeline.cluster plan \
  --campaign experiments/campaigns/2026-09-12-poe-base-z64-10b/campaign.yaml \
  --arm z64_poe_base --seed 0
```

After submission, use the recorded plan directory with the control-plane
`status` and `logs` verbs. Assess correct versus shuffled-code denoising and
code usage before interpreting tracker SR, MPJPE-L and MPJPE-G at matched
frames. The default diagnostics already include shuffled/zero-code probes.

## Submission

Submitted on ICE, 2026-09-12, with no plan-to-submit drift:

| Stage | Job | Dependency |
| --- | --- | --- |
| pretrain | 5766785 | none |
| lowlevel1 | 5766786 | afterok:5766785 |
| lowlevel2 | 5766787 | afterany:5766786 |

Initial scheduler check: pretrain RUNNING; both tracker stages PENDING on
their dependencies. The pretrain log reached environment initialization.

Plan ID: `poe-base-z64-10b-z64_poe_base-s0-20260912-150728-ad57f68d`.
Plan SHA: `ad57f68dd3afef5fde84f15be60b5054650435b2a9b6e2d238434ac5f7f73219`.
Verified source archive SHA-256:
`0be6a3b2c3d7e4dc8fea5703b3bc74a60810e5fc3db70e5ea1ffb977a18660b6`.
Archive size: 70.1 MB. The archive includes the working-tree RLOpt change;
no submodule commit or parent-pointer update was made. Existing worktree
changes were preserved. The source archive, not Git HEAD alone, identifies
the submitted implementation.

```bash
pixi run python -m imitation_experiments.pipeline.cluster status \
  --submission logs/cluster_control/poe-base-z64-10b/poe-base-z64-10b-z64_poe_base-s0-20260912-150728-ad57f68d
```
