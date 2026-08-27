# 2026-08-20 — Posterior interface: learning the code THROUGH the policy

Status, 2026-08-20: **designed, smoked, plans resolve, NOT submitted.**

All nine arms completed a 128-frame wiring smoke on the aligned input (`iter=1/1 | frames=128/128`,
zero errors), plans resolve offline with ICE preflight passing and
distinct `PLAN_SHA`s, and 43 contract tests hold the design. The W&B group `posterior-interface` is **confirmed**.

The smoke also checked the thing a smoke usually misses: the resolved configs
were diffed field by field and differ in **exactly** the intended axes
(`recon_coeff`, `train_posterior_through_policy`, `quantizer`,
`code_latent_dim` and the command width that follows from it). A silently-ignored override would otherwise have produced three identical
arms — the same duplicate-cell trap the JEPA arms hit in the frozen study.

## Why this is a separate campaign

Every arm of `2026-08-19-interface-design-study` pretrains a skill encoder
offline for 50,000 updates and then **freezes** it. That whole study therefore
varies *what the frozen code represents* and never *how the code is learned*.

This campaign takes the other route: `command_source=posterior`, where the code
is produced by an encoder that is itself learned during RL. It cannot be an arm
of the star — it differs from `ctrl` in the entire command-generation path, not
in one field — so it gets its own hub, and `ctrl` is a **cross-campaign
reference row**, never a one-field control.

## The 3 x 3 grid

Learning signal x latent space. The signal axis isolates what shapes the code;
the space axis asks whether the star's bottleneck ordering transfers to a
different implementation.

| | AE (`identity`, 256-D) | FSQ (64-D) | VQ (`vq_ema`, K=512) |
|---|---|---|---|
| **reconstruction only** | `post_recon_ae` | `post_recon_fsq` | `post_recon_vq` |
| **policy gradient only** | `post_pg_ae` | `post_pg_fsq` | `post_pg_vq` |
| **both** | `post_pgrecon_ae` | `post_pgrecon_fsq` | `post_pgrecon_vq` |

The reconstruction row is deliberately anchored on a proven setting:
`recon_coeff=1.0` with `train_posterior_through_policy=false` is what
`G1ImitationLatentPerStepVQRLOptIPMDConfig` already runs, so a failure there
indicts this campaign's wiring rather than the idea.

`post_recon` is deliberately the proven setting: `recon_coeff=1.0` with
`train_posterior_through_policy=false` is what
`G1ImitationLatentPerStepVQRLOptIPMDConfig` already runs, so a failure there
points at this campaign's wiring rather than at the idea.

KL (`kl_coeff`) is a fourth cell, not folded into these three — a VAE posterior
is a different claim from "reconstruction helps", and KL coefficients are
sensitive enough to deserve their own arm.

## Fixed across all nine

- `method=patch_autoencoder`; the quantizer and code width are the space axis
- `command_phase_mode=sin_cos`, so the command is the code plus 2 phase values
- `code_period=10`, `latent_steps_min/max=10` — **hold 10, matching `ctrl`**
- `patch_past_steps=0`, `patch_future_steps=9` — a 10-frame window, matching
  `ctrl`'s horizon 10
- everything else is the study's frozen contract: `Isaac-Imitation-G1-v2`,
  tuned entry point, tuned rewards, termination curriculum,
  `random80_adaptive20`, 16,384 x 24, gamma 0.97, 2B frames, seed 0
- **no pretrain stage** — there is no offline encoder to train, which is the
  point. One stage per arm.

## The input view is aligned with `ctrl` — one declaration, both halves

These arms encode the **same 38-value `root_qpos` frame** the frozen study's
control does. That took finding the right knob, and the wrong ones are worth
recording because three of them look like they should work and do not.

`EncoderViewCfg.components` defaults to the full-body trio
(`joint_qpos_qvel, root_pos, root_ori`) — that is where the reference joint
velocity enters. For a LATENT actor this view is the only source of expert
terms in the policy group, because
`policy_command_terms() = actor terms + encoder-view terms` and a latent actor
contributes none. So that one field decides what the posterior is able to read:

```
env.command_interface.encoder.components=[joint_qpos,root_pos,root_ori]
```

The other half is agent-side. `posterior_input_keys` **cannot** be set from the
CLI: `sync_input_keys` re-assigns it during `__post_init__`, after Hydra applies
overrides, so a CLI attempt is silently discarded and the run trains against the
default view while appearing to honour the flag. Hence
`rlopt_ipmd_posterior_root_qpos_cfg_entry_point`, which pins the matching triple.

The two must agree, and a contract test asserts they do — if they drift the run
either dies with `KeyError: 'expert_motion_qpos'` or silently encodes the wider
view the 2026-08-19 study measured as WORSE (-0.0271 success rate, +11.4%
MPJPE-L).

**Ruled out empirically**, so nobody retries them:

- `env.command_observation_terms=[expert_motion_qpos,...]` — reaches the env
  correctly and keeps the term in the resolved policy group, but the term has
  no producer. Keeping BOTH `expert_motion` and `expert_motion_qpos` still
  raises `KeyError` on the qpos one, which is what isolates this to publication
  rather than pruning.
- `env.command_interface.reference.critic_components=[joint_qpos,...]` — that
  is the critic's view, not the policy's.
- Setting `posterior_input_keys` from the CLI — silently discarded, as above.

## No pretrain is the POINT, not a caveat

The frozen arms get a 50,000-update encoder before frame 1; these build theirs
inside the same 2B budget. That is not a confound to apologise for — it is the
property under test. A policy-gradient route does not get a pretraining stage,
and whatever that costs is a result this campaign is here to measure and
report.

## The latent-space axis is the second half of the grid

The posterior path has its **own** bottleneck surface, separate from the offline
`latent_mode`: `latent_learning.quantizer` in `identity` / `fsq` / `vq_ema` /
`gumbel`, with independent `fsq_levels`, `codebook_size`, `codebook_embed_dim`,
`commitment_coeff`, `ema_decay` and Gumbel annealing.

**The star's bottleneck findings do not transfer to it.** Those were measured on
the offline path with a different implementation, so "continuous beats FSQ beats
learned codebooks", and `bn_vq_ema` collapsing to 0 successes in 4,096, are
hypotheses here rather than results. `post_*_vq` is the direct test of whether
that collapse is a property of EMA-VQ or of the offline path that hosted it.

`gumbel` is the one quantizer left out, to keep the grid at nine. It is the
obvious fourth column if the space axis turns out to matter.
