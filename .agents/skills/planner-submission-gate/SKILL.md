---
name: planner-submission-gate
description: Run the fail-closed audits that must pass before a planner run is submitted or its numbers are reported — encoder/checkpoint binding, DiffSR latent qualification, and the BONES-SEED or Phase-4 submission validators. Use before submitting a planner job, before Isaac evaluation of a latent arm, when pairing a policy with a skill encoder, when a qualification or audit JSON is requested, or when the user asks whether a checkpoint pair is valid.
---

# Planner submission gate

These audits fail closed. A missing audit JSON is a blocked submission, not a
warning. All of them live in
`source/imitation_experiments/imitation_experiments/audit/` and run in the
default Pixi environment.

Run them in this order.

## 1. Binding — the encoder inside the policy is the encoder you cite

A DiffSR qualification must prove that the selected skill checkpoint's
`skill_encoder_state_dict` is **tensor-identical** to the encoder embedded in
the latent low-level checkpoint.

```bash
pixi run python -m imitation_experiments.audit.validate_latent_skill_checkpoint_binding \
    --low_level_checkpoint <policy.pt> \
    --skill_checkpoint <encoder.pt> \
    --output_json <out.json>
```

Prefer the exact skill checkpoint path that low-level training recorded, even
when another checkpoint happens to hold identical runtime weights. Keep the
output JSON: later gates require the binding record.

Interface width is a separate question from binding. Invoke the
`g1-encoder-interface` skill before you pair an encoder whose interface
(`full_body` vs `root_qpos`) or macro-window stride is not already proven. A
mismatch fails loudly with an `hl/state shape` error; it never fails silently.

## 2. Qualification — the oracle run met its protocol

```bash
pixi run python -m imitation_experiments.audit.audit_diffsr_latent_qualification \
    --summary <qualification summary.json> \
    --low_level_checkpoint <policy.pt> \
    --skill_checkpoint <encoder.pt> \
    --manifest <dataset manifest.json> \
    --output_json <out.json> \
    [--expected_dataset_path ...] [--expected_num_envs 40] \
    [--expected_steps 1000] [--expected_seed 0] \
    [--expected_task Isaac-Imitation-G1-Latent-v0] \
    [--expected_planner_target_dim 256] [--success_threshold 0.8] [--require_pass]
```

Pass `--require_pass` for a real gate. Without it the command reports and
exits 0, which is a diagnostic, not a gate.

## 3. Submission validator — every artifact matches every other artifact

Pick the one that matches the campaign. Both take positional paths plus the
expected content hashes, and both refuse when any hash, count, or audit
verdict disagrees.

```bash
# BONES-SEED language planner
pixi run python -m imitation_experiments.audit.validate_bones_seed_planner_submission \
    <manifest> <language> <preparation> \
    <vanilla_checkpoint> <latent_checkpoint> <skill_checkpoint> \
    <vanilla_audit> <latent_audit> <equivalence> \
    <expected_latent_dataset_path> <expected_vanilla_dataset_path> \
    <expected_manifest_sha256> <expected_language_sha256> <expected_preparation_sha256>

# Phase-4 no-language sweep
pixi run python -m imitation_experiments.audit.validate_phase4_no_language_submission \
    <manifest> <vanilla_checkpoint> <latent_checkpoint> <skill_checkpoint> \
    <vanilla_audit> <latent_audit> <equivalence> \
    <expected_latent_dataset_path> <expected_vanilla_dataset_path> \
    <expected_manifest_sha256> [--expected_motion_count 40] \
    [--minimum_oracle_success 0.8] [--output_json <out.json>]
```

The `<equivalence>` argument is the streamed-versus-direct equivalence
certificate. It is mandatory for the focused interface comparison: the
streamed and direct vanilla paths must use the same ordered actor inputs and
the same frozen tracker weights, the restore must be strict, the module must
be frozen in evaluation mode, and the certificate must cover **all** actor
inputs and actions, phase-complete and asynchronous.

## 4. Comparison audit — the two main rows are actually comparable

```bash
pixi run python -m imitation_experiments.audit.audit_focused_causal_interface_comparison \
    --<latent|vanilla>_checkpoint ... --<...>_merge_manifest ... --<...>_summary ... \
    --direct_vanilla_summary ... --latent_oracle_summary ... --full_body_oracle_summary ... \
    --streamed_equivalence ... \
    --expected_seed ... --expected_num_envs ... --expected_history_steps ... \
    --expected_horizon_steps ... --expected_full_body_future_steps ... \
    --expected_planner_interval ... --expected_pretrain_updates ... \
    --expected_finetune_updates ... --expected_rows_per_stage ...
```

Both main rows must share the planner backbone, the training stages, the exact
positive sample budget, the optimizer budget, the seed, the evaluation start
range, and the low-level protocol.

## Protocol invariants the gates assume

- Planner inference uses **only** the causal robot history and the explicit
  task input: nine past frames plus the current one, 93 values per frame, so
  a `10 x 93` observation. Future reference data is allowed only for oracle
  commands, labels, and metrics. Never use
  `current_achieved_macro_transition_batch` as a deployable planner input.
- Planner collection and evaluation keep the 10-second, 500-control-step
  episode and the frozen random reference-start range 0–200 for both
  interfaces. The outer collector may continue across resets until it has the
  exact row count; the episode itself is not extended.
- The goal is published explicitly and per environment. Global timestep modulo
  logic is invalid, because environments reset asynchronously. Never choose or
  change the goal from a trajectory reassignment after a reset.
- Command-side expert noise stays disabled. Direct actor command terms and the
  matching critic command entries hold the same values; the critic may hold
  extra privileged state. That is not a noise difference.

## Related skills

`g1-encoder-interface` (interface and stride), `gr00t-planner` (the head and
its evaluation), `result-rigor` (whether the passing number may be cited),
`cluster-job-submission` (the submission itself).
