# Experiment script inventory

This is the complete classification of executable and Python source files under
`experiments/` after the 2026-07-23 cleanup. A path belongs here only while it
has a current caller, a reproducible supporting-study role, or focused test
coverage. Removed paths are recorded in [`PRUNED_SCRIPTS.md`](PRUNED_SCRIPTS.md).

Classes:

- **front door**: collaborator-facing campaign or release entrypoint;
- **guarded launcher**: validates prerequisites before scheduler mutation;
- **workflow**: canonical multi-stage experiment implementation;
- **qualification**: low-level evaluation or pass/fail gate;
- **library**: imported shared implementation, not a direct entrypoint;
- **audit/report**: validates or summarizes immutable artifacts;
- **diagnostic**: bounded smoke or checkpoint inspection helper;
- **supporting study**: retained reproducible experiment outside the main paper grid;
- **test**: focused automated coverage for retained code.

## Campaign, release, and supporting-study surfaces

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/campaigns/2026-07-22-bones-h10-scale/submit.sh` | front door | Dated wrapper for the retained BONES h10 scale screen. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/run.sh` | front door | Dated wrapper for the current latent-learning ablation. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit.sh` | front door | Preliminary latent-only Phase-5 pilot; fail-closed and excluded from paper aggregation. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/run.sh` | front door | Local latent-only ten-goal Phase-5 planner baseline: demonstration-pretrained plus rollout-finetuned shared planner, no scheduler. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_videos.sh` | diagnostic | Renders per-goal non-terminating closed-loop videos of the shared local10 planner. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_reference_comparison.sh` | diagnostic | Renders per-goal side-by-side reference-vs-planner videos (full 500-step horizon) for the local10 run. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/run_data_ablation_bc.sh` | supporting study | LOCAL finetune-data composition ablation arms B/C, reusing arm A's shared pools (superseded by the ICE launcher). |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/submit_ablation_arm_a_ice.sh` | guarded launcher | ICE submission for ablation arm A (DAgger+demo), large planner, via the repaired cluster pipeline. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/submit_ablation_ice.sh` | guarded launcher | Parameterized ICE submission for ablation arms A/B/C (ARM= selects data composition via the new pipeline flags). |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/render_ice_arm_videos.sh` | diagnostic | Pulls a finished ICE arm's planner back and renders per-goal reference-vs-planner videos locally. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/analyze_local_qualification.py` | audit/report | Validates all local ablation qualification arms. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/run_lafan1_local_10m_qualification.sh` | qualification | Runs the bounded local LAFAN1 gate. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_all_h200_after_local_qualification.sh` | guarded launcher | Submits all approved H200 arms only after a complete local gate. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_lafan1_diffsr_bottleneck_ablation_ice.sh` | guarded launcher | Submits the DiffSR bottleneck study. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/submit_lafan1_reconstruction_ablation_ice.sh` | guarded launcher | Submits the reconstruction-family study. |
| `experiments/paper/run.sh` | front door | Staging public reproduction surface; fails closed until release readiness. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit.sh` | front door | Dated wrapper for the preliminary latent-only Phase-5 H200 pilot. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_impl.sh` | guarded launcher | Implementation behind the H200 pilot wrapper. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_bones_seed_h10_gpu_lr_ablation_ice.sh` | supporting study | Canonical h10 GPU, rollout, and learning-rate screen. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-h200/submit_hl_skill_pipeline_pace_2b.sh` | workflow | Shared high-level skill pipeline used by retained launchers. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.example.env` | library | Template training profile for the ablation launchers. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.approved.env` | library | Approved H200 training profile consumed by the gated submission. |
| `experiments/campaigns/2026-07-22-latent-learning-ablation/latent_ablation/training_profile.h200.pending.env` | library | Pending H200 profile retained until the gate approves it. |

## Planner-capacity study

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_sweep.sh` | front door | Dated wrapper for the secondary planner-scaling study. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_point.sh` | workflow | Runs one capacity point of the scaling sweep. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_capacity_entry.py` | workflow | Python entry that dispatches a capacity point and aggregates it. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh` | qualification | Prepares the matched oracle baselines the sweep normalizes against. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/paths.env` | library | Frozen checkpoint and data paths for the capacity sweep. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_one_motion_capacity_scaling.py` | audit/report | Aggregates capacity points into the scaling curve. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/aggregate_one_motion_capacity_seeds.py` | audit/report | Aggregates repeated seeds at one capacity point. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/test_aggregate_one_motion_capacity_scaling.py` | test | Scaling-curve aggregation coverage. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/interface_baselines/test_aggregate_one_motion_capacity_seeds.py` | test | Capacity seed-aggregation coverage. |

## Low-level command support

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/command_space_ablation/evaluate_checkpoint.py` | qualification | Shared oracle evaluator used by current LAFAN1 and BONES gates. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/command_space_ablation/submit_cluster_oracle_ablation.sh` | guarded launcher | Shared low-level training submission used by current BONES workflows. |

## Paper comparison implementation

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/paper/aggregate_bones_seed_multiseed_results.py` | audit/report | Produces the fixed three-seed Phase-5 aggregate. |
| `experiments/paper/aggregate_phase4_no_language_results.py` | audit/report | Produces the complete Phase-4 aggregate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_bones_seed_language_interface.py` | audit/report | Audits a Phase-5 language-conditioned interface run. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_bones_seed_multigoal_language_comparison.py` | audit/report | Audits paired multi-goal Phase-5 artifacts. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_diffsr_latent_qualification.py` | qualification | Checks the DiffSR low-level oracle gate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_focused_causal_interface_comparison.py` | audit/report | Checks the focused two-row comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/audit_vanilla_tracker_qualification.py` | qualification | Checks direct and streamed vanilla qualification. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/balanced_motion_rows.py` | library | Enforces balanced per-motion sample selection. |
| `experiments/paper/build_paper_release_bundle.py` | audit/report | Builds the hash-verified Phase-4/5 release index. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/closed_loop_metrics.py` | library | Defines retained closed-loop paper metrics. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/collect_interface_rollout_samples.py` | workflow | Collects planner samples with the frozen causal protocol. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/eval_interface_planner_closed_loop.py` | workflow | Evaluates planners in the Isaac closed loop. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/eval_interface_planner_offline.py` | workflow | Runs retained offline planner diagnostics. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/interface_planner_common.py` | library | Shared planner models, checkpoints, and data loading. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/low_level_tracker.py` | library | Loads and freezes matched low-level trackers. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/merge_planner_samples.py` | workflow | Merges exact-budget demonstration and rollout samples. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/paper_protocol_metadata.py` | library | Records and validates frozen protocol metadata. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/phase4_no_language_matrix.py` | library | Defines the fixed Phase-4 task grid. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_latency.py` | library | Measures planner-only publication latency. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_publish_schedule.py` | library | Implements per-environment asynchronous publication. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/planner_sample_schema.py` | library | Defines chunked planner sample storage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/resolve_low_level_checkpoint.py` | qualification | Resolves content-specific low-level checkpoints. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_language_smoke.sh` | diagnostic | Tiny non-performance language wiring gate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_low_level_qualification.sh` | qualification | Paired Phase-5 low-level gate implementation. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_low_level_skynet.sh` | guarded launcher | Paired 1B-frame Phase-5 low-level candidate launcher. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_multigoal_language_comparison.py` | workflow | Python stage driver for the Phase-5 multi-goal comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_multigoal_language_comparison.sh` | workflow | Shell front end for the shared multi-goal workflow. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_dance102_strong_interface_comparison.sh` | workflow | Explicit-packet engine reused by the focused comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_focused_causal_interface_comparison.sh` | workflow | Canonical two-row comparison orchestrator. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_interface_baseline_job.py` | library | Cluster dispatcher restricted to active workflows. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_interface_baseline_job_impl.py` | library | Location-independent implementation behind the cluster dispatcher. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/_repo_paths.py` | library | Resolves the repository root and paper directory by marker, not by nesting depth. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/conftest.py` | test | Puts `experiments/paper/` on `sys.path` for tests that span the paper boundary. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_lafan1_diffsr_low_level_skynet.sh` | guarded launcher | Corrected-LAFAN1 latent low-level prerequisite. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_lafan1_low_level_qualification.sh` | qualification | Paired corrected-LAFAN1 low-level gate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_phase4_no_language_sweep.sh` | workflow | Executes one Phase-4 matrix task. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_shared_latent_interface_comparison.sh` | workflow | DiffSR latent row engine reused by the focused comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/smoke_test_causal_planner_env.py` | diagnostic | Minimal causal-planner environment wiring test. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_bones_seed_low_level_qualification_skynet.sh` | guarded launcher | Submits the fixed Phase-5 low-level qualification. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_bones_seed_multigoal_pipeline_skynet.sh` | guarded launcher | Submits one guarded Phase-5 seed dependency chain. |
| `experiments/paper/submit_bones_seed_multiseed_pipeline_skynet.sh` | guarded launcher | Paper-facing three-seed Phase-5 entrypoint. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_cluster_interface_baselines.sh` | guarded launcher | Generic cluster adapter requiring an explicit active mode. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_lafan1_low_level_qualification_skynet.sh` | guarded launcher | Submits the fixed LAFAN1 low-level qualification. |
| `experiments/paper/submit_phase4_no_language_skynet.sh` | guarded launcher | Paper-facing Phase-4 entrypoint. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/summarize_bones_seed_multigoal_language_comparison.py` | audit/report | Summarizes one paired Phase-5 seed. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/summarize_interface_comparison.py` | audit/report | Summarizes focused interface runs. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/train_chunked_transformer_planner.py` | workflow | Trains the shared retained planner families. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/validate_bones_seed_planner_submission.py` | audit/report | Preflights exact Phase-5 data and gate provenance. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/validate_latent_skill_checkpoint_binding.py` | qualification | Proves skill-encoder and low-level checkpoint identity. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/validate_phase4_no_language_submission.py` | audit/report | Preflights the fixed Phase-4 submission. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/write_interface_run_provenance.py` | audit/report | Writes source, checkpoint, and data provenance. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/write_single_motion_manifest.py` | library | Creates explicit single-motion manifests for fixed tasks. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/write_motion_subset_manifest.py` | library | Creates N-motion subset manifests with recorded source-manifest hash provenance. |

## Tests for the paper comparison

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_aggregate_bones_seed_multiseed_results.py` | test | Phase-5 aggregation invariants. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_audit_diffsr_latent_qualification.py` | test | DiffSR qualification audit coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_audit_vanilla_tracker_qualification.py` | test | Vanilla qualification audit coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_balanced_motion_rows.py` | test | Balanced-row selection coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_bones_seed_multigoal_stages.py` | test | Multi-goal stage contract coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_bones_seed_multiseed_submission.py` | test | Three-seed preflight coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_bones_seed_slurm_pipeline.py` | test | Phase-5 Slurm-chain coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_build_paper_release_bundle.py` | test | Release hash-chain coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_cluster_slurm_dependency.py` | test | Scheduler dependency parsing coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_continuous_planner_families.py` | test | Retained planner-family coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_low_level_tracker.py` | test | Frozen tracker loading coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_paper_protocol_metadata.py` | test | Frozen metadata coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_phase4_no_language.py` | test | Phase-4 grid and launcher coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_planner_latency.py` | test | Latency instrumentation coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_planner_publish_schedule.py` | test | Asynchronous renewal coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_planner_sample_schema.py` | test | Current sample-schema and language merge coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_resolve_low_level_checkpoint.py` | test | Checkpoint resolution coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_summarize_bones_seed_multigoal_language_comparison.py` | test | Phase-5 summary coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_validate_bones_seed_planner_submission.py` | test | Phase-5 preflight coverage. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/test_validate_latent_skill_checkpoint_binding.py` | test | Encoder-binding validator coverage. |
