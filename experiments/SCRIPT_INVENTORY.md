# Experiment script inventory

This is the complete classification of executable and Python source files under
`experiments/`, plus the `source/imitation_experiments/` package modules they
migrated into during the 2026-07-30 reorganization. A path belongs here only
while it has a current caller, a reproducible supporting-study role, or focused
test coverage. Removed paths are recorded in
[`PRUNED_SCRIPTS.md`](PRUNED_SCRIPTS.md).
`source/imitation_experiments/tests/test_script_inventory.py` fails when a row
goes stale or a new `experiments/` file is missing a row.

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
| `experiments/campaigns/2026-08-07-bones129k-latent-mode-stride5/run.sh` | guarded launcher | Smokes, validates, and submits three latent bottlenecks (deterministic, SONIC-FSQ64, group-Gumbel) at SONIC's stride-5 macro window, each with a dependent 5B G1-v2 controller on ICE. |
| `experiments/campaigns/2026-08-06-bones129k-skill-encoding/run.sh` | guarded launcher | Smokes, validates, and submits three endpoint-factorization ablations with dependent frozen-encoder G1-v2 low-level jobs on ICE. |
| `experiments/campaigns/2026-08-06-bones129k-skill-encoding/arms.sh` | library | Defines the occupancy, semi-Markov chain, and endpoint-delta encoder objectives shared by every campaign mode. |
| `experiments/campaigns/2026-08-04-bones129k-v2-adaptive-10b/run.sh` | guarded launcher | Validates the fresh full replay cache and accepted root+qpos encoder, then launches the local 32,768 × 6 low-level-from-scratch run with full-trajectory adaptive resets under a 10B cap. |
| `experiments/campaigns/2026-08-04-bones129k-v2-scale/run.sh` | guarded launcher | Runs the staged full-129,785-motion v2 DiffSR pretrain and completed local 32,768-environment 1B low-level scale probe. |
| `experiments/campaigns/2026-07-22-bones-h10-scale/submit.sh` | front door | Dated wrapper for the retained BONES h10 scale screen. |
| `experiments/campaigns/2026-07-29-sonic-official-fsq/run.sh` | front door | Dated wrapper for the official-window SONIC FSQ32 low-level campaign. |
| `experiments/campaigns/2026-07-29-sonic-official-fsq/sonic_official_fsq/submit_sonic_official_fsq_ice.sh` | guarded launcher | Validates corrected LAFAN1 inputs and submits one resumable ICE H200 segment under the 5B cap. |
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
| `source/imitation_experiments/imitation_experiments/capacity/enc380_capacity_grid.py` | library | Frozen walk1_subject1 × four sizes × three seeds definition and 0-11 array mapping. |
| `source/imitation_experiments/imitation_experiments/capacity/run_capacity_entry.py` | workflow | Python entry that dispatches legacy capacity cells and the fixed enc380 0-11 ICE grid. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_enc380_planner_route_comparison.sh` | workflow | Runs one stage/cell of the shared-tracker root+qpos-versus-latent capacity diagnostic. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_enc380_planner_route_ice.sh` | guarded launcher | Submits corrected qualification, one persistent 10-env/100-trajectory walk1 oracle collection, 12 capacity cells, and aggregation with dependencies. |
| `source/imitation_experiments/imitation_experiments/capacity/audit_enc380_motion_selection.py` | qualification | Binds the user-requested walk1_subject1 continuity diagnostic to corrected-manifest position 29 and labels it non-representative. |
| `source/imitation_experiments/imitation_experiments/capacity/audit_enc380_tracker_completion.py` | qualification | Binds the cross-segment ≥5B frame accounting, completed Slurm job, tracker hash, and encoder hash. |
| `source/imitation_experiments/imitation_experiments/capacity/audit_enc380_paired_demonstrations.py` | qualification | Verifies paired causal targets and exact completed `(env_id, episode_id)` segment counts while allowing variable rows per segment. |
| `source/imitation_experiments/imitation_experiments/capacity/materialize_paired_interface_samples.py` | workflow | Promotes root+qpos and latent targets from the same collected simulator rows. |
| `source/imitation_experiments/imitation_experiments/capacity/audit_packet_encoder_pin.py` | qualification | Certifies the explicit-packet-to-frozen-encoder route against oracle latent commands. |
| `source/imitation_experiments/imitation_experiments/capacity/aggregate_enc380_route_comparison.py` | audit/report | Aggregates walk1_subject1, four capacities, three seeds, one oracle-trained stage, and both routes. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/prepare_oracle_baselines.sh` | qualification | Prepares the matched oracle baselines the sweep normalizes against. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/paths.env` | library | Frozen checkpoint and data paths for the capacity sweep. |
| `source/imitation_experiments/imitation_experiments/capacity/aggregate_one_motion_capacity_scaling.py` | audit/report | Aggregates capacity points into the scaling curve. |
| `source/imitation_experiments/imitation_experiments/capacity/aggregate_one_motion_capacity_seeds.py` | audit/report | Aggregates repeated seeds at one capacity point. |
| `source/imitation_experiments/tests/test_aggregate_one_motion_capacity_scaling.py` | test | Scaling-curve aggregation coverage. |
| `source/imitation_experiments/tests/test_aggregate_one_motion_capacity_seeds.py` | test | Capacity seed-aggregation coverage. |

## Low-level command support

| Path | Class | Responsibility |
| --- | --- | --- |
| `source/imitation_experiments/imitation_experiments/lowlevel/evaluate_checkpoint.py` | qualification | Shared oracle evaluator used by current LAFAN1 and BONES gates. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/command_space_ablation/submit_cluster_oracle_ablation.sh` | guarded launcher | Shared low-level training submission used by current BONES workflows. |

## Paper comparison implementation

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/paper/aggregate_bones_seed_multiseed_results.py` | audit/report | Produces the fixed three-seed Phase-5 aggregate. |
| `experiments/paper/aggregate_phase4_no_language_results.py` | audit/report | Produces the complete Phase-4 aggregate. |
| `source/imitation_experiments/imitation_experiments/audit/audit_bones_seed_language_interface.py` | audit/report | Audits a Phase-5 language-conditioned interface run. |
| `source/imitation_experiments/imitation_experiments/audit/audit_bones_seed_multigoal_language_comparison.py` | audit/report | Audits paired multi-goal Phase-5 artifacts. |
| `source/imitation_experiments/imitation_experiments/audit/audit_diffsr_latent_qualification.py` | qualification | Checks the DiffSR low-level oracle gate. |
| `source/imitation_experiments/imitation_experiments/audit/audit_focused_causal_interface_comparison.py` | audit/report | Checks the focused two-row comparison. |
| `source/imitation_experiments/imitation_experiments/audit/audit_vanilla_tracker_qualification.py` | qualification | Checks direct and streamed vanilla qualification. |
| `source/imitation_experiments/imitation_experiments/data/balanced_motion_rows.py` | library | Enforces balanced per-motion sample selection. |
| `experiments/paper/build_paper_release_bundle.py` | audit/report | Builds the hash-verified Phase-4/5 release index. |
| `source/imitation_experiments/imitation_experiments/data/build_reference_arrays.py` | workflow | Builds training-shaped reference arrays from an NPZ tree, bypassing the Zarr and the persisted replay. |
| `source/imitation_experiments/imitation_experiments/data/publish_reference_arrays.py` | workflow | Validates, publishes, and retrieves a built reference-array directory via Hugging Face. |
| `source/imitation_experiments/imitation_experiments/evaluation/closed_loop_metrics.py` | library | Defines retained closed-loop paper metrics. |
| `source/imitation_experiments/imitation_experiments/data/collect_interface_rollout_samples.py` | workflow | Collects planner samples with the frozen causal protocol. |
| `source/imitation_experiments/imitation_experiments/evaluation/eval_interface_planner_closed_loop.py` | workflow | Evaluates planners in the Isaac closed loop. |
| `source/imitation_experiments/imitation_experiments/evaluation/eval_interface_planner_offline.py` | workflow | Runs retained offline planner diagnostics. |
| `source/imitation_experiments/imitation_experiments/planner/interface_planner_common.py` | library | Shared planner models, checkpoints, and data loading. |
| `source/imitation_experiments/imitation_experiments/lowlevel/low_level_tracker.py` | library | Loads and freezes matched low-level trackers. |
| `source/imitation_experiments/imitation_experiments/data/merge_planner_samples.py` | workflow | Merges exact-budget demonstration and rollout samples. |
| `source/imitation_experiments/imitation_experiments/provenance/paper_protocol_metadata.py` | library | Records and validates frozen protocol metadata. |
| `source/imitation_experiments/imitation_experiments/provenance/phase4_no_language_matrix.py` | library | Defines the fixed Phase-4 task grid. |
| `source/imitation_experiments/imitation_experiments/planner/planner_latency.py` | library | Measures planner-only publication latency. |
| `source/imitation_experiments/imitation_experiments/planner/planner_publish_schedule.py` | library | Implements per-environment asynchronous publication. |
| `source/imitation_experiments/imitation_experiments/data/planner_sample_schema.py` | library | Defines chunked planner sample storage. |
| `source/imitation_experiments/imitation_experiments/lowlevel/resolve_low_level_checkpoint.py` | qualification | Resolves content-specific low-level checkpoints. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_language_smoke.sh` | diagnostic | Tiny non-performance language wiring gate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_low_level_qualification.sh` | qualification | Paired Phase-5 low-level gate implementation. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_low_level_skynet.sh` | guarded launcher | Paired 1B-frame Phase-5 low-level candidate launcher. |
| `source/imitation_experiments/imitation_experiments/pipeline/run_bones_seed_multigoal_language_comparison.py` | workflow | Python stage driver for the Phase-5 multi-goal comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_bones_seed_multigoal_language_comparison.sh` | workflow | Shell front end for the shared multi-goal workflow. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_dance102_strong_interface_comparison.sh` | workflow | Explicit-packet engine reused by the focused comparison. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_focused_causal_interface_comparison.sh` | workflow | Canonical two-row comparison orchestrator. |
| `source/imitation_experiments/imitation_experiments/pipeline/run_interface_baseline_job.py` | library | Cluster dispatcher restricted to active workflows. |
| `source/imitation_experiments/imitation_experiments/pipeline/run_interface_baseline_job_impl.py` | library | Location-independent implementation behind the cluster dispatcher. |
| `source/imitation_experiments/imitation_experiments/paths.py` | library | Resolves the repository root and paper directory by marker, not by nesting depth. |
| `source/imitation_experiments/tests/conftest.py` | test | Puts `experiments/paper/` on `sys.path` for tests that span the paper boundary. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_lafan1_diffsr_low_level_skynet.sh` | guarded launcher | Corrected-LAFAN1 latent low-level prerequisite. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_lafan1_low_level_qualification.sh` | qualification | Paired corrected-LAFAN1 low-level gate. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_phase4_no_language_sweep.sh` | workflow | Executes one Phase-4 matrix task. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/run_shared_latent_interface_comparison.sh` | workflow | DiffSR latent row engine reused by the focused comparison. |
| `source/imitation_experiments/imitation_experiments/evaluation/smoke_test_causal_planner_env.py` | diagnostic | Minimal causal-planner environment wiring test. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_bones_seed_low_level_qualification_skynet.sh` | guarded launcher | Submits the fixed Phase-5 low-level qualification. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_bones_seed_multigoal_pipeline_skynet.sh` | guarded launcher | Submits one guarded Phase-5 seed dependency chain. |
| `experiments/paper/submit_bones_seed_multiseed_pipeline_skynet.sh` | guarded launcher | Paper-facing three-seed Phase-5 entrypoint. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_cluster_interface_baselines.sh` | guarded launcher | Generic cluster adapter requiring an explicit active mode. |
| `experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines/submit_lafan1_low_level_qualification_skynet.sh` | guarded launcher | Submits the fixed LAFAN1 low-level qualification. |
| `experiments/paper/submit_phase4_no_language_skynet.sh` | guarded launcher | Paper-facing Phase-4 entrypoint. |
| `source/imitation_experiments/imitation_experiments/evaluation/summarize_bones_seed_multigoal_language_comparison.py` | audit/report | Summarizes one paired Phase-5 seed. |
| `source/imitation_experiments/imitation_experiments/evaluation/summarize_interface_comparison.py` | audit/report | Summarizes focused interface runs. |
| `source/imitation_experiments/imitation_experiments/planner/train_chunked_transformer_planner.py` | workflow | Trains the shared retained planner families. |
| `source/imitation_experiments/imitation_experiments/audit/validate_bones_seed_planner_submission.py` | audit/report | Preflights exact Phase-5 data and gate provenance. |
| `source/imitation_experiments/imitation_experiments/audit/validate_latent_skill_checkpoint_binding.py` | qualification | Proves skill-encoder and low-level checkpoint identity. |
| `source/imitation_experiments/imitation_experiments/audit/validate_phase4_no_language_submission.py` | audit/report | Preflights the fixed Phase-4 submission. |
| `source/imitation_experiments/imitation_experiments/provenance/write_interface_run_provenance.py` | audit/report | Writes source, checkpoint, and data provenance. |
| `source/imitation_experiments/imitation_experiments/data/write_single_motion_manifest.py` | library | Creates explicit single-motion manifests for fixed tasks. |
| `source/imitation_experiments/imitation_experiments/data/write_motion_subset_manifest.py` | library | Creates N-motion subset manifests with recorded source-manifest hash provenance. |

## Tests for the paper comparison

| Path | Class | Responsibility |
| --- | --- | --- |
| `source/imitation_experiments/tests/test_aggregate_bones_seed_multiseed_results.py` | test | Phase-5 aggregation invariants. |
| `source/imitation_experiments/tests/test_audit_diffsr_latent_qualification.py` | test | DiffSR qualification audit coverage. |
| `source/imitation_experiments/tests/test_audit_vanilla_tracker_qualification.py` | test | Vanilla qualification audit coverage. |
| `source/imitation_experiments/tests/test_balanced_motion_rows.py` | test | Balanced-row selection coverage. |
| `source/imitation_experiments/tests/test_bones_seed_multigoal_stages.py` | test | Multi-goal stage contract coverage. |
| `source/imitation_experiments/tests/test_bones_seed_multiseed_submission.py` | test | Three-seed preflight coverage. |
| `source/imitation_experiments/tests/test_bones_seed_slurm_pipeline.py` | test | Phase-5 Slurm-chain coverage. |
| `source/imitation_experiments/tests/test_build_paper_release_bundle.py` | test | Release hash-chain coverage. |
| `source/imitation_experiments/tests/test_cluster_slurm_dependency.py` | test | Scheduler dependency parsing coverage. |
| `source/imitation_experiments/tests/test_continuous_planner_families.py` | test | Retained planner-family coverage. |
| `source/imitation_experiments/tests/test_low_level_tracker.py` | test | Frozen tracker loading coverage. |
| `source/imitation_experiments/tests/test_paper_protocol_metadata.py` | test | Frozen metadata coverage. |
| `source/imitation_experiments/tests/test_phase4_no_language.py` | test | Phase-4 grid and launcher coverage. |
| `source/imitation_experiments/tests/test_planner_latency.py` | test | Latency instrumentation coverage. |
| `source/imitation_experiments/tests/test_planner_publish_schedule.py` | test | Asynchronous renewal coverage. |
| `source/imitation_experiments/tests/test_planner_sample_schema.py` | test | Current sample-schema and language merge coverage. |
| `source/imitation_experiments/tests/test_resolve_low_level_checkpoint.py` | test | Checkpoint resolution coverage. |
| `source/imitation_experiments/tests/test_summarize_bones_seed_multigoal_language_comparison.py` | test | Phase-5 summary coverage. |
| `source/imitation_experiments/tests/test_validate_bones_seed_planner_submission.py` | test | Phase-5 preflight coverage. |
| `source/imitation_experiments/tests/test_validate_latent_skill_checkpoint_binding.py` | test | Encoder-binding validator coverage. |

## Rows added in the 2026-07-30 reorganization audit

These files predate the reorganization but had no inventory row; the coverage
test now enforces that every `experiments/` script is classified.

| Path | Class | Responsibility |
| --- | --- | --- |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/qualify_interface.sh` | qualification | Collects oracle qualification rollouts for one reduced command interface. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_bb1_shared_tracker_sweep.sh` | supporting study | BB1 shared-tracker mechanism sweep behind the 3.19x gap finding. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_bb3_noise_curves.sh` | supporting study | BB3 command-noise tolerance curves for both interfaces. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_finetune_method_b.sh` | supporting study | Method-B rollout finetuning arm of the capacity study. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_interface_ablation.sh` | supporting study | Reduced-interface ablation driver over the frozen streaming protocol. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_oracle_substitution_ladder.sh` | supporting study | Oracle-substitution ladder isolating where planner error enters. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_enc380_latent_low_level_ice.sh` | guarded launcher | Submits the enc380 latent low-level prerequisite to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_reduced_interface_low_level_ice.sh` | guarded launcher | Submits reduced-interface low-level trackers to ICE. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/run.sh` | front door | Dated wrapper for the GroupVQ capacity ablation. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/build_lafan1_cache_ice.sh` | workflow | Builds the content-addressed LAFAN1 cache on ICE for the GroupVQ grid. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/check_groupvq_encoder_grid.py` | audit/report | Validates the GroupVQ encoder grid outputs. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/groupvq_grid.sh` | workflow | Runs one GroupVQ grid cell. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/run_local_10m_qualification.sh` | qualification | Bounded local 10M gate for the GroupVQ arms. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/submit_groupvq_capacity_ablation_ice.sh` | guarded launcher | Submits the GroupVQ capacity grid to ICE. |
| `experiments/campaigns/2026-07-26-groupvq-capacity-ablation/groupvq_ablation/training_profile.h100.coe.env` | library | H100 COE training profile consumed by the GroupVQ launcher. |
| `experiments/campaigns/2026-07-27-sonic-env-latent-det/run.sh` | front door | Dated wrapper for the SONIC-env latent determinism campaign. |
| `experiments/campaigns/2026-07-27-sonic-env-latent-det/sonic_env_det/evaluate_stable_converged_checkpoint.sh` | qualification | Evaluates the stable converged SONIC-env checkpoint. |
| `experiments/campaigns/2026-07-27-sonic-env-latent-det/sonic_env_det/run_local_wiring_gate.sh` | qualification | Local wiring gate before SONIC-env submissions. |
| `experiments/campaigns/2026-07-27-sonic-env-latent-det/sonic_env_det/submit_latent_v0_reset_sampling_ice.sh` | guarded launcher | Submits the latent-v0 reset-sampling arm to ICE. |
| `experiments/campaigns/2026-07-27-sonic-env-latent-det/sonic_env_det/submit_sonic_env_latent_det_ice.sh` | guarded launcher | Submits the SONIC-env latent determinism arm to ICE. |
| `experiments/campaigns/2026-07-29-latent-holdout-horizon/latent_holdout/submit_latent_holdout_horizon_ice.sh` | guarded launcher | Submits the hold-in-{5,1} horizon ablation to ICE. |
| `experiments/paper/pipeline/pretrain_latent_encoder.py` | workflow | Hydra stage: pretrain the latent encoder for the paper pipeline. |
| `experiments/paper/pipeline/train_low_level.py` | workflow | Hydra stage: train a low-level tracker for the paper pipeline. |
| `experiments/paper/reference_buffer_workflow.py` | workflow | Reference-buffer workflow; the reference implementation of the paper-script standard. |
| `experiments/paper/run_enc380_planner_route_comparison.py` | workflow | Hydra driver for the enc380 planner-route comparison. |
| `experiments/paper/run_interface_capacity_study.py` | workflow | Hydra driver for the interface capacity study. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_enc380_h30_temporal_ensemble.sh` | workflow | Runs the H30 temporal-ensemble diagnostic over reused enc380 H10 rows. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_enc380_h30_temporal_ensemble_ice.sh` | guarded launcher | Submits the H30 temporal-ensemble diagnostic to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/resubmit_enc380_root_route_ice.sh` | guarded launcher | Resubmits the enc380 root+qpos route cells after a partial run. |
| `experiments/campaigns/2026-07-30-det-latent-e2e/run.sh` | front door | Dated pin of the deterministic-latent end-to-end chain; wraps `imitation_experiments.pipeline.run_latent_e2e` with the campaign config. |
| `experiments/campaigns/2026-08-05-bones-language10-oracle-pretrain/run.sh` | front door | Prepares the frozen selected-ten language data, collects complete oracle-policy trajectories, pretrains the shared planner, and evaluates 2k-update milestones. |
| `experiments/campaigns/2026-08-06-bones-language10-latent-receding/run.sh` | front door | Materializes matched three-token latent targets, trains the two H3 planners, evaluates the fixed seven-row overlap grid, and audits/aggregates the result. |
| `experiments/campaigns/2026-08-06-bones-latent-compositionality/run.sh` | front door | Reproduces the selected-30 collection, semantic phases, cross-motion retrieval controls, phase-level clustering and trajectory traversal, 500-family reference scale test, and median-query neighbor galleries. |
| `source/imitation_experiments/imitation_experiments/pipeline/run_latent_e2e.py` | workflow | Config-driven conductor for the latent chain: pretrain, low-level, binding gate, collect, merge, planner training, offline and closed-loop eval. |
| `source/imitation_experiments/imitation_experiments/data/prepare_language_motion_selection.py` | library | Builds the ordered selected-motion manifest, canonical language sidecar, and provenance record. |
| `source/imitation_experiments/imitation_experiments/planner/materialize_latent_receding_horizon.py` | library | Reuses complete-trajectory samples to build ordered H3 latent targets in future-publication or current-publication frames. |
| `source/imitation_experiments/imitation_experiments/planner/latent_receding_horizon.py` | library | Executes overlapping H3 predictions with fresh-only, exponential, or clipped/gated latent fusion on a per-environment renewal schedule. |
| `source/imitation_experiments/imitation_experiments/evaluation/aggregate_language_latent_receding.py` | audit/report | Enforces and ranks the fixed selected-ten seven-row latent receding-horizon evaluation grid. |
| `source/imitation_experiments/imitation_experiments/pipeline/run_language_planner_oracle_pretrain.py` | workflow | Runs trajectory-first selected-ten collection, oracle-only planner pretraining, milestone evaluation, and plateau aggregation. |
| `source/imitation_experiments/imitation_experiments/evaluation/analyze_collected_latent_space.py` | audit/report | Visualizes collected latent commands and runs phase-local and leave-one-motion-out semantic probes. |
| `source/imitation_experiments/imitation_experiments/evaluation/analyze_cross_motion_latent_structure.py` | audit/report | Aggregates randomized replicas and measures cross-motion latent-to-kinematic retrieval, semantic transfer, and clustering without using t-SNE for metrics. |
| `source/imitation_experiments/imitation_experiments/evaluation/analyze_semantic_latent_trajectories.py` | audit/report | Tests frozen shared semantic regions, leave-one-motion-out phase classification, unsupervised phase clustering, and time-ordered leave-and-return trajectories in latent PCA space. |
| `source/imitation_experiments/imitation_experiments/evaluation/analyze_reference_latent_scale.py` | audit/report | Encodes canonical root-qpos windows from distinct BONES action families for the large reference-only locality control. |
| `source/imitation_experiments/imitation_experiments/evaluation/build_semantic_phase_annotations.py` | library | Joins BONES temporal descriptions with complete manually curated semantic trait rows at exact 50 Hz steps. |
| `source/imitation_experiments/imitation_experiments/evaluation/segment_semantic_phase_videos.py` | workflow | Cuts full-horizon comparison videos into frame-accurate semantic phase clips. |
| `source/imitation_experiments/imitation_experiments/evaluation/build_latent_neighbor_gallery.py` | audit/report | Builds median-performance, distinct-motion video and contact-sheet galleries for cross-motion latent neighbors. |
| `source/imitation_experiments/imitation_experiments/capacity/aggregate_frame0_dr_baseonly_results.py` | audit/report | Aggregates the frame0 domain-randomization base-only evaluation results. |
| `source/imitation_experiments/imitation_experiments/capacity/aggregate_planner_budget_curve.py` | audit/report | Aggregates planner sample-budget curves across budget points and seeds. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_fb670_budget_curve.sh` | supporting study | Runs the FB670 explicit-packet planner sample-budget curve. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_fb670_via_latent_tracker_curve.sh` | supporting study | Runs the FB670 budget curve executed through the latent tracker route. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_latent_budget_curve.sh` | supporting study | Runs the latent-interface planner sample-budget curve. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_frame0_dr_baseonly_evaluation.sh` | supporting study | Frame0 evaluation isolating domain randomization to the base only. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/run_pure_root_qpos_planner.sh` | supporting study | Trains/evaluates the pure root+qpos planner variant. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_fb670_budget_curve_ice.sh` | guarded launcher | Submits the FB670 budget curve to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_fb670_via_latent_tracker_ice.sh` | guarded launcher | Submits the FB670-via-latent-tracker curve to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_latent_budget_curve_ice.sh` | guarded launcher | Submits the latent budget curve to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_pure_root_qpos_planner_ice.sh` | guarded launcher | Submits the pure root+qpos planner variant to ICE. |
| `experiments/campaigns/2026-07-23-lafan1-planner-capacity/submit_auxiliary_capacity_seed0_ice.sh` | guarded launcher | Submits the auxiliary seed-0 capacity cells to ICE. |
