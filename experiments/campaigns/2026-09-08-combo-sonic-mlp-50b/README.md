# SONIC-sized combo actor with a DAgger warm start

User-selected schedule: 200M frames of teacher-controlled supervised
imitation, then exponential teacher-probability decay from 1 to 0.01 over
200–400M, zero thereafter. Supervised distillation continues through 400M;
PPO starts only after that stage succeeds and gets its own full 50B budget.
Rates are evaluated at rollout boundaries. At 4096 x 24 the warm-start cap
rounds to 400,097,280 frames; PPO at 16384 x 24 rounds to 50,000,166,912.

The frozen teacher is combo's model_step_50000166912.pt. Student and teacher
consume the exact same p5_affine encoder weights, z64 + two phase channels,
and flat ten-step proprioception history (996 actor inputs). The student
MLP is [4096,4096,2048,2048,1024,1024,512,512], SiLU, 29 action means.
Only the actor grows. PPO retains the original six-layer critic, EE/wide
rewards, action-rate -0.03, Newton physics, full-batch three-epoch optimizer,
and original reset recipe. This is a capacity plus warm-start experiment,
not a single-variable capacity ablation or a SONIC reproduction.

DAgger uses a frozen teacher mean as the label on every visited state.
Each environment step chooses teacher versus student action by a Bernoulli
coin at the current teacher probability. The student receives labels even
when it controls the environment. The teacher normalizer and log_std are
copied to the student; both remain fixed during distillation. Only the mean
network trains, with Adam 1e-4, gradient clip 1, twelve 8192-row updates per
98304-frame rollout. A bounded 262144-row FIFO replay aggregates recent
rollouts; it is not a reservoir over all historical states. Distillation
uses the final combo reset mix (20% uniform), without a termination ramp.
Log training MSE, pre-update rollout MSE, teacher fraction, and frame count.

The IPMDDagger subclass reuses IPMD latent injection and the shared PPO
collect/record/checkpoint lifecycle, following L2T's detached action-MSE
pattern. It does not modify IPMDL2T's historical contract. Resume snapshots
retain student weights, distillation optimizer, replay, RNG and cumulative
warm-start frames. Physics/episode state restarts on resume. Exact teacher
encoder tensors and the teacher checkpoint hash are checked. A successful
warm-start exit atomically publishes initialization.pt with PPO frames zero,
actor plus encoder only, and no distillation optimizer. PPO critic and
optimizer start fresh. Existing 5B smoothness fine-tunes are independent.

The H200 chain starts with three full-size PPO rollout/updates as a memory
qualification, on a separate output path (1,179,648 qualification frames,
not used for initialization). Distillation is gated afterok on this job.
Two 15:59 DAgger segments use afterany, with PPO gated afterok on the second.
Sixteen 15:59 PPO segments continue from persistent checkpoints via afterany;
all carry the full 50B target and completed runs perform zero extra updates.
One H200, 16 CPUs, 160G host RAM per job. W&B has separate DAgger and PPO
identities. Output root: /data/combo_sonic_mlp_50b/sonic_mlp_seed0.
The original combo reset ramps are segment-local; changed throughput and
resume boundaries can shift their global-frame timing. Do not present the
comparison as matching a global reset schedule.

Validation: focused DAgger tests cover the exponential schedule, labels under
student control, frozen teacher/critic, optimizer/replay resume, and a clean
zero-frame PPO handoff; historical L2T tests pass. A full-sized local Newton
smoke exercises latent injection, strict encoder binding, mixed collection,
distillation and export. A second Newton smoke strictly loads that export
and completes three PPO updates from frame zero. H200 full-batch memory is gated by qualification.
Local audit and smoke evidence: logs/combo_actor_audit_20260908/.

The accompanying actor audit used 1400 captured states from six deterministic
PhysX clips and Gaussian actions resampled offline from their recorded
loc/scale. Reconstructed means differed by at most 3.81e-6; log probabilities
by 1.45e-4. This is not an archived PPO rollout replay. Three simulated
full-batch statistic updates changed the maximum ratio by 8.39e-4 at the
mature checkpoint. Sensitivities are in audit.json, in joint-target radians
for 0.1% and 1% feature-standard-deviation perturbations. They do not establish
that model capacity or normalization causes visible shaking.

## Submission verified, 2026-09-08 20:21 UTC

Qualification 5738762; DAgger 5738763 -> 5738764; PPO 5738765 through
5738780 (sixteen chained segments). All jobs were PENDING at the first
control-plane status check. Full H200 memory qualification has not run yet.

Plan: combo-sonic-mlp-50b-sonic_mlp-s0-20260908-201938-c498432a.
Plan SHA: c498432af2979f498e315b5cd8a7401c70395fed832a501297bf88665282e727.
Workspace SHA: 95f4a9ec4b76c0b8e7ceeb222e7760b0b74cdeded43143f57d44c04afda50a67.
The archive includes the uncommitted RLOpt implementation and its config,
plus the training-entrypoint integration. Their archived contents and the
campaign YAML were checked against the tested working-tree files.
Twelve focused DAgger/L2T tests passed. Local Newton distillation and PPO
handoff smokes both completed successfully; these establish wiring only.

## Memory repair r1 (2026-09-08)

Qualification 5738762 failed CUDA OOM in Newton/Warp graph launch during
collection, after TorchRL replay storage initialized. The log does not identify
which rollout failed, so it is not evidence that PPO learning never ran.
The GPU was empty at startup. The blocked original chain was canceled under
the user's fix-and-resubmit instruction; no DAgger/PPO training had started.

The replacement is campaign_memory_r1.yaml. PPO uses 8192 environments and
196608 frames per rollout, full-batch three epochs, with 254314 total iterations.
The total PPO cap remains 50000166912. DAgger remains 4096 environments and
400097280 frames, with the same 200–400M teacher-probability schedule.
The macro cache moves to CPU for every stage; runtime cache was already CPU.
These changes reduce GPU demand; H200 qualification is still required.
Qualification runs six rollouts, covering repeated collection/learning transitions.
Output root is /data/combo_sonic_mlp_50b_memr1/sonic_mlp_seed0 and W&B IDs
carry memr1. All 19 replacement stages passed remote preflight and resolved
budget/dependency checks. The original YAML and its frozen plan remain intact.

Replacement submission verified: qualification **5738999**, DAgger **5739000/5739001**,
PPO **5739002** through final segment **5739026** (IDs are noncontiguous;
use the submission record for the exact list). Plan SHA:
3b5b997594fd2c6248435c73bfc0eab7c6c4621c553d5296c9f28f0bf3673ada.
Workspace SHA: 81c950ec8289be74e7aa27c60113bd61149c678f2534b26fa5e89601a9e438e2.
The session-bound hourly monitor follows this replacement. For a persistent
app scheduled task, use [the prepared prompt](hourly_monitor_prompt.md);
creating that app schedule still requires the Scheduled UI because the
automation tool is unavailable in this task.

H200 repair qualification verified completed: job 5738999, six rollouts,
1,179,648 frames, exit 0:0, walltime 00:06:42. The log reports 34.02 seconds
of training and final reported throughput 49,252 frames/s. This establishes
full-batch memory qualification, not convergence. DAgger job 5739000 then
started; the remaining stages retain their recorded dependencies.
