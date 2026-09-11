# Repository context

IsaacLab-Imitation orchestrates G1 humanoid imitation experiments. It owns
Isaac Lab environment wiring, task registration, experiment code, and cluster
submission. RLOpt owns algorithms; ImitationLearningTools owns reusable data tools.

## Terms

- G1: Unitree G1 humanoid; robot configuration and assets are owned here.
- Tracker / low-level policy: 50 Hz policy producing joint actions.
- Planner: model publishing commands from causal robot history and explicit
  task input. The main comparison publishes at 5 Hz.
- Reference: dataset motion used for rewards, terminations, and metrics.
- Oracle: frozen tracker driven by fresh expert commands at 50 Hz.
- Command interface: latent, explicit (vanilla), or chunk (packet).
- DiffSR: diffusion-based skill representation encoding reference windows.
- root_qpos: joint positions plus root pose, 38 values per frame.
- Macro state / transition: state at the planner's window timescale.
- Explicit packet: ten vanilla full-body commands, [580, 30, 60], 670 values.
- Causal planner observation: nine past frames plus current, 10 x 93 values.
- M3: planner evaluation with 10-second episodes, tracking-error terminations
  disabled, and base_too_low active. Survival excludes base_too_low;
  time_out and reference_finished are successful ends.
- Qualification: strict oracle gate before planner submission.
- Binding: proof that tracker and skill checkpoint contain identical encoder
  tensors and compatible interfaces.
- Equivalence certificate: proof that streamed and direct paths feed identical
  ordered inputs to identical frozen actors.
- Manifest: declared dataset motions; provenance includes its path and hash.
- LAFAN1 / Phase 4: no-language motion comparison.
- BONES-SEED / Phase 5: language-annotated G1 motion comparison.
  Selected-ten / language10 is the local development subset.
- SONIC: upstream whole-body tracking formulation and reference implementation.
- IPMD-L2T: privileged explicit teacher controls rollouts; latent student learns
  executed actions.
- JEPA: online encoder on both branches with deterministic token prediction
  and SIGReg, without EMA or stop-gradient. EMA-target prediction is the
  EMA trick, not JEPA.
- SIGReg: regularizer of projected token distributions toward standard normal.

Pixi manages environments. ICE and Skynet are Slurm clusters. A workspace
archive is the submitted source snapshot; a smoke test checks wiring only.
The SONIC-calibrated sonic_capability124_v1 subset is in-distribution and
not an unbiased holdout.

Read directory CONTEXT.md files only for relevant local concepts. Current
recipes, job state, and research decisions live in the code and wiki.
