# IPMD Representation Learning

This repo is currently being used to study representation learning for
IPMD-family inverse RL on G1 imitation tasks.

The working direction is not just "pretrain a decoder". The core goal is to
learn a useful state representation `f(s)` from expert state trajectories and
then use that representation inside IPMD-style inverse RL / adversarial reward
learning. Offline learning should warm-start the representation, and online
training should keep adapting it instead of freezing the encoder permanently by
default.

## Research Target

The target stack is:

1. Expert state data provides motion/reference trajectories.
2. An offline stage learns representation and reward-relevant structure from
   those trajectories.
3. Online IPMD uses environment rollouts plus expert batches to refine the
   reward/representation and train the policy.
4. Variants such as bilinear IPMD and VQ-VAE/FSQ latent IPMD test different
   representation parameterizations.

The key point is that the expert data is mostly state trajectory data. Do not
assume action labels exist. Any offline objective that needs expert actions
must first prove where those labels come from.

## Ownership Boundary

`IsaacLab-Imitation` owns:

- G1 task registration.
- Env/config surfaces for vanilla, latent, VQ-VAE, and bilinear experiments.
- Expert batch sampling from trajectory data.
- Dataset manifests and Zarr-cache routing.
- Local smoke scripts and cluster submission scripts.
- Experiment context and command documentation.

`./RLOpt` owns:

- IPMD algorithm logic.
- Reward estimator architecture and update cadence.
- Latent learner implementation and checkpointing.
- Bilinear representation model internals.
- Offline pretraining implementation details.

Do not put algorithm workarounds into `scripts/rlopt/train.py` when the behavior
belongs in `RLOpt/rlopt/agent/ipmd/`.

## Current Env-Owned Surfaces

The important env-side surfaces in this repo are in
`source/isaaclab_imitation/isaaclab_imitation/envs/imitation_rl_env.py`:

- `sample_expert_batch(...)`: expert batch API consumed by imitation algorithms.
- `_sample_expert_batch_impl(...)`: maps requested nested keys into expert
  tensors.
- `_sample_expert_trajectory_batch(...)`: draws random expert transitions
  without advancing live env state.
- `_build_reward_input_cache(...)` and `_reward_input_expert_terms(...)`:
  pre-materialize expert-side `reward_input` values.
- `_map_requested_expert_observations(...)`: maps requested `policy`, `critic`,
  `reward_input`, `expert_state`, and `expert_window` keys.
- `_sample_expert_window_slice(...)` and `_build_expert_window_terms(...)`:
  expose temporal windows for latent encoders and codebooks.

For representation learning, `sample_expert_batch(...)` is the main bridge from
Isaac Lab trajectory data to RLOpt offline/IRL objectives.

## Task Surfaces

Current task surfaces:

- `Isaac-Imitation-G1-v0`: vanilla G1 tracking.
- `Isaac-Imitation-G1-LafanTrack-v0`: legacy alias for vanilla tracking.
- `Isaac-Imitation-G1-Latent-v0`: latent-conditioned task for ASE/IPMD-style
  latent command experiments.
- `Isaac-Imitation-G1-Latent-VQVAE-v0`: latent VQ-VAE/FSQ skill-codebook task.

RLOpt algorithm selection is done through `--algo` in `scripts/rlopt/train.py`.
For IPMD-family work, the important values are:

- `IPMD`: base IPMD path.
- `IPMD_BILINEAR`: IPMD plus bilinear representation/reward variant.
  Use this with `Isaac-Imitation-G1-Latent-v0` and
  `ipmd.use_latent_command=True` unless the user explicitly asks for a vanilla
  debug run. The vanilla/non-latent-command bilinear path is not a trusted
  comparison surface until it is explicitly fixed and revalidated.
- `IPMD_SR`: supported by the registry, but not the current main focus.

The task registry maps these through `rlopt_<algo>_cfg_entry_point` entries
under `source/isaaclab_imitation/.../config/g1/__init__.py`.

## Current Config Surfaces

Key config files:

- `rlopt_ipmd_cfg.py`: base vanilla/latent IPMD config.
- `rlopt_ipmd_bilinear_cfg.py`: bilinear IPMD config and offline pretrain
  toggles.
- `rlopt_ipmd_vqvae_cfg.py`: latent VQ-VAE/FSQ skill-codebook config.
- `imitation_g1_env_cfg.py`: vanilla env config and reward/expert observation
  groups.
- `imitation_g1_latent_env_cfg.py`: latent-command env config.
- `imitation_g1_latent_vqvae_env_cfg.py`: latent VQ-VAE env window config.

Important config concepts:

- `REWARD_INPUT_KEYS` define the expert/rollout reward estimator input surface.
- `LATENT_POSTERIOR_INPUT_KEYS` define what the posterior encoder consumes.
- `LATENT_POLICY_INPUT_KEYS` define what the policy sees when latent commands
  are enabled.
- `bilinear.offline_pretrain.enabled` controls the bilinear offline pretrain
  stage.
- `latent_learning.patch_past_steps` and `patch_future_steps` control expert
  window size.
- `ipmd.use_estimated_rewards_for_ppo` and `ipmd.env_reward_weight` determine
  how much PPO actually follows estimated rewards versus env rewards.

## Methodological Constraints

State-only expert data creates identifiability limits. Avoid claims that require
expert actions unless the data path explicitly provides them.

Good directions:

- State-representation learning from expert windows.
- IRL/adversarial reward objectives comparing rollout state distributions
  against expert state distributions.
- Reward structure that is semantically grouped, for example joint position,
  joint velocity, root state, and anchor terms.
- Offline warm-start followed by online fine-tuning.

Risky directions:

- Pure behavior cloning without real action labels.
- Treating next joint position as an action label without checking
  `JointPositionAction` scaling/offset semantics.
- Freezing the encoder after reconstruction if the goal is reusable `f(s)`.
- Repeating scalar reward regularization sweeps when the problem is reward
  parameterization or representation structure.

## Current Hypotheses

Use these as working hypotheses, not settled facts:

- A monolithic scalar reward MLP over flattened reward input can become a narrow
  separator rather than a smooth control-shaped reward basin.
- Grouped or structured rewards are likely more useful than only increasing
  scalar gradient penalty or L2 regularization.
- Bilinear IPMD is useful if the learned representation captures state structure
  that online reward learning can exploit.
- The bilinear policy path currently appears to depend on the latent-command
  task surface. Treat vanilla bilinear results as debugging evidence, not as
  final comparison data.
- VQ-VAE/FSQ latent IPMD is useful if discrete or held skill codes stabilize
  the high-level command geometry.

## Current Offline Pretrain Status

As of 2026-05-11, there are two offline-data paths:

- env-owned expert batches from `sample_expert_batch(...)`
- remote Unitree WBT LeRobot datasets streamed into a local TorchRL TensorDict
  replay cache

The immediate recovery target is still the bilinear SR warm-start path, not yet
the full offline IRL/GAIL objective. The LeRobot path starts with
`unitreerobotics/G1_WBT_Brainco_Pickup_Pillow` and maps low-dimensional
`robot_q_current` / `robot_q_desired` episodes into the same bilinear G1
TensorDict contract before training samples are drawn.

The Unitree WBT data does not currently provide qvel. The first mapper computes
joint and base angular velocities by finite differencing inside each episode.
This is a practical bootstrap choice, not a measured-velocity assumption.

See [LeRobot Offline Pretraining](lerobot-offline-pretraining.md) for the
dataset/cache contract and the current re-image checklist.

The recovered local test ladder is:

- `num_envs=128` for bug-finding smoke runs.
- `num_envs=1024-2048` for local performance/debug runs.
- `num_envs=4096` for the first cluster-ready run.
- Larger cluster env counts should be deliberate memory-headroom probes, not
  guesses.

Latest verified local results:

- 128-env bug smoke passed with 2 offline SR updates and 1 online iteration.
- 128-env functional smoke passed with 20 offline SR updates, sampling eval at
  updates 10 and 20, and 2 online iterations.
- 1024-env performance smoke passed with 2 offline SR updates and 1 online
  iteration.

Current cluster comparison target:

- `scratch`: no offline pretrain, online SR updates enabled.
- `pretrained_finetune`: offline SR pretrain, then online SR updates continue.
- `pretrained_frozen`: offline SR pretrain, then no online SR updates.
- `random_frozen`: no offline pretrain and no online SR updates.
- `pretrained_bc_finetune`: offline SR pretrain, offline policy BC on
  reconstructed expert actions, then online SR updates continue.

This comparison is now feature-only: `bilinear.policy_include_raw_state=False`,
so the policy input is `F(s)z` rather than `concat(F(s)z, s)`. This removes the
raw-state bypass that made `random_frozen` comparable to scratch in the earlier
batch.

The primary claim is online sample efficiency: whether pretraining reaches the
same online imitation/control quality using fewer environment interactions.
Track total wall-clock separately because offline pretraining adds startup cost.
The active comparison uses a 100M-frame online budget:
`max_iterations=1024` at `num_envs=4096`.

The current default pretrain budget is `2000` updates with batch size `8192`,
or about `16.4M` sampled expert transitions before online learning starts. In
the first feature-only 4096-env run, offline SR loss kept improving through the
end of pretraining: dynamics loss was about `12.1` at update 100 and `2.68` at
update 2000, while reconstruction MSE was about `44.0` at update 100 and `4.75`
at update 2000. This is enough for the first ablation, but not enough to claim
the offline representation is saturated.

The 20M-frame pretrained+finetune update-count sweep was cancelled after the
budget was raised. Run a future pretrain-length check with:

```bash
experiments/bilinear_pretrain/submit_pretrain_update_sweep.sh
```

The default sweep is `OFFLINE_NUM_UPDATES = 500, 1000, 2000, 4000`. Select the
pretrain budget by early online learning at fixed environment frames, not by
offline loss alone.

The next methodological step is still offline IRL/adversarial reward learning
once the pretrain/online bridge is stable.

## Implementation Rules

- Validate required observation keys, window sizes, and dimensional contracts at
  construction/config time where possible.
- Avoid defensive guards in the algorithmic hot path.
- Keep posterior and collector ownership in the IPMD agent, not in generic env
  helper layers.
- Keep vanilla and latent task surfaces distinct. IPMD can support both, but PPO
  should stay on vanilla surfaces and ASE should stay on latent surfaces.
- If a change is algorithmic, patch `./RLOpt`; if it is expert batch,
  env observation, task registration, manifest, or cluster routing, patch this
  repo.
