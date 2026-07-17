# ruff: noqa: E402
"""Roll out a trained low-level IPMD policy on RANDOM skill latents.

Probes the composability of a trained skill representation with the low-level
policy. During training the latent command ``z`` is the *deterministic encoding
of the expert reference window* the env is currently tracking (see
``FrozenHighLevelSkillCommandSampler.sample_for_step``). Here we instead draw
``z`` directly from the encoder's representation space -- uniform over the
codebook for discrete encoders, the ``N(0, I)`` prior for the gaussian encoder,
or an empirical encoded-data pool for the deterministic encoder -- hold each
``z`` for ``--hold-windows`` phase windows, and step the policy so we can watch
the resulting motion.

The command layout matches training exactly: ``[z (z_dim) ; sin(phase) ;
cos(phase)]`` where ``phase`` ramps ``0 -> (P-1)/P`` across each ``P``-step
window (``P = horizon_steps``). The z-refresh cadence is decoupled from the phase
so a single ``z`` can be sustained across several windows while the policy still
sees the 25-step phase signal it trained on.

Example (headless, no video)::

    pixi run -e isaaclab python scripts/rlopt/play_random_skill_latents.py \
        --task Isaac-Imitation-G1-Latent-v0 \
        --checkpoint logs/.../low_level.pt \
        --skill-checkpoint logs/hl_skill_diffsr/gumbel_multicat/checkpoints/best.pt \
        --num_envs 4 --steps 200 --hold-windows 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Roll out a low-level IPMD policy on random skill latents."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos."
)
parser.add_argument(
    "--video_length", type=int, default=400, help="Recorded video length (steps)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Imitation-G1-Latent-v0", help="Task name."
)
parser.add_argument(
    "--algo",
    "--algorithm",
    dest="algorithm",
    type=str.upper,
    default="IPMD",
    choices=["IPMD", "IPMD_SR", "IPMD_BILINEAR"],
    help="RLOpt algorithm (must match the low-level checkpoint).",
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Low-level policy checkpoint (.pt)."
)
parser.add_argument(
    "--skill-checkpoint",
    dest="skill_checkpoint",
    type=str,
    required=True,
    help="High-level skill-encoder checkpoint (.pt) providing the latent space.",
)
parser.add_argument(
    "--output_dir", type=str, default=None, help="Log/video output directory."
)
parser.add_argument(
    "--steps",
    type=int,
    default=1000,
    help="Total rollout steps (ignored under --video).",
)
parser.add_argument(
    "--hold-windows",
    dest="hold_windows",
    type=int,
    default=1,
    help="Number of P-step phase windows to hold each z (<=0 holds one z for the whole rollout).",
)
parser.add_argument(
    "--sample-source",
    dest="sample_source",
    type=str,
    default="auto",
    choices=["auto", "prior", "data"],
    help="Where to draw z: analytic prior/codebook, empirical encoded-data pool, or auto per encoder.",
)
parser.add_argument(
    "--phase-mode",
    dest="phase_mode",
    type=str,
    default="sin_cos",
    choices=["sin_cos", "none"],
    help="Command phase channel (must match how the low-level policy was trained).",
)
parser.add_argument(
    "--data-pool-size",
    dest="data_pool_size",
    type=int,
    default=8192,
    help="Number of encoded expert windows to build the empirical z pool (data source).",
)
parser.add_argument(
    "--dataset-path",
    dest="dataset_path",
    type=str,
    default="data/lafan1/g1_hl_diffsr",
    help="Expert dataset path (required by the G1 LAFAN env; also used by the data z-pool).",
)
parser.add_argument(
    "--manifest-path",
    dest="manifest_path",
    type=str,
    default="data/lafan1/manifests/g1_lafan1_manifest.json",
    help="LAFAN1 manifest path required by the G1 tracking env.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Environment / sampling seed."
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time if possible.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Inject the dataset + manifest as Hydra overrides so the env cfg's normal
# construction path expands the manifest into loader entries (setting these
# attributes after cfg construction skips that expansion). Skip any the user
# already passed explicitly.
_extra_overrides: list[str] = []
if not any(a.startswith("env.lafan1_manifest_path=") for a in hydra_args):
    _extra_overrides.append(
        f"env.lafan1_manifest_path={os.path.abspath(args_cli.manifest_path)}"
    )
if not any(a.startswith("env.dataset_path=") for a in hydra_args):
    _extra_overrides.append(
        f"env.dataset_path={os.path.abspath(args_cli.dataset_path)}"
    )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args + _extra_overrides
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random
import time

import gymnasium as gym
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.dict import print_dict
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import (
    IPMD,
    HighLevelSkillDiffSRConfig,
    IPMDBilinear,
    IPMDSR,
    build_skill_encoder,
)
from rlopt.agent.hl_skill_diffsr import (
    FrozenHighLevelSkillCommandSampler,
    _encoder_input_window,
    _encoder_window_steps,
    _validate_macro_batch,
)
from torchrl.envs import Compose, RewardClipping, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp
from tensordict.nn import InteractionType

import isaaclab_tasks  # noqa: F401
import isaaclab_imitation.tasks  # noqa: F401

ALGORITHM_CLASS_MAP = {
    "IPMD": IPMD,
    "IPMD_SR": IPMDSR,
    "IPMD_BILINEAR": IPMDBilinear,
}

ENTRY_POINT_ALGORITHM_MAP = {
    "rlopt_ipmd_cfg_entry_point": "IPMD",
    "rlopt_ipmd_sr_cfg_entry_point": "IPMD_SR",
    "rlopt_ipmd_bilinear_cfg_entry_point": "IPMD_BILINEAR",
}


def resolve_agent_cfg_entry_point(task_name: str | None, algorithm: str) -> str:
    """Resolve the agent config entry point based on algorithm and task registry."""
    if task_name is None:
        return f"rlopt_{algorithm.lower()}_cfg_entry_point"
    task_id = task_name.split(":")[-1]
    algo_entry_point = f"rlopt_{algorithm.lower()}_cfg_entry_point"
    spec = gym.spec(task_id)
    if spec.kwargs.get(algo_entry_point) is not None:
        print(f"[INFO] Using agent config entry point: {algo_entry_point}")
        return algo_entry_point
    supported = sorted(
        ENTRY_POINT_ALGORITHM_MAP[key]
        for key in ENTRY_POINT_ALGORITHM_MAP
        if spec.kwargs.get(key) is not None
    )
    raise ValueError(
        f"task '{task_id}' does not expose an RLOpt config for '{algorithm}'. "
        f"Supported: {supported}."
    )


def load_skill_encoder(skill_checkpoint_path: str, device: torch.device):
    """Load the frozen skill encoder standalone and return (encoder, config, state_dim)."""
    checkpoint = torch.load(
        os.path.abspath(skill_checkpoint_path), map_location=device, weights_only=False
    )
    config = HighLevelSkillDiffSRConfig.from_dict(checkpoint["config"])
    state_dict = checkpoint["skill_encoder_state_dict"]
    window_steps = _encoder_window_steps(config)
    state_dim = FrozenHighLevelSkillCommandSampler._state_dim_from_encoder_state(
        state_dict, window_steps=window_steps
    )
    encoder = build_skill_encoder(
        state_dim=state_dim,
        window_steps=window_steps,
        z_dim=config.z_dim,
        hidden_dims=config.encoder_hidden_dims,
        spec=config.latent_spec(),
    ).to(device)
    # ``strict=False`` tolerates the persisted-but-optional ``tau`` buffer.
    result = encoder.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if k.split(".")[-1] != "tau"]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Skill encoder state mismatch: missing={missing}, "
            f"unexpected={list(result.unexpected_keys)}."
        )
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, config, state_dim


def _build_data_pool(encoder, config, state_dim, command_env, device, pool_size):
    """Encode a pool of real expert windows -> realized-representation z cloud."""
    if not hasattr(command_env, "sample_expert_macro_transition_batch"):
        raise RuntimeError(
            "sample-source='data' requires the env to expose "
            "sample_expert_macro_transition_batch(...)."
        )
    zs: list[torch.Tensor] = []
    remaining = int(pool_size)
    while remaining > 0:
        bs = min(remaining, 4096)
        batch = command_env.sample_expert_macro_transition_batch(
            batch_size=bs,
            horizon_steps=int(config.horizon_steps),
            split=config.train_split,
            eval_fraction=float(config.eval_trajectory_fraction),
            split_seed=int(config.trajectory_split_seed),
        )
        state, future_window, _ = _validate_macro_batch(
            batch,
            batch_size=bs,
            horizon_steps=int(config.horizon_steps),
            device=device,
            state_dim=state_dim,
            source="Offline expert",
        )
        with torch.no_grad():
            z = encoder.encode(
                state, _encoder_input_window(config, future_window), deterministic=True
            )[0]
        zs.append(z)
        remaining -= bs
    pool = torch.cat(zs, dim=0)
    print(f"[INFO] Built empirical z pool: {tuple(pool.shape)} from expert windows.")
    return pool


def build_z_sampler(encoder, config, state_dim, command_env, device, args_cli):
    """Return ``sample_z(n) -> [n, z_dim]`` drawing from the encoder's z-space."""
    mode = config.latent_mode
    z_dim = int(config.z_dim)
    source = args_cli.sample_source

    def _codebook_multicat():
        codebook = encoder.codebook.detach()  # [G, C, code_dim]
        groups, categories, _ = codebook.shape
        group_index = torch.arange(groups, device=device)

        def sample_z(n: int) -> torch.Tensor:
            idx = torch.randint(0, categories, (n, groups), device=device)
            z_q = codebook[group_index, idx]  # [n, G, code_dim]
            return z_q.reshape(n, z_dim)

        return sample_z, f"uniform over per-group codebook [G={groups}, C={categories}]"

    def _codebook_embedding(weight: torch.Tensor, label: str):
        weight = weight.detach()  # [K, z_dim]
        num_codes = weight.shape[0]

        def sample_z(n: int) -> torch.Tensor:
            idx = torch.randint(0, num_codes, (n,), device=device)
            return weight[idx]

        return sample_z, f"uniform over {label} [K={num_codes}]"

    def _gaussian():
        def sample_z(n: int) -> torch.Tensor:
            return torch.randn(n, z_dim, device=device)

        return sample_z, "N(0, I) prior"

    def _data_pool():
        pool = _build_data_pool(
            encoder, config, state_dim, command_env, device, args_cli.data_pool_size
        )

        def sample_z(n: int) -> torch.Tensor:
            idx = torch.randint(0, pool.shape[0], (n,), device=device)
            return pool[idx]

        return sample_z, f"empirical encoded-data pool [P={pool.shape[0]}]"

    if source == "data":
        return _data_pool()

    # source in {"prior", "auto"}
    if mode in ("categorical", "gumbel_multicat"):
        return _codebook_multicat()
    if mode == "gumbel":
        return _codebook_embedding(encoder.gumbel.codebook.weight, "single codebook")
    if mode == "vq":
        return _codebook_embedding(encoder.vq.embedding, "VQ embedding")
    if mode == "gaussian":
        return _gaussian()
    # deterministic / fsq have no analytic prior.
    if source == "prior":
        raise ValueError(
            f"latent_mode={mode!r} has no analytic prior; use --sample-source data."
        )
    return _data_pool()


class RandomSkillLatentSampler:
    """Per-env random-latent source mirroring the training cadence + phase.

    Decouples the *phase counter* (period ``P``, resets every window, drives the
    ``sin/cos`` exactly as in training) from the *z-refresh counter* (redraws a
    new ``z`` every ``hold_windows`` windows), so a single ``z`` can be sustained
    while the policy keeps seeing the 25-step phase signal. ``done`` always forces
    a fresh ``z``.
    """

    def __init__(self, *, sample_z, z_dim, phase_period, hold_windows, phase_mode):
        self._sample_z = sample_z
        self.z_dim = int(z_dim)
        self.phase_period = int(phase_period)
        self.phase_dim = 2 if phase_mode == "sin_cos" else 0
        self.latent_dim = self.z_dim + self.phase_dim
        # Number of steps a z is held before a forced redraw. A window boundary
        # falls every ``phase_period`` steps, so an integer multiple keeps z
        # switches aligned to phase resets. ``hold_windows <= 0`` => hold forever.
        self.z_hold_steps = (
            int(hold_windows) * self.phase_period if hold_windows > 0 else 1 << 30
        )
        self._codes: torch.Tensor | None = None
        self._z_left: torch.Tensor | None = None
        self._phase_left: torch.Tensor | None = None
        self.last_refreshed: torch.Tensor | None = None

    def _ensure(self, batch_size, device, dtype):
        if (
            self._codes is None
            or self._codes.shape[0] != batch_size
            or self._codes.device != device
            or self._codes.dtype != dtype
        ):
            self._codes = torch.zeros(
                batch_size, self.z_dim, device=device, dtype=dtype
            )
            self._z_left = torch.zeros(batch_size, device=device, dtype=torch.long)
            self._phase_left = torch.zeros(batch_size, device=device, dtype=torch.long)

    def sample_for_step(self, td, *, device, dtype) -> torch.Tensor:
        batch_size = int(td.numel())
        self._ensure(batch_size, device, dtype)
        assert self._codes is not None and self._z_left is not None
        assert self._phase_left is not None

        done = FrozenHighLevelSkillCommandSampler._done_mask(
            td, batch_size=batch_size, device=device
        )
        # Redraw z on done or when its hold expires.
        z_renew = done | (self._z_left <= 0)
        self.last_refreshed = z_renew
        if bool(z_renew.any()):
            ids = torch.nonzero(z_renew, as_tuple=False).reshape(-1)
            new_z = self._sample_z(int(ids.numel())).to(device=device, dtype=dtype)
            self._codes.index_copy_(0, ids, new_z)
            self._z_left.index_fill_(0, ids, self.z_hold_steps)
        # Reset the phase window on done or when it elapses (independent of z).
        phase_renew = done | (self._phase_left <= 0)
        if bool(phase_renew.any()):
            ids = torch.nonzero(phase_renew, as_tuple=False).reshape(-1)
            self._phase_left.index_fill_(0, ids, self.phase_period)

        phase = (self.phase_period - self._phase_left).clamp(min=0).to(torch.float32)
        phase = phase / float(self.phase_period)
        latents = self._append_phase(self._codes, phase)

        self._z_left = self._z_left - 1
        self._phase_left = self._phase_left - 1
        return latents

    def _append_phase(self, codes: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.phase_dim == 0:
            return codes
        angle = phase.reshape(-1).to(device=codes.device, dtype=codes.dtype) * (
            2.0 * math.pi
        )
        phase_features = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat((codes, phase_features), dim=-1)


agent_entry_point = resolve_agent_cfg_entry_point(args_cli.task, args_cli.algorithm)


@hydra_task_config(args_cli.task, agent_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg,
):
    """Roll out the low-level policy on random skill latents."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # non-hydra overrides
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.env.num_envs = env_cfg.scene.num_envs
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    agent_cfg.collector.frames_per_batch *= env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # dataset_path / lafan1_manifest_path are applied via Hydra overrides above so
    # the cfg expands the manifest into loader entries at construction time.

    if args_cli.seed is not None:
        torch.manual_seed(int(args_cli.seed))

    device = torch.device(env_cfg.sim.device)

    # ------------------------------------------------------------------ #
    # Load the skill encoder + derive the command dimensions.
    # ------------------------------------------------------------------ #
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Low-level checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(os.path.abspath(args_cli.skill_checkpoint)):
        raise FileNotFoundError(
            f"Skill checkpoint not found: {args_cli.skill_checkpoint}"
        )

    encoder, skill_config, state_dim = load_skill_encoder(
        args_cli.skill_checkpoint, device
    )
    z_dim = int(skill_config.z_dim)
    phase_period = int(skill_config.horizon_steps)
    phase_dim = 2 if args_cli.phase_mode == "sin_cos" else 0
    latent_dim = z_dim + phase_dim
    print(
        f"[INFO] Skill encoder: latent_mode={skill_config.latent_mode} z_dim={z_dim} "
        f"horizon_steps={phase_period} -> latent_dim={latent_dim}"
    )

    # ------------------------------------------------------------------ #
    # Align the low-level agent config to the command layout and inject
    # our own command (command_source="random" avoids the expert-data path).
    # ------------------------------------------------------------------ #
    agent_cfg.ipmd.use_latent_command = True
    agent_cfg.ipmd.command_source = "random"
    agent_cfg.ipmd.latent_dim = latent_dim
    agent_cfg.ipmd.latent_steps_min = phase_period
    agent_cfg.ipmd.latent_steps_max = phase_period
    agent_cfg.ipmd.latent_learning.command_phase_mode = args_cli.phase_mode
    agent_cfg.ipmd.latent_learning.code_period = phase_period
    agent_cfg.ipmd.latent_learning.code_latent_dim = z_dim
    if hasattr(env_cfg, "latent_command_dim"):
        env_cfg.latent_command_dim = latent_dim
    sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync_input_keys):
        sync_input_keys()

    if args_cli.output_dir is None:
        log_dir = os.path.dirname(checkpoint_path)
    else:
        log_dir = os.path.abspath(args_cli.output_dir)
        os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    # ------------------------------------------------------------------ #
    # Build the env (play.py scaffolding).
    # ------------------------------------------------------------------ #
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported.")

    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "random_skill")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video ->", video_folder)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    command_env = (
        env  # IsaacLabWrapper: exposes set_agent_latent_command + expert samplers
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(1000), RewardClipping(-10.0, 5.0)),
    )

    # ------------------------------------------------------------------ #
    # Load the low-level policy.
    # ------------------------------------------------------------------ #
    agent = ALGORITHM_CLASS_MAP[args_cli.algorithm](env=env, config=agent_cfg)
    print(f"[INFO] Loading low-level checkpoint: {checkpoint_path}")
    # Weights-only load: we only need the policy (and value) network parameters for
    # playback. Skipping the optimizer/reward/latent-learner state avoids a
    # param-group mismatch, since our playback agent config (command_source=random)
    # differs from the training config the checkpoint was saved under.
    _checkpoint = torch.load(checkpoint_path, map_location=device)
    _loaded = []
    if (
        getattr(agent, "policy", None) is not None
        and "policy_state_dict" in _checkpoint
    ):
        agent.policy.load_state_dict(_checkpoint["policy_state_dict"])
        _loaded.append("policy")
    if (
        getattr(agent, "value_function", None) is not None
        and "value_state_dict" in _checkpoint
    ):
        agent.value_function.load_state_dict(_checkpoint["value_state_dict"])
        _loaded.append("value")
    if not _loaded:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no policy_state_dict/value_state_dict."
        )
    print(f"[INFO] Loaded network weights: {_loaded}")
    policy_operator = agent.actor_critic.get_policy_operator()
    policy_operator.eval()
    latent_key = getattr(agent, "_latent_key", ("policy", "latent_command"))

    # ------------------------------------------------------------------ #
    # Build the random-latent sampler.
    # ------------------------------------------------------------------ #
    sample_z, source_desc = build_z_sampler(
        encoder, skill_config, state_dim, command_env, device, args_cli
    )
    print(f"[INFO] z source: {source_desc}")
    sampler = RandomSkillLatentSampler(
        sample_z=sample_z,
        z_dim=z_dim,
        phase_period=phase_period,
        hold_windows=args_cli.hold_windows,
        phase_mode=args_cli.phase_mode,
    )
    if sampler.latent_dim != latent_dim:
        raise ValueError(
            f"sampler latent_dim {sampler.latent_dim} != expected {latent_dim}."
        )

    dt = getattr(env, "step_dt", None)
    num_envs = int(env_cfg.scene.num_envs)
    hold_desc = (
        "forever"
        if args_cli.hold_windows <= 0
        else f"{args_cli.hold_windows} window(s)"
    )
    print(
        f"[INFO] Rolling out: num_envs={num_envs} hold={hold_desc} "
        f"(z refresh every {sampler.z_hold_steps} steps, phase period {phase_period})"
    )

    td = env.reset()
    timestep = 0
    max_steps = args_cli.video_length if args_cli.video else args_cli.steps
    print("[INFO] Starting inference loop. Press Ctrl+C to stop.")

    while simulation_app.is_running() and timestep < max_steps:
        start_time = time.time()
        latents = sampler.sample_for_step(td, device=device, dtype=torch.float32)
        # Feed our command to the policy (td) and to the env (obs/reward terms).
        td.set(latent_key, latents.reshape(*td.batch_size, latent_dim))
        command_env.set_agent_latent_command(latents.reshape(-1, latent_dim))

        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = policy_operator(td)
            td = env.step(td)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )

        # Lightweight diagnostics for env 0.
        if timestep < 3 * phase_period or timestep % phase_period == 0:
            refreshed = (
                bool(sampler.last_refreshed[0])
                if sampler.last_refreshed is not None
                else False
            )
            z0 = sampler._codes[0]
            print(
                f"[step {timestep:04d}] phase0={float(latents[0, z_dim]):+.3f}/"
                f"{float(latents[0, z_dim + 1]):+.3f} (sin/cos) "
                f"z0_norm={float(z0.norm()):.3f} new_z={refreshed}"
            )

        timestep += 1
        if args_cli.real_time and dt is not None:
            sleep_time = float(dt) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"[INFO] Finished {timestep} steps.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
