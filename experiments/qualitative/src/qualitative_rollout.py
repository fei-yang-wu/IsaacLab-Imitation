"""Shared Isaac-side rollout scaffolding for the qualitative entrypoints.

One copy of the helpers every mode used to carry privately: environment
unwrapping, termination stripping, single-rank pinning, policy-weight
restoration, grid-camera framing, live-window encoding, and CLI list parsing.

Import this only AFTER the Isaac AppLauncher has started (the entrypoints
already order their imports that way); ``unwrap_imitation_env`` touches
``isaaclab_imitation`` lazily so importing this module itself stays cheap.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import qualitative_common as qc


def unwrap_imitation_env(env):
    from isaaclab_imitation.envs import ImitationRLEnv
    from isaaclab_imitation.envs.imitation_rl_env_legacy import ImitationRLEnvLegacy

    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (ImitationRLEnv, ImitationRLEnvLegacy)):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if isinstance(unwrapped, (ImitationRLEnv, ImitationRLEnvLegacy)):
            return unwrapped
        current = (
            getattr(current, "base_env", None)
            or getattr(current, "env", None)
            or getattr(current, "_env", None)
        )
    raise TypeError("Could not unwrap an imitation RL environment.")


def disable_all_terminations(env_cfg) -> list[str]:
    """Strip every termination term plus the curricula that mutate them.

    Every mode needs this for its own reason -- a reset would rewind one
    robot's reference cursor mid-protocol, cut a rollout short at a moment the
    script did not choose, or break an identical-start comparison -- but the
    mechanics are the same: no episode may end on its own. The episode length
    is pushed out of reach for the same reason.
    """
    disabled: list[str] = []
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is not None:
        for name in getattr(terminations, "__dataclass_fields__", {}):
            if getattr(terminations, name, None) is None:
                continue
            setattr(terminations, name, None)
            disabled.append(name)
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None:
        for name in getattr(curriculum, "__dataclass_fields__", {}):
            if getattr(curriculum, name, None) is not None:
                setattr(curriculum, name, None)
    if hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = 1.0e9
    return disabled


def pin_single_rank_on_reset(base_env, rank: int, start_step: int) -> None:
    """Every environment loads the same (rank, frame), on reset and on recovery.

    The identical start makes the injected codes the only difference between
    robots, and a mid-run recovery re-pins the SAME motion at the SAME frame,
    so a fallen robot returns to the identical pose rather than to an
    unrelated trajectory.
    """
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num_trajectories: int) -> torch.Tensor:
        return torch.full(
            (int(env_ids.numel()),), int(rank), dtype=torch.long, device=env_ids.device
        )

    tm.reset_schedule = "custom"
    tm.custom_reset_fn = _custom_reset_fn
    tm.reset_start_step = int(start_step)

    reference_term = getattr(base_env, "reference_command", None)
    selection = getattr(getattr(reference_term, "cfg", None), "selection", None)
    if selection is not None:
        selection.schedule = "custom"
        selection.full_trajectory = False
        selection.start_mode = "fixed"
        selection.start_frame = int(start_step)
        selection.random_step_min = int(start_step)
        selection.random_step_max = int(start_step)
        selection.adaptive_weight_fn = None
        reference_term._adaptive_failure_reset_sampler = None
        reference_term._build_reset_samplers()

    for attribute, value in (
        ("_random_reset_full_trajectory", False),
        ("_random_reset_step_min", 0),
        ("_random_reset_step_max", 0),
    ):
        if hasattr(base_env, attribute):
            setattr(base_env, attribute, value)


def load_policy_weights(agent, checkpoint_path: Path, device) -> list[str]:
    """Restore module weights only.

    The tuned checkpoints were trained with ``command_source=hl_skill`` and a
    different optimizer param-group layout than a playback agent builds
    (``command_source=random``), so a full ``load_model`` raises. Only the
    policy and value networks matter for inference. Both module layouts
    (``agent.policy`` and ``agent.actor_critic.policy``) and both key styles
    (``policy_state_dict`` and ``policy``) are accepted.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded: list[str] = []
    for name in ("policy", "value"):
        module = getattr(agent, name, None) or getattr(
            getattr(agent, "actor_critic", None), name, None
        )
        if module is None and name == "value":
            module = getattr(agent, "value_function", None)
        state = checkpoint.get(f"{name}_state_dict") or checkpoint.get(name)
        if module is None or state is None:
            continue
        module.load_state_dict(state, strict=True)
        loaded.append(name)
    if not loaded:
        msg = (
            f"No policy weights found in {checkpoint_path}; keys: "
            f"{sorted(checkpoint)[:8]}"
        )
        raise KeyError(msg)
    return loaded


def set_grid_camera(
    base_env,
    num_envs: int | None = None,
    *,
    framing: str = "wide",
    env_index: int | None = None,
    announce: bool = False,
) -> None:
    """One shot framing the robot grid; ``framing`` picks the trade-off.

    ``"static"`` frames the spawn grid from the env origins -- for identical
    -start interventions whose robots stay near their tiles. ``"wide"``
    recomputes from the live root positions so walking robots stay in frame.
    ``"close"`` also uses live positions but frames as tight as it can while
    holding every robot, because judging whether robots do the SAME thing
    needs joints to be visible, not silhouettes; it measures horizontal
    spread only, so a robot lying down does not pull the shot back.

    ``env_index`` narrows a live-position framing to ONE robot: with
    ``"close"`` the zero spread collapses the distance to its 3 m floor, a
    portrait shot with that robot centered and filling most of the frame.
    """
    if framing == "static":
        if env_index is not None:
            raise ValueError("env_index needs a live-position framing, not static.")
        points = base_env.scene.env_origins.detach()
    elif framing in ("wide", "close"):
        if env_index is not None:
            points = base_env.robot.data.root_pos_w.torch[
                env_index : env_index + 1
            ].detach()
        elif num_envs is not None:
            points = base_env.robot.data.root_pos_w.torch[:num_envs].detach()
        else:
            raise ValueError(f"framing={framing!r} needs num_envs or env_index.")
    else:
        raise ValueError(f"framing must be static, wide, or close; got {framing!r}.")
    center = points.mean(dim=0).clone()
    extent = points.max(dim=0).values - points.min(dim=0).values
    if framing == "close":
        span = float(torch.linalg.vector_norm(extent[:2]).item())
        if env_index is not None:
            # Single-robot portrait: tight enough that the robot fills most
            # of a square-cropped frame, centered on its torso.
            center[2] = 0.9
            distance = 2.4
        else:
            center[2] = 1.0
            distance = max(3.0, 0.75 * span + 2.0)
        tilt = (0.12, 0.30)
    else:
        span = float(torch.linalg.vector_norm(extent).item())
        center[2] = 0.9
        distance = max(6.0, (0.85 if framing == "static" else 0.8) * span)
        tilt = (0.15, 0.45)
    eye = center + torch.tensor(
        [tilt[0] * distance, -distance, tilt[1] * distance], device=base_env.device
    )
    if announce:
        count = int(points.shape[0])
        print(
            f"[INFO] camera: {count} robots span {span:.2f} m, "
            f"distance {distance:.2f} m"
        )
    push_camera(base_env, eye.cpu().tolist(), center.cpu().tolist())


def push_camera(base_env, eye, center) -> None:
    """Point both the visualizer camera and the video capture at ``center``.

    Headless, ``sim.set_camera_view`` reaches no visualizer (the base
    implementation is a no-op and the visualizer list is empty), so the video
    capture would keep its construction-time default camera. Push the framing
    straight into the capture object as well; when a live viewer exists the
    recorder re-syncs from it and this write is harmless.
    """
    eye_tuple = tuple(float(v) for v in eye)
    center_tuple = tuple(float(v) for v in center)
    base_env.sim.set_camera_view(eye_tuple, center_tuple)
    capture = getattr(getattr(base_env, "video_recorder", None), "_capture", None)
    if capture is not None and hasattr(capture, "update_camera"):
        capture.update_camera(eye_tuple, center_tuple)


def set_third_person_camera(
    base_env,
    env_index: int,
    *,
    distance: float = 2.4,
    eye_height: float = 1.5,
    center_height: float = 0.9,
    azimuth_deg: float = 45.0,
) -> None:
    """Third-person shot of ONE robot, at a fixed bearing off its heading.

    The camera sits ``distance`` metres from the robot at ``azimuth_deg``
    measured from its facing direction (yaw from the root quaternion) --
    0 is dead ahead looking back at the robot, 180 is directly behind it.
    The default 45 is a three-quarter FRONT view: a straight behind (or
    ahead) shot hides leg strides behind the torso, while the diagonal
    keeps both the stride and the front of the body readable. Slightly
    above, looking at the torso. Framing follows the MOTION rather than
    the world axes, so every filmstrip shows its robot from the same
    relative viewpoint whatever way that clip happens to be turned.
    """
    position = base_env.robot.data.root_pos_w.torch[env_index].detach()
    quat = base_env.robot.data.root_quat_w.torch[env_index].detach()
    # XYZW, not Isaac Lab's usual WXYZ: verified against the reference arrays
    # and rendered headings -- reading these values as wxyz puts the camera
    # ~90 degrees off and produces profile shots instead of a behind view.
    x, y, z, w = (float(v) for v in quat.cpu().tolist())
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # Direction from the robot TO the camera: the facing direction rotated by
    # the azimuth (counterclockwise, so positive puts the camera off the
    # robot's front-left; the mirror view is just a negative azimuth).
    camera_dir = yaw + math.radians(azimuth_deg)
    px, py = float(position[0]), float(position[1])
    eye = (
        px + math.cos(camera_dir) * distance,
        py + math.sin(camera_dir) * distance,
        eye_height,
    )
    push_camera(base_env, eye, (px, py, center_height))


def encode_live_window(
    bundle, base_env, sim_device, *, return_categories: bool = False
):
    """Encode the macro window at every environment's current reference cursor.

    The same call the frozen tracker makes at command-publication time, so a
    warmup prefix is ordinary encoder-driven tracking rather than a
    reimplementation of it. ``return_categories`` additionally returns the
    discrete assignments and is only valid for a discrete arm.
    """
    batch = base_env.current_expert_macro_transition_batch(
        horizon_steps=bundle.horizon_steps
    )["hl"]
    state = batch["state"].to(device=bundle.device, dtype=torch.float32)
    future_window = batch["future_window"].to(device=bundle.device, dtype=torch.float32)
    encoded = qc.encode_windows(bundle, state, future_window)
    z = encoded["z"].to(device=sim_device, dtype=torch.float32)
    if return_categories:
        return z, encoded["categories"].detach().cpu()
    return z


def parse_int_list(value: str | None, *, option: str = "--ranks") -> list[int] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option} must be a nonempty comma-separated list.")
    return [int(item) for item in items]


def parse_str_list(value: str | None, *, option: str = "--motions") -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{option} must be a nonempty comma-separated list.")
    return items
