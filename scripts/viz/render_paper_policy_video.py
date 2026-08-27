# Copyright (c) 2026, IsaacLab-Imitation Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: E402  # Isaac entrypoint: imports must follow AppLauncher.

"""Render paper-ready, policy-only videos of a low-level tracking checkpoint.

One video per requested trajectory rank, in one process. Unlike
``compare_policy_reference.py`` this script draws NO reference robot and NO
marker overlays: the frame contains only the policy-driven robot on a clean
studio floor, so the video argues quality, smoothness, and expressivity
directly, the way the SONIC paper's clips present a single performing robot.

The look is a **preset**, not a hard-coded scene. ``--style`` picks the studio
palette and ``--shot`` picks the framing; both are previewable as a contact
sheet before a single clip is rendered.

**``--style studio_light --shot hero_low`` is the recommended pair and the
default** (chosen for the paper on 2026-08-26): a seamless near-white cyclorama
with the robot at eye level on a 35 lens. It has no horizon to crop around and
sits on a white page without a visible frame edge. The other presets stay for
comparison, and ``--style light --shot ground_high`` reproduces what this script
rendered before the presets existed.

Presentation choices baked in:

- **Studio scene, styled at stage level.** The training env's Nucleus
  marble/grid floor is replaced with a plain slab and a three-point rig (key,
  fill, rim distant lights plus the dome). Every preset drives the same prims
  by attribute, so ``--preview`` can compare looks inside one Isaac launch.
  Lights are specified as elevation/azimuth in degrees, not as a quaternion.
- **Cyclorama by fog, not by geometry.** At a low camera angle the floor slab
  meets the dome in a hard seam. The studio presets enable RTX distance fog
  (``/rtx/fog/*``) tinted to the dome color, so the floor dissolves into the
  backdrop and the horizon disappears. ``--fog/--no-fog`` overrides the preset.
- **Follow camera with a real lens.** The Kit recording camera chases the robot
  root with an exponentially smoothed pursuit (``--camera_tau``), from a fixed
  azimuth and distance (``--camera_azimuth_deg``, ``--camera_distance``),
  optionally orbiting slowly (``--orbit_deg_per_s``). ``--shot`` sets pitch,
  distance, look-at height, and focal length together; any individual flag
  still overrides the preset. The lens matters: the Kit default focal length
  (18.15) is a 60-degree-horizontal wide angle that caricatures a close
  subject, while ``hero_low`` uses 35 for a compressed, portrait-like read.
- **Full clips.** Every rank plays from frame 0 to its reference's final
  frame. All termination and reward terms are disabled; a stumble stays in
  frame instead of resetting.
- **Deterministic.** Domain randomization and pushes are disabled and actions
  use the policy mode.

Frames for a paper figure come out as lossless PNG (``--stills_every`` or
``--stills_steps``) at the recorder's resolution, so a figure crop is never a
re-compression of an MP4. Pass ``--video_width 3840 --video_height 2160`` for a
print-resolution still pass.

``--shot sequence`` renders the stroboscopic composite instead: one image, one
scene, the robot drawn at several poses along the path it walked, the way
graphics papers show a continuous motion. It runs the clip twice -- once with
no rendering to learn the travel path and frame a locked camera to it, then
again to capture the poses -- cuts each pose out by differencing against a
background plate rendered with the robot hidden, and layers them. Two ordering
rules matter and both are geometric, not chronological: poses are spaced by
DISTANCE travelled (even time spacing bunches them wherever the robot slows,
which is where the interesting part of a motion is), and they are layered along
the SHADOW direction, so a pose's cast shadow never lands on top of the feet it
falls across.

The horizontal banding this used to show was the studio floor slab, not the
compositing and not the camera: see the size comment in ``_spawn_studio_rig``
before changing that geometry.

The Kit RTX camera is the only backend with real lighting, and it exists only
under PhysX, so this script REFUSES a Newton physics selection. Paper numbers
still come from the Newton evaluator; these videos are presentation renders
and the physics backend is recorded in the summary JSON.

Example (latent hold-1 recipe; everything after the flags is Hydra):

.. code-block:: bash

    # 1. compare every look in one launch, then pick one
    pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \\
        --checkpoint <model_step_N.pt> \\
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \\
        --ranks 606 --output_dir logs/showcase_videos/preview \\
        --preview --headless \\
        physics=physx env.data.reference_arrays_dir=... <latent overrides>

    # 2. render the clips plus print-resolution stills, on the recommended
    #    style and shot (both are the defaults, shown here for the record)
    pixi run -e isaaclab python scripts/viz/render_paper_policy_video.py \\
        --checkpoint <model_step_N.pt> \\
        --agent_entry_point rlopt_ipmd_tuned_cfg_entry_point \\
        --ranks 606 467 551 \\
        --output_dir logs/showcase_videos/paper_reel \\
        --style studio_light --shot hero_low \\
        --video_width 3840 --video_height 2160 --stills_every 50 --headless \\
        physics=physx env.data.reference_arrays_dir=... <latent overrides>
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# The pair chosen for the paper on 2026-08-26, and the default for that reason.
# A seamless near-white cyclorama with the subject at eye level on a 35 lens:
# the frame carries no horizon to crop around and drops onto a white page.
_RECOMMENDED_STYLE = "studio_light"
_RECOMMENDED_SHOT = "hero_low"

parser = argparse.ArgumentParser(
    description="Render paper-ready policy-only videos for a tracking checkpoint."
)
parser.add_argument("--task", type=str, default="Isaac-Imitation-G1-v2")
parser.add_argument("--algo", type=str, default="IPMD", choices=["PPO", "SAC", "IPMD"])
parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)."
)
parser.add_argument(
    "--agent_entry_point",
    type=str,
    required=True,
    help=(
        "Gym registry agent-config entry point. Required: the tuned checkpoints "
        "do not load under the default architecture."
    ),
)
parser.add_argument(
    "--ranks",
    type=int,
    nargs="+",
    required=True,
    help="Trajectory ranks to render, one video each, in order.",
)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Videos and the summary JSON land here.",
)
parser.add_argument(
    "--style",
    type=str,
    default=_RECOMMENDED_STYLE,
    choices=["light", "dark", "studio_light", "studio_dark", "photoreal"],
    help=(
        "Studio palette. 'studio_light' is the RECOMMENDED one and the default: "
        "a seamless cyclorama, a textureless dome plus matched distance fog, so "
        "there is no horizon line even at a low camera angle. 'studio_dark' is "
        "the same construction inverted. 'light'/'dark' are the original grey "
        "slab under the HDR sky, kept so earlier renders reproduce. 'photoreal' "
        "keeps the HDR sky visible and puts an MDL material on the ground "
        "(needs network on first use; falls back to a flat approximation)."
    ),
)
parser.add_argument(
    "--shot",
    type=str,
    default=_RECOMMENDED_SHOT,
    choices=["ground_high", "hero_low", "sequence", "orbit_hero"],
    help=(
        "Camera framing preset. 'hero_low' is the RECOMMENDED one and the "
        "default: eye level with a 35 lens, so the robot reads as a subject "
        "rather than as a wide-angle caricature. It needs a cyclorama style so "
        "the horizon does not show a seam. 'ground_high' is the original "
        "top-down chase (pitch 50, horizon out of frame). 'sequence' locks the "
        "camera and auto-frames the travel path, for --sequence_poses. "
        "'orbit_hero' adds a slow orbit. Any individual --camera_* flag "
        "overrides the preset."
    ),
)
parser.add_argument(
    "--backdrop",
    type=str,
    default=None,
    choices=["sky", "infinite"],
    help=(
        "Override the style's dome. 'sky' keeps an HDR texture, which is what "
        "casts a directional shadow. 'infinite' swaps in a uniform dome: "
        "seamless, but directionless, so the key light carries the shadow "
        "alone. Leave unset to use the style's own choice."
    ),
)
parser.add_argument(
    "--dome_hdr",
    type=str,
    default=None,
    help=(
        "Override the dome HDR texture URL for the sky backdrop. "
        "'studio' is an alias for the neutral photo-studio HDR."
    ),
)
parser.add_argument(
    "--floor_mdl",
    type=str,
    default=None,
    help="Override the MDL material URL used by --style photoreal.",
)
parser.add_argument(
    "--fog",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Force distance fog on or off, overriding the style.",
)
parser.add_argument(
    "--robot_tint",
    type=float,
    nargs=3,
    default=None,
    metavar=("R", "G", "B"),
    help=(
        "Bind one matte material over the whole robot for a flat 'clay "
        "render' figure. This REPLACES the Unitree two-tone materials, so it "
        "is off by default."
    ),
)
parser.add_argument(
    "--camera_distance",
    type=float,
    default=None,
    help="Horizontal chase distance from the robot root (m). Overrides --shot.",
)
parser.add_argument(
    "--camera_pitch_deg",
    type=float,
    default=None,
    help=(
        "Downward tilt of the chase camera (deg), overriding --shot. This is "
        "the primary vertical control: the camera height is derived from it "
        "and the distance. Above half the vertical field of view the horizon "
        "leaves the frame and the shot is ground-only. The half-angle is "
        "18 deg at the Kit default focal length and 9.6 deg at focal 35, so "
        "the lens decides where that threshold sits (see --focal_length)."
    ),
)
parser.add_argument(
    "--camera_height",
    type=float,
    default=None,
    help=(
        "Explicit camera height (m). Overrides --camera_pitch_deg; leave "
        "unset to derive the height from the pitch."
    ),
)
parser.add_argument(
    "--camera_azimuth_deg",
    type=float,
    default=215.0,
    help=(
        "World-frame azimuth of the camera around the robot (deg). The key, "
        "fill, and rim lights are placed relative to this, so the lighting "
        "follows the framing."
    ),
)
parser.add_argument(
    "--orbit_deg_per_s",
    type=float,
    default=None,
    help="Slow orbit rate; 0 keeps a fixed azimuth. Overrides --shot.",
)
parser.add_argument(
    "--camera_tau",
    type=float,
    default=None,
    help="Pursuit smoothing time constant (s); larger = calmer camera.",
)
parser.add_argument(
    "--lookat_height",
    type=float,
    default=None,
    help="Height of the camera target above the smoothed root (m).",
)
parser.add_argument(
    "--focal_length",
    type=float,
    default=None,
    help=(
        "Camera focal length in Kit units (a tenth of a world unit, so 35 "
        "reads like a 35 mm lens against the 20.955 default aperture). The "
        "Kit default is 18.15, a 60 deg horizontal wide angle. Overrides "
        "--shot."
    ),
)
parser.add_argument(
    "--f_stop",
    type=float,
    default=0.0,
    help=(
        "Aperture for depth of field; 0 disables it. When set, the focus "
        "distance tracks the camera-to-robot distance every frame."
    ),
)
parser.add_argument("--video_width", type=int, default=1920)
parser.add_argument("--video_height", type=int, default=1080)
parser.add_argument(
    "--stills_every",
    type=int,
    default=0,
    help="Also write every Nth captured frame as a lossless PNG; 0 disables.",
)
parser.add_argument(
    "--stills_steps",
    type=int,
    nargs="+",
    default=None,
    help="Captured-frame indices (per clip, 0-based) to write as PNG.",
)
parser.add_argument(
    "--aa",
    type=str,
    default=None,
    choices=["Off", "FXAA", "TAA", "DLSS", "DLAA"],
    help=(
        "Override the anti-aliasing mode. Leave unset to let the shot decide: "
        "a locked camera turns temporal AA off, because its history has "
        "nothing to average over a static frame and settles into visible "
        "horizontal seams."
    ),
)
parser.add_argument(
    "--sequence_poses",
    type=int,
    default=0,
    help=(
        "Composite this many poses of one clip into a single stroboscopic "
        "figure, the way graphics papers show a continuous motion. Requires a "
        "locked camera (--shot sequence); 0 disables. Passing --shot sequence "
        "without this uses 6."
    ),
)
parser.add_argument(
    "--sequence_alpha_min",
    type=float,
    default=0.4,
    help=(
        "Opacity of the OLDEST pose in the composite; the newest is always "
        "fully opaque. 1.0 disables the fade and draws every pose solid."
    ),
)
parser.add_argument(
    "--sequence_threshold",
    type=int,
    default=5,
    help=(
        "Per-channel difference from the background plate, 0-255, above which "
        "a pixel counts as robot or shadow. Raise it if renderer noise leaves "
        "speckle in the composite, lower it if soft shadow edges are cut off."
    ),
)
parser.add_argument(
    "--preview",
    action="store_true",
    help=(
        "Render a style x shot contact sheet instead of clips. The policy "
        "runs to --preview_step, then every combination is rendered from that "
        "one frozen pose so the looks are directly comparable."
    ),
)
parser.add_argument(
    "--preview_step",
    type=int,
    default=120,
    help="Control step to freeze at for --preview.",
)
parser.add_argument(
    "--preview_styles",
    type=str,
    nargs="+",
    default=None,
    help="Styles to include in the contact sheet. Defaults to all of them.",
)
parser.add_argument(
    "--preview_shots",
    type=str,
    nargs="+",
    default=None,
    help="Shots to include in the contact sheet. Defaults to the two stills shots.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# The recorder needs the offscreen camera pipeline.
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rlopt.agent import IPMD, PPO, SAC
from tensordict.nn import InteractionType
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import set_exploration_type, step_mdp

import isaaclab_imitation.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_imitation.envs.rlopt import IsaacLabTerminalObsReader, IsaacLabWrapper
from isaaclab_imitation.tasks.manager_based.imitation.command_interface import (
    bind_command_interface,
)

from imitation_experiments.provenance.paper_protocol_metadata import (
    disable_domain_randomization,
)

ALGORITHM_CLASS_MAP = {"PPO": PPO, "SAC": SAC, "IPMD": IPMD}

_PERSP_CAMERA = "/OmniverseKit_Persp"
_FLOOR_PRIM = "/World/studioFloor"
# ``ShapeCfg`` spawns the mesh under ``<prim>/geometry`` and puts a relative
# ``visual_material_path`` beside it, so the PreviewSurface shader lands here.
_FLOOR_MESH_PRIM = f"{_FLOOR_PRIM}/geometry/mesh"
_FLOOR_MATERIAL_PRIM = f"{_FLOOR_PRIM}/geometry/material"
_FLOOR_MDL_PRIM = "/World/Looks/studioFloorMdl"
_LIGHT_PRIMS = {
    "key": "/World/keyLight",
    "fill": "/World/fillLight",
    "rim": "/World/rimLight",
}

_SKY_HDR = (
    "{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
    "kloofendal_43d_clear_puresky_4k.hdr"
)
_STUDIO_HDR = "{NVIDIA_NUCLEUS_DIR}/Assets/Skies/Studio/photo_studio_01_4k.hdr"
_DEFAULT_FLOOR_MDL = (
    "{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
    "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
)


def _light(
    intensity: float, color: tuple, *, angle: float, elev: float, azim: float
) -> dict[str, Any]:
    """One rig light.

    ``angle`` is the angular diameter in degrees, which is what sets shadow
    softness: a small key keeps the contact shadow defined, a wide fill washes
    its own shadow out. ``elev`` is degrees above the horizon and ``azim`` is
    an offset from the camera azimuth, so the rig rotates with the framing.
    """
    return {
        "intensity": intensity,
        "color": color,
        "angle": angle,
        "elev_deg": elev,
        "azim_deg": azim,
    }


def _floor(
    color: tuple, *, roughness: float, metallic: float = 0.0, mdl: str | None = None
) -> dict[str, Any]:
    return {"color": color, "roughness": roughness, "metallic": metallic, "mdl": mdl}


def _dome(intensity: float, color: tuple, *, backdrop: str) -> dict[str, Any]:
    return {"intensity": intensity, "color": color, "backdrop": backdrop}


def _fog(
    color: tuple,
    *,
    start: float,
    end: float,
    density: float = 1.0,
    intensity: float = 1.0,
) -> dict[str, Any]:
    """Distance fog that lifts the far floor to the backdrop value.

    This is the cyclorama: without it a low camera sees a hard line where the
    slab meets the dome, and no amount of lighting hides it. ``start`` must sit
    well beyond the camera-to-robot distance -- fog that begins at the subject
    hazes the robot itself and flattens exactly the contrast a figure needs.

    ``intensity`` is the one knob that brightens the backdrop without lighting
    the scene. Dome intensity does both at once, so raising it to get a bright
    backdrop also blows out the floor; raising this instead leaves the rig
    where it is and lifts only the far field.
    """
    return {
        "color": color,
        "start": start,
        "end": end,
        "density": density,
        "intensity": intensity,
    }


# Studio palettes. Every value here is written onto live prims by
# ``_apply_style_stage``, never baked into the scene, so ``--preview`` can
# compare them all inside one Isaac launch.
#
# The light budget is calibrated against a measured reference: the original rig
# renders its floor at about sRGB 200 near the camera falling to 180 at the
# horizon, with no clipped pixels. Two things dominate and both must be set
# together. ``ambient`` is ``/rtx/sceneDb/ambientLightIntensity``, a flat term
# the rendering .kit leaves at 1.0 -- it puts a floor under the whole image that
# no amount of dimming the lights can get below, which is why the dark presets
# drop it to near zero. The rest is the three-point rig plus the dome.
_STYLES: dict[str, dict[str, Any]] = {
    "light": {
        "doc": "Original grey slab under the HDR sky; kept so old renders reproduce.",
        # Mid-gray: a white robot must separate from the floor, and near-white
        # blows out under the sky dome.
        "floor": _floor((0.46, 0.47, 0.51), roughness=0.85),
        # Matched to the floor's luminance at the horizon so the floor fades
        # into the backdrop without a visible seam.
        "dome": _dome(155.2, (0.87, 0.88, 0.90), backdrop="sky"),
        # Dome carries fill, key casts the shadow. Pushing the key far above
        # this saturates the floor and the shadow stops reading; the depth cue
        # in a ground-only shot is shadow CONTRAST, not brightness.
        "key": _light(759.0, (1.0, 0.98, 0.94), angle=1.0, elev=48.0, azim=40.0),
        "fill": _light(75.9, (0.92, 0.95, 1.0), angle=12.0, elev=25.0, azim=-75.0),
        "rim": _light(165.6, (1.0, 1.0, 1.0), angle=3.0, elev=30.0, azim=165.0),
        "fog": _fog(
            (0.87, 0.88, 0.90), start=18.0, end=95.0, density=0.8, intensity=1.7
        ),
        "ambient": 0.15,
        "ambient_occlusion": True,
    },
    "dark": {
        "doc": "Original dark slab under a dimmed HDR sky.",
        "floor": _floor((0.14, 0.15, 0.17), roughness=0.7),
        "dome": _dome(25.2, (0.09, 0.10, 0.12), backdrop="sky"),
        "key": _light(252.0, (1.0, 0.97, 0.9), angle=1.0, elev=45.0, azim=40.0),
        "fill": _light(16.8, (0.75, 0.83, 1.0), angle=15.0, elev=20.0, azim=-80.0),
        "rim": _light(320.0, (1.0, 1.0, 1.0), angle=2.0, elev=25.0, azim=168.0),
        "fog": _fog(
            (0.09, 0.10, 0.12), start=16.0, end=80.0, density=0.85, intensity=1.4
        ),
        "ambient": 0.01,
        "ambient_occlusion": True,
    },
    "studio_light": {
        "doc": "RECOMMENDED. Seamless light cyclorama: textureless dome plus fog.",
        # Darker than the dome on purpose. The fog lifts the far floor to the
        # backdrop value, so the near floor has to start below it or the frame
        # is one flat white with a robot in it.
        "floor": _floor((0.72, 0.72, 0.74), roughness=0.6),
        "dome": _dome(156.0, (0.93, 0.93, 0.95), backdrop="infinite"),
        "key": _light(600.0, (1.0, 0.98, 0.95), angle=0.8, elev=42.0, azim=35.0),
        "fill": _light(78.0, (0.95, 0.97, 1.0), angle=20.0, elev=18.0, azim=-70.0),
        "rim": _light(156.0, (1.0, 1.0, 1.0), angle=2.5, elev=28.0, azim=160.0),
        "fog": _fog((0.93, 0.93, 0.95), start=14.0, end=48.0, intensity=3.1),
        "ambient": 0.18,
        "ambient_occlusion": True,
    },
    "studio_dark": {
        "doc": "Dark glossy stage: low-roughness floor, strong rim separation.",
        # 0.45 and not lower: a distant light on a smoother floor throws one
        # broad specular blob that reads as a lens flare, not as a reflection.
        "floor": _floor((0.055, 0.06, 0.075), roughness=0.62),
        "dome": _dome(16.7, (0.04, 0.045, 0.058), backdrop="infinite"),
        "key": _light(261.8, (1.0, 0.96, 0.9), angle=0.7, elev=40.0, azim=32.0),
        "fill": _light(16.7, (0.62, 0.74, 1.0), angle=22.0, elev=15.0, azim=-82.0),
        "rim": _light(510.0, (0.92, 0.96, 1.0), angle=1.5, elev=22.0, azim=172.0),
        "fog": _fog((0.04, 0.045, 0.058), start=14.0, end=45.0, intensity=1.15),
        "ambient": 0.005,
        "ambient_occlusion": True,
    },
    "photoreal": {
        "doc": "HDR sky in frame, MDL ground, sun-hard shadows.",
        # The color/roughness here are only the fallback for an unreachable MDL.
        "floor": _floor((0.40, 0.39, 0.37), roughness=0.55, mdl=_DEFAULT_FLOOR_MDL),
        "dome": _dome(166.1, (1.0, 1.0, 1.0), backdrop="sky"),
        # 0.53 deg is the sun's angular diameter, so the shadow edge is as
        # sharp as an outdoor shot's.
        "key": _light(699.2, (1.0, 0.96, 0.88), angle=0.53, elev=38.0, azim=45.0),
        # The HDR sky already fills and rims; a second rig would double-light.
        "fill": _light(0.0, (1.0, 1.0, 1.0), angle=20.0, elev=20.0, azim=-70.0),
        "rim": _light(0.0, (1.0, 1.0, 1.0), angle=3.0, elev=25.0, azim=165.0),
        "fog": None,
        "ambient": 0.12,
        "ambient_occlusion": True,
    },
}

# Camera framings. ``focal_length`` is in Kit units (a tenth of a world unit),
# and every shot states it rather than leaving it None: the lens persists on the
# camera prim, so a shot that declines to set it inherits whichever lens the
# previous shot left behind. 18.147562 is the Kit default, 60 deg horizontal.
_SHOTS: dict[str, dict[str, Any]] = {
    "ground_high": {
        "doc": "Original top-down chase; horizon out, cast shadow carries depth.",
        "pitch_deg": 50.0,
        "distance": 2.8,
        "lookat_height": 0.75,
        "tau": 0.4,
        "focal_length": 18.147562,
        "orbit_deg_per_s": 0.0,
    },
    # A 35 lens sees 19.1 deg vertically, so the visible height at the subject is
    # 2 * distance * tan(9.55 deg) = 0.34 * distance. The G1 stands about 1.3 m,
    # and a mid-stride leg reaches lower still: below about 5 m the feet leave
    # the frame. Distance and focal length have to move together here.
    "hero_low": {
        "doc": "RECOMMENDED. Eye-level 35 lens; the robot reads as a subject.",
        "pitch_deg": 11.0,
        "distance": 5.6,
        "lookat_height": 0.85,
        "tau": 0.55,
        "focal_length": 35.0,
        "orbit_deg_per_s": 0.0,
    },
    # The stroboscopic-composite shot. The camera does NOT chase: a motion
    # sequence only reads as motion if the robot traverses a fixed frame, and a
    # chase camera keeps it centred in every pose, which argues the opposite.
    # Distance and azimuth are computed from the path the robot actually walks
    # (see ``_frame_travel_path``), so the values here are only fallbacks for a
    # motion that never leaves its start position.
    "sequence": {
        "doc": "Locked, auto-framed camera for a stroboscopic motion composite.",
        # 20 deg clears half of the default lens's 36 deg vertical FOV, which is
        # what keeps the horizon out; _frame_travel_path raises it if the lens
        # is changed to something wider.
        "pitch_deg": 20.0,
        "distance": 8.0,
        "lookat_height": 0.85,
        "tau": 0.4,
        # Wide: a sequence has to fit several metres of travel, and a long lens
        # would push the camera absurdly far back to do it.
        "focal_length": 18.147562,
        "orbit_deg_per_s": 0.0,
        "static": True,
    },
    "orbit_hero": {
        "doc": "hero_low with a slow orbit; for video, not for stills.",
        "pitch_deg": 15.0,
        "distance": 5.8,
        "lookat_height": 0.85,
        "tau": 0.8,
        "focal_length": 35.0,
        "orbit_deg_per_s": 6.0,
    },
}

# Stroboscopic-composite geometry, in metres.
_DEFAULT_SEQUENCE_POSES = 6  # past about 8 the figure turns to mush
_MIN_SEQUENCE_TRAVEL_M = 0.75  # below this the motion is "in place"
_SEQUENCE_MIN_BLOB_PX = 80.0  # reject renderer speckle, keep thin shadows
_SEQUENCE_MIN_BLOB_THICKNESS_PX = 6  # reject full-width scanline streaks
_SEQUENCE_HORIZON_MARGIN_DEG = 2.0  # pitch headroom over half the vertical FOV
_SEQUENCE_SETTLE_RENDERS = 4  # renders to let a visibility change land
_SEQUENCE_SEAM_LIMIT = 1.0  # grey levels; a clean chase render measures 0.67
_SEAM_WINDOW_ROWS = 21  # running-median span separating banding from gradient
_FLOOR_EXTENT_M = 400.0  # ~8x the furthest fog end; keeps the depth range sane
_FLOOR_THICKNESS_M = 0.02
_SEQUENCE_MARGIN_M = 1.1  # keeps the end poses off the frame edge
_SEQUENCE_SUBJECT_HEIGHT_M = 2.1  # G1 plus headroom and a little floor

# RTX distance fog. No numeric defaults are declared in any .kit file for this
# build, so every key is written explicitly rather than trusted.
_FOG_ENABLED = "/rtx/fog/enabled"
_FOG_SETTINGS = (
    ("/rtx/fog/fogDistanceBased/enabled", True),
    ("/rtx/fog/fogZup/enabled", True),
    # Height fog off: distance is the axis that hides the horizon.
    ("/rtx/fog/fogHeightDensity", 0.0),
    ("/rtx/fog/fogHeightFalloff", 1.0),
    ("/rtx/fog/fogStartHeight", 1000.0),
)


def _require_kit_camera_physics(env_cfg) -> str:
    """Refuse a physics selection whose recorder is the Newton GL viewer."""
    physics_name = type(env_cfg.sim.physics).__name__
    if "newton" in physics_name.lower():
        raise SystemExit(
            "render_paper_policy_video.py records through the Kit RTX camera, "
            "which only exists under PhysX. Pass `physics=physx`. "
            f"(got physics={physics_name})"
        )
    return physics_name


def _resolve_style(name: str, args) -> dict[str, Any]:
    """Expand a style preset and fold in the CLI overrides."""
    import copy

    style = copy.deepcopy(_STYLES[name])
    style["name"] = name
    if args.backdrop is not None:
        style["dome"]["backdrop"] = str(args.backdrop)
    if args.dome_hdr is not None:
        style["dome"]["texture"] = (
            _STUDIO_HDR if args.dome_hdr == "studio" else args.dome_hdr
        )
    else:
        style["dome"].setdefault("texture", _SKY_HDR)
    if args.floor_mdl is not None:
        style["floor"]["mdl"] = args.floor_mdl
    if args.fog is False:
        style["fog"] = None
    elif args.fog is True and style["fog"] is None:
        style["fog"] = {
            "color": style["dome"]["color"],
            "start": 8.0,
            "end": 50.0,
            "density": 1.0,
            "intensity": 1.0,
        }
    return style


def _direction_quat(
    elev_deg: float, azim_deg: float
) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) aiming a light's -Z axis down a direction.

    A USD ``DistantLight`` emits along its local -Z. The light is described by
    where it sits -- ``elev_deg`` above the horizon at world azimuth
    ``azim_deg`` -- and shines back at the origin, which is the direction
    computed here. Isaac Lab 3.0 takes ``init_state.rot`` as (x, y, z, w).
    """
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    # Unit vector from the light position toward the scene.
    dx = -math.cos(elev) * math.cos(azim)
    dy = -math.cos(elev) * math.sin(azim)
    dz = -math.sin(elev)
    # Shortest rotation carrying (0, 0, -1) onto (dx, dy, dz).
    dot = -dz
    if dot > 1.0 - 1.0e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)  # 180 deg about X
    cx, cy, cz = -dy, dx, 0.0  # cross((0, 0, -1), d)
    scale = math.sqrt((1.0 + dot) * 2.0)
    return (cx / scale, cy / scale, cz / scale, scale * 0.5)


def _spawn_studio_rig(env_cfg, style: dict[str, Any]) -> None:
    """Spawn the prims every style drives: floor slab, dome, three lights.

    The plane terrain type always spawns Isaac's grid ``default_environment``
    USD -- ``visual_material`` only tints it. So the physics ground stays (and
    is hidden after creation, see ``_hide_grid_ground``) while a visual-only
    slab with its top face exactly at z=0 provides the studio floor.

    Only the prim SET is fixed here. Every appearance value is (re)written by
    ``_apply_style_stage`` after the stage exists, which is what lets
    ``--preview`` compare styles inside one Isaac launch.
    """
    from isaaclab.assets import AssetBaseCfg

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.studio_floor = AssetBaseCfg(
        prim_path=_FLOOR_PRIM,
        spawn=sim_utils.CuboidCfg(
            # BOTH numbers are load-bearing, and both were established by
            # bisecting a horizontal-banding bug, so do not "tidy" either.
            #
            # Extent: the original 20 km slab put thin dark lines all over the
            # floor -- spaced by depth quantisation, so they crowded toward the
            # near field. The locked sequence camera scored 24 to 130 on
            # _seam_score against 0.67 for a chase shot; at 400 m it scores
            # 0.33. Keep it well past the largest fog end (110 m) so the far
            # edge stays hidden, and no bigger.
            #
            # Thickness: a thicker slab bands too, and at a shallow angle
            # instead of a grazing one -- 0.5 m took hero_low from 0.4 to 100.
            # A 2 cm slab is clean for every shot.
            size=(
                _FLOOR_EXTENT_M,
                _FLOOR_EXTENT_M,
                _FLOOR_THICKNESS_M,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=style["floor"]["color"],
                roughness=style["floor"]["roughness"],
                metallic=style["floor"]["metallic"],
            ),
        ),
        # Top face exactly at z=0.
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.5 * _FLOOR_THICKNESS_M)
        ),
    )
    sky = getattr(env_cfg.scene, "sky_light", None)
    if sky is not None:
        sky.spawn.intensity = style["dome"]["intensity"]
        sky.spawn.color = style["dome"]["color"]
        if style["dome"]["backdrop"] == "infinite":
            # A textureless dome IS the background: one flat value, no horizon.
            sky.spawn.texture_file = None
    # Three-point rig. The dome alone is directionless: without a key the robot
    # has no cast shadow and reads as floating, and without a rim a dark robot
    # merges into a dark backdrop.
    for role, prim_path in _LIGHT_PRIMS.items():
        spec = style[role]
        setattr(
            env_cfg.scene,
            f"{role}_light",
            AssetBaseCfg(
                prim_path=prim_path,
                spawn=sim_utils.DistantLightCfg(
                    intensity=spec["intensity"],
                    color=spec["color"],
                    angle=spec["angle"],
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    rot=_direction_quat(spec["elev_deg"], spec["azim_deg"])
                ),
            ),
        )


def _sky_light_prim_path(env_cfg) -> str | None:
    sky = getattr(env_cfg.scene, "sky_light", None)
    return None if sky is None else str(sky.prim_path)


def _set_prim_inputs(prim, values: dict[str, Any]) -> None:
    """Write ``inputs:<name>`` shader/light attributes.

    ``camel_case=False`` is required, not incidental: the helper's converter
    lowercases the whole string before rebuilding it, so asking it to convert
    ``inputs:diffuseColor`` yields ``inputs:diffusecolor`` -- a new attribute
    nothing reads, and the floor silently keeps its spawn-time color.
    """
    from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

    for name, value in values.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{name}", value, camel_case=False)


def _apply_ambient(intensity: float) -> None:
    """Set the renderer's flat ambient term.

    The rendering .kit leaves ``/rtx/sceneDb/ambientLightIntensity`` at 1.0,
    which puts a hard floor under every pixel. A dark studio is unreachable
    while it stands, however far the lights are dimmed.
    """
    import carb

    carb.settings.get_settings().set_float(
        "/rtx/sceneDb/ambientLightIntensity", float(intensity)
    )


def _apply_fog(fog: dict[str, Any] | None) -> None:
    """Fade the floor into the dome so a low camera sees no horizon seam."""
    import carb

    settings = carb.settings.get_settings()
    if fog is None:
        settings.set_bool(_FOG_ENABLED, False)
        return
    settings.set_bool(_FOG_ENABLED, True)
    for key, value in _FOG_SETTINGS:
        if isinstance(value, bool):
            settings.set_bool(key, value)
        else:
            settings.set_float(key, float(value))
    settings.set("/rtx/fog/fogColor", [float(c) for c in fog["color"]])
    settings.set_float("/rtx/fog/fogColorIntensity", float(fog.get("intensity", 1.0)))
    settings.set_float("/rtx/fog/fogStartDist", float(fog["start"]))
    settings.set_float("/rtx/fog/fogEndDist", float(fog["end"]))
    settings.set_float("/rtx/fog/fogDistanceDensity", float(fog["density"]))
    # No .kit file declares fog defaults in this build, so read back rather than
    # assume the keys took.
    if not settings.get(_FOG_ENABLED):
        print(
            "[RENDER] warning: /rtx/fog/enabled did not stick; horizon seam will show."
        )


def _valid_prim(stage, path: str):
    """Return the prim at ``path``, or None when it is absent or invalid."""
    prim = stage.GetPrimAtPath(path)
    return prim if prim is not None and prim.IsValid() else None


def _floor_shader_prim(stage):
    """Find the slab's PreviewSurface shader without assuming its leaf name."""
    material = _valid_prim(stage, _FLOOR_MATERIAL_PRIM)
    if material is None:
        return None
    shader = _valid_prim(stage, f"{_FLOOR_MATERIAL_PRIM}/Shader")
    if shader is not None:
        return shader
    for child in material.GetChildren():
        if child.GetTypeName() == "Shader":
            return child
    return None


def _apply_floor_mdl(stage, mdl_path: str) -> bool:
    """Swap in an MDL ground material. Returns False if it cannot be resolved."""
    from isaaclab.sim.utils import bind_visual_material
    from isaaclab.utils.assets import (
        ISAACLAB_NUCLEUS_DIR,
        NVIDIA_NUCLEUS_DIR,
        check_file_path,
    )

    resolved = mdl_path.format(
        ISAACLAB_NUCLEUS_DIR=ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR
    )
    if _valid_prim(stage, _FLOOR_MDL_PRIM) is None:
        try:
            # Nucleus is a remote HTTPS root in this install, so this is a
            # download on first use.
            if check_file_path(resolved) == 0:
                print(f"[RENDER] MDL not reachable, keeping the flat floor: {resolved}")
                return False
            sim_utils.spawn_from_mdl_file(
                _FLOOR_MDL_PRIM,
                sim_utils.MdlFileCfg(mdl_path=resolved, project_uvw=True),
            )
        except Exception as exc:  # noqa: BLE001 - a missing asset must not kill the render
            print(f"[RENDER] MDL floor failed ({exc}); keeping the flat floor.")
            return False
    bind_visual_material(_FLOOR_MESH_PRIM, _FLOOR_MDL_PRIM, stage=stage)
    return True


def _apply_style_stage(
    env_cfg, style: dict[str, Any], camera_azimuth_deg: float
) -> None:
    """Write a style onto the live stage. Re-callable, so a preview can switch."""
    from isaaclab.sim.utils import (
        bind_visual_material,
        get_current_stage,
        standardize_xform_ops,
    )
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR

    stage = get_current_stage()

    floor = style["floor"]
    bound_mdl = False
    if floor.get("mdl"):
        bound_mdl = _apply_floor_mdl(stage, floor["mdl"])
    if not bound_mdl:
        if _valid_prim(stage, _FLOOR_MDL_PRIM) is not None:
            # A previous style bound the MDL; put the flat material back.
            bind_visual_material(_FLOOR_MESH_PRIM, _FLOOR_MATERIAL_PRIM, stage=stage)
        shader = _floor_shader_prim(stage)
        if shader is not None:
            _set_prim_inputs(
                shader,
                {
                    "diffuseColor": tuple(float(c) for c in floor["color"]),
                    "roughness": float(floor["roughness"]),
                    "metallic": float(floor["metallic"]),
                },
            )

    dome = style["dome"]
    sky_path = _sky_light_prim_path(env_cfg)
    sky_prim = _valid_prim(stage, sky_path) if sky_path else None
    if sky_prim is not None:
        texture = ""
        if dome["backdrop"] == "sky":
            texture = str(dome.get("texture") or _SKY_HDR).format(
                ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR,
                NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR,
            )
        # A textureless dome IS the backdrop: one flat value with no horizon,
        # which the fog then matches the floor to.
        _set_prim_inputs(
            sky_prim,
            {
                "intensity": float(dome["intensity"]),
                "color": tuple(float(c) for c in dome["color"]),
                "texture:file": texture,
            },
        )

    for role, prim_path in _LIGHT_PRIMS.items():
        spec = style[role]
        prim = _valid_prim(stage, prim_path)
        if prim is None:
            continue
        _set_prim_inputs(
            prim,
            {
                "intensity": float(spec["intensity"]),
                "color": tuple(float(c) for c in spec["color"]),
                "angle": float(spec["angle"]),
            },
        )
        standardize_xform_ops(
            prim,
            orientation=_direction_quat(
                spec["elev_deg"], camera_azimuth_deg + spec["azim_deg"]
            ),
        )

    _apply_ambient(style["ambient"])
    _apply_fog(style["fog"])


def _tint_robot(rgb) -> None:
    """Bind one matte material over the whole robot for a flat figure render."""
    from isaaclab.sim.utils import bind_visual_material, get_current_stage

    stage = get_current_stage()
    material_path = "/World/Looks/robotTint"
    if _valid_prim(stage, material_path) is None:
        color = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        sim_utils.spawn_preview_surface(
            material_path,
            sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.55, metallic=0.0
            ),
        )
    envs = _valid_prim(stage, "/World/envs")
    if envs is None:
        print("[RENDER] no /World/envs prim; skipping the robot tint.")
        return
    for prim in envs.GetChildren():
        robot = _valid_prim(stage, f"{prim.GetPath()}/Robot")
        if robot is not None:
            bind_visual_material(str(robot.GetPath()), material_path, stage=stage)


def _hide_grid_ground() -> None:
    """Make the physics grid plane invisible; the studio slab is the visual."""
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom

    prim = get_current_stage().GetPrimAtPath("/World/ground")
    if prim is not None and prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()


def _disable_all_terminations(env_cfg) -> None:
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    for name in list(vars(terminations)):
        if name.startswith("_"):
            continue
        setattr(terminations, name, None)


def _disable_all_rewards(env_cfg) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    for name in list(vars(rewards)):
        if name.startswith("_"):
            continue
        term = getattr(rewards, name)
        if term is not None and hasattr(term, "weight"):
            term.weight = 0.0


def _unwrap_imitation_env(env):
    inner = env
    while hasattr(inner, "env") or hasattr(inner, "unwrapped"):
        candidate = getattr(inner, "unwrapped", None)
        if candidate is not None and candidate is not inner:
            inner = candidate
            continue
        nxt = getattr(inner, "env", None)
        if nxt is None or nxt is inner:
            break
        inner = nxt
    return inner


def _force_trajectory_on_reset(base_env, *, rank: int, start_step: int) -> None:
    """Pin every reset to one rank/frame (v2 command-term aware).

    Mirrors ``compare_policy_reference._force_policy_trajectory_on_reset``:
    the v2 reference command term owns start selection and would otherwise
    bypass the trajectory manager's custom rank callback.
    """
    tm = base_env.trajectory_manager

    def _custom_reset_fn(env_ids: torch.Tensor, _num: int) -> torch.Tensor:
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

    if hasattr(base_env, "_random_reset_full_trajectory"):
        base_env._random_reset_full_trajectory = False
    if hasattr(base_env, "_random_reset_step_min"):
        base_env._random_reset_step_min = 0
    if hasattr(base_env, "_random_reset_step_max"):
        base_env._random_reset_step_max = 0


def _resolve_shot(name: str, args) -> dict[str, Any]:
    """Expand a shot preset; any explicitly passed camera flag wins."""
    shot = dict(_SHOTS[name])
    shot["name"] = name
    for key, flag in (
        ("pitch_deg", "camera_pitch_deg"),
        ("distance", "camera_distance"),
        ("lookat_height", "lookat_height"),
        ("tau", "camera_tau"),
        ("focal_length", "focal_length"),
        ("orbit_deg_per_s", "orbit_deg_per_s"),
    ):
        value = getattr(args, flag)
        if value is not None:
            shot[key] = float(value)
    shot["azimuth_deg"] = float(args.camera_azimuth_deg)
    shot["height"] = (
        float(args.camera_height)
        if args.camera_height is not None
        else shot["lookat_height"]
        + shot["distance"] * math.tan(math.radians(shot["pitch_deg"]))
    )
    shot["f_stop"] = float(args.f_stop)
    shot["static"] = bool(_SHOTS[name].get("static", False))
    return shot


class _FollowCamera:
    """Exponentially smoothed chase camera driving the Kit recording prim.

    The Kit video capture samples ``/OmniverseKit_Persp`` every frame but only
    positions it once at construction, so re-aiming the prim each step is the
    supported way to get a moving recorded camera. ``ViewportManager`` writes
    only the transform, so the lens attributes authored in ``_apply_lens``
    survive every re-aim.
    """

    def __init__(self, base_env, shot: dict[str, Any]) -> None:
        self._env = base_env
        self._smoothed: torch.Tensor | None = None
        from isaacsim.core.rendering_manager import ViewportManager

        self._viewport_manager = ViewportManager
        self.configure(shot)

    def configure(self, shot: dict[str, Any]) -> None:
        """Re-point the camera at a different shot preset, mid-run if needed."""
        self._shot = shot
        self._tau = max(1.0e-3, float(shot["tau"]))
        self._distance = float(shot["distance"])
        self._height = float(shot["height"])
        self._azimuth = math.radians(float(shot["azimuth_deg"]))
        self._orbit_rate = math.radians(float(shot["orbit_deg_per_s"]))
        self._lookat_height = float(shot["lookat_height"])
        self._f_stop = float(shot.get("f_stop", 0.0))
        self._locked_center: tuple[float, float] | None = None

    def lock(self, framing: dict[str, Any]) -> None:
        """Pin the camera to one pose for the whole clip.

        A motion sequence only reads as motion because the robot crosses a
        fixed frame. Once this is set the chase is off and ``update`` rewrites
        the same transform every step.
        """
        self._locked_center = (float(framing["center"][0]), float(framing["center"][1]))
        self._azimuth = math.radians(float(framing["azimuth_deg"]))
        self._distance = float(framing["distance"])
        self._height = float(framing["height"])
        self._lookat_height = float(framing["lookat_height"])
        self._orbit_rate = 0.0

    @property
    def height(self) -> float:
        return self._height

    def reset(self) -> None:
        self._smoothed = None
        self.update(snap=True)

    def root_xy(self) -> tuple[float, float]:
        root = self._env.robot.data.root_pos_w
        root = (root.torch if hasattr(root, "torch") else root)[0].detach().cpu()
        return (float(root[0]), float(root[1]))

    def update(self, snap: bool = False) -> None:
        if self._locked_center is not None:
            self._write_view(self._locked_center)
            return
        root = self._env.robot.data.root_pos_w
        root = (root.torch if hasattr(root, "torch") else root)[0].detach().cpu()
        target = root.clone()
        target[2] = 0.0
        if self._smoothed is None or snap:
            self._smoothed = target
        else:
            dt = float(self._env.step_dt)
            alpha = 1.0 - math.exp(-dt / self._tau)
            self._smoothed = self._smoothed + alpha * (target - self._smoothed)
            self._azimuth += self._orbit_rate * dt
        self._write_view((float(self._smoothed[0]), float(self._smoothed[1])))

    def _write_view(self, center: tuple[float, float]) -> None:
        eye = [
            center[0] + self._distance * math.cos(self._azimuth),
            center[1] + self._distance * math.sin(self._azimuth),
            self._height,
        ]
        lookat = [center[0], center[1], self._lookat_height]
        self._viewport_manager.set_camera_view(_PERSP_CAMERA, eye=eye, target=lookat)
        if self._f_stop > 0.0:
            _set_camera_attr("focusDistance", math.dist(eye, lookat))


def _frame_travel_path(
    path: list[tuple[float, float]], lens: dict[str, Any], shot: dict[str, Any]
) -> dict[str, Any]:
    """Place a locked camera so the whole walked path fits the frame.

    The camera looks perpendicular to the direction of travel, which is what
    spreads the path across the frame instead of foreshortening it into a
    point. Distance is whatever it takes to fit the path horizontally AND the
    robot vertically, so the lens has to be known before this runs.
    """
    start, end = path[0], path[-1]
    travel = math.dist(start, end)
    center = (
        0.5 * (min(p[0] for p in path) + max(p[0] for p in path)),
        0.5 * (min(p[1] for p in path) + max(p[1] for p in path)),
    )
    if travel < _MIN_SEQUENCE_TRAVEL_M:
        # An in-place motion has no direction to stand perpendicular to, and
        # its poses will stack on top of each other in the composite whatever
        # the camera does. Keep the preset and say so.
        print(
            f"[SEQUENCE] the robot travels only {travel:.2f} m; poses will "
            "overlap heavily. Keeping the preset camera."
        )
        return {
            "center": center,
            "azimuth_deg": shot["azimuth_deg"],
            "distance": shot["distance"],
            "height": shot["height"],
            "lookat_height": shot["lookat_height"],
            "pitch_deg": shot["pitch_deg"],
            "travel_m": travel,
        }

    heading = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    azimuth = heading + 90.0

    # Fit the path across the frame, and the robot up it. Half-extents in
    # metres; the margins hold the end poses off the frame edge.
    half_across = 0.5 * travel + _SEQUENCE_MARGIN_M
    half_up = 0.5 * _SEQUENCE_SUBJECT_HEIGHT_M
    distance = max(
        half_across / math.tan(math.radians(0.5 * lens["hfov_deg"])),
        half_up / math.tan(math.radians(0.5 * lens["vfov_deg"])),
    )
    # Above half the vertical field of view the horizon leaves the frame. Below
    # it the floor slab meets the dome inside the shot, and that join is a hard
    # edge no fog setting reliably hides. The wide lens a sequence needs makes
    # this bite where hero_low never did, so derive the floor from the lens.
    min_pitch = 0.5 * lens["vfov_deg"] + _SEQUENCE_HORIZON_MARGIN_DEG
    pitch = max(float(shot["pitch_deg"]), min_pitch)
    if pitch > float(shot["pitch_deg"]) + 1.0e-6:
        print(
            f"[SEQUENCE] pitch raised {shot['pitch_deg']:.1f} -> {pitch:.1f} deg "
            f"to keep the horizon out of a {lens['vfov_deg']:.1f} deg frame."
        )
    height = shot["lookat_height"] + distance * math.tan(math.radians(pitch))
    return {
        "center": center,
        "azimuth_deg": azimuth,
        "distance": distance,
        "height": height,
        "lookat_height": shot["lookat_height"],
        "pitch_deg": pitch,
        "travel_m": travel,
    }


def _pose_indices(path: list[tuple[float, float]], count: int) -> list[int]:
    """Pick pose frames at equal DISTANCE along the path, not equal time.

    Even time spacing bunches poses wherever the robot slows down, which is
    exactly where the interesting part of a motion is; the figure then shows
    several near-identical poses and a gap.
    """
    count = max(2, int(count))
    steps = np.cumsum(
        [0.0] + [math.dist(path[i - 1], path[i]) for i in range(1, len(path))]
    )
    if steps[-1] < _MIN_SEQUENCE_TRAVEL_M:  # in place: fall back to time
        return sorted({int(round(t)) for t in np.linspace(0, len(path) - 1, count)})
    wanted = np.linspace(0.0, float(steps[-1]), count)
    return sorted({int(np.searchsorted(steps, w)) for w in wanted})


def _robot_prims(stage) -> list:
    envs = _valid_prim(stage, "/World/envs")
    if envs is None:
        return []
    found = []
    for env_prim in envs.GetChildren():
        robot = _valid_prim(stage, f"{env_prim.GetPath()}/Robot")
        if robot is not None:
            found.append(robot)
    return found


def _seam_score(frame) -> float:
    """How strong the coherent horizontal seams in a frame are, in grey levels.

    A regression guard, not a correction. Collapsing each row to one number
    kills per-pixel noise but keeps a seam, because a seam is the same
    deviation right across the row; subtracting a running median down the rows
    then removes the floor's own smooth gradient.

    The per-row statistic is the column MEDIAN, not the mean, and the WHOLE
    frame is scored. Both matter, and getting them wrong cost real time here: a
    mean is moved by the robot, so a band containing the subject reads as
    banded whether or not it is, and scoring only a strip is worse than useless
    as a gate, because whatever goes unmeasured is free to be terrible.

    A healthy frame scores well under 1. The old 20 km floor slab scored 24-130.
    """
    if frame is None:
        return 0.0
    grey = np.asarray(frame).astype(np.float32).mean(axis=2)
    profile = np.median(grey, axis=1)
    pad = _SEAM_WINDOW_ROWS // 2
    padded = np.pad(profile, pad, mode="edge")
    smooth = np.median(
        np.lib.stride_tricks.sliding_window_view(padded, _SEAM_WINDOW_ROWS), axis=-1
    )
    return float(np.abs(profile - smooth).max())


def _settled_render(base_env, warmup: int = _SEQUENCE_SETTLE_RENDERS):
    """Let the renderer converge on a still scene, then take ONE frame.

    Not a median of several. This renderer converges tile by tile, so
    consecutive frames of a static scene differ in tile-shaped patches, and a
    per-pixel median across them bakes those tile seams into the result --
    measurably: a plain single render shows no row-to-row step above 2 levels,
    a median of five shows dozens on a regular 112-row pitch. Warming up and
    taking one frame gives a clean image; what the warmup is for is the
    transient scanline garbage the first frames after a visibility change
    contain.
    """
    frame = None
    for _ in range(max(1, warmup)):
        frame = base_env.render()
    return frame


def _capture_plate(base_env, warmup: int = _SEQUENCE_SETTLE_RENDERS):
    """Render the empty set: the same scene with the robot hidden.

    Every pose in the composite is cut out by differencing against this, so it
    has to come from the same camera and the same lighting as the poses -- only
    the robot may change between them.

    Changing prim visibility leaves the renderer unsettled, and it recovers
    over renders rather than instantly: measured on a robot-free band, the
    plate and the first captured pose carry coherent horizontal seams up to 30
    levels deep while every later pose sits under 1.3. That is why the warmup
    here is tens of frames and not a handful, and why the caller settles again
    after the robot comes back before it starts recording. A streak in the
    plate is the worst case of all, because the plate is the composite's
    background: its artifacts print across the whole figure instead of inside
    one silhouette.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom

    robots = _robot_prims(get_current_stage())
    for prim in robots:
        UsdGeom.Imageable(prim).MakeInvisible()
    plate = _settled_render(base_env, warmup=warmup)
    score = _seam_score(plate)
    if score > _SEQUENCE_SEAM_LIMIT:
        print(
            f"[SEQUENCE] warning: the background plate shows {score:.1f}-level "
            "horizontal banding and the figure will carry it. Check the studio "
            "floor slab's extent and thickness before anything else."
        )
    for prim in robots:
        UsdGeom.Imageable(prim).MakeVisible()
    # Settle again with the robot back, so the clip's first frames -- and the
    # first captured pose with them -- are not still recovering.
    _settled_render(base_env, warmup=warmup)
    return plate


def _shadow_draw_order(poses: list, key_azimuth_deg: float) -> list:
    """Order poses so a cast shadow never lands on top of the feet it reaches.

    Layering is painter's order with no depth test, so whichever pose is drawn
    last wins every pixel it claims. Shadows all run the same way -- downwind of
    the key light -- so the pose a shadow falls ON must be drawn AFTER the pose
    the shadow comes FROM, or that shadow eats its feet. Sorting by how far
    each pose sits along the shadow direction gives exactly that order, and it
    stays correct whichever way the robot happens to walk.

    ``_direction_quat`` puts the key at ``key_azimuth_deg`` shining back at the
    origin, so shadows extend along the direction the light travels.
    """
    angle = math.radians(key_azimuth_deg)
    shadow = (-math.cos(angle), -math.sin(angle))
    return sorted(poses, key=lambda p: p["xy"][0] * shadow[0] + p["xy"][1] * shadow[1])


def _composite_sequence(plate, poses: list, *, threshold: int):
    """Layer the poses onto the background plate in the given order.

    A pixel belongs to a pose when it differs from the plate, which picks up
    the cast shadow along with the robot -- and the shadows are what keep the
    poses standing on the floor instead of floating. Each pose carries its own
    alpha (age) and the list is already in draw order (see
    ``_shadow_draw_order``); the two are deliberately independent, because the
    fade has to encode time while the layering has to encode geometry.
    """
    import cv2

    plate_i = plate.astype(np.int16)
    out = plate.astype(np.float32)
    close_k = np.ones((7, 7), np.uint8)
    for pose in poses:
        frame, alpha = pose["frame"], float(pose["alpha"])
        diff = np.abs(frame.astype(np.int16) - plate_i).max(axis=2)
        mask = (diff > int(threshold)).astype(np.uint8)
        # Bridge the gaps first. The robot's white shell sits within a few
        # levels of the backdrop, so differencing finds its outline and its
        # dark joints but drops the middle of a limb; closing rejoins the
        # outline so the fill below has something continuous to work with.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(mask)
        for contour in contours:
            # Area rejection rather than an opening: an opening eats the thin
            # end of a cast shadow, which is exactly the part that grounds the
            # pose. This drops renderer speckle and keeps the shadow.
            if cv2.contourArea(contour) < _SEQUENCE_MIN_BLOB_PX:
                continue
            # The renderer intermittently returns whole scanlines that differ by
            # tens of levels from their neighbours. Those streaks are full-frame
            # wide and a couple of pixels tall, so they clear any area limit;
            # thickness is what separates them from a robot or a shadow.
            _, _, box_w, box_h = cv2.boundingRect(contour)
            if min(box_w, box_h) < _SEQUENCE_MIN_BLOB_THICKNESS_PX:
                continue
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        # Filling enclosed holes -- between the legs, under an arm -- is safe:
        # the frame and the plate are identical everywhere outside the robot
        # and its shadow, so compositing those pixels is the identity.
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)
        weight = np.clip(soft, 0.0, 1.0)[..., None] * alpha
        out = out * (1.0 - weight) + frame.astype(np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _persp_camera():
    """The Kit recording camera as a ``UsdGeom.Camera``, or None.

    ``ViewportManager`` positions this prim but exposes no lens API, and its
    ``get_camera`` resolves viewports rather than camera paths, so the lens
    attributes are authored straight onto the prim.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdGeom

    prim = _valid_prim(get_current_stage(), _PERSP_CAMERA)
    return None if prim is None else UsdGeom.Camera(prim)


def _camera_edit_context():
    """Edit the persp camera where Kit actually authors it: the session layer.

    ``/OmniverseKit_Persp`` is defined in the stage's session layer, whose
    opinion is stronger than the root layer. Writing the lens through the
    default edit target silently loses to it -- the value reads back unchanged
    on the very next line.
    """
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd

    stage = get_current_stage()
    return Usd.EditContext(stage, stage.GetSessionLayer())


def _set_camera_attr(name: str, value) -> None:
    camera = _persp_camera()
    if camera is None:
        return
    attribute = camera.GetPrim().GetAttribute(name)
    if attribute:
        with _camera_edit_context():
            attribute.Set(value)


def _apply_lens(shot: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Author the recording camera's lens.

    Kit's default focal length of 18.15 against the 20.955 aperture is a
    60 degree horizontal wide angle: fine looking down at a floor, but it
    caricatures a subject at 4 m. The near clip also defaults to 1 m, which
    would slice through a close hero shot.
    """
    from pxr import Gf

    camera = _persp_camera()
    if camera is None:
        print(
            f"[RENDER] warning: {_PERSP_CAMERA} not on the stage; lens left at Kit defaults."
        )
        return {"focal_length": None, "f_stop": float(shot.get("f_stop", 0.0))}
    prim = camera.GetPrim()
    aperture = prim.GetAttribute("horizontalAperture").Get() or 20.955
    with _camera_edit_context():
        if shot.get("focal_length") is not None:
            camera.CreateFocalLengthAttr().Set(float(shot["focal_length"]))
        camera.CreateFStopAttr().Set(float(shot.get("f_stop", 0.0)))
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 1.0e6))

    focal = prim.GetAttribute("focalLength").Get() or 18.147562
    if (
        shot.get("focal_length") is not None
        and abs(focal - shot["focal_length"]) > 1.0e-3
    ):
        print(
            f"[RENDER] warning: focalLength read back as {focal:.4f}, "
            f"asked for {shot['focal_length']:.4f} (Kit may be re-asserting it)."
        )
    vertical_aperture = aperture * float(height) / float(width)
    return {
        "focal_length": float(focal),
        "horizontal_aperture": float(aperture),
        "hfov_deg": math.degrees(2.0 * math.atan(aperture / (2.0 * focal))),
        "vfov_deg": math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal))),
        "f_stop": float(shot.get("f_stop", 0.0)),
    }


def _walk_clip(env, base_env, camera, policy, td, clip_steps: int) -> list:
    """Step one clip without rendering and return the root path it walked.

    This is the first of the sequence figure's two passes. Nothing is recorded,
    so the expensive annotator read never runs; all it costs is the physics.
    Element ``i`` is the root position AFTER step ``i+1``, which is the same
    frame the recorder captures as ``recorded_frames[i]`` on the second pass.
    """
    path = []
    timestep = 0
    while simulation_app.is_running():
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = policy(td)
            td = env.step(td)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )
        path.append(camera.root_xy())
        timestep += 1
        if bool(base_env.current_reference_is_final_frame()[0].item()):
            break
        if timestep >= clip_steps + 2:
            break
    return path


def _write_png(frame, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


class _PaperRecordVideo(gym.wrappers.RecordVideo):
    """Recorder that aims the camera at the post-step pose and dumps PNGs.

    ``RecordVideo`` captures inside ``env.step``, after the physics has already
    advanced. Updating the chase camera in the outer loop therefore rendered
    every frame with the PREVIOUS step's camera pose -- a fixed one-frame lag
    that reads as the robot sliding off-centre during fast motion. Aiming from
    inside the capture hook removes it.
    """

    def __init__(
        self, env, *, camera, stills_dir, stills_steps, stills_every, **kwargs
    ):
        super().__init__(env, **kwargs)
        self._paper_camera = camera
        self._stills_dir = stills_dir
        self._stills_steps = set(int(s) for s in (stills_steps or ()))
        self._stills_every = max(0, int(stills_every))
        self.still_paths: list[str] = []

    def _wants_still(self, index: int) -> bool:
        if index in self._stills_steps:
            return True
        return self._stills_every > 0 and index % self._stills_every == 0

    def _capture_frame(self):
        self._paper_camera.update()
        before = len(self.recorded_frames)
        name = self._video_name
        super()._capture_frame()
        # The parent drops the frame, or stops the recording outright, when the
        # render returns something other than an array.
        if len(self.recorded_frames) != before + 1:
            return
        index = before
        if self._stills_dir is not None and self._wants_still(index):
            path = Path(self._stills_dir) / str(name) / f"frame_{index:05d}.png"
            _write_png(self.recorded_frames[-1], path)
            self.still_paths.append(str(path))


def _video_stem(rank: int, motion: str) -> str:
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in motion
    ).strip("_")
    return f"rank-{rank:06d}-{safe or 'motion'}"


def _contact_sheet(cells: list[dict[str, Any]], path: Path, columns: int) -> None:
    """Lay the preview renders out in a labelled grid."""
    from PIL import Image, ImageDraw

    thumbs = [Image.open(cell["path"]) for cell in cells]
    width = 640
    height = max(1, round(width * thumbs[0].height / thumbs[0].width))
    label_h = 28
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_h)), (24, 24, 26))
    draw = ImageDraw.Draw(sheet)
    for index, (thumb, cell) in enumerate(zip(thumbs, cells)):
        col, row = index % columns, index // columns
        x, y = col * width, row * (height + label_h)
        sheet.paste(
            thumb.resize((width, height), Image.Resampling.LANCZOS), (x, y + label_h)
        )
        label = f"{cell['style']}  /  {cell['shot']}"
        recommended = (
            cell["style"] == _RECOMMENDED_STYLE and cell["shot"] == _RECOMMENDED_SHOT
        )
        if recommended:
            # ASCII only: PIL's default bitmap font has no glyph for a star and
            # draws a tofu box instead.
            label += "     <<< RECOMMENDED"
        draw.text(
            (x + 8, y + 8),
            label,
            fill=(255, 214, 102) if recommended else (235, 235, 235),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _render_preview(
    *, base_env, env_cfg, camera, args, output_dir: Path, warmup: int = 3
) -> list[dict[str, Any]]:
    """Render the style x shot matrix from one frozen pose.

    Physics only advances through ``SimulationContext.step``, so the extra
    renders here cannot move the robot: every cell shows the same pose and the
    only difference between them is the look.
    """
    styles = args.preview_styles or list(_STYLES)
    shots = args.preview_shots or ["ground_high", "hero_low"]
    unknown = [s for s in styles if s not in _STYLES] + [
        s for s in shots if s not in _SHOTS
    ]
    if unknown:
        raise SystemExit(f"Unknown preview style/shot: {unknown}")

    preview_dir = output_dir / "preview"
    cells: list[dict[str, Any]] = []
    for style_name in styles:
        style = _resolve_style(style_name, args)
        _apply_style_stage(env_cfg, style, float(args.camera_azimuth_deg))
        for shot_name in shots:
            shot = _resolve_shot(shot_name, args)
            camera.configure(shot)
            camera.update(snap=True)
            _apply_lens(shot, int(args.video_width), int(args.video_height))
            frame = None
            for _ in range(max(1, warmup)):
                frame = base_env.render()
            if frame is None:
                print(f"[PREVIEW] no frame for {style_name}/{shot_name}; skipping.")
                continue
            path = preview_dir / f"{style_name}__{shot_name}.png"
            _write_png(frame, path)
            cells.append({"style": style_name, "shot": shot_name, "path": str(path)})
            print(f"[PREVIEW] {style_name} / {shot_name} -> {path}")

    if cells:
        _contact_sheet(
            cells, preview_dir / "contact_sheet.png", columns=max(1, len(shots))
        )
        print(f"[PREVIEW] contact sheet: {preview_dir / 'contact_sheet.png'}")
    return cells


def _run_preview(*, env, base_env, env_cfg, camera, policy, output_dir: Path) -> None:
    """Drive the policy to one representative pose, then render every look."""
    rank = int(args_cli.ranks[0])
    _force_trajectory_on_reset(base_env, rank=rank, start_step=int(args_cli.start_step))
    with torch.inference_mode():
        td = env.reset()
    camera.reset()

    clip_steps = int(base_env.trajectory_manager._length[rank].item())
    freeze_at = min(int(args_cli.preview_step), max(0, clip_steps - 1))
    print(f"[PREVIEW] rank={rank} stepping to frame {freeze_at} before rendering.")
    for _ in range(freeze_at):
        if not simulation_app.is_running():
            break
        with (
            torch.inference_mode(),
            set_exploration_type(InteractionType.DETERMINISTIC),
        ):
            td = policy(td)
            td = env.step(td)
            td = step_mdp(
                td, exclude_reward=True, exclude_done=False, exclude_action=True
            )
        if bool(base_env.current_reference_is_final_frame()[0].item()):
            break

    cells = _render_preview(
        base_env=base_env,
        env_cfg=env_cfg,
        camera=camera,
        args=args_cli,
        output_dir=output_dir,
    )
    summary_path = output_dir / "preview_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trajectory_rank": rank,
                "frozen_at_step": freeze_at,
                "resolution": [int(args_cli.video_width), int(args_cli.video_height)],
                "cells": cells,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[PREVIEW] summary: {summary_path}")


@hydra_task_config(args_cli.task, args_cli.agent_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    if bind_command_interface(agent_cfg, env_cfg) is None:
        sync_input_keys = getattr(agent_cfg, "sync_input_keys", None)
        if callable(sync_input_keys):
            sync_input_keys()

    physics_name = _require_kit_camera_physics(env_cfg)
    style = _resolve_style(str(args_cli.style), args_cli)
    shot = _resolve_shot(str(args_cli.shot), args_cli)

    sequence_poses = int(args_cli.sequence_poses)
    if shot["static"] and sequence_poses == 0:
        sequence_poses = _DEFAULT_SEQUENCE_POSES
    if sequence_poses and not shot["static"]:
        raise SystemExit(
            "--sequence_poses needs a locked camera: pass --shot sequence. A "
            "chase camera holds the robot in the middle of every frame, so the "
            "composite would stack every pose in one place."
        )

    env_cfg.scene.num_envs = 1
    agent_cfg.env.num_envs = 1
    agent_cfg.env.env_name = args_cli.task
    agent_cfg.seed = int(args_cli.seed)
    env_cfg.seed = int(args_cli.seed)

    logger_cfg = getattr(agent_cfg, "logger", None)
    if logger_cfg is not None:
        logger_cfg.backend = ""
        logger_cfg.video = False

    _spawn_studio_rig(env_cfg, style)
    # Ambient occlusion is what darkens the contact between foot and floor.
    # Without it the robot floats at a low camera angle, whatever the lighting.
    if style["ambient_occlusion"] and getattr(env_cfg.sim, "render", None) is not None:
        env_cfg.sim.render.enable_ambient_occlusion = True
    if args_cli.aa is not None and getattr(env_cfg.sim, "render", None) is not None:
        env_cfg.sim.render.antialiasing_mode = cast(
            'Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"]', args_cli.aa
        )
    _disable_all_terminations(env_cfg)
    _disable_all_rewards(env_cfg)
    disable_domain_randomization(env_cfg)
    if hasattr(env_cfg, "video_recorder") and env_cfg.video_recorder is not None:
        env_cfg.video_recorder.window_width = int(args_cli.video_width)
        env_cfg.video_recorder.window_height = int(args_cli.video_height)
    # A long ceiling; each clip is stopped manually at its reference's end.
    env_cfg.episode_length_s = 1.0e9

    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = _unwrap_imitation_env(env)
    _hide_grid_ground()
    _apply_style_stage(env_cfg, style, float(args_cli.camera_azimuth_deg))
    if args_cli.robot_tint is not None:
        _tint_robot(args_cli.robot_tint)
    # The recording camera prim and its render product are created lazily on the
    # first render, so the lens cannot be authored before this call.
    base_env.render()
    lens = _apply_lens(shot, int(args_cli.video_width), int(args_cli.video_height))
    fov = (
        f"hfov={lens['hfov_deg']:.1f}deg vfov={lens['vfov_deg']:.1f}deg"
        if lens.get("hfov_deg") is not None
        else "lens=kit-default"
    )
    print(
        f"[RENDER] style={style['name']} shot={shot['name']} {fov} "
        f"fog={'on' if style['fog'] else 'off'}"
    )

    num_trajectories = int(base_env.trajectory_manager._length.shape[0])
    invalid = [r for r in args_cli.ranks if not 0 <= r < num_trajectories]
    if invalid:
        raise SystemExit(
            f"Ranks {invalid} outside [0, {num_trajectories - 1}] for this source."
        )
    longest = max(
        int(base_env.trajectory_manager._length[int(r)].item()) for r in args_cli.ranks
    )

    # Built before the recorder: the recorder aims it from inside frame capture.
    camera = _FollowCamera(base_env, shot)
    want_stills = bool(args_cli.stills_every) or bool(args_cli.stills_steps)
    video_recorder = _PaperRecordVideo(
        env,
        camera=camera,
        stills_dir=str(output_dir / "stills") if want_stills else None,
        stills_steps=args_cli.stills_steps,
        stills_every=args_cli.stills_every,
        video_folder=str(output_dir / "videos"),
        step_trigger=lambda _step: False,  # every clip is started manually
        video_length=longest + 2,
        disable_logger=True,
    )
    env = video_recorder

    env = IsaacLabWrapper(env)
    env = env.set_info_dict_reader(
        IsaacLabTerminalObsReader(
            observation_spec=env.observation_spec, backend="gymnasium"
        )
    )
    env = TransformedEnv(
        base_env=env,
        transform=Compose(RewardSum(), StepCounter(longest + 2)),
    )

    agent = ALGORITHM_CLASS_MAP[args_cli.algo](env=env, config=agent_cfg)

    # Inference-only: strip optimizer state so param-group layout mismatches
    # from differently-configured training runs cannot block the restore.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and (
        "optimizer_state_dict" in payload or "reward_optimizer_state_dict" in payload
    ):
        stripped = {
            key: value
            for key, value in payload.items()
            if key not in ("optimizer_state_dict", "reward_optimizer_state_dict")
        }
        tmp = tempfile.NamedTemporaryFile(
            prefix="paper_video_weights_", suffix=".pt", delete=False
        )
        tmp.close()
        torch.save(stripped, tmp.name)
        agent.load_model(tmp.name)
        os.unlink(tmp.name)
    else:
        agent.load_model(checkpoint_path)

    policy = agent.collector_policy
    policy.eval()

    if args_cli.preview:
        _run_preview(
            env=env,
            base_env=base_env,
            env_cfg=env_cfg,
            camera=camera,
            policy=policy,
            output_dir=output_dir,
        )
        env.close()
        return

    results = []

    for index, rank in enumerate(args_cli.ranks):
        _force_trajectory_on_reset(
            base_env, rank=int(rank), start_step=int(args_cli.start_step)
        )
        with torch.inference_mode():
            td = env.reset()
        camera.reset()

        dataset, motion, trajectory = base_env.trajectory_manager.get_env_traj_info(0)
        clip_steps = int(base_env.trajectory_manager._length[int(rank)].item())
        stem = _video_stem(int(rank), motion)
        print(
            f"[RENDER] {index + 1}/{len(args_cli.ranks)} rank={rank} "
            f"motion={motion!r} steps={clip_steps}"
        )
        plate = None
        pose_at: set[int] = set()
        pose_frames: list = []
        if sequence_poses:
            # Pass 1 learns where the robot actually goes, so the locked camera
            # can be framed to it. The policy is deterministic and the reset is
            # pinned, so pass 2 retraces this path.
            path = _walk_clip(env, base_env, camera, policy, td, clip_steps)
            framing = _frame_travel_path(path, lens, shot)
            camera.lock(framing)
            camera.update()
            # The three-point rig is placed relative to the camera azimuth, and
            # locking just moved the camera somewhere the style never saw. Re-
            # apply it so key, fill, and rim keep their intended angles to the
            # shot -- and so the shadows run the way the draw order assumes.
            _apply_style_stage(env_cfg, style, framing["azimuth_deg"])
            pose_at = set(_pose_indices(path, sequence_poses))
            print(
                f"[SEQUENCE] travel {framing['travel_m']:.2f} m, camera locked "
                f"at {framing['distance']:.2f} m / azimuth "
                f"{framing['azimuth_deg']:.1f} deg, {len(pose_at)} poses"
            )
            _force_trajectory_on_reset(
                base_env, rank=int(rank), start_step=int(args_cli.start_step)
            )
            with torch.inference_mode():
                td = env.reset()
            camera.update()
            plate = _capture_plate(base_env)

        stills_before = len(video_recorder.still_paths)
        video_recorder.start_recording(stem)

        timestep = 0
        while simulation_app.is_running():
            with (
                torch.inference_mode(),
                set_exploration_type(InteractionType.DETERMINISTIC),
            ):
                td = policy(td)
                # The camera is aimed inside the recorder's capture hook, which
                # runs after the physics inside this step.
                td = env.step(td)
                td = step_mdp(
                    td, exclude_reward=True, exclude_done=False, exclude_action=True
                )
            if timestep in pose_at and video_recorder.recorded_frames:
                # Reuse the frame the recorder just captured. Never average
                # several renders instead: this renderer converges tile by
                # tile, so a per-pixel median across them bakes tile seams in.
                pose_frames.append(
                    {
                        "frame": np.asarray(video_recorder.recorded_frames[-1]).copy(),
                        "xy": camera.root_xy(),
                        "alpha": 1.0,
                    }
                )
            timestep += 1
            if bool(base_env.current_reference_is_final_frame()[0].item()):
                break
            if timestep >= clip_steps + 2:
                break

        if video_recorder.recording:
            video_recorder.stop_recording()
        video_path = output_dir / "videos" / f"{stem}.mp4"
        stills = video_recorder.still_paths[stills_before:]

        sequence_path = None
        if sequence_poses and plate is not None and pose_frames:
            sequence_dir = output_dir / "sequence"
            # Fade by age so the reader can see which way the motion runs...
            alpha_min = float(args_cli.sequence_alpha_min)
            last_pose = max(1, len(pose_frames) - 1)
            for pose_index, pose in enumerate(pose_frames):
                pose["alpha"] = alpha_min + (1.0 - alpha_min) * (pose_index / last_pose)
            # ...but layer by geometry, so no pose's shadow lands on the feet
            # of the pose it falls across.
            composite = _composite_sequence(
                plate,
                _shadow_draw_order(
                    pose_frames,
                    framing["azimuth_deg"] + float(style["key"]["azim_deg"]),
                ),
                threshold=int(args_cli.sequence_threshold),
            )
            sequence_path = sequence_dir / f"{stem}.png"
            _write_png(composite, sequence_path)
            # The plate and the individual poses are kept: they are what you
            # re-tune the threshold against, and the poses tile into a
            # conventional filmstrip without re-rendering.
            _write_png(plate, sequence_dir / f"{stem}_plate.png")
            for pose_index, pose in enumerate(pose_frames):
                _write_png(
                    pose["frame"],
                    sequence_dir / f"{stem}_poses" / f"pose_{pose_index:02d}.png",
                )
            print(f"[SEQUENCE] wrote {sequence_path}")
        elif sequence_poses:
            print(f"[SEQUENCE] no composite for {stem}: no plate or no poses.")
        print(
            f"[RENDER] wrote {video_path}"
            + (f" (+{len(stills)} stills)" if stills else "")
        )
        results.append(
            {
                "trajectory_rank": int(rank),
                "dataset": dataset,
                "motion": motion,
                "trajectory": trajectory,
                "steps": timestep,
                "video": str(video_path),
                "stills": stills,
                "sequence": str(sequence_path) if sequence_path else None,
            }
        )

    summary = {
        "checkpoint": checkpoint_path,
        "task": args_cli.task,
        "physics_cfg": physics_name,
        "seed": int(args_cli.seed),
        "resolution": [int(args_cli.video_width), int(args_cli.video_height)],
        "style": style,
        "shot": shot,
        "lens": lens,
        "robot_tint": args_cli.robot_tint,
        "sequence": {
            "poses": sequence_poses,
            "alpha_min": float(args_cli.sequence_alpha_min),
            "threshold": int(args_cli.sequence_threshold),
        }
        if sequence_poses
        else None,
        "clips": results,
    }
    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RENDER] summary: {summary_path}")
    for clip in results:
        print(f"[VIDEO] {clip['video']}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
