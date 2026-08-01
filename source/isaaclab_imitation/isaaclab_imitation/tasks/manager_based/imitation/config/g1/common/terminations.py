# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination and termination-curriculum settings for the G1 tracking tasks."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from .... import mdp
from .constants import G1_EE_BODY_NAMES


@configclass
class G1TerminationsCfg:
    """Termination terms aligned to the 29-DoF tracking environment."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reference_finished = DoneTerm(func=mdp.reference_trajectory_finished)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.25,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "torso_link",
            "threshold": 0.8,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_only,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=G1_EE_BODY_NAMES,
                preserve_order=True,
            ),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.25,
        },
    )
    # body too low
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.4,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


@configclass
class G1SonicTerminationsCfg(G1TerminationsCfg):
    """Strict adaptive release termination protocol from SONIC."""

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_adaptive,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "threshold": 0.15,
            "down_threshold": 0.75,
            "root_height_threshold": 0.5,
        },
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori_full,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "anchor_body_name": "pelvis",
            "threshold": 0.2,
        },
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_reference_body_pos_z_adaptive,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=G1_EE_BODY_NAMES, preserve_order=True
            ),
            "reference_body_names": G1_EE_BODY_NAMES,
            "threshold": 0.15,
            "down_threshold": 0.75,
            "root_height_threshold": 0.5,
        },
    )
    foot_pos_xyz = DoneTerm(
        func=mdp.bad_reference_body_pos_relative,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "reference_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ],
            "anchor_body_name": "pelvis",
            "threshold": 0.2,
        },
    )
    base_too_low = None


def _sonic_threshold_anneal_params(
    term_name: str,
    start_value: float,
    end_value: float,
) -> dict[str, object]:
    return {
        "term_name": term_name,
        "start_value": start_value,
        "end_value": end_value,
        "start_frames": 50_000_000,
        "end_frames": 500_000_000,
    }


@configclass
class G1SonicTerminationCurriculumCfg:
    """Anneal termination thresholds from SONIC base/eval values to strict.

    The release trains strict-from-scratch at 64+ GPU scale; locally that
    spends most of the early budget on ~5-step episodes. Starting at the
    release's own base/eval thresholds and reaching the strict release values
    by 500M frames recovers fast early learning while keeping every frame
    after the anneal - and the final policy's protocol - strictly SONIC.
    Override the shared window via
    ``env.curriculum.<term>.params.{start_frames,end_frames}``.
    """

    anchor_pos_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("anchor_pos", 0.25, 0.15),
    )
    anchor_ori_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("anchor_ori", 1.0, 0.2),
    )
    ee_body_pos_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("ee_body_pos", 0.25, 0.15),
    )
    foot_pos_xyz_threshold = CurrTerm(
        func=mdp.anneal_termination_threshold_by_frames,
        params=_sonic_threshold_anneal_params("foot_pos_xyz", 0.3, 0.2),
    )
