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


SONIC_WINDOW_TERM_NAMES = ("anchor_pos", "anchor_ori", "ee_body_pos", "foot_pos_xyz")
"""Tracking-error terms that can carry a persistence window.

Deliberately excludes ``base_too_low``: a fall is not a transient, and the M3
survival definition is stated in terms of that term firing.
"""

_DEFAULT_WINDOW_MIN_STEPS = 3

_WINDOWED_EQUIVALENT = {
    mdp.bad_anchor_pos_z_adaptive: mdp.PersistentBadAnchorPosZAdaptive,
    mdp.bad_anchor_ori_full: mdp.PersistentBadAnchorOriFull,
    mdp.bad_reference_body_pos_z_adaptive: mdp.PersistentBadReferenceBodyPosZAdaptive,
    mdp.bad_reference_body_pos_relative: mdp.PersistentBadReferenceBodyPosRelative,
}


def apply_termination_window(
    terminations: G1SonicTerminationsCfg,
    *,
    min_steps: int = _DEFAULT_WINDOW_MIN_STEPS,
    diagnostic_only: bool = False,
    term_names: tuple[str, ...] = SONIC_WINDOW_TERM_NAMES,
) -> None:
    """Give the strict tracking terms a persistence window, in place.

    Each term keeps its own threshold, anchor, and body set and only swaps the
    instantaneous predicate for the windowed wrapper around that same
    predicate, so the error geometry cannot drift away from
    :class:`G1SonicTerminationsCfg` -- only where the episode ends moves.

    This is the override path (the registered task ids stay on the
    instantaneous protocol), for launchers and evaluation scripts that already
    edit ``env_cfg.terminations``::

        apply_termination_window(env_cfg.terminations, min_steps=3)

    Idempotent: re-applying only updates ``min_steps`` / ``diagnostic_only``.
    Raises on a term whose predicate has no windowed equivalent rather than
    silently leaving it instantaneous.
    """
    for term_name in term_names:
        term = getattr(terminations, term_name, None)
        if term is None:
            continue
        windowed = _WINDOWED_EQUIVALENT.get(term.func)
        if windowed is not None:
            term.func = windowed
        elif term.func not in _WINDOWED_EQUIVALENT.values():
            raise ValueError(
                f"Termination term '{term_name}' uses {term.func!r}, which has no"
                " windowed equivalent. Windowing is defined for the strict SONIC"
                f" predicates only: {sorted(f.__name__ for f in _WINDOWED_EQUIVALENT)}."
            )
        term.params["min_steps"] = int(min_steps)
        term.params["diagnostic_only"] = bool(diagnostic_only)


def _run_parent_post_init(parent) -> None:
    parent_post_init = getattr(parent, "__post_init__", None)
    if callable(parent_post_init):
        parent_post_init()


@configclass
class G1SonicWindowedTerminationsCfg(G1SonicTerminationsCfg):
    """SONIC strict thresholds, ended only by a *persistent* violation.

    Every threshold is inherited from :class:`G1SonicTerminationsCfg`; the sole
    difference is that a tracking-error term must hold for ``min_steps``
    consecutive control steps before it terminates. That keeps the strict
    0.15 m / 0.2 bar as the value the policy must return to, while letting a
    single contact spike, retargeting glitch, or push recovery survive -- which
    threshold relaxation cannot do without also lowering the bar itself.

    Opt-in. The registered surfaces stay on :class:`G1SonicTerminationsCfg`,
    because termination causes define oracle qualification, M3 survival, and
    MPJPE truncation, so switching would make recorded gate numbers
    non-comparable.
    """

    def __post_init__(self) -> None:
        _run_parent_post_init(super())
        apply_termination_window(self, min_steps=_DEFAULT_WINDOW_MIN_STEPS)


@configclass
class G1SonicTerminationWindowProbeCfg(G1SonicTerminationsCfg):
    """Shadow measurement: record violation run lengths, terminate on none of them.

    Use this for the full-horizon diagnostic pass. Episodes end only on
    ``time_out`` / ``reference_finished``, so every violation onset is observed
    until it either resolves or the horizon runs out, and each term publishes
    ``Termination_Window/<term>/recovered_below_<k>_frac``: the fraction of
    onsets a window of length ``k`` would have survived. That measurement, not
    an assumption about transients, is what should decide whether
    :class:`G1SonicWindowedTerminationsCfg` is worth its protocol churn.

    The measurement requires not terminating: an instantaneous term destroys
    the episode before the run length it would have had is observable. So this
    removes every fall-stopping condition and is a diagnostic protocol only --
    never a qualification or paper-metric run.
    """

    def __post_init__(self) -> None:
        _run_parent_post_init(super())
        apply_termination_window(self, min_steps=1, diagnostic_only=True)


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
