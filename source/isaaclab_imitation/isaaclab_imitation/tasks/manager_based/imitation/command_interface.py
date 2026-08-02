# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The declared command interface of the imitation tasks (v2).

ONE place that says what the environment commands and who produces it. The
interface has exactly two channels:

- the **reference** channel, always present and always dataset-backed. It is
  what rewards, terminations, the MPJPE metric, and the ``reset_reference_state``
  teleport are measured against, in every mode -- including planner evaluation,
  where the reference is loaded purely to score a rollout it does not drive.
  What is pluggable there is *selection*: which motion and which start frame
  (:class:`~...mdp.commands.reference.ReferenceSelectionCfg`), so the
  environment no longer randomises that behind a default.
- the **actor** channel, exactly one of latent / explicit / chunk. This is the
  only thing that varies across the comparison rows:

  =========================  ===========================  ==================
  row                        actor command                source
  =========================  ===========================  ==================
  oracle / explicit tracker  :class:`ExplicitCommandCfg`   ``reference``
  latent (DiffSR / SONIC)    :class:`LatentCommandCfg`     ``agent``
  planner packet             :class:`ChunkCommandCfg`      ``external``
  =========================  ===========================  ==================

A third *view* exists but is not a channel: :class:`EncoderViewCfg` declares the
windowed reference terms a latent recipe's encoder (posterior / prior) reads.
It lives on the policy observation group next to the actor's latent command, so
the observation group is a superset of the actor contract -- the "the actor
consumes exactly one command source" invariant is enforced where it is real, on
the derived actor input keys (:func:`actor_command_keys`), not by splitting the
group.

The environment config carries one of these and is therefore the single
authority on what the actor, the critic, and the encoder read; the agent config
consumes the derivation here instead of restating it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .command_components import (
    COMMAND_COMPONENT_ORDER,
    COMMAND_COMPONENT_TERM_NAMES,
    COMMAND_SPACE_COMPONENTS,
    COMMAND_TERM_NAME_COMPONENTS,
    COMPONENT_BODY_SET_FIELDS,
    FULL_BODY_COMPONENTS,
    LATENT_COMMAND_TERM_NAME,
    command_space_components,
    component_term_names,
    is_missing,
    normalize_command_components,
)
from .mdp.commands.actor import (
    ChunkCommandCfg,
    ExplicitCommandCfg,
    LatentCommandCfg,
)
from .mdp.commands.reference import (
    ReferenceChannelCfg,
    ReferenceSelectionCfg,
)

_CHANNELS = frozenset({"reference", "actor"})

# Every command term an observation group may declare, in canonical order.
_ALL_COMMAND_TERM_NAMES: tuple[str, ...] = (
    LATENT_COMMAND_TERM_NAME,
    *component_term_names(COMMAND_COMPONENT_ORDER),
)


def _component_of_term(term_name: str) -> str:
    """The command component an observation term name stands for."""
    if term_name == LATENT_COMMAND_TERM_NAME:
        return LATENT_COMMAND_TERM_NAME
    return COMMAND_TERM_NAME_COMPONENTS[term_name]


@configclass
class EncoderViewCfg:
    """Windowed reference components a latent recipe's encoder reads.

    These live on the policy observation group beside the actor's latent command
    but are NOT the actor's command: the actor's input keys never contain them.
    The window is the recipe's (vqvae past=8, future-CVAE future=9, ...).
    """

    components: tuple[str, ...] = FULL_BODY_COMPONENTS
    past_steps: int = 0
    future_steps: int = 0

    def resolve(self) -> None:
        self.components = normalize_command_components(self.components)
        if int(self.past_steps) < 0 or int(self.future_steps) < 0:
            raise ValueError("encoder window steps must be >= 0.")

    def command_terms(self) -> tuple[str, ...]:
        return component_term_names(self.components)


@configclass
class CommandInterfaceCfg:
    """Two channels, one declared interface.

    Every environment config carries exactly one of these; the command terms,
    the observation-group command surface, and the actor / critic / encoder
    input keys are all derived from it.
    """

    # pyrefly: ignore[bad-assignment]  # Isaac Lab required-field idiom
    reference: ReferenceChannelCfg = MISSING

    actor: LatentCommandCfg | ExplicitCommandCfg | ChunkCommandCfg = LatentCommandCfg()

    encoder: EncoderViewCfg | None = None

    critic_channels: tuple[str, ...] = ("actor", "reference")
    """Channels the critic reads. The critic may read several; the actor may not."""

    def resolve(self) -> None:
        """Normalize and validate the whole interface. Idempotent."""
        if is_missing(self.reference):
            raise ValueError(
                "CommandInterfaceCfg.reference is required: the reference channel "
                "is always present (rewards, terminations, metrics, reset pose)."
            )
        self.reference.resolve()
        self.actor.resolve()
        if self.encoder is not None:
            self.encoder.resolve()
        channels = tuple(str(name).strip().lower() for name in self.critic_channels)
        unknown = sorted(set(channels) - _CHANNELS)
        if unknown:
            raise ValueError(
                f"Unknown critic channel(s) {unknown}; expected a subset of "
                f"{sorted(_CHANNELS)}."
            )
        if not channels:
            raise ValueError(
                "critic_channels is empty; the critic would see no command."
            )
        seen: set[str] = set()
        self.critic_channels = tuple(
            name for name in channels if not (name in seen or seen.add(name))
        )
        self._validate_body_sets()

    def _validate_body_sets(self) -> None:
        """Every selected component must have the bodies it is built from."""
        used = set(self.actor_components()) | set(self.critic_components())
        if self.encoder is not None:
            used |= set(self.encoder.components)
        for component in sorted(used & set(COMPONENT_BODY_SET_FIELDS)):
            field = COMPONENT_BODY_SET_FIELDS[component]
            if not getattr(self.reference, field):
                raise ValueError(
                    f"Command component {component!r} is selected but "
                    f"ReferenceChannelCfg.{field} is empty."
                )

    # -- kind / component queries -------------------------------------------

    def actor_kind(self) -> str:
        """``latent`` | ``explicit`` | ``chunk``."""
        if isinstance(self.actor, LatentCommandCfg):
            return "latent"
        if isinstance(self.actor, ChunkCommandCfg):
            return "chunk"
        if isinstance(self.actor, ExplicitCommandCfg):
            return "explicit"
        raise TypeError(
            f"Unsupported actor command config type {type(self.actor).__name__}."
        )

    def is_latent(self) -> bool:
        return self.actor_kind() == "latent"

    def actor_components(self) -> tuple[str, ...]:
        """Explicit components the actor consumes (empty for a latent actor)."""
        if isinstance(self.actor, (ExplicitCommandCfg, ChunkCommandCfg)):
            return tuple(self.actor.components)
        return ()

    def critic_components(self) -> tuple[str, ...]:
        """Reference components the critic reads.

        ``ReferenceChannelCfg.critic_components`` when set; otherwise the actor's
        components for an explicit/chunk actor, and the full-body trio for a
        latent actor.
        """
        if self.reference.critic_components is not None:
            return tuple(self.reference.critic_components)
        return self.actor_components() or FULL_BODY_COMPONENTS

    def policy_command_terms(self) -> tuple[str, ...]:
        """Every command term the policy group carries: actor + encoder view."""
        terms = list(self.actor.command_terms())
        if self.encoder is not None:
            for term_name in self.encoder.command_terms():
                if term_name not in terms:
                    terms.append(term_name)
        return tuple(terms)

    def critic_command_terms(self) -> tuple[str, ...]:
        """Every command term the critic group carries."""
        terms: list[str] = []
        if "actor" in self.critic_channels and self.is_latent():
            terms.append(LATENT_COMMAND_TERM_NAME)
        if "reference" in self.critic_channels:
            for term_name in component_term_names(self.critic_components()):
                if term_name not in terms:
                    terms.append(term_name)
        return tuple(terms)

    def expert_batch_window(self) -> tuple[int, int]:
        """Window the OFFLINE expert-batch mapper serves policy command keys at.

        The skill encoder's view when there is one -- an offline expert batch
        must be shaped like the live observation the encoder was trained on --
        otherwise the actor's own window.
        """
        if self.encoder is not None:
            return (int(self.encoder.past_steps), int(self.encoder.future_steps))
        if isinstance(self.actor, (ExplicitCommandCfg, ChunkCommandCfg)):
            return (int(self.actor.past_steps), int(self.actor.future_steps))
        return (0, 0)

    # -- derived surfaces ----------------------------------------------------

    def apply_to_observations(self, observations) -> None:
        """Narrow and parameterize the declared observation command terms.

        The observation surface declares every command term; this decides which
        ones survive, which channel each reads, and at what window -- the single
        place where "what the environment commands" becomes "what the networks
        see". Only ever narrows, so it is idempotent.
        """
        self._apply_to_group(
            getattr(observations, "policy", None),
            kept=self.policy_command_terms(),
            channel_of=self._policy_channel_of,
            window_of=self.policy_window_for,
            drop_noise=not self.is_latent(),
        )
        self._apply_to_group(
            getattr(observations, "critic", None),
            kept=self.critic_command_terms(),
            channel_of=lambda name: (
                "actor" if name == LATENT_COMMAND_TERM_NAME else "reference"
            ),
            # The critic reads the command it judges, single-frame.
            window_of=lambda name: (0, 0),
            drop_noise=True,
        )

    @staticmethod
    def _apply_to_group(group, *, kept, channel_of, window_of, drop_noise) -> None:
        if group is None:
            return
        for term_name in _ALL_COMMAND_TERM_NAMES:
            term = getattr(group, term_name, None)
            if term is None:
                continue
            if term_name not in kept:
                setattr(group, term_name, None)
                continue
            past_steps, future_steps = window_of(term_name)
            term.params = {
                "channel": channel_of(term_name),
                "component": _component_of_term(term_name),
                "past_steps": int(past_steps),
                "future_steps": int(future_steps),
            }
            if drop_noise:
                # Command-side expert noise stays disabled on explicit trackers
                # and on every critic input (frozen protocol).
                term.noise = None

    def _policy_channel_of(self, term_name: str) -> str:
        """Which channel a surviving policy command term reads."""
        return "actor" if term_name in self.actor.command_terms() else "reference"

    def policy_window_for(self, term_name: str) -> tuple[int, int]:
        """``(past_steps, future_steps)`` a policy-group command term asks for.

        Zero for the actor's own terms: the actor channel serves its configured
        window and rejects an override, so the observation term must not ask for
        one. A term that exists only for the encoder view reads the reference
        channel at the encoder's window.
        """
        if term_name in self.actor.command_terms():
            return (0, 0)
        if self.encoder is not None and term_name in self.encoder.command_terms():
            return (int(self.encoder.past_steps), int(self.encoder.future_steps))
        raise KeyError(
            f"{term_name!r} is not a policy-group command term of this interface."
        )


# ---------------------------------------------------------------------------
# Input-key derivation (the environment config is the authority).
# ---------------------------------------------------------------------------


def actor_command_keys(
    cfg: CommandInterfaceCfg, *, group: str = "policy"
) -> list[tuple[str, str]]:
    """Ordered command input keys of the ACTOR.

    Exactly one command source: the latent command, or the explicit/chunk
    component terms. Never the encoder view, even though it shares the group.
    """
    return [(group, term_name) for term_name in cfg.actor.command_terms()]


def critic_command_keys(
    cfg: CommandInterfaceCfg, *, group: str = "critic"
) -> list[tuple[str, str]]:
    """Ordered command input keys of the CRITIC (may span both channels)."""
    return [(group, term_name) for term_name in cfg.critic_command_terms()]


def encoder_command_keys(
    cfg: CommandInterfaceCfg, *, group: str = "policy"
) -> list[tuple[str, str]]:
    """Ordered command input keys of the skill encoder (posterior / prior)."""
    if cfg.encoder is None:
        return []
    return [(group, term_name) for term_name in cfg.encoder.command_terms()]


def actor_input_keys(
    cfg: CommandInterfaceCfg,
    *,
    proprio_keys: Sequence[tuple[str, str]],
    group: str = "policy",
) -> list[tuple[str, str]]:
    """Full ordered actor contract: its one command source, then proprioception."""
    return actor_command_keys(cfg, group=group) + list(proprio_keys)


def critic_input_keys(
    cfg: CommandInterfaceCfg,
    *,
    privileged_keys: Sequence[tuple[str, str]],
    group: str = "critic",
) -> list[tuple[str, str]]:
    """Full ordered critic contract: its command view, then privileged state."""
    return critic_command_keys(cfg, group=group) + list(privileged_keys)


def bind_command_interface(agent_cfg, env_cfg) -> CommandInterfaceCfg | None:
    """Bind an agent config to the environment's declared command interface.

    The environment config is the authority on what the actor, the critic, and
    the skill encoder read; this hands the agent that authority and re-derives
    its input keys. Called once by the training / play entry points, where both
    configs exist -- they are separate Hydra entry points, so the agent cannot
    reach the environment config on its own.

    Returns the bound interface, or ``None`` when either side does not
    participate (a legacy v0/v1 task, a non-imitation agent), leaving the
    agent's own key selection in place.
    """
    interface = getattr(env_cfg, "command_interface", None)
    if interface is None or not isinstance(interface, CommandInterfaceCfg):
        return None
    if not hasattr(agent_cfg, "sync_input_keys"):
        return None
    agent_cfg._command_interface = interface
    sync = getattr(agent_cfg, "sync_input_keys", None)
    if callable(sync):
        sync()
    return interface


__all__ = [
    "COMMAND_COMPONENT_ORDER",
    "COMMAND_COMPONENT_TERM_NAMES",
    "COMMAND_SPACE_COMPONENTS",
    "FULL_BODY_COMPONENTS",
    "LATENT_COMMAND_TERM_NAME",
    "ChunkCommandCfg",
    "CommandInterfaceCfg",
    "EncoderViewCfg",
    "ExplicitCommandCfg",
    "LatentCommandCfg",
    "ReferenceChannelCfg",
    "ReferenceSelectionCfg",
    "actor_command_keys",
    "actor_input_keys",
    "bind_command_interface",
    "command_space_components",
    "component_term_names",
    "critic_command_keys",
    "critic_input_keys",
    "encoder_command_keys",
    "normalize_command_components",
]
