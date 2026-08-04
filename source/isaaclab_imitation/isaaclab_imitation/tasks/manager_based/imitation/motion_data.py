# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What motion data an imitation environment tracks, and where it is cached.

The configuration here obeys one rule, which is the whole point of the module:

    **A field is either an input or an output, never both.**

:class:`MotionDataCfg` holds only *inputs* -- what the user (a class default, a
preset, a Hydra ``env.data.*`` override, or a job script) declares. Resolution
reads those inputs and returns a separate, frozen :class:`ResolvedMotionData`;
it writes nothing back. Two consequences make the whole surface simpler:

- ``None`` means "derive this" with no ambiguity, because nothing but the user
  can ever write the field. There is no need to guess, by comparing a live
  value against a class default, whether a cache path was configured or merely
  computed by an earlier resolution pass -- the guessing that
  ``dataset_path_explicit`` / ``motions_explicit`` / ``timing_explicit``
  existed to do.
- Resolution is idempotent by construction rather than by contract: resolving
  twice recomputes the same outputs from inputs that resolution never touched.

Naming here is functional. A *clip* is one motion file with a frame rate; a
*manifest* names an ordered set of them. Which dataset those clips came from
(LAFAN1, Dance102, BONES-SEED) is data, not code: it appears as a manifest
label used to make cache directories self-describing, and nothing branches on
it. The one historical spelling that survives, ``lafan1_csv``, is the ILTools
loader key (:data:`~.motion_manifest.CLIP_LOADER_KEY`) and lives only at that
submodule boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.utils.configclass import configclass

from .motion_manifest import (
    CLIP_LOADER_KEY,
    build_clip_loader_kwargs,
    cache_dir_from_entries,
    infer_manifest_control_freq,
    load_clip_loader_options,
    load_clip_manifest,
    load_manifest_family,
)

_TIMING_TOLERANCE_HZ = 1.0e-6


@configclass
class MotionDataCfg:
    """The declared motion-data inputs of an imitation environment.

    Every field is a *selection*. Resolution never writes here.
    """

    manifest: str | None = None
    """Path to the clip manifest that defines the reference set.

    The manifest is the only statement of what the data is, including its
    ``input_fps``; there is no separate rate field to disagree with it.
    Required unless ``cache_dir`` points at an already-built cache, which is
    the prebuilt-buffer-on-a-compute-node case.
    """

    clips: list[str] | None = None
    """Clip names to load, or ``None`` for every clip in the manifest."""

    takes: list[str] | None = None
    """Individual takes to load within the selected clips, or ``None`` for all.

    A clip groups one or more source files in the Zarr hierarchy; this selects
    inside that group. Almost always ``None`` -- one file per clip is the norm.
    """

    require_body_states: bool = True
    """Reject sources that are not NPZ files carrying body states.

    The tracking rewards and metrics read ``body_pos_w`` / ``body_quat_w`` /
    ``body_lin_vel_w`` / ``body_ang_vel_w``, which a CSV source does not carry.
    """

    cache_dir: str | None = None
    """Where the built Zarr cache lives, or ``None`` to derive it.

    The derived path is a function of the manifest identity and its entries, so
    two jobs naming the same manifest share a cache and a changed manifest
    yields a new one.
    """

    cache_refresh: bool = False
    """Delete and rebuild the Zarr cache before loading it."""

    storage_device: str = "cuda:0"
    """Device holding the reference replay buffer.

    ``cuda:0`` keeps every transition in VRAM (a 4096-row gather is ~15 us).
    Use ``cpu`` when the reference set is larger than the GPU: TorchRL then
    builds a ``LazyMemmapStorage`` and each sampled batch is copied to the
    compute device, at roughly 1.3 ms per 4096-row gather.
    """

    persist_dir: str | None = None
    """Reusable on-disk home for the CPU memmap buffer (CPU storage only).

    When this directory already holds a matching build, the buffer is mapped in
    milliseconds instead of refilled from Zarr -- filling costs about 66 ms per
    trajectory plus 53 us per frame, i.e. hours for a 129,785-clip set, on
    every job start.
    """

    persist_id: str | None = None
    """Content identity of the persisted buffer, which makes it relocatable.

    Name the source content (a manifest sha256, a dataset release tag) and the
    buffer can be built on one machine, copied to a compute node, and reopened
    there without the Zarr being present. Leave ``None`` when the buffer is
    built and consumed in place; the absolute Zarr path is the identity then.
    """

    persist_rebuild: bool = False
    """Force a refill of ``persist_dir``.

    A persisted buffer is NOT invalidated by a Zarr rebuilt in place, nor by a
    ``persist_id`` reused for changed content; set this whenever the source
    content changes.
    """

    keys: list[str] | None = None
    """Zarr arrays to load into the reference buffer, or ``None`` for all.

    The main lever on buffer size: for a 30-body G1 tree the full key set is
    2,696 B/frame, of which the eight transition-aligned ``next_*`` duplicates
    are 568 B (21%).
    """

    macro_cache_device: str | None = None
    """Optional device for a compact offline macro-state cache.

    This fast path currently supports the ``root_qpos`` macro-state terms. It
    materializes only joint positions and the selected anchor pose, avoiding a
    scattered gather over every field in a large CPU replay buffer for each
    encoder batch. Leave ``None`` to sample macro transitions directly from the
    reference replay buffer.
    """

    macro_cache_chunk_size: int = 262_144
    """Rows copied per chunk while materializing ``macro_cache_device``."""

    runtime_cache_device: str | None = None
    """Optional device for a compact, dense low-level reference cache.

    Large persisted CPU replay buffers are memory mapped. Randomly gathering
    every full-body field from such a buffer at every simulator step is much
    slower than simulation. When this is set, the data plane sequentially
    materializes only ``qpos``, ``qvel``, and the configured body states on the
    requested device, then uses that dense buffer for live trajectory sampling.
    ``qvel`` remains an internal reference source for velocity tracking and
    resets; it does not add velocity terms to the root+qpos macro command.
    """

    runtime_cache_body_names: list[str] | None = None
    """Dataset bodies retained by ``runtime_cache_device``.

    ``None`` uses the environment's MPJPE tracking-body set. The anchor and all
    configured command bodies must be present; construction fails otherwise.
    """

    runtime_cache_chunk_size: int = 262_144
    """Rows copied per chunk while materializing ``runtime_cache_device``."""

    wrap_steps: bool = False
    """Wrap the reference cursor at the end of a clip instead of terminating."""

    def resolve(
        self,
        *,
        sim_dt: float,
        decimation: int,
        joint_names: list[str],
        canonical_joint_names: list[str],
    ) -> ResolvedMotionData | None:
        """Derive the concrete dataset layout. Pure: ``self`` is not modified.

        ``sim_dt`` and ``decimation`` are the task's declared timing, not
        something to solve for: together they fix the control rate the whole
        protocol is defined at, and every recorded result assumes it. Clips are
        checked against that rate; they never move it.

        Returns ``None`` when no data is declared at all, which is a legitimate
        state for a config nobody has pointed at a dataset yet -- a layout test,
        a job submitter, an audit that only inspects the observation surface.
        The environment cannot load without one, and says so when it tries.

        Anything else that is wrong raises here and now: a missing clip file, a
        source without body states, or a clip rate that disagrees with the
        task's control rate are configuration errors, not conditions to paper
        over with a fallback.
        """
        control_rate = _control_rate(sim_dt=sim_dt, decimation=decimation)
        if self.manifest is None:
            if self.cache_dir is None:
                return None
            return self._resolve_prebuilt_cache(control_rate=control_rate)

        manifest_path, entries = load_clip_manifest(self.manifest)
        self._validate_sources(entries)
        clip_names = tuple(str(entry["name"]) for entry in entries)
        self._validate_clip_selection(clip_names, manifest_path)

        control_freq = _manifest_clip_rate(entries, manifest_path)
        _check_clip_rate(control_freq, control_rate, source=str(manifest_path))

        loader_kwargs = build_clip_loader_kwargs(
            entries=entries,
            sim_dt=float(sim_dt),
            decimation=int(decimation),
            control_freq=control_freq,
            joint_names=list(joint_names),
            canonical_joint_names=list(canonical_joint_names),
        )
        loader_kwargs.update(load_clip_loader_options(manifest_path))

        cache_dir = (
            str(Path(self.cache_dir).expanduser().resolve())
            if self.cache_dir is not None
            else cache_dir_from_entries(
                entries,
                manifest_path=manifest_path,
                family=load_manifest_family(manifest_path),
            )
        )

        return ResolvedMotionData(
            manifest_path=str(manifest_path),
            clip_names=clip_names,
            selected_clips=tuple(self.clips) if self.clips is not None else None,
            selected_takes=tuple(self.takes) if self.takes is not None else None,
            control_freq=control_freq,
            cache_dir=cache_dir,
            loader_kwargs=loader_kwargs,
        )

    # -- resolution steps ---------------------------------------------------

    def _resolve_prebuilt_cache(self, *, control_rate: float) -> ResolvedMotionData:
        """Resolve against an already-built cache, with no manifest to read.

        A cache is built from clips that already passed the rate check, so the
        task's control rate is what it holds.
        """
        assert self.cache_dir is not None
        return ResolvedMotionData(
            manifest_path=None,
            clip_names=(),
            selected_clips=tuple(self.clips) if self.clips is not None else None,
            selected_takes=tuple(self.takes) if self.takes is not None else None,
            control_freq=control_rate,
            cache_dir=str(Path(self.cache_dir).expanduser().resolve()),
            loader_kwargs={},
        )

    def _validate_clip_selection(
        self, clip_names: tuple[str, ...], manifest_path: Path
    ) -> None:
        """Every named clip must exist in the manifest.

        A misspelled name would otherwise just narrow the reference set without
        saying so, which looks like a worse result rather than a typo.
        """
        if self.clips is None:
            return
        missing = [name for name in self.clips if name not in set(clip_names)]
        if missing:
            raise ValueError(
                f"env.data.clips names {missing} which {manifest_path} does not "
                f"declare. It has {len(clip_names)} clips; the first few are "
                f"{list(clip_names[:5])}."
            )

    def _validate_sources(self, entries: list[dict[str, Any]]) -> None:
        """Every declared clip must exist, and carry body states when required."""
        for entry in entries:
            source_path = Path(str(entry["path"])).expanduser().resolve()
            entry["path"] = str(source_path)
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Motion clip is missing: {source_path}. Point "
                    "`env.data.manifest` at a manifest whose entries resolve to "
                    "repo-local clip files."
                )
            if self.require_body_states and source_path.suffix.lower() != ".npz":
                raise ValueError(
                    "This tracking environment needs clips carrying body states "
                    "(body_pos_w / body_quat_w / body_lin_vel_w / "
                    f"body_ang_vel_w), which only NPZ sources have. Got: "
                    f"{source_path}. Convert the sources to NPZ, or set "
                    "`env.data.require_body_states=false` if the consumer truly "
                    "does not need them."
                )


@dataclass(frozen=True)
class ResolvedMotionData:
    """The derived dataset layout. Outputs only; never a configuration input.

    Produced by :meth:`MotionDataCfg.resolve` and read by the data plane. A
    plain frozen dataclass rather than a ``configclass`` on purpose: it is not
    part of the override surface and must not round-trip through Hydra.
    """

    manifest_path: str | None
    clip_names: tuple[str, ...]
    """Every clip the manifest declares, in manifest order."""

    selected_clips: tuple[str, ...] | None
    """The subset to load, or ``None`` for all of :attr:`clip_names`."""

    selected_takes: tuple[str, ...] | None
    """Takes to load within the selected clips, or ``None`` for all."""

    control_freq: float
    """Clip rate, equal to the task's control rate (they are checked to agree)."""

    cache_dir: str
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    """ILTools clip-loader call arguments; empty when loading a prebuilt cache."""

    @property
    def loader_type(self) -> str:
        return CLIP_LOADER_KEY

    @property
    def can_build(self) -> bool:
        """Whether a missing cache can be built from these sources."""
        return bool(self.loader_kwargs)

    def motions(self) -> tuple[str, ...] | None:
        """Clip-name filter for the replay-buffer build.

        The MANIFEST is the reference set, not the cache. A cache is keyed by
        manifest identity but is often a superset in practice -- an explicit
        ``cache_dir`` is how several runs share one build, and how a 91-clip
        selection reads a 100-clip cache. Filtering to the manifest's own clips
        is therefore load-bearing: without it, naming a subset manifest while
        pointing at a shared cache would silently train on everything in that
        cache and look entirely normal in the logs.

        ``None`` only when there is no manifest at all, where the prebuilt cache
        IS the declared set.
        """
        if self.selected_clips is not None:
            return self.selected_clips
        return self.clip_names or None

    def trajectories(self) -> tuple[str, ...] | None:
        """Take-name filter for the replay-buffer build."""
        return self.selected_takes


def apply_motion_data(
    env_cfg: Any,
    *,
    manifest: str | Path | None = None,
    cache_dir: str | Path | None = None,
    clips: list[str] | None = None,
    takes: list[str] | None = None,
    cache_refresh: bool | None = None,
    wrap_steps: bool | None = None,
) -> None:
    """Point an environment config at motion data, whichever generation it is.

    Evaluation drivers, collectors, and audit scripts take a ``--task`` from the
    command line, so they can be handed either the current environment (which
    configures data under ``env.data``) or a frozen v0/v1 one (which keeps the
    flat fields). One helper knowing that is the alternative to every driver
    carrying its own ``hasattr`` ladder and each drifting differently.

    Only the arguments given are applied; ``None`` leaves the config's own
    choice alone.
    """
    data_cfg = getattr(env_cfg, "data", None)
    if isinstance(data_cfg, MotionDataCfg):
        if manifest is not None:
            data_cfg.manifest = str(manifest)
        if cache_dir is not None:
            data_cfg.cache_dir = str(cache_dir)
        if clips is not None:
            data_cfg.clips = list(clips)
        if takes is not None:
            data_cfg.takes = list(takes)
        if cache_refresh is not None:
            data_cfg.cache_refresh = bool(cache_refresh)
        if wrap_steps is not None:
            data_cfg.wrap_steps = bool(wrap_steps)
        return

    # Legacy (v0/v1) surface: flat fields plus its own manifest resolution.
    if manifest is not None:
        if not hasattr(env_cfg, "lafan1_manifest_path"):
            raise TypeError(
                f"{type(env_cfg).__name__} configures no motion data: it has "
                "neither `data` (current) nor `lafan1_manifest_path` (legacy)."
            )
        env_cfg.lafan1_manifest_path = str(manifest)
    if cache_dir is not None:
        env_cfg.dataset_path = str(cache_dir)
    if manifest is not None:
        resolve_manifest = getattr(env_cfg, "_resolve_manifest_config", None)
        if callable(resolve_manifest):
            resolve_manifest(
                dataset_path_explicit=cache_dir is not None,
                motions_explicit=clips is not None,
            )
    if clips is not None:
        env_cfg.motions = list(clips)
    if takes is not None:
        env_cfg.trajectories = list(takes)
    if cache_refresh is not None:
        env_cfg.refresh_zarr_dataset = bool(cache_refresh)
    if wrap_steps is not None:
        env_cfg.wrap_steps = bool(wrap_steps)


def _manifest_clip_rate(entries: list[dict[str, Any]], manifest_path: Path) -> float:
    """The one rate a manifest's clips are sampled at.

    Clips reach training as NPZ already resampled to the task rate by
    ``scripts/data/``; source formats that still need resampling (30 Hz LAFAN1
    CSV, say) are a converter input, never a training input. So a manifest has
    exactly one rate, and a manifest that does not is malformed rather than a
    case to accommodate.
    """
    rate = infer_manifest_control_freq(entries)
    if rate is None:
        raise ValueError(
            f"{manifest_path} declares no single clip rate: its entries either "
            "disagree on `input_fps` or are not all NPZ. Clips must be converted "
            "to the task's rate before training (`scripts/data/`), so a manifest "
            "mixing rates or formats is malformed."
        )
    return rate


def _control_rate(*, sim_dt: float, decimation: int) -> float:
    """The task's control rate in Hz, from its declared physics step and decimation."""
    if sim_dt <= 0.0:
        raise ValueError(f"sim.dt must be positive; got {sim_dt}.")
    if int(decimation) < 1:
        raise ValueError(f"decimation must be >= 1; got {decimation}.")
    return 1.0 / (float(sim_dt) * int(decimation))


def _check_clip_rate(clip_rate: float, control_rate: float, *, source: str) -> None:
    """Require the clips to be sampled at the task's control rate.

    ``sim.dt`` and ``decimation`` are protocol decisions, not free variables to
    solve for: they fix the rate every reward, termination threshold, episode
    length, and recorded result is defined at. Data that disagrees is data
    prepared for a different task, so this refuses rather than retuning the
    physics rate to accommodate it -- silently doing that would make two runs
    incomparable while both look fine in the logs.

    The conversion pipeline resamples sources to the task rate
    (``scripts/data/``), so a mismatch here means the wrong manifest, not a
    reason to loosen the check.
    """
    if abs(clip_rate - control_rate) <= _TIMING_TOLERANCE_HZ:
        return
    raise ValueError(
        f"Clips are {clip_rate:g} Hz but this task runs at {control_rate:g} Hz "
        f"(sim.dt x decimation). Source: {source}. Convert the clips to "
        f"{control_rate:g} Hz with `scripts/data/`, or point `env.data.manifest` "
        "at a manifest prepared for this task. Changing `env.sim.dt` or "
        "`env.decimation` to match the data instead would silently redefine the "
        "protocol every recorded result was measured under."
    )


__all__ = [
    "MotionDataCfg",
    "ResolvedMotionData",
    "apply_motion_data",
]
