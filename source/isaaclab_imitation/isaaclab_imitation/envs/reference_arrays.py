# Copyright (c) 2024-2025, The Isaac Lab Imitation Project Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read side of the prebuilt, training-shaped reference arrays.

The writer is
``imitation_experiments.data.build_reference_arrays``. It is deliberately not
imported here: that package is the experiment layer above this extension, so
depending on it would invert the repo's layering, and its builder must run in
the default Pixi environment without Isaac Lab. Only the sidecar's file name and
its four semantic key names are restated below; every array's shape and dtype is
read out of the sidecar itself, so the two sides cannot drift on layout.

What this buys, on the 129,785-motion BONES-SEED set: the `root_qpos` macro
cache and the dense runtime cache are otherwise re-derived inside every training
process by gathering over a 94.5 GiB, 30-body replay -- about 133 GB of reads to
keep about 55 GB, with ``body_pos_w`` and ``body_quat_w`` read twice, once by
each. These arrays are already in the layout both caches want, so the runtime
cache is memory-mapped in place instead of copied, and the macro cache is one
contiguous read.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import torch


logger = logging.getLogger(__name__)

#: Paired with ``build_reference_arrays.SIDECAR_NAME``.
SIDECAR_NAME = "reference_arrays_manifest.json"
FORMAT_VERSION = 1

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}

#: Fields the dense runtime reference cache is built from, in the order
#: ``_materialize_runtime_reference_cache`` declares them.
RUNTIME_FIELDS = (
    "qpos",
    "qvel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

#: Fields the ``root_qpos`` macro cache is built from. ``joint_pos`` is
#: ``qpos[:, 7:]``: storing it separately would duplicate 5.5 GB to avoid a
#: 1.24x read amplification.
MACRO_FIELDS = ("anchor_pos_w", "anchor_quat_w")


class ReferenceArrayStore:
    """A validated, memory-mapped view of one built reference-array directory."""

    def __init__(self, directory: Path, sidecar: dict[str, Any]) -> None:
        self._directory = directory
        self._sidecar = sidecar
        self._key = sidecar["key"]
        self._arrays: dict[str, torch.Tensor] = {}

    # -- construction -------------------------------------------------------

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        body_names: list[str],
        anchor_body: str,
        persist_id: str | None = None,
    ) -> ReferenceArrayStore:
        """Open a directory, refusing anything built for different content.

        The checks are the point. A reference-array directory is keyed by the
        body list and anchor body baked into it, and pairing one with an
        environment that expects a different set would silently train against
        the wrong columns rather than fail.
        """
        directory = Path(directory).expanduser().resolve()
        sidecar_path = directory / SIDECAR_NAME
        if not sidecar_path.is_file():
            raise FileNotFoundError(
                f"No {SIDECAR_NAME} in {directory}. A build that was interrupted "
                "writes its arrays but not the sidecar; quarantine that directory "
                "and build into a fresh, versioned one."
            )
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not read {sidecar_path}: {error}") from error

        if sidecar.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(
                f"{sidecar_path} has format_version="
                f"{sidecar.get('format_version')!r}, expected {FORMAT_VERSION}."
            )
        key = sidecar.get("key")
        if not isinstance(key, dict):
            raise RuntimeError(f"{sidecar_path} has no key object.")

        stored_bodies = [str(name) for name in key.get("body_names", [])]
        if stored_bodies != [str(name) for name in body_names]:
            raise RuntimeError(
                f"{directory} was built for bodies {stored_bodies} but this "
                f"environment tracks {list(body_names)}. Order matters: these are "
                "column positions, not a set. Build a directory for this body set."
            )
        # The anchor is the one part of the identity that does not have to
        # match. Its pose lives in `body_pos_w`/`body_quat_w` too, so any anchor
        # inside the retained body set can be derived at load time. That matters
        # for a published artifact: without it, a second anchor would mean a
        # second 49.4 GB upload. Anything outside the set still fails, because
        # those columns genuinely are not there.
        stored_anchor = key.get("anchor_body")
        if stored_anchor is not None and str(stored_anchor) != str(anchor_body):
            if str(anchor_body) not in stored_bodies:
                raise RuntimeError(
                    f"{directory} was built with anchor body {stored_anchor!r} "
                    f"and retains bodies {stored_bodies}, so it cannot serve an "
                    f"environment anchored on {anchor_body!r}. Rebuild with "
                    f"--anchor_body {anchor_body}, or add it to --body_names."
                )
            logger.warning(
                "%s bakes anchor %r but this environment anchors on %r; deriving "
                "the anchor pose from the retained body arrays instead. That is "
                "correct but reads the full body block once, so prefer a "
                "directory built with --anchor_body %s for repeated runs.",
                directory,
                stored_anchor,
                anchor_body,
                anchor_body,
            )
        # A directory built with no anchor body is legitimate -- not every
        # dataset needs one -- and only fails when the macro cache asks for the
        # anchor arrays, which say so by name.
        if persist_id is not None:
            source = key.get("source")
            if source != {"persist_id": persist_id}:
                raise RuntimeError(
                    f"{directory} was built from source {source!r}, but "
                    f"env.data.persist_id declares {persist_id!r}."
                )

        store = cls(directory, sidecar)
        store._validate_sizes()
        return store

    def _validate_sizes(self) -> None:
        """Catch a truncated or half-written array before training reads it."""
        for name, spec in self._key["arrays"].items():
            path = self._directory / f"{name}.memmap"
            if not path.is_file():
                raise FileNotFoundError(f"{self._directory} is missing {path.name}.")
            shape = tuple(int(value) for value in spec["shape"])
            dtype = self._torch_dtype(spec["dtype"])
            expected = math.prod(shape) * dtype.itemsize
            actual = path.stat().st_size
            if actual != expected:
                raise RuntimeError(
                    f"{path} is {actual} bytes but its sidecar declares shape "
                    f"{shape} of {spec['dtype']} ({expected} bytes)."
                )

    @staticmethod
    def _torch_dtype(name: str) -> torch.dtype:
        dtype = _DTYPES.get(str(name))
        if dtype is None:
            raise RuntimeError(f"Unsupported reference-array dtype {name!r}.")
        return dtype

    # -- metadata -----------------------------------------------------------

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def body_names(self) -> list[str]:
        return [str(name) for name in self._key["body_names"]]

    @property
    def joint_names(self) -> list[str]:
        return [str(name) for name in self._key.get("joint_names", [])]

    @property
    def anchor_body(self) -> str:
        return str(self._key["anchor_body"])

    def anchor_source(self, anchor_body: str) -> int | None:
        """Body index to derive the anchor pose from, or ``None`` if baked.

        ``None`` means ``anchor_pos_w``/``anchor_quat_w`` already hold this
        anchor and can be read directly; an index means they hold a different
        one and the pose must come out of the body arrays at that column.
        """
        stored = self._key.get("anchor_body")
        if stored is not None and str(stored) == str(anchor_body):
            return None
        names = self.body_names
        if str(anchor_body) not in names:
            raise KeyError(
                f"{self._directory} does not retain body {anchor_body!r}; it has "
                f"{names}."
            )
        return names.index(str(anchor_body))

    @property
    def available_arrays(self) -> frozenset[str]:
        """Arrays this directory holds.

        Not every source carries every field -- a dataset without body
        velocities, or one built with no anchor body, is a smaller set -- so
        consumers check for what they need and say what is absent.
        """
        return frozenset(self._key["arrays"])

    @property
    def num_rows(self) -> int:
        return int(self._sidecar["traj_info"]["written"])

    @property
    def total_bytes(self) -> int:
        return sum(
            math.prod(int(value) for value in spec["shape"])
            * self._torch_dtype(spec["dtype"]).itemsize
            for spec in self._key["arrays"].values()
        )

    def traj_info(self) -> dict[str, Any]:
        """Trajectory spans in the shape ``ParallelTrajectoryManager`` expects."""
        traj_info = dict(self._sidecar["traj_info"])
        traj_info["ordered_traj_list"] = [
            tuple(entry) for entry in traj_info["ordered_traj_list"]
        ]
        return traj_info

    # -- data ---------------------------------------------------------------

    def array(self, name: str) -> torch.Tensor:
        """Memory-map one array. Repeated calls share the same mapping."""
        cached = self._arrays.get(name)
        if cached is not None:
            return cached
        spec = self._key["arrays"].get(name)
        if spec is None:
            raise KeyError(
                f"{self._directory} has no array {name!r}; it holds "
                f"{sorted(self._key['arrays'])}."
            )
        shape = tuple(int(value) for value in spec["shape"])
        dtype = self._torch_dtype(spec["dtype"])
        tensor = torch.from_file(
            str(self._directory / f"{name}.memmap"),
            shared=True,
            size=math.prod(shape),
            dtype=dtype,
        ).view(shape)
        self._arrays[name] = tensor
        return tensor

    def warm(self, names: tuple[str, ...] | None = None, *, workers: int = 8) -> float:
        """Fault the mappings in with one parallel sequential pass.

        Without this the first training iterations pay the page faults inline,
        interleaved with simulation. Reading through separate file descriptors
        populates the same page cache the mappings resolve against, and threads
        scale because the work is I/O, not Python.
        """
        selected = names or tuple(self._key["arrays"])
        started = time.perf_counter()

        def _read(path: Path, start: int, end: int) -> None:
            buffer = bytearray(8 << 20)
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start
                while remaining > 0:
                    read = handle.readinto(
                        memoryview(buffer)[: min(remaining, len(buffer))]
                    )
                    if not read:
                        break
                    remaining -= read

        jobs: list[tuple[Path, int, int]] = []
        for name in selected:
            path = self._directory / f"{name}.memmap"
            size = path.stat().st_size
            span = max(size // max(workers, 1), 1)
            for start in range(0, size, span):
                jobs.append((path, start, min(start + span, size)))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda job: _read(*job), jobs))

        elapsed = time.perf_counter() - started
        total = sum(end - start for _path, start, end in jobs)
        logger.info(
            "Warmed %.1f GiB of reference arrays in %.1f s (%.0f MB/s) with %s threads.",
            total / 1024**3,
            elapsed,
            total / 1e6 / max(elapsed, 1e-9),
            workers,
        )
        return elapsed


def copy_to_device_parallel(
    source: torch.Tensor,
    *,
    device: torch.device,
    workers: int = 8,
    chunk_rows: int = 262_144,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Copy a large row-major tensor to ``device`` with concurrent readers.

    The single-threaded chunk loop this replaces is page-fault bound, not
    bandwidth bound: one thread sustains roughly 2 GB/s against a device that
    does far more with a deep queue. ``copy_`` releases the GIL, so threads
    genuinely overlap here.

    ``transform`` runs per chunk, before the copy, so selecting a column out of
    a large array never materializes the whole selection in host memory.
    """
    total = int(source.shape[0])
    probe = source[:1] if transform is None else transform(source[:1])
    target = torch.empty((total, *probe.shape[1:]), dtype=probe.dtype, device=device)
    if total == 0:
        return target
    bounds = [
        (start, min(start + chunk_rows, total))
        for start in range(0, total, max(chunk_rows, 1))
    ]

    def _copy(bound: tuple[int, int]) -> None:
        start, end = bound
        chunk = source[start:end]
        if transform is not None:
            chunk = transform(chunk)
        target[start:end].copy_(chunk.to(device=device))

    if len(bounds) == 1 or workers <= 1:
        for bound in bounds:
            _copy(bound)
        return target
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_copy, bounds))
    return target
