#!/usr/bin/env python3
"""Build, publish, and fetch the prebuilt reference replay buffer.

Why this exists
---------------
Training reads the reference motion set through a TorchRL replay buffer. Two
costs make that buffer worth treating as a first-class, published artifact
rather than something each job reconstructs:

* Filling it from Zarr costs about 66 ms per trajectory plus 53 us per frame.
  For the 129,785-clip BONES-SEED set that is roughly 3 h -- on every job start.
* Building the Zarr from NPZ first costs about 91.5 ms per clip plus 72 us per
  frame, roughly another 4.3 h.

Persisting the filled buffer collapses both to a memory-map: measured 26.27 s
to build versus 0.16 s to reopen, byte-identical, with the Zarr absent. So the
Zarr is consumed exactly once, ever, and never has to reach a compute node.

The published artifact is CPU memmap files, which is the portable on-disk form.
At train time `env.dataset_storage_device=cuda:0` materializes it into VRAM with
one sequential read, so sampling stays GPU-resident and training throughput is
unchanged (measured 8.28-8.38 s versus an 8.34 s baseline over 5 iterations).
Sampling a memmap directly is about 4x slower end-to-end; do not train off the
memmap when the set fits in VRAM.

Stages
------
    build   NPZ manifest -> Zarr -> persisted CPU memmap buffer
    pack    split the buffer into upload-sized parts + hashed index
    push    upload the packed parts to a Hugging Face dataset repo
    fetch   download + reassemble + verify (run this on the compute node)
    verify  reopen a buffer in place and check it against its manifest

Usage
-----
    # one-time, on a machine with the NPZ tree and plenty of disk
    pixi run python experiments/paper/reference_buffer_workflow.py \
        stages='[build,pack,push]'

    # on a compute node, before training
    pixi run python experiments/paper/reference_buffer_workflow.py \
        stages='[fetch]' paths.fetch_dest=/tmp/rb

    # then point training at it
    ... env.dataset_storage_device=cuda:0 \
        env.dataset_storage_persist_dir=/tmp/rb \
        env.dataset_storage_persist_id=<buffer.persist_id>

Every parameter lives in `conf/reference_buffer.yaml` next to this file.
Heavy imports are deferred per stage so `fetch` works on a node that has only
`huggingface_hub`, without Isaac Lab or iltools installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

LOGGER = logging.getLogger("reference_buffer")

#: Written by `pack`, consumed by `fetch`. Names every part and its digest.
PACK_INDEX_NAME = "buffer_pack_index.json"
#: Sidecar that `iltools.datasets.utils.make_rb_from` writes next to a buffer.
RB_MANIFEST_NAME = "iltools_rb_manifest.json"
#: TensorDict's own layout descriptor. Must survive pack/fetch untouched.
TD_META_NAME = "meta.json"
#: Written into a Zarr directory only after a build completes. Its presence is
#: the only evidence that a multi-hour build was not interrupted.
ZARR_COMPLETE_NAME = ".zarr_build_complete.json"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _expand(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def _sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _resolve_token(cfg: DictConfig) -> str | None:
    """Token from the configured file, then HF_TOKEN, then the cached login."""
    token_file = cfg.hf.get("token_file")
    if token_file:
        candidate = _expand(token_file)
        if candidate.is_file():
            token = candidate.read_text().strip()
            if token:
                return token
    return os.environ.get("HF_TOKEN") or None


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# stage: build
# --------------------------------------------------------------------------
def stage_build(cfg: DictConfig) -> None:
    """NPZ manifest -> Zarr -> persisted CPU memmap replay buffer."""
    from iltools.datasets.utils import make_rb_from

    manifest_path = _expand(cfg.paths.manifest)
    zarr_path = _expand(cfg.paths.zarr)
    buffer_path = _expand(cfg.paths.buffer)

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    num_motions = manifest.get("metadata", {}).get("num_motions", "unknown")
    manifest_sha = _sha256_file(manifest_path)
    LOGGER.info("Manifest %s (%s motions)", manifest_path, num_motions)
    LOGGER.info("Manifest sha256 %s", manifest_sha)

    # ---- Zarr -------------------------------------------------------------
    if bool(cfg.buffer.rebuild_zarr) and zarr_path.exists():
        LOGGER.info("rebuild_zarr=true; removing %s", zarr_path)
        shutil.rmtree(zarr_path)

    marker_path = zarr_path / ZARR_COMPLETE_NAME
    if zarr_path.exists() and marker_path.is_file():
        marker = json.loads(marker_path.read_text())
        if marker.get("manifest_sha256") != manifest_sha:
            raise RuntimeError(
                f"Zarr at {zarr_path} was built from manifest "
                f"{marker.get('manifest_sha256')}, but this run uses "
                f"{manifest_sha}. Pass buffer.rebuild_zarr=true to rebuild it, "
                "or point paths.zarr somewhere else."
            )
        LOGGER.info(
            "Reusing complete Zarr at %s (%s motions)",
            zarr_path,
            marker.get("num_motions"),
        )
    elif zarr_path.exists() and any(zarr_path.iterdir()):
        # A Zarr build takes hours. Without this check a run killed partway
        # leaves a directory that merely *exists*, and the next run would train
        # on a silently truncated reference set.
        raise RuntimeError(
            f"Zarr at {zarr_path} exists but has no {ZARR_COMPLETE_NAME} marker, "
            "so it is an incomplete or externally-produced build. Pass "
            "buffer.rebuild_zarr=true to rebuild it, or write the marker "
            "yourself if you are certain it is complete."
        )
    else:
        from iltools.datasets.lafan1.loader import Lafan1CsvLoader

        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        loader_cfg = OmegaConf.create({"dataset": manifest["dataset"]})
        relative = bool(
            manifest.get("metadata", {}).get("paths_are_relative_to_manifest", False)
        )
        # Motion paths in the manifest are stored relative to the manifest, so
        # the loader only resolves them from that directory. Restore the old cwd
        # afterwards: later stages use paths taken from the Hydra config.
        previous_cwd = Path.cwd()
        if relative:
            os.chdir(manifest_path.parent)
        started = time.perf_counter()
        try:
            Lafan1CsvLoader(
                cfg=loader_cfg,
                build_zarr_dataset=True,
                zarr_path=str(zarr_path),
            )
        finally:
            os.chdir(previous_cwd)
        # Written only after the loader returns, so the marker's presence is
        # exactly the statement "this Zarr is complete".
        marker_path.write_text(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "manifest_sha256": manifest_sha,
                    "num_motions": num_motions,
                }
            )
        )
        LOGGER.info(
            "Built Zarr in %.1f min (%s)",
            (time.perf_counter() - started) / 60.0,
            _human(_dir_bytes(zarr_path)),
        )

    # ---- replay buffer ----------------------------------------------------
    keys = cfg.buffer.get("keys")
    keys = list(keys) if keys is not None else None
    buffer_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    _, traj_info = make_rb_from(
        zarr_path=str(zarr_path),
        keys=keys,
        # Always CPU here: the persisted form is memmap files on disk. Training
        # chooses where the runtime buffer lives via env.dataset_storage_device.
        device="cpu",
        persist_dir=str(buffer_path),
        persist_id=str(cfg.buffer.persist_id),
        persist_rebuild=bool(cfg.buffer.rebuild),
        verbose_tree=False,
        prefetch=0,
    )
    LOGGER.info(
        "Buffer ready in %.1f min: %s transitions, %s trajectories, %s at %s",
        (time.perf_counter() - started) / 60.0,
        traj_info["written"],
        len(traj_info["ordered_traj_list"]),
        _human(_dir_bytes(buffer_path)),
        buffer_path,
    )
    if keys is not None:
        LOGGER.info(
            "Buffer holds a key subset; training must also set "
            "env.reconstructed_reference_action=false and "
            "env.observations.policy_supervision=null when next_* keys are absent."
        )


# --------------------------------------------------------------------------
# stage: pack
# --------------------------------------------------------------------------
def stage_pack(cfg: DictConfig) -> None:
    """Split the buffer into upload-sized parts and write a hashed index."""
    buffer_path = _expand(cfg.paths.buffer)
    packed_path = _expand(cfg.paths.packed)
    part_bytes = int(cfg.pack.part_bytes)
    hash_parts = bool(cfg.pack.hash_parts)

    if not (buffer_path / RB_MANIFEST_NAME).is_file():
        raise FileNotFoundError(
            f"{buffer_path} has no {RB_MANIFEST_NAME}; run the build stage first. "
            "A buffer without its manifest is an incomplete fill."
        )

    if packed_path.exists():
        shutil.rmtree(packed_path)
    packed_path.mkdir(parents=True)

    members: list[dict[str, Any]] = []
    sources = sorted(p for p in buffer_path.iterdir() if p.is_file())
    started = time.perf_counter()

    for source in sources:
        size = source.stat().st_size
        # The two small JSON descriptors are copied verbatim so a packed
        # directory is inspectable without reassembly.
        if source.name in {RB_MANIFEST_NAME, TD_META_NAME}:
            shutil.copy2(source, packed_path / source.name)
            members.append(
                {
                    "name": source.name,
                    "size": size,
                    "verbatim": True,
                    "sha256": _sha256_file(source),
                    "parts": [source.name],
                }
            )
            continue

        parts: list[str] = []
        digest = hashlib.sha256() if hash_parts else None
        with source.open("rb") as handle:
            index = 0
            while True:
                chunk = handle.read(part_bytes)
                if not chunk:
                    break
                part_name = f"{source.name}.part{index:04d}"
                (packed_path / part_name).write_bytes(chunk)
                if digest is not None:
                    digest.update(chunk)
                parts.append(part_name)
                index += 1
        if not parts:
            # A zero-byte member still needs a part so reassembly recreates it.
            part_name = f"{source.name}.part0000"
            (packed_path / part_name).write_bytes(b"")
            parts.append(part_name)
        members.append(
            {
                "name": source.name,
                "size": size,
                "verbatim": False,
                "sha256": digest.hexdigest() if digest is not None else None,
                "parts": parts,
            }
        )
        LOGGER.info(
            "Packed %s (%s) into %d part(s)", source.name, _human(size), len(parts)
        )

    index_payload = {
        "format_version": 1,
        "persist_id": str(cfg.buffer.persist_id),
        "part_bytes": part_bytes,
        "total_bytes": sum(m["size"] for m in members),
        "members": members,
    }
    (packed_path / PACK_INDEX_NAME).write_text(json.dumps(index_payload, indent=2))
    LOGGER.info(
        "Packed %s across %d members in %.1f min -> %s",
        _human(index_payload["total_bytes"]),
        len(members),
        (time.perf_counter() - started) / 60.0,
        packed_path,
    )


# --------------------------------------------------------------------------
# stage: push
# --------------------------------------------------------------------------
def stage_push(cfg: DictConfig) -> None:
    """Upload the packed parts to a Hugging Face dataset repo."""
    from huggingface_hub import HfApi

    packed_path = _expand(cfg.paths.packed)
    if not (packed_path / PACK_INDEX_NAME).is_file():
        raise FileNotFoundError(
            f"{packed_path} has no {PACK_INDEX_NAME}; run the pack stage first."
        )

    if bool(cfg.hf.disable_xet):
        # Must be set before the upload machinery initialises.
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    api = HfApi(token=_resolve_token(cfg))
    whoami = api.whoami()
    LOGGER.info(
        "Authenticated as %s (token role %s)",
        whoami.get("name"),
        whoami.get("auth", {}).get("accessToken", {}).get("role"),
    )
    api.create_repo(
        repo_id=str(cfg.hf.repo_id),
        repo_type=str(cfg.hf.repo_type),
        private=bool(cfg.hf.private),
        exist_ok=True,
    )
    LOGGER.info(
        "Uploading %s to %s (repo root; upload_large_folder takes no path prefix)",
        _human(_dir_bytes(packed_path)),
        cfg.hf.repo_id,
    )
    started = time.perf_counter()
    api.upload_large_folder(
        folder_path=str(packed_path),
        repo_id=str(cfg.hf.repo_id),
        repo_type=str(cfg.hf.repo_type),
    )
    LOGGER.info("Upload finished in %.1f min", (time.perf_counter() - started) / 60.0)


# --------------------------------------------------------------------------
# stage: fetch
# --------------------------------------------------------------------------
def stage_fetch(cfg: DictConfig) -> None:
    """Obtain, reassemble, and verify a published buffer on this machine.

    Parts come from Hugging Face by default, or from `fetch.local_source` when
    the packed directory is already reachable -- e.g. sitting on a shared
    filesystem, where reassembling straight into node-local scratch beats a
    round trip through the Hub.
    """
    dest = _expand(cfg.paths.fetch_dest)
    local_source = cfg.fetch.get("local_source")

    if local_source:
        snapshot_path = _expand(local_source)
        if not snapshot_path.is_dir():
            raise FileNotFoundError(
                f"fetch.local_source not a directory: {snapshot_path}"
            )
        LOGGER.info("Reassembling from local source %s", snapshot_path)
    else:
        from huggingface_hub import snapshot_download

        cache = _expand(cfg.paths.fetch_cache)
        if bool(cfg.hf.disable_xet):
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        started = time.perf_counter()
        cache.mkdir(parents=True, exist_ok=True)
        snapshot_path = Path(
            snapshot_download(
                repo_id=str(cfg.hf.repo_id),
                repo_type=str(cfg.hf.repo_type),
                revision=str(cfg.hf.revision),
                local_dir=str(cache),
                token=_resolve_token(cfg),
            )
        )
        LOGGER.info("Downloaded in %.1f min", (time.perf_counter() - started) / 60.0)

    # The index may sit under path_in_repo rather than at the snapshot root.
    candidates = sorted(snapshot_path.rglob(PACK_INDEX_NAME))
    if not candidates:
        raise FileNotFoundError(f"No {PACK_INDEX_NAME} inside {snapshot_path}")
    index_path = candidates[0]
    parts_root = index_path.parent
    index_payload = json.loads(index_path.read_text())

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    verify = bool(cfg.fetch.verify)
    started = time.perf_counter()
    for member in index_payload["members"]:
        target = dest / member["name"]
        if member["verbatim"]:
            shutil.copy2(parts_root / member["parts"][0], target)
        else:
            with target.open("wb") as out:
                for part_name in member["parts"]:
                    part_path = parts_root / part_name
                    if not part_path.is_file():
                        raise FileNotFoundError(f"Missing part {part_path}")
                    with part_path.open("rb") as handle:
                        shutil.copyfileobj(handle, out, length=8 << 20)
        actual_size = target.stat().st_size
        if actual_size != member["size"]:
            raise RuntimeError(
                f"{member['name']}: reassembled {actual_size} bytes, "
                f"expected {member['size']}"
            )
        if verify and member.get("sha256"):
            actual = _sha256_file(target)
            if actual != member["sha256"]:
                raise RuntimeError(
                    f"{member['name']}: sha256 {actual} != {member['sha256']}"
                )
    LOGGER.info(
        "Reassembled %s in %.1f min -> %s",
        _human(index_payload["total_bytes"]),
        (time.perf_counter() - started) / 60.0,
        dest,
    )

    if bool(cfg.fetch.cleanup_parts) and not local_source:
        shutil.rmtree(_expand(cfg.paths.fetch_cache), ignore_errors=True)
        LOGGER.info("Removed download staging directory %s", cfg.paths.fetch_cache)

    if bool(cfg.fetch.verify_open):
        _verify_buffer(dest, str(cfg.buffer.persist_id), cfg)

    LOGGER.info(
        "Train against it with: env.dataset_storage_device=cuda:0 "
        "env.dataset_storage_persist_dir=%s env.dataset_storage_persist_id=%s",
        dest,
        cfg.buffer.persist_id,
    )


# --------------------------------------------------------------------------
# stage: verify
# --------------------------------------------------------------------------
def _verify_buffer(buffer_path: Path, persist_id: str, cfg: DictConfig) -> None:
    """Reopen a buffer and check it agrees with its own manifest."""
    from iltools.datasets.utils import make_rb_from

    manifest = json.loads((buffer_path / RB_MANIFEST_NAME).read_text())
    recorded = manifest["key"]["source"].get("persist_id")
    if recorded != persist_id:
        raise RuntimeError(
            f"Buffer at {buffer_path} carries persist_id {recorded!r}, "
            f"expected {persist_id!r}."
        )

    keys = cfg.buffer.get("keys")
    started = time.perf_counter()
    _, traj_info = make_rb_from(
        # Deliberately bogus: a correct reuse must never touch the Zarr.
        zarr_path="/nonexistent/zarr",
        keys=list(keys) if keys is not None else None,
        device="cpu",
        persist_dir=str(buffer_path),
        persist_id=persist_id,
        verbose_tree=False,
        prefetch=0,
    )
    LOGGER.info(
        "Reopened in %.2f s without the Zarr: %s transitions, %s trajectories.",
        time.perf_counter() - started,
        traj_info["written"],
        len(traj_info["ordered_traj_list"]),
    )


def stage_verify(cfg: DictConfig) -> None:
    _verify_buffer(_expand(cfg.paths.buffer), str(cfg.buffer.persist_id), cfg)


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
STAGES = {
    "build": stage_build,
    "pack": stage_pack,
    "push": stage_push,
    "fetch": stage_fetch,
    "verify": stage_verify,
}


@hydra.main(version_base=None, config_path="conf", config_name="reference_buffer")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    requested = list(cfg.stages)
    unknown = [name for name in requested if name not in STAGES]
    if unknown:
        raise SystemExit(
            f"Unknown stage(s): {', '.join(unknown)}. Choose from: {', '.join(STAGES)}"
        )

    LOGGER.info("Stages: %s", " -> ".join(requested))
    LOGGER.info("persist_id: %s", cfg.buffer.persist_id)
    for name in requested:
        LOGGER.info("---- stage %s ----", name)
        started = time.perf_counter()
        STAGES[name](cfg)
        LOGGER.info(
            "---- stage %s done in %.1f min ----",
            name,
            (time.perf_counter() - started) / 60.0,
        )


if __name__ == "__main__":
    main()
