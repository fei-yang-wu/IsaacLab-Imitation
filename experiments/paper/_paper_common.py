"""Shared plumbing for the paper pipeline and evaluation entrypoints.

Every stage script in ``experiments/paper/pipeline/`` and ``experiments/paper/eval/``
is an orchestrator: it turns one Hydra config into an argv for an existing
training/collection/evaluation entrypoint, runs it, and writes a provenance
record next to the outputs. This module holds the parts they all need so the
records stay comparable across stages.

Two rules this module exists to enforce:

* **A stage that ran must be reconstructable from its record alone.** Every
  record carries the resolved config, the exact argv, the hash of every declared
  input file, the git state, and the hash of the entrypoint that produced it.
  Reruns that differ in any of those are detectably different runs.
* **A missing input fails before the simulator starts.** Isaac Sim takes minutes
  to boot; discovering a typo'd checkpoint path after that wastes a scheduler
  slot, and on ICE a wasted slot can mean a wasted 16-hour window.

Scripts import this by inserting their parent directory on ``sys.path``:

    _PAPER_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PAPER_DIR))
    from _paper_common import PipelineError, run_command
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

RECORD_SCHEMA_VERSION = 1


class PipelineError(RuntimeError):
    """A gate failed. Deliberately fatal: a partial run must not look complete."""


# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root containing ``pixi.toml`` and ``source/``.

    Marker-based rather than ``parents[N]``: this directory has been moved twice
    already, and each move silently broke every hard-coded nesting depth.
    """
    origin = (start or Path(__file__)).resolve()
    for candidate in [origin, *origin.parents]:
        if (candidate / "pixi.toml").is_file() and (candidate / "source").is_dir():
            return candidate
    raise PipelineError(f"Could not locate the repository root above {origin}")


REPO_ROOT = find_repo_root()
PAPER_DIR = REPO_ROOT / "experiments/paper"

#: The shared planner implementation now lives in the installable
#: ``imitation_experiments`` package; the campaign directory keeps only the
#: frozen shell launchers.
INTERFACE_BASELINES = (
    REPO_ROOT
    / "experiments/campaigns/2026-07-23-bones-phase5-language-local10/interface_baselines"
)
IMITATION_EXPERIMENTS_PKG = REPO_ROOT / "source/imitation_experiments/imitation_experiments"

SCRIPTS_RLOPT = REPO_ROOT / "scripts/rlopt"


# --------------------------------------------------------------------------
# Hashing and provenance
# --------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, streamed so multi-GB checkpoints are fine."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def file_provenance(path: str | Path, *, hash_file: bool = True) -> dict[str, Any]:
    """Describe one input artifact: resolved path, size, and content hash."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise PipelineError(f"Cannot record provenance for a missing file: {resolved}")
    record: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
    }
    if hash_file:
        record["sha256"] = sha256_file(resolved)
    return record


def git_state() -> dict[str, Any]:
    """Best-effort git description of the working tree.

    Cluster jobs run from an extracted tarball with no ``.git``, so every field
    is optional and a failure here is recorded rather than raised.
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "available": head is not None,
        "commit": head,
        "branch": _git("branch", "--show-current"),
        "dirty": None if status is None else bool(status.strip()),
    }


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def seed_environment(seed: int, *, deterministic: bool = True) -> dict[str, str]:
    """Environment variables that pin RNG behaviour in the child process.

    The stage scripts never train in-process, so seeding here means seeding what
    the child sees. ``CUBLAS_WORKSPACE_CONFIG`` is what makes cuBLAS reductions
    reproducible; without it ``torch.use_deterministic_algorithms`` raises on
    matmul-heavy workloads. Set ``deterministic: false`` when throughput matters
    more than bitwise reproducibility (long low-level training runs).
    """
    env = {"PYTHONHASHSEED": str(seed)}
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return env


# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------


def to_container(cfg: Any) -> Any:
    """Resolve an OmegaConf node to plain Python (interpolations expanded)."""
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def opt_str(value: Any) -> str | None:
    """Treat Hydra's empty-ish values as absent rather than as the string 'None'."""
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "null", "None"}:
        return None
    return text


def opt_path(value: Any) -> Path | None:
    text = opt_str(value)
    return None if text is None else Path(os.path.expandvars(text)).expanduser()


def require_file(value: Any, what: str) -> Path:
    """Resolve a required input path, failing loudly and early when absent."""
    path = opt_path(value)
    if path is None:
        raise PipelineError(f"{what} is required but was not set.")
    if not path.exists():
        raise PipelineError(f"{what} does not exist: {path}")
    return path


def refuse_existing_output(path: str | Path, *, allow_existing: bool, what: str) -> Path:
    """Refuse a populated output directory so a partial rerun cannot pass as complete.

    Mirrors the gate used by the aggregators and the release-bundle builder.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()) and not allow_existing:
        raise PipelineError(
            f"{what} already exists and is not empty: {resolved}\n"
            "Refusing so a partial rerun cannot be mistaken for a complete one. "
            "Set allow_existing_output=true only for an intentional, audited resume."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def hydra_override(key: str, value: Any) -> str:
    """Render one ``key=value`` Hydra override.

    Lists become the bracketed form Hydra parses (``[a,b]``); ``None`` becomes
    the explicit ``null`` that clears a config field.
    """
    if value is None:
        return f"{key}=null"
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    if isinstance(value, (list, tuple, ListConfig)):
        inner = ",".join(str(item) for item in to_container(value))
        return f"{key}=[{inner}]"
    return f"{key}={value}"


def extend_overrides(cmd: list[str], extra: Any) -> list[str]:
    """Append the free-form override escape hatch, if the config supplied one."""
    for item in to_container(extra) or []:
        text = opt_str(item)
        if text:
            cmd.append(text)
    return cmd


# --------------------------------------------------------------------------
# Interpreters
# --------------------------------------------------------------------------


def resolve_interpreter(cfg: DictConfig, kind: str) -> list[str]:
    """Return the argv prefix that runs a Python script for ``kind``.

    ``isaac`` needs the ``isaaclab`` Pixi environment (or, inside the cluster
    container, ``/isaac-sim/python.sh``); ``plain`` must NOT, because importing
    TorchRL under Isaac Sim pulls in Omniverse for no reason. Both are config
    values so a cluster profile can swap them without touching the scripts.
    """
    if kind not in {"isaac", "plain"}:
        raise PipelineError(f"Unknown interpreter kind {kind!r}")
    raw = cfg.python.get(kind)
    if isinstance(raw, (list, ListConfig)):
        parts = [str(item) for item in to_container(raw)]
    else:
        parts = shlex.split(str(raw))
    if not parts:
        raise PipelineError(f"python.{kind} resolved to an empty command")
    return parts


# --------------------------------------------------------------------------
# Command execution
# --------------------------------------------------------------------------


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_command(
    cmd: Sequence[str],
    *,
    dry_run: bool,
    log_path: Path | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    what: str = "stage",
) -> int:
    """Run one child command from the repository root, streaming and logging output.

    Returns the exit status. Raises :class:`PipelineError` on failure so a broken
    stage stops the pipeline instead of leaving a downstream stage to fail with a
    confusing missing-input error.
    """
    print(f"[CMD] {format_command(cmd)}", flush=True)
    if dry_run:
        print(f"[DRY-RUN] {what} not executed", flush=True)
        return 0

    child_env = dict(os.environ)
    child_env.update({str(k): str(v) for k, v in (env or {}).items()})

    handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        handle.write(f"# {format_command(cmd)}\n")
        handle.flush()

    try:
        process = subprocess.Popen(
            [str(part) for part in cmd],
            cwd=str(cwd or REPO_ROOT),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if handle is not None:
                handle.write(line)
        status = process.wait()
    finally:
        if handle is not None:
            handle.close()

    if status != 0:
        raise PipelineError(
            f"{what} failed with exit status {status}.\n"
            f"  command: {format_command(cmd)}"
            + (f"\n  log: {log_path}" if log_path else "")
        )
    return status


# --------------------------------------------------------------------------
# Stage records
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_stage_record(
    output_dir: Path,
    *,
    stage: str,
    entrypoint: Path,
    cfg: DictConfig | Mapping[str, Any],
    command: Sequence[str] | None,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    status: str = "complete",
    filename: str = "stage_record.json",
) -> Path:
    """Write the provenance record for one stage and return its path.

    ``inputs`` values may be paths (hashed here) or already-built dicts.
    """
    resolved_inputs: dict[str, Any] = {}
    for name, value in (inputs or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            resolved_inputs[name] = file_provenance(value)
        else:
            resolved_inputs[name] = to_container(value)

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "recorded_at_utc": utc_now(),
        "entrypoint": {
            "path": str(entrypoint.relative_to(REPO_ROOT)),
            "sha256": sha256_file(entrypoint),
        },
        "config": to_container(cfg),
        "command": [str(part) for part in command] if command else None,
        "command_line": format_command(command) if command else None,
        "inputs": resolved_inputs,
        "outputs": to_container(outputs or {}),
        "git": git_state(),
        "invocation": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    if extra:
        record.update(to_container(extra))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    payload = json.dumps(record, indent=2, sort_keys=True, default=str)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)
    print(f"[INFO] stage record: {path}", flush=True)
    return path


def read_stage_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_file():
        raise PipelineError(f"Stage record not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def latest_checkpoint(run_dir: str | Path, *, patterns: Iterable[str]) -> Path:
    """Find a produced checkpoint under ``run_dir`` by trying patterns in order.

    Training entrypoints disagree on layout (``checkpoints/latest.pt`` for the
    offline trainers, ``model_step_*.pt``/``model.pt`` for RLOpt), so the caller
    passes the patterns that apply to the stage it just ran.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise PipelineError(f"Run directory does not exist: {root}")
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            # Newest by mtime: RLOpt writes several model_step_*.pt per run.
            return max(matches, key=lambda item: item.stat().st_mtime)
    raise PipelineError(
        f"No checkpoint found under {root} matching any of {list(patterns)}. "
        "The training stage may have failed before writing one."
    )


def print_video_paths(search_root: str | Path) -> list[Path]:
    """Print absolute paths of retained videos so they are reachable over SSH.

    Required by the evaluation protocol: the Codex app does not pass video files
    through remote SSH targets, so the absolute path must appear on stdout for
    direct access on the target machine.
    """
    root = Path(search_root)
    if not root.exists():
        return []
    videos = sorted(
        path.resolve()
        for pattern in ("**/*.mp4", "**/*.webm")
        for path in root.glob(pattern)
    )
    for video in videos:
        print(f"[VIDEO] {video}", flush=True)
    if not videos:
        print(f"[VIDEO] none retained under {root.resolve()}", flush=True)
    return videos
