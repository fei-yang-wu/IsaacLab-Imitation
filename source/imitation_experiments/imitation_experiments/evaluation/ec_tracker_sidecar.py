"""Per-checkpoint CPU EC/MuJoCo evaluation sidecar.

The frequent evaluation signal of the three-level scheme: it watches a
training run's persistent checkpoint tree, and for every completed
``model_step_*.pt`` exports a fresh policy bundle, runs the fixed sidecar
board through the synchronous-lockstep Embodied-Control MuJoCo worker
(:mod:`imitation_experiments.evaluation.ec_sidecar_worker`), and writes one
canonical ``summary.json`` that :func:`imitation_experiments.reporting.records
.load_summary` reads unchanged.

Runs in the default Pixi environment on CPU only — never in the training
process, never on a GPU. Both heavy stages are subprocesses in their own
environments:

* bundle export: ``pixi run -e onnx-export python -m
  imitation_experiments.lowlevel.export_policy_bundle``
* rollout worker: ``pixi run -e lowlevel-sim python <ec_sidecar_worker.py>``
  with the Embodied-Control repository as the working directory

Usage from the repository root::

    pixi run python -m imitation_experiments.evaluation.ec_tracker_sidecar \
        run --checkpoint logs/<run>/model_step_500000000.pt \
        --preset fsq64_v2 --reference-root data/bones_seed_language10_v1/\
reference_arrays/root_qpos_v1 --output-root logs/eval/ec

    pixi run python -m imitation_experiments.evaluation.ec_tracker_sidecar \
        scan --checkpoint-tree logs/<run> --only-missing [--watch] ...

Work identity is ``checkpoint sha256 x profile hash``: an atomic
``O_CREAT|O_EXCL`` claim file makes concurrent sidecars (for example, the
sidecars of two chained training segments overlapping around a walltime
boundary) skip already-claimed work instead of duplicating it. A crashed
claim is reclaimed after ``--stale-claim-minutes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imitation_experiments.evaluation.protocol import (
    PROFILES,
    PROTOCOLS,
    BOARDS,
    EvalProfileV1,
    TrackerEvalContractV1,
    frozen_pairs,
)
from imitation_experiments.paths import REPO_ROOT

_WORKER_PATH = Path(__file__).with_name("ec_sidecar_worker.py")
_CHECKPOINT_PATTERN = "model_step_*.pt"
_STABILITY_SECONDS = 30.0


class SidecarError(RuntimeError):
    """A failure that invalidates one checkpoint's evaluation, not the sidecar."""


# ---------------------------------------------------------------------------
# Checkpoint discovery and eligibility
# ---------------------------------------------------------------------------


def sha256_of(path: Path, chunk_bytes: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(tree: Path) -> list[Path]:
    """Every ``model_step_<N>.pt`` under ``tree``, in cumulative-step order."""
    found = [
        path
        for path in tree.rglob(_CHECKPOINT_PATTERN)
        if path.is_file() and not path.name.endswith((".partial", ".tmp"))
    ]

    def _step(path: Path) -> int:
        stem = path.stem  # model_step_<N>
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return sorted(found, key=_step)


def wait_until_stable(
    path: Path, *, stability_seconds: float = _STABILITY_SECONDS
) -> bool:
    """True when size and mtime hold still across one stability interval."""
    try:
        before = path.stat()
    except FileNotFoundError:
        return False
    time.sleep(stability_seconds)
    try:
        after = path.stat()
    except FileNotFoundError:
        return False
    return before.st_size == after.st_size and before.st_mtime == after.st_mtime


def stage_checkpoint(checkpoint: Path, staging_dir: Path) -> tuple[Path, str]:
    """Copy the checkpoint into staging and verify the copy by sha256."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_of(checkpoint)
    staged = staging_dir / f"{source_sha}.pt"
    if not staged.is_file():
        tmp = staged.with_suffix(".copying")
        shutil.copy2(checkpoint, tmp)
        tmp.replace(staged)
    staged_sha = sha256_of(staged)
    if staged_sha != source_sha:
        staged.unlink(missing_ok=True)
        raise SidecarError(
            f"staged copy hash mismatch for {checkpoint}: {staged_sha} != {source_sha}"
        )
    return staged, source_sha


def resolve_skill_checkpoint(
    checkpoint: Path, explicit: Path | None = None
) -> Path | None:
    """Find the encoder checkpoint that belongs to this tracker checkpoint.

    An FSQ bundle export cannot be built from the tracker checkpoint alone: the
    lattice levels and bound constants live only in the skill checkpoint's
    config. The campaign tree pairs them by arm directory,

        <arm>/tracker/<run>/models/model_step_*.pt
        <arm>/encoder/checkpoints/latest.pt

    which is the exact path low-level training was given
    (``agent.ipmd.hl_skill_checkpoint_path``). Discovery is a convenience, not
    the guarantee: the exporter compares the discovered encoder tensor by
    tensor against the encoder embedded in the tracker checkpoint, so a wrong
    pairing fails loudly instead of scoring a mismatched pair.
    """
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_file():
            raise SidecarError(f"skill checkpoint not found: {resolved}")
        return resolved
    for parent in checkpoint.resolve().parents:
        candidate = parent / "encoder" / "checkpoints" / "latest.pt"
        if candidate.is_file():
            return candidate
    return None


def load_checkpoint_facts(staged: Path) -> dict[str, Any]:
    """CPU-load the checkpoint and pull the provenance the contract needs."""
    import torch

    try:
        payload = torch.load(staged, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - any load failure invalidates it
        raise SidecarError(f"checkpoint does not load on CPU: {exc}") from exc
    if not isinstance(payload, dict):
        raise SidecarError("checkpoint payload is not a dict")
    frames = payload.get("cumulative_env_frames")
    if frames is None:
        raise SidecarError("checkpoint carries no cumulative_env_frames")
    facts: dict[str, Any] = {"cumulative_env_frames": int(frames)}
    config = payload.get("config") or payload.get("agent_cfg") or {}
    if isinstance(config, dict):
        env_name = config.get("env", {})
        if isinstance(env_name, dict):
            env_name = env_name.get("env_name")
        if env_name:
            facts["task_id"] = str(env_name)
    return facts


# ---------------------------------------------------------------------------
# Atomic claims
# ---------------------------------------------------------------------------


def claim_path(output_root: Path, checkpoint_sha: str, profile_hash: str) -> Path:
    return output_root / checkpoint_sha / profile_hash / "claim.json"


def try_claim(path: Path, *, stale_minutes: float, payload: dict[str, Any]) -> bool:
    """Atomically claim one unit of work; reclaim only stale crashed claims."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = path.with_name("summary.json")
    if summary.is_file():
        return False
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age_minutes = (time.time() - path.stat().st_mtime) / 60.0
        if age_minutes < stale_minutes:
            return False
        # A crashed claim: replace it atomically so exactly one contender wins.
        stale = path.with_suffix(f".stale.{os.getpid()}")
        try:
            os.rename(path, stale)
        except FileNotFoundError:
            return False
        stale.unlink(missing_ok=True)
        return try_claim(path, stale_minutes=stale_minutes, payload=payload)
    with os.fdopen(descriptor, "w") as handle:
        json.dump({**payload, "pid": os.getpid(), "claimed_at": time.time()}, handle)
    return True


# ---------------------------------------------------------------------------
# Subprocess stages
# ---------------------------------------------------------------------------


def export_bundle(
    staged_checkpoint: Path,
    *,
    preset: str,
    bundle_dir: Path,
    pixi_bin: str,
    timeout_s: float,
    skill_checkpoint: Path | None = None,
    allow_finetuned_encoder: bool = False,
) -> dict[str, Any]:
    """Export and verify a fresh policy bundle; return its manifest."""
    if bundle_dir.exists():
        # A bundle is content-addressed by checkpoint sha; an existing one was
        # fully exported (the exporter refuses to write into an existing dir,
        # so a partial export cannot be mistaken for a finished one).
        manifest_path = bundle_dir / "manifest.json"
        if manifest_path.is_file():
            return json.loads(manifest_path.read_text())
        shutil.rmtree(bundle_dir)
    command = [
        pixi_bin,
        "run",
        "-e",
        "onnx-export",
        "python",
        "-m",
        "imitation_experiments.lowlevel.export_policy_bundle",
        "--checkpoint",
        str(staged_checkpoint),
        "--preset",
        preset,
        "--output",
        str(bundle_dir),
    ]
    if skill_checkpoint is not None:
        command.extend(["--skill-checkpoint", str(skill_checkpoint)])
    if allow_finetuned_encoder:
        command.append("--allow-finetuned-encoder")
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise SidecarError(
            f"bundle export failed ({completed.returncode}): "
            + completed.stderr[-2000:]
        )
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SidecarError("bundle export wrote no manifest.json")
    return json.loads(manifest_path.read_text())


def run_worker(
    *,
    bundle_dir: Path,
    model_xml: Path,
    reference_root: Path,
    cases: list[dict[str, Any]],
    max_steps: int,
    fall_height_m: float,
    noise: dict[str, float],
    work_dir: Path,
    ec_repo: Path,
    pixi_bin: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Run the sync-lockstep worker in the EC lowlevel-sim environment."""
    job = {
        "bundle": str(bundle_dir),
        "model": str(model_xml),
        "reference_root": str(reference_root),
        "cases": cases,
        "max_steps": int(max_steps),
        "fall_height_m": float(fall_height_m),
        "hold_steps": None,
        "noise": dict(noise),
        "output": str(work_dir / "worker_result.json"),
    }
    job_path = work_dir / "worker_job.json"
    job_path.write_text(json.dumps(job, indent=2))
    # Drop the caller's Pixi context: when the orchestrator itself runs under
    # `pixi run`, PIXI_PROJECT_MANIFEST points at THIS repo's manifest, which
    # has no `lowlevel-sim` environment. Pixi falls back to the local manifest
    # with a warning today, but relying on that is how a submission silently
    # resolves the wrong environment on a cluster node.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("PIXI_")
    }
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.setdefault("MUJOCO_GL", "egl")
    completed = subprocess.run(
        [
            pixi_bin,
            "run",
            "-e",
            "lowlevel-sim",
            "python",
            str(_WORKER_PATH),
            "--job",
            str(job_path),
        ],
        cwd=str(ec_repo),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    result_path = Path(job["output"])
    if completed.returncode != 0 or not result_path.is_file():
        raise SidecarError(
            f"EC worker failed ({completed.returncode}): " + completed.stderr[-2000:]
        )
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Canonical summary
# ---------------------------------------------------------------------------


def build_contract(
    *,
    checkpoint_sha: str,
    facts: dict[str, Any],
    preset: str,
    manifest: dict[str, Any],
    reference_manifest_sha: str,
    task_id: str,
    algorithm: str,
    protocol_tracked_bodies: tuple[str, ...],
    job_config_sha: str,
) -> TrackerEvalContractV1:
    return TrackerEvalContractV1(
        checkpoint_sha256=checkpoint_sha,
        cumulative_env_frames=facts.get("cumulative_env_frames"),
        task_id=task_id,
        algorithm=algorithm,
        agent_entry_point=None,
        actor_interface=frozen_pairs(
            {
                "preset": preset,
                "interface": manifest.get("interface", "latent"),
            }
        ),
        encoder_binding=frozen_pairs(
            {
                "source": "bundle_export_from_checkpoint",
                # The exporter records the encoder hash under `command`
                # (export_policy_bundle: manifest["command"]["encoder_sha256"]);
                # `models` holds only format/parity entries.
                "encoder_sha256": (manifest.get("command") or {}).get("encoder_sha256"),
                # The exporter proved this skill checkpoint tensor-identical to
                # the encoder embedded in the tracker checkpoint; recording it
                # is what lets a later gate re-check the pair.
                "skill_checkpoint_path": facts.get("skill_checkpoint_path"),
                "skill_checkpoint_sha256": facts.get("skill_checkpoint_sha256"),
                "binding": (manifest.get("source") or {}).get("encoder_binding"),
                "divergence_max_abs": (manifest.get("source") or {}).get(
                    "encoder_divergence_max_abs"
                ),
            }
        ),
        dataset=frozen_pairs({"reference_manifest_sha256": reference_manifest_sha}),
        tracked_body_names=protocol_tracked_bodies,
        policy=frozen_pairs({"bundle_preset": preset}),
        physics=frozen_pairs({"backend": "ec_mujoco_cpu"}),
        resolved_config_sha256=job_config_sha,
    )


def _episode_rows(worker_result: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = worker_result.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise SidecarError("worker result has no episodes")
    return episodes


def _assert_scored(episodes: list[dict[str, Any]]) -> None:
    failures = [row for row in episodes if row.get("artifact_failure")]
    if failures:
        detail = "; ".join(
            f"{row.get('motion_name')}: {row['artifact_failure']}" for row in failures
        )
        raise SidecarError(f"artifact failures invalidate this checkpoint: {detail}")


def to_summary(
    *,
    worker_result: dict[str, Any],
    profile: EvalProfileV1,
    contract: TrackerEvalContractV1,
    checkpoint_path: str,
    checkpoint_sha: str,
    bundle_dir: Path,
    bundle_manifest: dict[str, Any],
    label: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    """Reduce worker rows into the canonical ``records.load_summary`` schema."""
    protocol = PROTOCOLS[profile.protocol_id]
    board = BOARDS[profile.board_id]
    episodes = _episode_rows(worker_result)
    _assert_scored(episodes)

    # Requested versus realized: a rehearsal that silently ran without the
    # sensor noise its protocol declares would report an idealized number
    # under a protocol hash promising a noisy one.
    requested_noise = {key: float(value) for key, value in protocol.observation_noise}
    realized_noise = {
        key: float(value)
        for key, value in (worker_result.get("observation_noise") or {}).items()
    }
    if requested_noise != realized_noise:
        raise SidecarError(
            f"observation noise mismatch: protocol {requested_noise} "
            f"but worker ran {realized_noise}"
        )

    # A stratified board over-samples hard motions, so its raw mean is not a
    # population number. The case weights say what share of the population each
    # episode stands for; on a uniform board every weight is 1.0 and the
    # weighted and raw readings coincide.
    case_weight = {
        case.identity(): float(case.population_weight) for case in board.cases
    }

    per_environment = []
    frames_total = 0
    weighted_l = 0.0
    weighted_g = 0.0
    successful_l: list[float] = []
    fall_count = 0
    sonic_success_count = 0
    survival_steps: list[int] = []
    weight_total = 0.0
    weight_sonic_success = 0.0
    weight_fall_free = 0.0
    weight_successful_l = 0.0
    weighted_successful_l = 0.0
    for env_id, row in enumerate(episodes):
        frames = int(row["frames_scored"])
        expected = int(row["motion_length"]) - 1
        steps = int(row["steps"])
        # The most likely silent-wrong-number in this path: a frame-count
        # mismatch means the FK replay scored a different window than the
        # episode ran. One tick of slack covers the cursor's position when
        # exhaustion or a fall is detected; anything larger is a real skew.
        if abs(frames - steps) > 1:
            raise SidecarError(
                f"{row['motion_name']}: scored {frames} frames but ran {steps} steps"
            )
        if row.get("reference_finished") and abs(frames - expected) > 2:
            raise SidecarError(
                f"{row['motion_name']}: reference_finished with {frames} of "
                f"{expected} frames scored"
            )
        fell = bool(row["fell"])
        fall_count += int(fell)
        survival_steps.append(steps)
        frames_total += frames
        weighted_l += float(row["mpjpe_l_mm"]) * frames
        weighted_g += float(row["mpjpe_g_mm"]) * frames
        if not fell:
            successful_l.append(float(row["mpjpe_l_mm"]))

        sonic_success = bool((row.get("sonic") or {}).get("success", False))
        sonic_success_count += int(sonic_success)
        identity = (
            int(row["trajectory_rank"]),
            int(row["start_frame"]),
            int(row["env_seed"]),
            int(row["repeat_index"]),
        )
        weight = case_weight.get(identity)
        if weight is None:
            raise SidecarError(f"worker row is not on the board: {identity}")
        weight_total += weight
        weight_sonic_success += weight * float(sonic_success)
        weight_fall_free += weight * float(not fell)
        # Success-only quality, judged by the same criterion as the success
        # rate above, so both axes describe one population of episodes.
        if sonic_success:
            weight_successful_l += weight
            weighted_successful_l += weight * float(row["mpjpe_l_mm"])
        per_environment.append(
            {
                "env_id": env_id,
                "trajectory_rank": int(row["trajectory_rank"]),
                "motion_name": row["motion_name"],
                "start_frame": int(row["start_frame"]),
                "env_seed": int(row["env_seed"]),
                "repeat_index": int(row["repeat_index"]),
                "done": True,
                "fell": fell,
                "reference_finished": bool(row.get("reference_finished")),
                "survival_steps": steps,
                "termination_terms": [row["termination"]],
                "min_base_height_m": row["min_base_height_m"],
                "tracking_metrics": {
                    "tracking_mpjpe_mm": float(row["mpjpe_l_mm"]),
                    "tracking_mpjpe_g_mm": float(row["mpjpe_g_mm"]),
                },
                "tracking_metric_counts": {
                    "tracking_mpjpe_mm": frames,
                    "tracking_mpjpe_g_mm": frames,
                },
                "sonic": row.get("sonic", {}),
            }
        )

    count = len(per_environment)
    return {
        "schema_version": "ec_sidecar_summary_v1",
        "authority_status": "uncertified",
        "task": contract.task_id,
        "algorithm": contract.algorithm,
        "max_steps": protocol.outer_safety_cap_steps,
        "disable_push_event": True,
        "metadata": {
            "label": label,
            "seed": board.seeds[0] if len(board.seeds) == 1 else None,
            "fall_height_m": protocol.fall_height_m,
            "interface": dict(contract.actor_interface).get("interface"),
            "execution_mode": worker_result.get("execution_mode"),
            "low_level_tracker": {
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "policy_frozen": True,
            },
        },
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
            "cumulative_env_frames": contract.cumulative_env_frames,
        },
        "bundle": {
            "root": str(bundle_dir),
            "manifest_sha256": hashlib.sha256(
                json.dumps(bundle_manifest, sort_keys=True).encode()
            ).hexdigest(),
            "source_checkpoint_sha256": checkpoint_sha,
        },
        "contract": contract.stamp(),
        "protocol": protocol.stamp(),
        "board": board.stamp(),
        "profile": profile.stamp(),
        "realized_protocol": {
            "execution_mode": worker_result.get("execution_mode"),
            "backend": protocol.backend,
            "observation_corruption": bool(worker_result.get("observation_noise")),
            "observation_noise": worker_result.get("observation_noise") or {},
        },
        "runtime": runtime,
        "aggregate": {
            "episode_count": count,
            "completed_episode_count": count,
            "done_rate": 1.0,
            "fall_count": fall_count,
            "fall_free_rate": 1.0 - fall_count / count,
            "sonic_success_count": sonic_success_count,
            "sonic_success_rate": sonic_success_count / count,
            # Population estimates. Identical to the raw rates on a uniform
            # board; on a difficulty-stratified board these are the numbers
            # comparable with a full-population scoreboard.
            "population_weighted_sonic_success_rate": (
                weight_sonic_success / weight_total if weight_total else 0.0
            ),
            "population_weighted_fall_free_rate": (
                weight_fall_free / weight_total if weight_total else 0.0
            ),
            "survival_steps_mean": sum(survival_steps) / count,
            "valid_transition_count": frames_total,
        },
        "metric_means": {
            "tracking_mpjpe_mm": weighted_l / max(frames_total, 1),
            "tracking_mpjpe_g_mm": weighted_g / max(frames_total, 1),
        },
        "successful_trajectory_metrics": {
            "tracking_mpjpe_mm": (
                {"mean": sum(successful_l) / len(successful_l)} if successful_l else {}
            ),
            "population_weighted_sonic_success_mpjpe_mm": (
                {"mean": weighted_successful_l / weight_successful_l}
                if weight_successful_l
                else {}
            ),
        },
        "per_environment": per_environment,
    }


# ---------------------------------------------------------------------------
# W&B publication
# ---------------------------------------------------------------------------


_WANDB_RUN_ID_RE = re.compile(r"_wandb-([A-Za-z0-9]+)")

METRIC_PREFIX = "Eval/"


def wandb_payload(
    summary: dict[str, Any], prefix: str = METRIC_PREFIX
) -> tuple[dict[str, Any], int | None]:
    """Scalars plus the frame count they belong at.

    Frames, not wall clock or checkpoint index: RLOpt logs its own metrics with
    ``step=frames_processed``, so publishing at ``cumulative_env_frames`` puts
    a sidecar point on exactly the training curve's x-axis. A chained run's
    segment-local step counter restarts; the cumulative count does not.
    """
    aggregate = summary.get("aggregate") or {}
    means = summary.get("metric_means") or {}
    successful = (summary.get("successful_trajectory_metrics") or {}).get(
        "tracking_mpjpe_mm"
    ) or {}
    payload = {
        f"{prefix}mpjpe_l_mm": means.get("tracking_mpjpe_mm"),
        f"{prefix}mpjpe_g_mm": means.get("tracking_mpjpe_g_mm"),
        f"{prefix}fall_free_rate": aggregate.get("fall_free_rate"),
        f"{prefix}sonic_success_rate": aggregate.get("sonic_success_rate"),
        f"{prefix}sonic_success_rate_weighted": aggregate.get(
            "population_weighted_sonic_success_rate"
        ),
        f"{prefix}survival_steps_mean": aggregate.get("survival_steps_mean"),
        f"{prefix}episode_count": aggregate.get("episode_count"),
        f"{prefix}eval_seconds": (summary.get("runtime") or {}).get("eval_seconds"),
    }
    if "mean" in successful:
        payload[f"{prefix}mpjpe_l_mm_successful"] = successful["mean"]
    weighted_successful = (summary.get("successful_trajectory_metrics") or {}).get(
        "population_weighted_sonic_success_mpjpe_mm"
    ) or {}
    if "mean" in weighted_successful:
        payload[f"{prefix}mpjpe_l_mm_sonic_success_weighted"] = weighted_successful[
            "mean"
        ]
    frames = (summary.get("contract") or {}).get("cumulative_env_frames")
    if frames is not None:
        payload[f"{prefix}cumulative_env_frames"] = int(frames)
    clean = {key: value for key, value in payload.items() if value is not None}
    return clean, (int(frames) if frames is not None else None)


def infer_training_run_id(path: Path) -> str | None:
    """Read the trainer's W&B run id out of its checkpoint path.

    RLOpt nests checkpoints under ``<timestamp>_wandb-<run_id>``, so the run
    that produced a checkpoint is knowable from the file alone — no run
    registry, and no guessing from a name that several runs may share.
    """
    for part in path.resolve().parts:
        match = _WANDB_RUN_ID_RE.search(part)
        if match:
            return match.group(1)
    return None


class SidecarWandb:
    """Publishes sidecar points into W&B, lazily so the run id can be inferred.

    Two modes. ``attach`` writes ``Eval/*`` into the TRAINING run itself, using
    W&B shared mode (``x_primary=False``) so a second process may write a run
    the trainer owns; points land at the trainer's own frame step. Otherwise a
    companion run carries them, which is what a re-scored bundle or an
    already-finished run gets.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        group: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        attach: bool = False,
        run_id: str | None = None,
        prefix: str = METRIC_PREFIX,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._project = project
        self._run_name = run_name
        self._group = group
        self._entity = entity
        self._tags = tags or []
        self._config = config or {}
        self._attach = attach
        self._run_id = run_id
        self._prefix = prefix
        self.run: Any | None = None

    def _start(self, run_id: str | None) -> None:
        wandb = self._wandb
        settings = None
        resume = None
        if self._attach and run_id:
            # Shared mode is W&B's supported multi-writer path: the trainer is
            # the primary, this process is a labelled secondary. Without it,
            # two processes on one run id race over the step counter.
            #
            # The TRAINER must have created the run in shared mode
            # (WANDB_MODE=shared, WANDB__PRIMARY=true). W&B refuses to convert
            # an existing run -- "cannot enable shared mode for run <id> with
            # existing history" -- and the refusal arrives asynchronously on
            # the filestream, so the points are dropped rather than raised.
            #
            # x_update_finish_state=False is what keeps this process's
            # finish() from marking the TRAINER's run finished.
            settings = wandb.Settings(
                mode="shared",
                x_primary=False,
                x_label="ec-sidecar",
                x_update_finish_state=False,
            )
            resume = "allow"
        self.run = wandb.init(
            project=self._project,
            id=run_id if self._attach else None,
            resume=resume,
            name=None if (self._attach and run_id) else self._run_name,
            group=self._group,
            entity=self._entity,
            tags=self._tags,
            config=self._config,
            job_type=None if (self._attach and run_id) else "ec_sidecar",
            settings=settings,
            reinit=True,
        )
        wandb.define_metric(f"{self._prefix}cumulative_env_frames")
        wandb.define_metric(
            f"{self._prefix}*", step_metric=f"{self._prefix}cumulative_env_frames"
        )

    def publish(self, summary: dict[str, Any], source: Path | None = None) -> None:
        if self.run is None:
            run_id = self._run_id
            if self._attach and run_id is None and source is not None:
                run_id = infer_training_run_id(source)
            if self._attach and run_id is None:
                print(
                    "[WANDB] no training run id in the checkpoint path; "
                    "falling back to a companion run.",
                    file=sys.stderr,
                    flush=True,
                )
                self._attach = False
            self._start(run_id)
        payload, frames = wandb_payload(summary, self._prefix)
        if not payload:
            return
        assert self.run is not None
        # Explicit step = frames, so the point lands on the trainer's x-axis
        # rather than at whatever the writer's internal counter happens to be.
        self.run.log(payload, step=frames) if frames is not None else self.run.log(
            payload
        )

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------------------
# One checkpoint end to end
# ---------------------------------------------------------------------------


def evaluate_bundle_once(bundle_dir: Path, cfg: "SidecarConfig") -> Path | None:
    """Score an existing verified bundle, skipping staging and export.

    Checkpoint identity comes from the bundle manifest's ``source`` block, so
    the artifact is still keyed by the true checkpoint sha256. Used to
    re-score a bundle and to run the sidecar where the checkpoint itself is
    no longer on disk.
    """
    profile = PROFILES[cfg.profile_id]
    protocol = PROTOCOLS[profile.protocol_id]
    board = BOARDS[profile.board_id]

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SidecarError(f"bundle has no manifest.json: {bundle_dir}")
    manifest = json.loads(manifest_path.read_text())
    source = manifest.get("source") or {}
    checkpoint_sha = source.get("checkpoint_sha256")
    checkpoint_path = source.get("checkpoint_path", str(bundle_dir))
    if not checkpoint_sha:
        raise SidecarError("bundle manifest has no source.checkpoint_sha256")
    step_token = Path(checkpoint_path).stem.rsplit("_", 1)[-1]
    cumulative = int(step_token) if step_token.isdigit() else None

    work_dir = cfg.output_root / checkpoint_sha / profile.content_hash()
    claim = claim_path(cfg.output_root, checkpoint_sha, profile.content_hash())
    if not try_claim(
        claim,
        stale_minutes=cfg.stale_claim_minutes,
        payload={"bundle": str(bundle_dir), "profile": profile.profile_id},
    ):
        return None

    started = time.monotonic()
    try:
        reference_manifest = cfg.reference_root / "reference_arrays_manifest.json"
        reference_sha = sha256_of(reference_manifest)
        cases = [
            {
                "trajectory_rank": case.trajectory_rank,
                "motion_name": case.motion_name,
                "start_frame": case.start_frame,
                "env_seed": case.env_seed,
                "repeat_index": case.repeat_index,
            }
            for case in board.cases
        ]
        worker_result = run_worker(
            bundle_dir=bundle_dir,
            model_xml=cfg.model_xml,
            reference_root=cfg.reference_root,
            cases=cases,
            max_steps=protocol.outer_safety_cap_steps,
            fall_height_m=protocol.fall_height_m or 0.4,
            noise=dict(protocol.observation_noise),
            work_dir=work_dir,
            ec_repo=cfg.ec_repo,
            pixi_bin=cfg.pixi_bin,
            timeout_s=cfg.worker_timeout_s,
        )
        job_config_sha = hashlib.sha256(
            json.dumps(
                {"preset": cfg.preset, "profile": profile.stamp()}, sort_keys=True
            ).encode()
        ).hexdigest()
        contract = build_contract(
            checkpoint_sha=checkpoint_sha,
            facts={"cumulative_env_frames": cumulative},
            preset=str(source.get("preset", cfg.preset)),
            manifest=manifest,
            reference_manifest_sha=reference_sha,
            task_id=cfg.task_id,
            algorithm=cfg.algorithm,
            protocol_tracked_bodies=protocol.tracked_body_names,
            job_config_sha=job_config_sha,
        )
        summary = to_summary(
            worker_result=worker_result,
            profile=profile,
            contract=contract,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            bundle_dir=bundle_dir,
            bundle_manifest=manifest,
            label=f"ec-sidecar {profile.profile_id} {bundle_dir.name}",
            runtime={
                "export_seconds": 0.0,
                "eval_seconds": round(time.monotonic() - started, 3),
                "cpu_count": os.cpu_count(),
                **worker_result.get("runtime", {}),
            },
        )
        summary_path = work_dir / "summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(summary_path)
        if cfg.wandb is not None:
            cfg.wandb.publish(summary, source=bundle_dir)
        print(
            f"[PASS] {bundle_dir.name}: "
            f"mpjpe_l={summary['metric_means']['tracking_mpjpe_mm']:.2f}mm "
            f"fall_free={summary['aggregate']['fall_free_rate']:.3f} "
            f"-> {summary_path}",
            flush=True,
        )
        return summary_path
    except (SidecarError, subprocess.TimeoutExpired) as exc:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "failure.json").write_text(
            json.dumps(
                {
                    "bundle": str(bundle_dir),
                    "checkpoint_sha256": checkpoint_sha,
                    "profile_id": profile.profile_id,
                    "error": str(exc),
                    "failed_at": time.time(),
                },
                indent=2,
            )
        )
        print(f"[FAIL] {bundle_dir.name}: {exc}", file=sys.stderr, flush=True)
        return None
    finally:
        claim.unlink(missing_ok=True)


@dataclass
class SidecarConfig:
    output_root: Path
    reference_root: Path
    model_xml: Path
    ec_repo: Path
    preset: str
    profile_id: str
    pixi_bin: str
    task_id: str
    algorithm: str
    staging_dir: Path
    stale_claim_minutes: float
    export_timeout_s: float
    worker_timeout_s: float
    skill_checkpoint: Path | None = None
    allow_finetuned_encoder: bool = False
    # Optional companion W&B run; None keeps the sidecar file-only.
    wandb: SidecarWandb | None = None


def evaluate_checkpoint_once(checkpoint: Path, cfg: SidecarConfig) -> Path | None:
    """Evaluate one checkpoint; returns the summary path, or None if skipped."""
    profile = PROFILES[cfg.profile_id]
    protocol = PROTOCOLS[profile.protocol_id]
    board = BOARDS[profile.board_id]

    staged, checkpoint_sha = stage_checkpoint(checkpoint, cfg.staging_dir)
    work_dir = cfg.output_root / checkpoint_sha / profile.content_hash()
    claim = claim_path(cfg.output_root, checkpoint_sha, profile.content_hash())
    if not try_claim(
        claim,
        stale_minutes=cfg.stale_claim_minutes,
        payload={"checkpoint": str(checkpoint), "profile": profile.profile_id},
    ):
        return None

    started = time.monotonic()
    try:
        facts = load_checkpoint_facts(staged)
        skill_checkpoint = resolve_skill_checkpoint(checkpoint, cfg.skill_checkpoint)
        if skill_checkpoint is not None:
            facts["skill_checkpoint_path"] = str(skill_checkpoint)
            facts["skill_checkpoint_sha256"] = sha256_of(skill_checkpoint)
        bundle_dir = cfg.staging_dir / "bundles" / checkpoint_sha / cfg.preset
        export_started = time.monotonic()
        manifest = export_bundle(
            staged,
            preset=cfg.preset,
            bundle_dir=bundle_dir,
            pixi_bin=cfg.pixi_bin,
            timeout_s=cfg.export_timeout_s,
            skill_checkpoint=skill_checkpoint,
            allow_finetuned_encoder=cfg.allow_finetuned_encoder,
        )
        export_seconds = time.monotonic() - export_started

        reference_manifest = cfg.reference_root / "reference_arrays_manifest.json"
        reference_sha = sha256_of(reference_manifest)

        cases = [
            {
                "trajectory_rank": case.trajectory_rank,
                "motion_name": case.motion_name,
                "start_frame": case.start_frame,
                "env_seed": case.env_seed,
                "repeat_index": case.repeat_index,
            }
            for case in board.cases
        ]
        worker_result = run_worker(
            bundle_dir=bundle_dir,
            model_xml=cfg.model_xml,
            reference_root=cfg.reference_root,
            cases=cases,
            max_steps=protocol.outer_safety_cap_steps,
            fall_height_m=protocol.fall_height_m or 0.4,
            noise=dict(protocol.observation_noise),
            work_dir=work_dir,
            ec_repo=cfg.ec_repo,
            pixi_bin=cfg.pixi_bin,
            timeout_s=cfg.worker_timeout_s,
        )

        job_config_sha = hashlib.sha256(
            json.dumps(
                {"preset": cfg.preset, "profile": profile.stamp()}, sort_keys=True
            ).encode()
        ).hexdigest()
        contract = build_contract(
            checkpoint_sha=checkpoint_sha,
            facts=facts,
            preset=cfg.preset,
            manifest=manifest,
            reference_manifest_sha=reference_sha,
            task_id=facts.get("task_id", cfg.task_id),
            algorithm=cfg.algorithm,
            protocol_tracked_bodies=protocol.tracked_body_names,
            job_config_sha=job_config_sha,
        )
        summary = to_summary(
            worker_result=worker_result,
            profile=profile,
            contract=contract,
            checkpoint_path=str(checkpoint),
            checkpoint_sha=checkpoint_sha,
            bundle_dir=bundle_dir,
            bundle_manifest=manifest,
            label=f"ec-sidecar {profile.profile_id} {checkpoint.stem}",
            runtime={
                "export_seconds": round(export_seconds, 3),
                "eval_seconds": round(time.monotonic() - started, 3),
                "cpu_count": os.cpu_count(),
                **worker_result.get("runtime", {}),
            },
        )
        summary_path = work_dir / "summary.json"
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2))
        tmp.replace(summary_path)
        if cfg.wandb is not None:
            cfg.wandb.publish(summary, source=checkpoint)
        print(
            f"[PASS] {checkpoint.name}: "
            f"mpjpe_l={summary['metric_means']['tracking_mpjpe_mm']:.2f}mm "
            f"fall_free={summary['aggregate']['fall_free_rate']:.3f} "
            f"-> {summary_path}",
            flush=True,
        )
        return summary_path
    except (SidecarError, subprocess.TimeoutExpired) as exc:
        failure_path = work_dir / "failure.json"
        work_dir.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                    "profile_id": profile.profile_id,
                    "error": str(exc),
                    "failed_at": time.time(),
                },
                indent=2,
            )
        )
        print(f"[FAIL] {checkpoint.name}: {exc}", file=sys.stderr, flush=True)
        return None
    finally:
        claim.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default="sidecar_ec_v1", choices=sorted(PROFILES))
    parser.add_argument("--preset", default="fsq64_v2")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT
        / "source/isaaclab_imitation/isaaclab_imitation/assets/unitree/"
        "g1_description/g1_29dof_rev_1_0.xml",
    )
    parser.add_argument(
        "--ec-repo", type=Path, default=REPO_ROOT / "external/Embodied-Control"
    )
    parser.add_argument(
        "--skill-checkpoint",
        type=Path,
        default=None,
        help=(
            "encoder checkpoint for the bundle export; FSQ presets need it for "
            "the lattice levels. Default: <arm>/encoder/checkpoints/latest.pt "
            "discovered above the tracker checkpoint."
        ),
    )
    parser.add_argument(
        "--allow-finetuned-encoder",
        action="store_true",
        help=(
            "the arm fine-tunes its encoder during low-level training "
            "(hl_skill_finetune_enabled=true), so the bundle ships the "
            "tracker's own encoder and the divergence is recorded"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, default=None)
    parser.add_argument("--pixi-bin", default="pixi")
    parser.add_argument("--task-id", default="unknown")
    parser.add_argument("--algorithm", default="IPMD")
    parser.add_argument("--stale-claim-minutes", type=float, default=45.0)
    parser.add_argument("--export-timeout-s", type=float, default=900.0)
    parser.add_argument("--worker-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="publish each summary to a companion W&B run in this project",
    )
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-tags", default="", help="comma-separated")
    parser.add_argument(
        "--wandb-attach",
        action="store_true",
        help=(
            "write Eval/* into the TRAINING run instead of a companion run; "
            "the run id comes from the checkpoint path's _wandb-<id> segment "
            "unless --wandb-run-id says otherwise"
        ),
    )
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-metric-prefix", default=METRIC_PREFIX)


def _make_wandb(args: argparse.Namespace) -> SidecarWandb | None:
    if not args.wandb_project:
        return None
    profile = PROFILES[args.profile]
    return SidecarWandb(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"{args.task_id}-ec-sidecar",
        group=args.wandb_group,
        entity=args.wandb_entity,
        tags=[tag for tag in args.wandb_tags.split(",") if tag],
        attach=bool(args.wandb_attach),
        run_id=args.wandb_run_id,
        prefix=args.wandb_metric_prefix,
        # The identity of what these numbers mean travels with them.
        config={
            "profile_id": profile.profile_id,
            "profile_hash": profile.content_hash(),
            "protocol_id": profile.protocol_id,
            "protocol_hash": profile.protocol_hash,
            "board_id": profile.board_id,
            "board_hash": profile.board_hash,
            "preset": args.preset,
            "reference_root": str(args.reference_root),
        },
    )


def _config(args: argparse.Namespace) -> SidecarConfig:
    output_root = args.output_root.resolve()
    return SidecarConfig(
        output_root=output_root,
        reference_root=args.reference_root.resolve(),
        model_xml=args.model.resolve(),
        ec_repo=args.ec_repo.resolve(),
        preset=args.preset,
        profile_id=args.profile,
        pixi_bin=args.pixi_bin,
        task_id=args.task_id,
        algorithm=args.algorithm,
        staging_dir=(args.staging_dir or output_root / "staging").resolve(),
        skill_checkpoint=args.skill_checkpoint,
        allow_finetuned_encoder=bool(args.allow_finetuned_encoder),
        stale_claim_minutes=args.stale_claim_minutes,
        export_timeout_s=args.export_timeout_s,
        worker_timeout_s=args.worker_timeout_s,
        wandb=_make_wandb(args),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="evaluate one checkpoint")
    run_parser.add_argument("--checkpoint", type=Path, default=None)
    run_parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="score an existing verified bundle instead of exporting one",
    )
    _add_common(run_parser)

    scan_parser = commands.add_parser(
        "scan", help="evaluate every unseen checkpoint under a tree"
    )
    scan_parser.add_argument("--checkpoint-tree", type=Path, required=True)
    scan_parser.add_argument("--only-missing", action="store_true")
    scan_parser.add_argument(
        "--watch",
        action="store_true",
        help="keep scanning every --scan-interval-s until interrupted",
    )
    scan_parser.add_argument("--scan-interval-s", type=float, default=30.0)
    scan_parser.add_argument(
        "--skip-stability-wait",
        action="store_true",
        help="testing only: trust that checkpoint files are complete",
    )
    _add_common(scan_parser)

    publish_parser = commands.add_parser(
        "publish",
        help="re-publish summaries already on disk to W&B (backfill), no eval",
    )
    publish_parser.add_argument("--summary-root", type=Path, required=True)
    _add_common(publish_parser)

    args = parser.parse_args(argv)
    cfg = _config(args)
    try:
        return _dispatch(args, cfg, parser)
    finally:
        if cfg.wandb is not None:
            cfg.wandb.finish()


def _dispatch(
    args: argparse.Namespace, cfg: SidecarConfig, parser: argparse.ArgumentParser
) -> int:
    if args.command == "run":
        if (args.checkpoint is None) == (args.bundle is None):
            parser.error("run needs exactly one of --checkpoint or --bundle")
        if args.bundle is not None:
            result = evaluate_bundle_once(args.bundle.resolve(), cfg)
            return 0 if result is not None else 1
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f"checkpoint missing: {checkpoint}")
        result = evaluate_checkpoint_once(checkpoint, cfg)
        return 0 if result is not None else 1

    if args.command == "publish":
        if cfg.wandb is None:
            parser.error("publish needs --wandb-project")
        summaries = sorted(args.summary_root.resolve().rglob("summary.json"))
        if not summaries:
            parser.error(f"no summary.json under {args.summary_root}")
        # Frame order, so the backfilled series is monotonic like a live one.
        loaded = [json.loads(path.read_text()) for path in summaries]
        loaded.sort(
            key=lambda item: (
                (item.get("contract") or {}).get("cumulative_env_frames") or 0
            )
        )
        for summary in loaded:
            cfg.wandb.publish(summary, source=Path(summary["checkpoint"]["path"]))
            print(
                f"[PUBLISH] {(summary.get('contract') or {}).get('cumulative_env_frames')}"
                f" frames: mpjpe_l="
                f"{summary['metric_means']['tracking_mpjpe_mm']:.2f}mm",
                flush=True,
            )
        return 0

    tree = args.checkpoint_tree.resolve()
    if not tree.is_dir():
        parser.error(f"checkpoint tree missing: {tree}")
    while True:
        backlog = discover_checkpoints(tree)
        pending = 0
        for checkpoint in backlog:
            if not args.skip_stability_wait and not wait_until_stable(checkpoint):
                continue
            result = evaluate_checkpoint_once(checkpoint, cfg)
            if result is not None:
                pending += 1
        if not args.watch:
            return 0
        if pending == 0:
            time.sleep(args.scan_interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
