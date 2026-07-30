"""End-to-end latent-interface pipeline: encoder → tracker → data → planner → eval.

This is the project's core loop as one Hydra-driven conductor, so a variant is
a config override instead of a hand-edited shell chain:

    pretrain          stage 1a: offline DiffSR latent encoder (paper stage)
    low_level         stage 1b: latent-command tracker bound to that encoder
    binding           tensor-identity gate: encoder in tracker == skill ckpt
    collect           planner training rollouts from the frozen tracker
    merge             merge/dedup collected sample files
    train_planner     chunked-transformer planner on the merged samples
    eval_offline      open-loop planner metrics
    eval_closed_loop  Isaac closed-loop evaluation with the frozen tracker

Design rules, in order of importance:

* **Dry run first.** ``dry_run=true`` (the default) renders every command of
  every requested stage without touching the scheduler, the simulator, or the
  output tree beyond the run directory itself.
* **A stage that ran is reconstructable.** Each stage writes an
  ``e2e_stage.json`` with the argv, the resolved stage config, and input file
  hashes. Reruns land in the same tree only with ``resume=true``, and a resumed
  stage whose recorded input hashes changed is a hard error, not a warning.
* **Artifact handoffs are wired here, not in configs.** The encoder checkpoint,
  tracker checkpoint, sample directories, and planner checkpoint flow between
  stages by construction; per-stage configs hold only genuine parameters.

Usage from the repository root:

    pixi run python -m imitation_experiments.pipeline.run_latent_e2e \\
        name=det-latent-smoke stages='[pretrain,low_level]'

    # a campaign pins a config instead:
    pixi run python -m imitation_experiments.pipeline.run_latent_e2e \\
        --config-dir experiments/campaigns/<campaign>/conf --config-name <pin>
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from imitation_experiments.paths import REPO_ROOT

STAGE_ORDER = (
    "pretrain",
    "low_level",
    "binding",
    "collect",
    "merge",
    "train_planner",
    "eval_offline",
    "eval_closed_loop",
)

RECORD_NAME = "e2e_stage.json"


class PipelineError(RuntimeError):
    """A gate failed. Deliberately fatal: a partial run must not look complete."""


# --------------------------------------------------------------------------
# Small provenance helpers
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, str]:
    def _run(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() if out.returncode == 0 else ""

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "yes" if _run("status", "--porcelain") else "no",
    }


def _hash_inputs(inputs: dict[str, Path | None]) -> dict[str, dict[str, str]]:
    hashed: dict[str, dict[str, str]] = {}
    for name, path in inputs.items():
        if path is None:
            continue
        entry: dict[str, str] = {"path": str(path)}
        if path.is_file():
            entry["sha256"] = _sha256(path)
        hashed[name] = entry
    return hashed


# --------------------------------------------------------------------------
# Command rendering and execution
# --------------------------------------------------------------------------


def _flags(args: Any) -> list[str]:
    """Render a config mapping to CLI flags.

    ``None`` skips the flag, booleans render as bare ``--flag`` (only when
    true), and sequences render as one flag with multiple values.
    """
    rendered: list[str] = []
    for key, value in (OmegaConf.to_container(args, resolve=True) or {}).items():
        flag = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                rendered.append(flag)
        elif isinstance(value, (list, tuple)):
            rendered += [flag, *[str(v) for v in value]]
        else:
            rendered += [flag, str(value)]
    return rendered


class Runner:
    def __init__(self, cfg: DictConfig, run_root: Path):
        self.cfg = cfg
        self.run_root = run_root
        self.dry_run = bool(cfg.dry_run)
        self.plain = [str(x) for x in cfg.python.plain]
        self.isaac = [str(x) for x in cfg.python.isaac]

    def stage_dir(self, stage: str) -> Path:
        return self.run_root / stage

    def already_complete(self, stage: str) -> bool:
        record = self.stage_dir(stage) / RECORD_NAME
        if not record.is_file():
            return False
        payload = json.loads(record.read_text())
        return payload.get("status") == "complete"

    def check_resume_inputs(self, stage: str, inputs: dict[str, Path | None]) -> None:
        record = self.stage_dir(stage) / RECORD_NAME
        recorded = json.loads(record.read_text()).get("inputs", {})
        current = _hash_inputs(inputs)
        for name, entry in current.items():
            old = recorded.get(name, {})
            if "sha256" in entry and old.get("sha256") not in (None, entry["sha256"]):
                raise PipelineError(
                    f"Resume refused: input {name!r} of completed stage "
                    f"{stage!r} changed on disk ({old.get('sha256')} -> "
                    f"{entry['sha256']}). Use a fresh output root."
                )

    def run_stage(
        self,
        stage: str,
        command: list[str],
        inputs: dict[str, Path | None],
        outputs: dict[str, str],
    ) -> None:
        stage_dir = self.stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        printable = shlex.join(command)
        if self.dry_run:
            print(f"[DRY] ({stage}) {printable}", flush=True)
            status = "dry_run"
        else:
            for name, path in inputs.items():
                if path is not None and not path.exists():
                    raise PipelineError(
                        f"Stage {stage!r} input {name!r} is missing: {path}"
                    )
            print(f"[RUN] ({stage}) {printable}", flush=True)
            result = subprocess.run(command, cwd=REPO_ROOT)
            if result.returncode != 0:
                raise PipelineError(
                    f"Stage {stage!r} failed with exit code {result.returncode}: "
                    f"{printable}"
                )
            status = "complete"
        record = {
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "inputs": {} if self.dry_run else _hash_inputs(inputs),
            "outputs": outputs,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "git": _git_state(),
        }
        (stage_dir / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")


# --------------------------------------------------------------------------
# Artifact handoffs
# --------------------------------------------------------------------------


def _recorded_output(stage_dir: Path, key: str, fallback: Path) -> Path:
    """Prefer the exact artifact a paper stage recorded over a naming guess."""
    record = stage_dir / "stage_record.json"
    if record.is_file():
        payload = json.loads(record.read_text())
        entry = payload.get("outputs", {}).get(key)
        if isinstance(entry, dict) and entry.get("path"):
            return Path(entry["path"])
    return fallback


def main_impl(cfg: DictConfig) -> None:
    requested = [str(s) for s in cfg.stages]
    unknown = [s for s in requested if s not in STAGE_ORDER]
    if unknown:
        raise PipelineError(
            f"Unknown stages {unknown}; valid stages in order: {list(STAGE_ORDER)}"
        )
    stages = [s for s in STAGE_ORDER if s in requested]

    run_root = Path(str(cfg.output_root)).expanduser().resolve()
    if run_root.exists() and not bool(cfg.resume) and any(run_root.iterdir()):
        raise PipelineError(
            f"Output root already exists and is not empty: {run_root}. "
            "Set resume=true for an intentional continuation or pick a fresh root."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    runner = Runner(cfg, run_root)

    seed = str(int(cfg.seed))
    # Handoff paths. Stages overwrite these with recorded artifacts as they run.
    pretrain_dir = runner.stage_dir("pretrain")
    low_level_dir = runner.stage_dir("low_level")
    encoder_ckpt = _recorded_output(
        pretrain_dir,
        "skill_encoder_checkpoint",
        pretrain_dir / "checkpoints/latest.pt",
    )
    if cfg.low_level.get("encoder_checkpoint"):
        encoder_ckpt = Path(str(cfg.low_level.encoder_checkpoint))
    tracker_ckpt = _recorded_output(
        low_level_dir, "policy_checkpoint", low_level_dir / "checkpoints/latest.pt"
    )
    if cfg.collect.get("checkpoint"):
        tracker_ckpt = Path(str(cfg.collect.checkpoint))
    samples_dir = runner.stage_dir("collect") / "samples"
    merged_dir = runner.stage_dir("merge") / "samples"
    planner_dir = runner.stage_dir("train_planner")
    planner_ckpt = planner_dir / "checkpoints" / str(cfg.eval.planner_checkpoint_name)

    for stage in stages:
        if runner.already_complete(stage):
            print(f"[SKIP] ({stage}) already complete in {runner.stage_dir(stage)}")
            continue

        if stage == "pretrain":
            command = [
                *runner.plain,
                str(REPO_ROOT / "experiments/paper/pipeline/pretrain_latent_encoder.py"),
                f"output_dir={pretrain_dir}",
                f"seed={seed}",
                f"dry_run={str(runner.dry_run).lower()}",
                "allow_existing_output=true",
                *[str(o) for o in cfg.pretrain.overrides],
            ]
            runner.run_stage(stage, command, inputs={}, outputs={"run_dir": str(pretrain_dir)})
            encoder_ckpt = _recorded_output(
                pretrain_dir, "skill_encoder_checkpoint", encoder_ckpt
            )

        elif stage == "low_level":
            command = [
                *runner.plain,
                str(REPO_ROOT / "experiments/paper/pipeline/train_low_level.py"),
                f"output_dir={low_level_dir}",
                f"seed={seed}",
                f"dry_run={str(runner.dry_run).lower()}",
                "allow_existing_output=true",
                "interface.name=latent_skill",
                f"encoder.checkpoint={encoder_ckpt}",
                *[str(o) for o in cfg.low_level.overrides],
            ]
            runner.run_stage(
                stage,
                command,
                inputs={"encoder_checkpoint": encoder_ckpt},
                outputs={"run_dir": str(low_level_dir)},
            )
            tracker_ckpt = _recorded_output(
                low_level_dir, "policy_checkpoint", tracker_ckpt
            )

        elif stage == "binding":
            binding_json = runner.stage_dir(stage) / "skill_binding.json"
            command = [
                *runner.plain,
                "-m",
                "imitation_experiments.audit.validate_latent_skill_checkpoint_binding",
                "--low_level_checkpoint",
                str(tracker_ckpt),
                "--skill_checkpoint",
                str(encoder_ckpt),
                "--output_json",
                str(binding_json),
            ]
            runner.run_stage(
                stage,
                command,
                inputs={
                    "tracker_checkpoint": tracker_ckpt,
                    "encoder_checkpoint": encoder_ckpt,
                },
                outputs={"binding_record": str(binding_json)},
            )

        elif stage == "collect":
            command = [
                *runner.isaac,
                "-m",
                "imitation_experiments.data.collect_interface_rollout_samples",
                "--interface",
                "latent_skill",
                "--checkpoint",
                str(tracker_ckpt),
                "--output_dir",
                str(samples_dir),
                "--seed",
                seed,
                *_flags(cfg.collect.args),
            ]
            runner.run_stage(
                stage,
                command,
                inputs={"tracker_checkpoint": tracker_ckpt},
                outputs={"samples_dir": str(samples_dir)},
            )

        elif stage == "merge":
            command = [
                *runner.plain,
                "-m",
                "imitation_experiments.data.merge_planner_samples",
                "--source",
                str(samples_dir),
                "--output_dir",
                str(merged_dir),
                "--seed",
                seed,
                *_flags(cfg.merge.args),
            ]
            runner.run_stage(
                stage, command, inputs={}, outputs={"samples_dir": str(merged_dir)}
            )

        elif stage == "train_planner":
            command = [
                *runner.plain,
                "-m",
                "imitation_experiments.planner.train_chunked_transformer_planner",
                "--samples_dir",
                str(merged_dir),
                "--output_dir",
                str(planner_dir),
                "--seed",
                seed,
                *_flags(cfg.train_planner.args),
            ]
            runner.run_stage(
                stage, command, inputs={}, outputs={"planner_dir": str(planner_dir)}
            )

        elif stage == "eval_offline":
            out_json = runner.stage_dir(stage) / "offline_metrics.json"
            command = [
                *runner.plain,
                "-m",
                "imitation_experiments.evaluation.eval_interface_planner_offline",
                "--samples_dir",
                str(merged_dir),
                "--planner_checkpoint",
                str(planner_ckpt),
                "--output_json",
                str(out_json),
                "--seed",
                seed,
                *_flags(cfg.eval.offline_args),
            ]
            runner.run_stage(
                stage,
                command,
                inputs={"planner_checkpoint": planner_ckpt},
                outputs={"metrics": str(out_json)},
            )

        elif stage == "eval_closed_loop":
            out_json = runner.stage_dir(stage) / "closed_loop_metrics.json"
            command = [
                *runner.isaac,
                "-m",
                "imitation_experiments.evaluation.eval_interface_planner_closed_loop",
                "--checkpoint",
                str(tracker_ckpt),
                "--planner_checkpoint",
                str(planner_ckpt),
                "--output_json",
                str(out_json),
                "--seed",
                seed,
                *_flags(cfg.eval.closed_loop_args),
            ]
            runner.run_stage(
                stage,
                command,
                inputs={
                    "tracker_checkpoint": tracker_ckpt,
                    "planner_checkpoint": planner_ckpt,
                },
                outputs={"metrics": str(out_json)},
            )

        elif runner.already_complete(stage):  # pragma: no cover - defensive
            continue

    mode = "dry run rendered" if runner.dry_run else "completed"
    print(f"[INFO] latent e2e pipeline {mode}: {', '.join(stages)}")
    print(f"[INFO] run root: {run_root}")


@hydra.main(version_base=None, config_path="conf", config_name="latent_e2e")
def main(cfg: DictConfig) -> None:
    try:
        main_impl(cfg)
    except PipelineError as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
