from __future__ import annotations

from pathlib import Path

import pytest

from resolve_low_level_checkpoint import resolve_checkpoint


def _write_run(root: Path, name: str, experiment_name: str, checkpoint: str) -> Path:
    run = root / name
    (run / "models").mkdir(parents=True)
    (run / "command.txt").write_text(
        "python train.py "
        f"agent.logger.exp_name={experiment_name} "
        "agent.logger.backend=wandb\n",
        encoding="utf-8",
    )
    path = run / "models" / checkpoint
    path.write_bytes(b"checkpoint")
    return path.resolve()


def test_resolves_only_exact_recorded_experiment_name(tmp_path: Path) -> None:
    expected = _write_run(
        tmp_path,
        "2026-07-15_01-00-00",
        "paper_latent_train_oracle_low_level",
        "model_step_5000232960.pt",
    )
    _write_run(
        tmp_path,
        "2026-07-15_01-00-01",
        "paper_latent_train_debug_oracle_low_level",
        "model_step_5000232960.pt",
    )
    assert (
        resolve_checkpoint(
            tmp_path,
            run_id="paper_latent_train",
            checkpoint_basename="model_step_5000232960.pt",
        )
        == expected
    )


def test_rejects_duplicate_exact_run_records(tmp_path: Path) -> None:
    for index in range(2):
        _write_run(
            tmp_path,
            f"run_{index}",
            "paper_latent_train_oracle_low_level",
            "model_step_5000232960.pt",
        )
    with pytest.raises(ValueError, match="found 2"):
        resolve_checkpoint(
            tmp_path,
            run_id="paper_latent_train",
            checkpoint_basename="model_step_5000232960.pt",
        )


def test_rejects_missing_exact_final_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "command.txt").write_text(
        "python train.py "
        "agent.logger.exp_name=paper_latent_train_oracle_low_level\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Exact final checkpoint is missing"):
        resolve_checkpoint(
            tmp_path,
            run_id="paper_latent_train",
            checkpoint_basename="model_step_5000232960.pt",
        )
