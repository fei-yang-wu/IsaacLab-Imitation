"""The W&B run-id block must survive `set -euo pipefail`.

Regression for 2026-08-19: the block generated its random token with
`tr -dc 'a-z0-9' < /dev/urandom | head -c 6`. `head` closes the pipe after six
bytes while `tr` is still streaming an endless source, so `tr` takes SIGPIPE,
`pipefail` surfaces 141, and `set -e` kills the job at zero seconds with no
error message in the log.

It fired on every stage that declares a run id -- every low-level stage, and no
pretrain, which is exactly the signature that made it look like a per-stage
config problem. 21 arms of the interface design study died this way after their
pretrains had already completed.

These tests RUN the rendered shell rather than pattern-matching it, because the
failure was a runtime signal and not a visible syntax defect.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from imitation_experiments.pipeline.cluster.slurm import (
    _WANDB_RUN_ID_MAX,
    _render_run_id_block,
)


def _run(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _harness(tmp_path: Path, state_file: Path, declared: str = "ids-ctrl-s0") -> str:
    """The rendered block with the surrounding contract it runs under."""
    workspace = tmp_path / "ws" / "docker" / "cluster"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "job_env.resolved.sh").write_text(
        f"export WANDB_RUN_ID={declared}\n", encoding="utf-8"
    )
    return (
        "set -euo pipefail\n"
        f'extracted_workspace="{tmp_path / "ws"}"\n'
        + _render_run_id_block(str(state_file))
    )


def test_the_block_creates_a_run_id_under_pipefail(tmp_path: Path) -> None:
    state = tmp_path / "state" / "wandb_run_id"
    result = _run(_harness(tmp_path, state), tmp_path)
    assert result.returncode == 0, f"exit {result.returncode}: {result.stderr}"
    assert state.is_file()
    run_id = state.read_text(encoding="utf-8")
    assert run_id.startswith("ids-ctrl-s0-")
    assert len(run_id) <= _WANDB_RUN_ID_MAX


def test_the_created_id_is_random_per_chain(tmp_path: Path) -> None:
    ids = set()
    for index in range(4):
        state = tmp_path / f"state{index}" / "wandb_run_id"
        assert _run(_harness(tmp_path, state), tmp_path).returncode == 0
        ids.add(state.read_text(encoding="utf-8"))
    assert len(ids) == 4


def test_an_existing_id_is_reused_not_regenerated(tmp_path: Path) -> None:
    """A chain keeps one run id across resumes; W&B refuses a deleted id with a
    410 that kills the job, so this must never mint a second one."""
    state = tmp_path / "state" / "wandb_run_id"
    state.parent.mkdir(parents=True)
    state.write_text("ids-ctrl-s0-abc123", encoding="utf-8")
    result = _run(_harness(tmp_path, state), tmp_path)
    assert result.returncode == 0
    assert state.read_text(encoding="utf-8") == "ids-ctrl-s0-abc123"
    assert "ids-ctrl-s0-abc123" in result.stdout


def test_the_id_is_appended_to_the_env_file(tmp_path: Path) -> None:
    state = tmp_path / "state" / "wandb_run_id"
    assert _run(_harness(tmp_path, state), tmp_path).returncode == 0
    env = (tmp_path / "ws" / "docker" / "cluster" / "job_env.resolved.sh").read_text(
        encoding="utf-8"
    )
    assert env.count("export WANDB_RUN_ID=") == 2
    assert env.strip().endswith(state.read_text(encoding="utf-8"))


def test_a_stage_without_a_declared_id_is_a_no_op(tmp_path: Path) -> None:
    """Pretrain declares no run id. It must not create a state file."""
    state = tmp_path / "state" / "wandb_run_id"
    workspace = tmp_path / "ws" / "docker" / "cluster"
    workspace.mkdir(parents=True)
    (workspace / "job_env.resolved.sh").write_text("export FOO=1\n", encoding="utf-8")
    script = (
        "set -euo pipefail\n"
        f'extracted_workspace="{tmp_path / "ws"}"\n' + _render_run_id_block(str(state))
    )
    assert _run(script, tmp_path).returncode == 0
    assert not state.exists()


def test_compile_caches_are_scoped_to_the_job() -> None:
    """Concurrent jobs of one campaign must not share a compile cache.

    Regression for 2026-08-20: nine posterior arms started together and
    `post_recon_vq` died six minutes in with
    `InductorError: FileNotFoundError ... .tmp -> ....py` -- a lost rename in
    the shared `~/.cache/torchinductor`. Its eight siblings were unaffected,
    which is what makes this look like a bad arm rather than a race.
    """
    from imitation_experiments.pipeline.cluster.slurm import (
        SlurmDirectives,
        render_batch_script,
    )

    body = render_batch_script(
        SlurmDirectives(
            job_name="probe",
            log_dir="/tmp/logs",
            time_limit="00:10:00",
            cpus_per_task=1,
            gres="gpu:h200:1",
            mem="160G",
        ),
        remote_plan_dir="/remote/plan",
        stage="lowlevel",
        job_args=["--task", "X"],
        job_tmpdir_root="/tmp",
    )
    assert 'TORCHINDUCTOR_CACHE_DIR="$bootstrap_root/torchinductor"' in body
    assert 'TRITON_CACHE_DIR="$bootstrap_root/triton"' in body


def test_no_unbounded_source_feeds_a_truncating_head(tmp_path: Path) -> None:
    """The shape of the original bug, pinned so it cannot come back."""
    block = _render_run_id_block("/tmp/state")
    assert "/dev/urandom | head" not in block
    assert "urandom" not in block or "-N" in block


@pytest.mark.parametrize("declared", ["a", "ids-ctrl-s0", "x" * 40])
def test_the_id_never_exceeds_the_wandb_cap(tmp_path: Path, declared: str) -> None:
    """RLOpt adds a `logdir:` tag and W&B caps a tag at 64 characters, which
    leaves 31 for the id."""
    state = tmp_path / f"state-{len(declared)}" / "wandb_run_id"
    assert _run(_harness(tmp_path, state, declared), tmp_path).returncode == 0
    assert len(state.read_text(encoding="utf-8")) <= _WANDB_RUN_ID_MAX
