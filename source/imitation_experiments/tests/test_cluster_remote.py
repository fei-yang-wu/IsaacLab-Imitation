"""Gate: the control plane's ssh/tar/sbatch primitives are strict and lossless.

Convention: a fake ``ssh`` on PATH drops the host argument and executes the
remainder locally, so remote logic runs for real against a filesystem under
``tmp_path`` with zero network.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from imitation_experiments.paper.common import PipelineError, sha256_file
from imitation_experiments.pipeline.cluster.remote import (
    build_workspace_archive,
    sbatch_parsable,
    ssh_upload,
    sync_workspace_archive,
    upload_and_verify_archive,
)


def _install_fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake_ssh = bin_dir / "ssh"
    fake_ssh.write_text('#!/bin/bash\nshift\nexec bash -c "$*"\n')
    fake_ssh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/train.py").write_text("print('hi')\n")
    (repo / "logs").mkdir()
    (repo / "logs/big.ckpt").write_text("x" * 100)
    (repo / "wandb").mkdir()
    (repo / "wandb/run.log").write_text("log")
    (repo / "motion.npz").write_text("npz")
    (repo / ".git").mkdir()
    (repo / ".git/HEAD").write_text("ref")
    return repo


def test_archive_excludes_and_transform(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    archive = tmp_path / "workspace.tar.gz"
    sha = build_workspace_archive(repo, archive)
    assert sha == sha256_file(archive)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "workspace/scripts/train.py" in names
    # GNU tar leaves the bare "./" root entry untransformed; every real member
    # must land under workspace/.
    assert all(
        n in (".", "./", "workspace") or n.startswith("workspace/") for n in names
    )
    joined = "\n".join(names)
    for banned in ("logs/", "wandb/", ".git", ".npz"):
        assert banned not in joined, f"excluded pattern leaked into archive: {banned}"


def test_nested_environments_never_enter_the_archive(tmp_path: Path) -> None:
    """The 2026-08-17 bug: `./.pixi` was excluded, `external/*/.pixi` was not.

    A built Pixi environment inside a submodule put 7.4 GB into every
    submission. The compute node gets its torch and Isaac Lab from the
    container image, so no environment belongs in this archive at all.
    """
    repo = _fixture_repo(tmp_path)
    nested = repo / "external/Embodied-Control/.pixi/envs/lowlevel-sim/lib"
    nested.mkdir(parents=True)
    (nested / "libtorch.so").write_text("x" * 1000)
    (repo / "external/Embodied-Control/src").mkdir(parents=True)
    (repo / "external/Embodied-Control/src/mod.py").write_text("code\n")
    venv = repo / "third_party/tool/.venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("cfg")

    archive = tmp_path / "workspace.tar.gz"
    build_workspace_archive(repo, archive)

    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "workspace/external/Embodied-Control/src/mod.py" in names
    assert not [n for n in names if ".pixi" in n or ".venv" in n]


def test_oversized_archive_fails_and_names_the_offender(tmp_path: Path) -> None:
    """Silence is what made the first regression expensive."""
    repo = _fixture_repo(tmp_path)
    bulk = repo / "external/Bulky/payload"
    bulk.mkdir(parents=True)
    # Incompressible bytes, so the gzip result really does exceed the limit.
    (bulk / "blob.bin").write_bytes(os.urandom(2_000_000))

    with pytest.raises(PipelineError, match="over the") as excinfo:
        build_workspace_archive(repo, tmp_path / "workspace.tar.gz", max_bytes=500_000)

    assert "external/Bulky" in str(excinfo.value)


def test_workspace_store_is_content_addressed_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N plans from one tree hold one archive between them, not N copies."""
    _install_fake_ssh(tmp_path, monkeypatch)
    archive = tmp_path / "workspace.tar.gz"
    build_workspace_archive(_fixture_repo(tmp_path), archive)
    control_root = tmp_path / "remote/cluster_control"
    plan_one = control_root / "plans/p1"
    plan_two = control_root / "plans/p2"

    sha, store_path, reused = sync_workspace_archive(
        "ice", archive, control_root=str(control_root), remote_plan_dir=str(plan_one)
    )

    assert not reused
    assert store_path == f"{control_root}/workspaces/{sha}.tar.gz"
    assert Path(store_path).is_file()
    link = plan_one / "workspace.tar.gz"
    assert link.is_symlink() and link.resolve() == Path(store_path).resolve()
    # The batch script's `tar -xzf <plan_dir>/workspace.tar.gz` still works.
    with tarfile.open(link) as tar:
        assert "workspace/scripts/train.py" in tar.getnames()

    stored_mtime = Path(store_path).stat().st_mtime_ns
    _sha_again, _store_again, reused_again = sync_workspace_archive(
        "ice", archive, control_root=str(control_root), remote_plan_dir=str(plan_two)
    )

    assert reused_again
    assert Path(store_path).stat().st_mtime_ns == stored_mtime, "re-uploaded a copy"
    assert (plan_two / "workspace.tar.gz").resolve() == Path(store_path).resolve()


def test_workspace_store_replaces_a_corrupt_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored entry that fails its own name is repaired, not trusted."""
    _install_fake_ssh(tmp_path, monkeypatch)
    archive = tmp_path / "workspace.tar.gz"
    sha = build_workspace_archive(_fixture_repo(tmp_path), archive)
    control_root = tmp_path / "remote/cluster_control"
    store_path = control_root / "workspaces" / f"{sha}.tar.gz"
    store_path.parent.mkdir(parents=True)
    store_path.write_text("truncated")

    _sha, _path, reused = sync_workspace_archive(
        "ice",
        archive,
        control_root=str(control_root),
        remote_plan_dir=str(control_root / "plans/p1"),
    )

    assert not reused
    assert sha256_file(store_path) == sha


def test_upload_and_verify_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    archive = tmp_path / "workspace.tar.gz"
    build_workspace_archive(_fixture_repo(tmp_path), archive)
    remote_dir = tmp_path / "remote/plans/p1"
    sha = upload_and_verify_archive("ice", archive, str(remote_dir))
    assert (remote_dir / "workspace.tar.gz").is_file()
    assert (remote_dir / "workspace.tar.gz.sha256").read_text().strip() == sha
    assert not (remote_dir / "workspace.tar.gz.partial").exists()


def test_upload_detects_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = _install_fake_ssh(tmp_path, monkeypatch)
    # Variant fake ssh: truncates whatever arrives on stdin before storing.
    (bin_dir / "ssh").write_text(
        '#!/bin/bash\nshift\nhead -c 10 > /dev/null\nexec bash -c "$*" < /dev/null\n'
    )
    archive = tmp_path / "workspace.tar.gz"
    build_workspace_archive(_fixture_repo(tmp_path), archive)
    with pytest.raises(PipelineError, match="hash mismatch"):
        upload_and_verify_archive("ice", archive, str(tmp_path / "remote"))


def test_ssh_upload_is_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    payload = tmp_path / "payload.txt"
    payload.write_text("payload with spaces & specials\n")
    target = tmp_path / "remote dir/nested/upload.txt"
    ssh_upload("ice", payload, str(target))
    assert target.read_text() == payload.read_text()


def test_sbatch_parsable_strict_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = _install_fake_ssh(tmp_path, monkeypatch)
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\necho 12345\n")
    fake_sbatch.chmod(0o755)
    job_id = sbatch_parsable("ice", str(tmp_path / "batch.sh"), chdir=str(tmp_path))
    assert job_id == "12345"

    fake_sbatch.write_text("#!/bin/sh\necho 'Submitted batch job 12345'\n")
    with pytest.raises(PipelineError, match="unparseable"):
        sbatch_parsable("ice", str(tmp_path / "batch.sh"), chdir=str(tmp_path))


def test_sbatch_parsable_forwards_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = _install_fake_ssh(tmp_path, monkeypatch)
    argv_log = tmp_path / "sbatch_argv.log"
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(f'#!/bin/sh\necho "$@" >> "{argv_log}"\necho 777\n')
    fake_sbatch.chmod(0o755)
    job_id = sbatch_parsable(
        "ice", "/plans/p1/batch.sh", chdir=str(tmp_path), dependency="afterok:123"
    )
    assert job_id == "777"
    assert "--dependency=afterok:123" in argv_log.read_text()


def test_remote_failure_raises_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ssh(tmp_path, monkeypatch)
    from imitation_experiments.pipeline.cluster.remote import ssh_run

    with pytest.raises(PipelineError, match="remote command failed"):
        ssh_run("ice", "echo doomed >&2; exit 3")
    proc = subprocess.run(
        ["ssh", "ice", "bash", "-s"], input=b"echo ok", capture_output=True
    )
    assert proc.stdout.strip() == b"ok"
