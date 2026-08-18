"""ssh/sbatch primitives for the control plane.

Every network interaction funnels through :func:`ssh_run`, so tests fake one
``ssh`` binary and the rest of the stack runs for real. ``sbatch --parsable``
replaces the two incompatible "Submitted batch job ..." text parsers that the
legacy campaign wrappers grew.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from imitation_experiments.paper.common import PipelineError, sha256_file

# What the compute node needs is this repository's SOURCE. Every heavy
# dependency -- torch, Isaac Sim, Isaac Lab -- already lives inside the
# container image's `container-runtime` Pixi environment, so nothing that ships
# here should be a built environment or a media payload.
#
# The patterns below excluded `./.pixi` but not nested ones, which silently put
# `external/Embodied-Control/.pixi` (7.4 GB of built environments) into every
# submission until 2026-08-17. ARCHIVE_MAX_BYTES exists so the next such
# accident fails the submit instead of costing quota and upload time.
ARCHIVE_EXCLUDES: tuple[str, ...] = (
    "./.git",
    "*/.git",
    "./.pixi",
    # Built Pixi/venv trees anywhere in the workspace, not just at the root.
    "*/.pixi",
    "*/.venv",
    "*/node_modules",
    "./.codex",
    "./.claude",
    "./data",
    "./.tmp",
    "./logs",
    "./outputs",
    "./output",
    "./runs",
    "./videos",
    "./wandb",
    "*/__pycache__",
    "*/.pytest_cache",
    "*/.ruff_cache",
    "*.npz",
    "*.sif",
    # Upstream demo and media payloads under external/: the cluster imports the
    # `gr00t` package, never these.
    "./external/Isaac-GR00T/demo_data",
    "./external/Isaac-GR00T/media",
    "./external/Isaac-GR00T/examples",
    "./external/Isaac-GR00T/external_dependencies",
    "./external/Isaac-GR00T/scripts/deployment",
    # Generated run trees inside submodules (`./logs` covers the top level).
    "*/logs",
    # Deployment-rig assets (policy bundles, meshes) and notebooks: the
    # workstation reads these, no cluster job does.
    "./external/Embodied-Control/assets",
    "./external/Embodied-Control/notebooks",
)

# A workspace archive is source code. 400 MB is far above the real figure
# (~30 MB) and far below any accidental environment or dataset.
ARCHIVE_MAX_BYTES = 400 * 1024 * 1024

_PARSABLE_RE = re.compile(r"^([0-9]+)(;\S+)?$")


def ssh_run(
    login: str,
    script: str,
    *,
    stdin_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run a bash script on the remote host; optionally stream a file to stdin."""
    remote_cmd = script if stdin_path is None else f"( {script} )"
    argv = ["ssh", login, "bash", "-s"]
    if stdin_path is not None:
        # The file goes to stdin, so the script itself travels as an argument.
        argv = ["ssh", login, remote_cmd]
        with stdin_path.open("rb") as handle:
            proc = subprocess.run(argv, stdin=handle, capture_output=True, check=False)
    else:
        proc = subprocess.run(
            argv, input=script.encode(), capture_output=True, check=False
        )
    if check and proc.returncode != 0:
        raise PipelineError(
            f"remote command failed on '{login}' (exit {proc.returncode}): "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    return proc


def ssh_upload(login: str, local: Path, remote: str) -> None:
    """Atomic upload: write to ``.partial`` then rename (cluster_interface.sh:325)."""
    quoted_dir = shlex.quote(str(Path(remote).parent))
    quoted_partial = shlex.quote(f"{remote}.partial")
    quoted_final = shlex.quote(remote)
    ssh_run(
        login,
        f"mkdir -p {quoted_dir} && cat > {quoted_partial} && mv {quoted_partial} {quoted_final}",
        stdin_path=local,
    )


def build_workspace_archive(
    repo_root: Path, out: Path, *, max_bytes: int = ARCHIVE_MAX_BYTES
) -> str:
    """Pack the workspace source tree; return its sha256.

    Fails when the archive exceeds ``max_bytes``, naming what dominates it. An
    oversized archive is never harmless: it is uploaded per submit, stored per
    plan, and re-extracted by every chained segment.
    """
    cmd = ["tar", "-C", str(repo_root)]
    cmd.extend(f"--exclude={pattern}" for pattern in ARCHIVE_EXCLUDES)
    cmd.extend(["--transform", r"s#^\./#workspace/#", "-czf", str(out), "."])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise PipelineError(f"workspace archive build failed: {proc.stderr.strip()}")
    size = out.stat().st_size
    if size > max_bytes:
        raise PipelineError(
            f"workspace archive is {size / 1e6:.0f} MB, over the "
            f"{max_bytes / 1e6:.0f} MB limit. Heaviest paths:\n"
            + "\n".join(
                f"  {mb:8.1f} MB  {path}" for mb, path in archive_hot_spots(out)
            )
            + "\nAdd an ARCHIVE_EXCLUDES pattern, or raise the limit on purpose."
        )
    return sha256_file(out)


def archive_hot_spots(archive: Path, *, depth: int = 3, top: int = 10) -> list:
    """The heaviest ``depth``-deep prefixes inside an archive, largest first."""
    proc = subprocess.run(
        ["tar", "-tzvf", str(archive)], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        return []
    totals: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            size = int(fields[2])
        except ValueError:
            continue
        prefix = "/".join(fields[5].split("/")[:depth])
        totals[prefix] = totals.get(prefix, 0) + size
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [(size / 1e6, prefix) for prefix, size in ranked[:top]]


def upload_and_verify_archive(login: str, archive: Path, remote_dir: str) -> str:
    local_sha = sha256_file(archive)
    remote_path = f"{remote_dir}/workspace.tar.gz"
    ssh_upload(login, archive, remote_path)
    proc = ssh_run(login, f"sha256sum {shlex.quote(remote_path)}")
    remote_sha = proc.stdout.decode().split()[0] if proc.stdout.split() else ""
    if remote_sha != local_sha:
        raise PipelineError(
            f"workspace archive hash mismatch after upload: local={local_sha} remote={remote_sha}"
        )
    ssh_run(
        login,
        f"printf '%s\\n' {shlex.quote(local_sha)} > {shlex.quote(remote_path + '.sha256')}",
    )
    return local_sha


def sync_workspace_archive(
    login: str, archive: Path, *, control_root: str, remote_plan_dir: str
) -> tuple[str, str, bool]:
    """Publish the archive into the shared store; link the plan dir at it.

    The store is content addressed (``<control_root>/workspaces/<sha>.tar.gz``),
    so N plans built from one tree hold one copy between them instead of N.
    ``<plan_dir>/workspace.tar.gz`` stays a valid path -- it becomes a symlink
    into the store -- which keeps the batch script, the older submissions, and
    anyone reading a plan directory working unchanged.

    Returns ``(sha256, store_path, reused)``.
    """
    local_sha = sha256_file(archive)
    store_dir = f"{control_root}/workspaces"
    store_path = f"{store_dir}/{local_sha}.tar.gz"
    link_path = f"{remote_plan_dir}/workspace.tar.gz"

    ssh_run(login, f"mkdir -p {shlex.quote(store_dir)} {shlex.quote(remote_plan_dir)}")
    probe = ssh_run(
        login,
        f"if [ -f {shlex.quote(store_path)} ]; then sha256sum {shlex.quote(store_path)}; fi",
    )
    remote_sha = probe.stdout.decode().split()[0] if probe.stdout.split() else ""
    reused = remote_sha == local_sha
    if not reused:
        # A stored entry that fails its own name is corrupt; overwrite it.
        ssh_upload(login, archive, store_path)
        verify = ssh_run(login, f"sha256sum {shlex.quote(store_path)}")
        stored_sha = verify.stdout.decode().split()[0] if verify.stdout.split() else ""
        if stored_sha != local_sha:
            raise PipelineError(
                "workspace archive hash mismatch after upload: "
                f"local={local_sha} remote={stored_sha}"
            )
    ssh_run(
        login,
        f"printf '%s\\n' {shlex.quote(local_sha)} > {shlex.quote(store_path + '.sha256')} && "
        f"ln -sfn {shlex.quote(store_path)} {shlex.quote(link_path)} && "
        f"printf '%s\\n' {shlex.quote(local_sha)} > {shlex.quote(link_path + '.sha256')}",
    )
    return local_sha, store_path, reused


def sbatch_parsable(
    login: str,
    script_remote_path: str,
    *,
    chdir: str,
    dependency: str | None = None,
) -> str:
    """Submit and return the numeric job ID via the machine-readable interface."""
    dependency_arg = f" --dependency={shlex.quote(dependency)}" if dependency else ""
    proc = ssh_run(
        login,
        f"cd {shlex.quote(chdir)} && sbatch --parsable{dependency_arg} "
        f"{shlex.quote(script_remote_path)}",
    )
    stdout = proc.stdout.decode().strip()
    match = _PARSABLE_RE.match(stdout)
    if not match:
        raise PipelineError(
            f"sbatch --parsable returned unparseable output: '{stdout}'"
        )
    return match.group(1)
