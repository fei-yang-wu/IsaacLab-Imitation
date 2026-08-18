"""Login-node preflight: verify a plan's remote assumptions before any sbatch.

:func:`container_to_remote` is the single encoding of the job's bind model.
Every failure class it guards has a recorded incident: dataset invisible
without a bind ("manifest missing", ICE job 5577484), output path outside the
binds writing to node-local tmp until a multi-GB save dies (job 5577507), and
quota exhaustion silently killing checkpoint saves (the ICE 300 GB cap).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from imitation_experiments.paper.common import PipelineError

from .config import ClusterProfile, ResolvedJobSet
from .remote import ssh_run

# Absolute paths inside the container that are legitimate without a data bind.
CONTAINER_INTERNAL_PREFIXES: tuple[str, ...] = (
    "/workspace/isaaclab/project",
    "/isaac-sim",
    "/opt",
    "/tmp",
    "/dev",
)
_LOGS_CONTAINER_PREFIX = "/workspace/isaaclab/project/logs"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def container_to_remote(path: str, profile: ClusterProfile) -> str:
    """Map a container-visible absolute path to its login-node path."""
    if not path.startswith("/"):
        raise PipelineError(f"container path must be absolute: '{path}'")
    if path == "/data" or path.startswith("/data/"):
        return profile.data_dir + path[len("/data") :]
    if profile.project_logs_dir and (
        path == _LOGS_CONTAINER_PREFIX or path.startswith(_LOGS_CONTAINER_PREFIX + "/")
    ):
        return profile.project_logs_dir + path[len(_LOGS_CONTAINER_PREFIX) :]
    for bind in profile.extra_bind_paths:
        if path == bind or path.startswith(bind + "/"):
            return path
    raise PipelineError(
        f"path '{path}' is not visible under the job's binds "
        f"(/data -> {profile.data_dir}, extra: {list(profile.extra_bind_paths)}); "
        "inside --containall it would resolve to node-local temp"
    )


def _argv_path_checks(
    profile: ClusterProfile, jobset: ResolvedJobSet
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for stage in jobset.stages:
        for arg in stage.args:
            value = arg.partition("=")[2] if "=" in arg else arg
            if not value.startswith("/"):
                continue
            if value.startswith(CONTAINER_INTERNAL_PREFIXES):
                continue
            name = f"argv_path[{stage.name}]"
            try:
                container_to_remote(value, profile)
            except PipelineError as exc:
                results.append(CheckResult(name=name, ok=False, detail=str(exc)))
    return results


def _remote_check_script(profile: ClusterProfile, jobset: ResolvedJobSet) -> str:
    """One bash script, one ssh round-trip, TSV results on stdout."""
    lines = [
        'emit() { printf \'%s\\t%s\\t%s\\n\' "$1" "$2" "$3"; }',
        'nearest_existing() { p="$1"; while [ ! -e "$p" ] && [ "$p" != / ]; do p="$(dirname "$p")"; done; printf \'%s\' "$p"; }',
    ]

    def check_exists(name: str, remote_path: str) -> str:
        quoted = shlex.quote(remote_path)
        return (
            f"if [ -e {quoted} ]; then emit {shlex.quote(name)} OK {quoted}; "
            f'else emit {shlex.quote(name)} FAIL "missing: "{quoted}; fi'
        )

    for container_path in jobset.require_container_paths:
        remote_path = container_to_remote(container_path, profile)
        lines.append(check_exists(f"dataset:{container_path}", remote_path))

    output_remote = container_to_remote(jobset.output_container_path, profile)
    quoted_output = shlex.quote(output_remote)
    lines.append(
        f'anchor="$(nearest_existing {quoted_output})"; '
        f'if [ -w "$anchor" ]; then emit output_writable OK "$anchor"; '
        f'else emit output_writable FAIL "not writable: $anchor"; fi'
    )
    if profile.min_free_gb > 0:
        min_kb = profile.min_free_gb * 1024 * 1024
        lines.append(
            f'anchor="$(nearest_existing {quoted_output})"; '
            f"avail=$(df -Pk \"$anchor\" | awk 'NR==2 {{print $4}}'); "
            f'if [ "${{avail:-0}}" -ge {min_kb} ]; then emit output_free_space OK "${{avail}}K available"; '
            f'else emit output_free_space FAIL "only ${{avail:-0}}K available, need {min_kb}K"; fi'
        )

    quoted_log_dir = shlex.quote(profile.slurm.log_dir)
    lines.append(
        f"if mkdir -p {quoted_log_dir} 2>/dev/null && [ -w {quoted_log_dir} ]; then "
        f"emit slurm_log_dir OK {quoted_log_dir}; "
        f'else emit slurm_log_dir FAIL "cannot create/write: "{quoted_log_dir}; fi'
    )
    lines.append(check_exists("shared_sif", profile.shared_sif_path))
    if profile.hf_token_file:
        lines.append(check_exists("hf_token_file", profile.hf_token_file))
    if profile.wandb_api_key_file:
        lines.append(check_exists("wandb_api_key_file", profile.wandb_api_key_file))
    quoted_control = shlex.quote(profile.control_root)
    lines.append(
        f"if mkdir -p {quoted_control} 2>/dev/null && [ -w {quoted_control} ]; then "
        f"emit control_root OK {quoted_control}; "
        f'else emit control_root FAIL "cannot create/write: "{quoted_control}; fi'
    )
    return "\n".join(lines) + "\n"


def run_preflight(profile: ClusterProfile, jobset: ResolvedJobSet) -> list[CheckResult]:
    results = _argv_path_checks(profile, jobset)
    try:
        container_to_remote(jobset.output_container_path, profile)
    except PipelineError as exc:
        results.append(
            CheckResult(name="output_path_mapped", ok=False, detail=str(exc))
        )
        return results

    proc = ssh_run(profile.login, _remote_check_script(profile, jobset))
    for line in proc.stdout.decode().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        name, status, detail = parts
        results.append(CheckResult(name=name, ok=status == "OK", detail=detail))
    if not any(r.name.startswith(("dataset:", "output_writable")) for r in results):
        results.append(
            CheckResult(
                name="remote_script",
                ok=False,
                detail="remote preflight produced no recognizable results",
            )
        )
    return results
