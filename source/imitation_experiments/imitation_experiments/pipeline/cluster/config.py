"""Typed cluster profiles and campaign specs for the submission control plane.

The profile's open ``env`` dict is deliberate: every key placed there reaches
the job verbatim through the frozen env file. There is no forwarding
allow-list anywhere on the new path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from imitation_experiments.paper.common import PipelineError

from .slurm import external_dependency_job_id

# Campaign specs need small derived values (frames-per-batch arithmetic) and
# argv-list composition (shared blocks + per-arm blocks). Registered once,
# replace=True keeps re-imports idempotent.
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
OmegaConf.register_new_resolver(
    "ceil_div", lambda a, b: (int(a) + int(b) - 1) // int(b), replace=True
)
OmegaConf.register_new_resolver(
    "floor_div", lambda a, b: int(a) // int(b), replace=True
)
OmegaConf.register_new_resolver(
    "concat", lambda *lists: [item for chunk in lists for item in chunk], replace=True
)
OmegaConf.register_new_resolver(
    "dash", lambda value: str(value).replace("_", "-"), replace=True
)


@dataclass
class SlurmDefaults:
    gres: str = MISSING
    cpus_per_task: int = MISSING
    mem: str = MISSING
    log_dir: str = MISSING
    account: str | None = None
    qos: str | None = None
    partition: str | None = None
    mem_per_gpu: str | None = None
    nodes: int = 1
    ntasks: int = 1
    time_limit: str = "15:59:00"
    job_name_prefix: str = "imit"


@dataclass
class ClusterProfile:
    name: str = MISSING
    login: str = MISSING
    control_root: str = MISSING
    data_dir: str = MISSING
    shared_sif_path: str = MISSING
    isaac_cache_dir: str = MISSING
    job_tmpdir_root: str = "/tmp"
    project_logs_dir: str | None = None
    extra_bind_paths: list[str] = field(default_factory=list)
    hf_token_file: str | None = None
    wandb_api_key_file: str | None = None
    min_free_gb: int = 0
    slurm: SlurmDefaults = field(default_factory=SlurmDefaults)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class StageSpec:
    name: str = MISSING
    executable: str = MISSING
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    time_limit: str | None = None
    gres: str | None = None
    mem: str | None = None
    cpus_per_task: int | None = None
    # Comma-separated node list handed to `sbatch --exclude`. Use it to keep a
    # chain off a node that is delivering degraded throughput; record why in
    # the campaign README, because it changes which hardware a row ran on.
    exclude: str | None = None
    depends_on: str | None = None
    # "afterok" (default) runs only after a clean predecessor; "afterany" also
    # runs after TIMEOUT/FAILED, which is what a walltime-segmented resume
    # chain needs -- the successor picks up from the final resume checkpoint.
    dependency_kind: str = "afterok"


@dataclass
class ArmSpec:
    stages: list[StageSpec] = field(default_factory=list)
    vars: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightSpec:
    require_container_paths: list[str] = field(default_factory=list)
    output_container_path: str = MISSING


@dataclass
class CampaignSpec:
    name: str = MISSING
    profile: str = MISSING
    wandb_project: str = MISSING
    wandb_group: str = MISSING
    vars: dict[str, Any] = field(default_factory=dict)
    shared_env: dict[str, str] = field(default_factory=dict)
    preflight: PreflightSpec = field(default_factory=PreflightSpec)
    arms: dict[str, ArmSpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedStage:
    name: str
    executable: str
    args: tuple[str, ...]
    env: dict[str, str]
    time_limit: str | None
    gres: str | None
    mem: str | None
    cpus_per_task: int | None
    exclude: str | None
    depends_on: str | None
    dependency_kind: str


@dataclass(frozen=True)
class ResolvedJobSet:
    campaign_name: str
    campaign_path: str
    profile_name: str
    arm: str
    seed: int
    wandb_project: str
    wandb_group: str
    require_container_paths: tuple[str, ...]
    output_container_path: str
    shared_env: dict[str, str]
    stages: tuple[ResolvedStage, ...]


def _package_profile_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "conf" / f"profile_{name}.yaml"


def load_profile(name_or_path: str) -> ClusterProfile:
    candidate = Path(name_or_path)
    if not candidate.suffix == ".yaml":
        candidate = _package_profile_path(name_or_path)
    if not candidate.is_file():
        raise PipelineError(
            f"cluster profile not found: '{name_or_path}' ({candidate})"
        )
    try:
        merged = OmegaConf.merge(
            OmegaConf.structured(ClusterProfile), OmegaConf.load(candidate)
        )
        profile = cast(ClusterProfile, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        raise PipelineError(f"invalid cluster profile '{candidate}': {exc}") from exc
    _validate_profile(profile, candidate)
    return profile


def _validate_profile(profile: ClusterProfile, source: Path) -> None:
    if "." in profile.login:
        raise PipelineError(
            f"profile '{source}': login must be an ssh alias (got '{profile.login}'); "
            "keep host keys/config under the alias in ~/.ssh/config"
        )
    for label, value in (
        ("control_root", profile.control_root),
        ("data_dir", profile.data_dir),
        ("shared_sif_path", profile.shared_sif_path),
        ("isaac_cache_dir", profile.isaac_cache_dir),
        ("slurm.log_dir", profile.slurm.log_dir),
    ):
        if not value.startswith("/"):
            raise PipelineError(
                f"profile '{source}': {label} must be absolute, got '{value}'"
            )
    for bind in profile.extra_bind_paths:
        if not bind.startswith("/"):
            raise PipelineError(
                f"profile '{source}': extra_bind_paths entry not absolute: '{bind}'"
            )
    if profile.min_free_gb < 0:
        raise PipelineError(f"profile '{source}': min_free_gb must be >= 0")


def load_campaign(
    path: Path, *, arm: str, seed: int, overrides: list[str] | None = None
) -> ResolvedJobSet:
    if not path.is_file():
        raise PipelineError(f"campaign spec not found: {path}")
    try:
        cfg = OmegaConf.merge(OmegaConf.structured(CampaignSpec), OmegaConf.load(path))
    except OmegaConfBaseException as exc:
        raise PipelineError(f"invalid campaign spec '{path}': {exc}") from exc

    arm_names = list(cfg.arms.keys())
    if arm not in arm_names:
        raise PipelineError(
            f"unknown arm '{arm}' in {path}; arms: {', '.join(sorted(arm_names))}"
        )

    # Precedence: campaign vars < arm-local vars < CLI dotlist overrides;
    # arm/seed are identity and always win.
    try:
        cfg.vars = OmegaConf.merge(cfg.vars, cfg.arms[arm].vars)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    except OmegaConfBaseException as exc:
        raise PipelineError(f"invalid overrides for '{path}': {exc}") from exc
    cfg.vars.arm = arm
    cfg.vars.seed = seed
    try:
        OmegaConf.resolve(cfg)
        campaign = cast(CampaignSpec, OmegaConf.to_object(cfg))
    except OmegaConfBaseException as exc:
        raise PipelineError(
            f"unresolved campaign spec '{path}' (arm={arm}): {exc}"
        ) from exc

    stages = tuple(
        ResolvedStage(
            name=stage.name,
            executable=stage.executable,
            args=tuple(str(a) for a in stage.args),
            env={k: str(v) for k, v in stage.env.items()},
            time_limit=stage.time_limit,
            gres=stage.gres,
            mem=stage.mem,
            cpus_per_task=stage.cpus_per_task,
            exclude=getattr(stage, "exclude", None),
            depends_on=stage.depends_on,
            dependency_kind=str(getattr(stage, "dependency_kind", "afterok")),
        )
        for stage in campaign.arms[arm].stages
    )
    if not stages:
        raise PipelineError(f"arm '{arm}' in {path} declares no stages")
    stage_names = [s.name for s in stages]
    if len(set(stage_names)) != len(stage_names):
        raise PipelineError(
            f"arm '{arm}' in {path} has duplicate stage names: {stage_names}"
        )
    for stage in stages:
        if (
            stage.depends_on is not None
            and stage.depends_on not in stage_names
            and external_dependency_job_id(stage.depends_on) is None
        ):
            raise PipelineError(
                f"stage '{stage.name}' depends on unknown stage "
                f"'{stage.depends_on}' (a live Slurm job takes the form "
                f"'job:<id>')"
            )
    return ResolvedJobSet(
        campaign_name=campaign.name,
        campaign_path=str(path),
        profile_name=campaign.profile,
        arm=arm,
        seed=seed,
        wandb_project=campaign.wandb_project,
        wandb_group=campaign.wandb_group,
        require_container_paths=tuple(campaign.preflight.require_container_paths),
        output_container_path=campaign.preflight.output_container_path,
        shared_env={k: str(v) for k, v in campaign.shared_env.items()},
        stages=stages,
    )


def resolved_env(
    profile: ClusterProfile, jobset: ResolvedJobSet, stage: ResolvedStage
) -> dict[str, str]:
    """Full frozen environment for one job. Merge order (later wins):
    profile.env < fields derived from the typed profile < campaign.shared_env
    < stage.env; the stage executable always wins for CLUSTER_PYTHON_EXECUTABLE.
    """
    derived: dict[str, str] = {
        "CLUSTER_DATA_DIR": profile.data_dir,
        "CLUSTER_ISAAC_SIM_CACHE_DIR": profile.isaac_cache_dir,
        "CLUSTER_SHARED_SIF_PATH": profile.shared_sif_path,
        "CLUSTER_SIF_PATH": str(Path(profile.shared_sif_path).parent),
        "CLUSTER_JOB_TMPDIR_ROOT": profile.job_tmpdir_root,
        "CLUSTER_USE_SHARED_SIF": "1",
    }
    if profile.extra_bind_paths:
        derived["CLUSTER_EXTRA_BIND_PATHS"] = ":".join(profile.extra_bind_paths)
    if profile.project_logs_dir:
        derived["CLUSTER_PROJECT_LOGS_DIR"] = profile.project_logs_dir
    if profile.hf_token_file:
        derived["CLUSTER_HF_TOKEN_FILE"] = profile.hf_token_file
    if profile.wandb_api_key_file:
        derived["CLUSTER_WANDB_API_KEY_FILE"] = profile.wandb_api_key_file

    merged = dict(profile.env)
    merged.update(derived)
    merged.update(jobset.shared_env)
    merged.update(stage.env)
    merged["CLUSTER_PYTHON_EXECUTABLE"] = stage.executable
    return dict(sorted(merged.items()))
