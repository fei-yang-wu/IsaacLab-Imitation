"""Generate a cluster curve-evaluation campaign from a training campaign.

One Slurm job per arm scores that arm's whole budget axis in place, on the
cluster where its checkpoints already live. `eval_checkpoint_tree.py` keeps a
single Isaac Sim start and swaps policy weights across the milestones, which
collapses N container starts into one.

Every per-arm interface field -- code width, hold, phase, macro terms, stride,
anchor, horizon, and the posterior-versus-pretrained route -- is copied from
the TRAINING campaign, so an evaluation cannot silently disagree with the run
it scores. Writing these by hand for forty arms is how that drift happens.

    python -m imitation_experiments.reporting.build_curve_eval_campaign \
        --campaign experiments/campaigns/2026-08-30-latent-star-v2/campaign.yaml \
        --tree-root /storage/ice-shared/.../latent_star_v2_checkpoints \
        --out experiments/campaigns/2026-08-31-star-v2-curves/campaign.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from imitation_experiments.reporting.ablation_sections import load_campaign_arms

BODY_NAMES = (
    "[pelvis,left_hip_roll_link,left_knee_link,left_ankle_roll_link,"
    "right_hip_roll_link,right_knee_link,right_ankle_roll_link,torso_link,"
    "left_shoulder_roll_link,left_elbow_link,left_wrist_yaw_link,"
    "right_shoulder_roll_link,right_elbow_link,right_wrist_yaw_link]"
)
CELLS = "[2048,2048,1024,1024,512,512]"
REF_ARRAYS = (
    "/storage/ice-shared/vip-vwt/g1-imitation/datasets/bones_seed_full/"
    "ref_arrays/g1_bones_seed_sonic_full_129785_e714bbff_v1"
)
PERSIST_ID = "bones_seed_sonic_full_129785@e714bbff"


def _resolve(value: Any, merged: dict[str, Any], arm: str, seed: int) -> Any:
    """Expand the `${vars.X}` indirection the training campaign uses."""
    seen = 0
    while (
        isinstance(value, str) and value.startswith("${vars.") and value.endswith("}")
    ):
        value = merged[value[len("${vars.") : -1]]
        seen += 1
        if seen > 8:
            raise ValueError(f"cyclic var reference for {arm}")
    if isinstance(value, str):
        value = value.replace("${vars.output_root}", str(merged["output_root"]))
        value = value.replace("${vars.arm}", arm).replace("${vars.seed}", str(seed))
    return value


def encoder_path(merged: dict[str, Any], arm: str, seed: int, tree_root: str) -> str:
    """Where this arm's frozen encoder lives, inside the archived tree root.

    Most arms pretrain their own, but the cadence arms bind the HUB's file
    through a `${vars.hub_encoder}` indirection and never write one of their
    own. Assuming `<arm>/encoder/checkpoints/latest.pt` makes those jobs die on
    FileNotFoundError after paying for a full Isaac Sim start (2026-09-01).
    """
    raw = _resolve(merged.get("encoder_ckpt", ""), merged, arm, seed)
    text = str(raw)
    # The training campaign writes container paths under /data/<campaign>/;
    # the curves read the archived copy instead.
    marker = "/latent_star_v2/"
    if marker in text:
        return tree_root.rstrip("/") + "/" + text.split(marker, 1)[1]
    return f"{tree_root}/{arm}_seed{seed}/encoder/checkpoints/latest.pt"


def arm_stage(
    arm: str,
    merged: dict[str, Any],
    tree_root: str,
    eval_root: str,
    seed: int,
    cache_root: str,
    board: str,
    row: str,
) -> dict[str, Any]:
    posterior = str(merged.get("route", "pretrained")) == "posterior"
    tree = f"{tree_root}/{arm}_seed{seed}"
    z_dim = int(merged["z_dim"])
    code_latent = _resolve(merged.get("code_latent_dim", z_dim), merged, arm, seed)
    hold = int(merged["hold"])

    args: list[str] = [
        "--tree",
        tree,
        "--output_root",
        eval_root,
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--row",
        row,
        "--board",
        board,
        "--",
        "--task",
        str(merged["task"]),
        "--algo",
        "IPMD",
        "--agent_entry_point",
        "rlopt_ipmd_posterior_root_qpos_cfg_entry_point"
        if posterior
        else "rlopt_ipmd_tuned_cfg_entry_point",
        "--randomization",
        "none",
        "--action_sampling",
        "mode",
        "--steps",
        "10000",
        "--seed",
        "0",
        "--reference_start_frame",
        "0",
        "--reset_schedule",
        "sequential",
        "--skill_encoder_source",
        "checkpoint" if posterior else "pretrained",
        "--headless",
        "physics=newton_mjwarp",
        "env.sim.physics.solver_cfg.njmax=320",
        "env.sim.physics.solver_cfg.nconmax=200",
        "env.events.push_robot=null",
        "env.data.manifest=null",
        f"env.data.reference_arrays_dir={REF_ARRAYS}",
        f"env.data.persist_id={PERSIST_ID}",
        "env.data.reference_arrays_resident=true",
        "env.data.reference_arrays_warm_workers=16",
        "env.data.runtime_cache_device=cuda:0",
        "env.data.reference_prefetch_mode=off",
        "env.data.macro_cache_device=cuda:0",
        f"env.data.runtime_cache_body_names={BODY_NAMES}",
        "env.command_interface.actor=latent",
        f"env.command_interface.actor.dim={int(merged['command_dim'])}",
        "env.command_interface.encoder=single",
        f"agent.ipmd.latent_dim={int(merged['command_dim'])}",
    ]
    if posterior:
        args += [
            "env.command_interface.encoder.components=[joint_qpos,root_pos,root_ori]",
            "agent.ipmd.command_source=posterior",
            "agent.ipmd.latent_learning.method=patch_autoencoder",
            f"agent.ipmd.latent_learning.quantizer={merged.get('quantizer', 'identity')}",
            "agent.ipmd.latent_learning.patch_past_steps=0",
            "agent.ipmd.latent_learning.patch_future_steps=9",
            "agent.ipmd.latent_learning.train_posterior_through_policy="
            f"{merged.get('through_policy', 'true')}",
            f"agent.ipmd.latent_learning.recon_coeff={merged.get('recon_coeff', '1.0')}",
            f"agent.ipmd.latent_learning.kl_coeff={merged.get('kl_coeff', '0.0')}",
            *[str(t) for t in (merged.get("posterior_space_args") or [])],
        ]
    else:
        args += [
            "agent.ipmd.command_source=hl_skill",
            f"agent.ipmd.hl_skill_checkpoint_path={encoder_path(merged, arm, seed, tree_root)}",
            f"agent.ipmd.hl_skill_horizon_steps={int(merged['horizon'])}",
            f"agent.ipmd.hl_skill_command_mode={merged['command_mode']}",
            "agent.ipmd.hl_skill_finetune_enabled=false",
        ]
    args += [
        f"agent.ipmd.latent_steps_min={hold}",
        f"agent.ipmd.latent_steps_max={hold}",
        f"agent.ipmd.latent_learning.code_period={hold}",
        f"agent.ipmd.latent_learning.command_phase_mode={merged['phase_mode']}",
        f"agent.ipmd.latent_learning.command_phase_source={merged['phase_source']}",
        f"agent.ipmd.latent_learning.command_phase_period={int(merged['phase_period'])}",
        f"agent.ipmd.latent_learning.code_latent_dim={int(code_latent)}",
        f"env.expert_macro_state_terms={merged['macro_terms']}",
        f"env.expert_macro_frame_stride={int(merged['stride'])}",
        f"env.expert_macro_anchor_mode={merged['anchor_mode']}",
        "env.terminations.anchor_pos.params.threshold=0.25",
        "env.terminations.anchor_pos.params.down_threshold=0.25",
        "env.terminations.anchor_ori.params.threshold=1.0",
        "env.terminations.ee_body_pos.params.threshold=0.25",
        "env.terminations.ee_body_pos.params.down_threshold=0.25",
        "env.terminations.foot_pos_xyz=null",
        "env.terminations.base_too_low=null",
        f"agent.policy.num_cells={CELLS}",
        "agent.policy.activation_fn=silu",
        f"agent.value_function.num_cells={CELLS}",
        "agent.value_function.activation_fn=silu",
    ]
    # Env overrides the checkpoint's value function was built against must also
    # apply at eval; the restore is strict. Only `env.*` tokens pass through.
    args += [
        str(t)
        for t in (merged.get("extra_args") or [])
        if isinstance(t, str) and t.startswith("env.")
    ]
    return {
        "name": "score",
        "executable": "scripts/rlopt/eval_checkpoint_tree.py",
        # A PRIVATE Isaac Sim cache per arm. Two evaluation jobs sharing one
        # cache crashed the second inside Kit startup (ICE, 2026-08-27); with
        # one directory per arm there is nothing to contend for, so these can
        # run concurrently. A cold cache costs a slower first Kit start.
        "env": {"CLUSTER_ISAAC_SIM_CACHE_DIR": f"{cache_root}/{arm}"},
        # 25 milestones on the 4,096-clip board at roughly 3-4 minutes each,
        # plus one simulation start.
        "time_limit": "08:00:00",
        "gres": "gpu:h200:1",
        "partition": "coe-gpu",
        "qos": "coe-ice",
        "cpus_per_task": 16,
        "mem": "160G",
        "args": args,
    }


def build(
    campaign_path: Path,
    tree_root: str,
    eval_root: str,
    seed: int,
    cache_root: str = "/storage/ice-shared/vip-vwt/scratch-fwu91/isaac_cache_curves",
    board: str = "bones_testbed4096_v1",
    row: str = "clean",
) -> str:
    arms = load_campaign_arms(campaign_path)
    base = yaml.safe_load(campaign_path.read_text())["vars"]
    out_arms: dict[str, Any] = {}
    for arm, spec in sorted(arms.items()):
        merged = {**base, **spec}
        merged["output_root"] = f"{tree_root}/{arm}_seed{seed}"
        out_arms[arm] = {
            "vars": {},
            "stages": [
                arm_stage(
                    arm, merged, tree_root, eval_root, seed, cache_root, board, row
                )
            ],
        }
    doc = {
        "name": "star-v2-curves",
        "profile": "ice",
        "wandb_project": "g1-bs-ablation",
        "wandb_group": "latent-star-v2",
        "vars": {"eval_root": eval_root},
        "shared_env": {"CLUSTER_SIM_BACKEND": "newton"},
        "preflight": {
            "require_container_paths": [REF_ARRAYS],
            "output_container_path": eval_root,
        },
        "arms": out_arms,
    }
    header = (
        "# GENERATED by imitation_experiments.reporting.build_curve_eval_campaign.\n"
        "# Do not hand-edit: every interface field is copied from the training\n"
        "# campaign so evaluation cannot drift from the run it scores. Re-run the\n"
        "# generator instead.\n"
        "#\n"
        "# One job per arm. `eval_checkpoint_tree.py` skips already-scored cells\n"
        "# unless --rescore, so re-submitting as arms deepen is cheap.\n"
        "#\n"
        "# Each arm gets a PRIVATE Isaac Sim cache dir, which is what made the\n"
        "# 2026-08-27 concurrent-eval crash possible; with no shared cache these\n"
        "# can run in parallel. Ramp concurrency rather than trusting it blindly.\n"
    )
    return header + yaml.safe_dump(doc, sort_keys=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--tree-root", required=True)
    parser.add_argument("--eval-root", default="/data/eval/star_v2_curves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--board", default="bones_testbed4096_v1")
    parser.add_argument("--row", default="clean")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    text = build(
        args.campaign,
        args.tree_root,
        args.eval_root,
        args.seed,
        board=args.board,
        row=args.row,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
