"""Diagnose object force balance and normalization-only policy KL drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def contact_force_from_balance(
    mass, velocity_before, velocity_after, virtual_force, gravity, dt
):
    """Infer total physical-contact force from the COM momentum balance."""
    if mass <= 0 or dt <= 0:
        raise ValueError("Mass and control timestep must be positive.")
    return (
        mass * (np.asarray(velocity_after) - np.asarray(velocity_before)) / dt
        - np.asarray(virtual_force)
        - mass * np.asarray(gravity)
    )


def probe_kl(
    checkpoint: Path,
    observations,
    *,
    device="cpu",
    cold=False,
    loss_forward=False,
    freeze_meter=False,
):
    import torch
    from tensordict import TensorDict
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MLP
    from rlopt.agent.ppo.ppo import RunningMeanStdCatInputs
    from rlopt.base_class import BaseAlgorithm
    from rlopt.models import GaussianPolicyHead

    torch.manual_seed(42)
    state = torch.load(checkpoint, map_location=device, weights_only=False)[
        "policy_state_dict"
    ]
    mlp = MLP(
        in_features=562,
        out_features=58,
        num_cells=[1024, 512, 256, 128],
        activation_class=torch.nn.ELU,
        device=device,
    )
    norm = RunningMeanStdCatInputs(mlp, 562)
    head = GaussianPolicyHead(norm, 58, clip_log_std=True, device=device)
    head.load_state_dict(
        {k.removeprefix("module.0.module."): v for k, v in state.items()}, strict=True
    )
    if cold:
        norm.running_mean.zero_()
        norm.running_var.fill_(1)
        norm.count.fill_(1)
    op = TensorDictModule(head, in_keys=["policy"], out_keys=["loc", "scale"])
    op.requires_grad_(False).eval()
    obs = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=device)
    td = TensorDict({"policy": obs}, batch_size=[len(obs)], device=device)
    with torch.no_grad():
        old = op(td.clone())
    count_before = float(norm.count)
    parameters = {k: v.clone() for k, v in op.named_parameters()}
    optimizer = torch.optim.Adam(op.parameters(), lr=1e-3)
    owner = SimpleNamespace(optim=optimizer)
    cfg = SimpleNamespace(
        scheduler="adaptive",
        desired_kl=0.005,
        lr_adaptation_factor=1.5,
        min_lr=1e-5,
        max_lr=1e-2,
    )
    values = []
    op.train()
    with torch.no_grad():
        for _ in range(20):
            ids = torch.randint(len(obs), (768,), device=device)
            batch = old[ids].clone()
            context = BaseAlgorithm._prepare_kl_context(owner, batch, op)
            if loss_forward:
                op(batch.clone())
            if freeze_meter:
                op.eval()
            kl = BaseAlgorithm._compute_kl_after_update(owner, context, op)
            if freeze_meter:
                op.train()
            assert kl is not None
            BaseAlgorithm._maybe_adjust_lr(owner, kl, cfg)
            values.append(float(kl))
    assert all(torch.equal(parameters[k], v) for k, v in op.named_parameters())
    return {
        "cold_normalizer": cold,
        "simulate_loss_forward": loss_forward,
        "meter_forced_eval": freeze_meter,
        "optimizer_steps": 0,
        "parameters_unchanged": True,
        "normalizer_count_before": count_before,
        "normalizer_count_after": float(norm.count),
        "kl_per_call": values,
        "maximum_kl": max(values),
        "final_adaptive_lr": optimizer.param_groups[0]["lr"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    meta = json.loads(args.rollouts.with_suffix(".json").read_text())
    assert meta["complete"] and meta["protocol"]["diagnostic_traces"]
    diag = meta["diagnostics"]
    result = {
        "qualification": "mechanistic diagnostic on recorded rollout observations; not a training-performance comparison",
        "rollout_sha256": hashlib.sha256(args.rollouts.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "force_balance": [],
        "normalizer_kl": [],
    }
    with np.load(args.rollouts, allow_pickle=False) as arrays:
        for row in meta["results"]:
            key = row["trajectory_key"]

            def get(name):
                return arrays[key + "__" + name]

            contact = contact_force_from_balance(
                diag["mass_kg"],
                get("com_velocity_before_w"),
                get("com_velocity_after_w"),
                get("virtual_force_w"),
                diag["gravity_w"],
                diag["control_dt_s"],
            )
            window = slice(-40, None)
            result["force_balance"].append(
                {
                    "policy": row["policy"],
                    "assistance": row["assistance_scale"],
                    "window_control_steps": 40,
                    "mean_root_offset_w_m": get("root_error_w")[window]
                    .mean(0)
                    .tolist(),
                    "mean_virtual_force_w_n": get("virtual_force_w")[window]
                    .mean(0)
                    .tolist(),
                    "mean_inferred_physical_contact_force_w_n": contact[window]
                    .mean(0)
                    .tolist(),
                    "mean_measured_hand_contact_force_w_n": get("hand_contact_force_w")[
                        window
                    ]
                    .mean(0)
                    .tolist(),
                }
            )
        obs = arrays[args.trajectory_key + "__policy_observation"]
        for cold in (False, True):
            for loss_forward in (False, True):
                for freeze_meter in (False, True):
                    result["normalizer_kl"].append(
                        probe_kl(
                            args.checkpoint,
                            obs,
                            device=args.device,
                            cold=cold,
                            loss_forward=loss_forward,
                            freeze_meter=freeze_meter,
                        )
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
