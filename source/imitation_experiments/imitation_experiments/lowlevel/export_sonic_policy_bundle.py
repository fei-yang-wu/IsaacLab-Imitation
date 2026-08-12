"""Export a released SONIC tracker as an Embodied-Control policy bundle.

Run this command from the repository root:

    pixi run -e onnx-export python -m \
      imitation_experiments.lowlevel.export_sonic_policy_bundle \
      --checkpoint logs/downloaded_checkpoints/nvidia_GEAR_SONIC_9c0ff22/sonic_v1_1/last.pt \
      --output logs/policy_bundles/sonic_v1_1_native
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch import nn

from imitation_experiments.lowlevel.export_policy_bundle import (
    BUNDLE_API_VERSION,
    G1_ISAAC_JOINT_NAMES,
    ONNX_OPSET,
    _per_joint_actuation,
    _sdk_joint_names,
    _soft_joint_limits,
)
from imitation_experiments.lowlevel.sonic_release_actor import (
    SonicReleaseActor,
    load_sonic_release_actor,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


class _EncoderExport(nn.Module):
    def __init__(self, actor: SonicReleaseActor) -> None:
        super().__init__()
        self.encoder = actor.encoder
        self.quantizer = actor.quantizer

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(window)
        tokens = latent.reshape(latent.shape[0], 2, 32)
        return self.quantizer(tokens).reshape(latent.shape[0], 64)


class _DecoderExport(nn.Module):
    def __init__(self, actor: SonicReleaseActor) -> None:
        super().__init__()
        self.decoder = actor.decoder

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.decoder(observation)


def observation_contract() -> dict:
    terms = [
        {
            "name": "latent_command",
            "width": 64,
            "normalize": False,
            "history_length": 1,
            "history_stride": 1,
            "history_order": "oldest_first",
            "reset_fill": "repeat_first",
        }
    ]
    for name, width in (
        ("base_ang_vel", 3),
        ("joint_pos_rel", 29),
        ("joint_vel_rel", 29),
        ("last_action", 29),
        ("projected_gravity", 3),
    ):
        terms.append(
            {
                "name": name,
                "width": width,
                "normalize": False,
                "history_length": 10,
                "history_stride": 1,
                "history_order": "oldest_first",
                "reset_fill": "repeat_first",
            }
        )
    return {"terms": terms, "total_width": 994}


def action_contract() -> dict:
    actuation = _per_joint_actuation()
    lower, upper = _soft_joint_limits()
    sdk_names = _sdk_joint_names()
    return {
        "width": 29,
        "isaac_joint_names": G1_ISAAC_JOINT_NAMES,
        "sdk_joint_names": sdk_names,
        "isaac_to_sdk": [sdk_names.index(name) for name in G1_ISAAC_JOINT_NAMES],
        "default_joint_pos": actuation["default_joint_pos"],
        "default_joint_vel": [0.0] * 29,
        "action_scale": actuation["action_scale"],
        "stiffness": actuation["stiffness"],
        "damping": actuation["damping"],
        "armature": actuation["armature"],
        "effort_limit": actuation["effort_limit"],
        "last_action_is_raw": True,
        "raw_action_clip": 20.0,
        "joint_limits_lower": lower,
        "joint_limits_upper": upper,
    }


def command_contract(*, encoder_sha256: str) -> dict:
    return {
        "z_dim": 64,
        "phase_mode": "none",
        "phase_dim": 0,
        "hold_steps": 10,
        "state_dim": 64,
        "encoder_state_interface": "joint_qpos_qvel_anchor_ori",
        "window_steps": 9,
        "horizon_steps": 10,
        "encoder_window_mode": "intermediate",
        "macro_frame_stride": 5,
        "macro_anchor_mode": "robot_heading",
        "encoder_trigger": "every_control_tick",
        "activation": "silu",
        "layer_norm": False,
        "encoder_sha256": encoder_sha256,
        "quantizer": "none",
        "components": [],
    }


def _onnx_replay(
    path: Path, values: np.ndarray, *, input_name: str, output_name: str
) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([output_name], {input_name: values})[0]


def export_bundle(
    checkpoint: str | Path,
    output: str | Path,
    *,
    version: str = "auto",
    parity_atol: float = 1.0e-5,
) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {output_path}")
    output_path.mkdir(parents=True)

    actor = load_sonic_release_actor(checkpoint_path, version=version).eval()
    encoder = _EncoderExport(actor).eval()
    decoder = _DecoderExport(actor).eval()
    generator = torch.Generator().manual_seed(0)
    encoder_input = torch.randn(1, 640, generator=generator)
    decoder_input = torch.randn(1, 994, generator=generator)
    with torch.inference_mode():
        encoder_output = encoder(encoder_input)
        decoder_output = decoder(decoder_input)

    torch.onnx.export(
        encoder,
        encoder_input,
        output_path / "encoder.onnx",
        input_names=["reference_window"],
        output_names=["token"],
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    torch.onnx.export(
        decoder,
        decoder_input,
        output_path / "policy.onnx",
        input_names=["observation"],
        output_names=["action"],
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    torch.jit.trace(encoder, encoder_input).save(str(output_path / "encoder.pt"))
    torch.jit.trace(decoder, decoder_input).save(str(output_path / "policy.pt"))

    encoder_replay = _onnx_replay(
        output_path / "encoder.onnx",
        encoder_input.numpy(),
        input_name="reference_window",
        output_name="token",
    )
    decoder_replay = _onnx_replay(
        output_path / "policy.onnx",
        decoder_input.numpy(),
        input_name="observation",
        output_name="action",
    )
    encoder_error = float(np.max(np.abs(encoder_replay - encoder_output.numpy())))
    decoder_error = float(np.max(np.abs(decoder_replay - decoder_output.numpy())))
    if encoder_error > parity_atol or decoder_error > parity_atol:
        raise ValueError(
            "ONNX parity failed: "
            f"encoder={encoder_error}, decoder={decoder_error}, "
            f"tolerance={parity_atol}"
        )

    obs = observation_contract()
    action = action_contract()
    (output_path / "obs_contract.json").write_text(json.dumps(obs, indent=2) + "\n")
    (output_path / "action_contract.json").write_text(
        json.dumps(action, indent=2) + "\n"
    )
    np.savez(
        output_path / "golden_trace.npz",
        obs=decoder_input.numpy().astype(np.float32),
        action=decoder_output.numpy().astype(np.float32),
        encoder_in=encoder_input.numpy().astype(np.float32),
        encoder_out=encoder_output.numpy().astype(np.float32),
    )

    file_names = [
        "policy.pt",
        "policy.onnx",
        "encoder.pt",
        "encoder.onnx",
        "obs_contract.json",
        "action_contract.json",
        "golden_trace.npz",
    ]
    files = {name: _sha256(output_path / name) for name in file_names}
    import onnx
    import onnxruntime

    manifest = {
        "api_version": BUNDLE_API_VERSION,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "primary_policy_role": "student",
            "adapter": "sonic_release_actor",
            "sonic_version": actor.spec.version,
            "orientation_contract": actor.spec.orientation_contract,
            "repo_commit": _repo_commit(),
            "tool_versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__,
            },
        },
        "interface": "latent",
        "obs": obs,
        "action": action,
        "command": command_contract(encoder_sha256=files["encoder.pt"]),
        "rates": {"control_hz": 50, "physics_dt": 0.005, "decimation": 4},
        "models": {
            "policy_onnx": {
                "format": "onnx",
                "path": "policy.onnx",
                "input_name": "observation",
                "output_name": "action",
                "input_shape": [1, 994],
                "output_shape": [1, 29],
                "opset": ONNX_OPSET,
                "parity_atol": parity_atol,
                "max_abs_error": decoder_error,
            },
            "encoder_onnx": {
                "format": "onnx",
                "path": "encoder.onnx",
                "input_name": "reference_window",
                "output_name": "token",
                "input_shape": [1, 640],
                "output_shape": [1, 64],
                "opset": ONNX_OPSET,
                "parity_atol": parity_atol,
                "max_abs_error": encoder_error,
            },
        },
        "files": files,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--version", choices=("release", "v1_1", "auto"), default="auto"
    )
    parser.add_argument("--parity-atol", type=float, default=1.0e-5)
    args = parser.parse_args()
    output = export_bundle(
        args.checkpoint,
        args.output,
        version=args.version,
        parity_atol=args.parity_atol,
    )
    print(f"BUNDLE: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
