"""Exporter tests over a synthetic L2T-shaped checkpoint (no Isaac needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from imitation_experiments.lowlevel import export_policy_bundle as epb


def _policy_state(
    width=351,
    hidden=(64, 32),
    action_dim=29,
    mask_false_span=258,
    seed=0,
    with_mask=True,
):
    generator = torch.Generator().manual_seed(seed)
    state = {}
    prefix = "module.0.module."
    state[f"{prefix}base.running_mean"] = 0.1 * torch.randn(width, generator=generator)
    state[f"{prefix}base.running_var"] = torch.rand(width, generator=generator) + 0.5
    state[f"{prefix}base.count"] = torch.tensor(1.0e6)
    if with_mask:
        mask = torch.ones(width, dtype=torch.bool)
        mask[:mask_false_span] = False
        state[f"{prefix}base.normalize_mask"] = mask
    dims = [width, *hidden, action_dim]
    for layer, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
        index = 2 * layer
        state[f"{prefix}base.module.{index}.weight"] = 0.05 * torch.randn(
            n_out, n_in, generator=generator
        )
        state[f"{prefix}base.module.{index}.bias"] = torch.zeros(n_out)
    state[f"{prefix}log_std_module.log_std"] = torch.zeros(action_dim)
    return state


def _encoder_state(state_dim=38, hidden=(48, 24), z_dim=256, seed=1):
    generator = torch.Generator().manual_seed(seed)
    state = {}
    dims = [state_dim * 10, *hidden, z_dim]
    index = 0
    for layer, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
        state[f"net.{index}.weight"] = 0.05 * torch.randn(
            n_out, n_in, generator=generator
        )
        state[f"net.{index}.bias"] = torch.zeros(n_out)
        index += 1
        if layer < len(dims) - 2:
            state[f"net.{index}.weight"] = torch.ones(n_out)
            state[f"net.{index}.bias"] = torch.zeros(n_out)
            index += 2  # LayerNorm slot + activation gap
    return state


def _checkpoint(tmp_path, name="model.pt", **kwargs):
    payload = {
        "policy_state_dict": _policy_state(**kwargs),
        "hl_skill_command_sampler_state_dict": {
            "skill_encoder_state_dict": _encoder_state(),
            "finetune_updates": 0,
        },
        "checkpoint_metadata": {
            "algorithm": "IPMD_L2T",
            "primary_policy_role": "student",
        },
        "teacher_policy_state_dict": _policy_state(width=286, with_mask=False, seed=3),
    }
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _export(tmp_path, checkpoint, *extra):
    output = tmp_path / "bundle"
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--preset",
        "l2t_student_v2",
        "--output",
        str(output),
        "--macro-frame-stride",
        "1",
        "--macro-anchor-mode",
        "expert_heading",
        "--encoder-activation",
        "mish",
        "--encoder-layer-norm",
        *extra,
    ]
    assert epb.main(argv) == 0
    return output


def test_export_and_verify_round_trip(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output = _export(tmp_path, checkpoint, "--verify")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["api_version"] == "ec.bundle/v1"
    assert set(manifest["source"]["tool_versions"]) == {
        "torch",
        "numpy",
        "onnx",
        "onnxruntime",
    }
    assert manifest["source"]["export_command"].endswith("--verify")
    assert manifest["interface"] == "latent"
    assert manifest["obs"]["total_width"] == 351
    assert [t["name"] for t in manifest["obs"]["terms"]][0] == "latent_command"
    assert manifest["obs"]["terms"][0]["normalize"] is False
    assert manifest["command"]["macro_frame_stride"] == 1
    assert manifest["command"]["encoder_state_interface"] == "root_qpos"
    assert manifest["command"]["macro_anchor_mode"] == "expert_heading"
    assert manifest["command"]["window_steps"] == 9
    assert manifest["action"]["isaac_joint_names"][0] == "left_hip_pitch_joint"
    assert len(manifest["action"]["joint_limits_lower"]) == 29
    assert len(manifest["action"]["joint_limits_upper"]) == 29
    assert manifest["action"]["joint_limits_lower"][0] == pytest.approx(-2.260175)
    assert manifest["action"]["joint_limits_upper"][0] == pytest.approx(2.609275)
    assert len(manifest["files"]) == 8
    assert manifest["models"]["policy_onnx"]["input_shape"] == [1, 351]
    assert manifest["models"]["encoder_onnx"]["output_shape"] == [1, 256]
    report = epb.verify_bundle_dir(output)
    assert report["policy_max_abs_err"] <= 1e-6
    assert report["encoder_max_abs_err"] <= 1e-6
    assert report["policy_onnx_max_abs_err"] <= 2e-5
    assert report["encoder_onnx_max_abs_err"] <= 2e-5


def test_normalize_mask_span_check(tmp_path):
    checkpoint = _checkpoint(tmp_path, mask_false_span=200)
    with pytest.raises(ValueError, match="normalize mask disagrees"):
        _export(tmp_path, checkpoint)


def test_width_mismatch_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path, width=160)
    with pytest.raises(ValueError, match="Wrong preset"):
        _export(tmp_path, checkpoint)


def test_missing_provenance_rejected(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="macro_frame_stride"):
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "l2t_student_v2",
                "--output",
                str(output),
            ]
        )


def test_teacher_refused_without_flag(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "l2t_student_v2",
                "--output",
                str(tmp_path / "b"),
                "--role",
                "teacher",
            ]
        )


def test_teacher_export_with_flag_is_stamped(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "teacher_bundle"
    assert (
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "l2t_student_v2",
                "--output",
                str(output),
                "--role",
                "teacher",
                "--allow-privileged",
                "--skip-rlopt-parity",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["interface"] == "privileged-teacher"


def test_skill_binding_gate(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    matching = tmp_path / "skill.pt"
    torch.save(
        {
            "skill_encoder_state_dict": _encoder_state(),
            "config": {
                "macro_frame_stride": 1,
                "macro_anchor_mode": "expert_heading",
                "horizon_steps": 10,
                "encoder_window_mode": "intermediate",
                "encoder_activation": "mish",
                "encoder_layer_norm": True,
            },
        },
        matching,
    )
    output = _export(tmp_path, checkpoint, "--skill-checkpoint", str(matching))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["command"]["macro_anchor_mode"] == "expert_heading"
    assert manifest["source"]["skill_checkpoint_path"] == str(matching.resolve())
    assert manifest["source"]["skill_checkpoint_sha256"] == epb.sha256_file(matching)

    mismatching = tmp_path / "skill_bad.pt"
    torch.save(
        {
            "skill_encoder_state_dict": _encoder_state(seed=9),
            "config": {
                "macro_frame_stride": 1,
                "macro_anchor_mode": "expert_heading",
                "horizon_steps": 10,
                "encoder_window_mode": "intermediate",
                "encoder_activation": "mish",
                "encoder_layer_norm": True,
            },
        },
        mismatching,
    )
    with pytest.raises(ValueError, match="encoder tensor mismatch"):
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "l2t_student_v2",
                "--output",
                str(tmp_path / "b2"),
                "--skill-checkpoint",
                str(mismatching),
            ]
        )


def test_golden_trace_tamper_detected(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output = _export(tmp_path, checkpoint)
    trace = dict(np.load(output / "golden_trace.npz"))
    trace["action"] = trace["action"] + 0.01
    np.savez(output / "golden_trace.npz", **trace)
    with pytest.raises(ValueError, match="hash mismatch"):
        epb.verify_bundle_dir(output)


def test_export_refuses_existing_output_tree(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh path"):
        _export(tmp_path, checkpoint)


def _fsq_encoder_state(state_dim=38, hidden=(48, 24), z_dim=64, seed=2, levels=32):
    generator = torch.Generator().manual_seed(seed)
    state = {"_half_levels": torch.full((z_dim,), float(levels // 2))}
    dims = [state_dim * 10, *hidden, z_dim]
    index = 0
    for layer, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
        state[f"net.{index}.weight"] = 0.05 * torch.randn(
            n_out, n_in, generator=generator
        )
        state[f"net.{index}.bias"] = torch.zeros(n_out)
        index += 2  # plain [Linear, activation] pairs, no LayerNorm
    return state


def _fsq_checkpoint_and_skill(tmp_path, z_dim=64):
    encoder_state = _fsq_encoder_state(z_dim=z_dim)
    payload = {
        "policy_state_dict": _policy_state(width=159, mask_false_span=66),
        "hl_skill_command_sampler_state_dict": {
            "skill_encoder_state_dict": encoder_state,
            "finetune_updates": 0,
        },
    }
    checkpoint = tmp_path / "fsq_model.pt"
    torch.save(payload, checkpoint)
    skill = tmp_path / "fsq_skill.pt"
    torch.save(
        {
            "skill_encoder_state_dict": encoder_state,
            "config": {
                "macro_frame_stride": 1,
                "macro_anchor_mode": "robot",
                "horizon_steps": 10,
                "encoder_window_mode": "intermediate",
                "encoder_activation": "silu",
                "encoder_layer_norm": False,
                "sonic_fsq_levels": [32] * z_dim,
                "latent_mode": "sonic_fsq",
            },
        },
        skill,
    )
    return checkpoint, skill


def test_fsq_export_lattice_and_manifest(tmp_path):
    checkpoint, skill = _fsq_checkpoint_and_skill(tmp_path)
    output = tmp_path / "fsq_bundle"
    assert (
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "fsq64_v2",
                "--output",
                str(output),
                "--skill-checkpoint",
                str(skill),
                "--verify",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["obs"]["total_width"] == 159
    assert manifest["obs"]["terms"][0] == {
        "name": "latent_command",
        "width": 66,
        "normalize": False,
    }
    assert manifest["command"]["quantizer"] == "fsq"
    assert manifest["command"]["z_dim"] == 64
    assert manifest["command"]["hold_steps"] == 10
    assert manifest["command"]["fsq_half_levels"] == [16.0] * 64
    assert manifest["command"]["activation"] == "silu"
    assert manifest["command"]["layer_norm"] is False
    trace = np.load(output / "golden_trace.npz")
    z = trace["encoder_out"]
    lattice = np.round(z * 16.0) / 16.0
    np.testing.assert_array_equal(z, lattice)
    assert z.min() >= -1.0 and z.max() <= 15.0 / 16.0


def test_fsq_export_requires_skill_checkpoint(tmp_path):
    checkpoint, _skill = _fsq_checkpoint_and_skill(tmp_path)
    with pytest.raises(ValueError, match="FSQ presets require --skill-checkpoint"):
        epb.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--preset",
                "fsq64_v2",
                "--output",
                str(tmp_path / "b"),
                "--macro-frame-stride",
                "1",
                "--macro-anchor-mode",
                "robot",
                "--encoder-activation",
                "silu",
            ]
        )


def test_actuation_table_matches_known_values():
    actuation = epb._per_joint_actuation()
    names = epb.G1_ISAAC_JOINT_NAMES
    by_name = dict(zip(names, actuation["action_scale"], strict=True))
    assert by_name["left_hip_pitch_joint"] == pytest.approx(0.35066147, abs=1e-6)
    assert by_name["waist_yaw_joint"] == pytest.approx(0.54754647, abs=1e-6)
    assert by_name["left_ankle_pitch_joint"] == pytest.approx(0.43857731, abs=1e-6)
    assert by_name["left_wrist_pitch_joint"] == pytest.approx(0.07450087, abs=1e-6)
    defaults_by_name = dict(zip(names, actuation["default_joint_pos"], strict=True))
    assert defaults_by_name["left_knee_joint"] == pytest.approx(0.669)
    assert defaults_by_name["right_shoulder_roll_joint"] == pytest.approx(-0.2)
    assert defaults_by_name["left_shoulder_roll_joint"] == pytest.approx(0.2)
    assert min(actuation["stiffness"]) > 0 and min(actuation["damping"]) > 0
    effort_by_name = dict(zip(names, actuation["effort_limit"], strict=True))
    assert effort_by_name["left_hip_pitch_joint"] == pytest.approx(139.0)
    assert effort_by_name["left_wrist_yaw_joint"] == pytest.approx(5.0)
    armature_by_name = dict(zip(names, actuation["armature"], strict=True))
    assert armature_by_name["left_ankle_pitch_joint"] == pytest.approx(2 * 0.003609725)
    assert armature_by_name["left_hip_pitch_joint"] == pytest.approx(0.025101925)


def test_sdk_permutation_round_trip():
    sdk = epb._sdk_joint_names()
    assert len(sdk) == 29
    isaac_to_sdk = [sdk.index(name) for name in epb.G1_ISAAC_JOINT_NAMES]
    assert sorted(isaac_to_sdk) == list(range(29))
    assert (
        sdk[isaac_to_sdk[epb.G1_ISAAC_JOINT_NAMES.index("waist_yaw_joint")]]
        == "waist_yaw_joint"
    )
