from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = (
    Path(__file__).parent
    / "isaaclab_imitation"
    / "assets"
    / "robots"
    / "unitree_joint_order.py"
)


def _load_joint_order_module():
    spec = importlib.util.spec_from_file_location(
        "unitree_joint_order_test", _MODULE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unitree_joint_order = _load_joint_order_module()


def test_unitree_g1_joint_order_sources_match() -> None:
    assert (
        unitree_joint_order.UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES
        == unitree_joint_order.UNITREE_G1_29DOF_URDF_REVOLUTE_JOINT_NAMES
    )
    assert (
        unitree_joint_order.UNITREE_G1_29DOF_XML_MOTOR_JOINT_NAMES
        == unitree_joint_order.UNITREE_LEROBOT_G1_29DOF_JOINT_NAMES
    )
    assert (
        unitree_joint_order.UNITREE_G1_29DOF_SDK_JOINT_NAMES
        == unitree_joint_order.UNITREE_LEROBOT_G1_29DOF_JOINT_NAMES
    )
    assert (
        unitree_joint_order.UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES
        == unitree_joint_order.UNITREE_G1_29DOF_SDK_JOINT_NAMES
    )
    assert len(unitree_joint_order.UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES) == 29
    assert (
        unitree_joint_order.UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES[0]
        == "left_hip_pitch_joint"
    )
    assert (
        unitree_joint_order.UNITREE_G1_WBT_29DOF_DATASET_JOINT_NAMES[-1]
        == "right_wrist_yaw_joint"
    )


def test_unitree_g1_joint_order_assets_exist() -> None:
    assert unitree_joint_order.UNITREE_G1_29DOF_XML_FILE.is_file()
    assert unitree_joint_order.UNITREE_G1_29DOF_URDF_FILE.is_file()
