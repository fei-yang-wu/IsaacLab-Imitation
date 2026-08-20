"""Simulator-free tests for the Newton contact adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_imitation.tasks.manager_based.dexmanip.newton_contacts import (
    _aggregate_contact_pairs_kernel,
    _body_ids_from_names,
    _finalize_contact_pairs_kernel,
    _rows_from_environment_major_ids,
    build_contact_body_maps,
)


def test_body_maps_keep_environment_object_and_link_order() -> None:
    body_to_env, hand_to_link, object_to_index = build_contact_body_maps(
        8,
        link_body_ids=[[1, 2], [5, 6]],
        object_body_ids=[[3], [7]],
    )

    assert body_to_env == [-1, 0, 0, 0, -1, 1, 1, 1]
    assert hand_to_link == [-1, 0, 1, -1, -1, 0, 1, -1]
    assert object_to_index == [-1, -1, -1, 0, -1, -1, -1, 0]


@pytest.mark.parametrize(
    ("links", "objects", "match"),
    [
        ([[1]], [[1]], "occurs more than once"),
        ([[9]], [[1]], "outside the model range"),
        ([[1], [2]], [[3]], "same row count"),
    ],
)
def test_body_maps_reject_invalid_ids(
    links: list[list[int]], objects: list[list[int]], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_contact_body_maps(4, links, objects)


def test_object_scene_names_match_newton_body_path_segments() -> None:
    labels = (
        "/World/envs/env_0/smoke_cube/base",
        "/World/envs/env_1/smoke_cube/base",
    )

    resolved = _body_ids_from_names(
        labels,
        worlds=(0, 1),
        names=("smoke_cube",),
        num_envs=2,
        allowed_body_ids={0, 1},
        kind="object",
    )

    assert resolved == [[0], [1]]


def test_environment_major_sensor_ids_are_split_into_rows() -> None:
    rows = _rows_from_environment_major_ids(
        body_ids=(10, 11, 20, 21),
        num_envs=2,
        row_width=2,
        name="test sensor",
    )

    assert rows == [[10, 11], [20, 21]]


def _run_warp_aggregation(
    *,
    shape0: list[int],
    shape1: list[int],
    point0: list[tuple[float, float, float]],
    point1: list[tuple[float, float, float]],
    force0: list[tuple[float, float, float]],
    body_transforms: list[tuple[float, float, float, float, float, float, float]],
    valid_envs: list[bool] | None = None,
) -> SimpleNamespace:
    wp.init()
    device = "cpu"
    contact_count = len(shape0)
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([contact_count], dtype=wp.int32, device=device),
        rigid_contact_shape0=wp.array(shape0, dtype=wp.int32, device=device),
        rigid_contact_shape1=wp.array(shape1, dtype=wp.int32, device=device),
        rigid_contact_point0=wp.array(point0, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.array(point1, dtype=wp.vec3, device=device),
        force=wp.array(
            [(*force, 0.0, 0.0, 0.0) for force in force0],
            dtype=wp.spatial_vector,
            device=device,
        ),
        rigid_contact_max=contact_count,
    )
    shape_body = wp.array([0, 1], dtype=wp.int32, device=device)
    body_q = wp.array(body_transforms, dtype=wp.transform, device=device)
    body_to_env = wp.array([0, 0], dtype=wp.int32, device=device)
    hand_to_link = wp.array([0, -1], dtype=wp.int32, device=device)
    object_to_index = wp.array([-1, 0], dtype=wp.int32, device=device)
    valid = wp.array(valid_envs or [True], dtype=wp.bool, device=device)
    positions_sum = wp.zeros(1, dtype=wp.vec3, device=device)
    positions = wp.zeros(1, dtype=wp.vec3, device=device)
    forces = wp.zeros(1, dtype=wp.vec3, device=device)
    counts = wp.zeros(1, dtype=wp.int32, device=device)
    active = wp.zeros(1, dtype=wp.bool, device=device)

    wp.launch(
        _aggregate_contact_pairs_kernel,
        dim=contact_count,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.force,
            shape_body,
            body_q,
            body_to_env,
            hand_to_link,
            object_to_index,
            valid,
            1,
            1,
        ],
        outputs=[positions_sum, forces, counts],
        device=device,
    )
    wp.launch(
        _finalize_contact_pairs_kernel,
        dim=1,
        inputs=[positions_sum, forces, counts, 0.1],
        outputs=[positions, active],
        device=device,
    )
    return SimpleNamespace(
        positions=wp.to_torch(positions),
        forces=wp.to_torch(forces),
        counts=wp.to_torch(counts),
        active=wp.to_torch(active),
    )


def test_warp_adapter_uses_object_side_point_and_force_on_object() -> None:
    result = _run_warp_aggregation(
        shape0=[0],
        shape1=[1],
        point0=[(9.0, 9.0, 9.0)],
        point1=[(1.0, 2.0, 3.0)],
        force0=[(2.0, -3.0, 4.0)],
        body_transforms=[
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    torch.testing.assert_close(result.positions, torch.tensor([[11.0, 22.0, 33.0]]))
    torch.testing.assert_close(result.forces, torch.tensor([[-2.0, 3.0, -4.0]]))
    assert result.counts.tolist() == [1]
    assert result.active.tolist() == [True]


def test_warp_adapter_handles_reverse_pair_order() -> None:
    result = _run_warp_aggregation(
        shape0=[1],
        shape1=[0],
        point0=[(1.0, 2.0, 3.0)],
        point1=[(9.0, 9.0, 9.0)],
        force0=[(2.0, -3.0, 4.0)],
        body_transforms=[
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    torch.testing.assert_close(result.positions, torch.tensor([[11.0, 22.0, 33.0]]))
    torch.testing.assert_close(result.forces, torch.tensor([[2.0, -3.0, 4.0]]))


def test_warp_adapter_averages_points_and_sums_forces() -> None:
    result = _run_warp_aggregation(
        shape0=[0, 0],
        shape1=[1, 1],
        point0=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        point1=[(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)],
        force0=[(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
        body_transforms=[
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
    )

    torch.testing.assert_close(result.positions, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(result.forces, torch.tensor([[-1.0, -2.0, 0.0]]))
    assert result.counts.tolist() == [2]


def test_warp_adapter_masks_an_environment_after_reset() -> None:
    result = _run_warp_aggregation(
        shape0=[0],
        shape1=[1],
        point0=[(0.0, 0.0, 0.0)],
        point1=[(1.0, 2.0, 3.0)],
        force0=[(2.0, 0.0, 0.0)],
        body_transforms=[
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ],
        valid_envs=[False],
    )

    torch.testing.assert_close(result.positions, torch.zeros(1, 3))
    torch.testing.assert_close(result.forces, torch.zeros(1, 3))
    assert result.counts.tolist() == [0]
    assert result.active.tolist() == [False]
