"""Algebraic regression tests for the explicit toric logical operators."""

import numpy as np
import pytest

from lightstim.qec_code.surface_code.toric import ToricCode


def _support(record):
    return set(record["data_indices"])


def _opposite_type_overlap(left, right):
    if left["type"] == right["type"]:
        return 0
    return len(_support(left) & _support(right)) % 2


@pytest.mark.parametrize(
    "params",
    [
        {"distance": 2},
        {"distance": 3},
        {"l_z": 4, "l_x": 3},
        {"l_z": 3, "l_x": 2, "shift": (7, -2)},
    ],
)
def test_explicit_logical_operators_form_two_valid_pairs(params):
    code = ToricCode(**params)
    logical_z = [op for op in code.logical_ops if op["type"] == "Z"]
    logical_x = [op for op in code.logical_ops if op["type"] == "X"]

    assert len(logical_z) == len(logical_x) == code.num_logicals == 2

    # Every logical representative must lie in the stabilizer normalizer.
    for logical in code.logical_ops:
        assert set(logical["pauli"]) == _support(logical)
        assert set(logical["pauli"].values()) == {logical["type"]}
        assert _support(logical) <= code.data_indices
        assert all(
            _opposite_type_overlap(logical, stabilizer) == 0
            for stabilizer in code.stabilizers
        )

    # The declared order is (Z_0, X_0, Z_1, X_1), so the symplectic
    # pairing between the extracted X and Z lists must be the identity.
    pairing = np.array(
        [
            [len(_support(x_op) & _support(z_op)) % 2 for z_op in logical_z]
            for x_op in logical_x
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(pairing, np.eye(code.num_logicals, dtype=np.uint8))

    assert [len(_support(op)) for op in logical_z] == [code.l_z, code.l_x]
    assert [len(_support(op)) for op in logical_x] == [code.l_x, code.l_z]


@pytest.mark.parametrize(
    ("params", "expected_supports"),
    [
        (
            {"distance": 2},
            [
                {(1, 1), (3, 1)},
                {(1, 1), (1, 3)},
                {(0, 0), (0, 2)},
                {(0, 0), (2, 0)},
            ],
        ),
        (
            {"distance": 3},
            [
                {(1, 1), (3, 1), (5, 1)},
                {(1, 1), (1, 3), (1, 5)},
                {(0, 0), (0, 2), (0, 4)},
                {(0, 0), (2, 0), (4, 0)},
            ],
        ),
        (
            {"l_z": 2, "l_x": 3, "shift": (7, -2)},
            [
                {(8, -1), (10, -1)},
                {(8, -1), (8, 1), (8, 3)},
                {(7, -2), (7, 0), (7, 2)},
                {(7, -2), (9, -2)},
            ],
        ),
    ],
)
def test_logical_support_coordinates(params, expected_supports):
    code = ToricCode(**params)
    supports = [
        {code.qubit_coords[index] for index in op["data_indices"]}
        for op in code.logical_ops
    ]

    assert supports == expected_supports
