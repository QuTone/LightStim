"""Cache concurrency and backend-independent binary input behavior."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time

import numpy as np
import pytest

from lightstim.utils import pauli_backend


pytestmark = pytest.mark.smoke


def test_concurrent_basis_misses_do_not_duplicate_cache_accounting(monkeypatch):
    class FakeBasis:
        constructions = 0
        commuting = True
        estimated_bytes = 32

        def __init__(self, rows):
            type(self).constructions += 1
            # Expose a cache miss to the other workers. The cache lock must
            # cover construction, insertion, and memory accounting together.
            time.sleep(0.01)

    monkeypatch.setattr(pauli_backend, "_CommutingBasis", FakeBasis)
    monkeypatch.setattr(pauli_backend, "_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(pauli_backend, "_CACHE_MAX_BYTES", 4096)
    pauli_backend._clear_basis_cache()
    barrier = Barrier(8)
    matrix = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)

    def get_shared_basis(_):
        barrier.wait(timeout=5)
        return pauli_backend._get_basis(matrix)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            instances = list(pool.map(get_shared_basis, range(8)))
        assert all(instance is instances[0] for instance in instances)
        assert FakeBasis.constructions == 1
        assert len(pauli_backend._basis_cache) == 1
        assert pauli_backend._basis_cache_bytes == sum(
            size for _, size in pauli_backend._basis_cache.values()
        )
        assert pauli_backend._basis_cache_bytes <= pauli_backend._CACHE_MAX_BYTES
    finally:
        pauli_backend._clear_basis_cache()


@pytest.mark.parametrize("mode", ["reference", "auto", "native"])
def test_boolean_pauli_propagation_uses_xor_arithmetic(monkeypatch, mode):
    if mode == "native" and not pauli_backend._NATIVE_AVAILABLE:
        pytest.skip("optional native Pauli backend has not been built")
    monkeypatch.setenv("LIGHTSTIM_PAULI_BACKEND", mode)
    # A controlled-Z maps X0 -> X0 Z1 and X1 -> Z0 X1. The Y1 row
    # multiplied with X0 has two Z1 contributions, which must cancel.
    forward = np.asarray([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=bool)
    rows = np.asarray([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=bool)
    expected = rows.astype(np.uint8) @ forward.astype(np.uint8) % 2
    actual = pauli_backend.propagate_paulis(rows, forward)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
