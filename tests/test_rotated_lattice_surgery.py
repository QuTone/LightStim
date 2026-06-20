"""
Rotated surface code lattice surgery tests.

Stage 1 — RotatedTwoPatchCoupler (constructive merged-code diff):
  The coupler's job is purely geometric. Its correctness invariant is:

      (A's surviving stabilizers) ∪ (B's surviving stabilizers) ∪ (coupler stabilizers)
        ==  stabilizers of a single RotatedSurfaceCode of the merged size

  i.e. merging two rotated patches reproduces exactly the larger rotated code.

Stage 1b — circuit-level: the rotated two-patch LS circuit must be noiseless-clean
  (zero detection events, zero logical errors) and build a valid DEM.
"""
import io
import contextlib
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import assert_valid_circuit, assert_noiseless, assert_dem_valid, build_quiet

from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode


def _code_sigs(patch, drop_syn_coords=frozenset()):
    """Canonical {(type, frozenset(data_coords))} for an index-keyed code patch,
    dropping stabilizers whose syndrome coordinate is in drop_syn_coords."""
    sigs = set()
    for s in patch.stabilizers:
        if s.get("syn_coord") in drop_syn_coords:
            continue
        coords = frozenset(patch.qubit_coords[i] for i in s["pauli"])
        sigs.add((s["type"], coords))
    return sigs


def _coupler_sigs(coupler_patch):
    """Canonical {(type, frozenset(data_coords))} for a coordinate-keyed coupler patch."""
    return {(s["type"], frozenset(s["pauli"].keys())) for s in coupler_patch.stabilizers}


class TestRotatedTwoPatchCouplerGeometry:
    """Stage 1: pure-geometry invariant of the constructive coupler."""

    @pytest.mark.parametrize("d", [3, 5])
    def test_xx_merge_reproduces_merged_code(self, d):
        from lightstim.qec_code.surface_code.rotated import RotatedTwoPatchCoupler

        # XX merge = vertical stack; B sits one intermediate data row below A.
        a = RotatedSurfaceCode(distance=d)
        b = RotatedSurfaceCode(distance=d)
        b.shift_coords(0, 2 * d + 2)

        coupler = RotatedTwoPatchCoupler().create_coupler_patch([a, b], name="c", interaction_type="XX")

        merged = RotatedSurfaceCode(distance_z=d, distance_x=2 * d + 1)

        conflicts = coupler.conflicting_stabilizer_coords
        got = (
            _code_sigs(a, conflicts)
            | _code_sigs(b, conflicts)
            | _coupler_sigs(coupler)
        )
        assert got == _code_sigs(merged), "merged stabilizers != RotatedSurfaceCode(merged)"

    @pytest.mark.parametrize("d", [3, 5])
    def test_zz_merge_reproduces_merged_code(self, d):
        from lightstim.qec_code.surface_code.rotated import RotatedTwoPatchCoupler

        # ZZ merge = side-by-side; B sits one intermediate data column right of A.
        a = RotatedSurfaceCode(distance=d)
        b = RotatedSurfaceCode(distance=d)
        b.shift_coords(2 * d + 2, 0)

        coupler = RotatedTwoPatchCoupler().create_coupler_patch([a, b], name="c", interaction_type="ZZ")

        merged = RotatedSurfaceCode(distance_z=2 * d + 1, distance_x=d)

        conflicts = coupler.conflicting_stabilizer_coords
        got = (
            _code_sigs(a, conflicts)
            | _code_sigs(b, conflicts)
            | _coupler_sigs(coupler)
        )
        assert got == _code_sigs(merged), "merged stabilizers != RotatedSurfaceCode(merged)"

    def test_xx_intermediate_data_row(self):
        from lightstim.qec_code.surface_code.rotated import RotatedTwoPatchCoupler

        a = RotatedSurfaceCode(distance=3)
        b = RotatedSurfaceCode(distance=3)
        b.shift_coords(0, 8)
        coupler = RotatedTwoPatchCoupler().create_coupler_patch([a, b], name="c", interaction_type="XX")

        assert set(coupler.data_coords) == {(1.0, 7.0), (3.0, 7.0), (5.0, 7.0)}


class TestRotatedTwoPatchLSCircuit:
    """Stage 1b: end-to-end — the rotated two-patch LS circuit must be noiseless-clean."""

    def _build(self, interaction_type, offset, i1, i2):
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        from lightstim.qec_code.surface_code.rotated import (
            RotatedSurfaceCode as RSC,
            RotatedSurfaceCodeExtractionBlock,
            RotatedTwoPatchCoupler,
        )
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": 3}, patch2_config={"distance": 3},
            offset=offset, interaction_type=interaction_type,
            coupler_protocol=RotatedTwoPatchCoupler(),
            code_patch_class=RSC,
            extraction_block_class=RotatedSurfaceCodeExtractionBlock,
            initial_state_patch1=i1, initial_state_patch2=i2,
            measure_state_patch1=i1, measure_state_patch2=i2,
            rounds=2, noise_params=None, rotate_patch1=False,
        )
        return build_quiet(exp.build)

    @pytest.mark.smoke
    def test_xx_merge_noiseless(self):
        # X̄₁X̄₂ merge: patches in complementary bases (matches working unrotated convention)
        c = self._build("XX", (0, 8), "Z", "X")
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)

    @pytest.mark.smoke
    def test_zz_merge_noiseless(self):
        c = self._build("ZZ", (8, 0), "X", "Z")
        assert_valid_circuit(c); assert_noiseless(c); assert_dem_valid(c)
