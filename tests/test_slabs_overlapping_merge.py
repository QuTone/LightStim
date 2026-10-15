"""Retain both native PPM couplers until one combined final readout."""
from contextlib import redirect_stdout
from io import StringIO

import pytest

from lightstim.protocols.rotated_surface_ppm import RotatedSurfacePPMStep
from lightstim.qec_code.surface_code.rotated.ppm import RotatedSurfacePatchPlacement, origin_of
from lightstim.qec_code.surface_code.rotated.ppm.lowering import joint_pauli_vector, record_parity
from tests.test_slabs_three_patch_matrix import _BoundaryExperiment, validate_case


class HeldMergeExperiment(_BoundaryExperiment):
    """Use native primitives, omitting only intermediate split/readout calls."""

    def _setup(self):
        super()._setup()
        original_readout = self.builder.apply_data_readout

        def combined_readout(final_measurements, **kwargs):
            collection = {}
            for index, plan in self._plans.items():
                name = f"ppm_{index}"
                patch = self.system.coupler_patches[name]
                local = self.system.local_to_global_map[name]
                collection.update({local[q]: plan.corridor_init_basis for q in patch.data_indices})
            collection.update(final_measurements)
            return original_readout(final_measurements=collection, **kwargs)

        self.builder.apply_data_readout = combined_readout

    def _apply_ppm_step(self, index, step):
        name = f"ppm_{index}"
        pre = record_parity(self.tracker, joint_pauli_vector(self.system, step.targets))
        self.builder.activate_coupler(name)
        patch = self.system.coupler_patches[name]
        local = self.system.local_to_global_map[name]
        collection = {local[q]: self._plans[index].corridor_init_basis for q in patch.data_indices}
        self.builder.initialize(init_dict=collection, n=self.system.num_qubits)
        self.builder.apply_syndrome_extraction(self._merged_chunk(index), rounds=self.rounds)
        post = record_parity(self.tracker, joint_pauli_vector(self.system, step.targets))
        self._capture_outcome(index, step, pre, post=post)
        if index == len(self.ppm_sequence) - 1:
            # No individual patch X probe midway: it would violate the
            # merged code while a later SE still measures those checks.
            self._capture_boundary()


def physical_case(sequence, initials, finals, rounds):
    experiment = HeldMergeExperiment(
        [RotatedSurfacePatchPlacement(name, origin_of(x, 0, 3, seam=True), 3,
                                     "X_horizontal")
         for name, x in (("A", 0), ("B", 2), ("C", 4))],
        [RotatedSurfacePPMStep([(name, "Z") for name in pair],
                               route=[(1 if pair == "AB" else 3, 0)])
         for pair in sequence],
        initial_states=dict(zip("ABC", initials)),
        final_measure_states=dict(zip("ABC", finals)),
        rounds=rounds, rounds_init=1, logical_tracking="slabs",
    )
    with redirect_stdout(StringIO()):
        circuit = experiment.build()
    return experiment, circuit


@pytest.mark.parametrize("sequence", [("AB", "BC"), ("BC", "AB")])
@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize("initials,finals", [
    ("ZZZ", "ZZZ"), ("XXX", "ZZZ"), ("XXX", "XXX"),
    ("ZXZ", "XZX"),
])
def test_held_merges_then_one_readout(sequence, rounds, initials, finals):
    validate_case(sequence, initials, finals, rounds,
                  physical_factory=physical_case, boundary_indices=(0, -1),
                  split_after_each_ppm=False)
