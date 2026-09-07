"""The hole-free planar subsystem surface code of Bravyi et al."""

from collections import defaultdict
from numbers import Integral

from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode
from .SE_block import SubsystemSurfaceCodeExtractionBlock


class SubsystemSurfaceCode(QECPatch):
    """Square ``[[3d²-2d, 1, (d-1)², d]]`` subsystem surface code, d >= 2.

    Use the coordinates and boundary checks of a rotated surface-code patch
    of distance ``2*d-1``, omitting data sites ``(4*i+3, 4*j+3)``. The remaining
    bulk checks are weight-3 gauges whose ancillas sit at triangle hypotenuse
    midpoints. Two opposite same-basis triangles form each weight-6 center
    stabilizer. Weight-2 boundary checks belong to both the gauge group and
    its center. The X/Z convention follows LightStim's rotated patch.

    The omitted sites are not registered as qubits. Gauge qubits are encoded
    degrees of freedom, distinct from the dedicated measurement ancillas.
    """

    default_extraction_block_class = SubsystemSurfaceCodeExtractionBlock

    def _process_params(self):
        distance = self.params.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, Integral) or distance < 2:
            raise ValueError("Subsystem surface-code distance must be an integer >= 2.")
        self.distance = int(distance)
        self.parent_distance = 2 * self.distance - 1
        self.num_gauge_qubits = (self.distance - 1) ** 2

    def build(self):
        # Reuse the existing geometric template, then declare the subsystem
        # algebra independently; the parent is never registered in QECSystem.
        parent = RotatedSurfaceCode(distance=self.parent_distance)
        omitted = {
            (4 * i + 3, 4 * j + 3)
            for i in range(self.distance - 1)
            for j in range(self.distance - 1)
        }
        for index, coord in parent.qubit_coords.items():
            if coord in omitted:
                continue
            if index in parent.data_indices:
                role = "data"
            else:
                role = "syndrome_x" if index in parent.syndrome_indices_x else "syndrome_z"
            self.add_qubit(*coord, role=role)

        bulk_centers = defaultdict(set)
        for check in parent.stabilizers:
            basis = check["type"]
            support = {parent.qubit_coords[q] for q in check["data_indices"]}
            removed = support & omitted
            support -= omitted
            targets = {coord: basis for coord in sorted(support)}
            self.create_stim_gauge(targets, syn_coord=check["syn_coord"], type=basis)
            if removed:
                # Opposite triangle supports multiply by symmetric difference.
                key = (next(iter(removed)), basis)
                bulk_centers[key].symmetric_difference_update(support)
            else:
                self.create_stim_stabilizer(targets, type=basis)

        for (_, basis), support in sorted(bulk_centers.items()):
            self.create_stim_stabilizer(
                {coord: basis for coord in sorted(support)}, type=basis,
            )

        # These parent boundary strings avoid every omitted site and commute
        # with all gauges. Their weight is 2*d-1; dressed distance is d.
        for logical in parent.logical_ops:
            self.create_stim_logical(
                {parent.qubit_coords[q]: basis for q, basis in logical["pauli"].items()},
                logical["type"],
            )
        self.num_logicals = 1

    def get_info(self):
        info = super().get_info()
        info.update(
            distance=self.distance,
            parent_distance=self.parent_distance,
            num_data_qubits=len(self.data_indices),
            num_gauge_qubits=self.num_gauge_qubits,
        )
        return info
