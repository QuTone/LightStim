"""Product-coloration syndrome extraction for hypergraph-product codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import stim

from lightstim.qec_code.generic_css import color_bipartite_edges

from .code_patch import HGPCode


Basis = Literal["X", "Z"]
Direction = Literal["horizontal", "vertical"]
SeedEdge = tuple[int, int]
QubitPair = tuple[int, int]


@dataclass(frozen=True)
class HGPProductColorLayer:
    """One product-color layer and its stable semantic label."""

    basis: Basis
    direction: Direction
    color: int
    seed: Literal["H1", "H2"]
    seed_edges: tuple[SeedEdge, ...]
    cnot_pairs: tuple[QubitPair, ...]

    @property
    def label(self) -> str:
        return f"HGP:{self.basis}:{self.direction}:color={self.color}"


class HGPProductColorationExtractionBlock:
    """Algorithm-2 product coloration, expressed with CNOT gates.

    The first seed ``H1`` is horizontal and the second seed ``H2`` is
    vertical, matching :class:`HGPCode`.  Each seed Tanner graph is colored
    once and its colors are lifted across every independent product row or
    column.  Consequently, labels such as ``X/horizontal/color=2`` are stable
    properties of the seed code instead of incidental colors of the flattened
    quantum Tanner graph.

    X and Z checks are separate measurement blocks.  This is the CNOT
    equivalent of Algorithm 2 in arXiv:2308.08648: X ancillas are prepared and
    measured in the X basis, while Z ancillas use the Z basis.
    """

    def __init__(self, system: Any, patch_name: str | None = None):
        self.system = system
        self.patch_name, self.patch = self._find_patch(patch_name)
        self.local_to_global = self.system.local_to_global_map[self.patch_name]

        self.horizontal_seed_colors = self._color_seed(self.patch.h1)
        self.vertical_seed_colors = self._color_seed(self.patch.h2)

        # Preserve the established H2-then-H1 gate order while giving each
        # layer the direction label of the product-block display.
        self.x_layers = tuple(
            self._lift_layers("X", "vertical")
            + self._lift_layers("X", "horizontal")
        )
        self.z_layers = tuple(
            self._lift_layers("Z", "vertical")
            + self._lift_layers("Z", "horizontal")
        )
        self.layers = self.x_layers + self.z_layers
        self.depth_x = len(self.x_layers)
        self.depth_z = len(self.z_layers)
        self.cnot_depth = len(self.layers)

        self._validate_active_stabilizers()
        self.measurement_blocks = self._build_measurement_blocks()
        self.circuit = stim.Circuit()
        for block in self.measurement_blocks:
            self.circuit += block

    def _find_patch(self, requested_name: str | None) -> tuple[str, HGPCode]:
        active_patch_names = {
            stabilizer.get("patch_name")
            for stabilizer in self.system.active_stabilizers
        }
        candidates = [
            (name, patch)
            for name, (patch, _) in self.system.patches.items()
            if isinstance(patch, HGPCode) and name in active_patch_names
        ]
        if requested_name is not None:
            candidates = [item for item in candidates if item[0] == requested_name]
        if len(candidates) != 1:
            names = [name for name, _ in candidates]
            qualifier = f" named {requested_name!r}" if requested_name is not None else ""
            raise ValueError(
                "HGP product coloration requires exactly one active HGP patch"
                f"{qualifier}; found {names}."
            )
        return candidates[0]

    @staticmethod
    def _color_seed(seed: Any) -> tuple[tuple[SeedEdge, ...], ...]:
        edges = [
            (check, bit)
            for check, row in enumerate(seed.row_supports)
            for bit in row
        ]
        return tuple(tuple(color) for color in color_bipartite_edges(edges))

    def _g(self, local_index: int) -> int:
        return self.local_to_global[local_index]

    def _lift_layers(
        self,
        basis: Basis,
        direction: Direction,
    ) -> list[HGPProductColorLayer]:
        active_syndromes = (
            set(self.system.active_syndrome_indices_x)
            if basis == "X"
            else set(self.system.active_syndrome_indices_z)
        )
        seed = "H1" if direction == "horizontal" else "H2"
        colors = (
            self.horizontal_seed_colors
            if direction == "horizontal"
            else self.vertical_seed_colors
        )
        result = []

        for color_index, seed_edges in enumerate(colors):
            pairs: list[QubitPair] = []
            if basis == "X" and direction == "vertical":
                # H2[check_2, bit_2]: X(bit_1, check_2) -> VV(bit_1, bit_2)
                for check_2, bit_2 in seed_edges:
                    for bit_1 in range(self.patch.h1.num_bits):
                        syndrome = self._g(self.patch.x_check_qubits[(bit_1, check_2)])
                        if syndrome in active_syndromes:
                            data = self._g(self.patch.vv_qubits[(bit_1, bit_2)])
                            pairs.append((syndrome, data))
            elif basis == "X" and direction == "horizontal":
                # H1[check_1, bit_1]: X(bit_1, check_2) -> CC(check_1, check_2)
                for check_1, bit_1 in seed_edges:
                    for check_2 in range(self.patch.h2.num_checks):
                        syndrome = self._g(self.patch.x_check_qubits[(bit_1, check_2)])
                        if syndrome in active_syndromes:
                            data = self._g(self.patch.cc_qubits[(check_1, check_2)])
                            pairs.append((syndrome, data))
            elif basis == "Z" and direction == "vertical":
                # H2[check_2, bit_2]: CC(check_1, check_2) -> Z(check_1, bit_2)
                for check_2, bit_2 in seed_edges:
                    for check_1 in range(self.patch.h1.num_checks):
                        syndrome = self._g(self.patch.z_check_qubits[(check_1, bit_2)])
                        if syndrome in active_syndromes:
                            data = self._g(self.patch.cc_qubits[(check_1, check_2)])
                            pairs.append((data, syndrome))
            else:
                # H1[check_1, bit_1]: VV(bit_1, bit_2) -> Z(check_1, bit_2)
                for check_1, bit_1 in seed_edges:
                    for bit_2 in range(self.patch.h2.num_bits):
                        syndrome = self._g(self.patch.z_check_qubits[(check_1, bit_2)])
                        if syndrome in active_syndromes:
                            data = self._g(self.patch.vv_qubits[(bit_1, bit_2)])
                            pairs.append((data, syndrome))

            pairs.sort()
            flat_qubits = [qubit for pair in pairs for qubit in pair]
            if len(flat_qubits) != len(set(flat_qubits)):
                raise RuntimeError(
                    f"Product color {basis}/{direction}/{color_index} is not conflict-free."
                )
            result.append(
                HGPProductColorLayer(
                    basis=basis,
                    direction=direction,
                    color=color_index,
                    seed=seed,
                    seed_edges=tuple(seed_edges),
                    cnot_pairs=tuple(pairs),
                )
            )
        return result

    def _validate_active_stabilizers(self) -> None:
        patch_x = {self._g(index) for index in self.patch.x_check_qubits.values()}
        patch_z = {self._g(index) for index in self.patch.z_check_qubits.values()}
        active_all = set(self.system.active_syndrome_indices)
        unexpected = active_all - patch_x - patch_z
        if unexpected:
            raise ValueError(
                "HGP product coloration cannot schedule active syndrome qubits "
                f"outside patch {self.patch_name!r}: {sorted(unexpected)}."
            )

        for basis, layers, stabilizers in (
            ("X", self.x_layers, self.system.active_stabilizers_x),
            ("Z", self.z_layers, self.system.active_stabilizers_z),
        ):
            expected: dict[int, set[int]] = {}
            for layer in layers:
                for control, target in layer.cnot_pairs:
                    syndrome, data = (
                        (control, target) if basis == "X" else (target, control)
                    )
                    expected.setdefault(syndrome, set()).add(data)

            actual: dict[int, set[int]] = {}
            for stabilizer in stabilizers:
                syndrome = stabilizer.get("syn_idx")
                if syndrome in actual:
                    raise ValueError(
                        f"Multiple active {basis} stabilizers use syndrome qubit {syndrome}."
                    )
                actual[syndrome] = set(stabilizer.get("data_indices", ()))

            if expected != actual:
                raise ValueError(
                    f"Active {basis} stabilizers do not match the unmodified HGP "
                    "product structure; use the generic CSS extraction block for "
                    "a deformed or coupled patch."
                )

    @staticmethod
    def _append_layer(circuit: stim.Circuit, layer: HGPProductColorLayer) -> None:
        targets = [qubit for pair in layer.cnot_pairs for qubit in pair]
        if targets:
            circuit.append("CNOT", targets, tag=layer.label)
        circuit.append("TICK", tag=layer.label)

    def _build_measurement_blocks(self) -> tuple[stim.Circuit, stim.Circuit]:
        x_syndromes = sorted(self.system.active_syndrome_indices_x)
        z_syndromes = sorted(self.system.active_syndrome_indices_z)

        x_block = stim.Circuit()
        x_block.append("RX", x_syndromes)
        x_block.append("TICK", tag="SE_start")
        for layer in self.x_layers:
            self._append_layer(x_block, layer)
        x_block.append("MX", x_syndromes)

        z_block = stim.Circuit()
        z_block.append("R", z_syndromes)
        z_block.append("TICK", tag="HGP:Z:start")
        for layer in self.z_layers:
            self._append_layer(z_block, layer)
        z_block.append("M", z_syndromes)

        return x_block, z_block


HGPCodeExtractionBlock = HGPProductColorationExtractionBlock


__all__ = [
    "HGPCodeExtractionBlock",
    "HGPProductColorLayer",
    "HGPProductColorationExtractionBlock",
]
