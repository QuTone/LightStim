"""Generator-layer syndrome extraction for GALA codes.

The GALA construction is block circulant.  Holding one monomial ``F_i`` or
``G_i`` fixed while ranging over the active check rows produces a transversal
matching, so every group-ring monomial is naturally one conflict-free CNOT
layer.  This generator-aligned schedule is an ablation of the generic greedy
coloration used by the GALA circuit-level experiment.  Generic edge coloring
has the same depth but can mix generators within every layer, changing circuit
hook errors.

For the X checks the natural runs are ``F`` then ``G``.  For the Z checks the
inverse matchings run ``G`` then ``F``, as in the Chen-style Kasai extraction
block.  A preset may provide ``extraction_order`` to reorder the monomials
within those runs (the paper gives an explicit movement-minimizing order for
the compact ``[[132,30,12]]`` instance).
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import stim

from .code_patch import GalaCode, Monomial


LayerSpec = Tuple[str, int, int]


class GalaGeneratorExtractionBlock:
    """Transversal generator-layer extraction for one active GALA patch."""

    def __init__(self, system: Any):
        self.system = system
        self.circuit = stim.Circuit()
        self._find_patch()
        self.generator_order = self._validated_order(
            getattr(self._patch, "extraction_order", None))
        self.depth_x = len(self.generator_order)
        self.depth_z = len(self.generator_order)
        self.cnot_depth = self.depth_x + self.depth_z
        self._build_circuit()

    def _find_patch(self) -> None:
        matches = [
            (name, patch)
            for name, (patch, _offset) in self.system.patches.items()
            if isinstance(patch, GalaCode)
        ]
        if len(matches) != 1:
            raise ValueError(
                "GalaGeneratorExtractionBlock requires exactly one GALA patch; "
                f"found {len(matches)}."
            )
        self._patch_name, self._patch = matches[0]
        self._local_to_global = self.system.local_to_global_map[self._patch_name]
        self._global_to_local = {
            global_idx: local_idx
            for local_idx, global_idx in self._local_to_global.items()
        }

    def _all_specs(self) -> List[LayerSpec]:
        return [
            (family, i, term)
            for family, lift in (("F", self._patch.f), ("G", self._patch.g))
            for i, entry in enumerate(lift)
            for term in range(len(entry))
        ]

    def _validated_order(
        self, order: Sequence[LayerSpec] | None
    ) -> List[LayerSpec]:
        expected = self._all_specs()
        if order is None:
            return expected
        normalized = [
            (str(family).upper(), int(i), int(term))
            for family, i, term in order
        ]
        if len(normalized) != len(expected) or set(normalized) != set(expected):
            raise ValueError(
                "GALA extraction_order must contain every F/G monomial "
                "exactly once."
            )
        return normalized

    def _slots(self, basis: str) -> List[Tuple[int, int, int]]:
        """Return ``(global ancilla, active row, lift point)`` records."""
        code = self._patch
        stabilizers = (
            self.system.active_stabilizers_x
            if basis == "X" else self.system.active_stabilizers_z
        )
        base = code.n_data + (code.num_x_checks if basis == "Z" else 0)
        slots: List[Tuple[int, int, int]] = []
        for stabilizer in stabilizers:
            if stabilizer.get("patch_name") != self._patch_name:
                continue
            syn_idx = stabilizer["syn_idx"]
            local_syn = self._global_to_local[syn_idx] - base
            row, point = divmod(local_syn, code.lift_size)
            if not (0 <= row < code.J):
                raise ValueError(
                    f"Syndrome qubit {syn_idx} does not match the GALA layout."
                )
            slots.append((syn_idx, row, point))
        return slots

    def _data_index(self, block: int, point: int) -> int:
        return self._local_to_global[self._patch._data_uid(block, point)]

    def _monomial(self, spec: LayerSpec) -> Monomial:
        family, i, term = spec
        lift = self._patch.f if family == "F" else self._patch.g
        return lift[i][term]

    def _append_x_layer(
        self, spec: LayerSpec, slots: Sequence[Tuple[int, int, int]]
    ) -> None:
        code = self._patch
        family, i, _term = spec
        monomial = self._monomial(spec)
        targets: List[int] = []
        for syn_idx, row, point in slots:
            block = (row + i) % code.m_blocks
            if family == "G":
                block += code.m_blocks
            data_idx = self._data_index(
                block, code.alphabet.apply(monomial, point))
            targets.extend([syn_idx, data_idx])
        if targets:
            self.circuit.append("CNOT", targets)
        self.circuit.append("TICK")

    def _append_z_layer(
        self, spec: LayerSpec, slots: Sequence[Tuple[int, int, int]]
    ) -> None:
        code = self._patch
        family, i, _term = spec
        inverse = code.alphabet.invert_monomial(self._monomial(spec))
        targets: List[int] = []
        for syn_idx, row, point in slots:
            block = (row - i) % code.m_blocks
            if family == "F":
                block += code.m_blocks
            data_idx = self._data_index(
                block, code.alphabet.apply(inverse, point))
            targets.extend([data_idx, syn_idx])
        if targets:
            self.circuit.append("CNOT", targets)
        self.circuit.append("TICK")

    def _build_circuit(self) -> None:
        active_syn = sorted(self.system.active_syndrome_indices)
        active_x_syn = sorted(self.system.active_syndrome_indices_x)
        x_slots = self._slots("X")
        z_slots = self._slots("Z")

        self.circuit.append("R", active_syn)
        self.circuit.append("TICK", tag="SE_start")
        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")

        for spec in self.generator_order:
            self._append_x_layer(spec, x_slots)

        # Preserve each family's published order while swapping the F/G runs,
        # matching the inverse Z phase of the Kasai generator schedule.
        z_order = [s for s in self.generator_order if s[0] == "G"] + [
            s for s in self.generator_order if s[0] == "F"
        ]
        for spec in z_order:
            self._append_z_layer(spec, z_slots)

        if active_x_syn:
            self.circuit.append("H", active_x_syn)
        self.circuit.append("TICK")
        self.circuit.append("M", active_syn)


__all__ = ["GalaGeneratorExtractionBlock"]
