"""Syndrome extraction for the ``[[6, 2, 2]]`` code.

The code has no 2D locality, so there is no natural geometric CNOT schedule.
It reuses LightStim's generic bipartite edge-coloring extraction block -- the
same choice the Kasai codes make -- which colors the X and Z Tanner graphs
independently and emits one CNOT layer per color.

:class:`HSixLogicalXCheckBlock` is a second, optional block: the Bell-pair
"H-check" from ``get_dist_circ`` in CQCL/Magic-H6 (``Code614.py``). It reads out
both logical-X operators non-destructively so that a ``CircuitBuilder`` emits
DETECTORs which herald encoder faults that flip a logical without disturbing a
stabilizer (and so are invisible to the stabilizer extraction above).
"""

import stim

from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

# One noiseless SE round for a HSixCode system.
HSixExtractionBlock = GenericCSSColorationExtractionBlock


class HSixLogicalXCheckBlock:
    """The Magic-H6 Bell-pair "H-check": non-destructive readout of X0_L and X1_L.

    Verbatim port of the ancilla block in ``get_dist_circ`` (CQCL/Magic-H6,
    ``Stim/ConcatenatedMSProtocolSim/Code614.py``)::

        H   a0
        CX  a0 a1                       # a0,a1 -> Bell pair
        CX  a0 d0 , CX a1 d1
        CX  a0 d2 , CX a1 d3            # a0 copies X0_L support {d0,d2,d4}
        CX  a0 d4 , CX a1 d5            # a1 copies X1_L support {d1,d3,d5}
        CX  a0 a1
        H   a0
        M   a0 a1                       # get_dist_circ uses MR

    The Bell pair keeps the two logical degrees of freedom alive (a naive
    ``MX`` of two product-state ancillas would measure the logicals
    destructively and desync the tracker). On an encoded ``|++>_L`` both
    outcomes are deterministic, so each becomes a DETECTOR.

    Fed to the builder exactly like an SE block::

        block = HSixLogicalXCheckBlock(system)
        builder.apply_syndrome_extraction(circuit_chunk=block.circuit, rounds=1)

    Args:
        system: The QECSystem holding the ``[[6, 2, 2]]`` patch.
        ancilla_indices: The two check-ancilla global indices ``(a0, a1)``. If
            ``None``, the block takes the patch's bare ``role="syndrome"``
            ancillas -- those attached to neither an X nor a Z stabilizer, i.e.
            the ones added by ``HSixCode(h_check_ancillas=2)``.
        patch_name: Name of the patch to read; defaults to the system's only
            patch.
    """

    def __init__(self, system, ancilla_indices=None, patch_name=None):
        self.system = system
        self.patch_name = patch_name or self._sole_patch_name(system)

        anc = (
            list(ancilla_indices)
            if ancilla_indices is not None
            else self._bare_syndrome_ancillas(system, self.patch_name)
        )
        if len(anc) != 2:
            raise ValueError(
                f"The Bell-pair H-check needs exactly 2 ancillas, got {len(anc)}."
            )
        self.ancilla_indices = anc

        # X0_L / X1_L supports, from the patch's registered logical-X operators.
        supports = self._x_logical_supports(system, self.patch_name)
        if len(supports) != 2:
            raise ValueError(
                f"Expected 2 logical-X operators, got {len(supports)}."
            )
        self.x0_support, self.x1_support = supports

        self.circuit = stim.Circuit()
        self._build_circuit()

    def _build_circuit(self):
        a0, a1 = self.ancilla_indices

        # --- Step 1: reset ancillas, put a0 in |+> ---
        self.circuit.append("R", [a0, a1])

        #Bell state preparation: H on a0, then CNOT a0->a1
        self.circuit.append("H", [a0])
        # Critical tag for NoiseInjector (data-qubit error insertion point).
        self.circuit.append("TICK", tag="SE_start")
        # --- Step 2: Bell pair, then copy each logical-X support
        self.circuit.append("CX", [a0, a1])
        for d0, d1 in zip(self.x0_support, self.x1_support): #transversal CNOT
            self.circuit.append("CX", [a0, d0])
            self.circuit.append("CX", [a1, d1])

        #Mirror for bell state
        self.circuit.append("CX", [a0, a1])
        self.circuit.append("H", [a0])
        self.circuit.append("TICK")

        # --- Step 3: measure both ancillas (get_dist_circ: MR) ---
        self.circuit.append("M", [a0, a1])

    @staticmethod
    def _sole_patch_name(system):
        names = list(system.patches)
        if len(names) != 1:
            raise ValueError(
                f"patch_name is required when the system holds {len(names)} patches."
            )
        return names[0]

    @staticmethod
    def _bare_syndrome_ancillas(system, patch_name):
        patch = system.patches[patch_name][0]
        l2g = system.local_to_global_map[patch_name]
        bare = (
            patch.syndrome_indices
            - patch.syndrome_indices_x
            - patch.syndrome_indices_z
        )
        if not bare:
            raise ValueError(
                f"Patch '{patch_name}' has no spare ancillas. Build it with "
                "HSixCode(h_check_ancillas=2) or pass ancilla_indices."
            )
        return sorted(l2g[local] for local in bare)

    @staticmethod
    def _x_logical_supports(system, patch_name):
        supports = [
            sorted(op["pauli"])
            for op in system.logical_ops
            if op.get("type") == "X" and op.get("patch_name") == patch_name
        ]
        if not supports:
            raise ValueError(
                f"Patch '{patch_name}' registered no logical-X operators."
            )
        return supports


__all__ = ["HSixExtractionBlock", "HSixLogicalXCheckBlock"]
