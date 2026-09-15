"""Concurrent, unflagged extraction for the H family."""

from typing import Any

import stim

from .code_patch import HCode


class HCodeExtractionBlock:
    """Pipeline all four checks in n+2 CNOT layers, without flag ancillas.

    Let s=0,1,2,3 label XA,XB,ZA,ZB. Check s touches shared data 2 in
    layer s, its private data in layers s+1 onward, and shared data 3 in
    layer n-2+s. Private orders are (0,1) for A and (4,...,n-1) for B.
    Thus every X interaction precedes every Z interaction on the same data
    qubit. Interleaving only commutes mutually commuting gates: the ideal
    unitary equals serial X-then-Z extraction. The shared data at each end
    also prevent unchecked interior hook tails in the audited memory circuits.
    An individual round can still leave an unflagged weight-two data residual;
    the distance-two validation includes later checks and final memory readout.
    See docs/design/h_code_se_review.md for the output-boundary counterexample.

    x_layers/z_layers share the same time axis and contain (syndrome,data)
    pairs; cnot_layers contains directed (control,target) pairs. depth_x/z
    count occupied layers per basis, whose sum is NOT the concurrent depth.
    Multiple independent H patches run in parallel, using global indices.
    Active checks must be unmodified H-code checks; unsupported deformations
    or other code families raise ValueError rather than silently omitting them.
    """

    def __init__(self, system: Any):
        self.system = system
        active = {}
        for check in system.active_stabilizers:
            key = (check.get("syn_idx"), check["type"])
            if key in active:
                raise ValueError("HCode extraction requires one check per syndrome ancilla.")
            active[key] = check

        scheduled = []
        for name, (patch, _) in system.patches.items():
            if not isinstance(patch, HCode):
                continue
            mapping = system.local_to_global_map[name]
            for slot, check in enumerate(patch.stabilizers):
                basis = check["type"]
                syn = mapping[check["syn_idx"]]
                key = (syn, basis)
                if key not in active:
                    continue
                actual = active.pop(key)
                if slot >= 4 or basis != ("X", "X", "Z", "Z")[slot]:
                    raise ValueError("HCode extraction requires canonical H-code check ordering.")
                support = (0, 1, 2, 3) if slot % 2 == 0 else range(2, patch.n)
                expected = {mapping[q]: basis for q in support}
                if actual["pauli"] != expected or set(actual["data_indices"]) != set(expected):
                    raise ValueError("HCode extraction requires unmodified H-code checks.")
                private = (0, 1) if slot % 2 == 0 else tuple(range(4, patch.n))
                schedule = [
                    (slot, 2),
                    *((slot + 1 + i, q) for i, q in enumerate(private)),
                    (patch.n - 2 + slot, 3),
                ]
                scheduled.extend((t, basis, syn, mapping[q]) for t, q in schedule)
        if active or not scheduled:
            raise ValueError("HCode extraction requires active, unmodified H-code checks only.")

        self.cnot_depth = max(t for t, *_ in scheduled) + 1
        self.x_layers = [[] for _ in range(self.cnot_depth)]
        self.z_layers = [[] for _ in range(self.cnot_depth)]
        for t, basis, syn, data in scheduled:
            (self.x_layers if basis == "X" else self.z_layers)[t].append((syn, data))
        self.depth_x = sum(bool(layer) for layer in self.x_layers)
        self.depth_z = sum(bool(layer) for layer in self.z_layers)
        self.cnot_layers = [
            [*xs, *((data, syn) for syn, data in zs)]
            for xs, zs in zip(self.x_layers, self.z_layers)
        ]
        for layer in self.cnot_layers:
            targets = [q for pair in layer for q in pair]
            if len(targets) != len(set(targets)):
                raise ValueError("HCode extraction requires disjoint patches/qubits per layer.")

        syndrome = sorted({syn for _, _, syn, _ in scheduled})
        x_syndrome = sorted({syn for _, basis, syn, _ in scheduled if basis == "X"})
        self.circuit = stim.Circuit()
        self.circuit.append("R", syndrome)
        self.circuit.append("TICK", tag="SE_start")
        if x_syndrome:
            self.circuit.append("H", x_syndrome)
        self.circuit.append("TICK")
        for layer in self.cnot_layers:
            if layer:
                self.circuit.append("CX", [q for pair in layer for q in pair])
            self.circuit.append("TICK")
        if x_syndrome:
            self.circuit.append("H", x_syndrome)
        self.circuit.append("TICK")
        self.circuit.append("M", syndrome)


HSixExtractionBlock = HCodeExtractionBlock

__all__ = ["HCodeExtractionBlock", "HSixExtractionBlock"]
