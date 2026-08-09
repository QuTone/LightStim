# src/qec_code/surface_code/rotated/operation.py

"""
Logical operation set for Rotated Surface Code.

Provides state injection, logical unencoding, and logical Clifford gates as
LogicalOpSet methods, callable via
LogicalExecutor.apply_logical_operation().
"""

from typing import Literal, Tuple, Dict, Any, Type, List

from lightstim.ir.operation import CSSLogicalOpSet
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.surface_code.rotated.code_patch import RotatedSurfaceCode
import stim


# ------------------------------------------------------------------------------
# Module-level helpers — dynamical fold-transversal S
# ------------------------------------------------------------------------------

def _get_half_cycle_fold_yx_parts(
    system: Any,
    patch: QECPatch,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Return the fold-transversal layer for the rotated-code half-cycle.

    Two CNOT layers into the standard four-layer rotated surface-code syndrome
    extraction circuit, the data and interior syndrome qubits form an
    unrotated surface code.  In LightStim's coordinate frame, the fold axis of
    arXiv:2412.01391 is the local ``y=x`` diagonal.

    Boundary syndrome qubits are excluded: they are the unentangled boundary
    qubits outside the half-cycle unrotated-code bulk.  On the fold diagonal,
    data qubits receive S and syndrome qubits receive S†.  Off the diagonal,
    mirrored bulk qubits are paired by CZ.
    """
    if not isinstance(patch, RotatedSurfaceCode):
        raise TypeError(
            f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
        )
    if patch.distance_z != patch.distance_x:
        raise ValueError(
            "Dynamical fold-transversal S requires a square patch "
            f"(distance_z == distance_x). Got "
            f"{patch.distance_z} != {patch.distance_x}."
        )

    distance = patch.distance_z
    sx, sy = patch.shift
    target_syndromes = {
        stabilizer["syn_idx"]
        for stabilizer in patch.stabilizers
        if stabilizer.get("syn_idx") is not None
    }
    active_syndromes = set(system.active_syndrome_indices)
    missing_syndromes = sorted(target_syndromes - active_syndromes)
    if missing_syndromes:
        raise ValueError(
            "Dynamical fold-transversal S requires the patch's ordinary "
            "syndrome ancillas to be active. Missing global qubit indices "
            f"{missing_syndromes}."
        )

    fold_qubits = set(patch.data_indices) | target_syndromes
    diagonal_data: List[int] = []
    diagonal_syndromes: List[int] = []
    mirror_pairs: List[Tuple[int, int]] = []

    # The bulk occupies local coordinates 1..2d-1 on both axes. Syndrome
    # ancillas with a local coordinate 0 or 2d are the unentangled boundary
    # qubits excluded by the paper's protocol.
    bulk_min = 1
    bulk_max = 2 * distance - 1

    for qubit in sorted(fold_qubits):
        gx, gy = system.qubit_coords[qubit]
        lx = gx - sx
        ly = gy - sy
        if not (
            bulk_min <= lx <= bulk_max
            and bulk_min <= ly <= bulk_max
        ):
            continue

        if lx == ly:
            if qubit in patch.data_indices:
                diagonal_data.append(qubit)
            else:
                diagonal_syndromes.append(qubit)
            continue

        if lx > ly:
            continue

        reflected_coord = QECPatch.snap_coord((ly + sx, lx + sy))
        reflected_qubit = system.index_map.get(reflected_coord)
        if reflected_qubit is None or reflected_qubit not in fold_qubits:
            raise ValueError(
                f"Half-cycle mirror qubit at {reflected_coord} not found "
                f"for qubit {qubit} at {(gx, gy)}."
            )
        mirror_pairs.append((qubit, reflected_qubit))

    return diagonal_data, diagonal_syndromes, mirror_pairs


def _build_half_cycle_fold_s(
    system: Any,
    patch: QECPatch,
    *,
    inverse: bool,
) -> stim.Circuit:
    """Build the physical fold-transversal S or S† layer."""
    diagonal_data, diagonal_syndromes, mirror_pairs = (
        _get_half_cycle_fold_yx_parts(system, patch)
    )

    circuit = stim.Circuit()
    if mirror_pairs:
        circuit.append(
            "CZ",
            [qubit for pair in mirror_pairs for qubit in pair],
        )
    if diagonal_data:
        circuit.append(
            "S_DAG" if inverse else "S",
            sorted(diagonal_data),
        )
    if diagonal_syndromes:
        circuit.append(
            "S" if inverse else "S_DAG",
            sorted(diagonal_syndromes),
        )
    return circuit


def _insert_fold_after_second_cnot_layer(
    syndrome_extraction: stim.Circuit,
    fold_layer: stim.Circuit,
) -> stim.Circuit:
    """Insert ``fold_layer`` at the four-CNOT schedule's half-cycle."""
    result = stim.Circuit()
    cnot_layers = 0
    moment_has_cnot = False
    inserted = False

    for instruction in syndrome_extraction:
        if not isinstance(instruction, stim.CircuitInstruction):
            raise ValueError(
                "Dynamical fold-transversal S requires an explicit, "
                "non-REPEAT syndrome-extraction round."
            )

        result.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy(),
            tag=instruction.tag,
        )

        if instruction.name in ("CX", "CNOT"):
            moment_has_cnot = True
        if instruction.name != "TICK":
            continue

        if moment_has_cnot:
            cnot_layers += 1
            moment_has_cnot = False
        if cnot_layers == 2 and not inserted:
            result += fold_layer
            result.append("TICK")
            inserted = True

    if moment_has_cnot:
        cnot_layers += 1
    if cnot_layers != 4:
        raise ValueError(
            "Dynamical fold-transversal S requires exactly four CNOT "
            f"layers in one syndrome-extraction round; found {cnot_layers}."
        )
    if not inserted:
        raise ValueError(
            "Could not find the half-cycle boundary after the second CNOT layer."
        )
    return result


def _disable_extraction_idle_noise(circuit: stim.Circuit) -> stim.Circuit:
    """Retag an explicit SE round so noiseless mode also excludes idle noise."""
    result = stim.Circuit()
    for instruction in circuit:
        if not isinstance(instruction, stim.CircuitInstruction):
            raise ValueError("Expected an explicit syndrome-extraction round.")
        tag = (
            "noiseless"
            if instruction.name == "TICK" and instruction.tag == "SE_start"
            else instruction.tag
        )
        result.append(
            instruction.name,
            instruction.targets_copy(),
            instruction.gate_args_copy(),
            tag=tag,
        )
    return result


# ------------------------------------------------------------------------------
# Module-level helpers — state injection
# ------------------------------------------------------------------------------

def _get_injection_site(
    patch: QECPatch,
    system: Any,
    protocol: Literal["corner", "middle"],
) -> Tuple[Tuple[float, float], int]:
    """
    Return the (coord, global_index) of the injection site for the given protocol.

    Args:
        patch: Global patch (data_indices are global).
        system: QECSystem for coord lookup.
        protocol: 'corner' or 'middle'.

    Returns:
        (injection_coord, injection_global_index)
    """
    data_global_indices = sorted(patch.data_indices)
    coords = [(system.qubit_coords[gidx], gidx) for gidx in data_global_indices]

    if protocol == "corner":
        # Corner = qubit with smallest (x, y)
        corner_coord, corner_gidx = min(coords, key=lambda t: (t[0][0], t[0][1]))
        return corner_coord, corner_gidx
    else:
        # Middle = center qubit in logical grid
        d = patch.distance_z
        mid = d // 2 + 1
        # Find the corner to compute relative offsets
        corner_coord = min(c for c, _ in coords)
        # In rotated surface code, data qubits are at odd coords with spacing 2
        # Logical (mid, mid) maps to corner + (2*(mid-1), 2*(mid-1))
        mid_coord = (corner_coord[0] + 2 * (mid - 1), corner_coord[1] + 2 * (mid - 1))
        for c, gidx in coords:
            if abs(c[0] - mid_coord[0]) < 0.5 and abs(c[1] - mid_coord[1]) < 0.5:
                return c, gidx
        raise ValueError(
            f"Middle injection coordinate {mid_coord} not found in patch layout. "
            f"Middle injection may require odd distance."
        )


def _get_injection_init(
    patch: QECPatch,
    system: Any,
    inject_state: Literal["Z", "X", "Y"],
    protocol: Literal["corner", "middle"],
) -> Tuple[Dict[int, str], int]:
    """
    Build init_dict for state injection.

    For corner protocol:
        Corner qubit gets inject_state. Lower diagonal (rel_y >= rel_x) -> X, upper -> Z.
    For middle protocol:
        Center qubit gets inject_state. Diagonal split per middle-injection rule.

    Args:
        patch: Global patch (data_indices are global).
        system: QECSystem for coord lookup.
        inject_state: Target state ('Z', 'X', or 'Y').
        protocol: 'corner' or 'middle'.

    Returns:
        (init_dict, injection_global_index)
    """
    injection_coord, injection_gidx = _get_injection_site(patch, system, protocol)
    data_global_indices = sorted(patch.data_indices)

    init_dict: Dict[int, str] = {}

    if protocol == "corner":
        # Use injection site as origin for relative coordinates
        ox, oy = injection_coord
        for gidx in data_global_indices:
            cx, cy = system.qubit_coords[gidx]
            rel_x, rel_y = cx - ox, cy - oy
            if gidx == injection_gidx:
                init_dict[gidx] = inject_state
            elif rel_y >= rel_x:
                init_dict[gidx] = "X"   # lower diagonal -> |+>
            else:
                init_dict[gidx] = "Z"   # upper diagonal -> |0>
    else:
        # Middle injection: diagonal split based on relative position
        d = patch.distance_z
        corner_coord = min(
            (system.qubit_coords[gidx] for gidx in data_global_indices),
            key=lambda c: (c[0], c[1]),
        )
        ox, oy = corner_coord

        for gidx in data_global_indices:
            cx, cy = system.qubit_coords[gidx]
            # Convert to logical coords (1..d)
            lx = int(round((cx - ox) / 2)) + 1
            ly = int(round((cy - oy) / 2)) + 1

            if gidx == injection_gidx:
                init_dict[gidx] = inject_state
            elif (lx < ly and lx + ly <= d + 1) or (lx > ly and lx + ly >= d + 1):
                init_dict[gidx] = "Z"
            else:
                init_dict[gidx] = "X"

    return init_dict, injection_gidx


# ------------------------------------------------------------------------------
# LogicalOpSet implementation
# ------------------------------------------------------------------------------

class RotatedSurfaceCodeLogicalOpSet(CSSLogicalOpSet):
    """
    Logical operation set for Rotated Surface Code.

    Implements state injection, logical unencode, and surface-code-specific
    Clifford gates as composable operations for use with LogicalExecutor.
    """

    def __init__(self, extraction_block_class: Type):
        """
        Args:
            extraction_block_class: SE block class for this code
                (e.g. RotatedSurfaceCodeExtractionBlock). Takes system, has .circuit.
        """
        super().__init__()
        self.name = "RotatedSurfaceCode"
        self.extraction_block_class = extraction_block_class

    # ------------------------------------------------------------------
    # Dynamical fold-transversal phase gates
    # ------------------------------------------------------------------

    def _apply_dynamical_fold_s(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        *,
        inverse: bool,
        noiseless: bool,
    ) -> None:
        """Apply one complete S-SE or S†-SE round.

        The full round is deliberately passed through
        ``apply_syndrome_extraction`` as one measurement block.  This lets the
        tracker propagate the terminal syndrome measurements through the
        mid-cycle CZ/S/S† layer, apply the ancilla resets, and construct the
        mixed X/Z detectors automatically.
        """
        if self.extraction_block_class is None:
            raise ValueError(
                "extraction_block_class is required for the dynamical "
                "fold-transversal S protocol."
            )

        ordinary_round = self.extraction_block_class(builder.system).circuit
        fold_layer = _build_half_cycle_fold_s(
            builder.system,
            patch,
            inverse=inverse,
        )
        dynamical_round = _insert_fold_after_second_cnot_layer(
            ordinary_round,
            fold_layer,
        )
        if noiseless:
            dynamical_round = _disable_extraction_idle_noise(dynamical_round)
        builder.apply_syndrome_extraction(
            circuit_chunk=dynamical_round,
            rounds=1,
            noiseless=noiseless,
            # The mid-cycle fold layer needs the measurement-block engine's
            # retained-data semantics; opt in explicitly (the legacy engine
            # is the default for calls without measurement_blocks).
            measurement_blocks=(dynamical_round,),
        )

    def fold_transversal_s(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        noiseless: bool = False,
    ) -> None:
        """Implement logical S as one dynamical S-SE round.

        Protocol (arXiv:2412.01391):

        1. Reset the syndrome ancillas and run CNOT layers 1–2.
        2. On the half-cycle unrotated-code state, apply mirrored CZ pairs,
           S on diagonal data qubits, and S† on diagonal syndrome qubits.
        3. Run CNOT layers 3–4 and measure the syndrome ancillas.

        Logical action: Z_L -> Z_L and X_L -> Y_L.
        """
        self._apply_dynamical_fold_s(
            builder,
            patch,
            inverse=False,
            noiseless=noiseless,
        )

    def fold_transversal_s_dag(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        noiseless: bool = False,
    ) -> None:
        """Implement logical S† as the inverse dynamical S-SE round."""
        self._apply_dynamical_fold_s(
            builder,
            patch,
            inverse=True,
            noiseless=noiseless,
        )

    # ------------------------------------------------------------------
    # State preparation / teardown
    # ------------------------------------------------------------------

    def state_injection(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
        rounds: int = 0,
        post_select_coords=None,
        se_block_kwargs: dict = None,
        noiseless_init: bool = False,
    ):
        """
        State injection: initialize data qubits and optionally run SE rounds.

        The injection site receives the target state; surrounding qubits are initialized
        in a diagonal pattern (X/Z split). Syndrome detectors are tagged for post-selection.

        When rounds > 0, syndrome extraction is performed immediately (convenient for
        single-patch experiments). For multi-patch experiments where all patches must be
        initialized before SE, call with rounds=0 and run SE at the experiment level.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: Target logical state ('Z' -> |0>, 'X' -> |+>, 'Y' -> |+i>).
            protocol: Injection site — 'corner' or 'middle'.
            rounds: Number of SE rounds to run after init (0 = init only).
            post_select_coords: Set of (x, y, t) detector coords to tag for post-selection.
                If None (default), tag ALL syndrome coords (full post-selection).
                Pass empty set() for no post-selection (full QEC).
        """
        if not isinstance(patch, RotatedSurfaceCode):
            raise TypeError(
                f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
            )

        system = builder.system

        # Build qubit-initialization dict (global indices)
        init_dict, _ = _get_injection_init(patch, system, inject_state, protocol)

        # Tag syndrome coords for post-selection
        if post_select_coords is None:
            # Default: tag all syndrome coords (full post-selection)
            post_select_coords = set()
            for stab in patch.stabilizers:
                syn_idx = stab.get("syn_idx")
                if syn_idx is not None and syn_idx in system.qubit_coords:
                    coord = system.qubit_coords[syn_idx]
                    post_select_coords.add(tuple(coord) + (0.0,))
        builder.tracker.post_select_detector_coords |= post_select_coords

        # Emit reset instructions and update the tracker tableau
        builder.initialize(init_dict=init_dict, n=system.num_qubits, noiseless=noiseless_init)

        # Optional syndrome extraction rounds
        if rounds > 0:
            se_block = self.extraction_block_class(system, **(se_block_kwargs or {}))
            builder.apply_syndrome_extraction(
                circuit_chunk=se_block.circuit,
                rounds=rounds,
            )

    def logical_unencode(
        self,
        builder: CircuitBuilder,
        patch: QECPatch,
        inject_state: Literal["Z", "X", "Y"] = "Z",
        protocol: Literal["corner", "middle"] = "corner",
    ) -> int:
        """
        Unencode the logical state back to a single physical qubit (inverse of injection).

        Measures all data qubits of the patch EXCEPT the injection site in their
        initialization basis. The tracker automatically generates final-round detectors.
        The injection site qubit is left unmeasured, carrying the logical state.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch: Global QECPatch (returned by system.add_patch).
            inject_state: The state that was injected ('Z', 'X', or 'Y').
            protocol: Must match the protocol used in state_injection.

        Returns:
            Global qubit index of the unmeasured injection-site qubit.
        """
        if not isinstance(patch, RotatedSurfaceCode):
            raise TypeError(
                f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
            )

        system = builder.system

        # Reconstruct the init dict to know each qubit's basis
        init_dict, injection_gidx = _get_injection_init(
            patch, system, inject_state, protocol
        )

        # Measure all data qubits except the injection site, in their init basis
        unencode_measurements = {
            gidx: basis for gidx, basis in init_dict.items()
            if gidx != injection_gidx
        }

        # Use apply_data_readout for automatic detector/observable generation.
        # Noiseless: unencode measurements are deterministic given the stabilizer
        # state — noise here would corrupt the injection fidelity measurement.
        builder.apply_data_readout(final_measurements=unencode_measurements, noiseless=True)

        return injection_gidx

    # ------------------------------------------------------------------
    # Fold-transversal Hadamard
    # ------------------------------------------------------------------

    def _rot90_swap_layers(self, system, patch: QECPatch):
        """
        Partition the data qubits of `patch` into disjoint SWAP layers that
        realize a 90-degree rotation of the patch about its center.

        The rotated surface code's ZX-duality is the 90-degree rotation: its
        diagonal reflections are CSS automorphisms, NOT dualities, so the
        handbook's diagonal-reflection fold does not preserve the stabilizer
        group on this layout. The valid fold-transversal Hadamard therefore uses
        the rotation as its qubit permutation (cf. handbook Fig. 24, "a SWAP
        permutation then rotates the patch").

        Local-frame rotation (square patch, distance D, data coords in [1, 2D-1]):
            (x, y) -> (2D - y, x)

        Returns a list of layers, each a list of disjoint (idx_a, idx_b) pairs.
        """
        if patch.distance_z != patch.distance_x:
            raise ValueError(
                "Fold-transversal Hadamard requires a square patch "
                f"(distance_z == distance_x). Got {patch.distance_z} != {patch.distance_x}."
            )
        sx, sy = patch.shift
        D = patch.distance_z
        s = 2 * D
        index_map = system.index_map

        # Build the 90-degree rotation permutation (global idx -> global idx).
        perm = {}
        for g in sorted(patch.data_indices):
            gx, gy = patch.qubit_coords[g]
            lx, ly = gx - sx, gy - sy
            rcoord = QECPatch.snap_coord((s - ly + sx, lx + sy))
            if rcoord not in index_map:
                raise ValueError(
                    f"Rotated image {rcoord} of data qubit at {(gx, gy)} not found. "
                    "Fold-transversal Hadamard requires a square, odd-distance patch."
                )
            perm[g] = index_map[rcoord]

        # Decompose the permutation into ordered transpositions, cycle by cycle.
        swaps, visited = [], set()
        for start in perm:
            if start in visited:
                continue
            cycle, nxt = [start], perm[start]
            visited.add(start)
            while nxt != start:
                cycle.append(nxt)
                visited.add(nxt)
                nxt = perm[nxt]
            # reversed-order transpositions realize the cycle (forward = inverse)
            for j in range(len(cycle) - 2, -1, -1):
                swaps.append((cycle[j], cycle[j + 1]))

        # Pack transpositions into disjoint parallel layers (respecting deps).
        last, layer_map = {}, {}
        for a, b in swaps:
            lyr = max(last.get(a, 0), last.get(b, 0)) + 1
            last[a] = last[b] = lyr
            layer_map.setdefault(lyr, []).append((a, b))
        return [layer_map[k] for k in sorted(layer_map)]

    def fold_transversal_hadamard(self, builder: CircuitBuilder, patch: QECPatch,
                                  noiseless: bool = False):
        """
        Implement the logical Hadamard H_L on the rotated surface code.

        Physical circuit:
          Layer 0    — transversal H : H on every data qubit (X <-> Z)
          Layers 1.. — fold (SWAP)   : a 90-degree rotation of the patch,
                                       emitted as disjoint SWAP layers.

        Logical action: X_L <-> Z_L. Verified (tableau) to preserve the stabilizer
        group and exchange the logical operators for the LightStim rotated layout
        at d = 3, 5, 7. Unlike the unrotated code (y=x diagonal reflection), the
        rotated code's only ZX-duality is this 90-degree rotation.

        Args:
            builder: CircuitBuilder driving the experiment.
            patch:   The (global) RotatedSurfaceCode patch (must be square, odd d).
            noiseless: If True, tag gate instructions as noiseless.
        """
        if not isinstance(patch, RotatedSurfaceCode):
            raise TypeError(
                f"Expected RotatedSurfaceCode patch, got {type(patch).__name__}"
            )
        system = builder.system
        all_data = sorted(patch.data_indices)
        layers = self._rot90_swap_layers(system, patch)

        unitary = stim.Circuit()
        unitary.append("H", all_data)
        for layer in layers:
            unitary.append("TICK")
            unitary.append("SWAP", [q for pair in layer for q in pair])

        builder.apply_unitary_block(unitary, noiseless=noiseless)
