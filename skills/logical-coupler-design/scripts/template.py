"""
Logical Coupler Design — template script.

Demonstrates: implementing a new LogicalCouplerProtocol from scratch.

The example: a minimal LinearZZCoupler that couples two unrotated surface code patches
in a ZZ configuration using a hand-built 1-column corridor. This is a pedagogical
alternative to UnrotatedTwoPatchCoupler, showing every step explicitly.

Run from repo root:
    venv/bin/python skills/logical-coupler-design/scripts/template.py
"""
import sys
sys.path.insert(0, ".")

import math
from typing import List, Optional

from lightstim.ir.qec_system import QECSystem
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.coupler import LogicalCouplerProtocol, LogicalCouplerPatch
from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.surface_code.unrotated import (
    UnrotatedSurfaceCode,
    UnrotatedSurfaceCodeExtractionBlock,
)
from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: The coupler protocol
# ══════════════════════════════════════════════════════════════════════════════

class LinearZZCoupler(LogicalCouplerProtocol):
    """
    Minimal ZZ coupler for two horizontally-adjacent unrotated surface code patches.

    Assumptions (for pedagogical simplicity):
    - Patches are placed side by side along the X axis with a gap of exactly 2.
    - Both patches have the same distance d (same Y extent).
    - No transposition or rotation.

    Design:
    - Fills the 1-column gap with data qubits and Z-syndrome qubits.
    - Redefines boundary Z-syndrome qubits (at gap edges) to include the corridor.
    """

    EXPECTED_PATCH_COUNT = 2

    def __init__(self):
        super().__init__(name_prefix="linear_zz")

    def _build_coupler_geometry(self, coupler_patch: LogicalCouplerPatch,
                                patches: List[QECPatch], **params):
        p_left, p_right = self._order_patches(patches)

        # ── Step 1: Analyze geometry ──────────────────────────────────────────
        left_max_x  = p_left._get_bounds()[1]   # rightmost x of left patch
        right_min_x = p_right._get_bounds()[0]  # leftmost x of right patch
        gap = right_min_x - left_max_x

        if not math.isclose(gap, 2.0, abs_tol=0.1):
            raise ValueError(f"LinearZZCoupler requires gap=2, got gap={gap}.")

        # Corridor X position (midpoint of the gap)
        corridor_x = left_max_x + 1.0

        # Y range: span both patches (use intersection for safety)
        _, _, left_ymin, left_ymax = p_left._get_bounds()
        _, _, right_ymin, right_ymax = p_right._get_bounds()
        y_min = max(left_ymin, right_ymin)
        y_max = min(left_ymax, right_ymax)

        # ── Step 2: Register corridor qubits ─────────────────────────────────
        # Role is determined by parity: same parity as data → data, else → syndrome_z.
        # For unrotated SC: data qubits are at even+even or odd+odd integer coords.
        ref_data_x, ref_data_y = p_left.data_coords[0]
        y = y_min
        while y <= y_max + 1e-9:
            dx = corridor_x - ref_data_x
            dy = y - ref_data_y
            is_data = (int(round(dx)) % 2 == 0 and int(round(dy)) % 2 == 0) or \
                      (int(round(dx)) % 2 == 1 and int(round(dy)) % 2 == 1)
            if is_data:
                coupler_patch.add_qubit(corridor_x, y, role='data')
            else:
                coupler_patch.add_qubit(corridor_x, y, role='syndrome_z')
            y += 1.0

        # ── Step 3: Build stabilizers for corridor syndrome qubits ────────────
        for uid in coupler_patch.syndrome_indices_z:
            sx, sy = coupler_patch.qubit_coords[uid]
            neighbors = self._find_data_neighbors(coupler_patch, [p_left, p_right], sx, sy)
            if neighbors:
                coupler_patch.stabilizers.append({
                    "pauli":     {coord: "Z" for coord in neighbors},
                    "type":      "Z",
                    "syn_coord": (sx, sy),
                })

        # ── Step 4: Redefine boundary syndrome qubits ─────────────────────────
        # Left boundary: Z-syndrome qubits on p_left at x == left_max_x
        # Right boundary: Z-syndrome qubits on p_right at x == right_min_x
        for patch in [p_left, p_right]:
            for coord in patch.syndrome_coords:
                sx, sy = coord
                at_boundary = (math.isclose(sx, left_max_x, abs_tol=0.1) or
                               math.isclose(sx, right_min_x, abs_tol=0.1))
                if not at_boundary:
                    continue
                if coord not in patch.syndrome_coords_z:  # type: ignore[attr-defined]
                    # guard: only redefine Z-type boundary syndrome qubits
                    if not any(math.isclose(sy, c[1]) and
                               coord in getattr(patch, 'syndrome_coords_z', [])
                               for c in patch.syndrome_coords_z if math.isclose(c[0], sx)):
                        continue

                neighbors = self._find_data_neighbors(coupler_patch, [p_left, p_right], sx, sy)
                if len(neighbors) > 2:  # more than original weight → joint stabilizer formed
                    coupler_patch.stabilizers.append({
                        "pauli":     {c: "Z" for c in neighbors},
                        "type":      "Z",
                        "syn_coord": (sx, sy),
                    })
                    coupler_patch.conflicting_stabilizer_coords.add((sx, sy))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _order_patches(patches):
        bounds_a = patches[0]._get_bounds()
        bounds_b = patches[1]._get_bounds()
        if bounds_a[0] <= bounds_b[0]:
            return patches[0], patches[1]
        return patches[1], patches[0]

    @staticmethod
    def _find_data_neighbors(coupler_patch, other_patches, sx, sy):
        """Find all data qubits at distance 1 from (sx, sy) across all patches."""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            tx, ty = sx + dx, sy + dy
            # Check corridor
            if (tx, ty) in coupler_patch.index_map:
                uid = coupler_patch.index_map[(tx, ty)]
                if uid in coupler_patch.data_indices:
                    neighbors.append((tx, ty))
                    continue
            # Check neighboring patches
            for p in other_patches:
                if (tx, ty) in p.index_map:
                    uid = p.index_map[(tx, ty)]
                    if uid in p.data_indices:
                        neighbors.append((tx, ty))
                        break
        return neighbors


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Use the coupler in a circuit
# ══════════════════════════════════════════════════════════════════════════════

D = 3
P = 1e-3
ROUNDS = D

# Build system
system = QECSystem()
p_ctrl = system.add_patch(UnrotatedSurfaceCode(distance=D), name="ctrl", offset=(0, 0))
# Gap=2: right patch starts at x = left_max_x + 2 = (2*D) + 2
left_max_x = p_ctrl._get_bounds()[1] if hasattr(p_ctrl, '_get_bounds') else 2*D
p_tgt = system.add_patch(UnrotatedSurfaceCode(distance=D), name="tgt",
                          offset=(left_max_x + 2, 0))

# Register coupler
protocol = LinearZZCoupler()
system.register_coupler(protocol, ["ctrl", "tgt"], name="zz_coupler")
print(f"System after coupler registration: {system.num_qubits} qubits")

# Build circuit
tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
builder = CircuitBuilder(tracker, system)

def make_se():
    return UnrotatedSurfaceCodeExtractionBlock(system)

builder.write_coordinates()

# Initialize both patches in |0⟩
patch_data = {q: "Z" for q in system.data_indices
              if system.index_to_owner_map.get(q) in ("ctrl", "tgt")}
builder.initialize(patch_data, n=system.num_qubits)
builder.apply_syndrome_extraction(make_se().circuit, rounds=ROUNDS)

# Activate coupler
builder.activate_coupler("zz_coupler")
cp = system.coupler_patches["zz_coupler"]
cp_data = sorted(system.local_to_global_map["zz_coupler"][q] for q in cp.data_indices)
builder.initialize({q: "X" for q in cp_data}, n=system.num_qubits)
builder.apply_syndrome_extraction(make_se().circuit, rounds=1)
builder.apply_data_readout({q: "X" for q in cp_data})
builder.deactivate_coupler("zz_coupler")

builder.apply_syndrome_extraction(make_se().circuit, rounds=ROUNDS)
builder.apply_data_readout({q: "Z" for q in patch_data.keys()})

# Noiseless check
dets, obs = builder.circuit.compile_detector_sampler().sample(100, separate_observables=True)
ok = not dets.any() and not obs.any()
print(f"Noiseless check: {'PASS' if ok else 'FAIL'}")
print(f"Circuit: {builder.circuit.num_qubits}q "
      f"{builder.circuit.num_detectors}det {builder.circuit.num_observables}obs")

# Simulate
noisy = builder.build_noisy_circuit(NoiseConfig(p_2q=P, p_meas=P), "circuit_level")
stats = SimulationPipeline(
    decoder_config=DecoderConfig("pymatching"),
    max_errors=100, max_shots=300_000, print_progress=False
).run(noisy)
print(f"LER = {stats.logical_error_rate:.3e} ± {stats.ler_error_bar():.1e}")
