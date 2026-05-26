---
name: extend-new-code
description: >
  Add a new QEC code to LightStim by implementing QECPatch and a syndrome
  extraction block. Use this skill whenever the user asks to implement a new
  quantum error correcting code, extend LightStim with a custom code family,
  define stabilizers and logical operators for a new code, create a new
  SE_block, or understand the minimal interface a code needs to satisfy.
user-invocable: true
---

# Extend with a New QEC Code

Every LightStim code requires two classes:

1. **`QECPatch` subclass** — geometry (qubit positions) + physics (stabilizers + logicals)
2. **Extraction block** — builds one noiseless SE round as a `stim.Circuit`

## Part 1: `QECPatch` subclass

### The three build phases

```python
from lightstim.ir.qec_patch import QECPatch

class MyCode(QECPatch):
    def _process_params(self):
        self.distance = self.params['distance']  # validate params here

    def build(self):
        d = self.distance

        # Phase 1: geometry — register every qubit at its (x, y) coordinate
        self.add_qubit(x, y, role='data')         # data qubit
        self.add_qubit(x, y, role='syndrome_z')   # Z-ancilla
        self.add_qubit(x, y, role='syndrome_x')   # X-ancilla

        # Phase 2: physics — one stabilizer per ancilla
        # target_dict: {(x, y): 'X'|'Z'|'Y'} for each data qubit in the stabilizer support
        self.create_stim_stabilizer(
            target_dict={(x1, y1): 'Z', (x2, y2): 'Z'},
            syn_coord=(sx, sy),   # coordinate of the ancilla qubit
            type='Z',             # 'X' or 'Z' (or 'Mixed' for YY etc.)
        )

        # Phase 3: logical operators (must register both X and Z representatives)
        self.create_stim_logical({(x, y): 'Z', ...}, op_type='Z')
        self.create_stim_logical({(x, y): 'X', ...}, op_type='X')
        self.num_logicals = 1     # k = number of encoded logical qubits
```

### Key invariants

- `add_qubit` role must be one of `'data'`, `'syndrome_x'`, `'syndrome_z'`.
  Both `syndrome_x` and `syndrome_z` populate `syndrome_indices_x` / `syndrome_indices_z`
  — the SE block uses these to emit Hadamard gates and distinguish CX vs CZ layers.
- `create_stim_stabilizer` looks up `syn_coord` via `self.index_map` — register
  the ancilla with `add_qubit` BEFORE calling it.
- `create_stim_logical` only takes data qubit coordinates — syndrome coords are ignored.
- You MUST set `self.num_logicals` in `build()`.

### CSS vs non-CSS

For CSS codes (most common: SC, BB, toric), stabilizers are either all-X or all-Z.
Use `type='X'` or `type='Z'` respectively. The SE block will separate them.

For non-CSS codes with Y stabilizers, use `type='Mixed'` and handle gate decomposition
in the SE block manually.

---

## Part 2: Syndrome Extraction block

The SE block generates exactly **one noiseless SE round** as a `stim.Circuit`.
Noise is injected later by `builder.build_noisy_circuit()` — never put noise here.

### Structure

```python
import stim

class MyCodeExtractionBlock:
    def __init__(self, system):
        self.system = system
        self.circuit = stim.Circuit()
        self._build()

    def _build(self):
        c = self.circuit

        # 1. Reset all active syndrome qubits (both X and Z ancillas)
        syn_z = sorted(self.system.active_syndrome_indices_z)
        syn_x = sorted(self.system.active_syndrome_indices_x)
        c.append("R",  syn_z)
        c.append("RX", syn_x)

        # 2. TICK with SE_start tag — marks the idle window for noise injection
        c.append("TICK", tag="SE_start")

        # 3. Gate layers — one TICK between each layer
        # For Z-stabilizers: CX(data → ancilla)  [ancilla is "target" of CX]
        # For X-stabilizers: CX(ancilla → data)  [ancilla is "control" of CX]
        #
        # Ordering matters for hook errors — follow the code's canonical gate schedule.
        # Read system.active_stabilizers_z / _x and system.index_map to get global indices.
        for stab in self.system.active_stabilizers_z:
            for data_idx in stab['data_indices']:
                c.append("CX", [data_idx, stab['syn_idx']])
        c.append("TICK")

        for stab in self.system.active_stabilizers_x:
            for data_idx in stab['data_indices']:
                c.append("CX", [stab['syn_idx'], data_idx])
        c.append("TICK")

        # 4. Basis change for X-ancillas: H before measure
        if syn_x:
            c.append("H", syn_x)
            c.append("TICK")

        # 5. Measure — MUST be the LAST instruction (CircuitBuilder analyzes it)
        #    Z-ancillas measured in Z basis (M), X-ancillas in X basis (MX)
        if syn_z: c.append("M",  syn_z)
        if syn_x: c.append("MX", syn_x)
```

### Critical constraints on the SE block

1. **The last instruction must be `M` or `MX`** on syndrome qubits.
   `CircuitBuilder._get_back_propagated_pauli()` reads this last instruction to determine
   the measurement basis and which qubits were measured.

2. **Gate ordering within one SE round affects hook errors.** For an unrotated SC, the
   canonical order is: North → East → West → South neighbors. For your code, pick an
   order and be consistent across all rounds.

3. **Use `system.active_stabilizers_z` / `_x` not `system.stabilizers`.**
   When couplers are active, `active_stabilizers` includes coupler stabilizers.
   The SE block must work correctly in both coupled and uncoupled regimes.

4. **Use global indices from `system.index_map[(x, y)]` or `stab['syn_idx']`.**
   Never use local patch indices in the SE block circuit.

5. **No noise in the SE block.** The noise injector handles it via the `SE_start` tag.

---

## File layout

```
lightstim/qec_code/<your-code>/
├── __init__.py        # export patch class + extraction block
├── code_patch.py      # QECPatch subclass
└── SE_block.py        # extraction block class
```

---

## Verification sequence

After writing both classes, verify in this order:

```python
# 1. Patch builds without errors
patch = MyCode(distance=3)
print(patch.num_qubits, len(patch.stabilizers), patch.num_logicals)

# 2. System registration
system = QECSystem()
system.add_patch(patch, name="p")

# 3. SE block builds without errors
se = MyCodeExtractionBlock(system)

# 4. Full noiseless circuit (use builder directly)
from lightstim.ir.tracker import SyndromeTracker
from lightstim.ir.builder import CircuitBuilder
tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
builder = CircuitBuilder(tracker, system)
builder.write_coordinates()
builder.initialize({q: "Z" for q in system.data_indices}, n=system.num_qubits)
builder.apply_syndrome_extraction(se.circuit, rounds=3)
builder.apply_data_readout({q: "Z" for q in system.data_indices})

# 5. Must see 0 detection events and 0 observable flips
dets, obs = builder.circuit.compile_detector_sampler().sample(100, separate_observables=True)
assert not dets.any(),  "Noiseless circuit fires detectors — stabilizer or SE bug"
assert not obs.any(),   "Noiseless circuit flips observable — logical operator bug"
```

## Reference script

Read `scripts/template.py` for a complete `BitFlipStrip` repetition-code example
that you can run directly to verify the pattern works end-to-end.
