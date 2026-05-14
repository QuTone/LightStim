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

1. **`QECPatch` subclass** — defines geometry (qubit positions) and physics
   (stabilizers + logicals). Lives in `lightstim/qec_code/<your-code>/code_patch.py`.

2. **Extraction block** — generates one noiseless SE round as a `stim.Circuit`.
   Lives in `lightstim/qec_code/<your-code>/SE_block.py`.

## How to help the user

1. Ask: what is the code structure (topology, stabilizer types, distance)?
2. Read `scripts/template.py` — it implements a complete minimal example
   (BitFlipStrip, a repetition-code variant) from scratch.
3. Walk through the three build phases from the template:
   - **Phase 1**: register qubits with `self.add_qubit(x, y, role)`
   - **Phase 2**: register stabilizers with `self.create_stim_stabilizer(target_dict, syn_coord, type)`
   - **Phase 3**: register logicals with `self.create_stim_logical(target_dict, op_type)` + set `self.num_logicals`
4. For the SE block: Reset syndrome qubits → TICK(SE_start) → CX/CZ layers →
   TICK between each layer → Measure syndrome qubits.
5. After writing both classes, verify with a noiseless memory experiment
   (0 detection events = correct build).

## QECPatch API

```python
# Register a qubit at (x, y) with role 'data', 'syndrome_z', or 'syndrome_x'
self.add_qubit(x, y, role='data')

# Register a stabilizer: target_dict maps (x,y) coords → Pauli ('X','Z','Y')
# syn_coord is the syndrome qubit coord; type is 'X' or 'Z'
self.create_stim_stabilizer({'target_coord': 'Z', ...}, syn_coord=(sx, sy), type='Z')

# Register a logical operator: op_type is 'X' or 'Z'
self.create_stim_logical({'data_coord': 'Z', ...}, op_type='Z')
self.num_logicals = 1
```

## Key invariants

- All coordinates are `(float, float)` — use integers for simplicity
- Syndrome qubits must be in `self.syndrome_coords` (set by `add_qubit` with role `syndrome_*`)
- The SE block reads `system.active_stabilizers_z` / `system.active_stabilizers_x`
  and `system.index_map[(x,y)]` for global qubit indices

## File layout

```
lightstim/qec_code/<your-code>/
├── __init__.py
├── code_patch.py   ← QECPatch subclass
└── SE_block.py     ← extraction block
```

## Reference script

Read `scripts/template.py` for the complete BitFlipStrip example, which you can
run directly to verify the pattern works end-to-end.
