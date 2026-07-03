# Rotated Bent (XZ) Joint-Measurement Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A parameterized generator `build_rotated_bent_xz_layout([p1, p2], …)` that auto-builds the rotated bent (XZ) joint-measurement layout (data, CSS + mixed checks, boundary trim/replace, readout chain, no-MPP gate-level SE circuit) and self-verifies — reproducing the validated d=3 golden exactly and scaling to d=5,7.

**Architecture:** A construction pipeline in `rotated/bent_layout.py`: place two logical patches (reuse `RotatedSurfaceCode` + `shift`/`transpose_coords`), infer the L-shaped bus connector, emit bus CSS checks, trim/replace patch↔bus boundaries (CSS-fuse or mixed-domain-wall), truncate the inner bend corner, GF(2)-solve the readout chain, and emit the SE circuit via a generalized `RotatedBentJointMeasurement`. The d=3 golden is a frozen pytest fixture; the exact generalization rules are *derived to pass* the golden regression and *gated* by `.verify()` at d=5,7.

**Tech Stack:** Python, `stim`, `pymatching`, `numpy`; LightStim `RotatedSurfaceCode`, `QECPatch` helpers (`shift_coords`/`transpose_coords`/`create_stim_stabilizer`/`index_map`/`grid_map`/`get_grid_key`/`snap_coord`), `RotatedTwoPatchCoupler` conflict machinery, `protocols.routed_multi_patch_ls.solve_linear_decomposition`/`logical_pauli_product_vector`, existing `bent_joint_se.RotatedBentJointMeasurement`. Tests via `pytest`; run with the `light_stim` conda env (`/home/yuehan/miniconda3/envs/light_stim/bin/python -m pytest`).

## Global Constraints

- **Reuse the library convention; do NOT invent coordinates.** Data on `(odd,odd)`, ancillas `(even,even)`, spacing 2, corner data `(1,1)`; placement via `shift`; default X̄ vertical (x=1) / Z̄ horizontal (y=1), orientation flipped via `transpose_coords()`; coord↔index via `index_map`/`grid_map`/`snap_coord`/`get_grid_key`.
- **`PatchSpec` describes only the two logical patches** (p1=X, p2=Z). The bus/bend/routing region is NOT a `PatchSpec`.
- **Generator trims patch bus-facing boundaries** (delete/replace near-seam weight-2 checks); the seam/bus generator decides which checks are deleted vs replaced by mixed (XZ) checks.
- **d=3 golden is the regression oracle**; generated layout must equal it exactly (data, checks, x_logical, z_logical, readout_chain, circuit).
- **Check dict schema** (coord-keyed): `{'syn': (col,row), 'type': 'X'|'Z'|'M', 'pauli': {(col,row): 'X'|'Z'}, 'corners': sorted([...])}`. `type='M'` ⇒ pauli has both X and Z values but **no qubit carries both** (no twist/Y).
- **Eight acceptance checks** must pass: (1) all commute; (2) joint X̄₁·Z̄₂ in span; (3) single X̄₁ and single Z̄₂ NOT in span; (4) no Y/no twist; (5) `#data−rank==1`; (6) DEM valid + noiseless-deterministic; (7) no `MPP`; (8) no tick collision. Confirm the measured joint with `peek_observable_expectation`.
- **No `MPP`** anywhere; SE is gate-level (CNOT/CZ/H), one clean op-type per tick.
- **Commits deferred** — per the user's workflow, commit everything together at the end with results. Skip the per-task `git commit` until then.
- Spec: `docs/superpowers/specs/2026-06-26-rotated-bent-xz-generator-design.md`.

## File Structure

| File | Responsibility |
|---|---|
| `lightstim/qec_code/surface_code/rotated/bent_layout.py` (new) | `PatchSpec`, `BentLayout`, `build_rotated_bent_xz_layout`; the construction pipeline + `.verify()`. |
| `lightstim/qec_code/surface_code/rotated/bent_joint_se.py` (modify) | accept a `BentLayout` (or its `.data`/`.checks`/`.x_logical`); already coord-dict based. |
| `lightstim/qec_code/surface_code/rotated/__init__.py` (modify) | export `PatchSpec`, `build_rotated_bent_xz_layout`, `BentLayout`. |
| `tests/fixtures/bent_xz_golden_d3.json` (new) | frozen golden d=3 layout (data + checks + logicals + readout). |
| `tests/test_rotated_bent_layout.py` (new) | golden regression + 8 checks at d=3,5,7 + circuit equality at d=3. |
| `notebooks/LogicalOps/rotated_bent_XZ_LS.ipynb` (modify, last) | cell 1 → `layout = build_rotated_bent_xz_layout([p1,p2])`; keep viz/acceptance/detslice; LER over d=3,5,7. |

Run tests: `cd /nvme2n1/yuehan_zhang/resource_analsis/LightStim && /home/yuehan/miniconda3/envs/light_stim/bin/python -m pytest tests/test_rotated_bent_layout.py -q`

---

### Task 1: Golden fixture + acceptance-check helpers (the oracle)

Build the frozen d=3 golden and the reusable check functions first — everything downstream is gated by these.

**Files:**
- Create: `tests/fixtures/bent_xz_golden_d3.json`
- Create: `tests/test_rotated_bent_layout.py`

**Interfaces:**
- Produces: `GOLDEN` (dict: `data`, `checks`, `x_logical`, `z_logical`, `readout_chain`); helpers `symplectic(checks, data)`, `gf2_rank`, `in_span`, `commuting_pairs`, `has_twist`, `acceptance(data, checks, x_logical, z_logical)` returning the 8-check dict.

- [ ] **Step 1: Write the golden fixture JSON** (transcribe from the current notebook cell 1 — the validated table)

```json
{
  "distance": 3,
  "data": [[1,3],[1,5],[1,7],[1,9],[1,11],[3,1],[3,3],[3,5],[3,7],[3,9],[3,11],[5,1],[5,3],[5,5],[5,7],[5,9],[5,11],[7,1],[7,3],[7,5],[9,1],[9,3],[9,5]],
  "checks": [
    {"syn":[4,0],"type":"Z","pauli":{"3,1":"Z","5,1":"Z"}},
    {"syn":[8,0],"type":"Z","pauli":{"7,1":"Z","9,1":"Z"}},
    {"syn":[2,2],"type":"Z","pauli":{"1,3":"Z","3,1":"Z","3,3":"Z"}},
    {"syn":[4,2],"type":"X","pauli":{"3,1":"X","3,3":"X","5,1":"X","5,3":"X"}},
    {"syn":[6,2],"type":"Z","pauli":{"5,1":"Z","5,3":"Z","7,1":"Z","7,3":"Z"}},
    {"syn":[8,2],"type":"X","pauli":{"7,1":"X","7,3":"X","9,1":"X","9,3":"X"}},
    {"syn":[0,4],"type":"Z","pauli":{"1,3":"Z","1,5":"Z"}},
    {"syn":[2,4],"type":"X","pauli":{"1,3":"X","1,5":"X","3,3":"X","3,5":"X"}},
    {"syn":[4,4],"type":"Z","pauli":{"3,3":"Z","3,5":"Z","5,3":"Z","5,5":"Z"}},
    {"syn":[6,4],"type":"X","pauli":{"5,3":"X","5,5":"X","7,3":"X","7,5":"X"}},
    {"syn":[8,4],"type":"Z","pauli":{"7,3":"Z","7,5":"Z","9,3":"Z","9,5":"Z"}},
    {"syn":[10,4],"type":"X","pauli":{"9,3":"X","9,5":"X"}},
    {"syn":[2,6],"type":"M","pauli":{"1,5":"Z","1,7":"X","3,5":"Z","3,7":"X"}},
    {"syn":[4,6],"type":"M","pauli":{"3,5":"X","3,7":"Z","5,5":"X","5,7":"Z"}},
    {"syn":[6,6],"type":"M","pauli":{"5,5":"Z","5,7":"X","7,5":"Z"}},
    {"syn":[0,8],"type":"X","pauli":{"1,7":"X","1,9":"X"}},
    {"syn":[2,8],"type":"Z","pauli":{"1,7":"Z","1,9":"Z","3,7":"Z","3,9":"Z"}},
    {"syn":[4,8],"type":"X","pauli":{"3,7":"X","3,9":"X","5,7":"X","5,9":"X"}},
    {"syn":[2,10],"type":"X","pauli":{"1,9":"X","1,11":"X","3,9":"X","3,11":"X"}},
    {"syn":[4,10],"type":"Z","pauli":{"3,9":"Z","3,11":"Z","5,9":"Z","5,11":"Z"}},
    {"syn":[6,10],"type":"X","pauli":{"5,9":"X","5,11":"X"}},
    {"syn":[2,12],"type":"Z","pauli":{"1,11":"Z","3,11":"Z"}}
  ],
  "x_logical": [[1,7],[3,7],[5,7]],
  "z_logical": [[7,1],[7,3],[7,5]],
  "readout_chain": [[0,4],[2,2],[2,6],[4,0],[4,4],[6,2],[6,6]]
}
```

- [ ] **Step 2: Write the helpers + a self-test that the golden itself passes all 8 checks**

```python
import json, itertools, numpy as np, stim, pathlib

FIX = pathlib.Path(__file__).parent / "fixtures" / "bent_xz_golden_d3.json"

def load_golden():
    g = json.load(open(FIX))
    data = [tuple(c) for c in g["data"]]
    checks = []
    for ch in g["checks"]:
        pauli = {tuple(int(v) for v in k.split(",")): p for k, p in ch["pauli"].items()}
        checks.append({"syn": tuple(ch["syn"]), "type": ch["type"], "pauli": pauli,
                       "corners": sorted(pauli)})
    return dict(distance=g["distance"], data=data, checks=checks,
               x_logical=[tuple(c) for c in g["x_logical"]],
               z_logical=[tuple(c) for c in g["z_logical"]],
               readout_chain={tuple(c) for c in g["readout_chain"]})

def symplectic(checks, data):
    idx = {c: i for i, c in enumerate(sorted(data))}; n = len(idx)
    rows = []
    for ch in checks:
        v = np.zeros(2 * n, np.uint8)
        for c, P in ch["pauli"].items():
            if P in ("X", "Y"): v[idx[c]] ^= 1
            if P in ("Z", "Y"): v[n + idx[c]] ^= 1
        rows.append(v)
    return np.array(rows, np.uint8), idx, n

def gf2_rank(rows):
    M = np.array([r.copy() for r in rows], np.uint8); r = 0
    for c in range(M.shape[1]):
        piv = next((k for k in range(r, len(M)) if M[k, c]), None)
        if piv is None: continue
        M[[r, piv]] = M[[piv, r]]
        for k in range(len(M)):
            if k != r and M[k, c]: M[k] ^= M[r]
        r += 1
    return r

def in_span(rows, t): return gf2_rank(list(rows) + [t]) == gf2_rank(list(rows))

def _vec(support_pauli, idx, n):
    v = np.zeros(2 * n, np.uint8)
    for c, P in support_pauli.items():
        if P in ("X", "Y"): v[idx[c]] ^= 1
        if P in ("Z", "Y"): v[n + idx[c]] ^= 1
    return v

def acceptance(data, checks, x_logical, z_logical):
    S, idx, n = symplectic(checks, data)
    comm = lambda a, b: int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0
    bad = sum(1 for i in range(len(S)) for j in range(i+1, len(S)) if not comm(S[i], S[j]))
    twist = any(P == "Y" for ch in checks for P in ch["pauli"].values())
    rank = gf2_rank(list(S))
    vXZ = _vec({**{c:"X" for c in x_logical}, **{c:"Z" for c in z_logical}}, idx, n)
    vX = _vec({c:"X" for c in x_logical}, idx, n); vZ = _vec({c:"Z" for c in z_logical}, idx, n)
    return dict(commute=bad == 0, joint=in_span(S, vXZ),
                no_single=not in_span(S, vX) and not in_span(S, vZ),
                no_twist=not twist, one_logical=len(data) - rank == 1,
                n_mixed=sum(c["type"] == "M" for c in checks))

def test_golden_passes_acceptance():
    g = load_golden()
    a = acceptance(g["data"], g["checks"], g["x_logical"], g["z_logical"])
    assert a == {"commute": True, "joint": True, "no_single": True,
                 "no_twist": True, "one_logical": True, "n_mixed": 3}
```

- [ ] **Step 3: Run; verify the golden self-test passes**

Run: `… -m pytest tests/test_rotated_bent_layout.py::test_golden_passes_acceptance -q`
Expected: PASS. (If it fails, the fixture transcription is wrong — fix the JSON, not the helpers.)

- [ ] **Step 4: Commit** *(deferred)*

---

### Task 2: `PatchSpec` + single-patch placement

**Files:**
- Create: `lightstim/qec_code/surface_code/rotated/bent_layout.py`
- Modify: `tests/test_rotated_bent_layout.py`

**Interfaces:**
- Produces: `PatchSpec(name, origin, distance, measured_logical, orientation)`; `place_patch(spec) -> dict(data=set[(col,row)], checks=list[checkdict], x_support=[...], z_support=[...])` where checks are coord-keyed CSS dicts and `*_support` are the logical supports in the placed/oriented frame.

- [ ] **Step 1: Write the failing test** — a placed X_horizontal d=3 patch at origin (1,7) has the expected data + horizontal X̄ on its bottom row

```python
from lightstim.qec_code.surface_code.rotated.bent_layout import PatchSpec, place_patch

def test_place_patch_xhorizontal_d3():
    p = place_patch(PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"))
    # full 3x3 data block, corner at (1,7)
    assert p["data"] == {(c, r) for c in (1,3,5) for r in (7,9,11)}
    # X_horizontal => X-bar is a horizontal row; bus-facing (bottom) row 7
    assert sorted(p["x_support"]) == [(1,7),(3,7),(5,7)]
    # every check is pure CSS, weight 2 or 4, data on odd coords
    assert all(ch["type"] in ("X","Z") and len(ch["pauli"]) in (2,4) for ch in p["checks"])
```

- [ ] **Step 2: Run to verify it fails** (`ModuleNotFoundError`/`AttributeError`).

- [ ] **Step 3: Implement `PatchSpec` + `place_patch`**

Use `RotatedSurfaceCode(distance=d, shift=(origin[0]-1, origin[1]-1))` (so the library corner (1,1) lands at `origin`). For `orientation == "X_horizontal"` apply `transpose_coords()` (default X̄ is vertical). Extract `qubit_coords`/`data_indices`→data coords; convert each stabilizer to the coord-keyed dict schema (`{'syn','type','pauli','corners'}`) via `qubit_coords`. Read X̄/Z̄ supports from the patch's `logical_ops`. Pick the `*_support` on the **bus-facing** boundary (the boundary nearest the other patch; for "auto" this is decided in Task 4 — here just expose both the full logical and its boundary rows/cols).

```python
from dataclasses import dataclass
from ..rotated import RotatedSurfaceCode   # adjust import to avoid cycle; use direct module import

@dataclass(frozen=True)
class PatchSpec:
    name: str; origin: tuple; distance: int; measured_logical: str; orientation: str

def place_patch(spec):
    code = RotatedSurfaceCode(distance=spec.distance,
                              shift=(spec.origin[0]-1, spec.origin[1]-1))
    if spec.orientation == "X_horizontal":
        code.transpose_coords()
    qc = code.qubit_coords
    data = {tuple(qc[i]) for i in code.data_indices}
    checks = []
    for s in code.stabilizers:
        pauli = {tuple(qc[q]): P for q, P in s["pauli"].items()}
        checks.append({"syn": tuple(s["syn_coord"]), "type": s["type"],
                       "pauli": pauli, "corners": sorted(pauli)})
    xs = [tuple(qc[q]) for q in code.logical_ops_x["pauli"]]
    zs = [tuple(qc[q]) for q in code.logical_ops_z["pauli"]]
    return dict(data=data, checks=checks, x_support=xs, z_support=zs)
```
(Adjust attribute names to the real API found in Task-0 context: `logical_ops_x`/`logical_ops_z` or `logical_ops`; `transpose_coords` semantics. The test pins the expected result.)

- [ ] **Step 4: Run until the test passes.** Fix the shift/transpose/attribute details until `place_patch` yields exactly the asserted data + horizontal X̄. Add the symmetric test for p2 `(7,1,"Z","X_horizontal")` (Z̄ vertical on col 7).

- [ ] **Step 5: Commit** *(deferred)*

---

### Task 3: Bus inference + bus CSS checks

**Files:**
- Modify: `lightstim/qec_code/surface_code/rotated/bent_layout.py`
- Modify: `tests/test_rotated_bent_layout.py`

**Interfaces:**
- Produces: `infer_bus(p1_placed, p2_placed, bend="auto") -> dict(data=set, checks=list)` — the L-connector data + its CSS plaquettes (rotated convention), before trimming/seam.

- [ ] **Step 1: Write the failing test** — at d=3 the bus fills the lower-left block (cols 1-5, rows 1-5) and its data is exactly the golden's bus data (the lower-left region, including (1,1) at this pre-truncation stage)

```python
def test_infer_bus_d3_region():
    p1 = place_patch(PatchSpec("p1",(1,7),3,"X","X_horizontal"))
    p2 = place_patch(PatchSpec("p2",(7,1),3,"Z","X_horizontal"))
    bus = infer_bus(p1, p2, bend="auto")
    # bus is the L-connector lower-left block cols 1-5 rows 1-5 (pre-corner-cut)
    assert {(c,r) for c in (1,3,5) for r in (1,3,5)} <= bus["data"]
    assert all(ch["type"] in ("X","Z") for ch in bus["checks"])
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `infer_bus`** — derive the L-region geometry from the two placed patches' bounding boxes and bus-facing boundaries; build a `RotatedSurfaceCode` (or direct plaquette tiling) over the connector rectangle(s) and emit CSS checks in the coord schema. *The exact region + CSS pattern is derived to satisfy the d=3 region test and, downstream, the golden regression in Task 4.*

- [ ] **Step 4: Run until the bus-region test passes.**

- [ ] **Step 5: Commit** *(deferred)*

---

### Task 4: Assemble — trim/replace boundaries, mixed seam, corner cut → golden regression

The novel core. Combine placed patches + bus, trim bus-facing boundaries, emit the mixed domain wall, CSS-fuse the CSS junction, truncate the inner corner. **Gated by exact golden equality.**

**Files:**
- Modify: `lightstim/qec_code/surface_code/rotated/bent_layout.py`
- Modify: `tests/test_rotated_bent_layout.py`

**Interfaces:**
- Produces: `build_rotated_bent_xz_layout(patches, bend="auto", joint_type="XZ", readout_rule="auto") -> BentLayout` with `.data`, `.checks`, `.x_logical`, `.z_logical` (readout in Task 5). `BentLayout` is a dataclass.

- [ ] **Step 1: Write the failing golden-regression test** (canonicalize so order/representation don't matter)

```python
def _canon(checks):
    return sorted((c["type"], tuple(sorted(c["pauli"].items()))) for c in checks)

def test_generator_matches_golden_d3_layout():
    g = load_golden()
    lay = build_rotated_bent_xz_layout(
        [PatchSpec("p1",(1,7),3,"X","X_horizontal"),
         PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
    assert set(lay.data) == set(g["data"])
    assert _canon(lay.checks) == _canon(g["checks"])
    assert sorted(lay.x_logical) == sorted(g["x_logical"])
    assert sorted(lay.z_logical) == sorted(g["z_logical"])
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement the assembly** — pipeline stages 4–5 of the spec:
  - merge p1, p2, bus data/checks;
  - at each patch↔bus interface classify CSS-junction vs mixed-domain-wall (from the joint geometry: matching boundary types ⇒ CSS fuse; X-side meets Z-side ⇒ mixed wall);
  - CSS-fuse via the coupler's `syn_coord`-keyed conflict/replace pattern; emit `d` mixed checks with uniform-Pauli-per-side **alternating by seam index** (even j → patch-side Z / bus-side X; odd j → patch-side X / bus-side Z), end check weight-3;
  - truncate the inner bend corner (drop that data + its corner X-check; weight-3 the corner Z-plaquette).
  *Derive the exact trim set + seam placement + corner rule until `_canon(lay.checks) == _canon(golden)`.*

- [ ] **Step 4: Run until the golden-regression test passes exactly.** This is the gate that pins all trim/seam/corner rules. Do not weaken the assertion.

- [ ] **Step 5: Commit** *(deferred)*

---

### Task 5: Readout-chain auto + `BentLayout.verify()` + SE circuit

**Files:**
- Modify: `lightstim/qec_code/surface_code/rotated/bent_layout.py`
- Modify: `lightstim/qec_code/surface_code/rotated/bent_joint_se.py`
- Modify: `tests/test_rotated_bent_layout.py`

**Interfaces:**
- Consumes: Task-4 `BentLayout`; `RotatedBentJointMeasurement`; `solve_linear_decomposition`/`logical_pauli_product_vector`.
- Produces: `BentLayout.readout_chain: set`, `BentLayout.verify() -> dict` (8 checks), `BentLayout.build_circuit(rounds, p) -> stim.Circuit`.

- [ ] **Step 1: Write the failing tests** (readout matches golden; verify all True; circuit matches golden counts)

```python
def test_readout_chain_matches_golden_d3():
    lay = build_rotated_bent_xz_layout([PatchSpec("p1",(1,7),3,"X","X_horizontal"),
                                        PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
    assert set(lay.readout_chain) == load_golden()["readout_chain"]

def test_verify_all_pass_d3():
    lay = build_rotated_bent_xz_layout([PatchSpec("p1",(1,7),3,"X","X_horizontal"),
                                        PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
    v = lay.verify()
    assert all(v.values()), v

def test_circuit_matches_golden_d3():
    lay = build_rotated_bent_xz_layout([PatchSpec("p1",(1,7),3,"X","X_horizontal"),
                                        PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
    c = lay.build_circuit(rounds=3, p=0.0)
    assert (c.num_qubits, c.num_detectors, c.num_observables) == (45, 62, 1)
    assert "MPP" not in str(c)               # no MPP
    det, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not det.any() and not obs.any()   # noiseless-deterministic
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**
  - `readout_chain`: GF(2)-solve for the stabilizer subset whose symplectic product equals `_vec({X on x_logical, Z on z_logical})`; pick minimal-weight canonical solution. (Reuse `solve_linear_decomposition`.)
  - `verify()`: the `acceptance(...)` 6 algebraic checks **plus** DEM-valid + noiseless-deterministic (from `build_circuit`), no-`MPP`, and no-tick-collision (scan the circuit: no qubit appears twice between consecutive `TICK`s). Add `peek_observable_expectation` confirmation that the merge measures X̄₁·Z̄₂ and neither single.
  - `build_circuit`: call `RotatedBentJointMeasurement(lay.data, lay.checks, lay.x_logical).circuit(rounds, p)`; generalize that class to accept a `BentLayout` directly if convenient (keep behavior identical).

- [ ] **Step 4: Run until all three tests pass.**

- [ ] **Step 5: Commit** *(deferred)*

---

### Task 6: Scale to d=5,7 + swap notebook + LER curves

**Files:**
- Modify: `tests/test_rotated_bent_layout.py`
- Modify: `lightstim/qec_code/surface_code/rotated/__init__.py`
- Modify: `notebooks/LogicalOps/rotated_bent_XZ_LS.ipynb`

**Interfaces:**
- Consumes: the full generator.

- [ ] **Step 1: Write the failing scaling test** (d=5,7 generate + pass all 8 checks; invariants hold)

```python
import pytest
@pytest.mark.parametrize("d", [5, 7])
def test_generator_scales(d):
    p1 = PatchSpec("p1", (1, 2*d+1), d, "X", "X_horizontal")
    p2 = PatchSpec("p2", (2*d+1, 1), d, "Z", "X_horizontal")
    lay = build_rotated_bent_xz_layout([p1, p2])
    v = lay.verify()
    assert all(v.values()), (d, v)
    assert sum(c["type"] == "M" for c in lay.checks) == d        # #mixed == d
    assert len(lay.checks) == len(lay.data) - 1                  # #stab == #data-1
```
(p1/p2 origins for general d follow the d=3 pattern: p1 above the bus at `(1, 2d+1)`, p2 right of the bus at `(2d+1, 1)` — confirm the exact offsets reproduce the bend; adjust to what Task 4's geometry expects.)

- [ ] **Step 2: Run; if d=5/7 fail `.verify()`, the generalization rule is wrong — iterate Tasks 3–5** until both pass. This is the real generalization gate.

- [ ] **Step 3: Export from `__init__.py`**

```python
from .bent_layout import PatchSpec, BentLayout, build_rotated_bent_xz_layout
# add the three names to __all__
```

- [ ] **Step 4: Swap the notebook cell 1 to use the generator** (remove the hand-coded `DATA`/`CHECKS` literals; keep viz/acceptance/detslice cells unchanged since they consume `DATA`/`CHECKS`)

```python
from lightstim.qec_code.surface_code.rotated import PatchSpec, build_rotated_bent_xz_layout
layout = build_rotated_bent_xz_layout(
    [PatchSpec("p1",(1,7),3,"X","X_horizontal"),
     PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
DATA, CHECKS = layout.data, layout.checks
X1, Z2, CHAIN = layout.x_logical, layout.z_logical, layout.readout_chain
for c in CHECKS: c["corners"] = sorted(c["pauli"]); c["id"] = f"{c['type']}@{c['syn']}"
nX = sum(c["type"]=="X" for c in CHECKS); nZ = sum(c["type"]=="Z" for c in CHECKS); nM = sum(c["type"]=="M" for c in CHECKS)
di = {q:i for i,q in enumerate(DATA)}; nq = len(DATA)
print(f"data={nq} checks={len(CHECKS)} (X={nX} Z={nZ} M={nM})  [generated]")
```

- [ ] **Step 5: Replace the LER cell to sweep distance** (d=3,5,7), rebuilding the layout+circuit per d:

```python
def circuit_for(d, p):
    lay = build_rotated_bent_xz_layout([PatchSpec("p1",(1,2*d+1),d,"X","X_horizontal"),
                                        PatchSpec("p2",(2*d+1,1),d,"Z","X_horizontal")])
    return lay.build_circuit(rounds=d, p=p)
DISTANCES = (3,5,7)
# … estimate_ler over circuit_for(d, p) for each d; plot LER vs p, one curve per distance …
```

- [ ] **Step 6: Run the test suite + execute the notebook end-to-end**

Run: `… -m pytest tests/test_rotated_bent_layout.py -q` (all pass) then execute the notebook (via the `light_stim` kernel runner). Expected: generator drives the notebook; d=3 viz/acceptance unchanged; **LER curves show distance suppression** (higher d → lower LER sub-threshold). Final gate.

- [ ] **Step 7: Commit** *(deferred — present results to the user; commit generator + tests + notebook + spec together.)*

---

## Self-Review

**Spec coverage:** PatchSpec/interface → Task 2,4; bus inference → Task 3; trim/replace + mixed seam + corner cut → Task 4; readout auto → Task 5; SE circuit → Task 5; 8 verify checks → Task 1 (helpers) + Task 5 (circuit checks); d=3 golden regression → Task 1 (fixture) + Task 4 (layout) + Task 5 (circuit); d=5,7 + LER → Task 6; library convention reuse → Task 2; file layout → matches spec. ✅ All covered.

**Placeholder scan:** The "derive until the test passes" wording in Tasks 3–4 is intentional and unavoidable — the general bus/seam/corner rule is not known a priori; it is *defined* by the d=3 golden + `.verify()` oracle (the tests are fully written and concrete). All test code and the fixture are complete. No `TODO`/`TBD` left as deliverables. Library attribute names in Task 2 (`logical_ops_x`/`transpose_coords`) are flagged to confirm against the real API during execution, with the test pinning the expected output.

**Type consistency:** `place_patch`→dict(data,checks,x_support,z_support); `infer_bus`→dict(data,checks); `build_rotated_bent_xz_layout`→`BentLayout(.data,.checks,.x_logical,.z_logical,.readout_chain,.verify,.build_circuit)`; check dict schema `{'syn','type','pauli','corners'}` consistent across fixture, helpers, generator, and `bent_joint_se`. `acceptance()` keys (commute/joint/no_single/no_twist/one_logical/n_mixed) consistent with the Task-1 self-test and `verify()`.
