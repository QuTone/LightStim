# Rotated Multi-Patch Joint Measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `build_rotated_multi_patch_joint_layout(patches, routing="auto")` — a multi-patch joint measurement `M(∏ᵢ P̄ᵢ)` over a single **routed bent bus** (≥1 L-bend), starting with a verified 3-patch `M(X̄₁·Z̄₂·X̄₃)` (2 bends, 2 mixed walls), then N.

**Architecture:** A new module `multi_patch_joint.py` that reuses the two-patch primitives in `bent_layout.py` (`place_patch`, `_bent_plaquettes` re-typing, `_select_checks` keep, `_readout_chain`, `_arrange` bent connector, `BentLayoutError`). New surface: chain per-consecutive-pair bent connectors into a multi-bend bus, re-type the global X-side (all X̄ patches + X-typed bus segments), and verify the joint. The exact bus/wall geometry is derived against `verify()` (not hand-tuned), exactly as the two-patch boundary rules were.

**Tech Stack:** Python, `stim`, `pymatching`, `numpy`; LightStim rotated `bent_layout` primitives + `RotatedBentJointMeasurement`; `protocols.routed_multi_patch_ls.solve_linear_decomposition`. Tests: `pytest` via the `light_stim` env (`/home/yuehan/miniconda3/envs/light_stim/bin/python -m pytest`).

## Global Constraints

- **Reuse two-patch primitives; do NOT fork them.** Import `place_patch`, `_bent_plaquettes`, `_select_checks`, `_readout_chain`, `_arrange`, `_algebraic_valid`, `BentLayoutError`, `_symplectic`, `_gf2_rank`, `_in_span` from `bent_layout.py`. Reuse the rotated coordinate convention (data on (odd,odd); placement via `place_patch`).
- **Routed bent bus, not a straight spine:** v1 bus is a single non-self-crossing orthogonal path with ≥1 bend (chained per-pair bent connectors). Tree/branching and search-based Manhattan routing → `NotImplementedError`/`BentLayoutError` (scope, not impossible).
- **Check dict schema** (coord-keyed): `{'syn': (col,row), 'type': 'X'|'Z'|'M', 'pauli': {(col,row): 'X'|'Z'}, 'corners': sorted([...])}`. Mixed `type='M'` carries both X and Z values, no qubit both (no twist).
- **Eight verify checks:** (1) all commute; (2) joint `∏ᵢ P̄ᵢ` in span; (3) **each single** `P̄ᵢ` NOT in span; (4) no Y/no twist; (5) `#data − rank == N − 1`; (6) DEM valid + noiseless-deterministic; (7) no `MPP`; (8) no tick collision. Confirm the measured joint with `peek_observable_expectation`.
- **The two-patch `build_rotated_bent_xz_layout` stays unchanged** — multi-patch is additive.
- **Commit authorship:** author as `Yuehan Zhang <johnzhang514145@gmail.com>` (repo-local, already set); **no `Co-Authored-By` trailer**. Commits **deferred** until the user says so (bundle code + tests + spec).
- Spec: `docs/superpowers/specs/2026-06-26-rotated-multi-patch-joint-design.md`.

## File Structure

| File | Responsibility |
|---|---|
| `lightstim/qec_code/surface_code/rotated/bent_layout.py` (modify, small) | generalize `_select_checks` to preserve a **list** of logicals (two-patch passes a 2-list); keep everything else. |
| `lightstim/qec_code/surface_code/rotated/multi_patch_joint.py` (new) | `build_rotated_multi_patch_joint_layout`, `MultiPatchJointLayout` (`.verify()`, `.build_circuit()`); the routed-bent-bus construction. |
| `lightstim/qec_code/surface_code/rotated/__init__.py` (modify) | export the two new symbols. |
| `tests/test_rotated_multi_patch_joint.py` (new) | multi-logical verify helper; 3-patch `X̄₁Z̄₂X̄₃` build+verify; `X̄₁Z̄₂Z̄₃`; coordinate-aware; impossible/out-of-scope raise; N=4 sanity. |

Run tests: `cd /nvme2n1/yuehan_zhang/resource_analsis/LightStim && /home/yuehan/miniconda3/envs/light_stim/bin/python -m pytest tests/test_rotated_multi_patch_joint.py -q`

---

### Task 1: Generalize `_select_checks` to a list of logicals

The keep must preserve **every** patch logical, not just one X and one Z.

**Files:**
- Modify: `lightstim/qec_code/surface_code/rotated/bent_layout.py`
- Test: `tests/test_rotated_bent_layout.py` (existing suite must still pass)

**Interfaces:**
- Produces: `_select_checks(data, plaqs, preserve)` where `preserve` is a list of symplectic vectors (one per logical to keep out of span). The current two-patch call site passes `[vX, vZ]`.

- [ ] **Step 1: Change the signature + body** to accept a list of logical vectors

```python
def _select_checks(data, plaqs, preserve):
    """Maximal commuting, GF(2)-independent set that preserves EVERY logical in ``preserve``
    (none pulled into the stabilizer span). ``preserve`` is a list of symplectic vectors."""
    sv, n = _symplectic(data)
    forced = [ch for ch in plaqs if len(ch["pauli"]) >= 3]
    rest = sorted((ch for ch in plaqs if len(ch["pauli"]) == 2), key=lambda c: c["syn"])
    kept = list(forced)
    rows = [sv(ch["pauli"]) for ch in forced]
    for ch in rest:
        v = sv(ch["pauli"])
        if not all(_commute(v, r, n) for r in rows):
            continue
        if not _gf2_independent(rows, v):
            continue
        if any(_in_span(rows + [v], L) for L in preserve):
            continue
        rows.append(v); kept.append(ch)
    return kept
```

- [ ] **Step 2: Update the two-patch call site** in `build_rotated_bent_xz_layout`

```python
        sv, _ = _symplectic(data)
        vX = sv({c: "X" for c in x_logical}); vZ = sv({c: "Z" for c in z_logical})
        checks = _select_checks(data, _bent_plaquettes(data, retype, phase), [vX, vZ])
```
(Find the existing `_select_checks(data, plaqs, x_logical, z_logical)` call and replace; remove the old `x_logical, z_logical` parameter handling inside `_select_checks`.)

- [ ] **Step 3: Run the existing two-patch suite — must stay green**

Run: `… -m pytest tests/test_rotated_bent_layout.py -q`
Expected: all pass (no behavior change; the 2-logical list reproduces the old keep).

- [ ] **Step 4: Commit** *(deferred)*

---

### Task 2: Multi-logical verify helpers (the oracle)

**Files:**
- Create: `tests/test_rotated_multi_patch_joint.py`

**Interfaces:**
- Produces: `joint_acceptance(data, checks, logicals)` where `logicals = [(measured_logical, support), ...]`; returns a dict of the 5 algebraic checks generalized to N logicals.

- [ ] **Step 1: Write the helper + a self-test on a hand-made valid case**

```python
import numpy as np, pytest
from lightstim.qec_code.surface_code.rotated.bent_layout import _symplectic, _gf2_rank, _in_span

def _lvec(sv, measured, support):
    return sv({c: measured for c in support})

def joint_acceptance(data, checks, logicals):
    """5 algebraic checks for an N-term joint ∏ P̄_i. logicals = [(P, support), ...]."""
    sv, n = _symplectic(data)
    S = [sv(ch["pauli"]) for ch in checks]
    comm = lambda a, b: int((a[:n] & b[n:]).sum() + (a[n:] & b[:n]).sum()) % 2 == 0
    commute = all(comm(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S)))
    twist = any(P == "Y" for ch in checks for P in ch["pauli"].values())
    singles = [_lvec(sv, P, sup) for P, sup in logicals]
    joint = np.zeros(2 * n, np.uint8)
    for v in singles:
        joint ^= v
    N = len(logicals)
    return dict(commute=commute, joint=_in_span(S, joint),
                no_single=not any(_in_span(S, v) for v in singles),
                no_twist=not twist, logical_count=len(data) - _gf2_rank(S) == N - 1)

def test_joint_acceptance_on_two_patch_reference():
    # Sanity: the validated two-patch X̄₁Z̄₂ layout satisfies the N-logical helper with N=2.
    from lightstim.qec_code.surface_code.rotated.bent_layout import build_rotated_bent_xz_layout, PatchSpec
    lay = build_rotated_bent_xz_layout([PatchSpec("p1",(1,7),3,"X","X_horizontal"),
                                        PatchSpec("p2",(7,1),3,"Z","X_horizontal")])
    a = joint_acceptance(lay.data, lay.checks, [("X", lay.x_logical), ("Z", lay.z_logical)])
    assert a == {"commute": True, "joint": True, "no_single": True, "no_twist": True, "logical_count": True}
```

- [ ] **Step 2: Run; the two-patch sanity must pass** (validates the helper before it gates new code)

Run: `… -m pytest tests/test_rotated_multi_patch_joint.py::test_joint_acceptance_on_two_patch_reference -q`
Expected: PASS.

- [ ] **Step 3: Commit** *(deferred)*

---

### Task 3: 3-patch `X̄₁Z̄₂X̄₃` routed bent bus (the novel core)

Chain two bent connectors into a 2-bend staircase bus; re-type the global X-side; gate on `verify()`.

**Files:**
- Create: `lightstim/qec_code/surface_code/rotated/multi_patch_joint.py`
- Modify: `tests/test_rotated_multi_patch_joint.py`

**Interfaces:**
- Produces: `build_rotated_multi_patch_joint_layout(patches, routing="auto") -> MultiPatchJointLayout` with `.data`, `.checks`, `.logicals` (list of `(name, P, support)`), `.joint_support`, `.readout_chain`, `.verify()`, `.build_circuit(rounds, p)`.

- [ ] **Step 1: Write the failing test** — a 3-patch `X̄₁Z̄₂X̄₃` staircase builds and passes all 8 verify checks with 2 mixed walls

```python
from lightstim.qec_code.surface_code.rotated import (
    build_rotated_multi_patch_joint_layout, PatchSpec)

P1 = PatchSpec("p1", (1, 7), 3, "X", "X_horizontal")    # X-arm, lower-left
P2 = PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal")    # Z, upper-middle
P3 = PatchSpec("p3", (13, 7), 3, "X", "X_horizontal")   # X-arm, lower-right (2nd bend)

def test_three_patch_XZX_builds_and_verifies():
    lay = build_rotated_multi_patch_joint_layout([P1, P2, P3])
    v = lay.verify()
    assert all(v.values()), v
    assert sum(c["type"] == "M" for c in lay.checks) == 2 * 3   # 2 walls × d mixed checks each
    assert len(lay.logicals) == 3
```
(The exact p3 origin / arrangement may be adjusted during derivation so the staircase is a valid non-self-crossing bent bus; keep 2 walls and `X̄₁Z̄₂X̄₃`.)

- [ ] **Step 2: Run to verify it fails** (`ImportError`/build failure).

- [ ] **Step 3: Implement `MultiPatchJointLayout` + `build_rotated_multi_patch_joint_layout`**

Construction (derive the exact region against `verify()`, as the two-patch geometry was derived):
  1. `place_patch` each spec; order the patches along the bus (by the given list order).
  2. For each consecutive pair, build a **bent connector** reusing `_arrange`'s band + vertical-column logic (validate per-leg gaps `2+2k`/`2+2m`; raise `BentLayoutError` with the reason on misalignment/overlap/self-crossing).
  3. `data = ⋃ patches ∪ ⋃ legs − corner cuts`.
  4. `retype = ⋃ (X̄-patch data) ∪ (X-typed bus segments)` → `_bent_plaquettes(data, retype, phase)`; mixed checks appear at each X↔Z transition (one wall per transition).
  5. `preserve = [sv(P_i on support_i) for each patch]`; `_select_checks(data, plaqs, preserve)`; try `phase ∈ {0,1}` and pick the one with `joint_acceptance(...)` all-true (else next phase; else `BentLayoutError`).
  6. `joint_support` = XOR of per-patch logical vectors; `readout_chain = _readout_chain(...)` with that multi-term target.
  7. `verify()`: the 5 algebraic (via `joint_acceptance`) + DEM-valid + noiseless-deterministic + no-MPP + no-tick-collision (reuse the two-patch `_no_tick_collision` and the `BentLayout.verify` circuit checks); confirm with `peek_observable_expectation`.
  8. `build_circuit`: `RotatedBentJointMeasurement(data, checks, observable)` — observable = the joint readout; pin the init/observable wiring so the noiseless circuit is deterministic + DEM-valid.

- [ ] **Step 4: Iterate the geometry until `verify()` all-pass at 3 patches.** This is the gate; derive the leg/wall placement empirically (scratch script diffing against `joint_acceptance`) before finalizing. Do NOT weaken the asserts.

- [ ] **Step 5: Commit** *(deferred)*

---

### Task 4: `X̄₁Z̄₂Z̄₃` + coordinate-aware + impossible-placement tests

**Files:**
- Modify: `tests/test_rotated_multi_patch_joint.py`, `lightstim/qec_code/surface_code/rotated/__init__.py`

- [ ] **Step 1: Export from `__init__.py`**

```python
from .multi_patch_joint import build_rotated_multi_patch_joint_layout, MultiPatchJointLayout
# add both names to __all__
```

- [ ] **Step 2: Write the tests**

```python
def test_three_patch_XZZ_one_wall_one_css():
    lay = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
         PatchSpec("p2", (7, 1), 3, "Z", "X_horizontal"),
         PatchSpec("p3", (13, 1), 3, "Z", "X_horizontal")])   # Z-Z -> CSS junction, no 2nd wall
    assert all(lay.verify().values())
    assert sum(c["type"] == "M" for c in lay.checks) == 3      # only the 1↔2 X→Z wall (d mixed)

def test_multi_patch_coordinate_aware():
    A = build_rotated_multi_patch_joint_layout([P1, P2, P3])
    B = build_rotated_multi_patch_joint_layout(
        [PatchSpec("p1", (3, 9), 3, "X", "X_horizontal"),
         PatchSpec("p2", (9, 3), 3, "Z", "X_horizontal"),
         PatchSpec("p3", (15, 9), 3, "X", "X_horizontal")])    # translated +(2,2)
    assert list(A.data) != list(B.data)
    assert all(A.verify().values()) and all(B.verify().values())

def test_multi_patch_impossible_raises():
    from lightstim.qec_code.surface_code.rotated import BentLayoutError
    with pytest.raises(BentLayoutError):
        build_rotated_multi_patch_joint_layout(
            [PatchSpec("p1", (1, 7), 3, "X", "X_horizontal"),
             PatchSpec("p2", (6, 1), 3, "Z", "X_horizontal"),    # even-origin / odd gap -> impossible
             PatchSpec("p3", (13, 7), 3, "X", "X_horizontal")])
```

- [ ] **Step 3: Implement until all pass** (the X̄₁Z̄₂Z̄₃ same-type junction = CSS, no wall — falls out of the re-typing: p2,p3 both Z-side so no transition between them).

- [ ] **Step 4: Commit** *(deferred)*

---

### Task 5: Generalize to N patches

**Files:**
- Modify: `tests/test_rotated_multi_patch_joint.py`

- [ ] **Step 1: Write the N=4 sanity test**

```python
@pytest.mark.parametrize("seq", [("X","Z","X","Z"), ("X","Z","Z","X")])
def test_four_patch_sequences(seq):
    specs = [PatchSpec(f"p{i+1}", (1 + 6*i, 7 if seq[i]=="X" else 1), 3, seq[i], "X_horizontal")
             for i in range(4)]
    lay = build_rotated_multi_patch_joint_layout(specs)
    v = lay.verify()
    assert all(v.values()), (seq, v)
    walls = sum(1 for i in range(3) if seq[i] != seq[i+1])
    assert sum(c["type"] == "M" for c in lay.checks) == walls * 3   # d mixed per transition
```
(Adjust per-patch origins so the 4-patch staircase is a valid non-self-crossing bent bus; the construction already loops over consecutive pairs — N falls out.)

- [ ] **Step 2: Run; iterate the per-pair loop + arrangement until both N=4 sequences pass `verify()`.**

- [ ] **Step 3: Run the full multi-patch + two-patch + rotated/xzzx suites** (no regression)

Run: `… -m pytest tests/test_rotated_multi_patch_joint.py tests/test_rotated_bent_layout.py tests/test_rotated_lattice_surgery.py tests/test_xzzx_code.py -q`
Expected: all pass.

- [ ] **Step 4: Commit** *(deferred — present results; commit module + tests + spec together when the user approves.)*

---

## Self-Review

**Spec coverage:** interface → Task 3; routed bent bus (chained connectors) → Task 3; re-typing/walls → Task 3 step 3.4; multi-term readout → Task 3 step 3.6; reuse primitives → Task 1 (select_checks) + Task 3; 8-check verify (incl. `#data−rank=N−1`, each-single-not) → Task 2 helper + Task 3 verify; CSS-junction-for-same-type → Task 4 (`X̄₁Z̄₂Z̄₃`); coordinate-aware → Task 4; feasibility raises → Task 4; N generalization → Task 5; two-patch unchanged → Task 1 step 3 (suite stays green). ✅

**Placeholder scan:** The "derive the exact region against `verify()`" wording in Task 3/5 is intentional and unavoidable — the routed-bus/wall geometry is *defined by* the verify oracle (provided fully in Task 2), exactly as the two-patch boundary rules were derived; all test code and the `_select_checks`/`joint_acceptance` code are complete. No `TODO`/`TBD` as deliverables.

**Type consistency:** `_select_checks(data, plaqs, preserve)` (list of vectors) consistent across Task 1 + Task 3.5. `joint_acceptance(data, checks, logicals=[(P,support)])` consistent Task 2 + 3 + 4. `build_rotated_multi_patch_joint_layout(patches, routing) -> MultiPatchJointLayout(.data,.checks,.logicals,.joint_support,.readout_chain,.verify,.build_circuit)` consistent across Tasks 3–5. Check dict schema `{'syn','type','pauli','corners'}` matches `bent_layout`/`RotatedBentJointMeasurement` consumers.
