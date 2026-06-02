"""
LightStim local HTTP server.

Start with:
    venv/bin/uvicorn server.main:app --reload --host 0.0.0.0 --port 9999

Docs: http://localhost:9999/docs
"""
from __future__ import annotations
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lightstim.noise.config import NoiseConfig
from lightstim.frontend import export_all

app = FastAPI(title="LightStim API", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["POST", "GET"], allow_headers=["*"])

# ── Shared noise fields ────────────────────────────────────────────────────────

NoiseModel = Literal["circuit_level", "phenomenological", "code_capacity"]

class NoiseBase(BaseModel):
    noise_model: NoiseModel = "circuit_level"
    p: float = Field(1e-3, ge=0, le=0.5, description="Master error rate (used when individual rates are omitted)")
    p_1q:   Optional[float] = Field(None, ge=0, le=0.5)
    p_2q:   Optional[float] = Field(None, ge=0, le=0.5)
    p_meas: Optional[float] = Field(None, ge=0, le=0.5)
    p_reset: Optional[float] = Field(None, ge=0, le=0.5)
    p_idle: Optional[float] = Field(None, ge=0, le=0.5,
        description="Idle depolarizing on data qubits. Required for code_capacity / phenomenological.")
    decompose_errors: bool = False

    def noise_config(self) -> NoiseConfig:
        p = self.p
        # For code_capacity / phenomenological, p_idle is the primary parameter.
        # Fall back to master p so the model always has something to inject.
        p_idle_val = self.p_idle if self.p_idle is not None else p
        return NoiseConfig(
            p_1q=self.p_1q     if self.p_1q   is not None else p,
            p_2q=self.p_2q     if self.p_2q   is not None else p,
            p_meas=self.p_meas if self.p_meas is not None else p,
            p_reset=self.p_reset if self.p_reset is not None else p,
            p_idle=p_idle_val,
        )

def _export(circuit, req: NoiseBase, source: str, distance: int, rounds: int):
    return export_all(circuit, source=source, distance=distance, rounds=rounds,
                      noise_model=req.noise_model, physical_error_rate=req.p,
                      decompose_errors=req.decompose_errors)

def _err(e: Exception):
    raise HTTPException(status_code=500, detail=str(e))

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

_BB_PRESETS = {
    "bb_72_12_6":   {"l": 6,  "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 6},
    "bb_144_12_12": {"l": 12, "m": 6,  "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]], "d": 12},
    "bb_90_8_10":   {"l": 15, "m": 3,  "A": [[9,0],[0,1],[0,2]], "B": [[0,0],[2,0],[7,0]], "d": 10},
}

_4D_PRESETS = {
    "det3":     {"L": [[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,0,0,3]],  "n": 18,  "k": 6, "d": 3},
    "det9":     {"L": [[1,0,0,5],[0,1,0,6],[0,0,1,7],[0,0,0,9]],  "n": 54,  "k": 6, "d": 6},
    "hadamard": {"L": [[1,1,1,1],[0,2,0,2],[0,0,2,2],[0,0,0,4]],  "n": 96,  "k": 6, "d": 8},
}

class MemoryRequest(NoiseBase):
    code: Literal[
        "rotated", "unrotated", "toric", "repetition",
        "color",
        "bb_72_12_6", "bb_144_12_12", "bb_90_8_10",
        "4d_det3", "4d_det9", "4d_hadamard",
    ] = "rotated"
    distance: int = Field(3, ge=2, le=15)
    rounds:   int = Field(3, ge=1, le=30)
    basis: Literal["Z", "X"] = "Z"


@app.post("/api/circuit/memory")
def memory(req: MemoryRequest):
    try:
        from lightstim.ir.qec_system import QECSystem
        from lightstim.ir.tracker import SyndromeTracker
        from lightstim.ir.builder import CircuitBuilder

        system = QECSystem()

        if req.code == "rotated":
            from lightstim.qec_code.surface_code.rotated import RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock
            system.add_patch(RotatedSurfaceCode(distance=req.distance), name="main")
            se_cls = RotatedSurfaceCodeExtractionBlock
        elif req.code == "unrotated":
            from lightstim.qec_code.surface_code.unrotated import UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock
            system.add_patch(UnrotatedSurfaceCode(distance=req.distance), name="main")
            se_cls = UnrotatedSurfaceCodeExtractionBlock
        elif req.code == "toric":
            from lightstim.qec_code.surface_code.toric import ToricCode, ToricCodeExtractionBlock
            system.add_patch(ToricCode(distance=req.distance), name="main")
            se_cls = ToricCodeExtractionBlock
        elif req.code == "color":
            from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
            system.add_patch(ColorCode(distance=req.distance), name="main")
            se_cls = ColorCodeExtractionBlock
        elif req.code == "repetition":
            from lightstim.qec_code.repetition import RepetitionCode, RepetitionCodeExtractionBlock
            system.add_patch(RepetitionCode(distance=req.distance), name="main")
            se_cls = RepetitionCodeExtractionBlock
        elif req.code.startswith("bb_"):
            from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
            cfg = _BB_PRESETS[req.code]
            system.add_patch(BBCode(l=cfg["l"], m=cfg["m"], A=cfg["A"], B=cfg["B"]), name="main")
            se_cls = BBCodeExtractionBlock
        else:  # 4d_*
            from lightstim.qec_code.four_d_geo_code import FourDGeoCode, FourDGeoCodeExtractionBlock
            key = req.code.replace("4d_", "")
            cfg = _4D_PRESETS[key]
            system.add_patch(FourDGeoCode(L=cfg["L"], d=cfg["d"]), name="main")
            se_cls = FourDGeoCodeExtractionBlock

        tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
        builder = CircuitBuilder(tracker, system)
        se = se_cls(system)
        builder.write_coordinates()
        builder.initialize({q: req.basis for q in system.data_indices}, n=system.num_qubits)
        builder.apply_syndrome_extraction(se.circuit, rounds=req.rounds)
        builder.apply_data_readout({q: req.basis for q in system.data_indices})
        circuit = builder.build_noisy_circuit(req.noise_config(), noise_model=req.noise_model)

        return _export(circuit, req, f"{req.code}_memory_{req.basis.lower()}",
                       req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — LOGICAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Two-patch Lattice Surgery ─────────────────────────────────────────────────

class TwoPatchLSRequest(NoiseBase):
    distance1: int = Field(3, ge=2, le=9)
    distance2: int = Field(3, ge=2, le=9)
    interaction_type: Literal["ZZ", "XX"] = "ZZ"
    rounds: int = Field(3, ge=1, le=20)
    dx: Optional[float] = Field(None, description="Custom x-offset for patch2 (overrides default)")
    dy: Optional[float] = Field(None, description="Custom y-offset for patch2 (overrides default)")
    init_patch1: Optional[Literal["X", "Z"]] = Field(None, description="Initial state basis for patch 1 (overrides interaction_type default)")
    meas_patch1: Optional[Literal["X", "Z"]] = Field(None, description="Measurement basis for patch 1 (overrides interaction_type default)")
    init_patch2: Optional[Literal["X", "Z"]] = Field(None, description="Initial state basis for patch 2 (overrides interaction_type default)")
    meas_patch2: Optional[Literal["X", "Z"]] = Field(None, description="Measurement basis for patch 2 (overrides interaction_type default)")


@app.post("/api/circuit/two-patch-ls")
def two_patch_ls(req: TwoPatchLSRequest):
    try:
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        default_offset = (0.0, float(req.distance1 * 2)) if req.interaction_type == "ZZ" \
                         else (float(req.distance1 * 2), 0.0)
        offset = (req.dx if req.dx is not None else default_offset[0],
                  req.dy if req.dy is not None else default_offset[1])
        # Default basis derived from interaction type; each of init/meas independently overridable
        def_b1 = "X" if req.interaction_type == "ZZ" else "Z"
        def_b2 = "Z" if req.interaction_type == "ZZ" else "X"
        init1 = req.init_patch1 if req.init_patch1 is not None else def_b1
        meas1 = req.meas_patch1 if req.meas_patch1 is not None else def_b1
        init2 = req.init_patch2 if req.init_patch2 is not None else def_b2
        meas2 = req.meas_patch2 if req.meas_patch2 is not None else def_b2

        exp = TwoPatchLSExperiment(
            patch1_config={"distance": req.distance1},
            patch2_config={"distance": req.distance2},
            offset=offset, interaction_type=req.interaction_type,
            initial_state_patch1=init1, initial_state_patch2=init2,
            measure_state_patch1=meas1, measure_state_patch2=meas2,
            rounds=req.rounds, noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req,
                       f"two_patch_ls_{req.interaction_type.lower()}",
                       req.distance1, req.rounds)
    except Exception as e:
        _err(e)

# ── Transversal CNOT ──────────────────────────────────────────────────────────

class TransversalCNOTRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=9)
    rounds_before: int = Field(2, ge=1, le=20)
    rounds_after:  int = Field(2, ge=1, le=20)
    init_ctrl: Optional[Literal["X", "Z"]] = Field(None, description="Control init basis (default Z)")
    meas_ctrl: Optional[Literal["X", "Z"]] = Field(None, description="Control measure basis (default Z)")
    init_tgt:  Optional[Literal["X", "Z"]] = Field(None, description="Target init basis (default Z)")
    meas_tgt:  Optional[Literal["X", "Z"]] = Field(None, description="Target measure basis (default Z)")


@app.post("/api/circuit/transversal-cnot")
def transversal_cnot(req: TransversalCNOTRequest):
    try:
        from lightstim.protocols.cnot_trans import CNOTTransExperiment
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock)
        d = req.distance
        exp = CNOTTransExperiment(
            code_patch_class=UnrotatedSurfaceCode,
            extraction_block_class=UnrotatedSurfaceCodeExtractionBlock,
            code_params_control={"distance": d},
            code_params_target={"distance": d},
            offset_target=(float(d * 4), 0.0),
            rounds_before=req.rounds_before,
            rounds_after=req.rounds_after,
            initial_basis_control=req.init_ctrl or "Z",
            measure_basis_control=req.meas_ctrl or "Z",
            initial_basis_target=req.init_tgt  or "Z",
            measure_basis_target=req.meas_tgt  or "Z",
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req, "transversal_cnot",
                       d, req.rounds_before + req.rounds_after)
    except Exception as e:
        _err(e)

# ── CNOT Lattice Surgery (3-patch) ────────────────────────────────────────────

class CNOTLSRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=7)
    rounds: int = Field(2, ge=1, le=20)
    # Ancilla: init X → meas Z forced; init Z → meas X forced (protocol constraint)
    init_a: Optional[Literal["X", "Z"]] = Field(None, description="Ancilla init basis (default X)")
    init_c: Optional[Literal["X", "Z"]] = Field(None, description="Control init basis (default X)")
    meas_c: Optional[Literal["X", "Z"]] = Field(None, description="Control measure basis (default X)")
    init_t: Optional[Literal["X", "Z"]] = Field(None, description="Target init basis (default X)")
    meas_t: Optional[Literal["X", "Z"]] = Field(None, description="Target measure basis (default X)")


@app.post("/api/circuit/cnot-ls")
def cnot_ls(req: CNOTLSRequest):
    try:
        from lightstim.protocols.cnot_ls import CNOTLSExperiment
        d = req.distance
        init_a = req.init_a or "X"
        meas_a = "Z" if init_a == "X" else "X"  # protocol constraint
        init_c = req.init_c or "X"
        meas_c = req.meas_c or "X"
        init_t = req.init_t or "X"
        meas_t = req.meas_t or "X"
        exp = CNOTLSExperiment(
            patch_configs={"a": {"distance": d}, "c": {"distance": d}, "t": {"distance": d}},
            offset_ta=(float(2 * d), 0.0),   # target: right of ancilla (XX)
            offset_ca=(0.0, float(2 * d)),   # control: above ancilla (ZZ)
            rounds=req.rounds,
            initial_state_dict={"a": init_a, "c": init_c, "t": init_t},
            measure_state_dict={"a": meas_a, "c": meas_c, "t": meas_t},
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req, "cnot_ls", d, req.rounds)
    except Exception as e:
        _err(e)

# ── GHZ State Preparation ─────────────────────────────────────────────────────

class GHZRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=9)
    rounds_before: int = Field(2, ge=1, le=20)
    rounds_after:  int = Field(2, ge=1, le=20)
    # Per-patch init/meas basis (defaults match GHZExperiment: P1 X→Z, P2/P3 Z→Z)
    init_p1: Optional[Literal["X", "Z"]] = Field(None, description="Patch 1 init basis (default X)")
    meas_p1: Optional[Literal["X", "Z"]] = Field(None, description="Patch 1 measure basis (default Z)")
    init_p2: Optional[Literal["X", "Z"]] = Field(None, description="Patch 2 init basis (default Z)")
    meas_p2: Optional[Literal["X", "Z"]] = Field(None, description="Patch 2 measure basis (default Z)")
    init_p3: Optional[Literal["X", "Z"]] = Field(None, description="Patch 3 init basis (default Z)")
    meas_p3: Optional[Literal["X", "Z"]] = Field(None, description="Patch 3 measure basis (default Z)")


@app.post("/api/circuit/ghz")
def ghz(req: GHZRequest):
    try:
        from lightstim.protocols.ghz import GHZExperiment
        d = req.distance
        step = float(d * 4)
        exp = GHZExperiment(
            distance=d,
            offset_patch2=(step, 0.0),
            offset_patch3=(step * 2, 0.0),
            rounds_before=req.rounds_before,
            rounds_after=req.rounds_after,
            initial_basis_patch1=req.init_p1 or "X",
            measure_basis_patch1=req.meas_p1 or "Z",
            initial_basis_patch2=req.init_p2 or "Z",
            measure_basis_patch2=req.meas_p2 or "Z",
            initial_basis_patch3=req.init_p3 or "Z",
            measure_basis_patch3=req.meas_p3 or "Z",
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req, "ghz",
                       d, req.rounds_before + req.rounds_after)
    except Exception as e:
        _err(e)

# ── Logical H (fold-transversal Hadamard) ────────────────────────────────────

class LogicalHRequest(NoiseBase):
    distance: int = Field(3, ge=3, le=11, description="Must be odd")
    rounds: int = Field(2, ge=1, le=20)
    # Valid H-gate experiments: Z→X (|0⟩ prep, |+⟩ verify) or X→Z (|+⟩ prep, |0⟩ verify)
    init_basis:    Literal["Z", "X"] = "Z"
    measure_basis: Literal["X", "Z"] = "X"


@app.post("/api/circuit/logical-h")
def logical_h(req: LogicalHRequest):
    try:
        from lightstim.protocols.fold_transversal import build_gate_verification_circuit
        circuit = build_gate_verification_circuit(
            distance=req.distance,
            gates=["fold_transversal_hadamard"],
            init_basis=req.init_basis,
            measure_basis=req.measure_basis,
            rounds=req.rounds,
            unencode=False,
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(circuit, req, "logical_h", req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── Logical S (fold-transversal S gate, roundtrip) ────────────────────────────

class LogicalSRequest(NoiseBase):
    distance: int = Field(3, ge=3, le=11, description="Must be odd")
    rounds: int = Field(2, ge=1, le=20)
    variant: Literal["roundtrip", "oneway"] = "roundtrip"


@app.post("/api/circuit/logical-s")
def logical_s(req: LogicalSRequest):
    try:
        from lightstim.protocols.fold_transversal import (
            build_s_roundtrip_circuit, build_s_oneway_circuit)
        noise = req.noise_config()
        if req.variant == "oneway":
            circuit = build_s_oneway_circuit(
                distance=req.distance, rounds=req.rounds,
                noise_params=noise, noise_model=req.noise_model)
        else:
            circuit = build_s_roundtrip_circuit(
                distance=req.distance, rounds=req.rounds,
                noise_params=noise, noise_model=req.noise_model)
        return _export(circuit, req, f"logical_s_{req.variant}", req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── Multi-patch Lattice Surgery (ZZZ…Z product measurement) ──────────────────

class MultiPatchLSRequest(NoiseBase):
    n_patches: int = Field(3, ge=2, le=5, description="Number of patches (2–5)")
    distance: int = Field(3, ge=2, le=7)
    rounds: int = Field(2, ge=1, le=20)
    init_basis: Literal["X", "Z"] = Field("X", description="Init basis for all data qubits (default X)")


@app.post("/api/circuit/multi-patch-ls")
def multi_patch_ls(req: MultiPatchLSRequest):
    try:
        import io, contextlib
        from lightstim.qec_code.surface_code.unrotated import (
            UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock,
            UnrotatedMultiPatchCoupler)
        from lightstim.ir.qec_system import QECSystem
        from lightstim.ir.tracker import SyndromeTracker
        from lightstim.ir.builder import CircuitBuilder

        d, n = req.distance, req.n_patches
        # Layout: patches arranged in an L-shape / staircase to satisfy
        # the coupler's disjoint-y-range constraint.
        # p1 at (0,0), p2 at (step,0), remaining patches stacked below p1.
        step = float(d * 4)
        offsets = [(0.0, 0.0), (step, 0.0)]
        for i in range(2, n):
            offsets.append((0.0, float(i - 1) * step))

        system = QECSystem()
        patch_names = []
        for i, off in enumerate(offsets):
            name = f"p{i+1}"
            system.add_patch(UnrotatedSurfaceCode(distance=d), name=name, offset=off)
            patch_names.append(name)

        center_x = step / 2
        with contextlib.redirect_stdout(io.StringIO()):
            system.register_coupler(
                UnrotatedMultiPatchCoupler(), patch_names, "c",
                path_axis="vertical", center_axis=center_x)

        tracker = SyndromeTracker(system.num_qubits, system.num_logicals)
        builder = CircuitBuilder(tracker, system)
        builder.write_coordinates()

        ib = req.init_basis
        non_coupler = {q: ib for q in system.data_indices
                       if system.index_to_owner_map.get(q) != "c"}
        builder.initialize(non_coupler, n=system.num_qubits)
        se = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(se.circuit, rounds=req.rounds)

        builder.activate_coupler("c")
        coupler_data = {system.local_to_global_map["c"][q]: ib
                        for q in system.coupler_patches["c"].data_indices}
        builder.initialize(coupler_data, n=system.num_qubits)
        se2 = UnrotatedSurfaceCodeExtractionBlock(system)
        builder.apply_syndrome_extraction(se2.circuit, rounds=req.rounds)

        builder.apply_data_readout({**non_coupler, **coupler_data})
        circuit = builder.build_noisy_circuit(req.noise_config(), noise_model=req.noise_model)

        return _export(circuit, req, f"multi_patch_ls_{n}p_{ib.lower()}", d, req.rounds)
    except Exception as e:
        _err(e)

# ── State Injection ───────────────────────────────────────────────────────────

class StateInjectionRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=9)
    rounds: int = Field(3, ge=1, le=20)
    inject_state: Literal["Z", "X", "Y"] = "Y"
    protocol: Literal["corner", "middle"] = "corner"
    post_select_mode: Literal["full_postselection", "full_qec", "hybrid"] = "full_postselection"


@app.post("/api/circuit/state-injection")
def state_injection(req: StateInjectionRequest):
    try:
        from lightstim.protocols.state_injection import StateInjectionExperiment
        exp = StateInjectionExperiment(
            distance=req.distance,
            rounds=req.rounds,
            inject_state=req.inject_state,
            protocol=req.protocol,
            post_select_mode=req.post_select_mode,
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req,
                       f"state_injection_{req.inject_state.lower()}",
                       req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — LOGICAL CIRCUITS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Bell Teleportation ────────────────────────────────────────────────────────

class BellTeleportRequest(NoiseBase):
    variant: Literal["tg", "zz_ls", "xx_ls"] = "tg"
    distance: int = Field(3, ge=2, le=9)
    rounds_pre:  int = Field(2, ge=1, le=20, description="SE rounds before gate / LS")
    rounds_mid:  int = Field(1, ge=1, le=20, description="SE rounds during gate (TG only)")
    rounds_post: int = Field(1, ge=1, le=20, description="SE rounds after gate (TG only)")
    rounds_ls:   int = Field(2, ge=1, le=20, description="SE rounds for LS coupler (LS variants)")
    teleport_state: Literal["Z", "X"] = "Z"


@app.post("/api/circuit/bell-teleport")
def bell_teleport(req: BellTeleportRequest):
    try:
        from lightstim.protocols.bell_teleportation import (
            BellTeleportTG, BellTeleportZZLS, BellTeleportXXLS)
        noise = req.noise_config()
        if req.variant == "tg":
            exp = BellTeleportTG(distance=req.distance,
                                  rounds_pre=req.rounds_pre,
                                  rounds_mid=req.rounds_mid,
                                  rounds_post=req.rounds_post,
                                  teleport_state=req.teleport_state,
                                  noise_params=noise, noise_model=req.noise_model)
        elif req.variant == "zz_ls":
            exp = BellTeleportZZLS(distance=req.distance,
                                    rounds_pre=req.rounds_pre,
                                    rounds_ls=req.rounds_ls,
                                    teleport_state=req.teleport_state,
                                    noise_params=noise, noise_model=req.noise_model)
        else:
            exp = BellTeleportXXLS(distance=req.distance,
                                    rounds_pre=req.rounds_pre,
                                    rounds_ls=req.rounds_ls,
                                    teleport_state=req.teleport_state,
                                    noise_params=noise, noise_model=req.noise_model)
        return _export(exp.build(), req,
                       f"bell_teleport_{req.variant}",
                       req.distance, req.rounds_pre)
    except Exception as e:
        _err(e)

# ── T-gate Distillation ───────────────────────────────────────────────────────

class TGDistillationRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=5)   # d>5 takes too long
    rounds: int = Field(3, ge=1, le=20)


@app.post("/api/circuit/tg-distillation")
def tg_distillation(req: TGDistillationRequest):
    try:
        from lightstim.protocols.tg_distillation import (
            build_distillation_circuit, inject_noise, _TG_MAGIC_NAMES)
        circuit, _, system = build_distillation_circuit(
            d=req.distance, rounds_init=req.rounds, rounds_gate=1)
        magic_qubits = []
        for name in _TG_MAGIC_NAMES:
            patch, _ = system.patches[name]
            for local_q in patch.data_indices:
                magic_qubits.append(system.local_to_global_map[name][local_q])
        noisy = inject_noise(circuit, magic_qubits=magic_qubits,
                             p=req.p, p_injected=req.p, mode="full")
        return _export(noisy, req, "tg_distillation", req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── LS Distillation ───────────────────────────────────────────────────────────

class LSDistillationRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=5)   # d>5 takes too long
    rounds: int = Field(3, ge=1, le=20)


@app.post("/api/circuit/ls-distillation")
def ls_distillation(req: LSDistillationRequest):
    try:
        from lightstim.protocols.ls_distillation import (
            build_distillation_circuit, inject_noise, _LS_MAGIC_NAMES)
        circuit, _, system = build_distillation_circuit(d=req.distance, rounds=req.rounds)
        magic_qubits = set()
        for name in _LS_MAGIC_NAMES:
            patch, _ = system.patches[name]
            for local_q in patch.data_indices:
                magic_qubits.add(system.local_to_global_map[name][local_q])
        noisy = inject_noise(circuit, magic_qubits=magic_qubits,
                             p=req.p, p_injected=req.p, mode="full",
                             data_indices=list(system.data_indices))
        return _export(noisy, req, "ls_distillation", req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── CrossLS (Surface-PQRM Lattice Surgery) ────────────────────────────────────

_PQRM_PRESETS = {
    "(1,2,4)": [1, 2, 4],   # [[15,1,3]]
    "(1,3,5)": [1, 3, 5],   # [[31,1,3]]
    "(1,4,6)": [1, 4, 6],   # [[63,1,3]]
}

class CrossLSRequest(NoiseBase):
    pqrm_preset: Literal["(1,2,4)", "(1,3,5)", "(1,4,6)"] = "(1,2,4)"
    d_surf: int = Field(3, ge=2, le=9, description="Surface code distance")
    rounds: int = Field(2, ge=1, le=20)
    pqrm_state: Literal["Z", "X"] = "Z"
    surf_state: Literal["X", "Z"] = "X"
    post_select_hybrid: bool = Field(False, description="Enable hybrid post-selection on PQRM ancilla")


@app.post("/api/circuit/cross-ls")
def cross_ls(req: CrossLSRequest):
    try:
        from lightstim.protocols.cross_ls import CrossLSExperiment
        pqrm_para = _PQRM_PRESETS[req.pqrm_preset]
        exp = CrossLSExperiment(
            PQRM_para=pqrm_para,
            d_surf=req.d_surf,
            rounds=req.rounds,
            PQRM_state=req.pqrm_state,
            surf_state=req.surf_state,
            post_select_hybrid=req.post_select_hybrid,
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req, f"cross_ls_{req.pqrm_preset}",
                       req.d_surf, req.rounds)
    except Exception as e:
        _err(e)
