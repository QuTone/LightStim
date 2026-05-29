"""
LightStim local API server.

Start with:
    venv/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 9999

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
    decompose_errors: bool = False

    def noise_config(self) -> NoiseConfig:
        p = self.p
        return NoiseConfig(
            p_1q=self.p_1q   if self.p_1q   is not None else p,
            p_2q=self.p_2q   if self.p_2q   is not None else p,
            p_meas=self.p_meas  if self.p_meas  is not None else p,
            p_reset=self.p_reset if self.p_reset is not None else p,
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

class MemoryRequest(NoiseBase):
    code: Literal["rotated", "unrotated", "toric", "repetition"] = "rotated"
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
            from lightstim.qec_code.surface_code.rotated import (
                RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock)
            system.add_patch(RotatedSurfaceCode(distance=req.distance), name="main")
            se_cls = RotatedSurfaceCodeExtractionBlock
        elif req.code == "unrotated":
            from lightstim.qec_code.surface_code.unrotated import (
                UnrotatedSurfaceCode, UnrotatedSurfaceCodeExtractionBlock)
            system.add_patch(UnrotatedSurfaceCode(distance=req.distance), name="main")
            se_cls = UnrotatedSurfaceCodeExtractionBlock
        elif req.code == "toric":
            from lightstim.qec_code.surface_code.toric import (
                ToricCode, ToricCodeExtractionBlock)
            system.add_patch(ToricCode(distance=req.distance), name="main")
            se_cls = ToricCodeExtractionBlock
        else:  # repetition
            from lightstim.qec_code.repetition import (
                RepetitionCode, RepetitionCodeExtractionBlock)
            system.add_patch(RepetitionCode(distance=req.distance), name="main")
            se_cls = RepetitionCodeExtractionBlock

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


@app.post("/api/circuit/two-patch-ls")
def two_patch_ls(req: TwoPatchLSRequest):
    try:
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment
        offset = (0.0, float(req.distance1 * 4)) if req.interaction_type == "ZZ" \
                 else (float(req.distance1 * 4), 0.0)
        init1, meas1 = ("X", "X") if req.interaction_type == "ZZ" else ("Z", "Z")
        init2, meas2 = ("Z", "Z") if req.interaction_type == "ZZ" else ("X", "X")

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


@app.post("/api/circuit/cnot-ls")
def cnot_ls(req: CNOTLSRequest):
    try:
        from lightstim.protocols.cnot_ls import CNOTLSExperiment
        d = req.distance
        # XX coupler (target-ancilla) → horizontal; ZZ coupler (control-ancilla) → vertical
        exp = CNOTLSExperiment(
            patch_configs={"a": {"distance": d}, "c": {"distance": d}, "t": {"distance": d}},
            offset_ta=(float(2 * d), 0.0),   # target: right of ancilla (XX)
            offset_ca=(0.0, float(2 * d)),   # control: above ancilla (ZZ)
            rounds=req.rounds,
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
            noise_params=req.noise_config(),
            noise_model=req.noise_model,
        )
        return _export(exp.build(), req, "ghz",
                       d, req.rounds_before + req.rounds_after)
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
    rounds: int = Field(3, ge=1, le=20)
    teleport_state: Literal["Z", "X"] = "Z"


@app.post("/api/circuit/bell-teleport")
def bell_teleport(req: BellTeleportRequest):
    try:
        from lightstim.protocols.bell_teleportation import (
            BellTeleportTG, BellTeleportZZLS, BellTeleportXXLS)
        noise = req.noise_config()
        if req.variant == "tg":
            exp = BellTeleportTG(distance=req.distance,
                                  rounds_pre=req.rounds, rounds_mid=1, rounds_post=1,
                                  teleport_state=req.teleport_state,
                                  noise_params=noise, noise_model=req.noise_model)
        elif req.variant == "zz_ls":
            exp = BellTeleportZZLS(distance=req.distance,
                                    rounds_pre=req.rounds, rounds_ls=req.rounds,
                                    teleport_state=req.teleport_state,
                                    noise_params=noise, noise_model=req.noise_model)
        else:
            exp = BellTeleportXXLS(distance=req.distance,
                                    rounds_pre=req.rounds, rounds_ls=req.rounds,
                                    teleport_state=req.teleport_state,
                                    noise_params=noise, noise_model=req.noise_model)
        return _export(exp.build(), req,
                       f"bell_teleport_{req.variant}",
                       req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── T-gate Distillation ───────────────────────────────────────────────────────

class TGDistillationRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=7)
    rounds: int = Field(3, ge=1, le=20)


@app.post("/api/circuit/tg-distillation")
def tg_distillation(req: TGDistillationRequest):
    try:
        from lightstim.protocols.tg_distillation import build_distillation_circuit
        circuit, _, _ = build_distillation_circuit(d=req.distance,
                                                    rounds_init=req.rounds,
                                                    rounds_gate=1)
        return _export(circuit, req, "tg_distillation", req.distance, req.rounds)
    except Exception as e:
        _err(e)

# ── LS Distillation ───────────────────────────────────────────────────────────

class LSDistillationRequest(NoiseBase):
    distance: int = Field(3, ge=2, le=7)
    rounds: int = Field(3, ge=1, le=20)


@app.post("/api/circuit/ls-distillation")
def ls_distillation(req: LSDistillationRequest):
    try:
        from lightstim.protocols.ls_distillation import build_distillation_circuit
        circuit, _, _ = build_distillation_circuit(d=req.distance, rounds=req.rounds)
        return _export(circuit, req, "ls_distillation", req.distance, req.rounds)
    except Exception as e:
        _err(e)
