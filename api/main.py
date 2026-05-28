"""
LightStim local API server.

Start with:
    venv/bin/uvicorn api.main:app --reload --port 8000

Then the frontend can fetch:
    POST http://localhost:8000/api/circuit
"""
from __future__ import annotations

from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lightstim.noise.config import NoiseConfig
from lightstim.frontend import export_all

app = FastAPI(title="LightStim API", version="0.1.0")

# Allow the Lovable dev server (and localhost in general) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class RotatedMemoryRequest(BaseModel):
    distance: int = Field(3, ge=2, le=11)
    rounds: int = Field(3, ge=1, le=20)
    basis: Literal["Z", "X"] = "Z"
    p_1q: float = Field(1e-3, ge=0, le=0.5)
    p_2q: float = Field(1e-3, ge=0, le=0.5)
    p_meas: float = Field(1e-3, ge=0, le=0.5)
    p_reset: float = Field(1e-3, ge=0, le=0.5)
    noise_model: Literal["circuit_level", "phenomenological", "code_capacity"] = "circuit_level"
    decompose_errors: bool = False


class TwoPatchLSRequest(BaseModel):
    distance1: int = Field(3, ge=2, le=9)
    distance2: int = Field(3, ge=2, le=9)
    interaction_type: Literal["ZZ", "XX"] = "ZZ"
    rounds: int = Field(3, ge=1, le=20)
    p_1q: float = Field(1e-3, ge=0, le=0.5)
    p_2q: float = Field(1e-3, ge=0, le=0.5)
    p_meas: float = Field(1e-3, ge=0, le=0.5)
    p_reset: float = Field(1e-3, ge=0, le=0.5)
    noise_model: Literal["circuit_level", "phenomenological", "code_capacity"] = "circuit_level"
    decompose_errors: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/api/circuit/rotated-memory")
def rotated_memory(req: RotatedMemoryRequest):
    """Build a rotated surface code memory experiment and return DEM + Timeline + DetSlice."""
    try:
        from lightstim.qec_code.surface_code.rotated import (
            RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
        )
        from lightstim.ir.qec_system import QECSystem
        from lightstim.ir.tracker import SyndromeTracker
        from lightstim.ir.builder import CircuitBuilder

        system = QECSystem()
        system.add_patch(RotatedSurfaceCode(distance=req.distance), name="main")
        tracker = SyndromeTracker(system.num_qubits, expected_num_logicals=system.num_logicals)
        builder = CircuitBuilder(tracker, system)
        se = RotatedSurfaceCodeExtractionBlock(system)

        builder.write_coordinates()
        builder.initialize({q: req.basis for q in system.data_indices}, n=system.num_qubits)
        builder.apply_syndrome_extraction(se.circuit, rounds=req.rounds)
        builder.apply_data_readout({q: req.basis for q in system.data_indices})

        noise = NoiseConfig(p_1q=req.p_1q, p_2q=req.p_2q, p_meas=req.p_meas, p_reset=req.p_reset)
        circuit = builder.build_noisy_circuit(noise, noise_model=req.noise_model)

        return export_all(
            circuit,
            source=f"rotated_memory_{req.basis.lower()}",
            distance=req.distance,
            rounds=req.rounds,
            noise_model=req.noise_model,
            physical_error_rate=req.p_2q,
            decompose_errors=req.decompose_errors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/circuit/two-patch-ls")
def two_patch_ls(req: TwoPatchLSRequest):
    """Build a two-patch unrotated surface code lattice surgery experiment."""
    try:
        from lightstim.protocols.two_patch_ls import TwoPatchLSExperiment

        # ZZ: patch2 placed above patch1 (vertical), XX: placed to the right (horizontal)
        if req.interaction_type == "ZZ":
            offset = (0.0, float(req.distance1 * 4))
            init1, meas1 = "X", "X"
            init2, meas2 = "Z", "Z"
        else:
            offset = (float(req.distance1 * 4), 0.0)
            init1, meas1 = "Z", "Z"
            init2, meas2 = "X", "X"

        noise = NoiseConfig(p_1q=req.p_1q, p_2q=req.p_2q, p_meas=req.p_meas, p_reset=req.p_reset)
        exp = TwoPatchLSExperiment(
            patch1_config={"distance": req.distance1},
            patch2_config={"distance": req.distance2},
            offset=offset,
            interaction_type=req.interaction_type,
            initial_state_patch1=init1,
            initial_state_patch2=init2,
            measure_state_patch1=meas1,
            measure_state_patch2=meas2,
            rounds=req.rounds,
            noise_params=noise,
            noise_model=req.noise_model,
        )
        circuit = exp.build()

        return export_all(
            circuit,
            source=f"unrotated_ls_{req.interaction_type.lower()}",
            distance=req.distance1,
            rounds=req.rounds,
            noise_model=req.noise_model,
            physical_error_rate=req.p_2q,
            decompose_errors=req.decompose_errors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
