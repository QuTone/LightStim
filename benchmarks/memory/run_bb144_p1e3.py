"""bb_144_12_12 at p=1e-3 — GPU BP+OSD, high-statistics run.
Output: eval/memory_benchmark/results/fig2_bb_codes_bb_144_p1e3.csv
Merge into fig2_bb_codes_bb_144_12_12_gpu_bposd.csv after verification.
"""
import contextlib, io, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
from pathlib import Path
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

OUTPUT    = Path("eval/memory_benchmark/results/fig2_bb_codes_bb_144_p1e3.csv")
CODE_NAME = "bb_144_12_12"
BB_CFG    = {"l": 12, "m": 6, "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]]}
D         = 12
P         = 1e-3

if OUTPUT.exists():
    df = pd.read_csv(OUTPUT)
    done = df[(df["code"]==CODE_NAME) & (abs(df["p"]-P)<1e-12)]
    if not done.empty:
        print(f"Already done: {len(done)} row(s)")
        print(done[["shots","errors","logical_error_rate"]].to_string())
        exit(0)

system = QECSystem()
system.add_patch(BBCode(**BB_CFG), name=CODE_NAME)
N_DATA  = len(BBCode(**BB_CFG).data_indices)
N_TOTAL = system.num_qubits
K       = 12

noise = NoiseConfig(p_idle=P, p_1q=P, p_2q=P, p_meas=P, p_reset=P)
print(f"Building circuit: {CODE_NAME}  p={P:.1e}  rounds={D}")
with contextlib.redirect_stdout(io.StringIO()):
    exp = MemoryExperiment(
        qec_system=system,
        extraction_block_class=BBCodeExtractionBlock,
        rounds=D, noise_params=noise,
        noise_model="circuit_level", basis="Z",
    )
    circuit = exp.build()
print(f"  qubits={circuit.num_qubits}  detectors={circuit.num_detectors}  obs={circuit.num_observables}")

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig(
        "nv-qldpc-decoder",
        backend="gpu",
        params={
            "max_iterations": 1000,
            "osd_order": 10,
            "bp_method": "min_sum",
            "ms_scaling_factor": 0,
            "osd_method": "osd_cs",
            "use_osd": True,
        },
    ),
    max_errors=100,
    max_shots=1_000_000_000,
    num_workers=1,
    print_progress=True,
    progress_interval_sec=30.0,
)

meta = {"code": CODE_NAME, "p": P, "n_data": N_DATA, "n_total": N_TOTAL,
        "k": K, "figure": 2, "decoder_label": "gpu_bposd"}
print(f"\nRunning GPU BPOSD (max_shots=1e9, max_errors=100)...")
stats = pipeline.run(circuit, meta)

row = {**meta, "shots": stats.shots, "errors": stats.errors,
       "logical_error_rate": stats.logical_error_rate,
       "seconds": stats.seconds, "decoder": stats.decoder}
pd.DataFrame([row]).to_csv(OUTPUT, mode="a", header=not OUTPUT.exists(), index=False)
print(f"\nResult: LER={stats.logical_error_rate:.4e}  errors={stats.errors}  shots={stats.shots:,}  t={stats.seconds:.1f}s")
print(f"Saved to {OUTPUT}")
