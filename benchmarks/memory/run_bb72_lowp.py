"""bb_72_12_6 low-p extension: p=7e-4, 5e-4, 3e-4 — GPU BP+OSD, append to existing CSV."""
import contextlib, io
import pandas as pd
from pathlib import Path
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.BB_code import BBCode, BBCodeExtractionBlock
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

OUTPUT    = Path("benchmarks/memory/results/fig2_bb_codes_bb_72_12_6_gpu_bposd.csv")
CODE_NAME = "bb_72_12_6"
BB_CFG    = {"l": 6, "m": 6, "A": [[3,0],[0,1],[0,2]], "B": [[0,3],[1,0],[2,0]]}
D         = 6
P_VALUES  = [7e-4, 5e-4, 3e-4]

done_p = set()
if OUTPUT.exists():
    done_p = set(pd.read_csv(OUTPUT)["p"].values)
    print(f"Already done p: {sorted(done_p)}")

pending = [p for p in P_VALUES if not any(abs(p - dp) < 1e-12 for dp in done_p)]
print(f"Pending: {pending}\n")
if not pending:
    print("Nothing to do."); exit(0)

_sys = QECSystem()
_sys.add_patch(BBCode(**BB_CFG), name=CODE_NAME)
N_DATA  = len(BBCode(**BB_CFG).data_indices)
N_TOTAL = _sys.num_qubits
K       = 12

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("nv-qldpc-decoder", backend="gpu"),
    max_errors=100,
    max_shots=1_000_000_000,
    num_workers=1,
    print_progress=True,
)

for p in pending:
    system = QECSystem()
    system.add_patch(BBCode(**BB_CFG), name=CODE_NAME)
    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=BBCodeExtractionBlock,
            rounds=D, noise_params=noise,
            noise_model="circuit_level", basis="Z",
        )
        circuit = exp.build()

    meta = {"code": CODE_NAME, "p": p, "n_data": N_DATA, "n_total": N_TOTAL,
            "k": K, "figure": 2, "decoder_label": "gpu_bposd"}
    print(f"[p={p:.1e}]", flush=True)
    stats = pipeline.run(circuit, meta)
    row = {**meta, "shots": stats.shots, "errors": stats.errors,
           "logical_error_rate": stats.logical_error_rate,
           "seconds": stats.seconds, "decoder": stats.decoder}
    pd.DataFrame([row]).to_csv(OUTPUT, mode="a", header=not OUTPUT.exists(), index=False)
    print(f"  -> LER={stats.logical_error_rate:.3e}  errors={stats.errors}  shots={stats.shots:,}  t={stats.seconds:.1f}s\n")

print("Done.")
