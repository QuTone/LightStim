"""Color Code memory benchmark: d=3,5,7, p=1e-3, Z basis, MWPF decoder."""
import contextlib, io
import pandas as pd
from pathlib import Path
from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
from lightstim.protocols.memory import MemoryExperiment
from lightstim.simulation.decoder_backend import SimulationPipeline, DecoderConfig

RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT  = RESULTS / "fig3_color_code.csv"

pipeline = SimulationPipeline(
    decoder_config=DecoderConfig("mwpf", backend="cpu"),
    max_errors=100,
    max_shots=1_000_000_000,
    num_workers=16,
    print_progress=True,
)

rows = []
for d in [3, 5, 7]:
    p = 1e-3
    system = QECSystem()
    code = ColorCode(distance=d)
    system.add_patch(code, name="patch")
    n_total = system.num_qubits
    n_data  = len(code.data_indices)

    noise = NoiseConfig(p_idle=p, p_1q=p, p_2q=p, p_meas=p, p_reset=p)
    with contextlib.redirect_stdout(io.StringIO()):
        exp = MemoryExperiment(
            qec_system=system,
            extraction_block_class=ColorCodeExtractionBlock,
            rounds=d,
            noise_params=noise,
            noise_model="circuit_level",
            basis="Z",
        )
        circuit = exp.build()

    meta = {"code": "color_code", "distance": d, "p": p,
            "n_data": n_data, "n_total": n_total, "k": 1, "rounds": d}
    print(f"\n[d={d}] n_total={n_total}  n_data={n_data}", flush=True)
    stats = pipeline.run(circuit, meta)
    row = {**meta, "shots": stats.shots, "errors": stats.errors,
           "logical_error_rate": stats.logical_error_rate,
           "seconds": stats.seconds, "decoder": "mwpf"}
    rows.append(row)
    print(f"  → LER={stats.logical_error_rate:.3e}  errors={stats.errors}  shots={stats.shots:,}  t={stats.seconds:.1f}s", flush=True)

df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv(OUTPUT, index=False)
print(f"\nSaved {OUTPUT}")
