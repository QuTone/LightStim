"""Compare H-family schedules under all-detector memory postselection.

Run from the repository root, e.g.:
  PYTHONPATH=. python benchmarks/memory/audit_h_code_se.py --max-n 64 --output /tmp/h-se-audit.json

This is a finite-size circuit audit, not a proof of open-output extraction
fault tolerance. The default scans every even size from 6 to 64, both bases,
1/2/3 rounds, and zero/nonzero idle noise for both extraction schedules.
"""

import argparse
import json
from pathlib import Path

import stim

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.H_code import HCode, HCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock


def audit(max_n=64):
    rows = []
    counterexample = None
    for n in range(6, max_n + 1, 2):
        for schedule, block in (("dedicated", HCodeExtractionBlock),
                                ("coloration", GenericCSSColorationExtractionBlock)):
            system = QECSystem()
            system.add_patch(HCode(n), name="h")
            depth = block(system).cnot_depth
            for basis in ("X", "Z"):
                for rounds in (1, 2, 3):
                    for idle in (False, True):
                        p = .001
                        c = MemoryExperiment(
                            qec_patch=HCode(n), extraction_block_class=block,
                            basis=basis, rounds=rounds,
                            noise_params=NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p,
                                                     p_idle=p if idle else 0),
                        ).build()
                        dem = c.detector_error_model(decompose_errors=False)
                        errors = [e for e in dem.flattened() if e.type == "error"]
                        bad = [e for e in errors
                               if any(t.is_logical_observable_id() for t in e.targets_copy())
                               and not any(t.is_relative_detector_id() for t in e.targets_copy())]
                        witness = c.shortest_graphlike_error(canonicalize_circuit_errors=True)
                        assert all(e.circuit_error_locations for e in witness)
                        if len(witness) == 2:
                            a, b = [e.circuit_error_locations[0] for e in witness]
                            if a.stack_frames == b.stack_frames:
                                ta, tb = a.instruction_targets, b.instruction_targets
                                assert (ta.target_range_end <= tb.target_range_start
                                        or tb.target_range_end <= ta.target_range_start)
                        assert len(witness) in (1, 2)
                        distance = 1 if bad else 2
                        assert distance == len(witness)
                        rows.append(dict(n=n, schedule=schedule, basis=basis, rounds=rounds,
                                         idle=idle, cnot_depth=depth,
                                         undetected_single_fault_signatures=len(bad),
                                         circuit_fault_distance=distance))
                        if schedule == "coloration" and n == 8 and basis == "Z" and bad and counterexample is None:
                            counterexample = str(c.explain_detector_error_model_errors(
                                dem_filter=stim.DetectorErrorModel(str(bad[0])),
                                reduce_to_one_representative_error=True,
                            )[0])
        summary = {name: sorted({r['circuit_fault_distance'] for r in rows
                                if r['n'] == n and r['schedule'] == name})
                   for name in ('dedicated', 'coloration')}
        print(f"n={n}: {summary}", flush=True)
    return dict(stim_version=stim.__version__, max_n=max_n, rows=rows,
                h8_coloration_counterexample=counterexample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    HCode(args.max_n)  # Validate the size before starting.
    result = audit(args.max_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(result['rows'])} circuits; results: {args.output}", flush=True)
