"""Run with the Clifft interpreter, independently of LightStim imports."""

import json
from pathlib import Path
import sys


def main(config_path: Path) -> None:
    try:
        import clifft
    except ImportError as error:
        raise RuntimeError(
            "Clifft could not be imported. Install clifft==0.7.0 in your Python 3.12+ "
            "environment. To use a separate environment, set CLIFFT_PYTHON to its Python executable."
        ) from error
    import numpy as np

    config = json.loads(config_path.read_text(encoding="utf-8"))
    clifft.set_num_threads(config["threads"])
    program = clifft.compile(config["circuit_text"], normalize_syndromes=False)
    if program.peak_rank > config["max_peak_rank"]:
        raise ValueError(
            f"Circuit peak rank {program.peak_rank} exceeds max_peak_rank={config['max_peak_rank']}"
        )
    print(json.dumps({"widths": {
        "detectors": program.num_detectors,
        "observables": program.num_observables,
        "measurements": program.num_measurements,
        "expectations": program.num_exp_vals,
    }}))

    output_path = config_path.with_name("batch.npz")
    for line in sys.stdin:
        request = json.loads(line)
        result = clifft.sample(program, request["shots"], seed=request["seed"])
        np.savez(
            output_path,
            detectors=result.detectors,
            observables=result.observables,
            measurements=result.measurements,
            expectations=result.exp_vals,
        )
        print('{"ok": true}')


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(1)
