"""Run a resolved experiment with the existing LightStim simulation backend."""

import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from lightstim.noise.config import NoiseConfig
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline
from lightstim.simulation.decoder_backend.config import PipelineConfig
from lightstim.simulation.decoder_backend.post_select import get_post_select_detector_indices

from .results import append_result_row, canonical_json, config_digest, point_config_digest


def sample_full_postselection(circuit, *, max_shots, max_errors, batch_size, seed=0,
                              num_workers=1, target_observable_indices=None, observable_failure_mode="any"):
    """Count raw observable failures among shots with all detectors quiet."""
    if min(batch_size, max_errors, max_shots, num_workers) < 1:
        raise ValueError("Shot, error, worker and batch limits must be positive")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if observable_failure_mode not in {"any", "parity"}:
        raise ValueError(f"Unknown observable_failure_mode {observable_failure_mode!r}")
    circuit.detector_error_model(decompose_errors=False)
    nworkers = min(num_workers, (max_shots + batch_size - 1) // batch_size)
    seeds = ([seed] if nworkers == 1 else
             [int(s) for s in np.random.SeedSequence(seed).generate_state(nworkers, dtype=np.uint64)])
    samplers = [circuit.compile_detector_sampler(seed=s) for s in seeds]

    def sample_batch(sampler, size):
        dets, obs = sampler.sample(size, separate_observables=True)
        keep = ~dets.any(axis=1)
        selected = obs[keep]
        if target_observable_indices is not None:
            selected = selected[:, target_observable_indices]
        failures = (np.logical_xor.reduce(selected, axis=1) if observable_failure_mode == "parity"
                    else selected.any(axis=1))
        return size, int(keep.sum()), int(failures.sum())

    shots = accepted = errors = 0
    with (ThreadPoolExecutor(nworkers) if nworkers > 1 else nullcontext()) as pool:
        while shots < max_shots and errors < max_errors:
            sizes = [min(batch_size, max_shots - offset) for offset in
                     range(shots, min(shots + nworkers * batch_size, max_shots), batch_size)]
            batches = (pool.map(sample_batch, samplers, sizes) if pool
                       else [sample_batch(samplers[0], sizes[0])])
            for attempted, kept, failed in batches:
                shots += attempted
                accepted += kept
                errors += failed
    return {
        "shots": shots, "accepted": accepted, "rejected": shots - accepted, "errors": errors,
        "acceptance": accepted / shots, "logical_error_rate": errors / accepted if accepted else math.nan,
    }


def _check_indices(name, values, size):
    if values is not None and (len(set(values)) != len(values) or
                               any(i < 0 or i >= size for i in values)):
        raise ValueError(f"{name} must contain unique indices in [0, {size})")
    return values


def _sample(circuit, point, *, progress=False, progress_interval=30.0):
    sim = point["simulation"]
    mode = sim["mode"]
    policy = sim.get("observable_failure_mode", "any")
    target = _check_indices("target_observable_indices", sim.get("target_observable_indices"),
                            circuit.num_observables)
    settings = {k: sim.get(k, getattr(PipelineConfig, k)) for k in
                ("max_shots", "max_errors", "batch_size", "seed", "num_workers")}
    ps = sim.get("postselection", {})
    if mode != "hybrid" and ps:
        raise ValueError("simulation.postselection requires hybrid mode")
    if mode == "full_postselection":
        return sample_full_postselection(circuit, target_observable_indices=target,
                                         observable_failure_mode=policy, **settings)
    det = _check_indices("detector_indices", ps.get("detector_indices", []), circuit.num_detectors)
    obs = _check_indices("observable_indices", ps.get("observable_indices", []), circuit.num_observables)
    if ps.get("region"):
        if ps["region"] != "tagged":
            raise ValueError(f"Unknown postselection region {ps['region']!r} for this compiler. "
                             "Use detector_indices, observable_indices, or region 'tagged'.")
        det = sorted(set(det) | set(get_post_select_detector_indices(circuit)))
    if mode == "hybrid" and not (det or obs):
        raise ValueError("Hybrid mode requires a nonempty postselection region")
    pipeline = SimulationPipeline(
        decoder_config=DecoderConfig.from_spec(point["decoder"]), **settings,
        observable_failure_mode=policy, target_observable_indices=target,
        post_select_detector_indices=det, post_select_observable_indices=obs, print_progress=progress,
        progress_interval_sec=progress_interval,
    )
    stats = pipeline.run(circuit)
    return {"shots": stats.shots, "errors": stats.errors, "accepted": stats.post_selected_shots,
            "logical_error_rate": stats.logical_error_rate if stats.post_selected_shots else None}


def _result_metadata(point, system, circuit):
    parameters, noise, sim = point["parameters"], point["noise"], point["simulation"]
    rates = NoiseConfig.from_spec(noise)
    decode = sim["mode"] != "full_postselection"
    decoder = DecoderConfig.from_spec(point["decoder"]) if decode else None
    decoder_params = {}
    if decode:
        decoder_params = ({"chain": point["decoder"]["chain"]} if "chain" in point["decoder"]
                          else decoder.params)
    patches = point["system"]["patches"]
    row = {
        "protocol_id": point["protocol_id"], "parameters": parameters,
        "sweep_coordinates": point.coordinates, "p": noise["p"], "noise_model": noise["model"],
        "code": ",".join(dict.fromkeys(p["code"] for p in patches)), "mode": sim["mode"],
        "decoder_name": decoder.name if decode else "none", "backend": decoder.backend if decode else None,
        "decoder_params": decoder_params, "decoder_time_limit": decoder_params.get("time_limit"),
        "on_decode_failure": decoder.on_decode_failure if decode else None,
        "noise_custom_params": rates.custom_params,
        "postselection_region": sim.get("postselection", {}).get("region"),
        "seed": sim.get("seed", PipelineConfig.seed),
        "observable_failure_mode": sim.get("observable_failure_mode", "any"),
        "n_data": len(system.data_indices), "n_total": circuit.num_qubits, "k": system.num_logicals,
    }
    for name in ("p_1q", "p_2q", "p_meas", "p_reset", "p_idle"):
        row[name] = getattr(rates, name)
    for name in ("p_idle", "p_1q"):
        row[f"{name}_mode"] = "fixed" if noise.get("overrides", {}).get(name) is not None else "sweep"
    row.update({k: parameters[k] for k in ("basis", "rounds", "distance") if k in parameters})
    return row


def run_experiment(config, output, *, source_config=None, progress=False, progress_interval=30.0):
    """Run each sweep point and checkpoint it to CSV. Return the result path."""
    from . import compile, expand_sweep

    path = Path(output)
    spec = config.spec
    overrides = {k: v for k, v in spec.items() if source_config and v != source_config.spec.get(k)}
    run_sha = config_digest(spec)
    done = set()
    if path.exists():
        with path.open(encoding="utf-8", newline="") as src:
            done = {row["sweep_coordinates"] for row in csv.DictReader(src)
                    if row.get("run_config_sha256") == run_sha}
    last_message = float("-inf")
    for point in expand_sweep(config):
        coordinates = canonical_json(point.coordinates)
        if coordinates in done:
            continue
        start = perf_counter()
        if progress and start - last_message >= progress_interval:
            print(f"Running {spec['protocol_id']} {coordinates}", flush=True)
            last_message = start
        circuit, system = compile(point)
        result = _sample(circuit, point, progress=progress, progress_interval=progress_interval)
        row = _result_metadata(point, system, circuit)
        row.update(result, seconds=perf_counter() - start, config_sha256=point_config_digest(spec, point["parameters"]),
                   run_config_sha256=run_sha, config_overrides=overrides)
        # For memory the rounds parameter is the number of recovery cycles in a shot.
        # LS CNOT uses 'rounds' separately in each of three different stages.
        if spec["protocol_id"] == "rotated-surface-memory" and row.get("rounds") and row["accepted"]:
            rate = row["logical_error_rate"]
            rounds = row["rounds"]
            row["block_failure_se"] = math.sqrt(rate * (1 - rate) / row["accepted"])
            row["logical_failure_rate_per_round"] = 1 - (1 - rate) ** (1 / rounds)
            row["logical_failure_rate_per_round_se"] = (
                (1 - rate) ** (1 / rounds - 1) * row["block_failure_se"] / rounds if rate < 1 else 0
            )
        append_result_row(path, row)
        done.add(coordinates)
    return path


def run_config_cli(parser):
    """Dispatch an explicit --config before the runner parses legacy required flags."""
    import sys

    parser.add_argument("--config", metavar="JSON", help="Run an experiment config with explicit CLI overrides")
    if not any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]):
        return False
    from .cli import run_cli

    run_cli(sys.argv[1:], prog=parser.prog)
    return True
