"""Portable, fixed-shot BB PPM benchmark with resumable detailed Relay decoding.

This deliberately uses Relay's detailed API instead of changing the generic
SimulationPipeline's failure accounting. A logical failure is an observable
mismatch; an independently checked unsatisfied syndrome is reported separately.
No result from a completion-ordered subset is published as a final LER.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time
import uuid


DEFAULT_DECODER_PARAMS = dict(
    alpha=0.8, gamma0=0.5, pre_iter=320, num_sets=600, set_max_iter=60,
    gamma_dist_interval=(-0.1, 0.5), stop_nconv=5,
    stopping_criterion="nconv", logging=False, seed=0,
)
DEFAULT_P_VALUES = (0.0001, 0.0005, 0.001)
DEFAULT_OUTPUT = Path(__file__).parent / "results/bb_ppm_results.csv"
DEFAULT_RUN_DIR = Path(__file__).parent / "results/bb_ppm_run"
MATRIX_FILES = ("check_matrix.npz", "observable_matrix.npz", "error_priors.npy")
SAMPLE_FILES = ("detectors.npy", "observables.npy")
CSV_FIELDS = (
    "p", "shots", "logical_failures", "nonconverged", "converged_correct",
    "converged_logical_failure", "unconverged_correct_observables",
    "unconverged_logical_failure", "ler", "ler_95_exact_two_sided_low",
    "ler_95_exact_two_sided_high", "ler_95_one_sided_upper",
    "nonconvergence_rate", "sum_decode_cpu_seconds",
    "source_manifest_sha256", "decoder_manifest_sha256",
)
_WORKER = {}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    import numpy as np
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _lock(path):
    """One writer per run; file locks are released after a process exits."""
    with Path(path).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def available_workers():
    """Conservative process budget; every worker has one numerical thread."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return min(61 if os.name == "nt" else 192, cpus)


def normalize_decoder_params(overrides=None):
    """Validate the modern Relay profile; restore JSON arrays to Rust tuples."""
    overrides = {} if overrides is None else overrides
    _require(isinstance(overrides, dict) and not set(overrides) - set(DEFAULT_DECODER_PARAMS),
             "Unknown Relay parameter; use the documented detailed-decoder settings")
    params = dict(DEFAULT_DECODER_PARAMS, **overrides)
    for name, low, high in (("alpha", 0, 1), ("gamma0", 0, 1)):
        value = params[name]
        _require(type(value) in (int, float) and math.isfinite(value) and
                 low <= value <= high and (name != "alpha" or value > 0) and
                 (name != "gamma0" or value < 1), f"Invalid {name}")
    interval = params["gamma_dist_interval"]
    _require(isinstance(interval, (list, tuple)) and len(interval) == 2 and
             all(type(v) in (int, float) and math.isfinite(v) and -0.5 <= v < 1 for v in interval)
             and interval[0] < interval[1], "Invalid gamma_dist_interval")
    params["gamma_dist_interval"] = tuple(interval)
    for name, minimum in (("pre_iter", 1), ("num_sets", 0), ("set_max_iter", 1), ("stop_nconv", 1)):
        _require(type(params[name]) is int and params[name] >= minimum, f"Invalid {name}")
    _require(type(params["seed"]) is int and params["seed"] == 0 and
             params["logging"] is False and params["stopping_criterion"] == "nconv",
             "This benchmark fixes decoder seed=0, logging=False, stopping_criterion='nconv'")
    return params


def _versions():
    return {name: importlib.metadata.version(name) for name in ("stim", "relay-bp", "numpy", "scipy")}


def _source_hashes():
    """Fail closed on implementation changes, independent of checkout location."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "lightstim").rglob("*.py"))
    files += [Path(__file__).resolve()]
    return {p.relative_to(root).as_posix(): _sha(p) for p in files}


def compiled_plan_sha256(plans):
    # This is the executable-plan hash used by the BB PPM Notebook.
    payload = [{k: v for k, v in asdict(plan).items()
                if k not in ("search_trace", "search_status")} for plan in plans]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _audit_dem(dem):
    """Use the same upstream adapter as the audited run, checking every column."""
    import numpy as np
    from relay_bp.stim.sinter.check_matrices import CheckMatrices
    errors, detector_actions = defaultdict(list), {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        detectors, observables = set(), set()
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                detectors.symmetric_difference_update([target.val])
            elif target.is_logical_observable_id():
                observables.symmetric_difference_update([target.val])
            else:
                raise ValueError("Expected a strict, undecomposed DEM without separators")
        key = tuple(sorted(detectors)), tuple(sorted(observables))
        _require(bool(key[0]), "Relay adapter cannot preserve detector-free logical faults")
        _require(key[0] not in detector_actions or detector_actions[key[0]] == key[1],
                 "Same-detector/different-observable faults are unsupported by this Relay adapter")
        detector_actions[key[0]] = key[1]
        probability = inst.args_copy()[0]
        _require(0 < probability < 0.5, "Unsupported deterministic/high-probability DEM fault")
        errors[key].append(probability)
    _require(bool(errors), "Empty DEM: choose a positive noise point for this decoding benchmark")
    matrices = CheckMatrices.from_dem(dem)  # Default None is intentional in relay-bp 0.2.2.
    _require(matrices.syndrome_bias is None and matrices.observables_bias is None,
             "Unexpected Relay matrix bias")
    H, O = matrices.check_matrix.tocsc(), matrices.observables_matrix.tocsc()
    priors = np.asarray(matrices.error_priors, dtype=np.float64)
    _require(H.shape == (dem.num_detectors, len(errors)) and
             O.shape == (dem.num_observables, len(errors)), "DEM matrix dimensions changed")
    matched = set()
    for col in range(H.shape[1]):
        key = (tuple(sorted(H.indices[H.indptr[col]:H.indptr[col + 1]])),
               tuple(sorted(O.indices[O.indptr[col]:O.indptr[col + 1]])))
        _require(key in errors and key not in matched, "Missing or duplicated DEM matrix column")
        expected = 0.5 * (1 - np.prod(1 - 2 * np.asarray(errors[key])))
        _require(np.isclose(priors[col], expected, rtol=1e-12, atol=5e-15), "DEM prior parity mismatch")
        matched.add(key)
    _require(np.all(H.data == 1) and np.all(O.data == 1), "Nonbinary DEM matrices")
    return H, O, priors


def _verify_inputs(source):
    import numpy as np
    import scipy.sparse as sparse
    source = Path(source)
    manifest = _read(source / "manifest.json")
    _require(manifest["status"] == "prepared", "Saved inputs are incomplete")
    _require(set((*MATRIX_FILES, *SAMPLE_FILES)) <= set(manifest["artifact_hashes"]),
             "Input manifest does not hash every matrix and sample array")
    for name, expected in manifest["artifact_hashes"].items():
        _require(_sha(source / name) == expected, f"Saved input changed: {name}")
    H = sparse.load_npz(source / MATRIX_FILES[0]).tocsc()
    O = sparse.load_npz(source / MATRIX_FILES[1]).tocsc()
    priors = np.load(source / MATRIX_FILES[2], allow_pickle=False)
    detectors = np.load(source / SAMPLE_FILES[0], mmap_mode="r", allow_pickle=False)
    observables = np.load(source / SAMPLE_FILES[1], mmap_mode="r", allow_pickle=False)
    n = manifest["shots"]
    _require(type(n) is int and n > 0, "Invalid saved sample count")
    _require(H.shape[1] == O.shape[1] == len(priors) and H.shape[1] > 0 and O.shape[0] > 0,
             "Invalid matrix dimensions")
    _require(np.all(H.data == 1) and np.all(O.data == 1) and
             np.isfinite(priors).all() and np.all((priors > 0) & (priors < 0.5)), "Invalid H/O/priors")
    for values, bits in ((detectors, H.shape[0]), (observables, O.shape[0])):
        _require(values.dtype == np.uint8 and values.shape == (n, (bits + 7) // 8),
                 "Invalid packed sample dimensions")
        if bits % 8:
            _require(not np.any(values[:, -1] >> (bits % 8)), "Nonzero packed padding bits")
    return manifest, H, O, priors, detectors, observables


def _prepare_point(clean, point, *, p, shots, seed, point_index, batch_size,
                   noise_schedule=None):
    import numpy as np
    import scipy.sparse as sparse
    import stim
    from lightstim.noise.config import NoiseConfig
    from lightstim.noise.injector import NoiseInjector
    from lightstim.qec_code.BB_code.ppm.frame import strip_frame_activity
    source = point / "inputs"
    source.mkdir(parents=True, exist_ok=True)
    config = NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p)
    schedule = clean if noise_schedule is None else noise_schedule
    _require(strip_frame_activity(schedule) == clean,
             "Noise schedule differs from the clean BB circuit")
    noisy = NoiseInjector.from_circuit_level_with_idling(
        config, list(range(clean.num_qubits))).inject_noise(schedule)
    noisy = strip_frame_activity(noisy)
    _require(noisy.without_noise() == clean, "Noise injection changed the clean circuit")
    _require(stim.Circuit(str(noisy)) == noisy, "Circuit text roundtrip changed a noise argument")
    circuit_bytes = str(noisy).encode("utf-8")
    if (source / "manifest.json").exists():
        saved = _verify_inputs(source)[0]
        expected = dict(p=p, shots=shots, sample_seed=seed, point_index=point_index, batch_size=batch_size)
        _require(all(saved[key] == value for key, value in expected.items()) and
                 saved["artifact_hashes"].get("circuit.stim") == hashlib.sha256(circuit_bytes).hexdigest(),
                 "Prepared point differs from the requested noise circuit, count, or seed policy")
        return saved
    (source / "circuit.stim").write_bytes(circuit_bytes)
    dem = noisy.detector_error_model(decompose_errors=False, allow_gauge_detectors=False,
                                    approximate_disjoint_errors=False, ignore_decomposition_failures=False)
    H, O, priors = _audit_dem(dem)
    sparse.save_npz(source / MATRIX_FILES[0], H)
    sparse.save_npz(source / MATRIX_FILES[1], O)
    np.save(source / MATRIX_FILES[2], priors, allow_pickle=False)
    batches = point / "sample_batches"
    batches.mkdir(exist_ok=True)
    records = []
    for index, start in enumerate(range(0, shots, batch_size)):
        count = min(batch_size, shots - start)
        batch_seed = int(np.random.SeedSequence([seed, point_index, index]).generate_state(1, dtype=np.uint64)[0])
        path = batches / f"batch_{index:06d}.npz"
        marker = path.with_suffix(".json")
        identity = dict(seed=batch_seed, start=start, shots=count, circuit_sha256=_sha(source / "circuit.stim"))
        if marker.exists():
            record = _read(marker)
            _require(record["identity"] == identity and record["sha256"] == _sha(path), "Saved sample batch changed")
        else:
            dets, obs = noisy.compile_detector_sampler(seed=batch_seed).sample(
                count, separate_observables=True, bit_packed=True)
            _atomic_npz(path, detectors=dets, observables=obs)
            record = dict(identity=identity, sha256=_sha(path))
            _atomic_json(marker, record)
        records.append(record)
    # Merge disk checkpoints into memory-mapped packed arrays; workers do not
    # each copy the entire unpacked campaign into RAM.
    for key, name, bits in (("detectors", SAMPLE_FILES[0], H.shape[0]), ("observables", SAMPLE_FILES[1], O.shape[0])):
        destination = np.lib.format.open_memmap(source / name, mode="w+", dtype=np.uint8,
                                               shape=(shots, (bits + 7) // 8))
        for index, record in enumerate(records):
            with np.load(batches / f"batch_{index:06d}.npz", allow_pickle=False) as data:
                start, n = record["identity"]["start"], record["identity"]["shots"]
                _require(data[key].shape == (n, (bits + 7) // 8), "Sample batch dimensions changed")
                destination[start:start + n] = data[key]
        destination.flush()
        del destination
    names = (*MATRIX_FILES, *SAMPLE_FILES, "circuit.stim")
    manifest = dict(status="prepared", p=p, shots=shots, batch_size=batch_size,
                    sample_seed=seed, point_index=point_index, batches=records,
                    bit_order="little", artifact_hashes={name: _sha(source / name) for name in names},
                    dimensions=dict(detectors=H.shape[0], observables=O.shape[0], mechanisms=H.shape[1]))
    _atomic_json(source / "manifest.json", manifest)
    _verify_inputs(source)
    return manifest


def decode_one(H, O, priors, syndrome, truth, decoder_params=None):
    """Decode one saved input, independently recomputing Hc and Oc.

    This small API also supports paired regression against historical samples;
    it never samples a circuit. Call in a worker for single-thread isolation.
    """
    import numpy as np
    import relay_bp
    params = normalize_decoder_params(decoder_params)
    decoder = relay_bp.RelayDecoderF64(H, error_priors=priors, **params)
    runner = relay_bp.ObservableDecoderRunner(decoder, O, include_decode_result=True)
    syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
    start, cpu = time.perf_counter(), time.process_time()
    result = runner.decode_observables_detailed_batch(syndrome[None, :], parallel=False, progress_bar=False)[0]
    seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu
    physical = result.physical_decode_result
    correction = np.asarray(physical.decoding, dtype=np.uint8)
    _require(correction.shape == (H.shape[1],) and np.isin(correction, [0, 1]).all(), "Invalid Relay correction")
    predicted_det = np.asarray(H @ correction, dtype=np.uint8).ravel() % 2
    predicted_obs = np.asarray(O @ correction, dtype=np.uint8).ravel() % 2
    residual = predicted_det ^ syndrome
    wrong = predicted_obs ^ truth
    _require(np.array_equal(predicted_det, physical.decoded_detectors) and
             np.array_equal(predicted_obs, result.observables) and
             bool(result.converged) == bool(physical.success) == (not residual.any()),
             "Relay results disagree with independently recomputed Hc/Oc")
    iterations = int(result.iterations)
    _require(0 <= iterations <= params["pre_iter"] + params["num_sets"] * params["set_max_iter"],
             "Relay iteration count exceeds its configured budget")
    row = dict(converged=bool(result.converged), logical_failure=bool(wrong.any()),
               syndrome_residual_weight=int(residual.sum()), wrong_observable_count=int(wrong.sum()),
               total_bp_iterations=iterations, posterior_nonfinite_count=int((~np.isfinite(physical.posterior_ratios)).sum()),
               decode_seconds=seconds, decode_cpu_seconds=cpu_seconds)
    payload = dict(correction_packed=np.packbits(correction, bitorder="little"),
                   predicted_observables=predicted_obs, syndrome_residual=residual)
    return row, payload


def _worker_init(source, destination, identity, parameters):
    # The helper has no numerical imports at module scope. A spawned worker
    # sets its environment before importing NumPy, SciPy or the Rayon library.
    for name in ("RAYON_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    import numpy as np
    import scipy.sparse as sparse
    source = Path(source)
    _WORKER.update(H=sparse.load_npz(source / MATRIX_FILES[0]).tocsc(),
                   O=sparse.load_npz(source / MATRIX_FILES[1]).tocsc(),
                   priors=np.load(source / MATRIX_FILES[2], allow_pickle=False),
                   detectors=np.load(source / SAMPLE_FILES[0], mmap_mode="r", allow_pickle=False),
                   observables=np.load(source / SAMPLE_FILES[1], mmap_mode="r", allow_pickle=False),
                   destination=Path(destination), identity=identity, parameters=parameters)


def _worker(shot):
    import numpy as np
    state = _WORKER
    folder = state["destination"] / "per_shot" / f"shot_{shot:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_json(folder / "running.json", dict(shot_index=shot, pid=os.getpid(), started_utc=_utc()))
    syndrome = np.unpackbits(state["detectors"][shot], bitorder="little")[:state["H"].shape[0]]
    truth = np.unpackbits(state["observables"][shot], bitorder="little")[:state["O"].shape[0]]
    row, payload = decode_one(state["H"], state["O"], state["priors"], syndrome, truth, state["parameters"])
    row["shot_index"] = shot
    _atomic_npz(folder / "correction.npz", **payload)
    _atomic_json(folder / "complete.json", dict(identity=state["identity"], row=row,
                 payload_sha256=_sha(folder / "correction.npz"), finished_utc=_utc()))
    return row


def _read_checkpoint(folder, shot, identity, H, O, detectors, observables):
    import numpy as np
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    saved = _read(marker)
    _require(saved["identity"] == identity and saved["row"]["shot_index"] == shot and
             saved["payload_sha256"] == _sha(folder / "correction.npz"), "Shot checkpoint identity changed")
    with np.load(folder / "correction.npz", allow_pickle=False) as payload:
        packed = payload["correction_packed"]
        _require(packed.dtype == np.uint8 and packed.shape == ((H.shape[1] + 7) // 8,), "Invalid saved correction")
        correction = np.unpackbits(packed, bitorder="little")[:H.shape[1]]
        obs = np.asarray(O @ correction, dtype=np.uint8).ravel() % 2
        residual = (np.asarray(H @ correction, dtype=np.uint8).ravel() % 2) ^ np.unpackbits(detectors[shot], bitorder="little")[:H.shape[0]]
        wrong = obs ^ np.unpackbits(observables[shot], bitorder="little")[:O.shape[0]]
        row = saved["row"]
        _require(np.array_equal(obs, payload["predicted_observables"]) and
                 np.array_equal(residual, payload["syndrome_residual"]) and
                 type(row["converged"]) is bool and row["converged"] == (not residual.any()) and
                 type(row["logical_failure"]) is bool and row["logical_failure"] == bool(wrong.any()) and
                 row["syndrome_residual_weight"] == int(residual.sum()) and
                 row["wrong_observable_count"] == int(wrong.sum()), "Saved Hc/Oc or result flags changed")
    _require(math.isfinite(row["decode_cpu_seconds"]) and row["decode_cpu_seconds"] >= 0,
             "Invalid saved CPU duration")
    return row


def decode_saved_inputs(source, destination, *, decoder_params=None, workers=None, print_progress=False):
    """Return (rows, complete, stop_requested) after one prepared spawn-pool run."""
    source, destination = Path(source), Path(destination)
    parameters = normalize_decoder_params(decoder_params)
    workers = available_workers() if workers is None else workers
    _require(type(workers) is int and 1 <= workers <= available_workers(), "Invalid worker CPU budget")
    manifest, H, O, _, detectors, observables = _verify_inputs(source)
    destination.mkdir(parents=True, exist_ok=True)
    with _lock(destination / ".lock"):
        identity = dict(input_manifest_sha256=_sha(source / "manifest.json"),
                        parameters=json.loads(_canonical(parameters)), versions=_versions(),
                        source_sha256=_sha(__file__), input_hashes=manifest["artifact_hashes"])
        identity_sha = _fingerprint(identity)
        path = destination / "manifest.json"
        previous = _read(path) if path.exists() else {}
        _require(not previous or previous["identity"] == identity, "Decode configuration changed; use a new run directory")
        done = {}
        for shot in range(manifest["shots"]):
            row = _read_checkpoint(destination / "per_shot" / f"shot_{shot:06d}", shot,
                                   identity_sha, H, O, detectors, observables)
            if row is not None:
                done[shot] = row
        if len(done) == manifest["shots"] and previous.get("status") == "complete":
            # Verification is not a new decoding attempt. Keep the published
            # provenance hash stable when no work was required.
            return [done[i] for i in range(manifest["shots"])], True, False
        pending = iter(shot for shot in range(manifest["shots"]) if shot not in done)
        stop, errors = [False], []
        old_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        run = dict(identity=identity, identity_sha256=identity_sha, status="decoding", shots=manifest["shots"],
                   completed=len(done), attempts=previous.get("attempts", []) + [
                       dict(started_utc=_utc(), worker_processes=workers, threads_per_worker=1, start_method="spawn")])
        _atomic_json(path, run)
        try:
            if len(done) < manifest["shots"]:
                with ProcessPoolExecutor(max_workers=min(workers, manifest["shots"] - len(done)),
                        mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init,
                        initargs=(str(source), str(destination), identity_sha, parameters)) as pool:
                    active = {}
                    exhausted = False
                    last_print = time.monotonic()
                    while active or not exhausted:
                        while len(active) < workers and not exhausted and not stop[0]:
                            shot = next(pending, None)
                            if shot is None:
                                exhausted = True
                            else:
                                active[pool.submit(_worker, shot)] = shot
                        if not active:
                            break
                        finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                        for future in finished:
                            shot = active.pop(future)
                            try:
                                done[shot] = future.result()
                            except Exception as exc:
                                errors.append(dict(shot_index=shot, error=repr(exc)))
                                stop[0] = True
                        _atomic_json(destination / "progress.json", dict(updated_utc=_utc(), completed=len(done),
                            total=manifest["shots"], active=len(active), runtime_errors=errors, stop_requested=stop[0]))
                        if print_progress and time.monotonic() - last_print >= 10:
                            print(f"Completed {len(done)}/{manifest['shots']} shots; {len(active)} active", flush=True)
                            last_print = time.monotonic()
            rows = [done[i] for i in sorted(done)]
            run.update(status="complete" if len(done) == manifest["shots"] else "paused", completed=len(done),
                       updated_utc=_utc(), runtime_errors=errors)
            if errors:
                run["status"] = "failed"
            _atomic_json(destination / "shots.json", rows)
            _atomic_json(path, run)
            _verify_inputs(source)
            if errors:
                raise RuntimeError(f"Decoding failed; saved checkpoints remain resumable: {errors}")
            return rows, run["status"] == "complete", stop[0]
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def summarize(rows, requested):
    """Only complete fixed-N cohorts have a reported LER and binomial interval."""
    from scipy.stats import beta
    _require(len(rows) == requested and sorted(r["shot_index"] for r in rows) == list(range(requested)),
             "A final result requires every requested shot exactly once")
    categories = Counter((r["converged"], r["logical_failure"]) for r in rows)
    errors = sum(r["logical_failure"] for r in rows)
    invalid = sum(not r["converged"] for r in rows)
    return dict(shots=requested, logical_failures=errors, nonconverged=invalid,
                converged_correct=categories[True, False], converged_logical_failure=categories[True, True],
                unconverged_correct_observables=categories[False, False], unconverged_logical_failure=categories[False, True],
                ler=errors / requested, nonconvergence_rate=invalid / requested,
                ler_95_exact_two_sided_low=0.0 if errors == 0 else float(beta.ppf(.025, errors, requested-errors+1)),
                ler_95_exact_two_sided_high=1.0 if errors == requested else float(beta.ppf(.975, errors+1, requested-errors)),
                ler_95_one_sided_upper=1.0 if errors == requested else float(beta.ppf(.95, errors+1, requested-errors)),
                sum_decode_cpu_seconds=math.fsum(r["decode_cpu_seconds"] for r in rows))


def _claim_output(output, campaign):
    """Reserve output ownership before scientific work, including crash recovery."""
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        saved = _read(manifest_path)
        _require(saved.get("campaign_identity") == campaign["identity_sha256"] and
                 saved.get("run_id") == campaign["run_id"],
                 "Output belongs to another campaign; choose another --output")
    else:
        _require(not output.exists(), "Existing CSV has no matching manifest; choose another --output")
        # A crash between the CSV replace and final manifest publication must
        # remain recoverable without accepting an unrelated pre-existing CSV.
        _atomic_json(manifest_path, dict(status="preparing", run_id=campaign["run_id"],
                                        campaign_identity=campaign["identity_sha256"]))


def _publish(output, campaign, rows, workers, points):
    # Caller holds the output lock for the complete campaign, including setup.
    _claim_output(output, campaign)
    manifest_path = output.with_suffix(".manifest.json")
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    _atomic_json(manifest_path, dict(schema_version=1, kind="bb_ppm_recorded_results",
        status="complete" if len(rows) == len(campaign["identity"]["p_values"]) else "partial",
        raw_shots_in_repository=False,
        campaign_identity=campaign["identity_sha256"], run_id=campaign["run_id"],
        results_csv=dict(path=output.name, sha256=_sha(output)),
        experiment=campaign["experiment"], resources=campaign["resources"],
        decoder=dict(name="relay-bp", parameters=campaign["identity"]["decoder_params"], version=campaign["identity"]["versions"]["relay-bp"]),
        execution=dict(worker_processes=workers, threads_per_worker=1, start_method="spawn"),
        requested_shots_by_p={str(p): n for p, n in zip(campaign["identity"]["p_values"], campaign["identity"]["shots"])},
        sample_seed_policy="SeedSequence([sample_seed, point_index, batch_index]); fresh seed-0 Relay decoder per shot",
        no_outcome_based_stopping=True, points=points, rows=rows, updated_utc=_utc()))


def run_campaign(*, sequence=None, plans=None, p_values=DEFAULT_P_VALUES, shots=50, workers=None,
                 run_dir=DEFAULT_RUN_DIR, output=DEFAULT_OUTPUT, decoder_params=None, seed=0,
                 rounds=10, initial_states="X", final_measure_states="X", resume=True,
                 noise_model="circuit_level_with_idling", batch_size=1000, print_progress=False):
    """Build one BB PPM operation/sequence and run explicit fixed-shot points.

    ``plans`` are reused verbatim. Changing scientific settings or source code
    requires a new run directory; changing only the worker budget is allowed.
    Reusing a run directory never silently generates replacement samples.
    Repeating a seed/p ordering/batch size in another directory repeats the
    sample streams: choose a new seed for an independent experiment.
    """
    import stim
    from lightstim.protocols.bb_ppm import BBPPMExperiment, BBPPMStep
    from lightstim.qec_code.BB_code.ppm.lowering import Measure, Shift
    parameters = normalize_decoder_params(decoder_params)
    p_values = tuple(float(p) for p in p_values)
    counts = [shots] * len(p_values) if type(shots) is int else list(shots)
    workers = available_workers() if workers is None else workers
    _require(p_values and len(set(p_values)) == len(p_values) and
             all(math.isfinite(p) and 0 < p < .5 for p in p_values), "p values must be unique and in (0, .5)")
    _require(len(counts) == len(p_values) and all(type(n) is int and n > 0 for n in counts), "Use one positive shot count per p")
    _require(type(workers) is int and 1 <= workers <= available_workers(), "Invalid worker CPU budget")
    _require(type(seed) is int and 0 <= seed < 2**64, "Sample seed must be an unsigned 64-bit integer")
    _require(type(batch_size) is int and batch_size > 0, "batch_size must be positive")
    _require(noise_model == "circuit_level_with_idling", "This benchmark preserves circuit_level_with_idling")
    sequence = tuple(sequence) if sequence is not None else (BBPPMStep(((2, "X"), (7, "Z"), (12, "X"))),)
    experiment = BBPPMExperiment(None, sequence, plans=plans, rounds=rounds,
        initial_states=initial_states, final_measure_states=final_measure_states, noise_params=None)
    plan_sha = compiled_plan_sha256(experiment.plans)
    identity = json.loads(_canonical(dict(sequence=[asdict(s) for s in sequence], compiled_plan_sha256=plan_sha,
        p_values=p_values, shots=counts, rounds=rounds, initial_state=initial_states, final_basis=final_measure_states,
        noise_model=noise_model, noiseless_boundaries=True, rounds_init=1, rounds_final=1, idle_rounds=0,
        decoder_params=parameters, seed=seed, batch_size=batch_size, versions=_versions(), sources=_source_hashes())))
    run_dir, output = Path(run_dir).expanduser().resolve(), Path(output).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _lock(run_dir / ".campaign.lock"), _lock(output.with_suffix(".lock")):
        request = run_dir / "request.json"
        if request.exists():
            _require(resume and _read(request) == identity, "Run identity changed or resume disabled; use a new run directory")
        else:
            _atomic_json(request, identity)
        campaign_path = run_dir / "campaign.json"
        if campaign_path.exists():
            campaign = _read(campaign_path)
            _require(campaign["identity"] == identity and campaign["clean_sha256"] == _sha(run_dir / "clean.stim"),
                     "Saved clean circuit or campaign identity changed")
            clean = stim.Circuit.from_file(run_dir / "clean.stim")
            _require(campaign["noise_schedule_sha256"] == _sha(run_dir / "noise_schedule.stim"),
                     "Saved BB noise schedule changed")
            noise_schedule = stim.Circuit.from_file(run_dir / "noise_schedule.stim")
        else:
            clean = experiment.build()
            noise_schedule = experiment.noise_schedule
            _require(clean.num_observables > 0 and clean.detector_error_model().num_errors == 0,
                     "Clean BB PPM circuit has no evaluation observables or is not silent")
            _require(not clean.compile_detector_sampler(seed=0).sample(8, append_observables=True).any(),
                     "Noiseless detector/observable check failed")
            (run_dir / "clean.stim").write_bytes(str(clean).encode("utf-8"))
            (run_dir / "noise_schedule.stim").write_bytes(str(noise_schedule).encode("utf-8"))
            _require(stim.Circuit.from_file(run_dir / "clean.stim") == clean, "Clean circuit roundtrip failed")
            clean_sha = _sha(run_dir / "clean.stim")
            label = "; ".join(" ".join(f"{letter}{q}" for q, letter in step.targets) for step in sequence)
            campaign = dict(identity=identity, identity_sha256=_fingerprint(identity), run_id=uuid.uuid4().hex,
                clean_sha256=clean_sha, noise_schedule_sha256=_sha(run_dir / "noise_schedule.stim"),
                experiment=dict(target=label, compiled_plan_sha256=plan_sha,
                    clean_circuit_sha256=clean_sha, noise_model=noise_model, rounds=rounds,
                    initial_state=initial_states, final_basis=final_measure_states,
                    measurement_instructions=sum(isinstance(op, Measure) for plan in experiment.plans
                                                 for op in getattr(plan, "operations", ())),
                    shift_instructions=sum(isinstance(op, Shift) for plan in experiment.plans
                                           for op in getattr(plan, "operations", ()))),
                resources=dict(sequence=label, physical_qubits=clean.num_qubits, ticks=clean.num_ticks,
                    detectors=clean.num_detectors, observables=clean.num_observables, p0_silent=True,
                    plan_sha256=plan_sha, circuit_sha256=clean_sha))
            _atomic_json(campaign_path, campaign)
        _claim_output(output, campaign)
        rows, points = [], []
        if _read(output.with_suffix(".manifest.json"))["status"] == "preparing":
            _publish(output, campaign, rows, workers, points)
        for index, (p, n) in enumerate(zip(p_values, counts)):
            point = run_dir / f"point_{index:03d}"
            point.mkdir(exist_ok=True)
            _prepare_point(clean, point, p=p, shots=n, seed=seed, point_index=index,
                           batch_size=batch_size, noise_schedule=noise_schedule)
            decoded, complete, stopped = decode_saved_inputs(point / "inputs", point / "decoding",
                decoder_params=parameters, workers=workers, print_progress=print_progress)
            if not complete:
                if print_progress:
                    print("Paused; resume the same run directory to finish the fixed sample count.", flush=True)
                break
            source_sha, decode_sha = _sha(point / "inputs/manifest.json"), _sha(point / "decoding/manifest.json")
            row = dict(p=p, **summarize(decoded, n), source_manifest_sha256=source_sha, decoder_manifest_sha256=decode_sha)
            rows.append(row)
            points.append(dict(p=p, source_manifest_sha256=source_sha, decoder_manifest_sha256=decode_sha,
                               source_manifest=os.path.relpath(point / "inputs/manifest.json", output.parent),
                               decoder_manifest=os.path.relpath(point / "decoding/manifest.json", output.parent)))
            _publish(output, campaign, rows, workers, points)
            if print_progress:
                print(f"p={p:g}: completed {n} shots, {row['logical_failures']} observable errors, {row['nonconverged']} nonconverged", flush=True)
            if stopped:
                break
        return rows
