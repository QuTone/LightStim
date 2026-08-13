"""Reproduce the GALA paper's circuit-level memory experiment.

The target is Sec. S6.2 of arXiv:2608.07431: greedy-coloration syndrome
extraction, no idle noise, single-basis detector decoding, and the hierarchical
BP -> Relay-BP -> exact-MLE decoder used by arXiv:2604.16209. The GALA paper
does not state N_c. The default 12-round circuit follows its bundled notebook
and the earlier paper's W=d decoder window. Pass ``--rounds 32`` for a
monolithic full-memory ablation; this is not equivalent to that paper's
32-round experiment decoded through sliding windows.

Example:

    python benchmarks/memory/run_gala_replication.py \
        --p 0.0025 --shots 500 --workers 4 --tiers mle

The default is the full hierarchy with an unlimited exact-MLE fallback, as in
the paper.  It used Gurobi; LightStim's open-source SciPy/HiGHS backend can take
minutes or longer and consume substantial memory on one hard GALA residual.
Use ``--mle-time-limit`` when a bounded diagnostic is preferred over attempting
to complete every exact solve.

The paper models preparation and readout with single-qubit depolarization of
strength p and each two-qubit gate with two-qubit depolarization of strength p;
it does not add noise after the ideal basis-change H gates used to express
direct X preparation/readout in Stim. LightStim's reset/readout parameters are
basis-flip probabilities, so this script uses 2p/3 at those boundaries; the
other p/3 Pauli component is invisible in the prepared/measured basis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import beta

from lightstim.ir.qec_system import QECSystem
from lightstim.noise.config import NoiseConfig
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.gala_code import (
    GalaCode,
    GalaCodeExtractionBlock,
    GalaGeneratorExtractionBlock,
)
from lightstim.simulation.decoder_backend import DecoderConfig, SimulationPipeline


BP_PARAMS = {
    "max_iter": 200,
    "bp_method": "minimum_sum",
    "ms_scaling_factor": 0.0,
    "schedule": "serial",
}

RELAY_PARAMS = {
    "pre_iter": 0,
    "num_sets": 300,
    "set_max_iter": 60,
    "gamma0": 0.1,
    "alpha_iteration_scaling_factor": 1.0,
    "stop_nconv": 1,
    "precision": "f32",
    "seed": 0,
}


def build_circuit(
    *, preset: str, p: float, rounds: int, schedule: str = "coloration"
):
    code = GalaCode.from_preset(preset)
    system = QECSystem()
    system.add_patch(code, name="gala")
    extraction_block = (
        GalaCodeExtractionBlock
        if schedule == "coloration" else GalaGeneratorExtractionBlock
    )
    circuit = MemoryExperiment(
        qec_system=system,
        extraction_block_class=extraction_block,
        rounds=rounds,
        noise_params=NoiseConfig(
            p_idle=0.0,
            p_1q=0.0,
            p_2q=p,
            p_meas=2.0 * p / 3.0,
            p_reset=2.0 * p / 3.0,
        ),
        noise_model="circuit_level",
        basis="Z",
        z_only=True,
    ).build()
    return circuit, code


def decoder_config(
    *,
    tiers: str,
    mle_time_limit: float,
    bp_ms_scaling: float,
    relay_alpha: float | None,
    relay_alpha_scale: float,
    relay_gamma0: float,
) -> DecoderConfig:
    bp_params = {**BP_PARAMS, "ms_scaling_factor": bp_ms_scaling}
    relay_params = {
        **RELAY_PARAMS,
        "gamma0": relay_gamma0,
        "alpha": relay_alpha,
        "alpha_iteration_scaling_factor": relay_alpha_scale,
    }
    if tiers == "native-relay":
        # Closest available approximation to the paper's unpublished memory-BP
        # T1: Relay's native min-sum implementation performs the 200-iteration
        # memory-BP pre-pass, then continues into the published Relay tier.
        return DecoderConfig.chain([
            DecoderConfig(
                "relay-bp", params={**relay_params, "pre_iter": 200})
        ])
    stages = [DecoderConfig("ldpc-bp", params=bp_params)]
    if tiers in {"relay", "mle"}:
        stages.append(DecoderConfig("relay-bp", params=relay_params))
    if tiers == "mle":
        stages.append(DecoderConfig(
            "mle-ilp",
            params={"time_limit": mle_time_limit},
            on_decode_failure="error",
        ))
    return DecoderConfig.chain(stages)


def clopper_pearson(errors: int, shots: int, confidence: float = 0.95):
    alpha = 1.0 - confidence
    lower = (
        0.0 if errors == 0
        else float(beta.ppf(alpha / 2.0, errors, shots - errors + 1))
    )
    upper = (
        1.0 if errors == shots
        else float(beta.ppf(1.0 - alpha / 2.0, errors + 1, shots - errors))
    )
    return lower, upper


def per_logical_per_cycle(block_probability: float, k: int, rounds: int):
    if block_probability >= 1.0:
        return 1.0
    return -math.expm1(
        math.log1p(-block_probability) / (k * rounds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="gala_132_30_12")
    parser.add_argument("--p", type=float, action="append", required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument(
        "--schedule",
        choices=("coloration", "generator"),
        default="coloration",
        help="Paper LER coloration, or the movement-optimized generator order.",
    )
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tiers",
        choices=("bp", "relay", "native-relay", "mle"),
        default="mle",
    )
    parser.add_argument(
        "--mle-time-limit",
        type=float,
        default=0.0,
        help="Per-shot seconds; zero leaves exact MLE unbounded.",
    )
    parser.add_argument(
        "--bp-ms-scaling", type=float, default=BP_PARAMS["ms_scaling_factor"],
        help="T1 min-sum scaling; 0 uses ldpc's dynamic schedule.",
    )
    parser.add_argument(
        "--relay-alpha", type=float,
        help=("Relay min-sum alpha. Omit for the package default; exactly 0 "
              "enables Relay's iteration-dependent ramp."),
    )
    parser.add_argument(
        "--relay-alpha-scale", type=float,
        default=RELAY_PARAMS["alpha_iteration_scaling_factor"],
        help="Iteration scale used when --relay-alpha=0.",
    )
    parser.add_argument(
        "--relay-gamma0", type=float, default=RELAY_PARAMS["gamma0"],
        help="Ordered memory strength for Relay and its native BP pre-pass.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for p in args.p:
        circuit, code = build_circuit(
            preset=args.preset, p=p, rounds=args.rounds,
            schedule=args.schedule)
        stats = SimulationPipeline(
            decoder_config=decoder_config(
                tiers=args.tiers,
                mle_time_limit=args.mle_time_limit,
                bp_ms_scaling=args.bp_ms_scaling,
                relay_alpha=args.relay_alpha,
                relay_alpha_scale=args.relay_alpha_scale,
                relay_gamma0=args.relay_gamma0,
            ),
            max_shots=args.shots,
            max_errors=args.max_errors,
            batch_size=args.batch_size,
            num_workers=args.workers,
            seed=args.seed,
            print_progress=True,
        ).run(circuit)

        block_ler = stats.logical_error_rate
        block_low, block_high = clopper_pearson(
            stats.errors, stats.post_selected_shots)
        result = {
            "preset": args.preset,
            "p": p,
            "rounds": args.rounds,
            "schedule": args.schedule,
            "k": code.num_logicals,
            "tiers": args.tiers,
            "bp_ms_scaling": args.bp_ms_scaling,
            "relay_alpha": args.relay_alpha,
            "relay_alpha_scale": args.relay_alpha_scale,
            "relay_gamma0": args.relay_gamma0,
            "shots": stats.shots,
            "kept_shots": stats.post_selected_shots,
            "errors": stats.errors,
            "seconds": stats.seconds,
            "decoder_stage_attempts": stats.decoder_stage_attempts,
            "block_ler": block_ler,
            "block_ler_ci95": [block_low, block_high],
            "per_logical_per_cycle_ler": per_logical_per_cycle(
                block_ler, code.num_logicals, args.rounds),
            "per_logical_per_cycle_ler_ci95": [
                per_logical_per_cycle(
                    block_low, code.num_logicals, args.rounds),
                per_logical_per_cycle(
                    block_high, code.num_logicals, args.rounds),
            ],
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
