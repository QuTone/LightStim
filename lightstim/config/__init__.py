"""Declarative LightStim experiment configs and parameter sweeps."""

from .loader import (
    ExperimentConfig, ResolvedConfig, SCHEMA_VERSION, expand_sweep, load_config,
    resolve_compile_section,
)


def compile(spec):
    """Build the noisy Stim circuit and QEC system for a single experiment."""
    from .compiler import compile as compile_spec

    return compile_spec(spec)


def export_benchmarks_json(csv, config, output=None):
    """Write benchmark CSV results in the Protocol Zoo format."""
    from .results import export_benchmarks_json as export

    return export(csv, config, output)


__all__ = [
    "ExperimentConfig", "ResolvedConfig", "SCHEMA_VERSION", "compile",
    "expand_sweep", "export_benchmarks_json", "load_config", "resolve_compile_section",
]
