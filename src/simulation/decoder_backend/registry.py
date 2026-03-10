"""Decoder registry: name -> backend -> Decoder class."""

import sinter
from typing import Any, Dict, Type

# Registry: decoder_name -> { backend -> decoder_class }
_decoder_registry: Dict[str, Dict[str, type]] = {}


def register_decoder(
    name: str,
    decoder_class: Type,
    aliases: list[str] | None = None,
    backend: str = "cpu",
) -> None:
    """Register a decoder class under a name, backend, and optional aliases."""
    name = name.lower()
    _decoder_registry.setdefault(name, {})[backend] = decoder_class
    if aliases:
        for alias in aliases:
            _decoder_registry.setdefault(alias.lower(), {})[backend] = decoder_class


def get_decoder(
    name: str,
    backend: str = "cpu",
    **params,
) -> Any:
    """
    Get a decoder instance by name and backend.

    Args:
        name: Decoder name (e.g. 'pymatching', 'bposd').
        backend: 'cpu', 'gpu', or 'fpga'.
        **params: Decoder-specific parameters.

    Returns:
        Decoder instance implementing sinter.Decoder interface.
    """
    # Ensure decoders are loaded (registration happens on import)
    from . import decoders  # noqa: F401

    name = name.lower()

    # Check our registry first
    if name in _decoder_registry:
        by_backend = _decoder_registry[name]
        cls = by_backend.get(backend)
        if cls is None and backend != "cpu":
            hint = " Install cudaq_qec for GPU support: pip install cudaq_qec" if backend == "gpu" else ""
            raise ImportError(
                f"Decoder '{name}' has no '{backend}' backend registered.{hint}"
            )
        cls = cls or by_backend.get("cpu")
        if cls is not None:
            return cls(**params) if params else cls()

    # Fallback to sinter's built-in decoders (pymatching, fusion_blossom, vacuous)
    try:
        builtin = sinter._decoding.BUILT_IN_DECODERS.get(name)
        if builtin is not None:
            return builtin
    except AttributeError:
        pass

    # Show unique decoders: primary names only (aliases map to same backend)
    available = _unique_decoder_names()
    raise ValueError(
        f"Unknown decoder '{name}'. Available: {available}"
    )


def _unique_decoder_names() -> list[str]:
    """Return primary decoder names only (aliases like mwpm, bp_osd excluded)."""
    from . import decoders  # noqa: F401 - ensure decoders are loaded
    cls_to_names: Dict = {}
    for name, backends in _decoder_registry.items():
        for cls in backends.values():
            cls_to_names.setdefault(cls, []).append(name)
    # Prefer canonical name: pymatching > mwpm, bposd > bp_osd
    canonical_order = ["pymatching", "bposd", "mwpf", "nv-qldpc-decoder"]
    result = []
    for names in cls_to_names.values():
        chosen = next((n for n in canonical_order if n in names), names[0])
        result.append(chosen)
    return sorted(result)


def list_decoders() -> list[str]:
    """Return list of registered decoder names (without aliases)."""
    return _unique_decoder_names()
