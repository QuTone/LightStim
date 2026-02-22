"""Decoder registry: name -> Decoder class/instance factory."""

import sinter
from typing import Any, Callable, Dict, Type

# Registry: decoder_name -> (decoder_class_or_factory, backend)
_decoder_registry: Dict[str, tuple] = {}


def register_decoder(
    name: str,
    decoder_class: Type,
    aliases: list[str] | None = None,
) -> None:
    """Register a decoder class under a name and optional aliases."""
    name = name.lower()
    _decoder_registry[name] = (decoder_class,)
    if aliases:
        for alias in aliases:
            _decoder_registry[alias.lower()] = (decoder_class,)


def get_decoder(
    name: str,
    backend: str = "cpu",
    **params,
) -> Any:
    """
    Get a decoder instance by name.

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
        decoder_cls, = _decoder_registry[name]
        return decoder_cls(**params) if params else decoder_cls()

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
    for name, (cls,) in _decoder_registry.items():
        cls_to_names.setdefault(cls, []).append(name)
    # Prefer canonical name: pymatching > mwpm, bposd > bp_osd
    canonical_order = ["pymatching", "bposd", "mwpf"]
    result = []
    for names in cls_to_names.values():
        chosen = next((n for n in canonical_order if n in names), names[0])
        result.append(chosen)
    return sorted(result)


def list_decoders() -> list[str]:
    """Return list of registered decoder names (without aliases)."""
    return _unique_decoder_names()
