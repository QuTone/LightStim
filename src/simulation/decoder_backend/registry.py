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

    raise ValueError(
        f"Unknown decoder '{name}'. Available: {list(_decoder_registry.keys())}"
    )


def list_decoders() -> list[str]:
    """Return list of registered decoder names (without aliases)."""
    seen = set()
    result = []
    for name, (cls,) in _decoder_registry.items():
        if cls not in seen:
            seen.add(cls)
            result.append(name)
    return result
