"""PyMatching decoder (CPU, via sinter)."""

from ..registry import register_decoder

try:
    from sinter._decoding._decoding_pymatching import PyMatchingDecoder
except ImportError:
    PyMatchingDecoder = None  # type: ignore

if PyMatchingDecoder is not None:
    register_decoder("pymatching", PyMatchingDecoder, aliases=["mwpm"])
