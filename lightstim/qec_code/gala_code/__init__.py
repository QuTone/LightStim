from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

from .code_patch import GalaCode
from .group import LiftAlphabet, commutes, compose, invert, s3, s4
from .presets import GALA_CODE_PRESETS
from .SE_block import GalaGeneratorExtractionBlock

GalaCodeExtractionBlock = GenericCSSColorationExtractionBlock
GalaCode.default_extraction_block_class = GalaCodeExtractionBlock

__all__ = [
    "GalaCode",
    "GalaCodeExtractionBlock",
    "GalaGeneratorExtractionBlock",
    "GALA_CODE_PRESETS",
    "LiftAlphabet",
    "commutes",
    "compose",
    "invert",
    "s3",
    "s4",
]
