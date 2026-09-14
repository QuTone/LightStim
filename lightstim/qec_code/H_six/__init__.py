from .code_patch import HSixCode
from .SE_block import HSixExtractionBlock

HSixCode.default_extraction_block_class = HSixExtractionBlock

__all__ = [
    "HSixCode",
    "HSixExtractionBlock",
]
