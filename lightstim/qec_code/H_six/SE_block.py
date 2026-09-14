"""Baseline syndrome extraction for the [[6, 2, 2]] H6 code.

X and Z Tanner graphs are independently edge-colored by LightStim's generic
CSS block. Each color is a parallel CNOT layer; X checks use ancilla-to-data
CNOTs and Z checks use data-to-ancilla CNOTs. This baseline has no flags.
"""

from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock

HSixExtractionBlock = GenericCSSColorationExtractionBlock

__all__ = ["HSixExtractionBlock"]
