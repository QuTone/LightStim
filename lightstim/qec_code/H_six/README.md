# H6 compatibility imports

The implementation has moved to the general [H-code family](../H_code/README.md).
`HSixCode()` constructs `HCode(n=6)` with the original data numbering, checks,
logical representatives, coordinates, and parity helpers. Imports from
`lightstim.qec_code.H_six`, `.code_patch`, and `.SE_block` continue to work.
The default `HSixExtractionBlock` now uses the dedicated concurrent family
schedule. Select `GenericCSSColorationExtractionBlock` explicitly to reproduce
the earlier generic schedule.
