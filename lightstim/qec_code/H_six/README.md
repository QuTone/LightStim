# H6 compatibility imports

The implementation has moved to the general [H-code family](../H_code/README.md).
`HSixCode()` constructs `HCode(n=6)` with the original data numbering, checks,
logical representatives, coordinates, and parity helpers. Imports from
`lightstim.qec_code.H_six`, `.code_patch`, and `.SE_block` continue to work.
`HSixLogicalOpSet`, also available from `.operation`, inherits the family's
validated transversal-H operation; H acts on both logical slots together.
Its H6-only `encode(builder, patch, logical_bases=("X", "Y"))` prepares
independent logical Pauli eigenstates with the unflagged Fig. 1(d) encoder.
See the [encoding example](../H_code/README.md#h6-encoding-with-independent-logical-input-bases).
The default `HSixExtractionBlock` now uses the dedicated concurrent family
schedule. Select `GenericCSSColorationExtractionBlock` explicitly to reproduce
the earlier generic schedule.
