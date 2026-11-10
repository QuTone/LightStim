# Examples

Small, executable references for using LightStim in integrations and paper reconstructions.

Examples are consumers of the public `lightstim` API. They do not define core
semantics, and they are not the core protocol library or benchmark sweeps. A
real integration should normally move its adapter into the caller's repository
after the contract has been validated here.

## Categories

- [`integrations/`](integrations/): cross-project and cross-layer adapters.
- [`quantinuum_color_code_memory/`](quantinuum_color_code_memory/): a scoped
  d=3 reconstruction of Quantinuum's Bell-flag preparation and SWAP-superdense memory.

Generated circuits, detector error models, and manifests are written under
`build/examples/` and are intentionally not committed.
Small curated reference results and executed demonstration notebooks may be
included with an example, with their parameters, provenance and limitations.
