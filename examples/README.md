# Examples

Small, executable references for using LightStim from another project.

Examples are consumers of the public `lightstim` API. They do not define core
semantics, and they are not benchmark sweeps or protocol implementations. A
real integration should normally move its adapter into the caller's repository
after the contract has been validated here.

## Categories

- [`integrations/`](integrations/): cross-project and cross-layer adapters.

Generated circuits, detector error models, and manifests are written under
`build/examples/` and are intentionally not committed.
