# Integration Examples

This directory is the executable boundary between LightStim and external
systems. Each example should keep the external schema or mock IR local, map it
to public LightStim APIs, and expose the resulting physical artifacts and
lineage.

## Available

- [`logical_compiler_rotated_surface_ppm/`](logical_compiler_rotated_surface_ppm/):
  caller-owned logical PPM payload to rotated-surface lowering, Stim circuit,
  detector error model, and measurement-record manifest.

Integration examples may reveal a common contract over time. A single adapter
should not be treated as a backend-neutral IR until multiple real integrations
need the same abstraction.
