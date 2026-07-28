# Middle-Out Reference Circuits

These Stim circuits are snapshots generated with Chromobius/Clorco's
`midout_color_code_X` construction using uniform circuit noise at `p=0.001`.
The distance and generator-round count are encoded in each filename.

The Color Code tests use them as structural references for qubit coordinates,
CNOT layers, initialization resets, and the first detector layer. They are
kept in the test tree so the suite does not depend on a local Chromobius
checkout or ignored research artifacts.
