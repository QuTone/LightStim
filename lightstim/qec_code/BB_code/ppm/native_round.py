"""Native syndrome rounds expressed in a reversible Bell readout basis.

The tracker sees the joint Bell parity and its independent random bit. Export
restores the native physical circuit and substitutes these record coordinates
in the automatically generated annotations; it does not infer new checks.
"""
import stim

PREFIX = 'bb_native_basis_'


def apply_native_round(experiment, native):
    """Refresh each native check through a complete standard extraction round."""
    builder = experiment.builder
    tracker = experiment.tracker
    tag = PREFIX + str(len(experiment.basis_blocks))
    base = tracker.total_measurements
    right = {b for _, b in native.bell_pairs}
    sequence = [q for q in native.syndrome_qubits if q not in right]
    positions = {q: i for i, q in enumerate(sequence)}
    # The original vertex parity contains both Bell bits; in the new basis
    # it contains only the parity bit, never the independent random bit.
    outcome = [positions[native.syndrome_qubits[i]] for i in native.outcome_parity
               if native.syndrome_qubits[i] not in right]
    virtual_sequence = sequence + sorted(right)
    original_positions = {q: i for i, q in enumerate(native.syndrome_qubits)}
    left_to_right = dict(native.bell_pairs)
    mapping = {}
    for i, q in enumerate(virtual_sequence):
        records = {base + original_positions[q]}
        if q in left_to_right:
            records.add(base + original_positions[left_to_right[q]])
        mapping[base + i] = tuple(sorted(records))
    experiment.basis_blocks[tag] = {
        'physical': native.round_circuit, 'mapping': mapping, 'base': base,
    }
    # QUBIT_COORDS markers survive Stim.flattened(); SHIFT_COORDS does not.
    builder.circuit.append('QUBIT_COORDS', [0], experiment.system.qubit_coords[0],
                           tag=tag + '_begin')
    # This is an existing physical Bell partner in its legitimate reset state.
    # It is included in the round's physical resets and is not an extra qubit.
    if right:
        builder.initialize({q: 'Z' for q in sorted(right)}, tracker.num_qubits)
    block = stim.Circuit()
    for ins in native.round_circuit:
        if ins.name == 'MX':
            for a, b in native.bell_pairs:
                block.append('CX', [a, b], tag='bb_basis_closure')
            block.append('MX', sequence)
        else:
            block.append(ins)
    builder.apply_syndrome_extraction(block, rounds=1)
    if right:
        builder.apply_data_readout({q: 'X' for q in sorted(right)}, resolve_absorbed=False)
    builder.circuit.append('QUBIT_COORDS', [0], experiment.system.qubit_coords[0],
                           tag=tag + '_end')
    assert tracker.total_measurements == base + len(native.syndrome_qubits)
    return tuple(base + i for i in outcome)


def record_mapping(blocks):
    """Map absolute tracker record IDs to physical measurement parities."""
    return {r: parity for block in blocks.values() for r, parity in block['mapping'].items()}


def rewrite_parity(records, constant=False, blocks=None, *, mapping=None):
    if mapping is None:
        mapping = record_mapping(blocks or {})
    out = set()
    for r in records:
        out.symmetric_difference_update(mapping.get(r, (r,)))
    return tuple(sorted(out)), constant


def finalize_native_basis(circuit, blocks):
    """Restore physical readouts and translate the tracker's existing checks."""
    mapping = record_mapping(blocks)
    out = stim.Circuit()
    cursor = 0
    active = None
    annotations = []

    def append_rewritten(ins, source_cursor):
        targets = ins.targets_copy()
        if ins.name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            records, _ = rewrite_parity(
                [source_cursor + t.value for t in targets], mapping=mapping)
            if any(r >= out.num_measurements or r < 0 for r in records):
                raise ValueError(
                    f'Invalid mapped record in block {active!r}: '
                    f'source cursor={source_cursor}, output cursor={out.num_measurements}, '
                    f'instruction={ins}, mapped={records}')
            out.append(ins.name, [stim.target_rec(r - out.num_measurements) for r in records],
                       ins.gate_args_copy(), tag=ins.tag)
        elif any(t.is_measurement_record_target for t in targets):
            assert ins.name in ('CX', 'CY', 'CZ')
            for i in range(0, len(targets), 2):
                control, target = targets[i:i + 2]
                assert control.is_measurement_record_target and target.is_qubit_target
                record = source_cursor + control.value
                for r in mapping.get(record, (record,)):
                    out.append(ins.name, [stim.target_rec(r - out.num_measurements), target],
                               ins.gate_args_copy(), tag=ins.tag)
        else:
            out.append(ins)

    def expanded(source):
        for instruction in source:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                for _ in range(instruction.repeat_count):
                    yield from expanded(instruction.body_copy())
            else:
                yield instruction

    for ins in expanded(circuit):
        if ins.tag.startswith(PREFIX) and ins.tag.endswith('_begin'):
            assert active is None
            active = ins.tag[:-6]
            annotations = []
            assert cursor == blocks[active]['base'] == out.num_measurements
        elif ins.tag.startswith(PREFIX) and ins.tag.endswith('_end'):
            assert active == ins.tag[:-4]
            out += blocks[active]['physical']
            assert cursor == out.num_measurements
            for annotation, old_cursor in annotations:
                append_rewritten(annotation, old_cursor)
            active = None
        elif active is not None:
            if ins.name in ('DETECTOR', 'OBSERVABLE_INCLUDE', 'SHIFT_COORDS'):
                annotations.append((ins, cursor))
        else:
            append_rewritten(ins, cursor)
        cursor += ins.num_measurements
    assert active is None and cursor == out.num_measurements
    return out
