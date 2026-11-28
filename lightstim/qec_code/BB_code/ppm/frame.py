"""BB Pauli frame and compatibility with LightStim's existing noise schedule."""
from __future__ import annotations

from functools import lru_cache

import stim


# Bit 0 is the constant. Bit i+1 is raw absolute measurement record i.
def parity(records=(), constant=False):
    result = int(bool(constant))
    for record in records:
        result ^= 1 << (int(record) + 1)
    return result


def unpack(expression):
    records = []
    remaining = expression >> 1
    while remaining:
        low = remaining & -remaining
        records.append(low.bit_length() - 1)
        remaining ^= low
    return tuple(records), bool(expression & 1)


@lru_cache(None)
def _gate_images(name):
    data = stim.gate_data(name)
    tableau = data.tableau
    n = len(tableau)
    return n, tuple(tableau.x_output(k) for k in range(n)), tuple(tableau.z_output(k) for k in range(n))


class ClassicalPauliFrame:
    """|effective> = F(raw records) |physical>, up to an irrelevant phase."""
    def __init__(self, num_qubits):
        self.x = [0] * num_qubits
        self.z = [0] * num_qubits
        self.records = []
        self.removed_controls = 0

    def effective_parity(self, record_ids, constant=False):
        expression = int(bool(constant))
        for record in record_ids:
            expression ^= self.records[record]
        return expression

    def anticommutes(self, support):
        expression = 0
        for q, letter in support.items():
            if letter in 'YZ':
                expression ^= self.x[q]
            if letter in 'XY':
                expression ^= self.z[q]
        return expression

    def virtual_pauli(self, support, control):
        """control is already an affine expression of RAW records."""
        for q, letter in support.items():
            if letter in 'XY':
                self.x[q] ^= control
            if letter in 'YZ':
                self.z[q] ^= control

    def virtual_feedback(self, support, effective_records, invert=False):
        self.virtual_pauli(support, self.effective_parity(effective_records, invert))

    def reset(self, qubits):
        for q in qubits:
            self.x[q] = self.z[q] = 0

    def measure(self, support):
        value = (1 << (len(self.records) + 1)) ^ self.anticommutes(support)
        self.records.append(value)
        return value

    def clifford(self, name, qubits):
        n, xs, zs = _gate_images(name)
        if len(qubits) != n:
            raise ValueError((name, qubits))
        out_x = [0] * n
        out_z = [0] * n
        for i, q in enumerate(qubits):
            for coefficient, image in ((self.x[q], xs[i]), (self.z[q], zs[i])):
                for j in image.pauli_indices():
                    letter = image[j]
                    if letter in (1, 2):
                        out_x[j] ^= coefficient
                    if letter in (2, 3):
                        out_z[j] ^= coefficient
        for j, q in enumerate(qubits):
            self.x[q], self.z[q] = out_x[j], out_z[j]

    def consume(self, circuit, *, rewrite_annotations=False, virtualize_constant_paulis=False):
        """Remove feedback and optionally substitute existing record annotations.

        Annotation rewriting is for equivalence auditing only. Online unchanged
        core integration calls this with False and leaves its automatic D/O alone.
        Constants do not enter D/O flip parities; Stim's reference sample absorbs
        them. Constants remain in self.records for actual protocol outcomes.
        """
        output = stim.Circuit()
        for instruction in circuit.flattened():
            name = instruction.name
            targets = instruction.targets_copy()
            args = instruction.gate_args_copy()
            data = stim.gate_data(name)
            if name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
                if rewrite_annotations:
                    if not all(t.is_measurement_record_target for t in targets):
                        raise NotImplementedError('Non-record annotation target')
                    expr = self.effective_parity(len(self.records) + t.value for t in targets)
                    records, _ = unpack(expr)
                    output.append(name, [stim.target_rec(k - len(self.records)) for k in records],
                                  args, tag=instruction.tag)
                else:
                    output.append(instruction)
                continue
            if (virtualize_constant_paulis and name in ('X', 'Y', 'Z')
                    and instruction.tag == 'noiseless'):
                if not all(t.is_qubit_target for t in targets):
                    raise NotImplementedError((name, targets))
                self.virtual_pauli({t.value: name for t in targets}, 1)
                continue
            if data.is_unitary:
                n = 2 if data.is_two_qubit_gate else 1
                for start in range(0, len(targets), n):
                    group = targets[start:start + n]
                    if any(t.is_measurement_record_target for t in group):
                        if n != 2 or name not in ('CX', 'CY', 'CZ'):
                            raise NotImplementedError((name, group))
                        if group[0].is_measurement_record_target and group[1].is_qubit_target:
                            record, q = group[0].value + len(self.records), group[1].value
                        elif (name == 'CZ' and group[1].is_measurement_record_target
                              and group[0].is_qubit_target):
                            record, q = group[1].value + len(self.records), group[0].value
                        else:
                            raise NotImplementedError((name, group))
                        self.virtual_feedback({q: name[-1]}, (record,))
                        self.removed_controls += 1
                    else:
                        if not all(t.is_qubit_target for t in group):
                            raise NotImplementedError((name, group))
                        self.clifford(name, [t.value for t in group])
                        output.append(name, group, args, tag=instruction.tag)
                continue
            if name in ('R', 'RX', 'RY'):
                self.reset(t.value for t in targets)
            elif name in ('M', 'MX', 'MY', 'MR', 'MRX', 'MRY'):
                basis = {'M': 'Z', 'MX': 'X', 'MY': 'Y', 'MR': 'Z', 'MRX': 'X', 'MRY': 'Y'}[name]
                for target in targets:
                    # Raw record already includes any inverted-target convention.
                    self.measure({target.value: basis})
                    if data.is_reset:
                        self.reset((target.value,))
            elif name == 'MPP':
                support = {}
                for i, target in enumerate(targets):
                    if target.is_combiner:
                        continue
                    support[target.value] = 'X' if target.is_x_target else 'Y' if target.is_y_target else 'Z'
                    if i + 1 == len(targets) or not targets[i + 1].is_combiner:
                        self.measure(support)
                        support = {}
            elif data.produces_measurements:
                raise NotImplementedError(name)
            elif data.is_reset:
                raise NotImplementedError(name)
            # Pauli noise channels commute through a Pauli frame up to global
            # phase. This helper deliberately does not add or schedule noise.
            output.append(instruction)
        return output


class BBFrameAdapter:
    """Composition around unchanged CircuitBuilder, using its physical output.

    Effective record IDs stay stable. Protocol outcomes alone must be expanded
    with effective_parity before exposure to raw sampled measurements.
    """
    def __init__(self, builder):
        self._builder = builder
        self.frame = ClassicalPauliFrame(builder.tracker.num_qubits)
        self._length = 0
        self._last = None

    def __getattr__(self, key):
        return getattr(self._builder, key)

    def _sync(self):
        c = self._builder.circuit
        delta = stim.Circuit()
        start = self._length
        if self._last is not None and c[self._length - 1] != self._last:
            changed = c[self._length - 1]
            if isinstance(changed, stim.CircuitRepeatBlock):
                raise RuntimeError('Builder rewrote existing repeat block')
            old_targets = self._last.targets_copy()
            new_targets = changed.targets_copy()
            previous_header = (self._last.name, self._last.gate_args_copy(), self._last.tag)
            if ((changed.name, changed.gate_args_copy(), changed.tag) != previous_header
                    or new_targets[:len(old_targets)] != old_targets):
                raise RuntimeError('Builder rewrote existing physical instructions')
            delta.append(changed.name, new_targets[len(old_targets):], changed.gate_args_copy(), tag=changed.tag)
        delta += c[start:]
        self.frame.consume(delta)
        self._length = len(c)
        self._last = c[-1] if len(c) else None
        if len(self.frame.records) != self.tracker.total_measurements:
            raise RuntimeError(('Record count mismatch', len(self.frame.records), self.tracker.total_measurements))

    def _call(self, name, *args, **kwargs):
        self._sync()
        result = getattr(self._builder, name)(*args, **kwargs)
        self._sync()
        return result

    def initialize(self, *args, **kwargs):
        return self._call('initialize', *args, **kwargs)

    def apply_unitary_block(self, *args, **kwargs):
        return self._call('apply_unitary_block', *args, **kwargs)

    def apply_syndrome_extraction(self, *args, **kwargs):
        return self._call('apply_syndrome_extraction', *args, **kwargs)

    def apply_data_readout(self, *args, **kwargs):
        return self._call('apply_data_readout', *args, **kwargs)

    def apply_mid_data_readout(self, *args, **kwargs):
        return self._call('apply_mid_data_readout', *args, **kwargs)

    def apply_pauli_feedback(self, pauli, records, invert=False):
        self._sync()
        records = tuple(records)
        self.frame.virtual_feedback(pauli, records, invert)
        if pauli and (records or invert):
            # Preserve the historical feedback slot's operated-qubit mask for
            # MomentIdling. These identities enact no Pauli correction and are
            # removed after the existing NoiseInjector has consumed the mask.
            self.circuit.append('I', sorted(pauli), tag='noiseless')

    def effective_parity(self, records, constant=False):
        self._sync()
        return unpack(self.frame.effective_parity(records, constant))

    def finalize(self):
        self._sync()
        # History relations must be exported in the tracker's record basis,
        # before the native Bell-readout map is applied by the caller.
        return self._builder.to_stim_circuit().copy()


def strip_frame_activity(circuit):
    """Remove BB-owned identities after noise placement, or for clean output.

    This BB schedule does not otherwise use I[noiseless] instructions. The
    identities retain only feedback-slot activity, never an actual correction.
    """
    output = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            output.append(stim.CircuitRepeatBlock(
                instruction.repeat_count, strip_frame_activity(instruction.body_copy())))
        elif instruction.name != 'I' or instruction.tag != 'noiseless':
            output.append(instruction)
    return output
