"""QECSystem coupler adapter for routed rotated-surface-code PPM layouts."""

from types import SimpleNamespace

from lightstim.ir.coupler import LogicalCouplerProtocol

from .layout import build_explicit_ppm_layout
from .placement import _FLIP, place_patch

__all__ = ["RotatedSurfacePPMCoupler"]


class RotatedSurfacePPMCoupler(LogicalCouplerProtocol):
    """Materialize a verified rotated-surface PPM layout as an IR coupler."""

    EXPECTED_PATCH_COUNT = None
    _FLIP_O = {'X_horizontal': 'X_vertical', 'X_vertical': 'X_horizontal'}

    @staticmethod
    def _key(check):
        return (
            tuple(check['syn']),
            frozenset(
                (tuple(qubit), pauli)
                for qubit, pauli in check['pauli'].items()
            ),
        )

    def _build_coupler_geometry(
        self,
        coupler_patch,
        patches,
        *,
        placements,
        target,
        subset_route=None,
        seam=True,
        route=None,
        minority_names=frozenset(),
    ):
        result = subset_route
        if result is None:
            result = build_explicit_ppm_layout(
                placements,
                target,
                seam=seam,
                route=route,
            )
        if result.status != 'ok':
            raise ValueError(
                f'build_explicit_ppm_layout failed: {result.status} - '
                f'{result.message}'
            )

        layout = result.layout
        target_names = {name for name, _ in target}
        coupler_patch.conflicting_stabilizer_coords = set()

        merged_keys = {self._key(check) for check in layout.checks}
        kept = set()
        native_syndromes = set()
        patch_cells = set()
        for placement in placements:
            if placement.name not in target_names:
                continue
            patch_cells |= {
                (placement.origin[0] + 2 * i,
                 placement.origin[1] + 2 * j)
                for i in range(placement.distance)
                for j in range(placement.distance)
            }
            registered_orientation = (
                self._FLIP_O[placement.orientation]
                if placement.name in minority_names
                else placement.orientation
            )
            registered = place_patch(SimpleNamespace(
                origin=placement.origin,
                distance=placement.distance,
                orientation=registered_orientation,
            ))['checks']
            if placement.name in minority_names:
                registered = [
                    dict(
                        check,
                        type=_FLIP[check['type']],
                        pauli={
                            qubit: _FLIP[pauli]
                            for qubit, pauli in check['pauli'].items()
                        },
                    )
                    for check in registered
                ]
            for check in registered:
                native_syndromes.add(tuple(check['syn']))
                key = self._key(check)
                if key not in merged_keys:
                    coupler_patch.conflicting_stabilizer_coords.add(
                        tuple(check['syn'])
                    )
                else:
                    kept.add(key)

        for qubit in sorted(set(map(tuple, layout.data)) - patch_cells):
            coupler_patch.add_qubit(qubit[0], qubit[1], role='data')

        added = set()
        for check in layout.checks:
            if self._key(check) in kept:
                continue
            syndrome = tuple(check['syn'])
            check_type = check.get('type') or next(
                iter(set(check['pauli'].values()))
            )
            relay = check.get('kf')
            if relay is not None:
                if syndrome not in native_syndromes and syndrome not in added:
                    coupler_patch.add_qubit(
                        syndrome[0], syndrome[1], role='syndrome_x'
                    )
                    added.add(syndrome)
                for coord, role in (
                    (tuple(relay['flag']), 'syndrome_z'),
                    (tuple(relay['shared']), 'syndrome_z'),
                ):
                    if coord not in native_syndromes and coord not in added:
                        coupler_patch.add_qubit(coord[0], coord[1], role=role)
                        added.add(coord)
                coupler_patch.stabilizers.append({
                    'pauli': {
                        tuple(qubit): pauli
                        for qubit, pauli in check['pauli'].items()
                    },
                    'type': 'MIXED',
                    'syn_coord': syndrome,
                    'kf': {
                        'flag': tuple(relay['flag']),
                        'shared': tuple(relay['shared']),
                        'orient': relay['orient'],
                    },
                })
                continue
            if syndrome not in native_syndromes and syndrome not in added:
                coupler_patch.add_qubit(
                    syndrome[0],
                    syndrome[1],
                    role='syndrome_z' if check_type == 'Z' else 'syndrome_x',
                )
                added.add(syndrome)
            coupler_patch.stabilizers.append({
                'pauli': {
                    tuple(qubit): pauli
                    for qubit, pauli in check['pauli'].items()
                },
                'type': check_type if check_type in ('X', 'Z') else 'MIXED',
                'syn_coord': syndrome,
            })

        coupler_patch.routed_layout = layout
        coupler_patch.subset_route = result
