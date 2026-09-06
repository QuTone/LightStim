"""Square Bacon-Shor subsystem codes with separate center and gauge declarations."""

from numbers import Integral

from lightstim.ir.qec_patch import QECPatch
from lightstim.qec_code.generic_css.gauge_SE_block import GenericCSSGaugeExtractionBlock


class BaconShorCode(QECPatch):
    """The square ``[[d², 1, (d-1)², d]]`` Bacon-Shor subsystem code.

    Horizontal nearest-neighbor XX and vertical nearest-neighbor ZZ operators
    generate the gauge group. Products along complete pairs of columns/rows
    generate its stabilizer center. Each gauge has a dedicated readout ancilla;
    the default extraction circuit measures all X gauges, then all Z gauges.
    """

    default_extraction_block_class = GenericCSSGaugeExtractionBlock

    def _process_params(self):
        distance = self.params.get("distance")
        if isinstance(distance, bool) or not isinstance(distance, Integral) or distance < 2:
            raise ValueError("Bacon-Shor distance must be an integer >= 2.")
        self.distance = int(distance)

    def build(self):
        d = self.distance
        for row in range(d):
            for col in range(d):
                self.add_qubit(2 * col, 2 * row, role="data")

        for row in range(d):
            for col in range(d - 1):
                ancilla = (2 * col + 1, 2 * row)
                self.add_qubit(*ancilla, role="syndrome_x")
                self.create_stim_gauge(
                    {(2 * col, 2 * row): "X", (2 * col + 2, 2 * row): "X"},
                    syn_coord=ancilla,
                    type="X",
                )

        for row in range(d - 1):
            for col in range(d):
                ancilla = (2 * col, 2 * row + 1)
                self.add_qubit(*ancilla, role="syndrome_z")
                self.create_stim_gauge(
                    {(2 * col, 2 * row): "Z", (2 * col, 2 * row + 2): "Z"},
                    syn_coord=ancilla,
                    type="Z",
                )

        # Center checks are inferred from gauge outcomes, not measured by an
        # extra syndrome ancilla. The declaration stays fixed across X/Z phases.
        for col in range(d - 1):
            self.create_stim_stabilizer(
                {(2 * c, 2 * row): "X" for row in range(d) for c in (col, col + 1)},
                type="X",
            )
        for row in range(d - 1):
            self.create_stim_stabilizer(
                {(2 * col, 2 * r): "Z" for col in range(d) for r in (row, row + 1)},
                type="Z",
            )

        self.create_stim_logical({(0, 2 * row): "X" for row in range(d)}, "X")
        self.create_stim_logical({(2 * col, 0): "Z" for col in range(d)}, "Z")
        self.num_logicals = 1
