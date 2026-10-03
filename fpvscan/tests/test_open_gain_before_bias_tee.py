"""BladeRF.open must write gain before applying 4.5 V to the LNA."""
from __future__ import annotations

import inspect

from fpvscan.sdr import bladerf as B


def test_open_checks_fpga_then_gain_then_bias_tee():
    """`open()` with constructor `bias_tee=True` turns on the SMA
    rail. Doing that *before* `set_gain` parks a powered LNA on
    the default RFIC gain for a USB round-trip — ADC clip and a
    striped picture that looks like a dead decoder. FPGA-miss
    must still raise first so a blank bitstream never energizes
    the LNA at all.
    """
    src = inspect.getsource(B.BladeRF.open)
    fpga = src.index("is_fpga_configured")
    gain = src.index("self.set_gain(self.gain_db)")
    bias = src.index("self.set_bias_tee(True)")
    assert fpga < gain < bias
    assert "if self.bias_tee:" in src
    assert src.index("if self.bias_tee:") < bias
