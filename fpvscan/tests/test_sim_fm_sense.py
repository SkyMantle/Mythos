"""FPV transmitters put sync at the low end of the FM deviation."""
from __future__ import annotations

import numpy as np

from fpvscan.dsp.demod import inst_freq_hz
from fpvscan.sdr.sim import C_LINE_NTSC, Emitter, SimSource


def test_sim_fpv_sync_tip_is_below_blanking():
    """(v-0.30)*dev: sync=0 → below carrier, white=1 → above. Invert this and
    the discriminator's 'picture mean' AFC (if used) would walk the wrong way;
    CVBS still tries both polarities, but occupancy looks USB/LSB-flipped."""
    src = SimSource(
        emitters=[Emitter(5800e6, -10.0, "fpv", 10e6, C_LINE_NTSC, "t")],
        noise_db=-120.0,
        seed=1,
    )
    src.set_sample_rate(20e6)
    src.set_center_freq(5800e6)
    src.set_gain(40.0)
    src.open()
    iq = src.read(int(20e6 * 0.004))  # ~4 ms, several lines
    f = inst_freq_hz(iq, 20e6)
    lo, mid, hi = np.percentile(f, [5, 50, 95])
    assert lo < mid < hi
    # Sync dwells at the low edge; a flipped modulator would put lo above mid.
    assert mid - lo > 1e6
    assert hi - mid > 1e6
