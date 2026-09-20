"""Static LO-offset must zero when it would sit outside the analog LPF."""
from __future__ import annotations

from fpvscan.dsp import demod
from tests.helpers import MockSource, make_engine


def test_lo_offset_that_exceeds_nyquist_headroom_is_not_tuned():
    """Shipped lo_offset is 0 at 35 Msps, but a leftover 8 MHz offset at
    20 Msps with a 10 MHz channel is 13 MHz off DC — past 0.45·fs.
    Without the clamp, LOCK parks the LO beside the VTx and the
    channelizer looks at empty spectrum. The keep-offset path at
    40 Msps is already covered; this is the zeroing path.
    """
    fs = 20e6
    off = 8e6
    bw = 10e6
    assert abs(off) + bw / 2 > fs * 0.45

    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, lo_offset_hz=off, channel_bw_hz=bw, sample_rate=fs,
        capture_ms=8, idle_ms=0, afc=False,
    )
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    eng.state.mode = "LOCK"
    eng.state.lock_target = 5800e6

    seen_off: list[float] = []
    orig = demod.channelize

    def wrap(iq, rate, offset_hz, out_bw_hz, fast=True):
        seen_off.append(float(offset_hz))
        return orig(iq, rate, offset_hz, out_bw_hz, fast=fast)

    demod.channelize = wrap
    try:
        for _ in range(4):
            eng._do_lock()
            if seen_off:
                break
    finally:
        demod.channelize = orig
        eng._stop_reader()

    assert src.center_freq == 5800e6, (
        f"LO parked at {src.center_freq} — clamp must tune f, not f+off"
    )
    assert eng._lock_tuned == 5800e6
    assert seen_off, "channelize never ran (ring never filled)"
    assert seen_off[0] == 0.0
