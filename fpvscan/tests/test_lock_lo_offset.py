"""LOCK LO-offset: tune beside the channel, mix the inverse digitally."""
from __future__ import annotations

from fpvscan.dsp import demod
from tests.helpers import MockSource, make_engine


def test_lock_tunes_plus_offset_and_channelizes_minus_offset():
    """Wrong mixer sign looks at empty spectrum while the tuner sits on the VTx.

    Physical LO is f+off so the ADC DC hump is not in the picture. The
    channelizer must shift by −off (AFC off in this fixture) or LOCK shows
    snow on a live analog channel.
    """
    fs = 40e6
    off = 8e6
    src = MockSource(fs=fs)
    eng = make_engine(
        src=src, lo_offset_hz=off, channel_bw_hz=10e6, sample_rate=fs,
    )
    eng.cfg["scan"]["sample_rate"] = fs
    eng.cfg["video"]["sample_rate"] = fs
    # 8 + 5 = 13 MHz < 0.45*40 = 18 MHz → offset is kept
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

    assert eng._lock_tuned == 5800e6 + off
    assert src.center_freq == 5800e6 + off
    assert seen_off, "channelize never ran (ring never filled)"
    assert seen_off[0] == -off
