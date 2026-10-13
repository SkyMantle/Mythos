"""Omitted scan.channel_bw_hz must keep hop overlap above the analog gate."""
from __future__ import annotations

from tests.helpers import make_engine


def test_omitted_scan_channel_bw_keeps_half_channel_overlap():
    """#36 pinned shipped YAML 35 Msps / 10 MHz → 26.5 MHz hop.
    #58 pinned omitted LOCK `video.channel_bw_hz` at 20 MHz. Sweep
    uses `scan.channel_bw_hz` (fallback also 20 MHz) to size the
    auto-step: max(fs·0.25, fs·0.9 − ch/2). At helpers' 20 Msps that
    is 8 MHz, overlap = ch/2. A leftover 0 makes step = 0.9·fs, a
    stitch-sitting VTx falls through the 2.5 MHz analog gate, and
    INSPECT never runs.
    """
    eng = make_engine()
    eng.cfg["scan"].pop("channel_bw_hz")
    assert eng.cfg["scan"]["step_hz"] == 0
    fs = float(eng.cfg["scan"]["sample_rate"])
    eng.cfg["scan"]["start_hz"] = 5700e6
    eng.cfg["scan"]["stop_hz"] = 5900e6
    plan = eng._sweep_plan()
    step = plan[1] - plan[0]
    ch = 20e6
    want = max(fs * 0.25, fs * 0.9 - ch / 2)
    assert step == want == 8e6
    overlap = fs * 0.9 - step
    assert overlap == ch / 2
    visible = (overlap + ch) / 2
    assert visible > 2.5e6
    assert plan[0] == 5700e6 + fs / 2
