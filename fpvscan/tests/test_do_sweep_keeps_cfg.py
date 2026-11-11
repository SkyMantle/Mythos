"""A sweep hop must not replace the YAML dict `_run` already mutated."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_keeps_leftover_cfg():
    """`_run` *does* write the board's actual sample rate into
    `self.cfg["scan"]["sample_rate"]` and
    `self.cfg["video"]["sample_rate"]` before the loop, then
    holds `scan = self.cfg["scan"]` locally. #93 a later LOCK
    field keeps the top cfg dict and the nested video/scan
    dicts so the 1 Hz fs-tolerance restart still sees the
    board rate `_run` wrote. `_do_sweep` *does*
    `scan = self.cfg["scan"]` every hop and compares
    `src.sample_rate` to that rate with the same 1 Hz
    tolerance; a miss calls `set_sample_rate` on a live USB
    handle. The missing half: an 8 MHz hop must not
    `self.cfg = {}` / `self.cfg = dict(self.cfg)` /
    `self.cfg = self.cfg["scan"]` (copy-paste from
    `__init__` or a "fresh YAML view"). A leftover
    replacement still looks like a live walk (`mode=SWEEP`,
    reader down, `retune_calls==1`) while the next hop
    compares against 35e6 from a copied tree, retunes fs
    on the same handle the worker is grabbing, and
    `confirm_hits` / inspect_ms go back to whoever owns
    the old dict. A one-point 5.8 plan keeps this to one
    grab without `fixed_freq` (that flag skips the 1 Hz
    fs compare); zeros IQ so occupancy/inspect stay out
    of the path. Leftover is the same top dict *and* the
    same nested scan/video/sdr dicts after the hop, and
    the board actual rate `_run` wrote must still be what
    the hop reads — no `set_sample_rate` on a live handle.
    """
    src = MockSource()
    src._fs = 19_999_872.0
    eng = make_engine(src)
    # One-point plan, no `fixed_freq`: the hop *does* compare
    # `src.sample_rate` to `cfg["scan"]["sample_rate"]` and
    # would `set_sample_rate(20e6)` on a copied tree.
    eng.cfg["scan"]["start_hz"] = 5790e6
    eng.cfg["scan"]["stop_hz"] = 5810e6
    eng.cfg["scan"]["step_hz"] = 100e6
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    # `_run` writes the board's actual rate into the nested
    # dicts *before* the loop. Pin that mutation on the
    # leftover objects so a shallow `dict(self.cfg)` copy
    # that still holds 20e6 is a miss, not a free pass.
    eng.cfg["scan"]["sample_rate"] = 19_999_872.0
    eng.cfg["video"]["sample_rate"] = 19_999_872.0
    leftover = eng.cfg
    leftover_scan = eng.cfg["scan"]
    leftover_video = eng.cfg["video"]
    leftover_sdr = eng.cfg["sdr"]
    try:
        eng._do_sweep()
        assert src.retune_calls == 1
        assert src.set_sample_rate_calls == 0
        assert eng.cfg is leftover
        assert eng.cfg["scan"] is leftover_scan
        assert eng.cfg["video"] is leftover_video
        assert eng.cfg["sdr"] is leftover_sdr
        assert leftover_scan["sample_rate"] == 19_999_872.0
        assert leftover_video["sample_rate"] == 19_999_872.0
        assert leftover_scan["sample_rate"] != 20e6
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
