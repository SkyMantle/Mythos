"""A sweep hop on file/sim must still retune through the settled dump."""
from __future__ import annotations

from tests.helpers import MockSource, make_engine


def test_do_sweep_grab_uses_settled_retune_when_source_has_no_fast_path():
    """#100 `_grab` prefers `retune_and_read_fast` when the source
    has that callable — BladeRF's quick-tune profile. File/sim have
    no such method; factory replay is the operator's offline path.
    `_grab` is `getattr(src, "retune_and_read_fast", None)` then
    settled `retune_and_read`. A leftover always-fast
    (`self.src.retune_and_read_fast(f, n)` / dropping the getattr)
    AttributeErrors on the first 8 MHz step of a .cf32 / `--driver
    sim` session, the 400 MHz–6 GHz walk never starts, and
    `confirm_hits=2` never lights the bird. MockSource is file/sim
    shaped (no fast method). `fixed_freq` keeps this to one grab;
    zeros IQ so occupancy/inspect stay out of the path (inspect is
    #21; hop-with-fast is #100).
    """
    src = MockSource()
    src.fixed_freq = True
    src._fc = 5800e6
    assert not hasattr(src, "retune_and_read_fast")
    eng = make_engine(src)
    eng.cfg["scan"]["fft_size"] = 64
    eng.cfg["scan"]["averages"] = 1
    try:
        eng._do_sweep()
        assert src.retune_calls == 1
        assert src.center_freq == 5800e6
        assert src.retune_calls != 0
        assert not hasattr(src, "retune_and_read_fast")
        assert eng._lock_tuned is None
        assert eng._reader_thread is None
        assert eng._ring is None
        assert eng.state.mode == "SWEEP"
    finally:
        eng._stop_reader()
