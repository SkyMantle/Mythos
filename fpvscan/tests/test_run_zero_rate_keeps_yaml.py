"""USB reporting sample_rate 0 at open must not poison scan/video YAML."""
from __future__ import annotations

from queue import Queue

from fpvscan.engine import Engine
from tests.helpers import MockSource, engine_cfg


class _ZeroRate(MockSource):
    """Board/stream glitch: every read of sample_rate is 0."""

    @property
    def sample_rate(self):
        return 0.0

    def set_sample_rate(self, hz):
        self.set_sample_rate_calls += 1
        # Do not stick — boot must keep the configured rate.


def test_run_does_not_write_zero_rate_into_cfg():
    """#2 / test_engine_lock copies a neighboring actual rate into
    scan+video so LOCK's 1 Hz guard does not flap. #64 LOCK uses
    `or fs_want` when the board already reports 0. The boot path
    is `if self.src.sample_rate:` — a leftover unconditional copy
    writes 0 into YAML, every later sweep grab uses fs=0, and
    LOCK's omitted-key fallback (20e6) is overwritten so
    `or fs_want` still sees 0. USB drop at open() would then
    decode silence for the rest of the session.
    """
    src = _ZeroRate()
    cfg = engine_cfg()
    assert cfg["scan"]["sample_rate"] == 20e6
    assert cfg["video"]["sample_rate"] == 20e6
    eng = Engine(src, cfg, Queue())
    eng._stop.set()
    eng._run()
    assert src.opened is False
    assert src.set_sample_rate_calls >= 1
    assert cfg["scan"]["sample_rate"] == 20e6
    assert cfg["video"]["sample_rate"] == 20e6
