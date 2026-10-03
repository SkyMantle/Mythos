"""USB recover must restore LO, sample rate, and bias-tee — not just reopen."""
from __future__ import annotations

import inspect

from fpvscan.sdr import bladerf as B


BODY = inspect.getsource(B.BladeRF._recover)


def test_recover_snapshots_tune_before_close():
    """`close()` is allowed to drop `_fc` / `_fs`. Reading them after
    close restores 0 Hz / 0 Msps on the new handle: the next sweep
    step sits on DC and every occupancy is a LO leak.
    """
    snap = BODY.index("fc, fs, g = self._fc, self._fs, self.gain_db")
    close = BODY.index("self.close()")
    assert snap < close


def test_recover_zeros_fs_before_set_sample_rate():
    """After reopen, `_streaming` is False but `_fs` still holds the
    old rate. `set_sample_rate` skips a rewrite when `|Δfs| < 1`
    *and* the module is live; a later path that sets `_streaming`
    first then calls set_sample_rate with the same number never
    runs `_config_stream`. Zeroing `_fs` forces a full resync.
    """
    zero = BODY.index("self._fs = 0.0")
    rate = BODY.index("self.set_sample_rate(fs)")
    assert zero < rate
    assert BODY.index("self.set_gain(g)") < zero


def test_recover_restores_lo_as_int_hz_and_bias_tee():
    """A recovered board that stays on the previous hop's neighbor
    (or with bias-tee off) looks healthy in `snapshot()` — the
    Python flags are unchanged — while the picture is a different
    VTx or the LNA is dark. Frequency must be the saved `fc` as
    `int`, and 4.5 V must be re-applied when `bias_tee` was on.
    """
    rate = BODY.index("self.set_sample_rate(fs)")
    freq = BODY.index("bladerf_set_frequency")
    assert rate < freq
    assert "int(fc)" in BODY[freq:]
    assert "self._fc = fc" in BODY
    bias = BODY.index("if self.bias_tee:")
    assert freq < bias
    assert "self.set_bias_tee(True)" in BODY[bias:]


def test_recover_counts_an_io_error():
    """headless.py prints `src.io_errors`. Forgetting the increment
    makes a USB storm look like a clean run in the 3 s status line.
    """
    assert "self.io_errors += 1" in BODY
    assert BODY.index("self.io_errors += 1") < BODY.index("self.close()")
