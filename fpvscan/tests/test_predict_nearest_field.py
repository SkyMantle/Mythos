"""Tracked decode must wrap to the nearest field, not t0 + n·period drift."""
from __future__ import annotations

from fpvscan.dsp import cvbs


def test_predict_local_t0_rounds_to_the_nearest_field():
    """A ring snapshot can start several fields after the last lock.
    `_predict_local_t0` uses round((abs_start-abs_t0)/field) so the
    search window stays on the closest V-sync, not the previous one
    (`floor`) or a running sample count. #7 only compared PAL vs NTSC
    field length; a 50-line miss is a different bug than wrapping to
    the wrong *field* after a dropped snapshot.
    """
    period = 200.0
    state = cvbs.DecodeState(period=period, standard="PAL", abs_t0=1000.0)
    field = period * cvbs.FIELD_LINES["PAL"]

    late = cvbs._predict_local_t0(state, 1000.0 + 3.6 * field)
    assert late is not None
    local, n, fp = late
    assert n == 4
    assert abs(fp - field) < 1e-6
    assert abs(local - 0.4 * field) < 1e-6

    early = cvbs._predict_local_t0(state, 1000.0 + 3.4 * field)
    local2, n2, _ = early
    assert n2 == 3
    assert abs(local2 + 0.4 * field) < 1e-6
