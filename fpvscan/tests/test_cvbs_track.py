#!/usr/bin/env python3
"""Трекінг CVBS має цілити в наступний фронт поля, що ще є в знімку."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.dsp import cvbs

FS = 8e6
LINE_PAL = 15625.0
WIDTH = 160


def _pal_cvbs(n: int) -> np.ndarray:
    """PAL-подібний CVBS: рядкова 15625 Гц, поле 312.5 рядків."""
    t = np.arange(n, dtype=np.float64) / FS
    lt = 1.0 / LINE_PAL
    field_t = 312.5 * lt
    x = np.mod(t, lt)
    v = np.full(n, 0.30, dtype=np.float32)
    v[x < 4.7e-6] = 0.0
    in_act = (x >= 9.4e-6) & (x < lt - 1.5e-6)
    u = np.clip((x - 9.4e-6) / (lt - 10.9e-6), 0, 1)
    pic = 0.30 + 0.70 * (np.floor(u * 8) / 7.0)
    v[in_act] = pic[in_act].astype(np.float32)
    # широкі імпульси на початку кожного поля (~3 рядки)
    ft = np.mod(t, field_t)
    vs = ft < (3 * lt)
    if np.any(vs):
        x2 = np.mod(x, lt / 2)
        broad = np.where(x2 < (lt / 2 - 4.7e-6), 0.0, 0.30).astype(np.float32)
        v[vs] = broad[vs]
    return v


def _seed(v: np.ndarray) -> cvbs.DecodeState:
    state = cvbs.DecodeState()
    frame = cvbs.decode(v, FS, width=WIDTH, max_lines=280,
                        state=state, abs_start=0.0)
    assert frame is not None, "сліпий прохід не зібрав поле"
    assert state.period is not None and state.abs_t0 is not None
    assert state.lost == 0
    return state


def test_predict_next_vsync_when_snapshot_starts_mid_field():
    """5 мс після фронту — перша половина поля: round цілив назад."""
    v = _pal_cvbs(int(FS * 0.12))
    state = _seed(v)
    field_period = state.period * cvbs.FIELD_LINES[state.standard]
    abs_start = state.abs_t0 + 0.25 * field_period
    pred = cvbs._predict_local_t0(state, abs_start)
    assert pred is not None
    local_t0, n_fields, _fp = pred
    assert n_fields == 1, f"має бути наступне поле, n={n_fields}"
    assert local_t0 > 0, f"прогноз має бути в знімку, local_t0={local_t0}"
    # ~0.75 поля від початку знімка, не «мінус пів-поля»
    assert abs(local_t0 - 0.75 * field_period) < 0.05 * field_period


def test_tracked_decode_holds_across_mid_field_gap():
    """Типовий LOCK: знімок ~1.7 поля, початок всередині поля.

    Без ceil трекінг промахувався, lost зростав, сліпий пошук
    пересівав кадр — картинка стрибала по вертикалі.
    """
    v = _pal_cvbs(int(FS * 0.15))
    state = _seed(v)
    field_period = state.period * cvbs.FIELD_LINES[state.standard]
    gap = 0.25 * field_period          # 5 мс на PAL
    cap = int(1.7 * field_period)
    abs_start = state.abs_t0 + gap
    i0 = int(abs_start)
    chunk = v[i0:i0 + cap]
    assert len(chunk) == cap

    frame = cvbs.decode(chunk, FS, width=WIDTH, max_lines=280,
                        state=state, abs_start=abs_start)
    assert frame is not None, "трекінг мав зібрати поле"
    assert state.lost == 0, f"трекінг злетів, lost={state.lost}"
    assert frame.locked
    assert frame.lines >= 32


def test_tracked_decode_holds_on_integer_field_gap():
    """Ціле число полів: ceil і round збігаються — регресія не ламає це."""
    v = _pal_cvbs(int(FS * 0.15))
    state = _seed(v)
    field_period = state.period * cvbs.FIELD_LINES[state.standard]
    abs_start = state.abs_t0 + 3 * field_period
    cap = int(1.7 * field_period)
    i0 = int(abs_start)
    chunk = v[i0:i0 + cap]
    frame = cvbs.decode(chunk, FS, width=WIDTH, max_lines=280,
                        state=state, abs_start=abs_start)
    assert frame is not None
    assert state.lost == 0


if __name__ == "__main__":
    test_predict_next_vsync_when_snapshot_starts_mid_field()
    test_tracked_decode_holds_across_mid_field_gap()
    test_tracked_decode_holds_on_integer_field_gap()
    print("OK")
