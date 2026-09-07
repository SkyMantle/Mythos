from __future__ import annotations

import numpy as np

from fpvscan.dsp.cvbs import (
    STD_GEOM,
    TBC_DELTA_CLAMP,
    TBC_FOOTER_HOLD,
    TBC_SEARCH_FRAC,
    TBC_TRACK_FRAC,
    _h_crop,
    _render,
    _tbc_hold_footer,
    _tbc_line_edges,
    _tbc_smooth_edges,
    h_pit_in_mid_half,
    tbc_search_frac,
)


def _synth_mid_blank_field(period: float = 64.0, n_lines: int = 48):
    """Baseband field whose H-sync sits mid-line if sliced from a late t0."""
    line = np.ones(int(period), dtype=np.float32) * 0.72
    line[:5] = 0.05
    line[5:10] = 0.28
    line[10:58] = 0.55 + 0.25 * np.linspace(0.0, 1.0, 48, dtype=np.float32)
    line[58:] = 0.32
    v = np.tile(line, n_lines + 4)
    t0 = period * 0.5
    nom = t0 + np.arange(n_lines) * period
    return v, nom, period


def test_tbc_edges_find_sync_when_nominal_is_late() -> None:
    v, nom, period = _synth_mid_blank_field()
    edges = _tbc_line_edges(v, nom, period, thr=0.18)
    phase = np.mod(edges + 1.0, period)
    assert float(np.median(np.minimum(phase, period - phase))) < 3.0


def test_tbc_render_drops_midframe_hblank() -> None:
    v, nom, period = _synth_mid_blank_field()
    a0, a1, _ = STD_GEOM["PAL"]
    width = 80
    offs = np.linspace(a0 * period, a1 * period, width)
    naive = v[(nom[:, None] + offs[None, :]).astype(np.int32)]
    naive_u8 = np.clip((naive - 0.30) / 0.70, 0.0, 1.0)
    naive_col = int(np.argmin(naive_u8.mean(axis=0)))
    assert h_pit_in_mid_half(naive_col, width)

    out = _render(v, nom, period, a0, a1, width,
                  auto_levels=False, sharpen=0.0, h_phase_frac=0.0, thr=0.18)
    mid = out[:, width // 4: 3 * width // 4]
    assert float(mid.mean()) > 80
    pit = int(np.argmin(out.astype(np.float32).mean(axis=0)))
    assert not h_pit_in_mid_half(pit, width)
    cropped = _h_crop(out)
    cmid = cropped[:, cropped.shape[1] // 4: 3 * cropped.shape[1] // 4]
    assert float(cmid.min()) > 40


def test_tbc_median3_kills_single_line_spike() -> None:
    period = 64.0
    n = 24
    true = 10.0 + np.arange(n) * period
    spiked = true.copy()
    spiked[12] += 18.0
    sm = _tbc_smooth_edges(spiked, period)
    assert abs(sm[12] - true[12]) < abs(spiked[12] - true[12]) * 0.4
    delta = np.diff(sm) - period
    assert float(np.max(np.abs(delta))) <= TBC_DELTA_CLAMP + 1e-6


def test_tbc_smooth_bounds_noisy_line_jitter() -> None:
    period = 64.0
    n = 48
    rng = np.random.default_rng(0)
    jitter = rng.normal(0.0, 2.8, n)
    jitter[11] += 12.0
    jitter[27] -= 14.0
    chunks = []
    nom = np.empty(n, dtype=np.float64)
    pos = 8.0
    for i in range(n):
        line = np.ones(int(period), dtype=np.float32) * 0.72
        line[:5] = 0.05
        line[5:10] = 0.28
        line[10:58] = 0.70
        line[58:] = 0.32
        off = int(np.clip(np.round(jitter[i]), -6, 6))
        chunks.append(np.roll(line, off))
        nom[i] = pos + 2.0
        pos += period
    v = np.concatenate(chunks)
    raw = _tbc_line_edges(v, nom, period, thr=0.18, search_frac=TBC_TRACK_FRAC)
    sm = _tbc_smooth_edges(raw, period)
    delta = np.diff(sm) - period
    assert float(np.max(np.abs(delta))) <= TBC_DELTA_CLAMP + 1e-6
    assert tbc_search_frac(locked=False) == TBC_SEARCH_FRAC
    assert tbc_search_frac(locked=True) == TBC_TRACK_FRAC
    assert tbc_search_frac(locked=True, found_frac=0.4) == TBC_SEARCH_FRAC

    v, nom, period = _synth_mid_blank_field()
    a0, a1, _ = STD_GEOM["PAL"]
    width = 80
    out = _render(v, nom, period, a0, a1, width,
                  auto_levels=False, sharpen=0.0, h_phase_frac=0.0, thr=0.18,
                  tbc_locked=False)
    pit = int(np.argmin(out.astype(np.float32).mean(axis=0)))
    assert not h_pit_in_mid_half(pit, width)


def test_tbc_footer_hold_bounds_last_line_delta() -> None:
    period = 64.0
    n = 48
    true = 10.0 + np.arange(n) * period
    noisy = true.copy()
    noisy[-6:] += np.array([9.0, -11.0, 14.0, -8.0, 12.0, -10.0])
    held = _tbc_hold_footer(_tbc_smooth_edges(noisy, period), period)
    tail = np.diff(held[-TBC_FOOTER_HOLD - 1:]) - period
    assert float(np.max(np.abs(tail))) <= 1e-6
    body = np.diff(held[:-TBC_FOOTER_HOLD]) - period
    assert float(np.max(np.abs(body))) <= TBC_DELTA_CLAMP + 1e-6


def test_h_pll_nudge_updates_period_only_when_called() -> None:
    from fpvscan.dsp.cvbs import DecodeState, _pll_nudge_period
    state = DecodeState(period=64.0, standard="PAL")
    frozen = 64.0
    # Off path: helper is not called; period stays.
    assert state.period == frozen
    _pll_nudge_period(state, 64.0, t0=10.0, local_t0_pred=0.0, n_fields=8,
                      standard="PAL")
    assert state.period != frozen
    assert 0.5 * frozen < float(state.period) < 1.5 * frozen
