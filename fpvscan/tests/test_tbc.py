from __future__ import annotations

import json
from pathlib import Path
import time

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


def _pal_fields(fs: float = 8e6, n_fields: int = 3, *,
                start_line: int = 0, broad_vsync: bool = True,
                vblank_level: float = 0.26,
                active_lo: float = 0.40, active_hi: float = 0.80,
                fake_vsync_lines: int = 0) -> tuple:
    """Integer-period PAL-ish raster with a vertical bar and 25-line V-blank."""
    period_n = int(round(fs / 15625.0))
    vblank, active = 25, 288
    field = vblank + active
    sync = max(4, int(round(fs * 4.7e-6)))
    a0 = int(round(10.5 / 64.0 * period_n))
    a1 = int(round(62.5 / 64.0 * period_n))
    bar0 = a0 + int((a1 - a0) * 0.42)
    chunks = []
    for _ in range(n_fields):
        for i in range(field):
            line = np.full(period_n, 0.70, dtype=np.float32)
            line[:sync] = 0.0
            line[sync:sync + 6] = 0.30
            if i < 3 and broad_vsync:
                line[:] = 0.0
                half = period_n // 2
                line[half:half + sync] = 0.25
                line[:sync] = 0.0
            elif i < vblank:
                # Above the 0.18 sync threshold so broad-pulse V-sync is absent;
                # line energy still sees a dark 25-line trough (FPV / OSD cams).
                line[:] = float(vblank_level)
                line[:sync] = 0.0
            else:
                y = (i - vblank) / float(active)
                line[a0:a1] = float(active_lo) + float(active_hi - active_lo) * y
                line[bar0:bar0 + 8] = 0.98
            chunks.append(line)
    v = np.concatenate(chunks)
    if start_line:
        v = v[int(start_line) * period_n:]
    if fake_vsync_lines > 0:
        # Mid-field dark burst that used to steal pulse-detector t0.
        nfake = int(fake_vsync_lines) * period_n
        burst = np.zeros(nfake, dtype=np.float32)
        v = np.concatenate([burst, v])
    return v, fs, period_n


def test_tbc_render_holds_vertical_when_period_wrong() -> None:
    """Constant-period slice shears; TBC keeps the bar in one column."""
    true_p = 64.0
    n = 96
    line = np.ones(int(true_p), dtype=np.float32) * 0.55
    line[:5] = 0.05
    line[5:10] = 0.28
    line[28:36] = 0.95
    v = np.tile(line, n + 4)
    nom_p = 63.2
    nom = 3.0 + np.arange(n) * nom_p
    a0, a1, _ = STD_GEOM["PAL"]
    width = 80
    offs = np.linspace(a0 * nom_p, a1 * nom_p, width)
    naive = v[(nom[:, None] + offs[None, :]).astype(np.int32)]
    naive_cols = np.argmax(naive, axis=1)
    assert float(naive_cols.std()) > 8.0

    out = _render(v, nom, nom_p, a0, a1, width,
                  auto_levels=False, sharpen=0.0, h_phase_frac=0.0, thr=0.18)
    cols = np.argmax(out.astype(np.float32), axis=1)
    assert float(cols.std()) < 6.0


def _assert_framed_field(fr) -> None:
    from fpvscan.dsp.cvbs import row_correlation
    luma = fr.luma.astype(np.float32)
    h = luma.shape[0]
    rowm = luma.mean(axis=1)
    k = 20
    c = np.cumsum(np.concatenate(([0.0], rowm)))
    roll = (c[k:] - c[:-k]) / k
    j = int(np.argmin(roll))
    centre = j + k / 2.0
    assert centre < 0.22 * h or centre > 0.78 * h
    bright = rowm > float(np.median(rowm)) * 0.55
    assert int(bright.sum()) >= 40
    cols = np.argmax(luma[bright], axis=1)
    assert float(cols.std()) < 10.0
    assert row_correlation(fr.luma) > 0.4
    # Circular roll glued the previous field onto the footer (bright reset).
    body = rowm[int(0.12 * h):]
    assert float(np.min(np.diff(body))) > -40.0


def test_decode_midfield_weak_vsync_does_not_tear() -> None:
    """288-line pack from mid-field used to glue two half-fields + V-blank."""
    from fpvscan.dsp.cvbs import decode
    v, fs, _ = _pal_fields(start_line=140, broad_vsync=False)
    fr = decode(v, fs, width=160, max_lines=288, state=None)
    assert fr is not None
    _assert_framed_field(fr)


def test_decode_bright_hillside_weak_vblank_frames() -> None:
    """3470: PAL H-lock, V-blank only ~0.04 below a bright field."""
    from fpvscan.dsp.cvbs import decode
    v, fs, _ = _pal_fields(
        start_line=140, broad_vsync=False,
        vblank_level=0.50, active_lo=0.535, active_hi=0.555)
    fr = decode(v, fs, width=160, max_lines=288, state=None)
    assert fr is not None
    _assert_framed_field(fr)


def test_decode_false_early_pulse_loses_to_energy() -> None:
    """Weak pulse on a mid-field burst must not ship the torn 288 pack."""
    from fpvscan.dsp.cvbs import decode
    v, fs, _ = _pal_fields(
        start_line=140, broad_vsync=False, fake_vsync_lines=3)
    fr = decode(v, fs, width=160, max_lines=288, state=None)
    assert fr is not None
    _assert_framed_field(fr)


def test_v_unwrap_drops_preblank_does_not_roll() -> None:
    from fpvscan.dsp.cvbs import _v_blank_row, _v_unwrap
    h, w = 160, 48
    img = np.full((h, w), 160, dtype=np.uint8)
    img[:40] = 200
    img[70:95] = 55
    img[95:] = 90
    assert _v_blank_row(img, band=25) is not None
    out = _v_unwrap(img, band=25)
    assert out.shape[0] < h
    assert float(out[:20].mean()) < 80
    assert float(out[-8:].mean()) < 120


def test_choose_field_t0_prefers_energy_when_pulse_is_far() -> None:
    from fpvscan.dsp.cvbs import _choose_field_t0
    period = 64.0
    pulse = (10.0 * period, None)
    energy = (140.0 * period, 450.0 * period)
    hit = _choose_field_t0(pulse, energy, period)
    assert hit is energy
    agree = _choose_field_t0((140.0 * period, None), energy, period)
    assert agree is pulse or abs(agree[0] - energy[0]) < period


def test_cap_field_lines_never_exceeds_one_field() -> None:
    from fpvscan.dsp.cvbs import (
        Frame, ONE_FIELD_LINES, cap_field_lines, published_lines,
    )
    assert cap_field_lines(2804) == ONE_FIELD_LINES
    assert cap_field_lines(2000, 2000) == ONE_FIELD_LINES
    assert cap_field_lines(80, 80) == 80
    fat = Frame(
        luma=np.zeros((2804, 8), dtype=np.uint8),
        line_rate=15625.0, lines=2804, standard="PAL", locked=True,
    )
    assert published_lines(fat) == ONE_FIELD_LINES


def test_decode_caps_lines_to_one_field() -> None:
    from fpvscan.dsp.cvbs import decode, ONE_FIELD_LINES, published_lines
    v, fs, _ = _pal_fields(n_fields=5)
    fr = decode(v, fs, width=80, max_lines=2000, state=None)
    assert fr is not None
    assert fr.lines <= ONE_FIELD_LINES
    assert fr.luma.shape[0] <= ONE_FIELD_LINES
    assert published_lines(fr) <= ONE_FIELD_LINES


def test_sticky_bad_t0_does_not_ship_black_field() -> None:
    """Predicted t0 with no H-edge must not be the only 288-line output."""
    from fpvscan.dsp.cvbs import (
        DecodeState, analog_usable, choose_lock_raster, decode,
        free_run, needs_lock_fallback, raster_is_black,
    )
    v, fs, period_n = _pal_fields(n_fields=5, start_line=0, broad_vsync=True)
    state = DecodeState()
    n1 = int(fs * 0.040)
    fr1 = decode(v[:n1], fs, width=80, max_lines=288, state=state, abs_start=0.0)
    assert fr1 is not None
    assert state.sticky
    assert not raster_is_black(fr1)

    gap = np.zeros(int(fs * 0.008), dtype=np.float32)
    field = 25 + 288
    abs_pic = int(2 * field * period_n)
    pic = v[abs_pic:abs_pic + n1]
    chunk = np.concatenate([gap, pic])
    state.sticky = True
    state.lost = 0
    state.abs_t0 = 8.0
    fr2 = decode(chunk, fs, width=80, max_lines=288, state=state, abs_start=0.0)
    fallback = free_run(chunk, fs, width=80, max_lines=288)
    show = choose_lock_raster(fr2, fallback)
    assert show is not None
    assert not raster_is_black(show)
    assert float(show.luma.mean()) > 20
    assert show.luma.shape[0] >= 32
    assert state.period is not None
    if fr2 is None or needs_lock_fallback(fr2):
        assert not state.sticky
        assert state.abs_t0 is None
        assert show is fallback
    else:
        assert analog_usable(fr2)
        assert analog_usable(fr2) or show is fallback


def test_sticky_valid_skips_extra_hunt(monkeypatch) -> None:
    """Held lock: next IQ reuses t0 and must not re-run estimate_line_hz / _attempt."""
    import fpvscan.dsp.cvbs as cvbs
    from fpvscan.dsp.cvbs import DecodeState, analog_usable, raster_is_black

    hunts = {"n": 0}
    attempts = {"n": 0}
    real_hz = cvbs.estimate_line_hz
    real_att = cvbs._attempt

    def counted_hz(*a, **k):
        hunts["n"] += 1
        return real_hz(*a, **k)

    def counted_att(*a, **k):
        attempts["n"] += 1
        return real_att(*a, **k)

    monkeypatch.setattr(cvbs, "estimate_line_hz", counted_hz)
    monkeypatch.setattr(cvbs, "_attempt", counted_att)

    v, fs, period_n = _pal_fields(n_fields=5, start_line=0, broad_vsync=True)
    state = DecodeState()
    n1 = int(fs * 0.040)
    fr1 = cvbs.decode(v[:n1], fs, width=80, max_lines=288, state=state,
                      abs_start=0.0)
    assert fr1 is not None
    assert analog_usable(fr1)
    assert not raster_is_black(fr1)
    assert state.sticky
    assert state.abs_t0 is not None
    t0_first = float(state.abs_t0)
    hunts0, att0 = hunts["n"], attempts["n"]
    assert att0 >= 1

    field = 25 + 288
    field_len = field * period_n
    # First decode locks field-1 picture start (~173314 at 8 Msps). That
    # t0 already sits inside [field_len, field_len + n1), so a second
    # window there re-tracks the same H-edge and abs_t0 does not move.
    # Start at the next field boundary so tracking must advance ~one
    # field period without a new hunt.
    abs_start = (int(t0_first) // field_len + 1) * field_len
    chunk = v[abs_start:abs_start + n1]
    fr2 = cvbs.decode(chunk, fs, width=80, max_lines=288, state=state,
                      abs_start=float(abs_start))
    assert fr2 is not None
    assert analog_usable(fr2)
    assert not raster_is_black(fr2)
    assert state.sticky
    assert abs(float(state.abs_t0) - t0_first) > 10.0
    field_t0 = abs_start + 25 * period_n
    assert abs(float(state.abs_t0) - field_t0) < 50 * period_n
    assert hunts["n"] == hunts0
    assert attempts["n"] == att0


def test_weak_window_shares_one_bounded_hunt_with_fallback(monkeypatch) -> None:
    """One IQ/baseband window may run one bounded hunt, including snow."""
    import fpvscan.dsp.cvbs as cvbs

    rng = np.random.default_rng(41)
    fs = 2.0e6
    weak = rng.normal(0.0, 0.03, int(fs * 0.056)).astype(np.float32)
    calls = {"n": 0}
    real = cvbs.estimate_line_hz

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cvbs, "estimate_line_hz", counted)
    shared = cvbs.LineRateHint()
    frame = cvbs.decode(
        weak, fs, width=80, max_lines=96,
        state=cvbs.DecodeState(), line_hint=shared,
    )
    fallback = cvbs.free_run(
        weak, fs, width=80, max_lines=96, line_hint=shared,
    )

    assert calls["n"] == 1
    assert shared.input_samples <= cvbs.LINE_HUNT_MAX_INPUT_SAMPLES
    assert shared.nfft <= cvbs.LINE_HUNT_MAX_NFFT
    assert fallback is not None
    assert fallback.free_run
    assert not cvbs.raster_is_black(fallback)
    assert cvbs.choose_lock_raster(frame, fallback) is not None


def test_line_hunt_budget_is_independent_of_lock_capture_length() -> None:
    import fpvscan.dsp.cvbs as cvbs

    rng = np.random.default_rng(7)
    for fs in (20.0e6, 35.0e6):
        weak = rng.normal(0.0, 0.02, int(fs * 0.056)).astype(np.float32)
        hint = cvbs.LineRateHint()
        hint.resolve(weak, fs)
        assert hint.input_samples <= cvbs.LINE_HUNT_MAX_INPUT_SAMPLES
        assert hint.nfft <= cvbs.LINE_HUNT_MAX_NFFT


def test_existing_weak_iq_fixture_returns_fresh_snow_with_one_hunt(
        monkeypatch) -> None:
    """Project IQ regression only; its recorded center is not operational data."""
    from fpvscan.dsp import cvbs, demod

    caps = Path(__file__).resolve().parents[1] / "caps"
    matches = []
    for sidecar in caps.glob("iq_*.json"):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("note") == "bounded diagnostic":
            matches.append((sidecar, meta))
    if not matches:
        return
    sidecar, meta = matches[0]
    capture = sidecar.with_suffix(".cf32")
    if not capture.exists():
        return

    calls = {"n": 0}
    real = cvbs.estimate_line_hz

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cvbs, "estimate_line_hz", counted)
    iq = np.fromfile(capture, dtype=np.complex64)
    fs = float(meta["sample_rate"])
    channel_bw = float(meta["channel_bw_hz"])
    started = time.perf_counter()
    channel, fs_ch = demod.channelize(iq, fs, 0.0, channel_bw)
    base = demod.fm_demod(channel, fs_ch, deviation_hz=channel_bw / 4.0)
    base = demod.deemphasis(base, fs_ch)
    hint = cvbs.LineRateHint()
    frame = cvbs.decode(
        base, fs_ch, width=160, state=cvbs.DecodeState(), line_hint=hint)
    fallback = cvbs.free_run(base, fs_ch, width=160, line_hint=hint)
    elapsed = time.perf_counter() - started

    assert calls["n"] == 1
    assert hint.nfft <= cvbs.LINE_HUNT_MAX_NFFT
    assert frame is None or not cvbs.analog_usable(frame)
    assert fallback is not None and fallback.free_run
    assert not cvbs.raster_is_black(fallback)
    assert elapsed < 2.0


def test_sticky_midfield_resnaps_not_173_torn() -> None:
    """Sticky holding a mid-picture t0 must snap to V-blank, not publish 173 lines."""
    from fpvscan.dsp.cvbs import (
        DecodeState, ONE_FIELD_LINES, analog_usable, decode, field_needs_resnap,
        published_lines, raster_is_black,
    )
    assert field_needs_resnap(173)
    assert not field_needs_resnap(288)

    v, fs, period_n = _pal_fields(n_fields=5, start_line=0, broad_vsync=True)
    state = DecodeState()
    n1 = int(fs * 0.056)
    fr1 = decode(v[:n1], fs, width=80, max_lines=288, state=state, abs_start=0.0)
    assert fr1 is not None
    assert analog_usable(fr1)
    assert not raster_is_black(fr1)
    assert state.sticky
    assert state.period is not None

    # Mid-active line (~140) is half a PAL field + wrap — operator 173 lines.
    mid = float(140 * period_n)
    state.abs_t0 = mid
    state.sticky = True
    state.lost = 0
    state.t0_err = 0.0
    chunk = v[int(mid):int(mid) + n1]
    fr2 = decode(chunk, fs, width=80, max_lines=288, state=state,
                 abs_start=mid)
    assert fr2 is not None
    assert analog_usable(fr2)
    assert not raster_is_black(fr2)
    assert state.sticky
    assert published_lines(fr2) >= int(ONE_FIELD_LINES * 0.72)
    field = 25 + 288
    field_len = field * period_n
    t0 = float(state.abs_t0)
    # Held geometry must sit near a V-blank, not the planted mid-picture t0.
    assert abs(t0 - mid) > 30 * period_n
    phase = (t0 - 25 * period_n) % field_len
    assert phase < 50 * period_n or phase > field_len - 50 * period_n


def test_dead_sticky_no_hedge_drops_keeps_period() -> None:
    """No H-edge at predicted t0: drop sticky immediately, keep line period."""
    from fpvscan.dsp.cvbs import (
        DecodeState, analog_usable, decode, luma_mean, needs_lock_fallback,
        raster_is_black,
    )
    from fpvscan.scan_view import freeze_lock_afc

    v, fs, _ = _pal_fields(n_fields=4, start_line=0, broad_vsync=True)
    state = DecodeState()
    n1 = int(fs * 0.040)
    fr1 = decode(v[:n1], fs, width=80, max_lines=288, state=state, abs_start=0.0)
    assert fr1 is not None and state.sticky
    kept = state.period
    dead = np.zeros(n1, dtype=np.float32)
    state.sticky = True
    state.lost = 0
    state.abs_t0 = 8.0
    fr2 = decode(dead, fs, width=80, max_lines=288, state=state, abs_start=0.0)
    assert not state.sticky
    assert state.period == kept
    assert needs_lock_fallback(fr2)
    analog_ok = analog_usable(fr2)
    assert not analog_ok
    assert not freeze_lock_afc(
        analog_ok=analog_ok,
        pic_locked=bool(fr2.locked) if fr2 is not None else False,
        pic_score=0.9,
        row_corr=0.9,
        luma_mean=luma_mean(fr2) if fr2 is not None else 0.0,
    )
    if fr2 is not None:
        assert raster_is_black(fr2) or not analog_usable(fr2)


def test_sticky_tracking_holds_t0_across_blocks() -> None:
    """A framed analog hit must not re-hunt t0 on the next IQ block."""
    from fpvscan.dsp.cvbs import DecodeState, decode, field_hold_ok
    v, fs, period_n = _pal_fields(n_fields=5, start_line=0, broad_vsync=True)
    state = DecodeState()
    n1 = int(fs * 0.040)
    fr1 = decode(v[:n1], fs, width=80, max_lines=288, state=state, abs_start=0.0)
    assert fr1 is not None
    assert state.period is not None
    assert field_hold_ok(fr1)
    assert state.sticky
    _assert_framed_field(fr1)

    field = 25 + 288
    abs_start = int(field * period_n)
    chunk = v[abs_start:abs_start + n1].copy()
    burst0 = 90 * period_n
    chunk[burst0:burst0 + 22 * period_n] = 0.02
    fr2 = decode(chunk, fs, width=80, max_lines=288, state=state,
                 abs_start=float(abs_start))
    assert fr2 is not None
    assert state.sticky
    # Burst is in the picture on purpose — do not require a clean field.
    # t0 must stay on the real V-blank, not jump to the painted trough.
    burst_abs = abs_start + burst0
    assert abs(float(state.abs_t0) - burst_abs) > 30 * period_n
    field_t0 = abs_start + 25 * period_n
    assert abs(float(state.abs_t0) - field_t0) < 50 * period_n


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
