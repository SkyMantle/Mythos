from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from fpvscan.dsp.streaming import (
    ACTIVE_LINES,
    BasebandChunk,
    BasebandFIFO,
    StreamingDemodulator,
    StreamingFieldAssembler,
    VerticalSyncDetector,
    analog_line_hint_hz,
    cvbs_frame,
    interpolate_isolated_h,
)
from fpvscan.iqbuffer import IQRingBuffer
from fpvscan.video_publisher import VideoPublisher


def _fm_iq(fs: float, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    inst = 75e3 + 23e3 * np.sin(2.0 * np.pi * 15_625.0 * t)
    phase = np.cumsum(inst) * (2.0 * np.pi / fs)
    return np.exp(1j * phase).astype(np.complex64)


@pytest.mark.parametrize("decimation", [1, 2, 3])
def test_streaming_demod_matches_one_chunk_at_random_boundaries(
        decimation: int) -> None:
    fs = 2.4e6
    iq = _fm_iq(fs, 120_003)
    kwargs = dict(
        decimation=decimation, mix_hz=137_250.0,
        deviation_hz=480e3, deemphasis_tau=0.5e-6,
    )
    reference = StreamingDemodulator(fs, **kwargs).process(iq, 10_000).samples

    rng = np.random.default_rng(20260915 + decimation)
    streamed = StreamingDemodulator(fs, **kwargs)
    pieces = []
    cursor = 10_000
    offset = 0
    while offset < iq.size:
        count = min(iq.size - offset, int(rng.integers(1, 4097)))
        result = streamed.process(iq[offset:offset + count], cursor)
        pieces.append(result.samples)
        cursor += count
        offset += count
    actual = np.concatenate(pieces)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


def test_nco_fm_and_deemphasis_states_continue_across_single_samples() -> None:
    fs = 1.6e6
    iq = _fm_iq(fs, 20_001)
    one = StreamingDemodulator(
        fs, mix_hz=-211_111.0, deviation_hz=400e3,
    ).process(iq, 0).samples
    stream = StreamingDemodulator(
        fs, mix_hz=-211_111.0, deviation_hz=400e3,
    )
    chunks = [
        stream.process(iq[i:i + 1], i).samples for i in range(iq.size)
    ]
    many = np.concatenate([part for part in chunks if part.size])
    np.testing.assert_allclose(many, one, rtol=2e-5, atol=2e-5)


def test_fifo_absolute_positions_and_gap_reset() -> None:
    fifo = BasebandFIFO(12)
    fifo.append(BasebandChunk(
        np.arange(8, dtype=np.float32), 100, 108, 50, 58,
    ))
    fifo.append(BasebandChunk(
        np.arange(8, 14, dtype=np.float32), 108, 114, 58, 64,
    ))
    assert (fifo.start, fifo.end, fifo.size) == (52, 64, 12)
    np.testing.assert_array_equal(
        fifo.slice(54, 62), np.arange(4, 12, dtype=np.float32),
    )

    was_gap = fifo.append(BasebandChunk(
        np.array([20, 21], dtype=np.float32), 200, 202, 100, 102, gap=True,
    ))
    assert was_gap
    assert fifo.gap_count == 1
    assert (fifo.start, fifo.end, fifo.size) == (100, 102, 2)
    assert fifo.slice(54, 62) is None


def test_iq_ring_unseen_cursor_and_overrun_are_explicit() -> None:
    ring = IQRingBuffer(8)
    ring.write(np.arange(6, dtype=np.float32).astype(np.complex64))
    out = np.empty(8, dtype=np.complex64)
    first, start, cursor, gap = ring.read_since_into(out, 0)
    assert (start, cursor, gap) == (0, 6, False)
    np.testing.assert_array_equal(first.real, np.arange(6))
    empty, start2, cursor2, gap2 = ring.read_since_into(out, cursor)
    assert empty.size == 0
    assert (start2, cursor2, gap2) == (6, 6, False)

    ring.write(np.arange(6, 16, dtype=np.float32).astype(np.complex64))
    tail, start3, cursor3, gap3 = ring.read_since_into(out, cursor)
    assert gap3
    assert (start3, cursor3) == (8, 16)
    np.testing.assert_array_equal(tail.real, np.arange(8, 16))
    ring.write(np.array([30, 31], np.complex64), discontinuity=True)
    jumped, start4, cursor4, gap4 = ring.read_since_into(out, cursor3)
    assert gap4
    assert (start4, cursor4) == (16, 18)
    np.testing.assert_array_equal(jumped.real, [30, 31])


def _synthetic_fields(
    standard: str,
    *,
    fs: float | None = None,
    fields: int = 4,
    missing_active: tuple[int, int] | None = None,
    missing_span: tuple[int, int, int] | None = None,
    serrations: bool = True,
    vblank_level: float | None = None,
) -> tuple[np.ndarray, float, int, list[int]]:
    line_hz = 15_625.0 if standard == "PAL" else 15_734.264
    if fs is None:
        fs = line_hz * 128.0
    period = int(round(fs / line_hz))
    field_lines = 312.5 if standard == "PAL" else 262.5
    vblank = 25 if standard == "PAL" else 20
    field_samples = int(round(field_lines * period))
    lead = 12 * period
    total = lead + fields * field_samples + 20 * period
    x = np.full(total, 0.72, dtype=np.float32)
    sync = max(4, int(round(4.7e-6 * fs)))
    for pos in range(0, total - sync, period):
        x[pos:pos + sync] = 0.02
        row = pos // period
        x[pos + int(0.3 * period):pos + int(0.45 * period)] = (
            0.42 + 0.20 * ((row % 37) / 36.0)
        )
    starts = []
    short = max(2, int(round(2.35e-6 * fs)))
    long = int(round(0.40 * period))
    dark = 0.10 if vblank_level is None else float(vblank_level)
    for field in range(fields):
        v0 = lead + field * field_samples
        starts.append(v0)
        if serrations:
            stop = min(total, v0 + int(8 * period))
            x[v0:stop] = 0.72
            for k in range(15):
                pos = int(round(v0 + 0.5 * k * period))
                width = short if k < 5 or k >= 10 else long
                x[pos:pos + width] = 0.02
        else:
            for line in range(int(vblank)):
                pos = v0 + line * period
                x[pos + sync:pos + int(0.95 * period)] = dark
    if missing_active is not None:
        field, every = missing_active
        v0 = starts[field]
        active = int(v0 + vblank * period)
        end = starts[field + 1]
        first_sync = int(math.ceil(active / period) * period)
        for pos in range(first_sync, end, every * period):
            x[pos:pos + sync] = 0.72
    if missing_span is not None:
        field, start_line, n_lines = missing_span
        v0 = starts[field]
        first = int(math.ceil(
            (v0 + (vblank + int(start_line)) * period) / period,
        ) * period)
        for i in range(int(n_lines)):
            pos = first + i * period
            x[pos:pos + sync] = 0.72
    return x, fs, period, starts


@pytest.mark.parametrize("standard", ["PAL", "NTSC"])
def test_equalizing_serration_vsync_detection(standard: str) -> None:
    base, fs, period, expected = _synthetic_fields(standard)
    detector = VerticalSyncDetector(fs, period)
    found = []
    for start in range(0, len(base), 7331):
        found.extend(detector.feed(base[start:start + 7331], start))
    positions = [item.position for item in found]
    assert len(positions) >= 3
    for want in expected[1:-1]:
        assert min(abs(got - want) for got in positions) < period


@pytest.mark.parametrize(
    ("standard", "expected_lines"),
    [("PAL", 288), ("NTSC", 240)],
)
def test_full_fields_assemble_across_random_chunks(
        standard: str, expected_lines: int) -> None:
    base, fs, period, _ = _synthetic_fields(standard)
    assembler = StreamingFieldAssembler(
        fs, width=96, line_hint_hz=fs / period,
    )
    rng = np.random.default_rng(91)
    output = []
    start = 0
    while start < len(base):
        count = min(len(base) - start, int(rng.integers(5000, 19000)))
        output.extend(assembler.feed(BasebandChunk(
            base[start:start + count],
            start, start + count, start, start + count,
        )))
        start += count
    assert len(output) >= 2
    assert all(frame.locked and not frame.free_run for frame in output)
    assert all(abs(frame.lines - expected_lines) <= 1 for frame in output)
    assert all(abs(frame.luma.shape[0] - expected_lines) <= 1 for frame in output)
    assert assembler.diagnostics()["v_start_jitter_p95_lines"] < 1.0


def test_incomplete_field_is_never_locked_or_submitted() -> None:
    base, fs, period, _ = _synthetic_fields("PAL", missing_active=(1, 2))
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=fs / period,
    )
    output = []
    for start in range(0, len(base), 11_111):
        output.extend(assembler.feed(BasebandChunk(
            base[start:start + 11_111], start, min(len(base), start + 11_111),
            start, min(len(base), start + 11_111),
        )))
    assert assembler.metrics["fields_incomplete"] >= 1
    assert all(frame.lines >= math.ceil(ACTIVE_LINES["PAL"] * 0.95) for frame in output)


def test_randomized_chunk_boundaries_keep_geometry_parity_and_picture() -> None:
    from fpvscan.dsp.cvbs import row_correlation

    base, fs, period, _ = _synthetic_fields("PAL", fields=5)

    def run(seed: int):
        rng = np.random.default_rng(seed)
        assembler = StreamingFieldAssembler(
            fs, width=80, line_hint_hz=fs / period,
        )
        frames = []
        cursor = 0
        while cursor < len(base):
            count = min(len(base) - cursor, int(rng.integers(700, 23_000)))
            frames.extend(assembler.feed(BasebandChunk(
                base[cursor:cursor + count], cursor, cursor + count,
                cursor, cursor + count,
            )))
            cursor += count
        return frames

    left, right = run(3), run(71)
    assert [frame.lines for frame in left] == [frame.lines for frame in right]
    assert [frame.field_parity for frame in left] == [
        frame.field_parity for frame in right
    ]
    np.testing.assert_allclose(
        [row_correlation(frame.luma) for frame in left],
        [row_correlation(frame.luma) for frame in right],
        atol=0.03,
    )


def test_incomplete_height_padding_is_black_not_repeated_picture() -> None:
    from fpvscan.dsp.cvbs import _fit_height

    raster = np.arange(24, dtype=np.uint8).reshape(3, 8)
    fitted = _fit_height(raster, 6)
    np.testing.assert_array_equal(fitted[:3], raster)
    assert np.all(fitted[3:] == 0)
    assert not np.array_equal(fitted[3], raster[-1])


def test_parity_alternates_and_becomes_unknown_after_gap() -> None:
    base, fs, period, _ = _synthetic_fields("PAL", fields=5)
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=fs / period,
    )
    first = assembler.feed(BasebandChunk(base, 0, len(base), 0, len(base)))
    assert [frame.field_parity for frame in first[:2]] == [0, 1]

    shifted = len(base) + 10_000
    after = assembler.feed(BasebandChunk(
        base, shifted, shifted + len(base), shifted, shifted + len(base), gap=True,
    ))
    assert after
    assert after[0].field_parity is None


def test_confirmed_coarse_hint_seeds_once_and_fine_pll_persists() -> None:
    base, fs, period, _ = _synthetic_fields("PAL", fields=5)
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=15_625.0,
    )
    assembler.feed(BasebandChunk(base, 0, len(base), 0, len(base)))
    refined = float(assembler.period)
    assert assembler.seed_coarse_hint(15_500.0) is False
    assert assembler.period == refined
    assert assembler.coarse_hint_hz == pytest.approx(15_625.0)
    assert assembler.seed_coarse_hint(15_500.0, confirmed=True) is False
    assert assembler.period == refined


def test_analog_line_hint_hz_keeps_unconfirmed_comb() -> None:
    assert analog_line_hint_hz(15_641.8) == pytest.approx(15_641.8)
    assert analog_line_hint_hz(0.0) is None
    assert analog_line_hint_hz(None) is None
    assert analog_line_hint_hz(1.2e6) is None


def test_cvbs_frame_drops_unknown_kwargs() -> None:
    frame = cvbs_frame(
        luma=np.zeros((8, 8), np.uint8),
        line_rate=15_625.0,
        lines=8,
        standard="PAL",
        locked=False,
        free_run=True,
        incomplete=True,
        not_a_real_field=True,
    )
    assert frame.free_run is True
    assert frame.lines == 8
    assert not hasattr(frame, "not_a_real_field")


def test_unconfirmed_ntsc_hunt_yields_to_confirmed_pal() -> None:
    assembler = StreamingFieldAssembler(6.7e6, width=64)
    assert assembler.seed_coarse_hint(15_840.0, confirmed=False) is True
    assert assembler.standard == "NTSC"
    assert assembler._period_confirmed is False
    assert assembler.seed_coarse_hint(15_641.0, confirmed=True) is True
    assert assembler.standard == "PAL"
    assert assembler._period_confirmed is True
    period = float(assembler.period)
    assert assembler.seed_coarse_hint(15_840.0, confirmed=True) is False
    assert assembler.period == period


def test_latest_free_run_uses_seeded_period_without_hunting(monkeypatch) -> None:
    from fpvscan.dsp import cvbs

    base, fs, period, _ = _synthetic_fields("PAL", fields=4)
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=fs / period,
    )
    assembler.fifo.append(BasebandChunk(base, 0, len(base), 0, len(base)))

    def boom(*_args, **_kwargs):
        raise AssertionError("line hunt must not run")

    monkeypatch.setattr(cvbs, "estimate_line_hz", boom)
    monkeypatch.setattr(cvbs, "_tbc_line_edges", boom)
    snow = assembler.latest_free_run()
    assert snow is not None
    assert snow.free_run
    assert snow.lines >= int(0.90 * ACTIVE_LINES["PAL"])


def test_complete_latest_fields_use_phase1_single_slot() -> None:
    entered = threading.Event()
    release = threading.Event()

    def encoder(luma, _fmt, _quality, _height, _method):
        entered.set()
        assert release.wait(2)
        return bytes([int(luma[0, 0])])

    publisher = VideoPublisher(encoder=encoder)
    publisher.set_clients(1)
    try:
        assert publisher.submit(np.full((288, 8), 1, np.uint8), {"frame_seq": 1})
        assert entered.wait(2)
        for seq in range(2, 8):
            assert publisher.submit(
                np.full((288, 8), seq, np.uint8), {"frame_seq": seq},
            )
        assert publisher.metrics()["video_pending"] == 1
        assert publisher.metrics()["video_dropped"] >= 5
    finally:
        release.set()
        publisher.stop()


def _feed_random_chunks(
    assembler: StreamingFieldAssembler,
    base: np.ndarray,
    fs: float,
    seed: int,
    *,
    min_s: float = 0.005,
    max_s: float = 0.020,
) -> list:
    rng = np.random.default_rng(seed)
    lo = max(1, int(min_s * fs))
    hi = max(lo + 1, int(max_s * fs) + 1)
    frames = []
    cursor = 0
    n = int(base.size)
    while cursor < n:
        count = min(n - cursor, int(rng.integers(lo, hi)))
        frames.extend(assembler.feed(BasebandChunk(
            base[cursor:cursor + count], cursor, cursor + count,
            cursor, cursor + count,
        )))
        cursor += count
    return frames


@pytest.mark.parametrize("standard", ["PAL", "NTSC"])
def test_hybrid_v_from_energy_troughs_without_serrations(
        standard: str) -> None:
    expected = ACTIVE_LINES[standard]
    base, fs, period, _ = _synthetic_fields(
        standard, fields=5, serrations=False, vblank_level=0.08,
    )
    assembler = StreamingFieldAssembler(
        fs, width=96, line_hint_hz=fs / period,
    )
    output = _feed_random_chunks(assembler, base, fs, 44)
    assert assembler.metrics["v_starts_energy"] >= 2
    assert len(output) >= 1
    assert all(frame.locked and not frame.free_run for frame in output)
    assert all(abs(frame.lines - expected) <= 1 for frame in output)
    assert all(frame.standard == standard for frame in output)


def test_snow_baseband_never_locks_complete_fields() -> None:
    fs = 2.0e6
    rng = np.random.default_rng(2026)
    snow = rng.standard_normal(int(fs * 0.12)).astype(np.float32)
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=15_625.0,
    )
    output = _feed_random_chunks(assembler, snow, fs, 9)
    assert output == []
    assert assembler.metrics["fields_complete"] == 0
    assert all(not frame.locked for frame in output)


def test_isolated_missing_h_edges_complete_via_pll_fill() -> None:
    base, fs, period, _ = _synthetic_fields(
        "PAL", fields=5, missing_active=(1, 9),
    )
    assembler = StreamingFieldAssembler(
        fs, width=80, line_hint_hz=fs / period,
    )
    output = _feed_random_chunks(assembler, base, fs, 12)
    assert assembler.metrics["h_edges_interpolated"] >= 1
    assert assembler.metrics["fields_complete"] >= 1
    assert output
    assert all(frame.locked for frame in output)
    assert all(abs(frame.lines - 288) <= 1 for frame in output)


def test_heavy_missing_h_is_never_locked() -> None:
    every_other, fs, period, _ = _synthetic_fields(
        "PAL", fields=4, missing_active=(1, 2),
    )
    assembler = StreamingFieldAssembler(
        fs, width=64, line_hint_hz=fs / period,
    )
    output = _feed_random_chunks(assembler, every_other, fs, 3)
    assert assembler.metrics["fields_incomplete"] >= 1
    assert all(abs(frame.lines - 288) <= 1 for frame in output)

    hole, fs2, period2, _ = _synthetic_fields(
        "PAL", fields=4, missing_span=(1, 20, 48),
    )
    assembler2 = StreamingFieldAssembler(
        fs2, width=64, line_hint_hz=fs2 / period2,
    )
    output2 = _feed_random_chunks(assembler2, hole, fs2, 5)
    assert assembler2.metrics["fields_complete"] >= 1
    assert all(frame.locked and abs(frame.lines - 288) <= 1 for frame in output2)


def test_interpolate_isolated_h_caps_every_other_gap() -> None:
    period = 100.0
    real = np.arange(0.0, 288.0 * period, 2.0 * period)
    filled, n_interp = interpolate_isolated_h(real, period, 288)
    assert n_interp <= int(0.18 * 288)
    assert filled.size < math.ceil(288 * 0.95)


_PHASE2_IQ_CAPTURES = (
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T110149.590230Z_3370.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T121015.255322Z_3412.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T121357.733502Z_3412.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260916T072228.023433Z_3500.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260916T111105.370536Z_4990.000M.cf32",
)


def _lock_like_stream(path: Path, seed: int):
    from fpvscan.dsp import adaptive_if, cvbs
    from fpvscan.dsp.streaming import StreamingDemodulator

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    fs = float(meta["sample_rate"])
    cap = float(meta.get("channel_bw_hz") or 0.0) or None
    deviation = max(1e5, (cap or fs) / 4.0)
    iq = np.fromfile(path, dtype=np.complex64)
    preview = np.array(iq, copy=True)
    np.subtract(preview, np.mean(preview), out=preview)
    selected = adaptive_if.select(
        preview, fs, base_mix_hz=0.0, channel_cap_hz=cap,
        deviation_hz=deviation, width=160,
    ).selection
    trusted = analog_line_hint_hz(selected.line_rate_hz)
    stream = StreamingDemodulator(
        fs, decimation=int(selected.decimation),
        mix_hz=float(selected.offset_hz), deviation_hz=deviation,
    )
    assembler = StreamingFieldAssembler(
        stream.output_rate_hz, width=160, line_hint_hz=trusted,
    )
    rng = np.random.default_rng(seed)
    lo = max(1, int(0.005 * fs))
    hi = max(lo + 1, int(0.020 * fs) + 1)
    frames = []
    cursor = 0
    while cursor < iq.size:
        count = min(iq.size - cursor, int(rng.integers(lo, hi)))
        chunk = stream.process(iq[cursor:cursor + count], cursor)
        frames.extend(assembler.feed(chunk, line_hint_hz=trusted))
        cursor += count
    return frames, assembler, selected, cvbs.row_correlation


@pytest.mark.parametrize(
    "capture",
    _PHASE2_IQ_CAPTURES,
    ids=lambda path: path.name,
)
def test_lock_like_iq_chunks_assemble_complete_pal_field(
        capture: Path) -> None:
    if not capture.exists() or not capture.with_suffix(".json").exists():
        pytest.skip(f"IQ fixture missing: {capture.name}")
    left, assembler, selected, row_corr = _lock_like_stream(capture, 21)
    right, assembler2, _, _ = _lock_like_stream(capture, 77)
    complete = [frame for frame in left if frame.locked and not frame.free_run]
    complete2 = [frame for frame in right if frame.locked and not frame.free_run]
    print(
        capture.name,
        "v_pulse", assembler.metrics["v_starts_pulse"],
        "v_energy", assembler.metrics["v_starts_energy"],
        "h_real", assembler.metrics["h_edges_real"],
        "h_interp", assembler.metrics["h_edges_interpolated"],
        "fields_complete", assembler.metrics["fields_complete"],
        "lines", [frame.lines for frame in complete],
        "corr", [round(row_corr(frame.luma), 3) for frame in complete],
        "dec", selected.decimation,
        "line_hz", round(float(selected.line_rate_hz), 2),
        "alt_complete", assembler2.metrics["fields_complete"],
        "alt_lines", [frame.lines for frame in complete2],
    )
    assert assembler.metrics["fields_complete"] >= 1
    assert assembler2.metrics["fields_complete"] >= 1
    assert complete and complete2
    for frame in complete + complete2:
        assert frame.standard == "PAL"
        assert 287 <= int(frame.lines) <= 288
        assert 287 <= int(frame.luma.shape[0]) <= 288


def test_predicted_fields_continue_without_new_v_detect() -> None:
    base, fs, period, _ = _synthetic_fields(
        "PAL", fields=6, serrations=False, vblank_level=0.08,
    )
    assembler = StreamingFieldAssembler(
        fs, width=96, line_hint_hz=fs / period,
    )
    chunk = max(1, int(0.008 * fs))
    cursor = 0
    first: list = []
    while cursor < len(base) and assembler.metrics["fields_complete"] < 1:
        count = min(len(base) - cursor, chunk)
        first.extend(assembler.feed(BasebandChunk(
            base[cursor:cursor + count], cursor, cursor + count,
            cursor, cursor + count,
        )))
        cursor += count
    assert assembler.metrics["fields_complete"] == 1
    assert first and first[0].locked and not first[0].free_run
    assert abs(first[0].lines - 288) <= 1

    real_feed = assembler._detector.feed

    def _pulses_without_v(samples, abs_start, gap=False):
        real_feed(samples, abs_start, gap=gap)
        return []

    assembler._detector.feed = _pulses_without_v
    later = []
    while cursor < len(base):
        count = min(len(base) - cursor, chunk)
        later.extend(assembler.feed(BasebandChunk(
            base[cursor:cursor + count], cursor, cursor + count,
            cursor, cursor + count,
        )))
        cursor += count
    predicted = [frame for frame in later if frame.locked and not frame.free_run]
    assert len(predicted) >= 2
    assert assembler.metrics["v_starts_predict"] >= 2
    assert all(abs(frame.lines - 288) <= 1 for frame in predicted)
    t0 = [
        float(frame.field_t0)
        for frame in first + predicted
        if frame.field_t0 is not None
    ]
    assert len(t0) >= 3
    field_period = 312.5 * float(assembler.period)
    deltas = np.diff(t0)
    assert all(abs(float(delta) - field_period) / field_period < 0.08 for delta in deltas)


_POST_PHASE2_IQ_CAPTURES = (
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T133643.428986Z_3700.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T133916.270575Z_3700.000M.cf32",
)
_OPTIONAL_PHASE2_IQ_CAPTURES = (
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T124853.580533Z_5906.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T125240.312895Z_3270.000M.cf32",
    Path(__file__).resolve().parents[1] / "caps"
    / "iq_20260915T125258.114561Z_3860.000M.cf32",
)


@pytest.mark.parametrize(
    "capture",
    _POST_PHASE2_IQ_CAPTURES,
    ids=lambda path: path.name,
)
def test_post_phase2_iq_sustains_complete_pal_fields(capture: Path) -> None:
    if not capture.exists() or not capture.with_suffix(".json").exists():
        pytest.skip(f"IQ fixture missing: {capture.name}")
    frames, assembler, selected, row_corr = _lock_like_stream(capture, 21)
    complete = [frame for frame in frames if frame.locked and not frame.free_run]
    assert selected.confirmed
    assert assembler.metrics["fields_complete"] >= 2
    assert len(complete) >= 2
    t0 = [
        float(frame.field_t0)
        for frame in complete
        if frame.field_t0 is not None
    ]
    if len(t0) >= 2:
        field_period = 312.5 * float(assembler.period)
        assert all(
            abs(float(delta) - field_period) / field_period < 0.20
            for delta in np.diff(t0)
        )
    for frame in complete:
        assert frame.standard == "PAL"
        assert 287 <= int(frame.lines) <= 288
        assert 287 <= int(frame.luma.shape[0]) <= 288


@pytest.mark.parametrize(
    "capture",
    _OPTIONAL_PHASE2_IQ_CAPTURES,
    ids=lambda path: path.name,
)
def test_optional_lock_iq_assembles_complete_pal_field(capture: Path) -> None:
    if not capture.exists() or not capture.with_suffix(".json").exists():
        pytest.skip(f"IQ fixture missing: {capture.name}")
    frames, assembler, selected, _ = _lock_like_stream(capture, 21)
    complete = [frame for frame in frames if frame.locked and not frame.free_run]
    assert selected.confirmed
    assert assembler.metrics["fields_complete"] >= 1
    assert complete
    for frame in complete:
        assert frame.standard == "PAL"
        assert 287 <= int(frame.lines) <= 288


def test_snap_grid_keeps_constant_period() -> None:
    period = 100.0
    grid = 12.0 + np.arange(12, dtype=np.float64) * period
    measured = grid + 6.5
    measured[4] += 18.0
    out = StreamingFieldAssembler._snap_grid_to_h(grid, measured, period)
    assert np.allclose(np.diff(out), period, atol=1e-9)
    assert abs(float(np.median(out - grid)) - 6.5) < 2.0
