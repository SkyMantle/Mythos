"""Stateful Phase-2 LOCK DSP and complete-field assembly.

The classes in this module consume monotonically increasing absolute sample
ranges.  They deliberately do not know the RF centre: source rate, selected
integer decimation and signal-derived PAL/NTSC timing are the only geometry
inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from scipy import signal

from . import cvbs, demod


ACTIVE_LINES = {"PAL": 288, "NTSC": 240}
MIN_COMPLETE_FRAC = 0.95
MAX_INTERP_FRAC = 0.18
# FPV often misses ~25% of H pulses.  A plausible comb plus analog_usable
# still yields a 288/240 raster; every-other-H snow stays incomplete.
PERIOD_LOCK_MIN_FRAC = 0.70
# Free-run snow may wrap ~two analog fields so a mid-window V start
# still has a full raster.  Shorter windows published 187-line shears.
_FREE_RUN_WINDOW_S = 0.040
_FREE_RUN_MIN_S = 0.020
_FREE_RUN_MIN_FRAC = 0.90


def analog_line_hint_hz(line_hz: float | None) -> float | None:
    """Keep a PAL/NTSC comb even before adaptive IF marks it confirmed."""
    try:
        hz = float(line_hz or 0.0)
    except (TypeError, ValueError):
        return None
    if cvbs.ANALOG_LINE_LO_HZ < hz < cvbs.ANALOG_LINE_HI_HZ:
        return hz
    return None


def cvbs_frame(**kwargs) -> cvbs.Frame:
    """Build a Frame even when Pi still has an older cvbs.Frame layout."""
    fields = getattr(cvbs.Frame, "__dataclass_fields__", None)
    if isinstance(fields, dict) and fields:
        kwargs = {key: value for key, value in kwargs.items() if key in fields}
    return cvbs.Frame(**kwargs)


FIFO_FIELDS = 4.0
_H_WIDTH_MIN = 0.012
_H_WIDTH_MAX = 0.18
_H_THIN_MIN = 0.80
_H_GRID_LO = 0.90
_H_GRID_HI = 1.10
_PREDICT_REFINE_LINES = 8.0
_PREDICT_NEXT_FRAC = 0.05
_PREDICT_MAX_FIELDS = 12
_PHASE_SCALE = np.float32(2.0 * np.pi / (1 << 32))
_PHASE_MASK = np.int64(0xFFFFFFFF)


@dataclass(frozen=True)
class BasebandChunk:
    samples: np.ndarray
    source_start: int
    source_end: int
    base_start: int
    base_end: int
    gap: bool = False


class StreamingDemodulator:
    """Persistent NCO, anti-alias FIR, FM discriminator and deemphasis."""

    def __init__(
        self,
        source_rate_hz: float,
        *,
        decimation: int = 1,
        mix_hz: float = 0.0,
        deviation_hz: float = 4.8e6,
        deemphasis_tau: float = 0.5e-6,
    ) -> None:
        self.source_rate_hz = float(source_rate_hz)
        self.decimation = max(1, int(decimation))
        self.output_rate_hz = self.source_rate_hz / self.decimation
        self.deviation_hz = float(deviation_hz)
        self.deemphasis_tau = float(deemphasis_tau)
        self.mix_hz = float(mix_hz)
        self._nco_phase = np.int64(0)
        self._nco_word = self._phase_word(self.mix_hz)
        self._fir = self._design_fir(self.decimation)
        self._fir_zi = np.zeros(max(0, len(self._fir) - 1), dtype=np.complex64)
        self._fm_previous: np.complex64 | None = None
        self._deemphasis_zi = np.zeros(1, dtype=np.float64)
        self._source_cursor: int | None = None
        self._source_count = 0
        self._base_cursor = 0
        self.last_fm = np.empty(0, dtype=np.float32)
        self.reset_count = 0
        self.timings_ms = {
            "nco": 0.0,
            "filter_decimate": 0.0,
            "fm": 0.0,
            "deemphasis": 0.0,
        }

    def _phase_word(self, hz: float) -> np.int64:
        turns = (float(hz) / self.source_rate_hz) % 1.0
        return np.int64(round(turns * (1 << 32)))

    @staticmethod
    def _design_fir(decimation: int) -> np.ndarray:
        if decimation <= 1:
            return np.ones(1, dtype=np.float32)
        taps = max(9, 8 * int(decimation) + 1)
        return signal.firwin(
            taps, 0.90 / float(decimation), window="hamming",
        ).astype(np.float32)

    @staticmethod
    def _ema(old: float, value: float) -> float:
        return value if old == 0.0 else 0.8 * old + 0.2 * value

    def set_mix_hz(self, mix_hz: float) -> None:
        """Change NCO frequency without resetting accumulated phase."""
        self.mix_hz = float(mix_hz)
        self._nco_word = self._phase_word(self.mix_hz)

    def reset(self, source_start: int = 0) -> None:
        self._nco_phase = np.int64(0)
        self._fir_zi.fill(0)
        self._fm_previous = None
        self._deemphasis_zi.fill(0)
        self._source_cursor = int(source_start)
        self._source_count = 0
        self._base_cursor = int(source_start) // self.decimation
        self.reset_count += 1

    def _mix(self, iq: np.ndarray) -> np.ndarray:
        x = np.asarray(iq, dtype=np.complex64)
        if x.size == 0 or abs(self.mix_hz) < 1.0:
            return x
        phase = (
            self._nco_phase
            + np.arange(x.size, dtype=np.int64) * self._nco_word
        ) & _PHASE_MASK
        angle = phase.astype(np.float32) * _PHASE_SCALE
        lo = np.empty(x.size, dtype=np.complex64)
        lo.real = np.cos(angle)
        lo.imag = -np.sin(angle)
        self._nco_phase = np.int64(
            (self._nco_phase + np.int64(x.size) * self._nco_word) & _PHASE_MASK
        )
        return x * lo

    def _filter_decimate(self, mixed: np.ndarray) -> np.ndarray:
        if self.decimation == 1:
            return np.asarray(mixed, dtype=np.complex64)
        filtered, self._fir_zi = signal.lfilter(
            self._fir, [1.0], mixed, zi=self._fir_zi,
        )
        first = (
            self.decimation - 1 - (self._source_count % self.decimation)
        ) % self.decimation
        return np.asarray(filtered[first::self.decimation], dtype=np.complex64)

    def _fm(self, channel: np.ndarray) -> np.ndarray:
        if channel.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._fm_previous is None:
            if channel.size < 2:
                self._fm_previous = np.complex64(channel[-1])
                return np.empty(0, dtype=np.float32)
            out = demod.fm_demod(
                channel, self.output_rate_hz, deviation_hz=self.deviation_hz,
            )
        else:
            joined = np.empty(channel.size + 1, dtype=np.complex64)
            joined[0] = self._fm_previous
            joined[1:] = channel
            out = demod.fm_demod(
                joined, self.output_rate_hz, deviation_hz=self.deviation_hz,
            )
        self._fm_previous = np.complex64(channel[-1])
        return np.asarray(out, dtype=np.float32)

    def _deemphasis(self, fm: np.ndarray) -> np.ndarray:
        if fm.size == 0:
            return np.empty(0, dtype=np.float32)
        a = math.exp(-1.0 / (self.output_rate_hz * self.deemphasis_tau))
        out, self._deemphasis_zi = signal.lfilter(
            [1.0 - a], [1.0, -a], fm, zi=self._deemphasis_zi,
        )
        return np.asarray(out, dtype=np.float32)

    def process(
        self,
        iq: np.ndarray,
        source_start: int,
        *,
        gap: bool = False,
    ) -> BasebandChunk:
        x = np.asarray(iq, dtype=np.complex64)
        start = int(source_start)
        discontinuous = bool(
            gap
            or (self._source_cursor is not None and start != self._source_cursor)
        )
        if self._source_cursor is None or discontinuous:
            self.reset(start)
        base_start = self._base_cursor

        t0 = time.perf_counter()
        mixed = self._mix(x)
        t1 = time.perf_counter()
        channel = self._filter_decimate(mixed)
        t2 = time.perf_counter()
        fm = self._fm(channel)
        self.last_fm = fm
        t3 = time.perf_counter()
        base = self._deemphasis(fm)
        t4 = time.perf_counter()
        for key, value in (
            ("nco", (t1 - t0) * 1000.0),
            ("filter_decimate", (t2 - t1) * 1000.0),
            ("fm", (t3 - t2) * 1000.0),
            ("deemphasis", (t4 - t3) * 1000.0),
        ):
            self.timings_ms[key] = self._ema(self.timings_ms[key], value)

        self._source_count += x.size
        self._source_cursor = start + x.size
        self._base_cursor += base.size
        return BasebandChunk(
            samples=base,
            source_start=start,
            source_end=start + x.size,
            base_start=base_start,
            base_end=self._base_cursor,
            gap=discontinuous,
        )


class BasebandFIFO:
    """Bounded circular FIFO addressed by absolute baseband positions."""

    def __init__(self, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError("FIFO capacity must be positive")
        self.capacity = int(capacity)
        self._buffer = np.empty(self.capacity, dtype=np.float32)
        self._write = 0
        self.start = 0
        self.end = 0
        self.size = 0
        self.gap_count = 0

    def reset(self, base_start: int = 0, *, count_gap: bool = False) -> None:
        self._write = 0
        self.start = int(base_start)
        self.end = int(base_start)
        self.size = 0
        if count_gap:
            self.gap_count += 1

    def append(self, chunk: BasebandChunk) -> bool:
        x = np.asarray(chunk.samples, dtype=np.float32)
        gap = bool(chunk.gap or (self.size and chunk.base_start != self.end))
        if gap:
            self.reset(chunk.base_start, count_gap=True)
        elif self.size == 0:
            self.start = int(chunk.base_start)
            self.end = int(chunk.base_start)
        if x.size >= self.capacity:
            tail = x[-self.capacity:]
            self._buffer[:] = tail
            self._write = 0
            self.end = int(chunk.base_end)
            self.start = self.end - self.capacity
            self.size = self.capacity
            return gap
        n = int(x.size)
        if n == 0:
            return gap
        first = min(n, self.capacity - self._write)
        self._buffer[self._write:self._write + first] = x[:first]
        if first < n:
            self._buffer[:n - first] = x[first:]
        self._write = (self._write + n) % self.capacity
        self.end = int(chunk.base_end)
        self.size = min(self.capacity, self.size + n)
        self.start = self.end - self.size
        return gap

    def slice(self, abs_start: int, abs_end: int) -> np.ndarray | None:
        lo, hi = int(abs_start), int(abs_end)
        if lo < self.start or hi > self.end or hi < lo:
            return None
        n = hi - lo
        out = np.empty(n, dtype=np.float32)
        if n == 0:
            return out
        oldest = (self._write - self.size) % self.capacity
        offset = lo - self.start
        pos = (oldest + offset) % self.capacity
        first = min(n, self.capacity - pos)
        out[:first] = self._buffer[pos:pos + first]
        if first < n:
            out[first:] = self._buffer[:n - first]
        return out

    def latest(self, count: int) -> tuple[np.ndarray, int]:
        n = min(max(0, int(count)), self.size)
        start = self.end - n
        data = self.slice(start, self.end)
        return (
            np.empty(0, dtype=np.float32) if data is None else data,
            start,
        )


@dataclass(frozen=True)
class SyncPulse:
    start: float
    end: float
    sign: int

    @property
    def width(self) -> float:
        return self.end - self.start

    @property
    def centre(self) -> float:
        return (self.start + self.end) * 0.5


@dataclass(frozen=True)
class VStart:
    position: float
    active_start: float
    standard: str
    sign: int
    structure_votes: int
    source: str = "pulse"


def thin_h_edges(edges: np.ndarray, period: float) -> np.ndarray:
    """Keep at most one measured edge per analog line."""
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.unique(np.asarray(edges, dtype=np.float64))
    min_sep = _H_THIN_MIN * float(period)
    selected = [float(order[0])]
    for t in order[1:]:
        if t - selected[-1] >= min_sep:
            selected.append(float(t))
    return np.asarray(selected, dtype=np.float64)


def interpolate_isolated_h(
    edges: np.ndarray,
    period: float,
    target: int,
    *,
    max_frac: float = MAX_INTERP_FRAC,
) -> tuple[np.ndarray, int]:
    """Fill single-line holes from the PLL period. Long gaps stay empty."""
    if edges.size == 0:
        return np.empty(0, dtype=np.float64), 0
    real = np.unique(np.asarray(edges, dtype=np.float64))
    max_fill = int(math.floor(float(max_frac) * max(1, int(target))))
    p = float(period)
    out = [float(real[0])]
    n_interp = 0
    for a, b in zip(real[:-1], real[1:]):
        ratio = float(b - a) / p
        if n_interp < max_fill and 1.65 <= ratio <= 2.35:
            filled = float(a + p)
            if abs(float(b) - filled) >= 0.65 * p:
                out.append(filled)
                n_interp += 1
        out.append(float(b))
    return np.asarray(out, dtype=np.float64), n_interp


def h_grid_is_plausible(edges: np.ndarray, period: float) -> bool:
    """True when reconstructed starts follow the analog line comb."""
    if edges.size < 24:
        return False
    d = np.diff(np.asarray(edges, dtype=np.float64))
    p = float(period)
    median = float(np.median(d))
    return _H_GRID_LO * p <= median <= _H_GRID_HI * p


class VerticalSyncDetector:
    """Detect equalizing/serration cadence from measured pulse widths."""

    def __init__(self, sample_rate_hz: float, period: float) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.period = float(period)
        self._raw_tail = np.empty(0, dtype=np.float32)
        self._tail_start = 0
        self._pulses: dict[int, list[SyncPulse]] = {1: [], -1: []}
        self._last_candidate: dict[int, float] = {1: -1e30, -1: -1e30}
        self._sign: int | None = None
        self._line_energy: dict[int, list[tuple[float, float]]] = {1: [], -1: []}
        self._energy_pos: dict[int, float | None] = {1: None, -1: None}
        self._energy_troughs: dict[int, list[float]] = {1: [], -1: []}
        self._sync_lo: dict[int, float | None] = {1: None, -1: None}
        self._sync_hi: dict[int, float | None] = {1: None, -1: None}

    def reset(self, position: int = 0) -> None:
        self._raw_tail = np.empty(0, dtype=np.float32)
        self._tail_start = int(position)
        self._pulses = {1: [], -1: []}
        self._last_candidate = {1: -1e30, -1: -1e30}
        self._sign = None
        self._line_energy = {1: [], -1: []}
        self._energy_pos = {1: None, -1: None}
        self._energy_troughs = {1: [], -1: []}
        self._sync_lo = {1: None, -1: None}
        self._sync_hi = {1: None, -1: None}

    def update_period(self, period: float) -> None:
        self.period = float(period)

    @staticmethod
    def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        padded = np.concatenate(([False], mask, [False]))
        d = np.diff(padded.astype(np.int8))
        return np.flatnonzero(d == 1), np.flatnonzero(d == -1)

    def _sync_limits(self, x: np.ndarray, sign: int) -> tuple[float, float] | None:
        if x.size == 0:
            if self._sync_lo[sign] is None:
                return None
            return float(self._sync_lo[sign]), float(self._sync_hi[sign])
        probe = x[::max(1, x.size // 16384)]
        lo, hi = np.percentile(probe, [0.5, 99.5])
        span = float(hi - lo)
        if span < 1e-8:
            if self._sync_lo[sign] is None:
                return None
            return float(self._sync_lo[sign]), float(self._sync_hi[sign])
        if self._sync_lo[sign] is None:
            self._sync_lo[sign], self._sync_hi[sign] = float(lo), float(hi)
        else:
            ref = float(self._sync_hi[sign] - self._sync_lo[sign])
            if span >= 0.45 * max(ref, 1e-8):
                a = 0.18
                self._sync_lo[sign] = (1.0 - a) * float(self._sync_lo[sign]) + a * float(lo)
                self._sync_hi[sign] = (1.0 - a) * float(self._sync_hi[sign]) + a * float(hi)
        return float(self._sync_lo[sign]), float(self._sync_hi[sign])

    def _extract(self, raw: np.ndarray, start: int, sign: int) -> list[SyncPulse]:
        x = raw if sign > 0 else -raw
        if self._sync_lo[sign] is None or self._sync_hi[sign] is None:
            return []
        lo = float(self._sync_lo[sign])
        hi = float(self._sync_hi[sign])
        span = float(hi - lo)
        if span < 1e-8:
            return []
        threshold = float(lo + 0.20 * span)
        begins, ends = self._runs(x < threshold)
        p = self.period
        result = []
        for a, b in zip(begins, ends):
            width = int(b - a)
            if 0.012 * p <= width <= 0.62 * p:
                result.append(SyncPulse(start + float(a), start + float(b), sign))
        return result

    @staticmethod
    def _half_cadence(pulses: list[SyncPulse], period: float) -> bool:
        if len(pulses) < 3:
            return False
        d = np.diff([pulse.centre for pulse in pulses])
        return bool(np.count_nonzero(
            (d >= 0.35 * period) & (d <= 0.65 * period)
        ) >= len(d) - 1)

    def _find_candidates(self, sign: int) -> list[VStart]:
        p = self.period
        pulses = self._pulses[sign]
        short = [q for q in pulses if 0.012 * p <= q.width <= 0.055 * p]
        long = [q for q in pulses if 0.25 * p <= q.width <= 0.58 * p]
        found: list[VStart] = []
        if len(short) < 6 or len(long) < 3:
            return found
        for anchor in long:
            serr = [
                q for q in long
                if anchor.start - 0.8 * p <= q.start <= anchor.start + 3.5 * p
            ]
            pre = [
                q for q in short
                if anchor.start - 4.0 * p <= q.start < anchor.start
            ]
            post = [
                q for q in short
                if anchor.start < q.start <= anchor.start + 7.0 * p
            ]
            if not (
                len(serr) >= 3
                and len(pre) >= 3
                and len(post) >= 3
                and self._half_cadence(serr, p)
                and self._half_cadence(pre[-4:], p)
                and self._half_cadence(post[:4], p)
            ):
                continue
            position = float(pre[-min(5, len(pre))].start)
            if position - self._last_candidate[sign] < 0.40 * 262.5 * p:
                continue
            # Standard is finalized from the seeded/fine line rate, not RF.
            line_hz = self.sample_rate_hz / p
            standard = demod.standard_from_line_rate(line_hz, max_err_hz=None)
            vblank = cvbs.STD_GEOM[standard][2]
            found.append(VStart(
                position=position,
                active_start=position + float(vblank) * p,
                standard=standard,
                sign=sign,
                structure_votes=min(len(pre), 4) + min(len(serr), 4)
                + min(len(post), 4),
                source="pulse",
            ))
            self._last_candidate[sign] = position
        return found

    def _geometry(self) -> tuple[str, int, float]:
        line_hz = self.sample_rate_hz / float(self.period)
        standard = demod.standard_from_line_rate(line_hz, max_err_hz=None)
        vblank = int(cvbs.STD_GEOM[standard][2])
        field_lines = float(cvbs.FIELD_LINES[standard])
        return standard, vblank, field_lines

    def _line_energy_update(
        self, raw: np.ndarray, start: int, sign: int,
    ) -> None:
        x = raw if sign > 0 else -raw
        p = float(self.period)
        if p < 8.0:
            return
        pos = self._energy_pos.get(sign)
        if pos is None or pos < float(start) - p:
            pos = float(start)
        end = float(start + raw.size)
        series = self._line_energy[sign]
        while pos + p <= end + 1e-6:
            rel = int(round(pos - start))
            hi = int(round(rel + p))
            if rel < 0:
                pos += p
                continue
            if hi > raw.size:
                break
            sl = x[rel:hi]
            a = int(0.18 * sl.size)
            b = max(a + 1, int(0.92 * sl.size))
            series.append((pos, float(np.mean(sl[a:b]))))
            pos += p
        self._energy_pos[sign] = pos
        _, _, field_lines = self._geometry()
        keep_from = end - 2.6 * field_lines * p
        self._line_energy[sign] = [
            item for item in series if item[0] >= keep_from
        ]

    def _energy_trough_positions(self, sign: int) -> list[float]:
        series = self._line_energy[sign]
        _, vblank, field_lines = self._geometry()
        k = max(8, int(vblank))
        if len(series) < k + 48:
            return []
        positions = np.fromiter(
            (item[0] for item in series), dtype=np.float64, count=len(series),
        )
        means = np.fromiter(
            (item[1] for item in series), dtype=np.float64, count=len(series),
        )
        c = np.cumsum(np.concatenate(([0.0], means)))
        roll = (c[k:] - c[:-k]) / k
        mid = float(np.median(roll))
        q5, q95 = np.percentile(roll, [5.0, 95.0])
        span = float(q95 - q5)
        if span < 1e-8:
            return []
        need = max(0.10 * span, 0.04 * max(abs(mid), span))
        min_sep = int(round(0.80 * field_lines))
        cand: list[int] = []
        for i in range(1, roll.size - 1):
            if (
                roll[i] <= roll[i - 1]
                and roll[i] <= roll[i + 1]
                and (mid - float(roll[i])) >= need
            ):
                cand.append(i)
        kept: list[int] = []
        for i in cand:
            if kept and i - kept[-1] < min_sep:
                if roll[i] < roll[kept[-1]]:
                    kept[-1] = i
                continue
            kept.append(i)
        return [float(positions[i]) for i in kept]

    def _trough_extra_evidence(self, sign: int, trough: float) -> bool:
        p = float(self.period)
        _, vblank, _ = self._geometry()
        tb = float(trough)
        te = tb + float(vblank) * p
        pulses = self._pulses[sign]
        trough_p = [q for q in pulses if tb <= q.start < te]
        active_p = [q for q in pulses if te <= q.start < te + float(vblank) * p]
        dens_t = len(trough_p) / max(float(vblank), 1.0)
        dens_a = len(active_p) / max(float(vblank), 1.0)
        if dens_t >= dens_a + 0.35 or dens_t >= 1.35:
            return True
        if len(trough_p) < 3:
            return False
        d = np.diff([q.centre for q in trough_p])
        if d.size == 0:
            return False
        half = float(np.mean((d >= 0.35 * p) & (d <= 0.65 * p)))
        full = float(np.mean((d >= 0.80 * p) & (d <= 1.20 * p)))
        return half >= 0.35 and half > full

    def _energy_candidates(
        self, raw: np.ndarray, start: int, sign: int,
    ) -> list[VStart]:
        self._line_energy_update(raw, start, sign)
        p = float(self.period)
        standard, vblank, field_lines = self._geometry()
        expected = field_lines * p
        troughs = self._energy_trough_positions(sign)
        hist = self._energy_troughs[sign]
        for pos in troughs:
            matched = False
            for i, prev in enumerate(hist):
                if abs(pos - prev) < 8.0 * p:
                    hist[i] = pos
                    matched = True
                    break
            if not matched:
                hist.append(pos)
        hist.sort()
        self._energy_troughs[sign] = [
            item for item in hist if item > start - 6.0 * expected
        ][-12:]
        confirmed: list[float] = []
        for pos in self._energy_troughs[sign]:
            partners = [
                item for item in self._energy_troughs[sign]
                if item != pos
                and 0.95 * expected <= abs(pos - item) <= 1.05 * expected
            ]
            if partners:
                confirmed.append(pos)
        if not confirmed:
            return []
        latest = confirmed[-1]
        partners = [
            item for item in confirmed
            if item < latest
            and 0.95 * expected <= latest - item <= 1.05 * expected
        ]
        if not partners:
            return []
        previous = min(partners, key=lambda item: abs(latest - item - expected))
        found: list[VStart] = []
        extra = self._trough_extra_evidence(sign, latest)
        votes = 4 + min(3, len(partners)) + (2 if extra else 0)
        for pos in (previous, latest):
            if pos - self._last_candidate[sign] < 0.40 * 262.5 * p:
                continue
            found.append(VStart(
                position=pos,
                active_start=pos + float(vblank) * p,
                standard=standard,
                sign=sign,
                structure_votes=votes,
                source="energy",
            ))
        return found

    def _sign_h_score(self, sign: int, position: float) -> int:
        p = float(self.period)
        end = position + 80.0 * p
        return sum(
            1 for pulse in self._pulses[int(sign)]
            if position <= pulse.start < end
            and _H_WIDTH_MIN * p <= pulse.width <= _H_WIDTH_MAX * p
        )

    def _preferred_energy_sign(self) -> int | None:
        p = float(self.period)
        counts = {}
        for sign in (1, -1):
            counts[sign] = sum(
                1 for pulse in self._pulses[sign]
                if 0.040 * p <= pulse.width <= 0.13 * p
            )
        best = max(counts, key=counts.get)
        other = -best
        if counts[best] < 24:
            return None
        if counts[best] < max(counts[other] * 1.2, counts[other] + 8):
            return None
        return int(best)

    def _prefer_pulse(
        self, pulse: list[VStart], energy: list[VStart],
    ) -> list[VStart]:
        p = float(self.period)
        merged = list(pulse)
        for item in energy:
            if any(
                item.sign == other.sign
                and abs(item.position - other.position) < 8.0 * p
                for other in pulse
            ):
                continue
            merged.append(item)
        return sorted(merged, key=lambda item: item.position)

    def feed(
        self,
        samples: np.ndarray,
        abs_start: int,
        *,
        gap: bool = False,
    ) -> list[VStart]:
        if gap:
            self.reset(abs_start)
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return []
        keep = max(64, int(round(self._geometry()[2] * self.period)))
        if self._raw_tail.size:
            raw = np.concatenate((self._raw_tail, x))
            start = self._tail_start
        else:
            raw = x
            start = int(abs_start)
        signs: Iterable[int] = (self._sign,) if self._sign is not None else (1, -1)
        pulse_cands: list[VStart] = []
        energy_cands: list[VStart] = []
        cutoff = float(abs_start) - 2.0 * self.period
        overlap = int(round(3.0 * self.period))
        extract_from = max(int(abs_start) - overlap, start)
        extract_rel = max(0, extract_from - start)
        for sign in signs:
            # Keep a pulse that began before the overlap window even when the
            # new window starts in its low run.  Reconstructing that truncated
            # run made one H edge depend on arbitrary chunk boundaries.
            self._sync_limits(raw if sign > 0 else -raw, sign)
            existing = [
                pulse for pulse in self._pulses[sign] if pulse.start < extract_from
            ]
            scanned = self._extract(raw[extract_rel:], extract_from, sign)
            dedup: list[SyncPulse] = []
            for pulse in existing + scanned:
                if dedup and abs(pulse.start - dedup[-1].start) < 0.08 * self.period:
                    if pulse.width > dedup[-1].width:
                        dedup[-1] = pulse
                else:
                    dedup.append(pulse)
            self._pulses[sign] = [
                pulse for pulse in dedup
                if pulse.end >= cutoff - 4.0 * 312.5 * self.period
            ]
            pulse_cands.extend(self._find_candidates(sign))
            energy_cands.extend(self._energy_candidates(raw, start, sign))
        candidates = self._prefer_pulse(pulse_cands, energy_cands)
        if self._sign is None:
            pulse_hits = [item for item in candidates if item.source == "pulse"]
            if pulse_hits:
                best = max(
                    pulse_hits,
                    key=lambda item: (
                        item.structure_votes,
                        self._sign_h_score(item.sign, item.position),
                    ),
                )
                self._sign = best.sign
            else:
                preferred = self._preferred_energy_sign()
                if preferred is None:
                    candidates = []
                else:
                    self._sign = preferred
        if self._sign is not None:
            candidates = [item for item in candidates if item.sign == self._sign]
        for item in candidates:
            if item.position > self._last_candidate[item.sign]:
                self._last_candidate[item.sign] = item.position
        self._raw_tail = np.array(raw[-keep:], copy=True)
        self._tail_start = start + max(0, raw.size - keep)
        return sorted(candidates, key=lambda item: item.position)

    def h_edges(self, sign: int, start: float, end: float) -> np.ndarray:
        p = self.period
        return np.asarray([
            pulse.start for pulse in self._pulses[int(sign)]
            if start <= pulse.start < end
            and _H_WIDTH_MIN * p <= pulse.width <= _H_WIDTH_MAX * p
        ], dtype=np.float64)


class StreamingFieldAssembler:
    """Accumulate baseband and emit only structurally complete fields."""

    def __init__(
        self,
        sample_rate_hz: float,
        *,
        width: int = 640,
        line_hint_hz: float | None = None,
        auto_levels: bool = True,
        sharpen: float = 0.0,
    ) -> None:
        self.sample_rate_hz = float(sample_rate_hz)
        self.width = int(width)
        self.auto_levels = bool(auto_levels)
        self.sharpen = float(sharpen)
        self.period: float | None = None
        self.standard = "?"
        self.coarse_hint_hz: float | None = None
        self._period_confirmed = False
        self.pll_correction = 0.0
        self._detector: VerticalSyncDetector | None = None
        cap = int(self.sample_rate_hz * 0.10)
        self.fifo = BasebandFIFO(max(4096, cap))
        self._last_v: VStart | None = None
        self._next_v: VStart | None = None
        self._tracking = False
        self._v_history: list[VStart] = []
        self._level_lo: float | None = None
        self._level_hi: float | None = None
        self._parity_known = True
        self._next_parity = 0
        self._field_index = 0
        self._jitter_lines: list[float] = []
        self._last_snow_at = 0.0
        self.metrics = {
            "fields_detected": 0,
            "fields_complete": 0,
            "fields_incomplete": 0,
            "fields_dropped": 0,
            "fields_period_locked": 0,
            "h_edges": 0,
            "h_edges_real": 0,
            "h_edges_interpolated": 0,
            "v_starts_pulse": 0,
            "v_starts_energy": 0,
            "v_starts_predict": 0,
            "gap_count": 0,
            "parity": None,
            "field_index": 0,
        }
        if line_hint_hz is not None:
            self.seed_coarse_hint(line_hint_hz, confirmed=True)

    def _apply_period(self, hz: float, *, confirmed: bool) -> None:
        self.coarse_hint_hz = float(hz)
        self.period = self.sample_rate_hz / float(hz)
        self.standard = demod.standard_from_line_rate(hz, max_err_hz=None)
        self._detector = VerticalSyncDetector(self.sample_rate_hz, self.period)
        self._period_confirmed = bool(confirmed)
        self._last_v = None
        self._next_v = None
        self._tracking = False
        self._v_history = []
        self.pll_correction = 0.0

    def seed_coarse_hint(
        self,
        line_hz: float | None,
        *,
        confirmed: bool = False,
    ) -> bool:
        hz = analog_line_hint_hz(line_hz)
        if hz is None:
            return False
        if self.period is None:
            self._apply_period(hz, confirmed=confirmed)
            return True
        if not confirmed:
            return False
        current_hz = self.sample_rate_hz / float(self.period)
        new_std = demod.standard_from_line_rate(hz, max_err_hz=None)
        old_std = demod.standard_from_line_rate(current_hz, max_err_hz=None)
        if self._period_confirmed and (
            new_std == old_std or abs(current_hz - hz) < 250.0
        ):
            return False
        self._apply_period(hz, confirmed=True)
        return True

    def reset(self, base_start: int = 0, *, gap: bool = True) -> None:
        self.fifo.reset(base_start, count_gap=gap)
        if self._detector is not None:
            self._detector.reset(base_start)
        self._last_v = None
        self._next_v = None
        self._tracking = False
        self._v_history = []
        self._parity_known = not gap
        self._next_parity = 0
        if gap:
            self.metrics["gap_count"] += 1
            self.metrics["parity"] = None

    def _ensure_period(self, samples: np.ndarray) -> bool:
        if self.period is not None:
            return True
        probe = np.asarray(samples, dtype=np.float32)
        needed = int(self.sample_rate_hz * cvbs.LINE_HUNT_WINDOW_S)
        if probe.size < needed and self.fifo.size >= int(self.sample_rate_hz * 0.012):
            probe, _ = self.fifo.latest(needed)
        hz = cvbs.estimate_line_hz(probe, self.sample_rate_hz)
        return self.seed_coarse_hint(hz, confirmed=False)

    def _pair_aligned(self, previous: VStart, current: VStart) -> bool:
        if previous.standard != current.standard or previous.sign != current.sign:
            return False
        expected = cvbs.FIELD_LINES[current.standard] * float(self.period)
        error = float(current.position - previous.position - expected)
        return abs(error) <= 0.05 * expected

    def _spacing_ok(self, previous: VStart, current: VStart) -> bool:
        if not self._pair_aligned(previous, current):
            return False
        expected = cvbs.FIELD_LINES[current.standard] * float(self.period)
        error = float(current.position - previous.position - expected)
        self._jitter_lines.append(error / float(self.period))
        self._jitter_lines = self._jitter_lines[-128:]
        # Free-running analog field oscillators can be several percent away
        # while their H comb remains stable.  Structural pulse votes still
        # gate the boundary; this tolerance only links consecutive fields.
        return True

    def _matching_previous(self, current: VStart) -> VStart | None:
        if self.period is None:
            return None
        expected = cvbs.FIELD_LINES[current.standard] * self.period
        compatible = [
            item for item in self._v_history
            if item.standard == current.standard
            and item.sign == current.sign
            and 0.95 * expected <= current.position - item.position
            <= 1.05 * expected
        ]
        if not compatible:
            return None
        return min(
            compatible,
            key=lambda item: abs((current.position - item.position) - expected),
        )

    def _update_pll(self, edges: np.ndarray) -> None:
        if edges.size < 24 or self.period is None:
            return
        diff = np.diff(edges)
        good = diff[
            (diff >= 0.80 * self.period) & (diff <= 1.20 * self.period)
        ]
        if good.size < 16:
            return
        measured = float(np.median(good))
        correction = measured - self.period
        limit = 0.01 * self.period
        correction = float(np.clip(correction, -limit, limit))
        self.period += 0.08 * correction
        self.pll_correction = correction
        if self._detector is not None:
            self._detector.update_period(self.period)

    def _levels(self, samples: np.ndarray) -> tuple[float, float]:
        probe = samples[::max(1, samples.size // 32768)]
        lo_m, hi_m = np.percentile(probe, [2.0, 99.0])
        if hi_m - lo_m < 1e-5:
            lo_m, hi_m = 0.0, 1.0
        if self._level_lo is None:
            self._level_lo, self._level_hi = float(lo_m), float(hi_m)
        else:
            a = 0.18
            self._level_lo = (1.0 - a) * self._level_lo + a * float(lo_m)
            self._level_hi = (1.0 - a) * self._level_hi + a * float(hi_m)
        return self._level_lo, self._level_hi

    @staticmethod
    def _snap_grid_to_h(
        grid: np.ndarray, measured: np.ndarray, period: float,
    ) -> np.ndarray:
        """Keep a constant line period; apply one H-phase to the whole field.

        Snapping every row to the nearest edge independently sheared analog
        FPV into a 234-line diagonal wrap whenever H pulses were noisy.
        """
        if measured.size == 0:
            return np.asarray(grid, dtype=np.float64)
        out = np.asarray(grid, dtype=np.float64)
        measured = np.asarray(measured, dtype=np.float64)
        period = float(period)
        limit = 0.35 * period
        residuals: list[float] = []
        for g in out:
            j = int(np.searchsorted(measured, g))
            best = None
            best_d = limit
            for k in (j - 1, j):
                if 0 <= k < measured.size:
                    d = float(measured[k]) - float(g)
                    if abs(d) <= best_d:
                        best_d = abs(d)
                        best = d
            if best is not None:
                residuals.append(best)
        if not residuals:
            return out.copy()
        return out + float(np.median(np.asarray(residuals, dtype=np.float64)))

    def _sample_edges(
        self, boundary: VStart, edges: np.ndarray,
    ) -> np.ndarray | None:
        assert self.period is not None
        a0, a1, _ = cvbs.STD_GEOM[boundary.standard]
        lo_abs = int(math.floor(edges[0]))
        hi_abs = int(math.ceil(edges[-1] + a1 * self.period + 2))
        raw = self.fifo.slice(lo_abs, hi_abs)
        if raw is None:
            return None
        v = raw if boundary.sign > 0 else -raw
        local = edges - lo_abs
        offsets = np.linspace(
            a0 * self.period, a1 * self.period, self.width, dtype=np.float32,
        )
        index = local[:, None].astype(np.float32) + offsets[None, :]
        if float(index.max(initial=0.0)) >= len(v) - 1:
            return None
        i0 = index.astype(np.int32)
        frac = index - i0
        raster = v[i0] * (1.0 - frac) + v[i0 + 1] * frac
        level_lo, level_hi = self._levels(raster)
        luma = np.clip(
            (raster - level_lo) / max(1e-6, level_hi - level_lo), 0.0, 1.0,
        )
        if self.sharpen > 0.0 and luma.shape[1] > 2:
            blur = luma.copy()
            blur[:, 1:-1] = (
                0.25 * luma[:, :-2] + 0.5 * luma[:, 1:-1]
                + 0.25 * luma[:, 2:]
            )
            luma = np.clip(
                luma + self.sharpen * (luma - blur), 0.0, 1.0,
            )
        return np.asarray(luma * 255.0, dtype=np.uint8)

    def _finish_field(
        self,
        boundary: VStart,
        image: np.ndarray,
        *,
        lines: int,
        period_locked: bool = False,
    ) -> cvbs.Frame:
        parity = self._next_parity if self._parity_known else None
        if self._parity_known:
            self._next_parity ^= 1
        else:
            self._parity_known = True
            self._next_parity = 0
        index_value = self._field_index
        self._field_index += 1
        self.metrics["fields_complete"] += 1
        if period_locked:
            self.metrics["fields_period_locked"] += 1
        self.metrics["parity"] = parity
        self.metrics["field_index"] = index_value
        return cvbs_frame(
            luma=image,
            line_rate=self.sample_rate_hz / float(self.period),
            lines=int(lines),
            standard=boundary.standard,
            locked=True,
            field_parity=parity,
            free_run=False,
            field_t0=boundary.active_start,
        )

    def _drop_incomplete(self) -> None:
        self.metrics["fields_incomplete"] += 1
        self.metrics["fields_dropped"] += 1

    def _render(self, boundary: VStart, next_boundary: VStart) -> cvbs.Frame | None:
        assert self.period is not None and self._detector is not None
        target = ACTIVE_LINES[boundary.standard]
        measured = thin_h_edges(
            self._detector.h_edges(
                boundary.sign,
                boundary.active_start - 0.35 * self.period,
                next_boundary.position - 0.10 * self.period,
            ),
            self.period,
        )
        if measured.size:
            first = int(np.argmin(np.abs(measured - boundary.active_start)))
            measured = measured[first:]
        n_real = int(measured.size)
        self.metrics["h_edges_real"] += n_real
        edges, n_interp = interpolate_isolated_h(
            measured, self.period, target,
        )
        self.metrics["h_edges_interpolated"] += int(n_interp)
        self.metrics["h_edges"] += int(edges.size)
        minimum = int(math.ceil(target * MIN_COMPLETE_FRAC))
        count = min(target, int(edges.size))
        measured_complete = (
            edges.size >= minimum
            and count >= target - 1
            and h_grid_is_plausible(edges, self.period)
        )
        period_ok = (
            n_real >= int(math.ceil(target * PERIOD_LOCK_MIN_FRAC))
            and h_grid_is_plausible(measured, self.period)
        )
        if not measured_complete and not period_ok:
            self._drop_incomplete()
            return None
        p = float(self.period)
        grid = (
            float(boundary.active_start)
            + np.arange(target, dtype=np.float64) * p
        )
        grid = self._snap_grid_to_h(
            grid, measured if measured.size else edges, p,
        )
        self._update_pll(grid)
        image = self._sample_edges(boundary, grid)
        if image is None:
            self._drop_incomplete()
            return None
        if not measured_complete:
            frame = cvbs_frame(
                luma=image,
                line_rate=self.sample_rate_hz / p,
                lines=int(target),
                standard=boundary.standard,
                locked=True,
                free_run=False,
                field_t0=boundary.active_start,
            )
            if not cvbs.analog_usable(frame):
                self._drop_incomplete()
                return None
        return self._finish_field(
            boundary, image, lines=target, period_locked=not measured_complete,
        )

    def _field_period_samples(self, standard: str) -> float:
        return float(cvbs.FIELD_LINES[standard]) * float(self.period)

    def _predicted_vstart(self, previous: VStart) -> VStart:
        p = float(self.period)
        position = float(previous.position) + self._field_period_samples(
            previous.standard,
        )
        vblank = float(cvbs.STD_GEOM[previous.standard][2])
        return VStart(
            position=position,
            active_start=position + vblank * p,
            standard=previous.standard,
            sign=previous.sign,
            structure_votes=0,
            source="predict",
        )

    def _field_window(self, start: VStart, end: VStart) -> tuple[float, float]:
        p = float(self.period)
        lo = float(start.active_start) - 0.35 * p
        hi = float(end.position) + p + 2.0
        return lo, hi

    def _field_in_fifo(self, start: VStart, end: VStart) -> bool:
        lo, hi = self._field_window(start, end)
        return lo >= float(self.fifo.start) and hi <= float(self.fifo.end)

    def _field_aged_out(self, start: VStart) -> bool:
        p = float(self.period)
        lo = float(start.active_start) - 0.35 * p
        return lo < float(self.fifo.start)

    def _nudge_tracking(self, current: VStart) -> None:
        if self._last_v is None or self.period is None:
            return
        if (
            current.standard != self._last_v.standard
            or current.sign != self._last_v.sign
        ):
            return
        p = float(self.period)
        delta = float(current.position) - float(self._last_v.position)
        if abs(delta) <= _PREDICT_REFINE_LINES * p:
            self._last_v = current
            return
        expected = self._field_period_samples(current.standard)
        if abs(delta - expected) <= _PREDICT_NEXT_FRAC * expected:
            self._next_v = current

    def _peek_next_boundary(self) -> VStart | None:
        if self._last_v is None or self.period is None:
            return None
        nxt = self._next_v
        if nxt is not None:
            if (
                nxt.position > self._last_v.position
                and self._pair_aligned(self._last_v, nxt)
            ):
                return nxt
            self._next_v = None
        predicted = self._predicted_vstart(self._last_v)
        if predicted.position <= self._last_v.position:
            return None
        return predicted

    def _consume_next_boundary(self, nxt: VStart) -> None:
        if self._next_v is not None and nxt is self._next_v:
            self._next_v = None
        elif (
            self._next_v is not None
            and abs(float(nxt.position) - float(self._next_v.position))
            < 0.25 * float(self.period)
        ):
            self._next_v = None

    def _emit_predicted_fields(self) -> list[cvbs.Frame]:
        """Render complete fields from predicted V after the first confirmed pair."""
        output: list[cvbs.Frame] = []
        if not self._tracking or self._last_v is None or self.period is None:
            return output
        for _ in range(_PREDICT_MAX_FIELDS):
            nxt = self._peek_next_boundary()
            if nxt is None or self._last_v is None:
                break
            if not self._field_in_fifo(self._last_v, nxt):
                if self._field_aged_out(self._last_v):
                    self.metrics["fields_dropped"] += 1
                    self._consume_next_boundary(nxt)
                    self._last_v = nxt
                    continue
                break
            frame = self._render(self._last_v, nxt)
            self._consume_next_boundary(nxt)
            if nxt.source == "predict":
                self.metrics["v_starts_predict"] += 1
            if nxt.source == "predict" or nxt not in self._v_history:
                self._v_history.append(nxt)
                self._v_history = self._v_history[-16:]
            self._last_v = nxt
            if frame is None:
                continue
            output.append(frame)
        return output

    def line_comb_present(self, *, field_span: float = 1.2) -> bool:
        """True when recent H pulses still follow the analog line comb."""
        if self.period is None or self._detector is None:
            return False
        sign = self._detector._sign
        if sign is None:
            return False
        p = float(self.period)
        field = float(cvbs.FIELD_LINES.get(self.standard, cvbs.FIELD_LINES["?"]))
        start = max(
            float(self.fifo.start),
            float(self.fifo.end) - float(field_span) * field * p,
        )
        edges = thin_h_edges(
            self._detector.h_edges(int(sign), start, float(self.fifo.end)),
            p,
        )
        return h_grid_is_plausible(edges, p)

    def feed(
        self,
        chunk: BasebandChunk,
        *,
        line_hint_hz: float | None = None,
    ) -> list[cvbs.Frame]:
        gap = self.fifo.append(chunk)
        if gap:
            self.reset(chunk.base_start, gap=True)
            self.fifo.append(BasebandChunk(
                chunk.samples, chunk.source_start, chunk.source_end,
                chunk.base_start, chunk.base_end, False,
            ))
        if line_hint_hz is not None:
            self.seed_coarse_hint(line_hint_hz, confirmed=True)
        if not self._ensure_period(chunk.samples):
            return []
        assert self._detector is not None
        candidates = self._detector.feed(
            chunk.samples, chunk.base_start, gap=gap,
        )
        output: list[cvbs.Frame] = []
        for current in candidates:
            self.metrics["fields_detected"] += 1
            if current.source == "energy":
                self.metrics["v_starts_energy"] += 1
            else:
                self.metrics["v_starts_pulse"] += 1
            self._v_history.append(current)
            self._v_history = self._v_history[-16:]
            if not self._tracking:
                previous = self._matching_previous(current)
                if previous is not None and self._spacing_ok(previous, current):
                    self._tracking = True
                    self._last_v = previous
                    self._next_v = current
            else:
                self._nudge_tracking(current)
        if self._tracking:
            output.extend(self._emit_predicted_fields())
        return output

    def latest_free_run(self, *, max_lines: int = 288) -> cvbs.Frame | None:
        """Wrap unsynced analog at the known line period without TBC.

        ``cvbs.free_run`` genlocks every row (±52 % search).  At the unified
        hardware rate that is ~200 ms of GIL and is why LOCK sat at 0.3 fps
        while publishing 288-line snow.
        """
        if self.period is None:
            return None
        p = float(self.period)
        count = min(
            self.fifo.size,
            int(self.sample_rate_hz * _FREE_RUN_WINDOW_S),
        )
        base, _ = self.fifo.latest(count)
        if base.size < int(self.sample_rate_hz * _FREE_RUN_MIN_S):
            return None
        sign = 1 if self._detector is None or self._detector._sign is None else int(
            self._detector._sign
        )
        if sign < 0:
            base = -base
        std = self.standard if self.standard in cvbs.STD_GEOM else "PAL"
        target = ACTIVE_LINES.get(std, 288)
        max_lines = min(int(max_lines), target)
        a0, a1, _ = cvbs.STD_GEOM[std]
        n_lines = min(max_lines, int((len(base) - 2 - a1 * p) / p))
        if n_lines < int(math.ceil(_FREE_RUN_MIN_FRAC * target)):
            return None
        starts = np.arange(n_lines, dtype=np.float64) * p
        offsets = np.linspace(a0 * p, a1 * p, self.width, dtype=np.float32)
        index = starts[:, None].astype(np.float32) + offsets[None, :]
        i0 = index.astype(np.int32)
        frac = index - i0
        raster = base[i0] * (1.0 - frac) + base[i0 + 1] * frac
        lo, hi = np.percentile(raster, [2.0, 99.0])
        if hi - lo < 1e-6:
            return None
        luma = np.clip((raster - lo) / (hi - lo), 0.0, 1.0)
        return cvbs_frame(
            luma=np.asarray(luma * 255.0, dtype=np.uint8),
            line_rate=self.sample_rate_hz / p,
            lines=n_lines,
            standard=std,
            locked=False,
            free_run=True,
            incomplete=True,
        )

    def diagnostics(self) -> dict:
        jitter = np.abs(np.asarray(self._jitter_lines, dtype=np.float64))
        return {
            **self.metrics,
            "fifo_samples": int(self.fifo.size),
            "fifo_age_ms": round(1000.0 * self.fifo.size / self.sample_rate_hz, 2),
            "fifo_start": int(self.fifo.start),
            "fifo_end": int(self.fifo.end),
            "v_start_jitter_p95_lines": (
                0.0 if jitter.size == 0 else round(float(np.percentile(jitter, 95)), 4)
            ),
            "h_period_samples": (
                None if self.period is None else round(float(self.period), 6)
            ),
            "h_pll_correction_samples": round(float(self.pll_correction), 6),
            "h_edges_real": int(self.metrics["h_edges_real"]),
            "h_edges_interpolated": int(self.metrics["h_edges_interpolated"]),
            "v_starts_pulse": int(self.metrics["v_starts_pulse"]),
            "v_starts_energy": int(self.metrics["v_starts_energy"]),
            "v_starts_predict": int(self.metrics["v_starts_predict"]),
            "coarse_hint_hz": self.coarse_hint_hz,
            "standard": self.standard,
        }
