"""Bounded, signal-driven DSP channelization for analog LOCK.

RF center is deliberately absent from this module.  Candidate widths come
from the capture sample rate and integer decimations supported by the
channelizer.  A short preview chooses among them using analog line-comb and
raster evidence; the chosen path is then cached by the engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import time

import numpy as np

from . import cvbs, demod


MAX_CANDIDATES = 3
FILTER_USABLE_FRAC = 0.90
PREVIEW_SECONDS = 0.024
MAX_WINDOWS = 3
MIN_CONFIRM_WINDOWS = 2
OFFSET_HYSTERESIS = 0.12
CHANNEL_HYSTERESIS = 0.02
AGGREGATE_HYSTERESIS = 0.01
MIN_SPECTRAL_RETENTION = 0.30
ANALOG_WINDOW_ROW_CORR = 0.06
ANALOG_MEDIAN_ROW_CORR = 0.08
ANALOG_MEDIAN_PROMINENCE_DB = 12.0
ANALOG_MIN_VERTICAL_DETAIL = 1.0
DETAIL_ROW_WEIGHT = 0.10
DETAIL_PROMINENCE_WEIGHT = 0.06
OFFSET_PROBE_RETENTION = 0.55
OFFSET_PROBE_ROW_CORR = 0.25
OFFSET_PROBE_PROMINENCE_DB = 18.0
LOSS_FRAMES = 8


def analog_line_present(line_rate_hz: float | None) -> bool:
    """True for a PAL/NTSC comb even when the assembler has no period yet."""
    try:
        hz = float(line_rate_hz or 0.0)
    except (TypeError, ValueError):
        return False
    return cvbs.ANALOG_LINE_LO_HZ < hz < cvbs.ANALOG_LINE_HI_HZ


def keep_cached_selection(
    *,
    line_rate_hz: float | None = None,
    assembler_period: float | None = None,
    assembler_sample_rate_hz: float | None = None,
) -> bool:
    """Do not recache IF while a PAL/NTSC line rate is already in the frame."""
    if analog_line_present(line_rate_hz):
        return True
    period = assembler_period
    fs = assembler_sample_rate_hz
    if period is None or period <= 0 or fs is None or float(fs) <= 0.0:
        return False
    return analog_line_present(float(fs) / float(period))


@dataclass(frozen=True)
class Candidate:
    decimation: int
    output_rate_hz: float
    effective_bw_hz: float
    cutoff_hz: float


@dataclass(frozen=True)
class CandidateScore:
    score: float
    line_rate_hz: float
    line_prominence_db: float
    line_confidence: float
    row_corr: float
    sync_score: float
    spectral_retention: float
    clip_penalty: float
    noise_penalty: float
    analog: bool
    vertical_detail: float = 0.0


@dataclass(frozen=True)
class Selection:
    decimation: int
    effective_bw_hz: float
    cutoff_hz: float
    offset_hz: float
    score: float
    line_rate_hz: float
    line_prominence_db: float
    row_corr: float
    analog: bool
    reason: str
    generation: int
    candidates_evaluated: int
    confirmed: bool = False
    stable: bool = False
    windows_evaluated: int = 1
    winner_votes: int = 1
    spectral_retention: float = 0.0
    line_confidence: float = 0.0
    sync_score: float = 0.0
    vertical_detail: float = 0.0

    def diagnostics(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Evaluation:
    selection: Selection
    base: np.ndarray
    fm_base: np.ndarray
    fs_ch: float
    line_hint: cvbs.LineRateHint
    input_offset_samples: int = 0


def numeric_channel_cap(value: Any) -> float | None:
    """Numeric ``video.channel_bw_hz`` is a maximum preferred DSP width.

    ``auto``/missing/non-positive values impose no cap.  Decimation 1 is
    retained as a safety baseline even when its transition-safe width exceeds
    the cap, so an unsuitable narrow path can never win merely by policy.
    """
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0.0 else None


def generate_candidates(
    sample_rate_hz: float,
    *,
    channel_cap_hz: float | None = None,
    minimum_effective_bw_hz: float | None = None,
    max_candidates: int = MAX_CANDIDATES,
    usable_fraction: float = FILTER_USABLE_FRAC,
) -> list[Candidate]:
    """Generate bounded integer-decimation candidates, independent of RF.

    The transition-safe effective width is ``usable_fraction * fs / dec``;
    cutoff is half of that.  A numeric operator width is only a preferred
    upper cap.  No candidate is tied to an RF center or named band.
    """
    fs = float(sample_rate_hz)
    if fs <= 0.0:
        return []
    limit = max(1, min(int(max_candidates), MAX_CANDIDATES))
    frac = float(np.clip(usable_fraction, 0.5, 0.98))
    cap = numeric_channel_cap(channel_cap_hz)
    minimum = numeric_channel_cap(minimum_effective_bw_hz)
    out: list[Candidate] = []
    for dec in range(1, MAX_CANDIDATES + 1):
        output = fs / dec
        effective = output * frac
        if dec > 1 and cap is not None and effective > cap:
            continue
        # FM deviation supplies a signal-derived lower width bound.  At a
        # minimal shared source rate this naturally leaves dec=1; at higher
        # rates valid decimated paths remain available.
        if dec > 1 and minimum is not None and effective < minimum:
            continue
        out.append(Candidate(
            decimation=dec,
            output_rate_hz=output,
            effective_bw_hz=effective,
            cutoff_hz=effective / 2.0,
        ))
        if len(out) >= limit:
            break
    if not out:
        out.append(Candidate(1, fs, fs * frac, fs * frac / 2.0))
    return out


def _clip_fraction(iq: np.ndarray) -> float:
    if iq.size < 32:
        return 0.0
    mag = np.abs(iq[::max(1, iq.size // 8192)])
    peak = float(np.max(mag))
    if peak <= 1e-12:
        return 0.0
    return float(np.mean(mag >= peak * 0.995))


def _occupied_spectrum(iq: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One shared short FFT representing observed occupied baseband energy."""
    n = min(len(iq), 4096)
    if n < 256:
        return np.empty(0), np.empty(0)
    x = np.asarray(iq[-n:], dtype=np.complex64)
    power = np.abs(np.fft.fftshift(
        np.fft.fft(x * np.hanning(n).astype(np.float32))
    )) ** 2
    floor = float(np.percentile(power, 25.0))
    occupied = np.maximum(power - floor, 0.0)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / float(fs)))
    return freqs, occupied


def _spectral_retention(
    profile: tuple[np.ndarray, np.ndarray],
    *,
    mix_hz: float,
    cutoff_hz: float,
) -> float:
    freqs, occupied = profile
    total = float(np.sum(occupied))
    if total <= 1e-20 or freqs.size == 0:
        return 0.0
    keep = np.abs(freqs - float(mix_hz)) <= float(cutoff_hz)
    return float(np.clip(np.sum(occupied[keep]) / total, 0.0, 1.0))


def _candidate_score(
    base: np.ndarray,
    fs_ch: float,
    *,
    input_clip_frac: float,
    spectral_retention: float,
    width: int,
) -> tuple[CandidateScore, cvbs.LineRateHint]:
    video = demod.classify_video(
        base, fs_ch, tol_hz=180.0, min_prominence_db=0.0,
        min_conf=0.0, min_harmonics=0,
    )
    standard_ok = video.standard in ("PAL", "NTSC")
    # Share the comb result with free_run: no second line-rate estimator FFT.
    line_hz = float(video.line_rate) if standard_ok else None
    hint = cvbs.LineRateHint(line_hz=line_hz, attempted=True)
    frame = cvbs.free_run(
        base, fs_ch, width=max(80, min(int(width), 160)), line_hint=hint,
    )
    picture = cvbs.score_picture(frame)
    vertical_detail = 0.0
    if frame is not None and frame.luma is not None and frame.luma.shape[0] >= 4:
        luma = np.asarray(frame.luma, dtype=np.float32)
        row_delta = np.mean(np.abs(np.diff(luma, axis=0)), axis=1)
        vertical_detail = float(np.median(row_delta))
    prom_n = float(np.clip((video.prominence_db - 3.0) / 30.0, 0.0, 1.0))
    corr_n = float(np.clip(picture.row_corr / 0.35, 0.0, 1.0))
    line_n = float(np.clip(video.confidence, 0.0, 1.0)) if standard_ok else 0.0
    clip_penalty = float(np.clip(input_clip_frac * 2.5, 0.0, 0.25))
    noise_penalty = 0.12 if picture.row_corr <= 0.0 else 0.0
    occupancy_penalty = 0.08 * (1.0 - float(np.clip(spectral_retention, 0.0, 1.0)))
    total = (
        0.42 * prom_n
        + 0.38 * corr_n
        + 0.12 * line_n
        + 0.08 * float(np.clip(picture.value, 0.0, 1.0))
        - clip_penalty
        - noise_penalty
        - occupancy_penalty
    )
    analog = bool(
        standard_ok
        and video.prominence_db >= 8.0
        and picture.row_corr >= ANALOG_WINDOW_ROW_CORR
        and picture.lines >= 32
        and vertical_detail >= ANALOG_MIN_VERTICAL_DETAIL
    )
    return CandidateScore(
        score=float(np.clip(total, 0.0, 1.0)),
        line_rate_hz=float(video.line_rate),
        line_prominence_db=float(video.prominence_db),
        line_confidence=float(video.confidence),
        row_corr=float(picture.row_corr),
        sync_score=float(picture.value),
        spectral_retention=float(spectral_retention),
        clip_penalty=clip_penalty,
        noise_penalty=noise_penalty,
        analog=analog,
        vertical_detail=vertical_detail,
    ), hint


def _aggregate_scores(scores: list[CandidateScore]) -> CandidateScore:
    """Median temporal evidence; ``analog`` requires repeated good windows."""
    if not scores:
        raise ValueError("at least one candidate score is required")

    def med(name: str) -> float:
        return float(np.median([float(getattr(item, name)) for item in scores]))

    evidence = sum(bool(item.analog) for item in scores)
    enough = min(MIN_CONFIRM_WINDOWS, len(scores))
    analog = bool(
        len(scores) >= MIN_CONFIRM_WINDOWS
        and evidence >= enough
        and med("row_corr") >= ANALOG_MEDIAN_ROW_CORR
        and med("line_prominence_db") >= ANALOG_MEDIAN_PROMINENCE_DB
        and med("line_confidence") >= 0.25
        and med("vertical_detail") >= ANALOG_MIN_VERTICAL_DETAIL
    )
    return CandidateScore(
        score=med("score"),
        line_rate_hz=med("line_rate_hz"),
        line_prominence_db=med("line_prominence_db"),
        line_confidence=med("line_confidence"),
        row_corr=med("row_corr"),
        sync_score=med("sync_score"),
        spectral_retention=med("spectral_retention"),
        clip_penalty=med("clip_penalty"),
        noise_penalty=med("noise_penalty"),
        analog=analog,
        vertical_detail=med("vertical_detail"),
    )


def _guarded_score(score: CandidateScore) -> float:
    """Rank robust evidence without flattening strong candidate differences.

    The bounded base score intentionally saturates once a candidate is clearly
    analog.  At that point adjacent channel widths can otherwise tie even when
    one preserves materially better raster correlation and line-comb detail.
    These small, capped tie-breakers retain that information; confirmation
    still uses the independent multi-window thresholds above.
    """
    value = float(score.score)
    value += DETAIL_ROW_WEIGHT * float(np.clip(
        (score.row_corr - 0.35) / 0.65, 0.0, 1.0,
    ))
    value += DETAIL_PROMINENCE_WEIGHT * float(np.clip(
        (score.line_prominence_db - 33.0) / 20.0, 0.0, 1.0,
    ))
    if score.spectral_retention < MIN_SPECTRAL_RETENTION and not score.analog:
        value -= 0.25
    return value


def _preview_windows(
    iq: np.ndarray,
    fs: float,
    *,
    max_windows: int = MAX_WINDOWS,
    preview_seconds: float = PREVIEW_SECONDS,
) -> list[tuple[int, np.ndarray]]:
    """Up to three non-overlapping-equivalent temporal previews."""
    width = max(1, int(float(fs) * float(preview_seconds)))
    if len(iq) <= width:
        return [(0, np.asarray(iq, dtype=np.complex64))]
    count = min(max(1, int(max_windows)), max(1, len(iq) // width))
    starts = np.linspace(0, len(iq) - width, count, dtype=np.int64)
    return [
        (int(start), np.asarray(iq[int(start):int(start) + width],
                                dtype=np.complex64))
        for start in starts
    ]


def _demodulate(
    iq: np.ndarray,
    sample_rate_hz: float,
    mix_hz: float,
    candidate: Candidate,
    deviation_hz: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    channel, fs_ch = demod.channelize(
        iq, sample_rate_hz, mix_hz, candidate.effective_bw_hz,
        decimation=candidate.decimation,
    )
    fm_base = demod.fm_demod(channel, fs_ch, deviation_hz=deviation_hz)
    return demod.deemphasis(fm_base, fs_ch), fm_base, fs_ch


def offset_improves(
    baseline: CandidateScore,
    trial: CandidateScore,
    *,
    hysteresis: float = OFFSET_HYSTERESIS,
) -> bool:
    """Accept a prelock/AFC nudge only after material signal improvement."""
    if trial.score < baseline.score + max(0.0, float(hysteresis)):
        return False
    if baseline.analog and not trial.analog:
        return False
    return trial.line_prominence_db >= baseline.line_prominence_db - 1.0


def select(
    iq: np.ndarray,
    sample_rate_hz: float,
    *,
    base_mix_hz: float = 0.0,
    channel_cap_hz: float | None = None,
    deviation_hz: float = 4.8e6,
    width: int = 160,
    proposed_nudge_hz: float = 0.0,
    generation: int = 1,
    offset_hysteresis: float = OFFSET_HYSTERESIS,
    max_candidates: int = MAX_CANDIDATES,
    max_windows: int = MAX_WINDOWS,
    preview_seconds: float = PREVIEW_SECONDS,
) -> Evaluation:
    """Confirm a path over bounded temporal previews, then return latest base.

    Three candidates × at most three 24 ms windows are scored.  A proposed
    offset is checked only for the confirmed candidate and on at most two
    windows.  Thus selection is bounded to 11 candidate-window evaluations,
    once per lifecycle generation rather than once per frame.
    """
    candidates = generate_candidates(
        sample_rate_hz, channel_cap_hz=channel_cap_hz,
        minimum_effective_bw_hz=max(0.0, float(deviation_hz) * 1.8),
        max_candidates=max_candidates,
    )
    if not candidates:
        raise ValueError("sample rate must be positive")
    fs = float(sample_rate_hz)
    windows = _preview_windows(
        np.asarray(iq, dtype=np.complex64),
        fs,
        max_windows=max_windows,
        preview_seconds=preview_seconds,
    )
    records: dict[
        int,
        list[
            tuple[
                CandidateScore, np.ndarray, np.ndarray, float,
                cvbs.LineRateHint, int,
            ]
        ],
    ] = {candidate.decimation: [] for candidate in candidates}
    votes = {candidate.decimation: 0 for candidate in candidates}

    for start, preview in windows:
        clip = _clip_fraction(preview)
        occupied = _occupied_spectrum(preview, fs)
        window_scores: list[tuple[Candidate, CandidateScore]] = []
        for candidate in candidates:
            base, fm_base, fs_ch = _demodulate(
                preview, fs, base_mix_hz, candidate, deviation_hz,
            )
            score, hint = _candidate_score(
                base, fs_ch, input_clip_frac=clip,
                spectral_retention=_spectral_retention(
                    occupied,
                    mix_hz=base_mix_hz,
                    cutoff_hz=candidate.cutoff_hz,
                ),
                width=width,
            )
            records[candidate.decimation].append(
                (score, base, fm_base, fs_ch, hint, start),
            )
            window_scores.append((candidate, score))

        peak = max(_guarded_score(item[1]) for item in window_scores)
        # A near-tied width is supported by this window too.  Counting only
        # one exact winner made two consistently good adjacent decimations
        # look temporally unstable and forced the visibly worse baseline.
        for candidate, candidate_score in window_scores:
            if _guarded_score(candidate_score) >= peak - CHANNEL_HYSTERESIS:
                votes[candidate.decimation] += 1

    aggregates = {
        candidate.decimation: _aggregate_scores(
            [item[0] for item in records[candidate.decimation]],
        )
        for candidate in candidates
    }
    needed_votes = min(MIN_CONFIRM_WINDOWS, len(windows))
    voted = [
        candidate for candidate in candidates
        if votes[candidate.decimation] >= needed_votes
    ]
    baseline = min(candidates, key=lambda item: item.decimation)
    stable = bool(voted and len(windows) >= MIN_CONFIRM_WINDOWS)
    if stable:
        aggregate_peak = max(
            _guarded_score(aggregates[item.decimation]) for item in voted
        )
        candidate = min(
            (
                item for item in voted
                if _guarded_score(aggregates[item.decimation])
                >= aggregate_peak - AGGREGATE_HYSTERESIS
            ),
            key=lambda item: item.decimation,
        )
    else:
        candidate = baseline

    score = aggregates[candidate.decimation]
    confirmed = bool(stable and score.analog)
    if not confirmed and candidate.decimation != baseline.decimation:
        candidate = baseline
        score = aggregates[baseline.decimation]

    offset = 0.0
    evaluated = len(candidates) * len(windows)
    if not stable:
        reason = "unstable windows; conservative path"
    elif not confirmed:
        reason = "unconfirmed analog; conservative path"
    else:
        reason = "multi-window signal score"

    nudge = float(proposed_nudge_hz)
    probe_offset = bool(
        abs(nudge) >= 1.0
        and confirmed
        and (
            score.spectral_retention < OFFSET_PROBE_RETENTION
            or score.row_corr < OFFSET_PROBE_ROW_CORR
            or score.line_prominence_db < OFFSET_PROBE_PROMINENCE_DB
        )
    )
    if probe_offset:
        trial_scores: list[CandidateScore] = []
        offset_windows = windows if len(windows) == 1 else [windows[0], windows[-1]]
        for _, preview in offset_windows:
            occupied = _occupied_spectrum(preview, fs)
            trial_base, _, trial_fs = _demodulate(
                preview, fs, base_mix_hz + nudge, candidate, deviation_hz,
            )
            trial_score, _ = _candidate_score(
                trial_base, trial_fs, input_clip_frac=_clip_fraction(preview),
                spectral_retention=_spectral_retention(
                    occupied,
                    mix_hz=base_mix_hz + nudge,
                    cutoff_hz=candidate.cutoff_hz,
                ),
                width=width,
            )
            trial_scores.append(trial_score)
        evaluated += len(offset_windows)
        trial = _aggregate_scores(trial_scores)
        if trial.analog and offset_improves(
                score, trial, hysteresis=offset_hysteresis):
            score = trial
            offset = nudge
            reason = "offset improved signal score"
        else:
            reason = f"{reason}; zero/current offset won hysteresis"
    elif abs(nudge) >= 1.0 and confirmed:
        reason = f"{reason}; strong centered evidence skipped offset probe"

    selection = Selection(
        decimation=candidate.decimation,
        effective_bw_hz=candidate.effective_bw_hz,
        cutoff_hz=candidate.cutoff_hz,
        offset_hz=offset,
        score=score.score,
        line_rate_hz=score.line_rate_hz,
        line_prominence_db=score.line_prominence_db,
        row_corr=score.row_corr,
        analog=confirmed,
        reason=reason,
        generation=int(generation),
        candidates_evaluated=evaluated,
        confirmed=confirmed,
        stable=stable,
        windows_evaluated=len(windows),
        winner_votes=votes[candidate.decimation],
        spectral_retention=score.spectral_retention,
        line_confidence=score.line_confidence,
        sync_score=score.sync_score,
        vertical_detail=score.vertical_detail,
    )
    # Candidate scoring is preview-only, but the first decoder frame must use
    # the complete engine capture.  Returning the final 24 ms preview caused
    # 79/93/123-line startup rasters and seeded sticky geometry from a fragment.
    base, fm_base, fs_ch = _demodulate(
        np.asarray(iq, dtype=np.complex64),
        fs,
        base_mix_hz + offset,
        candidate,
        deviation_hz,
    )
    trusted_hint = bool(confirmed and stable and score.analog)
    hint = cvbs.LineRateHint(
        line_hz=score.line_rate_hz if trusted_hint else None,
        attempted=trusted_hint,
        confirmed=trusted_hint,
    )
    return Evaluation(selection, base, fm_base, fs_ch, hint, 0)


def demodulate_cached(
    iq: np.ndarray,
    sample_rate_hz: float,
    selection: Selection,
    *,
    base_mix_hz: float,
    deviation_hz: float,
    timings_ms: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidate = Candidate(
        selection.decimation,
        float(sample_rate_hz) / selection.decimation,
        selection.effective_bw_hz,
        selection.cutoff_hz,
    )
    started = time.perf_counter()
    channel, fs_ch = demod.channelize(
        iq,
        sample_rate_hz,
        base_mix_hz + selection.offset_hz,
        candidate.effective_bw_hz,
        decimation=candidate.decimation,
    )
    channelized = time.perf_counter()
    fm_base = demod.fm_demod(channel, fs_ch, deviation_hz=deviation_hz)
    demodulated = time.perf_counter()
    base = demod.deemphasis(fm_base, fs_ch)
    finished = time.perf_counter()
    if timings_ms is not None:
        timings_ms.update({
            "channelize": (channelized - started) * 1000.0,
            "demod": (demodulated - channelized) * 1000.0,
            "deemphasis": (finished - demodulated) * 1000.0,
        })
    return base, fm_base, fs_ch


class Lifecycle:
    """Cache and controlled re-evaluation state for one Engine LOCK."""

    def __init__(self, loss_frames: int = LOSS_FRAMES) -> None:
        self.loss_frames = max(2, int(loss_frames))
        self.selection: Selection | None = None
        self.unusable_frames = 0
        self.generation = 0
        self.reason = "first lock"
        self.evaluations = 0
        self.sample_rate_hz: float | None = None

    def reset(self, reason: str) -> None:
        self.selection = None
        self.unusable_frames = 0
        self.reason = str(reason)
        self.sample_rate_hz = None

    def needs_evaluation(self, sample_rate_hz: float) -> bool:
        fs = float(sample_rate_hz)
        if self.selection is None:
            return True
        if self.sample_rate_hz is None or abs(self.sample_rate_hz - fs) > 1.0:
            self.reset("sample-rate change")
            return True
        return False

    def adopt(self, selection: Selection, sample_rate_hz: float) -> Selection:
        self.generation += 1
        self.evaluations += 1
        self.sample_rate_hz = float(sample_rate_hz)
        trigger = self.reason
        reason = (
            selection.reason
            if trigger == "first lock"
            else f"{trigger}; {selection.reason}"
        )
        self.selection = Selection(
            **{**selection.diagnostics(), "generation": self.generation,
               "reason": reason}
        )
        self.reason = self.selection.reason
        self.unusable_frames = 0
        return self.selection

    def observe(self, usable: bool) -> bool:
        if usable:
            self.unusable_frames = 0
            return False
        self.unusable_frames += 1
        if self.unusable_frames < self.loss_frames:
            return False
        self.selection = None
        self.reason = f"{self.unusable_frames} consecutive unusable frames"
        self.unusable_frames = 0
        return True

    def diagnostics(self) -> dict[str, Any]:
        if self.selection is None:
            return {
                "enabled": True,
                "selected": False,
                "score": 0.0,
                "reason": self.reason,
                "generation": self.generation,
                "evaluations": self.evaluations,
                "unusable_frames": self.unusable_frames,
            }
        return {
            "enabled": True,
            "selected": True,
            **self.selection.diagnostics(),
            "evaluations": self.evaluations,
            "unusable_frames": self.unusable_frames,
        }
