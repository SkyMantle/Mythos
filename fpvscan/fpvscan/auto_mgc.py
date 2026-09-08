"""Software MGC for LOCK: step catalog ``sdr.gain_db``, not BladeRF AGC.

Bias-T (4.5 V) is a separate on/off. When it is already on, the LNA and
the existing ``_apply_bias_tee_gain`` offset already change ADC level —
this loop must not add a fixed +15 dB and must never toggle Bias-T.

Hill-climb: after a step up, wait ``SETTLE_S``. If pic_score / lock
quality does not improve, revert that step and mark a ceiling. Frozen
(near-identical) luma with a weak picture steps down, not up. A locked
usable picture is not brightened. Operator catalog writes of ``gain_db``
turn auto off; they click авто to resume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

MIN_DB = 0.0
MAX_DB = 60.0
STEP_DB = 3.0
INTERVAL_S = 0.40
HOLD_S = 1.5
SETTLE_S = 0.70
CLIP_FRAC = 0.002
RMS_THRESH = 0.40
SAT_FRAC = 0.12
FREEZE_CORR = 0.985
FREEZE_DOWN_DB = 36.0
IMPROVE_EPS = 0.02
WEAK_SCORE = 0.20
HOLD_SCORE = 0.35
USABLE_SCORE = 0.18
MIN_LINES = 80
MIN_CORR = 0.08
SIG_N = 12
IQ_RMS_N = 4096


@dataclass(frozen=True)
class MgcSample:
    gain_db: float
    pic_locked: bool
    pic_score: float
    pic_lines: int
    row_corr: float
    clip_frac: float
    enabled: bool = True
    operator_hold: bool = False
    now_s: float = 0.0
    adc_rms: float = 0.0
    sat_frac: float = 0.0
    frame_sig: tuple[float, ...] = ()
    min_db: float = MIN_DB
    max_db: float = MAX_DB
    step_db: float = STEP_DB
    clip_thresh: float = CLIP_FRAC
    rms_thresh: float = RMS_THRESH
    sat_thresh: float = SAT_FRAC
    freeze_corr: float = FREEZE_CORR
    freeze_down_db: float = FREEZE_DOWN_DB
    settle_s: float = SETTLE_S
    improve_eps: float = IMPROVE_EPS
    weak_score: float = WEAK_SCORE
    hold_score: float = HOLD_SCORE
    usable_score: float = USABLE_SCORE
    min_lines: int = MIN_LINES
    min_corr: float = MIN_CORR


@dataclass
class MgcState:
    """Probe / ceiling / previous luma signature across LOCK MGC ticks."""

    probe_from_db: float | None = None
    probe_to_db: float | None = None
    probe_at_s: float = 0.0
    probe_score: float = 0.0
    probe_locked: bool = False
    probe_quality: float = 0.0
    ceiling_db: float | None = None
    last_sig: tuple[float, ...] = field(default_factory=tuple)


def due(now_s: float, last_eval_s: float, interval_s: float = INTERVAL_S) -> bool:
    return (float(now_s) - float(last_eval_s)) >= float(interval_s)


def has_raster(sample: MgcSample) -> bool:
    return (
        int(sample.pic_lines) >= int(sample.min_lines)
        and float(sample.row_corr) >= float(sample.min_corr)
    )


def picture_is_good(sample: MgcSample) -> bool:
    """Dead zone: locked raster with a usable score — do not pump gain."""
    return (
        bool(sample.pic_locked)
        and float(sample.pic_score) >= float(sample.hold_score)
        and has_raster(sample)
    )


def picture_is_usable(sample: MgcSample) -> bool:
    """Locked with a watchable score — do not raise 'to make it brighter'."""
    return bool(sample.pic_locked) and float(sample.pic_score) >= float(sample.usable_score)


def picture_is_weak(sample: MgcSample) -> bool:
    """Unlocked and (low score or no raster). Snow-lock does not count as weak."""
    if sample.pic_locked:
        return False
    return float(sample.pic_score) < float(sample.weak_score) or not has_raster(sample)


def lock_quality(sample: MgcSample) -> float:
    """Scalar used to judge whether an up-step helped."""
    q = float(sample.pic_score)
    if sample.pic_locked:
        q += 0.12
    if has_raster(sample):
        q += 0.05
    return q


def luma_signature(luma: Any, n: int = SIG_N) -> tuple[float, ...]:
    """Cheap spatial fingerprint of a luma field for freeze detection."""
    if luma is None:
        return ()
    arr = np.asarray(luma)
    if arr.ndim != 2 or arr.size == 0:
        return ()
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if h < 2 or w < 2:
        return ()
    ys = np.linspace(0, h - 1, int(n)).astype(np.int32)
    xs = np.linspace(0, w - 1, int(n)).astype(np.int32)
    grid = np.asarray(arr, dtype=np.float32)[ys[:, None], xs[None, :]]
    return tuple(float(v) for v in grid.ravel())


def luma_sat_frac(luma: Any) -> float:
    """Fraction of pixels crushed to black or white (overload proxy)."""
    if luma is None:
        return 0.0
    arr = np.asarray(luma)
    if arr.size == 0:
        return 0.0
    x = arr.reshape(-1)
    return float(np.mean((x <= 3) | (x >= 252)))


def iq_rms(iq: Any, n: int = IQ_RMS_N) -> float:
    """Linear RMS of a complex IQ slice (1.0 ≈ full scale)."""
    if iq is None:
        return 0.0
    arr = np.asarray(iq)
    if arr.size == 0:
        return 0.0
    sl = arr.reshape(-1)[: int(n)]
    return float(np.sqrt(np.mean(np.real(sl) ** 2 + np.imag(sl) ** 2)))


def sig_corr(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Pearson correlation of two luma signatures. Flat+flat → 1.0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    va = 0.0
    vb = 0.0
    cov = 0.0
    for x, y in zip(a, b):
        dx = x - ma
        dy = y - mb
        va += dx * dx
        vb += dy * dy
        cov += dx * dy
    if va < 1e-6 and vb < 1e-6:
        return 1.0
    if va < 1e-6 or vb < 1e-6:
        return 0.0
    return float(cov / (va * vb) ** 0.5)


def adc_hot(sample: MgcSample) -> bool:
    """True when clip, IQ RMS, or luma crush say the front-end is hot."""
    if float(sample.clip_frac) >= float(sample.clip_thresh):
        return True
    if float(sample.adc_rms) >= float(sample.rms_thresh):
        return True
    if float(sample.sat_frac) >= float(sample.sat_thresh):
        return True
    return False


def picture_frozen(sample: MgcSample, state: MgcState) -> bool:
    """Near-identical successive fields with a weak/unlocked picture."""
    if picture_is_good(sample) or picture_is_usable(sample):
        return False
    corr = sig_corr(sample.frame_sig, state.last_sig)
    return corr >= float(sample.freeze_corr)


def _clear_probe(state: MgcState) -> None:
    state.probe_from_db = None
    state.probe_to_db = None
    state.probe_at_s = 0.0
    state.probe_score = 0.0
    state.probe_locked = False
    state.probe_quality = 0.0


def _improved(sample: MgcSample, state: MgcState) -> bool:
    if bool(sample.pic_locked) and not bool(state.probe_locked):
        return True
    return lock_quality(sample) >= float(state.probe_quality) + float(sample.improve_eps)


def _remember_sig(state: MgcState, sample: MgcSample) -> None:
    if sample.frame_sig:
        state.last_sig = sample.frame_sig


def _start_probe(state: MgcState, sample: MgcSample, from_db: float, to_db: float) -> None:
    state.probe_from_db = float(from_db)
    state.probe_to_db = float(to_db)
    state.probe_at_s = float(sample.now_s)
    state.probe_score = float(sample.pic_score)
    state.probe_locked = bool(sample.pic_locked)
    state.probe_quality = lock_quality(sample)


def step(sample: MgcSample, state: MgcState) -> float:
    """One MGC decision. Mutates ``state`` (probe, ceiling, freeze signature).

    Returns:
        Catalog ``gain_db`` after at most one step.
    """
    lo = float(sample.min_db)
    hi = float(sample.max_db)
    gain = max(lo, min(hi, float(sample.gain_db)))
    step_db = abs(float(sample.step_db)) or STEP_DB
    if not sample.enabled or sample.operator_hold:
        _clear_probe(state)
        _remember_sig(state, sample)
        return gain

    frozen = picture_frozen(sample, state)
    if adc_hot(sample):
        nxt = max(lo, gain - step_db)
        _clear_probe(state)
        state.ceiling_db = nxt
        _remember_sig(state, sample)
        return nxt
    if frozen:
        _clear_probe(state)
        if gain >= float(sample.freeze_down_db):
            nxt = max(lo, gain - step_db)
            state.ceiling_db = nxt
            _remember_sig(state, sample)
            return nxt
        cap = gain if state.ceiling_db is None else min(float(state.ceiling_db), gain)
        state.ceiling_db = cap
        _remember_sig(state, sample)
        return gain

    if state.probe_to_db is not None:
        elapsed = float(sample.now_s) - float(state.probe_at_s)
        if elapsed < float(sample.settle_s):
            _remember_sig(state, sample)
            return gain
        from_db = float(state.probe_from_db if state.probe_from_db is not None else gain)
        improved = _improved(sample, state)
        _clear_probe(state)
        if not improved:
            state.ceiling_db = from_db
            _remember_sig(state, sample)
            return max(lo, min(hi, from_db))
        _remember_sig(state, sample)
        return gain

    if picture_is_good(sample) or picture_is_usable(sample) or sample.pic_locked:
        _remember_sig(state, sample)
        return gain

    if picture_is_weak(sample):
        cap = hi if state.ceiling_db is None else min(hi, float(state.ceiling_db))
        if gain >= cap - 1e-9:
            _remember_sig(state, sample)
            return gain
        nxt = min(cap, gain + step_db)
        if nxt > gain + 1e-9:
            _start_probe(state, sample, gain, nxt)
        _remember_sig(state, sample)
        return nxt

    _remember_sig(state, sample)
    return gain


def next_gain_db(sample: MgcSample, state: MgcState | None = None) -> float:
    """Return catalog gain after at most one step. Clip / freeze win over weak."""
    return step(sample, state if state is not None else MgcState())
