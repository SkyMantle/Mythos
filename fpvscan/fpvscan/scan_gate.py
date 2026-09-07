"""Sweep inspect / auto-peek gates. Keep engine._do_sweep thin."""
from __future__ import annotations

from typing import Any

from fpvscan.bands import band_of

HIT_TOL_HZ = 14.0e6
FFT_MIN_BW_HZ = 2.5e6
INSPECT_MIN_BW_HZ = 5.0e6
INSPECT_BW_HZ = 12.0e6
INSPECT_BW_5G8_HZ = 18.0e6
INSPECT_MS = 40.0
INSPECT_MS_5G8 = 70.0
AUTO_PEEK_MIN_PIC = 0.12
SOFT_MAX_BW_HZ = 15.0e6


def fft_min_bw_hz(scan: dict[str, Any]) -> float:
    return float(scan.get("fft_min_bw_hz") or FFT_MIN_BW_HZ)


def in_5g8(freq_hz: float) -> bool:
    return band_of(freq_hz) == "5G8"


def inspect_bw_hz(scan: dict[str, Any], freq_hz: float) -> float:
    if in_5g8(freq_hz):
        return float(scan.get("inspect_bw_5g8_hz") or INSPECT_BW_5G8_HZ)
    return float(scan.get("inspect_bw_hz") or INSPECT_BW_HZ)


def inspect_ms(scan: dict[str, Any], freq_hz: float) -> float:
    if in_5g8(freq_hz):
        return float(scan.get("inspect_ms_5g8") or INSPECT_MS_5G8)
    return float(scan.get("inspect_ms") or INSPECT_MS)


def should_full_inspect(
    *,
    dwell_hz: float,
    center_hz: float,
    bandwidth_hz: float,
    snr_db: float,
    from_extra: bool,
    scan: dict[str, Any],
    line_hint: bool = False,
) -> bool:
    """Expensive inspect (retune + offsets + decode) only near a dwell or extra.

    Far coarse LTE blobs still queue cluster extras; they do not stall the pass.
    Narrow analog next to the dwell keeps the 0.6× min_bw path.
    Extras need a cheap 15.7 kHz comb hint so DJI/Wi-Fi skip decode.
    """
    hit_tol = float(scan.get("hit_tol_hz") or HIT_TOL_HZ)
    min_bw = float(scan.get("min_bw_hz") or INSPECT_MIN_BW_HZ)
    max_bw = float(scan.get("max_bw_hz") or 25e6)
    near = abs(float(center_hz) - float(dwell_hz)) <= hit_tol
    bw = float(bandwidth_hz)
    if not (min_bw <= bw <= max_bw):
        if not (near and bw >= 0.6 * min_bw and float(snr_db) >= 8):
            return False
    if from_extra:
        return bool(line_hint)
    return bool(near)


def inspect_soft_ok(
    *,
    standard: str,
    bandwidth_hz: float,
    prominence_db: float,
    confidence: float,
    harmonics: int,
    row_corr: float,
    pic_lines: int,
    scan: dict[str, Any],
    center_hz: float = 0.0,
) -> bool:
    """Spectral PAL/NTSC bypass when 40 ms did not assemble a frame.

    PAL/NTSC with line harmonics skip the 15 MHz cap and 0.90×inspect_bw
    5018-snow rule. Snow rasters (many lines, no row corr) still fail.
    """
    if standard not in ("PAL", "NTSC"):
        return False
    bw = float(bandwidth_hz)
    has_comb = int(harmonics) >= 1
    if has_comb:
        max_bw = float(scan.get("max_bw_hz") or 25e6)
        if not (5.0e6 <= bw <= max_bw):
            return False
    elif not (5.0e6 <= bw <= SOFT_MAX_BW_HZ):
        return False
    corr = float(row_corr)
    min_lines = int(scan.get("inspect_min_lines", 80))
    if int(pic_lines) >= min_lines and corr < 0.06:
        return False
    if not has_comb:
        cap = inspect_bw_hz(scan, center_hz)
        if bw >= 0.90 * cap and corr < 0.08:
            return False
    prom = float(scan.get("line_prominence_db", 10))
    if float(prominence_db) < prom:
        return False
    bypass = float(scan.get("inspect_conf_bypass", 0.70))
    if float(confidence) >= bypass:
        return True
    return has_comb


def auto_peek_allowed(det: Any, scan: dict[str, Any]) -> bool:
    """Steal sweep time only when a real picture (or lock) is already in hand."""
    if not scan.get("auto_peek", True):
        return False
    conf = float(getattr(det, "confidence", 0.0) or 0.0)
    if conf < float(scan.get("auto_peek_min_conf", 0.6)):
        return False
    pic = float(getattr(det, "pic_score", 0.0) or 0.0)
    locked = bool(getattr(det, "pic_locked", False))
    min_pic = float(scan.get("auto_peek_min_pic") or AUTO_PEEK_MIN_PIC)
    if pic < min_pic and not locked:
        return False
    return True
