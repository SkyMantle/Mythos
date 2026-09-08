"""Published detection list filters. Raw engine peaks stay in memory."""
from __future__ import annotations

from typing import Any, Iterable

HIT_FILTERS = ("all", "hide_weak", "hide_no_video", "hide_near_dup")
HIT_FILTER_DEFAULT = "hide_weak"
NEAR_DUP_HZ = 8.0e6
WEAK_REL_DB = 12.0
WEAK_ABS_DB = 8.0
NO_VIDEO_SCORE = 0.12
KEEP_PIC_SCORE = 0.20
INSPECT_MIN_ROW_CORR = 0.12
MIN_RASTER_LINES = 80
VIDEO_STANDARDS = frozenset({"PAL", "NTSC"})
FRAME_FRESH_S = 2.5
PRUNE_SETTLE_S = 5.0
HIT_SELECT_HZ = 2.0e6
# While LOCKED: digital AFC residual vs inspect peak is the same analog bird.
# AFC cap is ~1.5 MHz; this matches the UI “current” highlight. Not 10 MHz
# FPV channel spacing. Never use this to ignore ±0.1 force-tune.
MERGE_LOCK_HZ = HIT_SELECT_HZ
# Unlocked inspect identity: Hz–kHz peak interp. 100 kHz operator step is real.
MERGE_CHANNEL_HZ = 100e3
HIT_KEY_HZ = 50e3
# Unforced republish / list-echo only. ±0.1 MHz = 100 kHz must retune.
LOCK_SPURIOUS_HZ = 20e3
LOCK_HOLD_HZ = LOCK_SPURIOUS_HZ


def hit_key(freq_hz: float) -> int:
    """Stable list/dict identity: 50 kHz bins, not 1 Hz or MHz truncation."""
    return int(round(float(freq_hz) / HIT_KEY_HZ))


def same_channel(
    a_hz: float,
    b_hz: float,
    tol_hz: float = MERGE_CHANNEL_HZ,
) -> bool:
    return abs(float(a_hz) - float(b_hz)) <= float(tol_hz)


def lock_retune(
    want_hz: float,
    lock_target: float | None,
    *,
    force: bool = False,
    locked: bool = False,
) -> tuple[float, bool]:
    """Return (target_hz, skip).

    MERGE_LOCK_HZ is list identity while locked, not a tune no-op.
    Only unforced Hz/kHz republish is skipped. force=True (nudge /
    popover / auto-relock) always retunes to want_hz.
    """
    want = float(want_hz)
    if force:
        return want, False
    if (
        locked
        and lock_target is not None
        and same_channel(lock_target, want, LOCK_SPURIOUS_HZ)
    ):
        return float(lock_target), True
    return want, False


def sticky_published_hz(
    old_hz: float,
    new_hz: float,
    *,
    lock_target: float | None = None,
    locked: bool = False,
) -> float:
    """Keep one published center per bird; while locked, show the lock target."""
    old = float(old_hz)
    new = float(new_hz)
    if locked and lock_target is not None:
        lock = float(lock_target)
        if same_channel(old, lock, MERGE_LOCK_HZ) and same_channel(new, lock, MERGE_LOCK_HZ):
            return lock
    if same_channel(old, new):
        return old
    return new


def filter_published(
    dets: Iterable[dict[str, Any]],
    mode: str | None = HIT_FILTER_DEFAULT,
) -> list[dict[str, Any]]:
    items = [dict(d) for d in dets if d.get("freq_hz") is not None]
    kind = str(mode or HIT_FILTER_DEFAULT)
    if kind == "all" or not items:
        return items
    if kind == "hide_weak":
        return _hide_weak(items)
    if kind == "hide_no_video":
        return [d for d in items if _has_video(d)]
    if kind == "hide_near_dup":
        return _hide_near_dup(items)
    return items


def keeps(det: dict[str, Any], pool: Iterable[dict[str, Any]], mode: str | None) -> bool:
    shown = filter_published(list(pool), mode)
    freq = float(det.get("freq_hz") or 0)
    return any(same_channel(float(d["freq_hz"]), freq) for d in shown)


def _snr(d: dict[str, Any]) -> float:
    return float(d.get("snr_db") or 0.0)


def _pic(d: dict[str, Any]) -> float:
    return float(d.get("pic_score") or 0.0)


def _has_video(d: dict[str, Any]) -> bool:
    if d.get("pic_locked"):
        return True
    return _pic(d) >= NO_VIDEO_SCORE


def analog_standard(std: Any) -> bool:
    return str(std or "").strip().upper() in VIDEO_STANDARDS


def _row_corr(d: dict[str, Any]) -> float:
    return float(d.get("row_corr") or 0.0)


def _pic_lines(d: dict[str, Any]) -> int:
    return int(d.get("pic_lines") or d.get("lines") or 0)


def video_confirmed(d: dict[str, Any]) -> bool:
    """Analog video, not SNR. pic_locked or inspect-level row/score."""
    if not analog_standard(d.get("standard")):
        return False
    if d.get("pic_locked"):
        return True
    return _row_corr(d) >= INSPECT_MIN_ROW_CORR or _pic(d) >= KEEP_PIC_SCORE


def frames_flowing(
    d: dict[str, Any] | None = None,
    *,
    streaming: bool = False,
    now: float | None = None,
    frame_age_s: float | None = None,
    fresh_s: float = FRAME_FRESH_S,
) -> bool:
    if streaming:
        return True
    if frame_age_s is not None:
        return 0.0 <= float(frame_age_s) <= float(fresh_s)
    last = (d or {}).get("last_picture_at")
    if last and now is not None:
        return (float(now) - float(last)) <= float(fresh_s)
    return False


def is_video_green(
    d: dict[str, Any],
    *,
    streaming: bool = False,
    now: float | None = None,
    frame_age_s: float | None = None,
    fresh_s: float = FRAME_FRESH_S,
) -> bool:
    """Grid/list green: video-confirmed AND JPEG/frames are actually flowing."""
    return video_confirmed(d) and frames_flowing(
        d, streaming=streaming, now=now, frame_age_s=frame_age_s, fresh_s=fresh_s,
    )


def has_raster(d: dict[str, Any]) -> bool:
    return _pic_lines(d) >= MIN_RASTER_LINES and _row_corr(d) >= INSPECT_MIN_ROW_CORR


def empty_lock_picture(d: dict[str, Any]) -> bool:
    """No usable picture after operator lock: unlocked, weak score, no raster."""
    if d.get("pic_locked"):
        return False
    if _pic(d) >= KEEP_PIC_SCORE:
        return False
    if has_raster(d):
        return False
    return True


def should_prune_empty_lock(
    *,
    operator_selected: bool,
    auto_lock: bool,
    elapsed_s: float,
    sample: dict[str, Any],
    settle_s: float = PRUNE_SETTLE_S,
) -> bool:
    """Drop only the hit the operator locked, after auto-gain has had time."""
    if not operator_selected or auto_lock:
        return False
    if float(elapsed_s) < float(settle_s):
        return False
    return empty_lock_picture(sample)


def _hide_weak(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for d in items:
        groups.setdefault(str(d.get("band") or "—"), []).append(d)
    keep: list[dict[str, Any]] = []
    for group in groups.values():
        peak = max(_snr(d) for d in group)
        for d in group:
            if d.get("pic_locked") or _pic(d) >= KEEP_PIC_SCORE:
                keep.append(d)
                continue
            if _snr(d) >= peak - WEAK_REL_DB:
                keep.append(d)
                continue
            if len(group) == 1 and _snr(d) >= WEAK_ABS_DB:
                keep.append(d)
    return keep


def _hide_near_dup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        items,
        key=lambda d: (_pic(d), _snr(d), float(d.get("confidence") or 0.0)),
        reverse=True,
    )
    keep: list[dict[str, Any]] = []
    for d in ranked:
        freq = float(d["freq_hz"])
        if any(abs(freq - float(k["freq_hz"])) <= NEAR_DUP_HZ for k in keep):
            continue
        keep.append(d)
    return keep
