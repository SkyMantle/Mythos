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
    return any(abs(float(d["freq_hz"]) - freq) < 1.0 for d in shown)


def _snr(d: dict[str, Any]) -> float:
    return float(d.get("snr_db") or 0.0)


def _pic(d: dict[str, Any]) -> float:
    return float(d.get("pic_score") or 0.0)


def _has_video(d: dict[str, Any]) -> bool:
    if d.get("pic_locked"):
        return True
    return _pic(d) >= NO_VIDEO_SCORE


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
