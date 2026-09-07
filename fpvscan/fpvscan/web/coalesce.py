"""Drop-to-latest for live WS events so the console never plays a backlog."""
from __future__ import annotations

from queue import Empty, Full
from typing import Any


def coalesce_ws_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep notices/detections in order; keep only the newest frame and spectrum."""
    if len(events) <= 1:
        return events
    latest_frame: dict[str, Any] | None = None
    latest_spectrum: dict[str, Any] | None = None
    others: list[dict[str, Any]] = []
    for ev in events:
        kind = ev.get("type")
        if kind == "frame":
            latest_frame = ev
        elif kind == "spectrum":
            latest_spectrum = ev
        else:
            others.append(ev)
    if latest_spectrum is not None:
        others.append(latest_spectrum)
    if latest_frame is not None:
        others.append(latest_frame)
    return others


def enqueue_live_event(q, ev: dict[str, Any]) -> None:
    """Put an event; if the queue is full, keep the latest frame and spectrum.

    Frames used to evict old items; spectrum was discarded. During LOCK that
    muted FFT updates whenever video filled the queue (record makes it worse).
    """
    try:
        q.put_nowait(ev)
        return
    except (Full, Exception):
        pass
    dumped: list[dict[str, Any]] = []
    try:
        while True:
            dumped.append(q.get_nowait())
    except Empty:
        pass
    dumped.append(ev)
    coalesced = coalesce_ws_events(dumped)
    live = [e for e in coalesced if e.get("type") in ("spectrum", "frame")]
    others = [e for e in coalesced if e.get("type") not in ("spectrum", "frame")]
    for item in live + others[-8:]:
        try:
            q.put_nowait(item)
        except (Full, Exception):
            break
