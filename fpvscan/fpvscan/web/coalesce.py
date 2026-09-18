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
    """Put an event while only coalescing frame/spectrum traffic.

    State, notice, catalog, and detection events retain their insertion order.
    Normal Engine video uses a separate latest-only publisher, but frame
    handling remains here for compatibility with small standalone producers.
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
    kind = ev.get("type")
    live_kind = kind in ("spectrum", "frame")
    reliable = [
        item for item in dumped
        if item.get("type") not in ("spectrum", "frame")
    ]
    live = coalesce_ws_events([
        item for item in dumped
        if item.get("type") in ("spectrum", "frame")
    ] + ([ev] if live_kind else []))
    # Reliable events go back first and in their original order.  The incoming
    # reliable event is never displaced by high-rate live data.
    restore = reliable + ([] if live_kind else [ev]) + live
    for item in restore:
        try:
            q.put_nowait(item)
        except (Full, Exception):
            break
