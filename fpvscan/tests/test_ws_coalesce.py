from __future__ import annotations

from queue import Queue

from fpvscan.web.coalesce import coalesce_ws_events, enqueue_live_event


def test_coalesce_keeps_latest_frame_and_spectrum() -> None:
    events = [
        {"type": "spectrum", "data": {"bins": [1], "nfft": 8}},
        {"type": "frame", "data": {"img": b"old", "lines": 100}},
        {"type": "notice", "data": {"text": "ok"}},
        {"type": "spectrum", "data": {"bins": [2], "nfft": 8, "rate_hz": 10.0}},
        {"type": "frame", "data": {"img": b"new", "lines": 288}},
    ]
    out = coalesce_ws_events(events)
    kinds = [e["type"] for e in out]
    assert kinds == ["notice", "spectrum", "frame"]
    assert out[1]["data"]["bins"] == [2]
    assert out[1]["data"]["rate_hz"] == 10.0
    assert out[2]["data"]["img"] == b"new"


def test_enqueue_when_full_keeps_latest_spectrum() -> None:
    q: Queue = Queue(maxsize=2)
    enqueue_live_event(q, {"type": "frame", "data": {"img": b"a"}})
    enqueue_live_event(q, {"type": "frame", "data": {"img": b"b"}})
    enqueue_live_event(q, {"type": "spectrum", "data": {"bins": [1], "cursor_hz": 1e9}})
    enqueue_live_event(q, {"type": "frame", "data": {"img": b"c"}})
    enqueue_live_event(q, {"type": "spectrum", "data": {"bins": [9], "cursor_hz": 2e9}})
    dumped = []
    while not q.empty():
        dumped.append(q.get_nowait())
    kinds = [e["type"] for e in dumped]
    assert "spectrum" in kinds
    assert "frame" in kinds
    spec = next(e for e in dumped if e["type"] == "spectrum")
    frame = next(e for e in dumped if e["type"] == "frame")
    assert spec["data"]["bins"] == [9]
    assert spec["data"]["cursor_hz"] == 2e9
    assert frame["data"]["img"] == b"c"
