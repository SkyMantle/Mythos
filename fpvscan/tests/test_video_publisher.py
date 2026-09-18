from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import numpy as np

from fpvscan.engine import Engine
from fpvscan.video_publisher import SnapshotRequest, VideoPublisher
from fpvscan.web.coalesce import enqueue_live_event


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("condition was not reached")
        threading.Event().wait(min(0.005, remaining))


def _frame(value: int = 0) -> np.ndarray:
    return np.full((24, 32), value, dtype=np.uint8)


def test_rapid_submits_keep_one_pending_and_emit_newest() -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def slow_encoder(luma, _fmt, _quality, _height, _method):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2)
        return bytes([int(luma[0, 0])])

    pub = VideoPublisher(encoder=slow_encoder)
    pub.set_clients(1)
    try:
        assert pub.submit(_frame(1), {"frame_seq": 1})
        assert entered.wait(2)
        for seq in range(2, 101):
            assert pub.submit(_frame(seq), {"frame_seq": seq})
            assert pub.metrics()["video_pending"] <= 1
        assert pub.metrics()["video_dropped"] >= 98
        release.set()
        _wait_for(lambda: pub.metrics()["emitted_frame_seq"] == 100)
        ready = pub.take_latest()
        assert ready is not None
        assert ready["frame_seq"] == 100
        assert pub.metrics()["video_pending"] == 0
    finally:
        release.set()
        pub.stop()


def test_slow_encoder_drops_without_blocking_or_growth() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_encoder(luma, _fmt, _quality, _height, _method):
        entered.set()
        assert release.wait(2)
        return bytes([int(luma[0, 0])])

    pub = VideoPublisher(encoder=slow_encoder)
    pub.set_clients(1)
    try:
        pub.submit(_frame(1), {"frame_seq": 1})
        assert entered.wait(2)
        submitted = threading.Event()

        def submit_many() -> None:
            for seq in range(2, 42):
                pub.submit(_frame(seq), {"frame_seq": seq})
            submitted.set()

        thread = threading.Thread(target=submit_many)
        thread.start()
        assert submitted.wait(1), "submits blocked behind the encoder"
        thread.join()
        metrics = pub.metrics()
        assert metrics["video_pending"] == 1
        assert metrics["video_dropped"] >= 39
    finally:
        release.set()
        pub.stop()


def test_reliable_events_keep_order_when_live_data_coalesces() -> None:
    events: Queue = Queue(maxsize=4)
    enqueue_live_event(events, {"type": "notice", "data": {"n": 1}})
    enqueue_live_event(events, {"type": "spectrum", "data": {"n": 1}})
    enqueue_live_event(events, {"type": "state", "data": {"n": 2}})
    enqueue_live_event(events, {"type": "frame", "data": {"n": 1}})
    enqueue_live_event(events, {"type": "catalog", "data": {"n": 3}})
    enqueue_live_event(events, {"type": "spectrum", "data": {"n": 2}})
    drained = []
    while True:
        try:
            drained.append(events.get_nowait())
        except Empty:
            break
    reliable = [
        item for item in drained
        if item["type"] not in ("frame", "spectrum")
    ]
    assert [item["type"] for item in reliable] == [
        "notice", "state", "catalog",
    ]
    assert [item["data"]["n"] for item in reliable] == [1, 2, 3]


def test_client_demand_starts_with_next_fresh_frame() -> None:
    encoded = threading.Event()
    calls = []

    def encoder(luma, _fmt, _quality, _height, _method):
        calls.append(int(luma[0, 0]))
        encoded.set()
        return b"image"

    pub = VideoPublisher(encoder=encoder)
    try:
        assert not pub.submit(_frame(1), {"frame_seq": 1})
        assert calls == []
        pub.set_clients(1)
        assert pub.take_latest() is None
        assert pub.submit(_frame(2), {"frame_seq": 2})
        assert encoded.wait(2)
        _wait_for(lambda: pub.metrics()["emitted_frame_seq"] == 2)
        assert calls == [2]
        assert pub.take_latest()["frame_seq"] == 2
    finally:
        pub.stop()


def test_recording_and_snapshot_work_without_ws_client(tmp_path) -> None:
    recorded = threading.Event()
    snapped = threading.Event()

    class Recorder:
        def push(self, luma):
            self.value = int(luma[0, 0])
            recorded.set()

    def encoder(luma, fmt, _quality, height, _method):
        assert fmt == "webp"
        assert height == 576
        return bytes([int(luma[0, 0])])

    path = tmp_path / "shot.webp"
    request = SnapshotRequest(path, "out/photos/shot.webp", "shot.webp")
    pub = VideoPublisher(
        encoder=encoder,
        snapshot_done=lambda _request, _size: snapped.set(),
    )
    recorder = Recorder()
    try:
        assert pub.submit(
            _frame(7), {"frame_seq": 7},
            recorder=recorder, snapshot=request,
        )
        assert recorded.wait(2)
        assert snapped.wait(2)
        assert path.read_bytes() == b"\x07"
        assert recorder.value == 7
        assert pub.metrics()["video_encoded"] == 0
        assert pub.take_latest() is None
    finally:
        pub.stop()


def test_encoder_exception_recovers_on_next_frame() -> None:
    failed = threading.Event()
    recovered = threading.Event()
    calls = 0

    def encoder(_luma, _fmt, _quality, _height, _method):
        nonlocal calls
        calls += 1
        if calls == 1:
            failed.set()
            raise RuntimeError("synthetic")
        recovered.set()
        return b"ok"

    pub = VideoPublisher(encoder=encoder)
    pub.set_clients(1)
    try:
        pub.submit(_frame(1), {"frame_seq": 1})
        assert failed.wait(2)
        _wait_for(lambda: pub.metrics()["video_encode_errors"] == 1)
        pub.submit(_frame(2), {"frame_seq": 2})
        assert recovered.wait(2)
        _wait_for(lambda: pub.metrics()["emitted_frame_seq"] == 2)
        assert pub.metrics()["video_encoded"] == 1
    finally:
        pub.stop()


def test_submitted_luma_is_not_mutated_by_owner() -> None:
    entered = threading.Event()
    release = threading.Event()
    observed = []

    def encoder(luma, _fmt, _quality, _height, _method):
        entered.set()
        assert release.wait(2)
        observed.append(np.array(luma, copy=True))
        return b"ok"

    pub = VideoPublisher(encoder=encoder)
    pub.set_clients(1)
    source = _frame(19)
    try:
        pub.submit(source, {"frame_seq": 1})
        assert entered.wait(2)
        source.fill(231)
        release.set()
        _wait_for(lambda: bool(observed))
        assert np.all(observed[0] == 19)
    finally:
        release.set()
        pub.stop()


class _Source:
    name = "test"
    sample_rate = 20e6
    overflows = 0
    clip_frac = 0.0
    adc_rms = 0.0
    bias_tee = False


def test_engine_metrics_distinguish_stages_and_reset() -> None:
    cfg = {
        "scan": {"sample_rate": 20e6},
        "video": {},
        "sdr": {"gain_db": 20, "auto_gain": False},
    }
    eng = Engine(_Source(), cfg, Queue())
    eng.set_video_clients(1)
    try:
        eng._frame_seq = 8
        eng._frame_mono = time.monotonic()
        eng._fps_ema = 12.0
        eng.video_publisher.submit(_frame(8), {"frame_seq": 8})
        _wait_for(lambda: eng.video_publisher.metrics()["emitted_frame_seq"] == 8)
        snap = eng.snapshot()
        assert snap["processed_fps"] == 12.0
        assert snap["submitted_frame_seq"] == 8
        assert snap["emitted_frame_seq"] == 8
        assert snap["frame_age_ms"] is not None
        assert snap["submitted_frame_age_ms"] is not None
        assert snap["emitted_frame_age_ms"] is not None
        eng._timings["lock_total"] = 25.0
        assert "encode" not in eng._timings
        assert eng.snapshot()["timings_ms"]["encode"] >= 0.0
        eng._reset_lock_metrics()
        reset = eng.snapshot()
        assert reset["processed_fps"] == 0.0
        assert reset["frame_seq"] == 0
        assert reset["submitted_frame_seq"] == 0
        assert reset["emitted_frame_seq"] == 0
        assert reset["frame_age_ms"] is None
        assert reset["submitted_frame_age_ms"] is None
        assert reset["emitted_frame_age_ms"] is None
    finally:
        eng.stop()
