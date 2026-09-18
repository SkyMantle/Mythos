"""Bounded latest-only handoff from LOCK DSP to WebSocket publication."""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from .dsp import cvbs


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotRequest:
    path: Path
    ref: str
    name: str


@dataclass(frozen=True)
class _Task:
    buffer_index: int
    metadata: Mapping[str, Any]
    generation: int
    preview: bool
    recorder: Any | None
    snapshot: SnapshotRequest | None
    fmt: str
    quality: int
    method: int


class VideoPublisher:
    """Encode the newest requested frame without building a frame backlog.

    The DSP thread copies into one of two owned arrays.  One array may be in
    the encoder while the other is the sole replaceable pending frame.
    """

    def __init__(
        self,
        *,
        encoder: Callable[[np.ndarray, str, int, int | None, int], bytes] | None = None,
        snapshot_done: Callable[[SnapshotRequest, int], None] | None = None,
        snapshot_error: Callable[[SnapshotRequest, Exception], None] | None = None,
    ):
        self._encoder = encoder or self._encode_luma
        self._snapshot_done = snapshot_done
        self._snapshot_error = snapshot_error
        self._condition = threading.Condition(threading.RLock())
        self._buffers: list[np.ndarray | None] = [None, None]
        self._pending: _Task | None = None
        self._active_index: int | None = None
        self._ready: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self._stop = False
        self._generation = 0
        self._clients = 0
        self._dropped = 0
        self._encoded = 0
        self._encode_errors = 0
        self._submitted_seq = 0
        self._emitted_seq = 0
        self._submitted_mono = 0.0
        self._emitted_mono = 0.0
        self._submitted_tick: float | None = None
        self._emitted_tick: float | None = None
        self._submitted_fps = 0.0
        self._emitted_fps = 0.0
        self._encode_ms = 0.0

    @staticmethod
    def _encode_luma(
        luma: np.ndarray,
        fmt: str,
        quality: int,
        height: int | None,
        method: int,
    ) -> bytes:
        frame = cvbs.Frame(
            luma=luma,
            line_rate=0.0,
            lines=int(luma.shape[0]),
            standard="?",
            locked=False,
        )
        return cvbs.encode(
            frame, fmt, quality, height=height, method=method,
        )

    @staticmethod
    def _tick(
        now: float, previous: float | None, ema: float,
    ) -> tuple[float, float]:
        if previous is None or now <= previous:
            return now, ema
        instant = 1.0 / (now - previous)
        return now, instant if ema == 0.0 else ema * 0.8 + instant * 0.2

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._run, name="fpv-video-publisher", daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.error("video publisher did not stop within %.1f s", timeout)

    def reset(self) -> None:
        """Retire pending/output frames and zero per-mode publication metrics."""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._ready = None
            self._dropped = 0
            self._encoded = 0
            self._encode_errors = 0
            self._submitted_seq = 0
            self._emitted_seq = 0
            self._submitted_mono = 0.0
            self._emitted_mono = 0.0
            self._submitted_tick = None
            self._emitted_tick = None
            self._submitted_fps = 0.0
            self._emitted_fps = 0.0
            self._encode_ms = 0.0

    def set_clients(self, count: int) -> None:
        with self._condition:
            old = self._clients
            self._clients = max(0, int(count))
            # A 0 -> 1 client transition must wait for the next fresh submit.
            if old == 0 and self._clients > 0:
                self._generation += 1
                self._ready = None

    @property
    def clients(self) -> int:
        with self._condition:
            return self._clients

    def submit(
        self,
        luma: np.ndarray,
        metadata: Mapping[str, Any],
        *,
        recorder: Any | None = None,
        snapshot: SnapshotRequest | None = None,
        fmt: str = "jpeg",
        quality: int = 75,
        method: int = 0,
    ) -> bool:
        """Copy one frame into the replaceable pending slot.

        Returns false when no WebSocket, recorder, or snapshot needs the frame.
        """
        array = np.asarray(luma)
        if array.ndim != 2:
            raise ValueError("video luma must be a two-dimensional array")
        with self._condition:
            preview = self._clients > 0
            if not preview and recorder is None and snapshot is None:
                return False
            self.start()
            pending = self._pending
            if pending is not None:
                # A requested still owns its exact accepted luma until written;
                # drop this preview submit instead of silently changing the
                # POST /api/snapshot result underneath the operator.
                if pending.snapshot is not None:
                    self._dropped += 1
                    return False
                index = pending.buffer_index
                self._dropped += 1
            else:
                index = 0 if self._active_index != 0 else 1
            buf = self._buffers[index]
            if (
                buf is None
                or buf.shape != array.shape
                or buf.dtype != np.uint8
            ):
                buf = np.empty(array.shape, dtype=np.uint8)
                self._buffers[index] = buf
            np.copyto(buf, array, casting="unsafe")
            seq = int(metadata.get("frame_seq", 0))
            now = time.perf_counter()
            self._submitted_tick, self._submitted_fps = self._tick(
                now, self._submitted_tick, self._submitted_fps,
            )
            self._submitted_seq = max(self._submitted_seq, seq)
            self._submitted_mono = time.monotonic()
            self._pending = _Task(
                buffer_index=index,
                metadata=MappingProxyType(dict(metadata)),
                generation=self._generation,
                preview=preview,
                recorder=recorder,
                snapshot=snapshot,
                fmt=str(fmt),
                quality=int(quality),
                method=int(method),
            )
            self._condition.notify()
            return True

    def take_latest(self) -> dict[str, Any] | None:
        """Return and clear the latest encoded WS-ready frame."""
        with self._condition:
            ready = self._ready
            self._ready = None
            return ready

    def note_emitted(self, frame_seq: int) -> None:
        """Compatibility hook for externally produced frame payloads."""
        seq = int(frame_seq)
        with self._condition:
            if seq <= self._emitted_seq:
                return
            now = time.perf_counter()
            self._emitted_tick, self._emitted_fps = self._tick(
                now, self._emitted_tick, self._emitted_fps,
            )
            self._emitted_seq = seq
            self._emitted_mono = time.monotonic()

    def metrics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            submitted_age = (
                None if self._submitted_mono <= 0.0
                else max(0.0, (now - self._submitted_mono) * 1000.0)
            )
            emitted_age = (
                None if self._emitted_mono <= 0.0
                else max(0.0, (now - self._emitted_mono) * 1000.0)
            )
            return {
                "submitted_fps": round(self._submitted_fps, 2),
                "emitted_fps": round(self._emitted_fps, 2),
                "submitted_frame_seq": int(self._submitted_seq),
                "emitted_frame_seq": int(self._emitted_seq),
                "submitted_frame_age_ms": (
                    None if submitted_age is None else round(submitted_age, 1)
                ),
                "emitted_frame_age_ms": (
                    None if emitted_age is None else round(emitted_age, 1)
                ),
                "video_pending": int(self._pending is not None),
                "video_dropped": int(self._dropped),
                "video_encoded": int(self._encoded),
                "video_encode_errors": int(self._encode_errors),
                "video_clients": int(self._clients),
                "video_encode_ms": round(self._encode_ms, 1),
            }

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop:
                    self._condition.wait()
                if self._pending is None and self._stop:
                    return
                task = self._pending
                self._pending = None
                self._active_index = task.buffer_index
                luma = self._buffers[task.buffer_index]
            if luma is None:
                continue
            try:
                if task.recorder is not None:
                    try:
                        task.recorder.push(luma)
                    except Exception:
                        log.exception("video recorder handoff failed")
                if task.snapshot is not None:
                    self._write_snapshot(luma, task)
                with self._condition:
                    preview = task.preview and self._clients > 0
                if preview:
                    self._publish_preview(luma, task)
            except Exception:
                log.exception("unexpected video publisher failure")
            finally:
                with self._condition:
                    self._active_index = None

    def _write_snapshot(self, luma: np.ndarray, task: _Task) -> None:
        request = task.snapshot
        if request is None:
            return
        try:
            raw = self._encoder(luma, "webp", 90, 576, 1)
            request.path.write_bytes(raw)
            if self._snapshot_done is not None:
                self._snapshot_done(request, len(raw))
        except Exception as exc:
            with self._condition:
                self._encode_errors += 1
            log.exception("snapshot encoding failed path=%s", request.path)
            if self._snapshot_error is not None:
                self._snapshot_error(request, exc)

    def _publish_preview(self, luma: np.ndarray, task: _Task) -> None:
        started = time.perf_counter()
        try:
            raw = self._encoder(
                luma, task.fmt, task.quality, None, task.method,
            )
            payload = dict(task.metadata)
            payload["img"] = base64.b64encode(raw).decode("ascii")
            event = {"type": "frame", "data": payload}
            ws_json = json.dumps(event, separators=(",", ":"), default=str)
        except Exception:
            with self._condition:
                self._encode_errors += 1
            log.exception(
                "preview encoding failed frame_seq=%s",
                task.metadata.get("frame_seq"),
            )
            return
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        seq = int(task.metadata.get("frame_seq", 0))
        with self._condition:
            if task.generation != self._generation:
                return
            self._encode_ms = (
                elapsed_ms if self._encode_ms == 0.0
                else self._encode_ms * 0.8 + elapsed_ms * 0.2
            )
            self._encoded += 1
            if self._clients <= 0:
                return
            pending_seq = (
                int(self._pending.metadata.get("frame_seq", 0))
                if self._pending is not None else 0
            )
            if pending_seq > seq:
                self._dropped += 1
                return
            if self._ready is not None:
                self._dropped += 1
            now = time.perf_counter()
            self._emitted_tick, self._emitted_fps = self._tick(
                now, self._emitted_tick, self._emitted_fps,
            )
            self._emitted_seq = max(self._emitted_seq, seq)
            self._emitted_mono = time.monotonic()
            self._ready = {
                "frame_seq": seq,
                "data": payload,
                "json": ws_json,
            }
