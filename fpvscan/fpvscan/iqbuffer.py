"""Кільцевий буфер IQ-відліків для безперервного захоплення.

Нитка приймача (SdrSource.read у циклі) постійно дописує сюди свіжі
відліки. Нитка декодера (Engine._do_lock) бере знімки (snapshot)
незалежно від темпу запису — без блокування приймача на час обробки
кадру і без розриву потоку між викликами decode(), як було раніше
(кожен виклик = окреме retune_and_read блоками "ривками").

Абсолютний лічильник записаних відліків (`filled`) — це те, що дає
змогу cvbs.decode() узгоджувати фазу рядкової/кадрової синхри між
послідовними знімками, навіть якщо вони не йдуть впритул один за
одним.
"""
from __future__ import annotations
import threading
import time
from collections import deque
import numpy as np


class IQRingBuffer:
    """Потокобезпечний кільцевий буфер комплексних IQ-відліків."""

    def __init__(self, capacity: int, dtype=np.complex64):
        if capacity <= 0:
            raise ValueError("capacity має бути додатним")
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=dtype)
        self._pos = 0        # куди писати наступний відлік (індекс у _buf)
        self._filled = 0     # скільки всього відліків записано (монотонно)
        self._last_write_mono = 0.0
        self._gap_positions: deque[int] = deque(maxlen=32)
        self._gap_count = 0
        self._lock = threading.Lock()

    @property
    def filled(self) -> int:
        """Загальна кількість колись записаних відліків (абсолютна позиція
        потоку — саме її використовує cvbs.DecodeState для узгодження фази)."""
        with self._lock:
            return self._filled

    def write(self, chunk: np.ndarray, *, discontinuity: bool = False) -> None:
        """Append source samples, optionally marking a stream discontinuity.

        ``filled`` tracks the absolute source position even when an individual
        write is larger than the ring.  Readers can therefore distinguish a
        normal unread range from an overrun instead of gluing unrelated IQ.
        """
        original_n = len(chunk)
        if original_n == 0:
            return
        with self._lock:
            if discontinuity:
                self._gap_positions.append(self._filled)
                self._gap_count += 1
            if original_n >= self.capacity:
                # Preserve the absolute cursor and arrange the retained tail so
                # ``_pos`` still denotes the next logical source sample.
                chunk = chunk[-self.capacity:]
                n = self.capacity
                final_pos = (self._pos + original_n) % self.capacity
                first = self.capacity - final_pos
                self._buf[final_pos:] = chunk[:first]
                if first < n:
                    self._buf[:n - first] = chunk[first:]
                self._pos = final_pos
                self._filled += original_n
                self._last_write_mono = time.monotonic()
                return
            n = original_n
            end = self._pos + n
            if end <= self.capacity:
                self._buf[self._pos:end] = chunk
            else:
                first = self.capacity - self._pos
                self._buf[self._pos:] = chunk[:first]
                self._buf[:end - self.capacity] = chunk[first:]
            self._pos = end % self.capacity
            self._filled += n
            self._last_write_mono = time.monotonic()

    @property
    def gap_count(self) -> int:
        with self._lock:
            return self._gap_count

    @property
    def last_write_age_s(self) -> float:
        """Seconds since the reader last appended samples."""
        with self._lock:
            stamp = self._last_write_mono
        if stamp <= 0:
            return float("inf")
        return max(0.0, time.monotonic() - stamp)

    def snapshot_into(self, out: np.ndarray, n: int | None = None
                      ) -> tuple[np.ndarray, int]:
        """Copy newest samples into caller-owned contiguous storage.

        The destination is allocated before the ring mutex is acquired.  The
        mutex still covers the memory copy so the returned window is coherent,
        but allocation and all later processing happen after it is released.
        """
        dst = np.asarray(out)
        if dst.ndim != 1 or dst.dtype != self._buf.dtype:
            raise ValueError("snapshot buffer must be a 1-D ring-dtype array")
        if not dst.flags.c_contiguous or not dst.flags.writeable:
            raise ValueError("snapshot buffer must be writable and contiguous")
        wanted = dst.size if n is None else int(n)
        if wanted < 0 or wanted > dst.size:
            raise ValueError("snapshot sample count exceeds destination")

        with self._lock:
            avail = min(wanted, self._filled, self.capacity)
            abs_start = self._filled - avail
            view = dst[:avail]
            if avail == 0:
                return view, abs_start
            start = (self._pos - avail) % self.capacity
            if start + avail <= self.capacity:
                np.copyto(view, self._buf[start:start + avail])
            else:
                first = self.capacity - start
                np.copyto(view[:first], self._buf[start:])
                np.copyto(view[first:], self._buf[:avail - first])
            return view, abs_start

    def snapshot(self, n: int) -> tuple[np.ndarray, int]:
        """Останні `n` відліків суцільним масивом (копія) + абсолютна
        позиція першого з них у потоці.

        Якщо записано менше за `n`, повертає все, що є. Абсолютна
        позиція нехай і не збігається з попереднім знімком впритул —
        decode() рахує зсув сам, спираючись на різницю абсолютних
        позицій, а не на суміжність викликів.
        """
        wanted = int(n)
        if wanted < 0:
            raise ValueError("snapshot sample count must be non-negative")
        out = np.empty(min(wanted, self.capacity), dtype=self._buf.dtype)
        return self.snapshot_into(out)

    def read_since_into(
        self,
        out: np.ndarray,
        cursor: int | None,
        *,
        max_samples: int | None = None,
    ) -> tuple[np.ndarray, int, int, bool]:
        """Copy an unread absolute range into ``out`` exactly once.

        Returns ``(view, abs_start, next_cursor, gap)``.  ``gap`` is true when
        the requested cursor was overwritten or the source explicitly marked a
        discontinuity.  In that case reading resumes at the first contiguous
        retained sample after the newest known gap.
        """
        dst = np.asarray(out)
        if dst.ndim != 1 or dst.dtype != self._buf.dtype:
            raise ValueError("read buffer must be a 1-D ring-dtype array")
        if not dst.flags.c_contiguous or not dst.flags.writeable:
            raise ValueError("read buffer must be writable and contiguous")
        limit = dst.size if max_samples is None else min(dst.size, int(max_samples))
        if limit < 0:
            raise ValueError("max_samples must be non-negative")

        with self._lock:
            filled = self._filled
            retained = min(filled, self.capacity)
            oldest = filled - retained
            requested = oldest if cursor is None else int(cursor)
            gap = requested < oldest or requested > filled
            if requested > filled:
                requested = filled
            start = max(oldest, requested)
            for position in self._gap_positions:
                if start <= position < filled:
                    start = max(start, position)
                    gap = True
            available = min(max(0, filled - start), limit)
            end_abs = start + available
            view = dst[:available]
            if available:
                ring_start = (self._pos - (filled - start)) % self.capacity
                if ring_start + available <= self.capacity:
                    np.copyto(view, self._buf[ring_start:ring_start + available])
                else:
                    first = self.capacity - ring_start
                    np.copyto(view[:first], self._buf[ring_start:])
                    np.copyto(view[first:], self._buf[:available - first])
            while self._gap_positions and self._gap_positions[0] < oldest:
                self._gap_positions.popleft()
            return view, start, end_abs, gap
