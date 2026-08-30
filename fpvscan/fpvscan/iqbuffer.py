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
        self._lock = threading.Lock()

    @property
    def filled(self) -> int:
        """Загальна кількість колись записаних відліків (абсолютна позиція
        потоку — саме її використовує cvbs.DecodeState для узгодження фази)."""
        with self._lock:
            return self._filled

    def write(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        if n == 0:
            return
        if n >= self.capacity:
            # шматок сам по собі більший за буфер — лишаємо тільки хвіст
            chunk = chunk[-self.capacity:]
            n = self.capacity
        with self._lock:
            end = self._pos + n
            if end <= self.capacity:
                self._buf[self._pos:end] = chunk
            else:
                first = self.capacity - self._pos
                self._buf[self._pos:] = chunk[:first]
                self._buf[:end - self.capacity] = chunk[first:]
            self._pos = end % self.capacity
            self._filled += n

    def snapshot(self, n: int) -> tuple[np.ndarray, int]:
        """Останні `n` відліків суцільним масивом (копія) + абсолютна
        позиція першого з них у потоці.

        Якщо записано менше за `n`, повертає все, що є. Абсолютна
        позиція нехай і не збігається з попереднім знімком впритул —
        decode() рахує зсув сам, спираючись на різницю абсолютних
        позицій, а не на суміжність викликів.
        """
        with self._lock:
            avail = min(n, self._filled, self.capacity)
            abs_start = self._filled - avail
            if avail == 0:
                return np.zeros(0, dtype=self._buf.dtype), abs_start
            start = (self._pos - avail) % self.capacity
            if start + avail <= self.capacity:
                out = self._buf[start:start + avail].copy()
            else:
                first = self.capacity - start
                out = np.concatenate((self._buf[start:], self._buf[:avail - first]))
            return out, abs_start
