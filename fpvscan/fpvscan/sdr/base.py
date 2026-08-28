"""Єдиний інтерфейс джерела IQ.

Робочий драйвер — BladeRF (fpvscan/sdr/bladerf.py), ctypes напряму
до libbladeRF. FileSource відтворює записаний ефір: це потрібно, щоб
міняти алгоритми, не тягаючись щоразу до стенду.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class SdrSource(ABC):
    """Джерело комплексних відліків.

    Контракт:
      * read(n) повертає рівно n відліків complex64, нормованих у ±1.0
      * зміна center_freq/sample_rate може відкидати частину буфера
      * джерело потокобезпечне тільки в межах одного потоку-власника
    """

    name = "base"
    # Джерело з фіксованою частотою (запис із файлу). Рушій тоді
    # не робить свіп: інакше та сама передача «знаходиться» на
    # кожному кроці й список знахідок забивається дублями.
    fixed_freq = False

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def read(self, n: int) -> np.ndarray: ...

    @abstractmethod
    def set_center_freq(self, hz: float) -> None: ...

    @abstractmethod
    def set_sample_rate(self, hz: float) -> None: ...

    @abstractmethod
    def set_gain(self, db: float) -> None: ...

    @property
    @abstractmethod
    def center_freq(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> float: ...

    # --- необов'язкове ---

    def retune_and_read(self, hz: float, n: int) -> np.ndarray:
        """Перебудова + читання. Реалізації з quick-tune перевизначають це."""
        self.set_center_freq(hz)
        self.read(n // 4)  # скидаємо перехідний процес PLL
        return self.read(n)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
