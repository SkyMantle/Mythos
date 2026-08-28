#!/usr/bin/env python3
"""Замір реальної швидкості перебудови на твоїй платі.

Від цього залежить час повного проходу 400 МГц – 6 ГГц, і зміряти це
можна тільки на залізі: цифри в документації стосуються синтезатора,
а не всього ланцюжка USB -> прошивка -> RFIC.

    python scripts/bench_retune.py

Виводить час одного кроку звичайною перебудовою і профілем швидкої,
та перерахунок у час повного проходу. Якщо quick tune на цій збірці
libbladeRF не працює — так і скаже, без вигадок.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr.bladerf import BladeRF, BladeRFError


def bench(fn, freqs, reps=3):
    ts = []
    for _ in range(reps):
        for f in freqs:
            t = time.perf_counter()
            fn(f)
            ts.append((time.perf_counter() - t) * 1e6)   # мкс
    return statistics.median(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--rate", type=float, default=40.0, help="Мвідл/с")
    ap.add_argument("--device", default="")
    ap.add_argument("--lib", help="точний шлях до bladeRF.dll / libbladeRF.so")
    a = ap.parse_args()

    fs = a.rate * 1e6
    step = fs * 0.8
    freqs = [int(f) for f in
             __import__("numpy").arange(400e6 + fs / 2, 6000e6, step)]
    print(f"кроків у повному проході: {len(freqs)} "
          f"(смуга {fs/1e6:.0f} МГц, крок {step/1e6:.0f} МГц)")

    src = BladeRF(device=a.device, lib_path=a.lib, use_meta=True)
    src.open()
    i = src.info()
    print(f"плата: {i['board']}  sn={i['serial'][:8]}  FPGA {i['fpga_kle']}kLE")
    src.set_sample_rate(fs)
    src.set_gain(30)

    sample = freqs[::max(1, len(freqs) // 25)]

    print("\n1. звичайна перебудова")
    med, mx = bench(src.set_center_freq, sample)
    print(f"   медіана {med:7.1f} мкс, максимум {mx:7.1f} мкс")
    print(f"   -> повний прохід тільки на перебудову: "
          f"{med * len(freqs) / 1e6:.2f} с")

    print("\n2. профілі швидкої перебудови")
    t = time.perf_counter()
    ok = src.prime_quick_tune(freqs)
    print(f"   знято {ok} з {len(freqs)} за {time.perf_counter()-t:.1f} с")
    if ok < len(freqs):
        print("   !! профілі знялись не для всіх частот.")
        print("      Найімовірніше ця збірка libbladeRF або прошивка не")
        print("      підтримує quick tune для xA4. Лишай quick_tune: false —")
        print("      звичайної перебудови для FPV цілком вистачає.")
    if ok:
        med2, mx2 = bench(lambda f: src.quick_retune(f), sample[:ok])
        print(f"   медіана {med2:7.1f} мкс, максимум {mx2:7.1f} мкс")
        print(f"   -> повний прохід: {med2 * len(freqs) / 1e6:.3f} с "
              f"(прискорення {med/max(med2,1e-9):.1f}×)")

    print("\n3. захоплення на крок (те, що насправді визначає час проходу)")
    n = 4096 * 8
    t = time.perf_counter()
    for f in sample:
        src.retune_and_read(f, n)
    per = (time.perf_counter() - t) / len(sample) * 1e3
    print(f"   {per:.2f} мс на крок (перебудова + осідання + {n} відліків)")
    print(f"   -> повний прохід: {per * len(freqs) / 1e3:.2f} с")
    print(f"   зривів потоку: {src.overflows}")

    src.close()


if __name__ == "__main__":
    try:
        main()
    except BladeRFError as e:
        print(f"\nbladeRF: {e}")
        sys.exit(1)
