#!/usr/bin/env python3
"""Запис ефіру у файл.

    python scripts/record.py -f 5800 -s 40 -d 3 -g 30 -o caps/vtx_f4.cf32

Далі цей файл можна ганяти на Windows скільки завгодно разів:

    python run.py --driver file --file caps/vtx_f4.cf32

Обсяг: complex64 = 8 байт на відлік. 40 Мвідл/с × 3 с ≈ 960 МБ.
Тому 2–4 секунди зазвичай достатньо: у них уже 60–100 кадрів відео.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr.bladerf import BladeRF
from fpvscan.sdr.file import write_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--freq", type=float, required=True, help="МГц")
    ap.add_argument("-s", "--rate", type=float, default=40.0, help="Мвідл/с")
    ap.add_argument("-d", "--dur", type=float, default=3.0, help="секунд")
    ap.add_argument("-g", "--gain", type=float, default=30.0, help="дБ")
    ap.add_argument("--agc", action="store_true")
    ap.add_argument("-o", "--out", default="capture.cf32")
    ap.add_argument("--note", default="")
    ap.add_argument("--device", default="", help="ідентифікатор плати, напр. *:serial=abcd")
    ap.add_argument("--lib", help="точний шлях до bladeRF.dll / libbladeRF.so")
    a = ap.parse_args()

    fc, fs = a.freq * 1e6, a.rate * 1e6
    n = int(fs * a.dur)
    print(f"{fc/1e6:.1f} МГц, {fs/1e6:.1f} Мвідл/с, {a.dur} с "
          f"= {n*8/1e6:.0f} МБ")

    src = BladeRF(device=a.device, lib_path=a.lib, gain_db=a.gain, agc=a.agc)
    src.open()
    src.set_sample_rate(fs)
    src.set_gain(a.gain)
    i = src.info()
    print(f"плата {i['board']} sn={i['serial'][:8]} FPGA {i['fpga_kle']}kLE "
          f"підсилення {i['gain_range'][0]}..{i['gain_range'][1]} дБ")
    src.set_center_freq(fc)
    src.read(int(fs * 0.05))                 # даємо тракту встановитись

    print("запис...")
    iq = src.read(n)
    src.close()

    peak = 20 * np.log10(np.max(np.abs(iq)) + 1e-12)
    rms = 20 * np.log10(np.sqrt(np.mean(np.abs(iq) ** 2)) + 1e-12)
    print(f"пік {peak:.1f} дБ від шкали, СКЗ {rms:.1f} дБ, "
          f"обрізано {src.clip_frac*100:.2f}%, зривів потоку {src.overflows}")
    if src.clip_frac > 0.001:
        print("!! АЦП у насиченні — зменш підсилення або відсунь антену")
    elif peak < -35:
        print("!! сигнал надто слабкий — додай підсилення")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p = write_capture(a.out, iq, fc, fs, a.gain, a.note)
    print(f"-> {p}  (+ {p.with_suffix('.json').name})")


if __name__ == "__main__":
    main()
