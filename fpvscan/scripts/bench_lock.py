#!/usr/bin/env python3
"""Офлайн-бенч гарячого циклу LOCK — без заліза, на симуляторі.

Повторює те, що робить Engine._do_lock() на кожен кадр:
channelize -> afc -> fm_demod -> deemphasis -> decode(трекінг) -> encode,
міряє кожен етап окремо і сумарний fps. Дозволяє A/B будь-якої
оптимізації без bladeRF (тест на реальному стенді вже завершено).

    py scripts/bench_lock.py                      # поточні дефолти
    py scripts/bench_lock.py --fs 20 --encode webp1
    py scripts/bench_lock.py --fs 35 --margin 1.25

Друкує середні часи (мс) по кожному етапу за N ітерацій, відкидаючи
перші кілька (прогрів JIT кешів BLAS/варму трекінгу).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr.sim import SimSource, Emitter
from fpvscan.dsp import demod, cvbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=20.0, help="Мвідл/с у LOCK")
    ap.add_argument("--bw", type=float, default=12.0, help="смуга каналу, МГц")
    ap.add_argument("--dev", type=float, default=10.0, help="девіація ЧМ, МГц")
    ap.add_argument("--capture-ms", type=float, default=56.0)
    ap.add_argument("--margin", type=float, default=1.25,
                    help="track_window_margin: скільки полів беремо на кадр")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--max-lines", type=int, default=288)
    ap.add_argument("--encode", default="webp0",
                    choices=["webp4", "webp1", "webp0", "jpeg", "none"])
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--std", default="PAL", choices=["PAL", "NTSC"])
    a = ap.parse_args()

    fs = a.fs * 1e6
    ch_bw = a.bw * 1e6
    line = cvbs.__dict__.get("_LINE", None)
    line_rate = 15625.0 if a.std == "PAL" else 15734.264

    # Один сильний відео-передавач рівно в центрі смуги (offset=0), як у
    # стабільному LOCK: несуча вже зведена AFC до нуля.
    src = SimSource(emitters=[Emitter(5800e6, -12.0, "fpv",
                                      a.dev * 1e6, line_rate, "bench")],
                    noise_db=-70.0)
    src.open()
    src.set_sample_rate(fs)
    src.set_center_freq(5800e6)

    capture_s = a.capture_ms / 1000.0
    n_full = int(fs * capture_s)
    dec = max(1, int(fs / ch_bw))
    field_lines = cvbs.FIELD_LINES[a.std]

    state = cvbs.DecodeState()
    abs_start = 0.0
    stages = {k: [] for k in ("channelize", "afc", "demod_fm",
                              "deemphasis", "decode", "encode", "total")}
    ok = 0
    lines_seen = []

    def enc(frame):
        if a.encode == "none":
            return b""
        if a.encode == "jpeg":
            return cvbs.encode(frame, "jpeg", a.quality, height=None)
        method = {"webp4": 4, "webp1": 1, "webp0": 0}[a.encode]
        return cvbs.encode(frame, "webp", a.quality, height=None, method=method)

    for it in range(a.iters):
        # розмір вікна на кадр — як в Engine: у трекінгу коротший
        n = n_full
        if state.period is not None and state.lost == 0:
            n_track = int(state.period * field_lines * dec * a.margin)
            n = max(int(fs * 0.02), min(n_full, n_track))

        iq = src.read(n)
        abs_start += 0  # позиція старту знімку у шкалі потоку (тут = 0 зсув)

        t = time.perf_counter()
        base_iq, fs_ch = demod.channelize(iq, fs, 0.0, ch_bw)
        t1 = time.perf_counter()
        # як у Engine: спочатку ЧМ, AFC з того ж сигналу (без 2-го arctan2)
        base = demod.fm_demod(base_iq, fs_ch, deviation_hz=ch_bw / 4)
        t2 = time.perf_counter()
        _ = demod.freq_error_from_demod(base, ch_bw / 4)
        t3 = time.perf_counter()
        base = demod.deemphasis(base, fs_ch)
        t4 = time.perf_counter()
        frame = cvbs.decode(base, fs_ch, width=a.width, max_lines=a.max_lines,
                            state=state, abs_start=abs_start + 1,
                            auto_levels=True, sharpen=0.5)
        t5 = time.perf_counter()
        img = enc(frame) if frame is not None else b""
        t6 = time.perf_counter()

        abs_start += n / dec  # наступний знімок іде далі по потоку (в шкалі каналу)

        if it < a.warmup:
            continue
        stages["channelize"].append((t1 - t) * 1000)
        stages["demod_fm"].append((t2 - t1) * 1000)
        stages["afc"].append((t3 - t2) * 1000)
        stages["deemphasis"].append((t4 - t3) * 1000)
        stages["decode"].append((t5 - t4) * 1000)
        stages["encode"].append((t6 - t5) * 1000)
        stages["total"].append((t6 - t) * 1000)
        if frame is not None:
            ok += 1
            lines_seen.append(frame.lines)

    n_meas = a.iters - a.warmup
    print(f"\nfs={a.fs:.0f} МГц  bw={a.bw:.0f} МГц  dec={dec}  "
          f"capture={a.capture_ms:.0f}мс  margin={a.margin}  "
          f"encode={a.encode} q={a.quality}")
    print(f"кадрів декодовано: {ok}/{n_meas}"
          + (f"  рядків~{int(np.median(lines_seen))}" if lines_seen else ""))
    print("-" * 46)
    for k in ("channelize", "afc", "demod_fm", "deemphasis",
              "decode", "encode", "total"):
        arr = np.array(stages[k])
        if len(arr):
            print(f"  {k:12s} {arr.mean():7.1f} мс  (медіана {np.median(arr):6.1f})")
    tot = np.median(stages["total"]) if stages["total"] else 0
    if tot > 0:
        print("-" * 46)
        print(f"  => {1000.0/tot:5.1f} к/с  (за медіаною total {tot:.1f} мс)")


if __name__ == "__main__":
    main()
