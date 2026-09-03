#!/usr/bin/env python3
"""Проганяє ВЕСЬ ланцюг рушія по записаному .cf32 — офлайн, без ефіру.

    py analyze_cap.py cap_5010.788.cf32 --center 5010.788

Робить рівно те саме, що робить `_do_lock()`, але покроково й з друком
проміжних чисел, яких у бойовому лозі не видно:

  1. psd_db + find_occupied — де САМЕ пошук бачить зайняту смугу і який
     центр він назвав би (те, з чого народжується `lock_target`);
  2. центроїд і пік потужності — де сигнал стоїть насправді;
  3. freq_error_hz — що каже AFC на цьому ж записі;
  4. channelize -> fm_demod -> deemphasis -> cvbs.decode — чи збирається
     кадр, і якщо ні, то на якому саме рубежі відвалюється.

Далі — те, чого в бойовому режимі зробити не можна: перебирає ЗСУВ
каналайзера від -15 до +15МГц і показує, на якому зсуві кадр нарешті
декодується. Це прямо відповідає на питання "чи є взагалі такий зсув,
при якому воно працює" — без AFC, без перебудов, без ефіру.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="файл .cf32 (numpy complex64, raw)")
    ap.add_argument("--center", type=float, default=0.0,
                    help="частота, на яку був налаштований приймач, МГц")
    ap.add_argument("--fs", type=float, default=35.0, help="Мвідл/с")
    ap.add_argument("--ch-bw", type=float, default=20.0, help="смуга каналу, МГц")
    ap.add_argument("--scan", default="/tmp/fp",
                    help="каталог, де лежить пакет fpvscan (для імпорту dsp)")
    args = ap.parse_args()

    sys.path.insert(0, args.scan)
    sys.path.insert(0, str(Path(__file__).parent))
    from fpvscan.dsp import spectrum, demod, cvbs

    fs = args.fs * 1e6
    ch_bw = args.ch_bw * 1e6
    c0 = args.center * 1e6

    iq = np.fromfile(args.path, dtype=np.complex64)
    print(f"файл: {args.path}  {len(iq)} відліків = {len(iq)/fs*1000:.1f} мс "
          f"при fs={fs/1e6:.2f}МГц")
    mag = np.abs(iq)
    print(f"рівень: медіана |iq|={np.median(mag):.4f} max={mag.max():.4f} "
          f"частка>0.99={float((mag>0.99).mean()):.4f}")
    iq = iq - iq.mean()

    # --- 1. те, що побачив би свіп ---
    nfft, avg = 8192, 8
    if len(iq) >= nfft * avg:
        psd = spectrum.psd_db(iq, nfft, avg)
        occ = spectrum.find_occupied(psd, c0, fs, threshold_db=4.0,
                                     min_bw_hz=2e6, dc_notch_hz=200e3)
        edge = (nfft / 2 - int(nfft * 0.05)) * (fs / nfft)
        print(f"\n[1] find_occupied: знайдено {len(occ)} ділянок "
              f"(край корисного вікна ±{edge/1e6:.2f}МГц)")
        for o in occ:
            off = o.center_hz - c0
            lo, hi = off - o.bandwidth_hz / 2, off + o.bandwidth_hz / 2
            clipped = " <-- ПРИТИСНУТО ДО КРАЮ ВІКНА" if (
                hi >= edge - 2 * fs / nfft or lo <= -edge + 2 * fs / nfft) else ""
            print(f"    центр={o.center_hz/1e6:.3f}МГц (зсув {off/1e6:+.2f}) "
                  f"смуга={o.bandwidth_hz/1e6:.2f}МГц SNR={o.snr_db:.1f}дБ "
                  f"[{lo/1e6:+.2f}..{hi/1e6:+.2f}]{clipped}")

    # --- 2. де сигнал насправді ---
    m = min(len(iq), 1 << 16)
    sp = np.abs(np.fft.fftshift(np.fft.fft(iq[:m] * np.hanning(m)))) ** 2
    fr = np.fft.fftshift(np.fft.fftfreq(m, 1 / fs))
    pk = float(fr[int(np.argmax(sp))])
    cen = float((fr * sp).sum() / sp.sum())
    print(f"\n[2] спектр: пік={pk/1e6:+.2f}МГц центроїд={cen/1e6:+.2f}МГц "
          f"(абсолютно: пік={(c0+pk)/1e6:.3f} центроїд={(c0+cen)/1e6:.3f}МГц)")

    # --- 3. що каже AFC ---
    base_iq, fs_ch = demod.channelize(iq, fs, 0.0, ch_bw)
    err = demod.freq_error_hz(base_iq, fs_ch)
    print(f"\n[3] freq_error_hz на зсуві 0: {err/1e3:+.0f}кГц "
          f"(fs_ch={fs_ch/1e6:.2f}МГц, децимація {int(fs/ch_bw)}x)")

    # --- 4. декодування на зсуві 0 ---
    def try_decode(shift_hz, verbose=False):
        b_iq, f_ch = demod.channelize(iq, fs, shift_hz, ch_bw)
        base = demod.fm_demod(b_iq, f_ch, deviation_hz=ch_bw / 4)
        base = demod.deemphasis(base, f_ch)
        sc = demod.classify_video(base, f_ch)
        fr_ = cvbs.decode(base, f_ch, width=640, presmooth_us=1.0e-6,
                          min_score=0.4, min_row_corr=0.4)
        return sc, fr_

    sc, frame = try_decode(0.0)
    print(f"\n[4] на зсуві 0: classify_video is_video={sc.is_video} "
          f"line={sc.line_rate:.0f}Гц conf={sc.confidence:.2f} "
          f"prom={sc.prominence_db:.1f}дБ | cvbs.decode -> "
          + ("КАДР Є" if frame is not None else "None"))

    # --- 5. перебір зсуву: чи існує зсув, на якому воно працює ---
    print(f"\n[5] перебір зсуву каналайзера (те, чого AFC зробити не може):")
    print(f"    {'зсув':>8} {'is_video':>9} {'conf':>6} {'prom':>7} {'кадр':>6}")
    best = None
    for sh_mhz in np.arange(-15, 15.5, 1.0):
        sc, frame = try_decode(sh_mhz * 1e6)
        got = "ТАК" if frame is not None else "-"
        if sc.confidence > 0.1 or frame is not None:
            print(f"    {sh_mhz:>+7.1f}МГц {str(sc.is_video):>9} "
                  f"{sc.confidence:>6.2f} {sc.prominence_db:>6.1f}дБ {got:>6}")
        if best is None or sc.confidence > best[1]:
            best = (sh_mhz, sc.confidence, frame is not None)
    print(f"\n    найкращий зсув: {best[0]:+.1f}МГц (впевненість {best[1]:.2f}, "
          f"кадр {'є' if best[2] else 'нема'})")
    if abs(best[0]) > 1.0:
        print(f"    -> сигнал стоїть на {best[0]:+.1f}МГц від центру запису, "
              f"тобто абсолютно ≈{(c0 + best[0]*1e6)/1e6:.3f}МГц")


if __name__ == "__main__":
    main()
