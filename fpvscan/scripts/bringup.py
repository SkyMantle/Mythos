#!/usr/bin/env python3
"""Первинне підняття заліза на реальному стенді.

Наводиш антену на увімкнений передавач, кажеш скрипту приблизну
частоту — він підбирає підсилення, міряє зайняту смугу, перевіряє
рядкову частоту, декодує кадр і зберігає його у PNG.

    python scripts/bringup.py -f 5800            # реальний bladeRF
    python scripts/bringup.py --file caps/x.cf32 # перевірка на записі

Якщо на якомусь етапі зупиняється — у виводі написано, що саме
дивитись. Це швидше, ніж гадати за веб-консоллю.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.dsp import spectrum, demod, cvbs
from fpvscan.bands import nearest_channel


def grab(src, fc, fs, n):
    src.set_center_freq(fc)
    src.read(int(fs * 0.02))
    return src.read(n)


def level(x):
    """Рівень сигналу без постійної складової.

    Витік гетеродина і зміщення АЦП сидять на нулі, не залежать від
    підсилення і, якщо їх не прибрати, дають фальшивий «пік», за яким
    неможливо підібрати підсилення.
    """
    import numpy as np
    d = x - np.mean(x)
    peak = 20 * np.log10(np.max(np.abs(d)) + 1e-12)
    rms = 20 * np.log10(np.sqrt(np.mean(np.abs(d) ** 2)) + 1e-12)
    dc = 20 * np.log10(np.abs(np.mean(x)) + 1e-12)
    return peak, rms, dc

def _sync_report(base, fs):
    """Що саме бачить декодер: чи є взагалі синхроімпульси і чи рівно
    вони йдуть. Кореляція рядків каже «шум», а це каже чому."""
    import numpy as np
    for sign, label in ((1.0, "пряма"), (-1.0, "інверсна")):
        v = base * sign
        lo, hi = np.percentile(v, [0.5, 99.5])
        if hi - lo < 1e-9:
            continue
        v = (v - lo) / (hi - lo)
        below = v < 0.18
        edges = np.flatnonzero(below[1:] & ~below[:-1]) + 1
        if len(edges) < 10:
            print(f"  полярність {label}: фронтів синхри {len(edges)} — мало")
            continue
        d = np.diff(edges).astype(float)
        med = float(np.median(d[d > np.percentile(d, 40)]))
        good = float(((d > med * 0.8) & (d < med * 1.2)).mean())
        print(f"  полярність {label}: фронтів {len(edges)}, "
              f"період {med:.1f} відл. = {fs/med:.0f} Гц, "
              f"рівних інтервалів {good*100:.0f}%")

def scan_band(src, lo_hz, hi_hz, fs, gain=None):
    """Прочісування діапазону: де взагалі щось є."""
    import numpy as np
    from fpvscan.dsp import spectrum
    step = max(fs * 0.25, fs * 0.9 - 10e6)   # запас під ширину каналу
    freqs = np.arange(lo_hz + fs / 2, hi_hz, step)
    print(f"  {lo_hz/1e6:.0f}–{hi_hz/1e6:.0f} МГц, {len(freqs)} кроків "
          f"по {step/1e6:.0f} МГц")
    hits = []
    for f in freqs:
        x = grab(src, f, fs, 4096 * 8)
        x = x - np.mean(x)
        psd = spectrum.psd_db(x, 4096, 8)
        view = spectrum.usable_view(psd, fs)
        nf = spectrum.noise_floor_db(view)
        occ = spectrum.find_occupied(psd, f, fs, threshold_db=6, min_bw_hz=1e6)
        mark = ""
        if occ:
            best = max(occ, key=lambda z: z.snr_db)
            hits += occ
            mark = (f"  <- {best.center_hz/1e6:.1f} МГц, "
                    f"{best.bandwidth_hz/1e6:.1f} МГц, +{best.snr_db:.1f} дБ")
        print(f"    {f/1e6:7.1f} МГц  підлога {nf:6.1f}  макс {view.max():6.1f}"
              f"  динаміка {view.max()-nf:5.1f} дБ{mark}")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--freq", type=float, help="МГц")
    ap.add_argument("-s", "--rate", type=float, default=40.0, help="Мвідл/с")
    ap.add_argument("--file", help="перевірити на записі замість заліза")
    ap.add_argument("--device", default="")
    ap.add_argument("--lib", help="точний шлях до bladeRF.dll / libbladeRF.so")
    ap.add_argument("--scan", nargs=2, type=float, metavar=("LO", "HI"),
                    help="прочесати діапазон, МГц: --scan 3000 3900")
    ap.add_argument("-o", "--out", default="bringup.png")
    ap.add_argument("--save", help="зберегти захоплений IQ у .cf32 для розбору")
    a = ap.parse_args()

    # --- джерело ---
    if a.file:
        from fpvscan.sdr.file import FileSource
        src = FileSource(a.file)
        src.open()
        fc, fs = src.center_freq, src.sample_rate
        gains = [None]
    else:
        if not a.freq and not a.scan:
            ap.error("вкажи -f (МГц), або --scan LO HI, aбо --file")
        from fpvscan.sdr.bladerf import BladeRF
        src = BladeRF(device=a.device, lib_path=a.lib)
        src.open()
        i = src.info()
        print(f"плата: {i['board']}  sn={i['serial'][:8]}  FPGA {i['fpga_kle']}kLE")
        fc = a.freq * 1e6 if a.freq else (a.scan[0] + a.scan[1]) / 2 * 1e6
        fs = a.rate * 1e6
        src.set_sample_rate(fs)
        print(f"смуга ФНЧ {src.bandwidth/1e6:.1f} МГц")
        g_lo, g_hi = i['gain_range']
        gains = list(range(int(g_lo) + 5, int(g_hi) + 1, 10))

    print(f"\n=== 1. рівень і підсилення === {fc/1e6:.1f} МГц, "
          f"{fs/1e6:.1f} Мвідл/с")
    best_g, iq = None, None
    for g in gains:
        if g is not None:
            src.set_gain(g)
        x = grab(src, fc, fs, 1 << 18)
        peak, rms, dc = level(x)
        clip = float(np.mean(np.abs(x.real) > 0.98))
        tag = ""
        if clip > 1e-3:
            tag = "  <- насичення"
        elif -30 < peak < -6:
            tag = "  <- добре"
        print(f"  підсилення {str(g):>4} дБ: пік {peak:6.1f}  СКЗ {rms:6.1f}"
              f"  пост.скл. {dc:6.1f}  обрізано {clip*100:5.2f}%{tag}")
        if clip < 1e-3 and (best_g is None or peak > -30):
            best_g, iq = g, x
    if iq is None:
        iq = x
    if best_g is not None:
        src.set_gain(best_g)
        print(f"  беремо {best_g} дБ")

    if level(iq)[0] < -45 and not a.scan:
        print("  !! сигналу майже немає. Перевір: чи увімкнений передавач,")
        print("     чи та антена в тому роз'ємі (RX1), чи та частота.")

    # --- прочісування діапазону, якщо частота невідома ---
    if a.scan:
        print("\n=== 1б. прочісування діапазону ===")
        hits = scan_band(src, a.scan[0] * 1e6, a.scan[1] * 1e6, fs)
        if not hits:
            print("  нічого не знайдено. Передавач увімкнений? Антена в RX1?")
            return
        top = max(hits, key=lambda z: z.snr_db)
        fc = top.center_hz
        print(f"  -> найсильніше на {fc/1e6:.1f} МГц, далі працюємо з ним")
        iq = grab(src, fc, fs, 1 << 18)

    print("\n=== 2. спектр ===")
    iq = iq - np.mean(iq)
    psd = spectrum.psd_db(iq, 4096, 16)
    view = spectrum.usable_view(psd, fs)
    nf = spectrum.noise_floor_db(view)
    print(f"  шумова підлога {nf:.1f} дБ, максимум {view.max():.1f} дБ, "
          f"динаміка {view.max()-nf:.1f} дБ  (без країв смуги і нуля)")
    occ = spectrum.find_occupied(psd, fc, fs, threshold_db=6, min_bw_hz=1e6)
    if not occ:
        print("  !! нічого не піднімається над підлогою на 6 дБ.")
        print("     Якщо не знаєш точну частоту — прочеши діапазон:")
        print("       py scripts/bringup.py --scan 3000 3900")
        print("     Якщо частота відома — перевір антену (роз'єм RX1),")
        print("     відстань до передавача і чи він взагалі випромінює.")
        return
    for o in sorted(occ, key=lambda z: -z.snr_db)[:5]:
        ch = nearest_channel(o.center_hz)
        print(f"  {o.center_hz/1e6:9.1f} МГц  смуга {o.bandwidth_hz/1e6:5.1f} МГц  "
              f"С/Ш {o.snr_db:5.1f} дБ  {ch or ''}")
    top = max(occ, key=lambda z: z.snr_db)
    print(f"  -> найсильніший канал на {top.center_hz/1e6:.1f} МГц")
        # Аналогове відео має широку «полицю»: ширина слабо росте при
    # зниженні порогу. Якщо на 3 дБ смуга різко ширша, ніж на 20 —
    # це вузький сигнал із пологими схилами, тобто не відео.
    shape = []
    for th in (3, 6, 10, 20, 30):
        o2 = spectrum.find_occupied(psd, fc, fs, threshold_db=th,
                                    min_bw_hz=0.3e6)
        near = [z for z in o2 if abs(z.center_hz - top.center_hz) < 15e6]
        shape.append((th, max((z.bandwidth_hz for z in near), default=0.0)))
    print("  ширина смуги за порогами: " +
          "  ".join(f"{th}дБ={b/1e6:.1f}" for th, b in shape) + "  МГц")
    if top.bandwidth_hz > fs * 0.75:
        print("  !! канал ширший за смугу приймача — підніми -s "
              f"(зараз {fs/1e6:.0f}, спробуй {min(61, fs/1e6*1.5):.0f})")

    print("\n=== 3. рядкова частота ===")
    n_long = int(fs * 0.06)
    x = grab(src, fc, fs, n_long) if not a.file else src.read(n_long)
    if a.save:
        from fpvscan.sdr.file import write_capture
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        write_capture(a.save, x, fc, fs, 0.0, f"bringup {fc/1e6:.1f} МГц")
        print(f"  IQ збережено -> {a.save}")
    x = x - np.mean(x)
    ch_iq, fs2 = demod.channelize(x, fs, top.center_hz - fc,
                                  max(top.bandwidth_hz, 8e6))
    base = demod.fm_demod(ch_iq, fs2, top.bandwidth_hz / 5)
    sc = demod.classify_video(base, fs2)
    print(f"  {sc.line_rate:.0f} Гц  {sc.standard}  впевненість {sc.confidence}"
          f"  -> {'аналогове відео' if sc.is_video else 'НЕ схоже на відео'}")
    if not sc.is_video:
        print("  Це або цифровий канал, або сигнал надто слабкий для")
        print("  вимірювання рядкової. Кадр нижче все одно спробуємо.")

    print("\n=== 4. декодування кадру ===")
    x = x - np.mean(x)
    bw = max(top.bandwidth_hz * 1.6, 8e6)
    ch2, fs_ch = demod.channelize(x, fs, top.center_hz - fc, bw)
    print(f" канал {bw/1e6:.1f} МГц -> {fs_ch/1e6:.1f} Мвідл/с, "
          f"{fs_ch/15625:.0f} відліків на рядок")
    base = demod.fm_demod(ch2, fs_ch, deviation_hz=bw / 4)
    base = demod.deemphasis(base, fs_ch)
    fr = cvbs.decode(base, fs_ch, width=640)
    _sync_report(base, fs_ch)
    if fr is None:
        print("  !! кадр не зібрався. Найчастіші причини, за спаданням:")
        print("     1) смуга приймача вужча за канал — підніми -s")
        print("     2) АЦП у насиченні — знизь підсилення")
        print("     3) С/Ш замалий — ближче до передавача або краща антена")
        return
    print(f"  {fr.standard}  {fr.line_rate:.1f} Гц  рядків {fr.lines}  "
          f"кадрова синхра {'є' if fr.locked else 'нема'}")
    corr = float(np.corrcoef(fr.luma[50].astype(float),
                             fr.luma[51].astype(float))[0, 1])
    print(f"  кореляція сусідніх рядків {corr:.3f} "
          f"({'картинка тримається' if corr > 0.6 else 'схоже на шум'})")

    from PIL import Image
    Image.fromarray(fr.luma, "L").resize((640, 480)).save(a.out)
    print(f"  -> {a.out}")

    if hasattr(src, "overflows"):
        print(f"\nзривів потоку USB: {src.overflows}")
        if src.overflows:
            print("  Перевір кабель USB 3.0 і живлення Pi (5 В / 5 А).")
    src.close()


if __name__ == "__main__":
    main()
