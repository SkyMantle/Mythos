#!/usr/bin/env python3
"""Наскрізний самотест без заліза.

Ганяє симулятор -> спектр -> детектор -> класифікатор -> декодер CVBS
і зберігає decoded.png. Якщо на картинці видно градаційні смуги —
весь тракт живий.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fpvscan.sdr.sim import SimSource
from fpvscan.dsp import spectrum, demod, cvbs

FS_SCAN, FS_VID = 40e6, 30e6
src = SimSource(); src.open(); src.set_sample_rate(FS_SCAN)

print("1) спектр і пошук зайнятих смуг на 5800 МГц")
iq = src.retune_and_read(5800e6, 4096 * 8)
psd = spectrum.psd_db(iq, 4096, 8)
occ = spectrum.find_occupied(psd, 5800e6, FS_SCAN, threshold_db=8, min_bw_hz=4e6)
print(f"   підлога {spectrum.noise_floor_db(psd):.1f} дБ, знайдено {len(occ)}")
for o in occ:
    print(f"   {o.center_hz/1e6:8.1f} МГц  смуга {o.bandwidth_hz/1e6:5.1f} МГц  "
          f"С/Ш {o.snr_db:5.1f} дБ")
assert occ, "детектор нічого не знайшов"

print("2) класифікація за рядковою частотою")
for o in occ:
    long_iq = src.retune_and_read(5800e6, int(FS_SCAN * 0.025))
    ch, fs2 = demod.channelize(long_iq, FS_SCAN, o.center_hz - 5800e6,
                               max(o.bandwidth_hz, 8e6))
    base = demod.fm_demod(ch, fs2, o.bandwidth_hz / 5)
    s = demod.classify_video(base, fs2)
    print(f"   {o.center_hz/1e6:8.1f} МГц -> відео={s.is_video} "
          f"{s.standard} {s.line_rate:.0f} Гц conf={s.confidence}")

print("3) захоплення та декодування кадру")
src.set_sample_rate(FS_VID)
t = time.perf_counter()
iq = src.retune_and_read(5800e6, int(FS_VID * 0.12))
t_cap = time.perf_counter() - t

t = time.perf_counter()
base = demod.fm_demod(iq, FS_VID, 10e6)
base = demod.deemphasis(base, FS_VID)
fr = cvbs.decode(base, FS_VID, width=640)
t_dsp = time.perf_counter() - t

assert fr is not None, "декодер не зібрав кадр"
print(f"   {fr.standard} {fr.line_rate:.1f} Гц, рядків {fr.lines}, "
      f"кадрова синхра={fr.locked}")
print(f"   генерація {t_cap*1000:.0f} мс, обробка {t_dsp*1000:.0f} мс "
      f"на 120 мс ефіру")

out = Path(__file__).parent / "decoded.png"
from PIL import Image
Image.fromarray(fr.luma, "L").resize((640, 480)).save(out)
print(f"   кадр -> {out}")

# перевірка, що це не шум: рядки мають бути схожі один на одного
rows = fr.luma.astype(np.float32)
corr = np.corrcoef(rows[50], rows[51])[0, 1]
print(f"4) кореляція сусідніх рядків {corr:.3f} (шум дав би ~0)")
assert corr > 0.7, "картинка не тримається"
print("\nOK — тракт працює")
