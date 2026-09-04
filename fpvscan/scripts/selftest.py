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

print("5) відсіювання шуму класифікатором")
tone_n = int(FS_VID * 0.03)
t = np.arange(tone_n) / FS_VID
tone = np.sin(2 * np.pi * 15625.0 * t).astype(np.float32)
ts = demod.classify_video(tone, FS_VID, min_harmonics=1)
print(f"   тон 15625 Гц: відео={ts.is_video} harm={ts.harmonics} ({ts.reason})")
assert not ts.is_video, "чистий тон без гармонік не має проходити як відео"

print("6) оцінка картинки (спільна з INSPECT/пошуком частоти)")
pic = cvbs.score_picture(fr)
print(f"   score={pic.value:.2f} corr={pic.row_corr:.2f} locked={pic.locked} lines={pic.lines}")
assert pic.is_analog(min_corr=0.25), "справжній кадр не пройшов оцінку"
noise_luma = np.random.randint(0, 256, size=fr.luma.shape, dtype=np.uint8)
noise_fr = cvbs.Frame(luma=noise_luma, line_rate=15625.0, lines=fr.lines,
                      standard="PAL", locked=False)
npic = cvbs.score_picture(noise_fr)
print(f"   шум: score={npic.value:.2f} corr={npic.row_corr:.2f}")
assert not npic.is_analog(min_corr=0.22), "шум не має проходити димову перевірку"

print("7) оцінка зсуву частоти (те, чим полює hunt)")
src.set_sample_rate(FS_VID)
iq_h = src.retune_and_read(5800e6, int(FS_VID * 0.04))
scores = {}
for df, label in ((0.0, "0"), (8e6, "+8 МГц")):
    ch, fs2 = demod.channelize(iq_h, FS_VID, df, 12e6)
    b = demod.deemphasis(demod.fm_demod(ch, fs2, 3e6), fs2)
    fr2 = cvbs.decode(b, fs2, width=320)
    p2 = cvbs.score_picture(fr2)
    scores[label] = p2
    print(f"   {label}: score={p2.value:.2f} locked={p2.locked} corr={p2.row_corr:.2f}")
assert scores["0"].is_analog(min_corr=0.25), "центр має збиратись як відео"
assert scores["0"].value > scores["+8 МГц"].value + 0.05, "далеко від центру картинка гірша"

print("\nOK — тракт працює")
