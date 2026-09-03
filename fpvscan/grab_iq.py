
"""Знімок сирого IQ у .cf32 — щоб діагностувати офлайн, без ефіру.
 
Кладеться поруч із run.py і запускається звідти ж:
 
    py grab_iq.py 5010.788            # проблемний широкий канал
    py grab_iq.py 4988.0              # для порівняння - вузький, робочий
    py grab_iq.py 5010.788 --seconds 0.12 --out big.cf32
 
Формат .cf32 (numpy complex64, raw) — рівно той, який уміє читати
власний файловий драйвер проєкту, тож той самий запис можна не лише
переслати на аналіз, а й прокрутити локально:
 
    py run.py --file cap_5010.788.cf32
 
Частота задається в МГц. Типово знімається 0.06 с ≈ 17 МБ при
sample_rate 35 Мвідл/с — цього досить і на пошук зайнятої смуги
(find_occupied), і на кілька повних напівкадрів PAL.
"""
import argparse
import sys
from pathlib import Path
 
import numpy as np
 
sys.path.insert(0, str(Path(__file__).parent))
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freq_mhz", type=float, help="частота центру, МГц")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--seconds", type=float, default=0.06)
    ap.add_argument("--out")
    ap.add_argument("--driver")
    ap.add_argument("--lib")
    args = ap.parse_args()
 
    # Імпорт проєкту після розбору аргументів, щоб `--help` працював
    # навіть тоді, коли скрипт запущено не з кореня fpvscan.
    from fpvscan import config
    from fpvscan.sdr.factory import make_source
 
    cfg = config.load(args.config)
    if args.driver:
        cfg["sdr"]["driver"] = args.driver
    if args.lib:
        cfg["sdr"]["lib_path"] = args.lib
 
    # Беремо ту саму частоту дискретизації, що й режим утримання, —
    # інакше знімок не буде відтворювати те, що бачить _do_lock().
    fs = float(cfg["video"].get("sample_rate", cfg["scan"].get("sample_rate", 35e6)))
    hz = args.freq_mhz * 1e6
    n = int(fs * args.seconds)
 
    src = make_source(cfg)
    src.open()
    try:
        src.set_sample_rate(fs)
 
        # Підсилення рахуємо ТОЧНО так, як це робить рушій, інакше
        # знімок не відтворює те, що бачить _do_lock(). Помилка першої
        # версії цього скрипта: ставив голий sdr.gain_db (35дБ), тоді
        # як рушій при увімкненому bias-tee віднімає
        # bias_tee_gain_offset_db (15дБ) — LNA на bias-tee додає своє
        # підсилення поверх. Через це запис вийшов із 91% відліків у
        # кліпі (|iq| упирався в 1.414 = обидва канали АЦП на межі) і
        # для аналізу не годився.
        bt = bool(cfg["sdr"].get("bias_tee", False))
        if bt and hasattr(src, "set_bias_tee"):
            src.set_bias_tee(True)
        base_gain = float(cfg["sdr"].get("gain_db", 30))
        offset = float(cfg["sdr"].get("bias_tee_gain_offset_db", 18))
        gain = base_gain - offset if bt else base_gain
        if hasattr(src, "set_gain"):
            src.set_gain(gain)
        print(f"підсилення: {gain:.0f}дБ (базове {base_gain:.0f}"
              + (f" − {offset:.0f} за bias-tee" if bt else "") + ")", flush=True)
 
        # Той самий автопідбір, що й у рушії: якщо АЦП усе одно клипить,
        # ступінчасто підрізаємо, поки частка кліпу не впаде нижче порогу.
        if cfg["sdr"].get("gain_autolevel", True) and hasattr(src, "set_gain"):
            target = float(cfg["sdr"].get("clip_target", 0.02))
            step = float(cfg["sdr"].get("gain_step_db", 3.0))
            floor_db = float(cfg["sdr"].get("gain_floor_db", -10.0))
            for _ in range(int(cfg["sdr"].get("gain_autolevel_steps", 6))):
                src.retune_and_read(hz, max(4096, int(fs * 0.005)))
                clip = float(getattr(src, "clip_frac", 0.0))
                if clip <= target:
                    break
                gain -= step
                if gain < floor_db:
                    break
                print(f"  кліп {clip:.3f} > {target} -> підсилення {gain:.0f}дБ",
                      flush=True)
                src.set_gain(gain)
        print(f"частота {hz/1e6:.3f} МГц, fs={fs/1e6:.2f} Мвідл/с, "
              f"{args.seconds:.3f} с = {n} відліків", flush=True)
        iq = src.retune_and_read(hz, n)
    finally:
        try:
            src.close()
        except Exception:
            pass
 
    iq = np.asarray(iq).astype(np.complex64)
    out = Path(args.out or f"cap_{args.freq_mhz:g}.cf32")
    iq.tofile(out)
 
    # Коротка самоперевірка, щоб одразу було видно, що запис не порожній
    # і не в кліпі — інакше є ризик надіслати на аналіз тишу.
    mag = np.abs(iq)
    print(f"записано: {out}  ({out.stat().st_size/1e6:.1f} МБ, {len(iq)} відліків)")
    print(f"рівень: медіана |iq|={np.median(mag):.4f}  max={mag.max():.4f}  "
          f"частка >0.99 (кліп)={float((mag > 0.99).mean()):.4f}")
    if hasattr(src, "center_freq"):
        print(f"драйвер каже center_freq={float(src.center_freq)/1e6:.3f} МГц")
    if np.median(mag) < 1e-4:
        print("!! майже тиша — перевір антену/bias-tee/підсилення перед надсиланням")
 
 
if __name__ == "__main__":
    main()
 



