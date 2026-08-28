#!/usr/bin/env python3
"""Рушій без веб-сервера.

Потрібен, щоб розділити дві причини мовчання: чи не працює обробка
сигналу, чи не працює веб-шар. Тут немає ні uvicorn, ні браузера —
усе, що рушій віддає, друкується в консоль.

    py scripts/headless.py                    свіп за конфігом
    py scripts/headless.py --lock 3896        одразу стати на канал
    py scripts/headless.py --secs 60          скільки працювати
"""
import argparse
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan import config
from fpvscan.engine import Engine
from fpvscan.sdr.factory import make_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--driver", help="bladerf | sim | file")
    ap.add_argument("--lock", type=float, help="одразу стати на частоту, МГц")
    ap.add_argument("--debug", action="store_true",
    help="показувати кожного кандидата і причину рішення")
    ap.add_argument("--secs", type=float, default=45.0)
    a = ap.parse_args()

    cfg = config.load(a.config)
    if a.driver:
        cfg["sdr"]["driver"] = a.driver
    if a.debug:
        cfg["scan"]["debug_candidates"] = True
        cfg["scan"]["confirm_hits"] = 1

    src = make_source(cfg["sdr"])
    q: queue.Queue = queue.Queue(maxsize=2000)
    eng = Engine(src, cfg, q)

    print(f"приймач: {src.name}, свіп "
          f"{cfg['scan']['start_hz']/1e6:.0f}-{cfg['scan']['stop_hz']/1e6:.0f} МГц")
    eng.start()
    if a.lock:
        time.sleep(1)
        eng.command("lock", freq_hz=a.lock * 1e6)
        print(f"стаю на {a.lock} МГц")

    t0 = time.time()
    n_spec = n_frame = 0
    last = 0.0
    try:
        while time.time() - t0 < a.secs:
            try:
                ev = q.get(timeout=0.5)
            except queue.Empty:
                ev = None
            if ev:
                k = ev["type"]
                if k == "spectrum":
                    n_spec += 1
                elif k == "frame":
                    n_frame += 1
                    d = ev["data"]
                    print(f"  кадр: {d['standard']} {d['line_rate']} Гц, "
                          f"рядків {d['lines']}, синхра {d['locked']}, "
                          f"{len(d['img'])} Б")
                elif k == "detection":
                    d = ev["data"]
                    print(f"  ЗНАЙДЕНО {d['freq_hz']/1e6:.1f} МГц "
                          f"{d['standard']} С/Ш {d['snr_db']} дБ "
                          f"смуга {d['bandwidth_hz']/1e6:.1f} МГц")
                elif k == "candidate":
                  d = ev["data"]	
                  mark = "+" if d.get("accepted") else "-"
                  print(f"  {mark} кандидат {d['freq_hz']/1e6:7.1f} МГц  "
                          f"смуга {d.get('bandwidth_hz',0)/1e6:4.1f} МГц  "
                          f"С/Ш {d.get('snr_db',0):5.1f}  "
                          f"рядкова {d.get('line_rate',0):8.1f} Гц "
                          f"{d.get('standard','?'):>4}  "
                          f"підйом {d.get('prominence_db',0):5.1f} дБ  "
                          f"гарм {d.get('harmonics',0)}  "
                          f"впевн {d.get('confidence',0)}")
                elif k == "notice":
                    print(f"  [{ev['data']['level']}] {ev['data']['text']}")

            now = time.time()
            if now - last >= 3:
                last = now
                s = eng.snapshot()
                alive = eng._thread.is_alive() if eng._thread else False
                print(f"[{now-t0:5.1f}с] {s['mode']:5} "
                      f"{s['tuned_hz']/1e6:7.1f} МГц  проходів {s['sweeps_done']}  "
                      f"знахідок {len(s['detections'])}  "
                      f"спектрів {n_spec} кадрів {n_frame}  "
                      f"нитка {'жива' if alive else 'МЕРТВА'}  "
                      f"зривів {s.get('overflows', 0)}/"
                      f"{getattr(src, 'timeouts', 0)}/"
                      f"{getattr(src, 'io_errors', 0)}")
                if not alive:
                    print("  Робоча нитка померла — дивись traceback вище.")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()

    print(f"\nразом: спектрів {n_spec}, кадрів {n_frame}, "
          f"знахідок {len(eng.snapshot()['detections'])}")
    if n_spec == 0:
        print("Жодного спектра. Рушій не читає з плати — проблема нижче веба.")


if __name__ == "__main__":
    main()