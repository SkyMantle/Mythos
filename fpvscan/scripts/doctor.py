#!/usr/bin/env python3
"""Діагностика підключення bladeRF.

    python scripts/doctor.py
    python scripts/doctor.py --lib "C:\\Program Files\\bladeRF\\x64\\bladeRF.dll"

Перевіряє по черзі: розрядність Python, чи є бібліотека на диску, чи
є поруч її залежності, чи вона завантажується, чи видно плату, чи
залита FPGA. Зупиняється на першому, що не так, і каже що робити.
"""
import argparse
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpvscan.sdr import bladerf as B


def ok(msg): print(f"  [+] {msg}")
def bad(msg): print(f"  [!] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", help="точний шлях до bladeRF.dll / libbladeRF.so")
    a = ap.parse_args()

    print("\n1. оточення")
    bits = 64 if ctypes.sizeof(ctypes.c_void_p) == 8 else 32
    ok(f"Python {sys.version.split()[0]}, {bits}-бітний, {sys.platform}")
    if bits == 32:
        bad("32-бітний Python не завантажить x64-збірку bladeRF.dll. "
            "Постав 64-бітний Python.")
    if os.environ.get("BLADERF_LIB_DIR"):
        ok(f"BLADERF_LIB_DIR = {os.environ['BLADERF_LIB_DIR']}")

    print("\n2. пошук бібліотеки на диску")
    files = [a.lib] if a.lib else B.find_lib_files()
    if not files:
        bad("не знайдено жодного файлу")
        print("\n     Шукав у:")
        for d in B._search_dirs()[:12]:
            print(f"       {d}")
        print("\n     Далі: чи взагалі стоїть bladeRF? Перевір командою")
        print("       bladeRF-cli -e info")
        print("     Якщо ця команда працює — знайди, де лежить bladeRF.dll,")
        print("     і передай шлях: python scripts/doctor.py --lib \"...\"")
        print("     Якщо не працює — постав bladeRF для Windows від Nuand.")
        return 1
    for f in files:
        size = os.path.getsize(f)
        print(f"  [+] {f}  ({size/1024:.0f} КБ)")

    print("\n3. залежності поруч")
    for f in files:
        d = Path(f).parent
        miss = B.missing_deps(f)
        have = [x for x in B.WIN_DEPS if (d / x).is_file()]
        if have:
            ok(f"{d}: є {', '.join(have)}")
        if miss and sys.platform.startswith("win"):
            bad(f"{d}: бракує {', '.join(miss)}")
            print("      libusb-1.0.dll — найчастіша причина того, що DLL")
            print("      «не знайдено», хоча вона на місці. Знайди її в")
            print("      каталозі установки bladeRF і поклади поруч.")

    print("\n4. завантаження")
    try:
        lib = B.load_lib(a.lib)
        ok(f"завантажено: {B.lib_path()}")
    except B.BladeRFError as e:
        bad("не завантажується")
        print("\n" + str(e))
        return 1

    print("\n5. плата")
    dev = ctypes.c_void_p()
    rc = lib.bladerf_open(ctypes.byref(dev), None)
    if rc < 0:
        msg = lib.bladerf_strerror(rc)
        bad(f"bladerf_open: {msg.decode() if msg else rc}")
        print("      Плату не видно. Перевір:")
        print("      - кабель у порт USB 3.0 (синій), не через хаб")
        print("      - у Диспетчері пристроїв має бути Nuand bladeRF")
        print("      - чи не тримає плату інша програма (bladeRF-cli, SDR#)")
        return 1
    name = lib.bladerf_get_board_name(dev)
    buf = ctypes.create_string_buffer(64)
    lib.bladerf_get_serial(dev, buf)
    ok(f"{name.decode() if name else '?'}  sn={buf.value.decode(errors='ignore')}")

    print("\n6. FPGA")
    conf = lib.bladerf_is_fpga_configured(dev)
    if conf > 0:
        size = ctypes.c_int(0)
        lib.bladerf_get_fpga_size(dev, ctypes.byref(size))
        ok(f"залита, {size.value}kLE")
    else:
        bad("не залита")
        print("      Windows: bladeRF-cli -l hostedxA4.rbf")
        print("      Ubuntu:  sudo apt install bladerf-fpga-hostedxa4")
    lib.bladerf_close(dev)

    print("\nГотово. Далі: python scripts/bringup.py -f 5800")
    return 0


if __name__ == "__main__":
    sys.exit(main())
