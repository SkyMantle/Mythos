#!/usr/bin/env python3
"""Точка входу. Однаково запускається на Windows і на Pi 5."""
import argparse
import queue
import sys
from pathlib import Path

import yaml
import uvicorn
from fpvscan import config

sys.path.insert(0, str(Path(__file__).parent))

from fpvscan.engine import Engine
from fpvscan.sdr.factory import make_source
from fpvscan.web.server import create_app

def _port_busy(port: int) -> bool:
    import socket
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--driver", help="перекрити sdr.driver: bladerf | file | sim")
    ap.add_argument("--file", help="запис .cf32 для відтворення (вмикає driver=file)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--lib", help="точний шлях до bladeRF.dll / libbladeRF.so")
    args = ap.parse_args()

    cfg = config.load(args.config)
    if args.file:
        cfg["sdr"]["driver"] = "file"
        cfg["sdr"]["path"] = args.file
    if args.lib:
        cfg["sdr"]["lib_path"] = args.lib
    if args.driver:
        cfg["sdr"]["driver"] = args.driver
    if args.port:
        cfg["web"]["port"] = args.port

    src = make_source(cfg["sdr"])
    engine = Engine(src, cfg, queue.Queue(maxsize=64))
    app = create_app(engine)
    engine.start()

    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    if _port_busy(port):
        print(f"Порт {port} уже зайнятий. Найімовірніше запущена інша копія —\n"
            f"вона ж тримає і плату. Заверши її або візьми інший порт: "
            f"--port {port + 1}")
        engine.stop()
        return
    # 0.0.0.0 — це «слухати на всіх інтерфейсах», а не адреса для браузера
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Приймач: {src.name}   ->   http://{shown}:{port}")
    try:
                uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
