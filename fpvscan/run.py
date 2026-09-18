#!/usr/bin/env python3
"""Точка входу. Однаково запускається на Windows і на Pi 5."""
import os
# До імпорту numpy: на Pi 5 чотири ядра — DSP + USB + BLAS. Без ліміту
# hunt/decode роздувають OpenBLAS на всі ядра і LOCK падає до ~3 к/с.
if not os.environ.get("OPENBLAS_NUM_THREADS") and not os.environ.get("OMP_NUM_THREADS"):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import queue
import sys
import threading
from pathlib import Path

import yaml
import uvicorn
from fpvscan import config

sys.path.insert(0, str(Path(__file__).parent))

from fpvscan.engine import Engine
from fpvscan.sdr.factory import make_source
from fpvscan.web.server import create_app


def _watch_engine(engine, server, stop: threading.Event,
                  interval_s: float = 0.5) -> None:
    """Stop Uvicorn if its engine worker dies unexpectedly."""
    while not stop.wait(interval_s):
        alive = getattr(engine, "worker_alive", None)
        if (
            getattr(engine, "_worker_started", False)
            and callable(alive)
            and not alive()
            and not getattr(engine, "_stop", stop).is_set()
        ):
            print(
                "[watchdog] нитка рушія завершилась; зупиняємо процес "
                "для перезапуску systemd",
                flush=True,
            )
            server.should_exit = True
            return

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
    # Ordered low-rate state/notice/catalog events are reliable.  Video has
    # its own bounded single-slot publisher and never enters this queue.
    engine = Engine(src, cfg, queue.Queue())
    app = create_app(engine)
    engine.start()

    host, port = cfg["web"]["host"], int(cfg["web"]["port"])
    if _port_busy(port):
        print(f"Порт {port} уже зайнятий. ...")
        engine.stop()
        return
    bind = f"{host}:{port}"
    print(f"Приймач: {src.name}   слухає {bind}")
    if host in ("0.0.0.0", "::"):
        print(f"Консоль: http://<IP-цього-вузла>:{port}  (усі інтерфейси, включно з ZeroTier; не лише localhost)")
    else:
        print(f"Консоль: http://{host}:{port}")
    server = uvicorn.Server(uvicorn.Config(
        app, host=host, port=port, log_level="warning"))
    watchdog_stop = threading.Event()
    watchdog = threading.Thread(
        target=_watch_engine,
        args=(engine, server, watchdog_stop),
        daemon=True,
        name="fpvscan-watchdog",
    )
    watchdog.start()
    try:
        server.run()
    finally:
        watchdog_stop.set()
        watchdog.join(timeout=1)
        engine.stop()

if __name__ == "__main__":
    main()
