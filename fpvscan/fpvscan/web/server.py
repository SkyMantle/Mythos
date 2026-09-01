"""Веб-сервер: віддає консоль і транслює події рушія у WebSocket."""
from __future__ import annotations
import asyncio
import base64
import json
from pathlib import Path
from queue import Queue, Empty

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fpvscan import engine

STATIC = Path(__file__).parent / "static"


def create_app(engine) -> FastAPI:
    app = FastAPI(title="FPV Scan")
    clients: set[WebSocket] = set()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def state():
        return engine.snapshot()

    @app.post("/api/lock/{freq_hz}")
    async def lock(freq_hz: float):
        engine.command("lock", freq_hz=freq_hz)
        return {"ok": True}

    @app.post("/api/sweep")
    async def sweep():
        engine.command("sweep")
        return {"ok": True}

    @app.post("/api/snapshot")
    async def snapshot():
        engine.command("snapshot")
        return {"ok": True}

    @app.post("/api/record/{on}")
    async def record(on: str):
        engine.command("rec_start" if on == "start" else "rec_stop")
        return {"ok": True}

    @app.post("/api/bias_tee/{action}")
    async def api_bias_tee(action: str):
        engine.command("bias_tee", on=(action == "on"))
        return {"ok": True}

    @app.post("/api/clear")
    async def clear():
        engine.command("clear")
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        clients.add(sock)
        await sock.send_text(json.dumps({"type": "state", "data": engine.snapshot()}))
        try:
            while True:
                await sock.receive_text()   # клієнт нічого не шле, тримаємо канал
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(sock)

    async def pump():
        """Перекладає події з нитки рушія у сокети."""
        q: Queue = engine.events
        loop = asyncio.get_running_loop()
        while True:
            try:
                ev = await loop.run_in_executor(None, q.get, True, 0.5)
            except Empty:
                continue
            except Exception:
                await asyncio.sleep(0.1)
                continue
            if ev["type"] == "frame":
                ev["data"]["img"] = base64.b64encode(ev["data"]["img"]).decode()
            msg = json.dumps(ev)
            dead = []
            for c in list(clients):
                try:
                    await c.send_text(msg)
                except Exception:
                    dead.append(c)
            for c in dead:
                clients.discard(c)

    async def heartbeat():
        while True:
            await asyncio.sleep(2)
            msg = json.dumps({"type": "state", "data": engine.snapshot()})
            for c in list(clients):
                try:
                    await c.send_text(msg)
                except Exception:
                    clients.discard(c)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(pump())
        asyncio.create_task(heartbeat())

    return app
