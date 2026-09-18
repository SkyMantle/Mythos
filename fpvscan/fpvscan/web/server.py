"""Веб-сервер: віддає консоль і транслює події рушія у WebSocket."""
from __future__ import annotations
import asyncio
import base64
import json
import logging
from pathlib import Path
from queue import Queue, Empty

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fpvscan.app.bootstrap import attach_test_api
from fpvscan.engine import (
    IQCaptureBusy,
    IQCaptureNoSpace,
    IQCaptureTooLarge,
    IQCaptureUnavailable,
)
from fpvscan.web.coalesce import coalesce_ws_events

STATIC = Path(__file__).parent / "static"
WS_SEND_S = 1.5
log = logging.getLogger(__name__)


class IQCaptureRequest(BaseModel):
    seconds: float | None = Field(default=None, gt=0, le=2.0)
    note: str = Field(default="", max_length=256)


def _dumps(obj) -> str:
    return json.dumps(obj, default=str)


def create_app(engine) -> FastAPI:
    app = FastAPI(title="FPV Scan")
    attach_test_api(app, engine)
    clients: set[WebSocket] = set()
    rotator_lock = asyncio.Lock()

    def _update_video_clients() -> None:
        setter = getattr(engine, "set_video_clients", None)
        if callable(setter):
            setter(len(clients))

    def _register_client(sock: WebSocket) -> None:
        if sock not in clients:
            clients.add(sock)
            _update_video_clients()

    def _unregister_client(sock: WebSocket) -> None:
        if sock in clients:
            clients.remove(sock)
            _update_video_clients()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def state():
        # Always off the loop: even a cached JSON read needs the GIL.
        fn = getattr(engine, "snapshot_json", None)
        if callable(fn):
            blob = await asyncio.to_thread(fn)
            if isinstance(blob, str) and blob:
                return Response(content=blob, media_type="application/json")
        return await asyncio.to_thread(engine.snapshot)

    @app.post("/api/lock/{freq_hz}")
    async def lock(freq_hz: float, force: bool = False):
        engine.command("lock", freq_hz=freq_hz, force=force)
        return {"ok": True}

    @app.post("/api/sweep")
    async def sweep(auto_lock: bool = False):
        """Start a new sweep generation.

        Operator/API calls are a manual hold by default.  Automated scanners
        opt in explicitly with ``?auto_lock=1``.
        """
        engine.command("sweep", auto_lock=auto_lock)
        return {"ok": True, "sweep_auto_lock": auto_lock}

    @app.post("/api/snapshot")
    async def snapshot():
        engine.command("snapshot")
        return {"ok": True}

    @app.post("/api/record/{on}")
    async def record(on: str):
        engine.command("rec_start" if on == "start" else "rec_stop")
        return {"ok": True}

    @app.post("/api/iq/capture")
    async def iq_capture(payload: IQCaptureRequest | None = Body(default=None)):
        capture = getattr(engine, "capture_iq", None)
        if not callable(capture):
            raise HTTPException(
                status_code=503, detail="рушій не підтримує IQ capture")
        req = payload or IQCaptureRequest()
        try:
            return await asyncio.to_thread(
                capture, seconds=req.seconds, note=req.note)
        except IQCaptureBusy as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IQCaptureUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except IQCaptureTooLarge as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except IQCaptureNoSpace as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("IQ capture failed")
            raise HTTPException(
                status_code=500, detail="не вдалося записати IQ capture") from exc

    @app.post("/api/bias_tee/{action}")
    async def api_bias_tee(action: str):
        engine.command("bias_tee", on=(action == "on"))
        return {"ok": True}

    @app.post("/api/clear")
    async def clear():
        engine.command("clear")
        return {"ok": True}

    def _rotator():
        rot = getattr(engine, "rotator", None)
        if rot is None:
            from fpvscan.rotator import AntennaRotator
            rot = AntennaRotator((engine.cfg or {}).get("rotator") or {})
            engine.rotator = rot
        return rot

    @app.get("/api/rotator")
    async def rotator_get():
        return await asyncio.to_thread(_rotator().status)

    def _rotator_apply(payload: dict):
        rot = _rotator()
        begin_guard = getattr(engine, "begin_motor_guard", None)
        hold_mgc = getattr(engine, "hold_auto_mgc", None)
        def _guard_sdr_controls():
            if callable(begin_guard):
                # This must happen before nudge/set_azimuth can write PWM.
                begin_guard()
            elif callable(hold_mgc):
                # Compatibility for a non-Engine embedding.
                hold_mgc(0.5, reason="rotator command")
        if "step" in payload:
            try:
                step = float(payload["step"])
                _guard_sdr_controls()
                st = rot.nudge(step)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="step має бути числом") from exc
        elif "azimuth" in payload:
            try:
                azimuth = float(payload["azimuth"])
                _guard_sdr_controls()
                st = rot.set_azimuth(azimuth)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="azimuth має бути числом") from exc
        else:
            raise HTTPException(status_code=400, detail="потрібен azimuth або step")
        emit = getattr(engine, "_emit", None)
        note = getattr(engine, "note_rotator", None)
        if callable(note):
            try:
                note(st)
            except Exception:
                pass
        if callable(emit):
            try:
                emit("rotator", st)
            except Exception:
                pass
        return st

    @app.put("/api/rotator")
    async def rotator_put(payload: dict = Body(...)):
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="потрібен azimuth або step")
        async with rotator_lock:
            return await asyncio.to_thread(_rotator_apply, payload)

    async def _ws_send(sock: WebSocket, msg: str) -> bool:
        try:
            await asyncio.wait_for(sock.send_text(msg), timeout=WS_SEND_S)
            return True
        except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
            _unregister_client(sock)
            try:
                await sock.close()
            except (WebSocketDisconnect, RuntimeError):
                pass
            return False
        except asyncio.CancelledError:
            _unregister_client(sock)
            raise
        except Exception as exc:
            # Transport failures are scoped to this peer.  Do not let one
            # browser connection escape into Uvicorn's application lifecycle.
            log.debug("WebSocket send ended: %s", exc)
            _unregister_client(sock)
            try:
                await sock.close()
            except Exception:
                pass
            return False

    async def _ws_broadcast(msg: str) -> int:
        peers = list(clients)
        if not peers:
            return 0
        sent = await asyncio.gather(
            *(_ws_send(c, msg) for c in peers),
            return_exceptions=True,
        )
        return sum(result is True for result in sent)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        accepted = False
        try:
            # Registration happens only after a successful handshake.  This
            # keeps the Phase1 publisher's client count balanced when a browser
            # disappears during accept or reconnects rapidly.
            await sock.accept()
            accepted = True
            _register_client(sock)
            ws_json = getattr(engine, "ws_state_json", None)
            if callable(ws_json):
                blob = await asyncio.to_thread(ws_json)
                sent = await _ws_send(sock, blob)
            else:
                snap = await asyncio.to_thread(engine.snapshot)
                sent = await _ws_send(
                    sock, _dumps({"type": "state", "data": snap}))
            if not sent:
                return
            while True:
                # Outbound-only clients send no application messages, but an
                # ASGI receive is still the reliable disconnect notification.
                message = await sock.receive()
                if message.get("type") == "websocket.disconnect":
                    return
        except (WebSocketDisconnect, RuntimeError):
            # Disconnects before/during accept and normal ASGI state races are
            # connection-local lifecycle events, never application failures.
            return
        except asyncio.CancelledError:
            # Uvicorn cancels connection tasks during shutdown.
            return
        finally:
            _unregister_client(sock)
            if accepted:
                try:
                    await sock.close()
                except (WebSocketDisconnect, RuntimeError):
                    pass

    async def pump():
        """Перекладає події з нитки рушія у сокети."""
        q: Queue = engine.events
        loop = asyncio.get_running_loop()
        while True:
            ev = None
            try:
                ev = await loop.run_in_executor(None, q.get, True, 0.02)
            except Empty:
                pass
            except Exception:
                await asyncio.sleep(0.1)
            if ev is not None:
                batch = [ev]
                while True:
                    try:
                        batch.append(q.get_nowait())
                    except Empty:
                        break
                items = coalesce_ws_events(batch)

                def _encode(batch_items=items):
                    out = []
                    for item in batch_items:
                        # Compatibility for non-Engine producers.  Normal
                        # video never enters this reliable event queue.
                        if item["type"] == "frame":
                            raw = item["data"].get("img")
                            if isinstance(raw, (bytes, bytearray)):
                                item["data"]["img"] = base64.b64encode(raw).decode()
                        out.append(_dumps(item))
                    return out

                msgs = await asyncio.to_thread(_encode)
                for msg in msgs:
                    await _ws_broadcast(msg)

            take_video = getattr(engine, "take_video_frame", None)
            ready = take_video() if callable(take_video) else None
            if ready is not None and clients:
                await _ws_broadcast(str(ready["json"]))

    async def heartbeat():
        while True:
            await asyncio.sleep(2)
            # Ping first so зв'язок stays up even if snapshot waits on GIL.
            await _ws_broadcast('{"type":"ping"}')
            ws_json = getattr(engine, "ws_state_json", None)
            try:
                if callable(ws_json):
                    msg = await asyncio.to_thread(ws_json)
                else:
                    snap = await asyncio.to_thread(engine.snapshot)
                    msg = await asyncio.to_thread(
                        _dumps, {"type": "state", "data": snap})
            except Exception:
                continue
            await _ws_broadcast(msg)

    @app.on_event("startup")
    async def _start():
        app.state.ws_tasks = [
            asyncio.create_task(pump()),
            asyncio.create_task(heartbeat()),
        ]

    @app.on_event("shutdown")
    async def _shutdown():
        tasks = list(getattr(app.state, "ws_tasks", ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for sock in list(clients):
            _unregister_client(sock)
            try:
                await sock.close()
            except (WebSocketDisconnect, RuntimeError):
                pass

    return app
