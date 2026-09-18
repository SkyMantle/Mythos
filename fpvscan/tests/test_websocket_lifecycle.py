from __future__ import annotations

import asyncio
from queue import Queue

from fastapi import WebSocketDisconnect

from fpvscan.web.server import create_app


class _Engine:
    cfg = {"rotator": {"enable": False}}

    def __init__(self) -> None:
        self.events = Queue()
        self.client_counts: list[int] = []

    def set_video_clients(self, count: int) -> None:
        self.client_counts.append(count)

    def ws_state_json(self) -> str:
        return '{"type":"state","data":{"mode":"SWEEP"}}'

    def snapshot(self) -> dict:
        return {"mode": "SWEEP", "engine_alive": True, "sdr_state": "OK"}


class _Socket:
    def __init__(
        self,
        *,
        accept_error: Exception | None = None,
        send_error: Exception | None = None,
        disconnect_error=False,
    ) -> None:
        self.accept_error = accept_error
        self.send_error = send_error
        self.disconnect_error = disconnect_error
        self.accepts = 0
        self.sends = 0
        self.receives = 0
        self.closes = 0

    async def accept(self) -> None:
        self.accepts += 1
        if self.accept_error is not None:
            raise self.accept_error

    async def send_text(self, _message: str) -> None:
        self.sends += 1
        if self.send_error is not None:
            raise self.send_error

    async def receive(self) -> dict:
        self.receives += 1
        if self.disconnect_error:
            raise WebSocketDisconnect(code=1001)
        return {"type": "websocket.disconnect", "code": 1001}

    async def close(self) -> None:
        self.closes += 1


def _endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/ws"
    )


def test_accept_then_immediate_disconnect_balances_client_count() -> None:
    engine = _Engine()
    endpoint = _endpoint(create_app(engine))
    sock = _Socket(disconnect_error=True)

    asyncio.run(endpoint(sock))

    assert sock.accepts == 1
    assert sock.sends == 1
    assert sock.receives == 1
    assert engine.client_counts == [1, 0]


def test_accept_failure_is_clean_and_never_registers_client() -> None:
    engine = _Engine()
    endpoint = _endpoint(create_app(engine))
    sock = _Socket(accept_error=RuntimeError("browser left during accept"))

    asyncio.run(endpoint(sock))

    assert sock.accepts == 1
    assert sock.sends == 0
    assert sock.receives == 0
    assert engine.client_counts == []


def test_initial_send_disconnect_does_not_enter_receive_loop() -> None:
    engine = _Engine()
    endpoint = _endpoint(create_app(engine))
    sock = _Socket(send_error=RuntimeError("socket closed after accept"))

    asyncio.run(endpoint(sock))

    assert sock.accepts == 1
    assert sock.sends == 1
    assert sock.receives == 0
    assert engine.client_counts == [1, 0]


def test_repeated_browser_reconnects_stay_connection_local() -> None:
    engine = _Engine()
    app = create_app(engine)
    endpoint = _endpoint(app)

    async def reconnect() -> None:
        for _ in range(25):
            await endpoint(_Socket())

    asyncio.run(reconnect())

    assert engine.client_counts == [value for _ in range(25) for value in (1, 0)]
    assert engine.snapshot()["engine_alive"] is True
