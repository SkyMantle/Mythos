"""Wire the testing-panel API into the existing FastAPI app."""
from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from fpvscan import paths
from fpvscan.app.adapters.engine_adapter import EngineAdapter
from fpvscan.app.core.errors import install_error_handlers
from fpvscan.app.core.logging import configure_test_logging, request_id_var
from fpvscan.app.repositories.sqlite_store import SqliteTestStore
from fpvscan.app.routers import include_test_routers


def attach_test_api(app: FastAPI, engine, store: SqliteTestStore | None = None) -> None:
    configure_test_logging()
    adapter = EngineAdapter(engine, paths.ROOT / "config.yaml")
    app.state.engine_port = adapter
    app.state.store = store or SqliteTestStore(paths.OUT / "test_panel.sqlite3")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_error_handlers(app)
    include_test_routers(app)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = rid
        return response
