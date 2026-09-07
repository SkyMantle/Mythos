from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from fpvscan.app.core.logging import request_id_var, session_id_var
from fpvscan.app.domain.clock import utc_now_iso
from fpvscan.app.domain.exceptions import NotFoundError, ValidationError
from fpvscan.app.domain.models import SESSION_MODES, SESSION_STATUSES, Session
from fpvscan.app.repositories.protocols import TestStoreProtocol
from fpvscan.app.services.catalog import catalog_values
from fpvscan.app.services.idempotency import remember, replay_or_none
from fpvscan.app.services.ports import EnginePort

log = logging.getLogger("fpvscan.test.sessions")


class SessionService:
    def __init__(self, engine: EnginePort, repo: TestStoreProtocol) -> None:
        self._engine = engine
        self._repo = repo

    async def create_session(
        self,
        idempotency_key: UUID,
        mode: str,
        title: str | None = None,
        parameter_snapshot: dict[str, Any] | None = None,
        schema_id: str | None = None,
    ) -> Session:
        if mode not in SESSION_MODES:
            raise ValidationError(f"invalid mode: {mode}", {"mode": "enum"})
        body = {
            "mode": mode,
            "title": title,
            "parameter_snapshot": parameter_snapshot,
            "schema_id": schema_id,
        }
        replay = await replay_or_none(self._repo, idempotency_key, "session.create", body)
        if replay is not None:
            return Session.from_dict(replay)
        schema = (
            await self._repo.get_schema(schema_id)
            if schema_id
            else await self._repo.get_active_schema()
        )
        if schema is None:
            raise NotFoundError(f"observation schema {schema_id} not found")
        now = utc_now_iso()
        session = Session(
            id=str(uuid4()),
            title=(title or "").strip() or f"{mode} session",
            mode=mode,
            status="active",
            notes="",
            schema_id=schema.id,
            schema_snapshot=list(schema.fields),
            parameter_snapshot=parameter_snapshot or catalog_values(self._engine.current_cfg()),
            created_at=now,
            updated_at=now,
        )
        await self._repo.create_session(session)
        await remember(self._repo, idempotency_key, "session.create", body, session.to_dict())
        session_id_var.set(session.id)
        log.info("session_created id=%s mode=%s rid=%s", session.id, mode, request_id_var.get())
        return session

    async def list_sessions(self) -> list[Session]:
        return await self._repo.list_sessions()

    async def get_session(self, session_id: str) -> Session:
        session = await self._repo.get_session(session_id)
        if session is None:
            raise NotFoundError(f"session {session_id} not found")
        return session

    async def patch_session(
        self,
        session_id: str,
        title: str | None = None,
        notes: str | None = None,
        status: str | None = None,
    ) -> Session:
        session = await self.get_session(session_id)
        if status is not None:
            if status not in SESSION_STATUSES:
                raise ValidationError(f"invalid status: {status}", {"status": "enum"})
            if session.status == "completed" and status != "completed":
                raise ValidationError("completed session is terminal", {"status": "terminal"})
            session.status = status
        if title is not None:
            session.title = title.strip()
        if notes is not None:
            session.notes = notes
        session.updated_at = utc_now_iso()
        return await self._repo.update_session(session)

    async def complete_session(self, session_id: str) -> Session:
        session = await self.get_session(session_id)
        if session.status == "completed":
            return session
        session.status = "completed"
        session.updated_at = utc_now_iso()
        return await self._repo.update_session(session)
